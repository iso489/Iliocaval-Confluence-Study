"""Tests of the class-score resampling experiment on synthetic examinations (no GPU, no TotalSegmentator)."""
import ast
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import class_score_resampling as CS
import model_grid_audit as MG
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
IVC, LEFT, RIGHT = 13, 17, 18
LATER = 20                                 # a muscle label, written after the cardiac submodel
CHANNELS = 19


def class_map():
    mapping = {i: f"structure_{i}" for i in range(1, 23)}
    mapping.update({IVC: "inferior_vena_cava", LEFT: "iliac_vena_left", RIGHT: "iliac_vena_right"})
    return mapping


def internal_labels():
    """A vertical IVC above a two-plane gap, the left iliac below it and a separate right iliac."""
    volume = np.zeros((24, 24, 61), np.uint8)
    volume[10:14, 10:14, 36:61] = IVC
    volume[10:14, 10:14, 0:34] = LEFT
    volume[17:20, 10:14, 0:30] = RIGHT
    return volume


def cardiac_view(labels):
    """What the cardiac submodel predicted: a later submodel's label overwrote the IVC beneath it."""
    return np.where(labels == LATER, IVC, labels)


def scores_for(labels, seed=20261025):
    """Softmax with slight voxel noise (no exact ties), one dominant class per voxel."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.001, 0.002, (CHANNELS, *labels.shape)).astype(np.float32)
    for channel in range(CHANNELS):
        raw[channel][labels == channel] = rng.uniform(0.95, 0.99, int((labels == channel).sum()))
    return raw / raw.sum(axis=0, keepdims=True)


def build_stage(path, labels=None, native_shape=(24, 24, 21), swap_axes=(0, 3, 2, 1), tamper_native=False):
    """The files one GPU inference leaves in its stage directory."""
    labels = internal_labels() if labels is None else labels
    stage = Path(path)
    stage.mkdir(parents=True, exist_ok=True)
    mapping = class_map()
    cardiac = {str(i): {"name": mapping[i], "total_id": i} for i in range(1, CHANNELS)}
    (stage / MG.INTERNAL_LABELS).write_text(json.dumps({
        "total_class_map": {str(k): v for k, v in mapping.items()},
        "cardiac_submodel": {"task_id": 293, "part": "class_map_part_cardiac", "classes": cardiac}}))
    nib.save(nib.Nifti1Image(labels, np.diag([1.5, 1.5, 1.5, 1])), stage / MG.INTERNAL_MULTILABEL)
    factors = np.array(native_shape) / np.array(labels.shape)
    native = ndi.zoom(labels, factors, order=0, mode="nearest")
    if tamper_native:
        native[0, 0, 0] = IVC
    affine = np.diag([*(1.5 * (np.array(labels.shape) - 1) / (np.array(native_shape) - 1)), 1])
    nib.save(nib.Nifti1Image(native.astype(np.uint8), affine), stage / MG.CASE_OUTPUTS[0])
    # nnU-Net's reader stores the array as (channel, z, y, x)
    np.savez_compressed(stage / CS.SCORES, probabilities=np.transpose(scores_for(cardiac_view(labels)), swap_axes))
    return stage


def class_scores(labels):
    scores = np.log(np.maximum(scores_for(cardiac_view(labels)), CS.EPS))
    for channel in (IVC, LEFT, RIGHT):
        scores[channel][labels == LATER] = np.log(CS.EPS)
    return scores


def expected_linear(labels, native_shape):
    """The whole-volume answer, through scipy, with no working box."""
    scores = class_scores(labels)
    factors = np.array(native_shape) / np.array(labels.shape)
    resampled = np.stack([ndi.zoom(channel, factors, order=1, mode="nearest") for channel in scores])
    code = np.zeros(CHANNELS, np.uint8)
    code[[IVC, LEFT, RIGHT]] = [1, 2, 3]
    return code[np.argmax(resampled, axis=0)]


@pytest.mark.parametrize("n_in,n_out", [(61, 21), (61, 40), (24, 47), (30, 30)])
def test_coordinates_and_interpolation_match_scipy_zoom(n_in, n_out):
    rng = np.random.default_rng(7)
    values = rng.normal(size=(1, n_in))
    mine = CS.along_axis(values, CS.zoom_coordinates(n_in, n_out), 1)[0]
    theirs = ndi.zoom(values[0], n_out / n_in, order=1, mode="nearest")
    assert theirs.shape == mine.shape
    assert np.allclose(mine, theirs, atol=1e-9)


def test_argmax_of_interpolated_log_softmax_equals_that_of_interpolated_logits():
    rng = np.random.default_rng(11)
    logits = rng.normal(0, 4, (5, 20, 4, 4)).astype(np.float64)
    softmax = np.exp(logits) / np.exp(logits).sum(axis=0, keepdims=True)
    coordinates = [CS.zoom_coordinates(20, 9), CS.zoom_coordinates(4, 4), CS.zoom_coordinates(4, 4)]
    from_logits = CS.resample_argmax(logits, coordinates, [19 / 8, 1, 1], slab=False)
    from_softmax = CS.resample_argmax(np.log(softmax), coordinates, [19 / 8, 1, 1], slab=False)
    assert np.array_equal(from_logits, from_softmax)


def test_slab_mean_integrates_a_single_plane():
    values = np.zeros((1, 21), np.float64)
    values[0, 10] = 1.0
    # the tent of plane 10 has area 1; over a slice 3 planes thick its mean is 1/3
    assert CS.slab_mean(values, np.array([10.0]), 3.0, 1)[0, 0] == pytest.approx(1 / 3, abs=0.02)


def test_slab_equals_linear_when_no_axis_is_coarsened():
    rng = np.random.default_rng(13)
    scores = rng.normal(size=(4, 12, 12, 12))
    coordinates = [CS.zoom_coordinates(12, 23)] * 3
    assert np.array_equal(CS.resample_argmax(scores, coordinates, [11 / 22] * 3, slab=False),
                          CS.resample_argmax(scores, coordinates, [11 / 22] * 3, slab=True))


@pytest.mark.parametrize("native_shape", [(24, 24, 21), (47, 47, 21), (24, 24, 40)])
def test_linear_arm_matches_the_whole_volume_result(tmp_path, native_shape):
    labels = internal_labels()
    stage = build_stage(tmp_path / "stage", labels, native_shape)
    out = CS.analyze(stage, tmp_path / "case", later_ids=[LATER])
    saved = np.asanyarray(nib.load(tmp_path / "case" / CS.MASKS["linear"]).dataobj)
    assert np.array_equal(saved, expected_linear(labels, native_shape))
    assert out["axis_order"] == [2, 1, 0] and out["tie_mismatch_voxels"] == 0


def test_slab_arm_matches_an_independent_average_over_each_slice(tmp_path):
    labels = internal_labels()
    labels[10:14, 10:14, 30] = 0                            # a one-plane hole exactly on native plane 10
    stage = build_stage(tmp_path / "stage", labels)
    CS.analyze(stage, tmp_path / "case", later_ids=[LATER])
    saved = np.asanyarray(nib.load(tmp_path / "case" / CS.MASKS["slab"]).dataobj)
    scores = class_scores(labels)
    step = 60 / 20
    offsets = ((np.arange(CS.SLAB_SAMPLES) + 0.5) / CS.SLAB_SAMPLES - 0.5) * step
    x, y, z = np.meshgrid(np.arange(24), np.arange(24), np.arange(21) * step, indexing="ij")
    mean = np.mean([[ndi.map_coordinates(channel, [x, y, np.clip(z + offset, 0, 60)], order=1, mode="nearest")
                     for channel in scores] for offset in offsets], axis=0)
    code = np.zeros(CHANNELS, np.uint8)
    code[[IVC, LEFT, RIGHT]] = [1, 2, 3]
    assert np.array_equal(saved, code[np.argmax(mean, axis=0)])
    assert not np.array_equal(saved, expected_linear(labels, (24, 24, 21)))   # the arm differs from linear


def test_classes_follow_the_masks_and_the_gap_is_seen_on_the_inference_grid(tmp_path):
    stage = build_stage(tmp_path / "stage")
    classes = CS.analyze(stage, tmp_path / "case", later_ids=[LATER])["classes"]
    assert classes["inference"]["class26"] == "disconnected"
    assert classes["nearest"]["class26"] == "direct"        # planes 33 and 36 become adjacent
    assert set(classes) == set(CS.ARMS)


def test_a_later_submodel_label_is_not_venous_in_the_class_score_arms(tmp_path):
    labels = internal_labels()
    labels[10:14, 10:14, 45] = LATER                        # the muscle label cuts the IVC on one plane
    stage = build_stage(tmp_path / "stage", labels)
    result = CS.analyze(stage, tmp_path / "case", later_ids=[LATER])
    assert result["venous_voxels_overwritten_by_later_submodels"] == 16
    assert np.array_equal(np.asanyarray(nib.load(tmp_path / "case" / CS.MASKS["linear"]).dataobj),
                          expected_linear(labels, (24, 24, 21)))


def test_an_export_that_is_not_the_nearest_plane_zoom_is_refused(tmp_path):
    stage = build_stage(tmp_path / "stage", tamper_native=True)
    with pytest.raises(ValueError, match="does not reproduce the exported native labels"):
        CS.analyze(stage, tmp_path / "case", later_ids=[LATER])


def test_scores_stored_in_the_wrong_position_are_refused(tmp_path):
    labels = internal_labels()
    stage = build_stage(tmp_path / "stage", labels)
    with np.load(stage / CS.SCORES) as saved:
        moved = np.roll(saved["probabilities"], 5, axis=1)   # same shape, wrong voxels
    np.savez_compressed(stage / CS.SCORES, probabilities=moved)
    with pytest.raises(ValueError, match="do not reproduce the captured labels"):
        CS.analyze(stage, tmp_path / "case", later_ids=[LATER])


def test_a_split_image_with_a_partial_score_grid_is_refused(tmp_path):
    stage = build_stage(tmp_path / "stage")
    with np.load(stage / CS.SCORES) as saved:
        partial = saved["probabilities"][:, :40]
    np.savez_compressed(stage / CS.SCORES, probabilities=partial)
    with pytest.raises(ValueError, match="cannot match the captured grid"):
        CS.analyze(stage, tmp_path / "case", later_ids=[LATER])


def test_only_the_cardiac_submodel_saves_scores():
    calls = []
    module = SimpleNamespace(nnUNetv2_predict=lambda d, o, task, *a, **k: calls.append((task, k.get("save_probabilities_path"))))
    seen = CS.capture_scores(module, "/stage/scores.npz")
    for task in (291, 292, 293, 294, 295):
        module.nnUNetv2_predict("in", "out", task, save_probabilities_path=None)
    assert calls == [(291, None), (292, None), (293, Path("/stage/scores.npz")), (294, None), (295, None)]
    assert seen == [293]


def test_a_pipeline_that_already_saves_scores_is_refused():
    module = SimpleNamespace(nnUNetv2_predict=lambda *a, **k: None)
    CS.capture_scores(module, "/stage/scores.npz")
    with pytest.raises(ValueError, match="already requests"):
        module.nnUNetv2_predict("in", "out", 293, save_probabilities_path=Path("/other.npz"))


def _inference_call(path, function):
    tree = ast.parse(Path(path).read_text())
    body = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
    call = next(n for n in ast.walk(body) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "totalsegmentator")
    return sorted((k.arg, ast.dump(k.value)) for k in call.keywords)


def test_the_inference_call_is_the_one_the_archive_rerun_used():
    assert _inference_call(CS.__file__, "score_worker") == _inference_call(MG.__file__, "inference_worker")


def test_the_summary_counts_arms_and_keeps_paths_out_of_the_record(tmp_path):
    rows = []
    args = SimpleNamespace(out=tmp_path / "out", model_grid_root=tmp_path)
    for i, (spacing, near, linear) in enumerate([(1.5, "disconnected", "disconnected"), (2.5, "direct", "disconnected"),
                                                 (5.0, "disconnected", "direct"), (5.0, "nonmeasurable", "nonmeasurable")]):
        row = {"cohort": "AMOS", "case": f"case_{i}", "mask_case": f"case_{i}", "participant": f"p{i}"}
        rows.append(row)
        block = {arm: {"class26": near, "class6": near, "voxels": [1, 1, 1]} for arm in CS.ARMS}
        block["linear"] = {"class26": linear, "class6": linear, "voxels": [1, 1, 1]}
        MG.atomic_json(MG.case_folder(args.out, row) / "qc.json",
                       {"status": "pass", **row, "archive_class26": near, "archive_class6": near, "classes": block,
                        "native_spacing_mm": [1.0, 1.0, spacing], "tie_mismatch_voxels": 0,
                        "venous_voxels_overwritten_by_later_submodels": 0, "total_seconds": 1.0})
    failed = {"cohort": "SAROS", "case": "case_9", "mask_case": "case_9", "participant": "p9"}
    rows.append(failed)
    MG.atomic_json(MG.case_folder(args.out, failed) / "qc.json",
                   {"status": "failed", "error": "OSError: /home/someone/secret/file.nii.gz"})
    (tmp_path / "cases.json").write_text(json.dumps(rows))
    assert CS.summarize(args, rows) == 0
    summary = pd.read_csv(args.out / "class_score_summary.csv")
    line = summary[(summary.group == "all") & (summary.adjacency == 26)].iloc[0]
    assert (line.n_measurable, line.nearest_disconnected, line.linear_disconnected) == (3, 2, 2)
    assert (line.linear_closed_vs_nearest, line.linear_opened_vs_nearest) == (1, 1)
    assert set(summary.group) == {"all", "spacing_le_1p5mm", "spacing_gt1p5_to3mm", "spacing_gt3mm"}
    record = (args.out / "run_record.json").read_text()
    assert "/home/" not in record and "<path>" in record
    assert json.loads(record)["complete"] == 4


def _rows_with_records(tmp_path, statuses):
    args = SimpleNamespace(out=tmp_path / "out", workers=2, max_cases=None, stop_launching_after=None)
    rows = []
    for i, status in enumerate(statuses):
        row = {"cohort": "AMOS", "case": f"case_{i}"}
        rows.append(row)
        if status is not None:
            MG.atomic_json(MG.case_folder(args.out, row) / "qc.json", {"status": status})
    return args, rows


def test_all_mode_runs_only_unfinished_cases_and_in_parallel(tmp_path):
    args, rows = _rows_with_records(tmp_path, ["pass", None, "failed", "started", None, None])
    lock, running, peak, launched = threading.Lock(), [0], [0], []

    def launch(_args, index):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
            launched.append(index)
        time.sleep(0.05)
        with lock:
            running[0] -= 1
        return 0

    assert CS.run_all(args, rows, launch=launch) == 0
    assert sorted(launched) == [1, 2, 3, 4, 5]
    assert peak[0] == 2


def test_a_pilot_stops_at_max_cases(tmp_path):
    args, rows = _rows_with_records(tmp_path, ["pass", None, None, None, None])
    args.max_cases = 2
    launched = []
    CS.run_all(args, rows, launch=lambda _a, i: launched.append(i) or 0)
    assert sorted(launched) == [1, 2]


def test_no_case_starts_after_the_launch_limit(tmp_path):
    args, rows = _rows_with_records(tmp_path, [None, None, None])
    args.stop_launching_after = 0.0
    launched = []
    assert CS.run_all(args, rows, launch=lambda _a, i: launched.append(i) or 0) == 0
    assert launched == []


def test_one_failed_case_does_not_stop_the_others(tmp_path):
    args, rows = _rows_with_records(tmp_path, [None, None, None, None])
    launched = []

    def launch(_args, index):
        launched.append(index)
        return 1 if index == 1 else 0

    assert CS.run_all(args, rows, launch=launch) == 1
    assert sorted(launched) == [0, 1, 2, 3]


def test_a_damaged_record_counts_as_unfinished(tmp_path):
    args, rows = _rows_with_records(tmp_path, ["pass", "pass"])
    (MG.case_folder(args.out, rows[1]) / "qc.json").write_text("")
    assert CS._unfinished(args, rows) == [1]


def test_each_case_gets_its_own_totalsegmentator_home(tmp_path, monkeypatch):
    ct = tmp_path / "ct.nii.gz"
    ct.write_bytes(b"ct")
    args = SimpleNamespace(index=0, out=tmp_path / "out", recorded_root="/r", study_root=tmp_path,
                           stage_root=tmp_path, weights_root=tmp_path / "weights")
    row = {"cohort": "AMOS", "case": "case_0", "mask_case": "case_0", "participant": "p",
           "expected_class26": "direct", "expected_class6": "direct"}
    seen = {}
    monkeypatch.setattr(CS, "verified_sources", lambda *a: {"ct_path": ct, "frozen_native": ct})

    def fake_run(command, env, stdout, stderr):
        seen.update(env)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(CS.subprocess, "run", fake_run)
    assert CS.run_case(args, [row]) == 1
    assert Path(seen["TOTALSEG_HOME_DIR"]).parent == Path(seen["TMPDIR"])
    assert seen["TMPDIR"].startswith(str(tmp_path))
