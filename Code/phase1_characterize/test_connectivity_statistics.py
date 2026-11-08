
import hashlib
import subprocess
import sys
from pathlib import Path

import connectivity_statistics as FS
import numpy as np
import pandas as pd
import pytest
import recover_raw_connectivity as RC
import SimpleITK as sitk
import stats as ST

import model_grid_audit as MG


@pytest.mark.parametrize(
    "script", ["connectivity_statistics.py", "recover_raw_connectivity.py"]
)
def test_analysis_commands_start_without_pytest_path_bootstrap(script, tmp_path):
    path = Path(__file__).with_name(script)
    result = subprocess.run(
        [sys.executable, str(path), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_model_grid_canonical_frame_includes_empty_cases_and_exact_crosswalk():
    frame = MG.canonical_frame(FS.OUT / "analysis_cases.csv", FS.OUT / "participant_mapping.csv")
    from collections import Counter
    assert len(frame) == 1899
    assert Counter(r["expected_class26"] for r in frame) == MG.EXPECTED_NATIVE
    assert len({r["participant"] for r in frame}) == 1784
    empty = [r for r in frame if r["expected_class26"] == "nonmeasurable"]
    assert len(empty) == 26 and {r["cohort"] for r in empty} == {"SAROS"}
    assert all(r["case"] != r["mask_case"] for r in frame if r["cohort"] == "autoPET")
    wrong = pd.read_csv(FS.OUT / "participant_mapping.csv")
    assert set(wrong.cohort) == set(MG.EXPECTED)


def test_model_grid_rejects_canonical_identity_tampering(tmp_path):
    people = pd.read_csv(FS.OUT / "participant_mapping.csv")
    people.loc[0, "participant"] = "wrong-person"
    people.to_csv(tmp_path / "people.csv", index=False)
    with pytest.raises(ValueError, match="participant/case crosswalk"):
        MG.canonical_frame(FS.OUT / "analysis_cases.csv", tmp_path / "people.csv")


def test_model_grid_ct_paths_come_from_original_reports(tmp_path):
    ct = tmp_path / "Data" / "AMOS" / "imagesTr" / "ct.nii.gz"
    ct.parent.mkdir(parents=True)
    ct.write_bytes(b"ct")
    assert MG.ct_from_report({"input": str(ct)}, tmp_path) == ct
    assert MG.ct_from_report({"input": "/old/site/Data/AMOS/imagesTr/ct.nii.gz"}, tmp_path) == ct
    with pytest.raises(FileNotFoundError, match="no directory-scan"):
        MG.ct_from_report({"input": "/old/site/unknown.nii.gz"}, tmp_path)
    with pytest.raises(ValueError, match="absolute"):
        MG.ct_from_report({"input": "relative.nii.gz"}, tmp_path)
    for bad in ["..", "/absolute", "a/b", "a\\b", ""]:
        with pytest.raises(ValueError, match="unsafe"):
            MG.case_folder(tmp_path, {"cohort": "AMOS", "case": bad})
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "root").mkdir()
    (tmp_path / "root" / "AMOS").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        MG.case_folder(tmp_path / "root", {"cohort": "AMOS", "case": "x"})


def _total_class_map():
    from totalsegmentator.map_to_binary import class_map
    return {int(k): v for k, v in class_map["total"].items()}


def _capture_context(names):
    return {"task_name": "total", "output_task_name": "total", "resample": 1.5,
            "crop": None, "roi_subset": None, "save_lowres": False,
            "remove_small_blobs": False, "test": 0, "class_map": {"total": names}}


def test_model_grid_capture_is_passive_and_rejects_wrong_modes(tmp_path, monkeypatch):
    import nibabel as nib
    names = _total_class_map()
    ids = {v: k for k, v in names.items()}
    artery = ids["iliac_artery_right"]
    data = np.zeros((7, 8, 9), np.uint8)
    data[1:3, :, :] = ids[MG.NAMES[0]]
    data[3:5, :, :] = ids[MG.NAMES[1]]
    data[5:7, :, :] = ids[MG.NAMES[2]]
    data[3:5, 0, :] = artery
    image = nib.Nifti1Image(data, np.diag([1.5, 1.5, 1.5, 1.]))
    before_header = image.header.binaryblock
    before_array = data.copy()
    output = tmp_path / "internal.nii.gz"
    monkeypatch.setenv("TS_AUDIT_CAPTURE", str(output))
    context = _capture_context(names)
    MG.capture_internal(image, context)
    actual = nib.load(output)
    assert np.array_equal(actual.affine, image.affine)
    assert set(np.unique(actual.dataobj)) == {0, 1, 2, 3}
    assert np.array_equal(data, before_array) and image.header.binaryblock == before_header
    with pytest.raises(ValueError, match="multiple capture"):
        MG.capture_internal(image, context)
    for name in MG.INTERNAL_OUTPUTS:
        output.with_name(name).unlink(missing_ok=True)
    output.unlink()
    for field, value in [("task_name", "total_v3"), ("resample", 3.), ("save_lowres", True),
                         ("roi_subset", ["inferior_vena_cava"]), ("remove_small_blobs", True)]:
        with pytest.raises(ValueError, match="capture requires"):
            MG.capture_internal(image, {**context, field: value})
        assert not output.exists()
    image = nib.Nifti1Image(data, np.diag([1.5, 1.5, 1.6, 1.]))
    with pytest.raises(ValueError, match="1.5-mm"):
        MG.capture_internal(image, context)


def test_model_grid_capture_keeps_the_complete_label_map_and_its_mapping(tmp_path, monkeypatch):
    import json

    import nibabel as nib
    names = _total_class_map()
    ids = {v: k for k, v in names.items()}
    data = np.zeros((6, 7, 8), np.uint8)
    data[1:3, :, :] = ids["inferior_vena_cava"]
    data[4:6, :, :] = ids["iliac_vena_left"]
    data[3, 2:5, :] = ids["iliac_artery_right"]
    data[3, 0:2, :] = ids["aorta"]
    data[0, 0, 0] = ids["vertebrae_L5"]
    affine = np.diag([1.5, 1.5, 1.5, 1.])
    affine[:3, 3] = [-100., 50., 7.5]
    image = nib.Nifti1Image(data, affine)
    venous = tmp_path / "internal_venous.nii.gz"
    monkeypatch.setenv("TS_AUDIT_CAPTURE", str(venous))
    MG.capture_internal(image, _capture_context(names))
    assert set(MG.INTERNAL_OUTPUTS) == {"internal_venous.nii.gz", "internal_multilabel.nii.gz",
                                        "internal_labels.json"}
    full = nib.load(tmp_path / "internal_multilabel.nii.gz")
    assert np.array_equal(np.asanyarray(full.dataobj), data)
    assert full.get_data_dtype() == np.uint8 and np.array_equal(full.affine, affine)
    compact = np.asanyarray(nib.load(venous).dataobj)
    assert np.array_equal(compact, MG.compact_venous(np.asanyarray(full.dataobj), [ids[n] for n in MG.NAMES]))
    labels = json.loads((tmp_path / "internal_labels.json").read_text())
    assert {int(k): v for k, v in labels["total_class_map"].items()} == names
    cardiac = labels["cardiac_submodel"]
    assert cardiac["task_id"] == 293 and len(cardiac["classes"]) == 18
    by_name = {c["name"]: c["total_id"] for c in cardiac["classes"].values()}
    assert by_name["iliac_artery_left"] == ids["iliac_artery_left"]
    assert by_name["iliac_artery_right"] == ids["iliac_artery_right"]
    assert by_name["inferior_vena_cava"] == ids["inferior_vena_cava"]
    assert labels["venous_compact"] == {"1": "inferior_vena_cava", "2": "iliac_vena_left", "3": "iliac_vena_right"}
    assert np.allclose(labels["grid"]["affine"], affine) and labels["grid"]["shape"] == [6, 7, 8]
    counts = labels["voxel_counts"]
    assert counts["iliac_artery_right"] == 3 * 8 and counts["aorta"] == 2 * 8 and counts["vertebrae_L5"] == 1
    assert "iliac_vena_right" not in counts and sum(counts.values()) == int(np.count_nonzero(data))


def test_model_grid_capture_refuses_foreign_labels_and_an_incomplete_cardiac_map(tmp_path, monkeypatch):
    import nibabel as nib
    names = _total_class_map()
    ids = {v: k for k, v in names.items()}
    data = np.zeros((4, 4, 4), np.uint8)
    data[0] = ids["inferior_vena_cava"]
    data[2] = ids["iliac_vena_left"]
    image = nib.Nifti1Image(data, np.diag([1.5, 1.5, 1.5, 1.]))
    venous = tmp_path / "internal_venous.nii.gz"
    monkeypatch.setenv("TS_AUDIT_CAPTURE", str(venous))
    foreign = data.copy()
    foreign[3] = 250
    assert 250 not in names
    with pytest.raises(ValueError, match="outside the class map"):
        MG.capture_internal(nib.Nifti1Image(foreign, image.affine), _capture_context(names))
    assert not any((tmp_path / name).exists() for name in MG.INTERNAL_OUTPUTS)
    partial = {k: v for k, v in names.items() if v != "iliac_artery_left"}
    with pytest.raises(ValueError, match="cardiac"):
        MG.capture_internal(image, _capture_context(partial))
    assert not any((tmp_path / name).exists() for name in MG.INTERNAL_OUTPUTS)


def test_model_grid_instrumentation_anchor_rejects_foreign_source():
    with pytest.raises(ValueError, match="reviewed v2.17.0"):
        MG.patched_source("# newer or modified inference code\n")


def test_model_grid_topology_empty_labels_and_face_only_sensitivity():
    image = np.zeros((5, 5, 5), np.uint8)
    image[1, 1, 1] = 1
    image[2, 2, 2] = 2
    result = MG.topology(image, [1, 2, 3])
    assert result["class26"] == "direct" and result["class6"] == "disconnected"
    image[image == 2] = 0
    assert MG.topology(image, [1, 2, 3])["class26"] == "nonmeasurable"


def test_model_grid_unknown_outcomes_and_paired_participant_bootstrap():
    frame = pd.DataFrame({"cohort": ["X"] * 5, "case": list("abcde"),
                          "participant": ["repeat", "repeat", "repeat", "p2", "p3"],
                          "native_spacing_mm": [1.] * 5,
                          "native_class26": ["disconnected", "direct", "direct", "nonmeasurable", "disconnected"],
                          "internal_class26": ["direct", "direct", "disconnected", "nonmeasurable", "nonmeasurable"]})
    frame["native_class6"] = frame.native_class26
    frame["internal_class6"] = frame.internal_class26
    outputs = FS.model_grid_summaries(frame, n_boot=400, seed=11)
    pairs = outputs["model_grid_paired.csv"].query("stratum=='Pooled' and connectivity==26").set_index("outcome")
    assert pairs.loc["lower_bound", "mean_change"] == -.4
    assert pairs.loc["upper_bound", "mean_change"] == .2
    # Independent direct enumeration of participant-block draws, including all visits.
    hi = np.array([1, 0, 0])
    sizes = np.array([1, 1, 3])
    # The actual per-visit bound vectors are [-1,0,1,-1,-1] and [-1,0,1,1,0].
    lo = np.array([-1, -1, 0])
    rng = np.random.default_rng(11)
    draws = rng.integers(0, 3, (400, 3))
    for outcome, sums in [("lower_bound", lo), ("upper_bound", hi)]:
        values = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
        assert np.allclose(pairs.loc[outcome, ["ci_lo", "ci_hi"]].to_numpy(float), np.quantile(values, [.025, .975]))
    transitions = outputs["model_grid_transitions.csv"].query("stratum=='Pooled' and connectivity==26")
    assert len(transitions) == 16 and transitions.events.sum() == 5
    assert transitions.query("native=='nonmeasurable' and internal=='nonmeasurable'").events.iloc[0] == 1


def _model_grid_case_fixture(tmp_path, monkeypatch, mismatch=False, inference_exit=0,
                             capture="complete"):
    """`capture` selects what the synthetic inference leaves behind: the real capture's three
    files (`complete`), the venous file alone (`venous_only`), a venous file that disagrees
    with the complete map (`compact_mismatch`), or a sidecar whose class map was altered
    (`sidecar_mismatch`). `mismatch` selects how the synthetic rerun departs from the archive:
    `False` reproduces it voxel for voxel, `True` adds one stray IVC voxel far from the veins (a
    boundary difference that leaves the connectivity class as archived), and `"class"` removes
    the IVC slab touching the left iliac so the rerun's native class becomes disconnected."""
    import json
    from types import SimpleNamespace

    import nibabel as nib
    from totalsegmentator.map_to_binary import class_map
    folder = tmp_path / "archive"
    folder.mkdir()
    ids = [next(k for k, v in class_map["total"].items() if v == name) for name in MG.NAMES]
    artery = next(k for k, v in class_map["total"].items() if v == "iliac_artery_right")
    data = np.zeros((6, 6, 6), np.uint8)
    data[1:3, 2:4, 2:4] = ids[0]
    data[3:5, 2:4, 2:4] = ids[1]
    data[5, 2:4, 2:4] = artery
    ct = folder / "CT.nii.gz"
    native = folder / "segmentation.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros(data.shape, np.int16), np.eye(4)), ct)
    nib.save(nib.Nifti1Image(data, np.eye(4)), native)
    report = folder / "report.json"
    report.write_text("{}")
    row = {"cohort": "AMOS", "case": "a", "participant": "AMOS:p", "mask_case": "a",
           "ct_path": str(ct), "frozen_native": str(native), "frozen_report": str(report),
           "label_indices": ids, "expected_class26": "direct", "expected_class6": "direct",
           "native_geometry": MG.geometry(nib.load(native))}
    row["source_hashes"] = MG._source_hashes(row)
    pre = {"manifest_sha256": "frozen"}
    env = {"fingerprint": "environment", "identity": {"weights_root": str(tmp_path / "weights")}}
    monkeypatch.setattr(MG, "load_contract", lambda root, recheck_environment=False: ([row], pre, env))
    monkeypatch.setattr(MG, "environment_record", lambda path: env["identity"])
    calls = []
    def fake_inference(command, env, stdout, stderr):
        calls.append(command)
        assert MG.sha(command[3]) == row["source_hashes"]["ct_path"]
        assert env["TOTALSEG_WEIGHTS_PATH"].endswith("weights")
        stdout.write("synthetic inference subprocess\n")
        rerun = data.copy()
        if mismatch == "class":
            rerun[2, 2:4, 2:4] = 0
        elif mismatch:
            rerun[0, 0, 0] = ids[0]          # one stray voxel; the class stays direct
        nib.save(nib.Nifti1Image(rerun, np.eye(4)), command[4])
        # The synthetic inference runs the REAL capture on the internal grid, so the files a
        # case folder receives are the ones the instrumented module would write.
        internal = nib.Nifti1Image(data, np.diag([1.5, 1.5, 1.5, 1.]))
        monkeypatch.setenv("TS_AUDIT_CAPTURE", command[5])
        MG.capture_internal(internal, _capture_context({int(k): v for k, v in class_map["total"].items()}))
        if capture == "venous_only":
            for name in MG.INTERNAL_OUTPUTS[1:]:
                Path(command[5]).with_name(name).unlink()
        elif capture == "compact_mismatch":
            compact = np.asanyarray(nib.load(command[5]).dataobj).copy()
            compact[5, 2, 2] = 3
            nib.save(nib.Nifti1Image(compact, internal.affine), command[5])
        elif capture == "sidecar_mismatch":
            sidecar_path = Path(command[5]).with_name(MG.INTERNAL_LABELS)
            sidecar = json.loads(sidecar_path.read_text())
            sidecar["total_class_map"][str(ids[2])] = "renamed_label"
            sidecar_path.write_text(json.dumps(sidecar))
        return SimpleNamespace(returncode=inference_exit)
    monkeypatch.setattr(MG.subprocess, "run", fake_inference)
    root, stage = tmp_path / "experiment", tmp_path / "stage"
    stage.mkdir()
    return root, stage, row, pre, env, calls


def test_model_grid_run_resume_and_output_tamper_refusal(tmp_path, monkeypatch):
    import json
    root, stage, row, pre, env, calls = _model_grid_case_fixture(tmp_path, monkeypatch)
    sources = dict(row["source_hashes"])
    MG.run_case(root, 0, stage)
    qc = MG.verify_qc(root, row, pre, env)
    assert qc["status"] == "pass" and all(v == 0 for v in qc["unequal_voxels"].values())
    MG.run_case(root, 0, stage)
    assert len(calls) == 1 and MG._source_hashes(row) == sources
    output = MG.case_folder(root, row) / "internal_venous.nii.gz"
    output.write_bytes(output.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="saved case output changed"):
        MG.run_case(root, 0, stage)
    assert len(calls) == 1
    assert MG.status(root)[0]["status"] == "invalid"
    with pytest.raises(ValueError, match="analysis blocked"):
        MG.analyze(root, 30, 11)
    assert json.loads((root / "analysis_status.json").read_text())["status"] == "blocked"
    assert not (root / "paired_connectivity.summary.json").exists()


@pytest.mark.parametrize("failure", ["inference_exit", "source_changed", "environment_changed"])
def test_model_grid_retains_failures_without_selecting_passed_cases(tmp_path, monkeypatch, failure):
    """A failed inference, a changed source or a changed environment is retained as `failed` and
    blocks the analysis; nothing is selected around it."""
    import json
    root, stage, row, pre, env, calls = _model_grid_case_fixture(tmp_path, monkeypatch,
        inference_exit=3 if failure == "inference_exit" else 0)
    if failure == "source_changed":
        Path(row["ct_path"]).write_bytes(b"changed")
    if failure == "environment_changed":
        monkeypatch.setattr(MG, "environment_record", lambda p: {"changed": True})
    with pytest.raises((ValueError, RuntimeError)):
        MG.run_case(root, 0, stage)
    qc = json.loads((MG.case_folder(root, row) / "qc.json").read_text())
    assert qc["status"] == "failed"
    assert (MG.case_folder(root, row) / "attempts.jsonl").is_file()
    if failure == "source_changed":
        assert not calls
    with pytest.raises(ValueError, match="analysis blocked"):
        MG.analyze(root, 30, 11)


def test_model_grid_boundary_voxel_differences_count_as_reproduced_when_the_class_agrees(tmp_path, monkeypatch):
    import json
    root, stage, row, pre, env, calls = _model_grid_case_fixture(tmp_path, monkeypatch, mismatch=True)
    MG.run_case(root, 0, stage)                       # no exception: the class is reproduced
    qc = MG.verify_qc(root, row, pre, env)
    assert qc["status"] == "mismatch" and qc["class_reproduced"] is True
    assert qc["unequal_voxels"] == {MG.NAMES[0]: 1, MG.NAMES[1]: 0, MG.NAMES[2]: 0}
    MG.run_case(root, 0, stage)                       # the resume verifies and skips it
    assert len(calls) == 1
    inventory = MG.status(root)
    assert inventory[0]["status"] == "mismatch" and inventory[0]["class_reproduced"] is True
    qc_status = json.loads((root / "qc_status.json").read_text())
    assert qc_status["status"] == "complete"
    assert qc_status["reproduction"] == {"exact": 0, "boundary": 1, "class_divergent": 0}
    monkeypatch.setattr(MG, "EXPECTED_NATIVE", {"direct": 1})
    MG.analyze(root, 30, 11)
    completion = json.loads((root / "analysis_status.json").read_text())
    assert completion["status"].startswith("complete_")
    assert completion["native_reproduction"]["exact_cases"] == 0
    assert completion["native_reproduction"]["boundary_cases"] == 1
    assert completion["native_reproduction"]["class_agreement"] == "1/1"
    assert completion["native_reproduction"]["max_unequal_voxels"][MG.NAMES[0]] == 1
    assert "reproduction_fidelity.csv" in completion["outputs"]
    fidelity = pd.read_csv(root / "reproduction_fidelity.csv")
    assert fidelity.reproduction.tolist() == ["boundary"]
    assert fidelity[f"unequal_voxels_{MG.NAMES[0]}"].tolist() == [1]
    assert fidelity[f"rerun_voxels_{MG.NAMES[0]}"].tolist() == [fidelity[f"archive_voxels_{MG.NAMES[0]}"].iloc[0] + 1]
    assert fidelity.archive_class26.tolist() == fidelity.rerun_native_class26.tolist() == ["direct"]
    # The recorded counts are re-derived, never trusted: a record claiming exactness is refused.
    qc_path = MG.case_folder(root, row) / "qc.json"
    forged = json.loads(qc_path.read_text())
    forged["unequal_voxels"] = {name: 0 for name in MG.NAMES}
    forged["status"] = "pass"
    MG.atomic_json(qc_path, forged)
    with pytest.raises(ValueError, match="independently detected"):
        MG.analyze(root, 30, 11)
    assert not (root / "paired_connectivity.summary.json").exists()
    # A record whose status and counts contradict each other is refused before any hash is read.
    forged["status"] = "mismatch"
    MG.atomic_json(qc_path, forged)
    with pytest.raises(ValueError, match="identity-bound native reproduction record"):
        MG.verify_qc(root, row, pre, env)


def test_model_grid_class_divergent_case_is_reported_and_kept_in_the_pairing(tmp_path, monkeypatch):
    import json
    root, stage, row, pre, env, calls = _model_grid_case_fixture(tmp_path, monkeypatch, mismatch="class")
    MG.run_case(root, 0, stage)                       # no exception
    qc = MG.verify_qc(root, row, pre, env)
    assert qc["status"] == "mismatch" and qc["class_reproduced"] is False
    assert qc["unequal_voxels"][MG.NAMES[0]] == 4 and qc["native_topology"]["class26"] == "disconnected"
    inventory = MG.status(root)
    assert inventory[0]["status"] == "mismatch" and inventory[0]["class_reproduced"] is False
    qc_status = json.loads((root / "qc_status.json").read_text())
    assert qc_status["status"] == "complete"
    assert qc_status["reproduction"] == {"exact": 0, "boundary": 0, "class_divergent": 1}
    monkeypatch.setattr(MG, "EXPECTED_NATIVE", {"direct": 1})   # the ARCHIVE's counts, which still hold
    MG.analyze(root, 30, 11)
    completion = json.loads((root / "analysis_status.json").read_text())
    assert completion["status"].startswith("complete_")
    reproduction = completion["native_reproduction"]
    assert reproduction["class_divergent_cases_n"] == 1
    assert reproduction["class_divergent_cases"] == [
        {"cohort": "AMOS", "case": "a", "participant": "AMOS:p", "adjacency": "26", "archive": "direct", "rerun": "disconnected"},
        {"cohort": "AMOS", "case": "a", "participant": "AMOS:p", "adjacency": "6", "archive": "direct", "rerun": "disconnected"}]
    assert reproduction["class_agreement"] == "0/1" and reproduction["class_agreement26"] == "0/1"
    assert reproduction["archive_native_class26_counts"] == {"direct": 1}
    assert reproduction["rerun_native_class26_counts"] == {"disconnected": 1}
    rows = pd.read_csv(root / "paired_connectivity.csv")
    assert rows.native_class26.tolist() == ["disconnected"] and rows.archive_class26.tolist() == ["direct"]
    assert rows.internal_class26.tolist() == ["direct"]
    fidelity = pd.read_csv(root / "reproduction_fidelity.csv")
    assert fidelity.reproduction.tolist() == ["class_divergent"]
    for name in ["model_grid_prevalence", "model_grid_paired", "model_grid_transitions"]:
        assert f"{name}_archive_class.csv" in completion["outputs"] and (root / f"{name}_archive_class.csv").is_file()
    main = pd.read_csv(root / "model_grid_transitions.csv")
    sensitivity = pd.read_csv(root / "model_grid_transitions_archive_class.csv")

    def pooled(table):
        part = table[(table.stratum == "Pooled") & (table.connectivity == 26) & (table.events > 0)]
        return part[["native", "internal"]].values.tolist()
    assert pooled(main) == [["disconnected", "direct"]] and pooled(sensitivity) == [["direct", "direct"]]


def test_model_grid_case_folder_keeps_the_complete_internal_capture(tmp_path, monkeypatch):
    import json

    import nibabel as nib
    root, stage, row, pre, env, _calls = _model_grid_case_fixture(tmp_path, monkeypatch)
    MG.run_case(root, 0, stage)
    folder = MG.case_folder(root, row)
    qc = MG.verify_qc(root, row, pre, env)
    assert set(qc["outputs"]) == {"native_multilabel.nii.gz", "internal_venous.nii.gz",
                                  "internal_multilabel.nii.gz", "internal_labels.json", "inference.log"}
    assert qc["internal_capture"]["cardiac_classes"] == 18
    full = np.asanyarray(nib.load(folder / "internal_multilabel.nii.gz").dataobj)
    compact = np.asanyarray(nib.load(folder / "internal_venous.nii.gz").dataobj)
    assert np.array_equal(compact, MG.compact_venous(full, row["label_indices"]))
    assert (full == row["label_indices"][0]).sum() == 8 and (full > 0).sum() > (compact > 0).sum()
    labels = json.loads((folder / "internal_labels.json").read_text())
    assert labels["cardiac_submodel"]["task_id"] == 293
    # A QC record without the capture inventory is refused before any hash is read.
    stripped = dict(qc)
    stripped.pop("internal_capture")
    MG.atomic_json(folder / "qc.json", stripped)
    with pytest.raises(ValueError, match="complete internal capture record"):
        MG.verify_qc(root, row, pre, env)
    MG.atomic_json(folder / "qc.json", qc)
    # A tampered complete map is refused exactly as a tampered venous file is.
    target = folder / "internal_multilabel.nii.gz"
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="saved case output changed"):
        MG.verify_qc(root, row, pre, env)


@pytest.mark.parametrize("capture,message", [("venous_only", "complete internal capture"),
                                             ("compact_mismatch", "disagrees"),
                                             ("sidecar_mismatch", "sidecar disagrees")])
def test_model_grid_run_refuses_an_incomplete_or_inconsistent_capture(tmp_path, monkeypatch, capture, message):
    import json
    root, stage, row, _pre, _env, _calls = _model_grid_case_fixture(tmp_path, monkeypatch, capture=capture)
    with pytest.raises(ValueError, match=message):
        MG.run_case(root, 0, stage)
    qc = json.loads((MG.case_folder(root, row) / "qc.json").read_text())
    assert qc["status"] == "failed"
    with pytest.raises(ValueError, match="analysis blocked"):
        MG.analyze(root, 30, 11)


def test_model_grid_analysis_recomputes_output_topology_and_publishes_last(tmp_path, monkeypatch):
    import json
    root, stage, row, pre, env, calls = _model_grid_case_fixture(tmp_path, monkeypatch)
    MG.run_case(root, 0, stage)
    monkeypatch.setattr(MG, "EXPECTED_NATIVE", {"direct": 1})
    MG.analyze(root, 30, 11)
    completion = json.loads((root / "analysis_status.json").read_text())
    assert completion["processed"] == 1 and completion["status"].startswith("complete_")
    assert completion["internal_capture"] == "complete total class map with venous compact codes and label sidecar"
    assert completion["native_reproduction"]["exact_cases"] == 1 and completion["native_reproduction"]["boundary_cases"] == 0
    assert pd.read_csv(root / "reproduction_fidelity.csv").reproduction.tolist() == ["exact"]
    for path, digest in completion["outputs"].items():
        assert MG.sha(root / path) == digest
    rows = pd.read_csv(root / "paired_connectivity.csv")
    assert rows.native_class26.tolist() == rows.internal_class26.tolist() == ["direct"]
    qc_path = MG.case_folder(root, row) / "qc.json"
    qc = json.loads(qc_path.read_text())
    qc["status"] = "failed"
    MG.atomic_json(qc_path, qc)
    with pytest.raises(ValueError, match="analysis blocked"):
        MG.analyze(root, 30, 11)
    assert not (root / "paired_connectivity.summary.json").exists()


def test_model_grid_analysis_independently_rechecks_the_complete_capture(tmp_path, monkeypatch):
    import json

    import nibabel as nib
    root, stage, row, _pre, _env, _calls = _model_grid_case_fixture(tmp_path, monkeypatch)
    MG.run_case(root, 0, stage)
    folder = MG.case_folder(root, row)
    complete = folder / "internal_multilabel.nii.gz"
    image = nib.load(complete)
    data = np.asanyarray(image.dataobj).copy()
    data[0, 0, 0] = row["label_indices"][1]
    nib.save(nib.Nifti1Image(data, image.affine), complete)
    qc = json.loads((folder / "qc.json").read_text())
    qc["outputs"][complete.name] = MG.sha(complete)
    MG.atomic_json(folder / "qc.json", qc)
    with pytest.raises(ValueError, match="internal venous/complete capture disagreement"):
        MG.analyze(root, 30, 11)
    assert not (root / "paired_connectivity.summary.json").exists()


def test_model_grid_analysis_does_not_trust_a_forged_voxel_equality_marker(tmp_path, monkeypatch):
    import json
    import nibabel as nib
    root, stage, row, pre, env, calls = _model_grid_case_fixture(tmp_path, monkeypatch)
    MG.run_case(root, 0, stage)
    folder = MG.case_folder(root, row)
    native = folder / "native_multilabel.nii.gz"
    image = nib.load(native)
    data = np.asanyarray(image.dataobj).copy()
    data[0, 0, 0] = row["label_indices"][0]
    nib.save(nib.Nifti1Image(data, image.affine), native)
    qc = json.loads((folder / "qc.json").read_text())
    qc["outputs"][native.name] = MG.sha(native)
    MG.atomic_json(folder / "qc.json", qc)
    with pytest.raises(ValueError, match="independently detected"):
        MG.analyze(root, 30, 11)
    assert not (root / "paired_connectivity.summary.json").exists()


@pytest.mark.parametrize("corrupt", [False, True])
def test_model_grid_preflight_checks_canonical_binary_bytes_without_omitting_failures(tmp_path, monkeypatch, corrupt):
    import json
    import nibabel as nib
    row, folder = _write_multilabel_fixture(tmp_path)
    report = json.loads((folder / "report.json").read_text())
    ct = tmp_path / "ct_source" / "CT.nii.gz"
    ct.parent.mkdir()
    native = nib.load(folder / "segmentation.nii.gz")
    nib.save(nib.Nifti1Image(np.zeros(native.shape, np.int16), native.affine), ct)
    report["input"] = str(ct)
    (folder / "report.json").write_text(json.dumps(report))
    original = {p.name: p.read_bytes() for p in folder.iterdir()}
    prepared = {"fingerprint": "fixture-environment"}
    monkeypatch.setattr(MG, "prepare", lambda root, weights_root=None: prepared)
    canonical = {k: row[k] for k in ["cohort", "case", "mask_case", "participant"]}
    canonical.update({"expected_class26": row["class26"], "expected_class6": row["class6"]})
    # Hashes are generated by the original exporter and intentionally nonstandard label IDs.
    outputs, error = FS.EM.export_case(("X", "a", str(tmp_path / "multilabel"), str(tmp_path / "reference"), True, "", None), structures=MG.NAMES)
    assert error is None
    canonical.update({x["structure"] + "_sha256": x["sha256"] for x in outputs})
    if corrupt:
        canonical[MG.NAMES[0] + "_sha256"] = "0" * 64
    monkeypatch.setattr(MG, "canonical_frame", lambda a, b: [dict(canonical)])
    stage, root = tmp_path / "stage", tmp_path / "experiment"
    stage.mkdir()
    args = [root, FS.OUT / "analysis_cases.csv", FS.OUT / "participant_mapping.csv",
            tmp_path / "multilabel", tmp_path, stage]
    if corrupt:
        with pytest.raises(ValueError, match="preflight failed"):
            MG.preflight(*args)
        assert json.loads((root / "preflight.failed.json").read_text())["verified"] == 0
        assert not (root / "cases.json").exists()
    else:
        MG.preflight(*args)
        verified = json.loads((root / "cases.json").read_text())
        assert len(verified) == 1 and verified[0]["label_indices"] == [13, 42, 99]
        assert verified[0]["source_hashes"] == MG._source_hashes(verified[0])
        assert json.loads((root / "preflight.json").read_text())["manifest_sha256"] == MG.sha(root / "cases.json")
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == original


def test_model_grid_contract_rejects_manifest_code_and_roster_changes(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    root.mkdir()
    artifact = tmp_path / "roster.csv"
    artifact.write_text("roster")
    code = tmp_path / "code.py"
    code.write_text("code")
    monkeypatch.setattr(MG.ST, "REPO_ROOT", tmp_path)
    rows = [{"cohort": c, "case": f"c{i}"} for c, count in MG.EXPECTED.items() for i in range(count)]
    MG.atomic_json(root / "cases.json", rows)
    env = {"identity": {"code": {"code.py": MG.sha(code)}}}
    env["fingerprint"] = MG.identity(env["identity"])
    MG.atomic_json(root / "environment.json", env)
    pre = {"status": "pass", "verified": 1899, "manifest_sha256": MG.sha(root / "cases.json"),
           "environment_fingerprint": env["fingerprint"], "canonical_inputs": {str(artifact): MG.sha(artifact)}}
    MG.atomic_json(root / "preflight.json", pre)
    assert len(MG.load_contract(root)[0]) == 1899
    for path, message in [(code, "code changed"), (artifact, "roster changed"), (root / "cases.json", "preflight")]:
        original = path.read_bytes()
        path.write_bytes(original + b"changed")
        with pytest.raises(ValueError, match=message):
            MG.load_contract(root)
        path.write_bytes(original)


def test_model_grid_code_supersession_is_recorded_bound_and_the_only_way_past_a_code_change(tmp_path, monkeypatch):
    """Editing environment.json is never the way."""
    import json
    root = tmp_path / "experiment"
    root.mkdir()
    code = tmp_path / "code.py"
    code.write_text("code")
    other = tmp_path / "other.py"
    other.write_text("other")
    monkeypatch.setattr(MG.ST, "REPO_ROOT", tmp_path)
    rows = [{"cohort": c, "case": f"c{i}"} for c, count in MG.EXPECTED.items() for i in range(count)]
    MG.atomic_json(root / "cases.json", rows)
    env = {"identity": {"code": {"code.py": MG.sha(code), "other.py": MG.sha(other)}}}
    env["fingerprint"] = MG.identity(env["identity"])
    MG.atomic_json(root / "environment.json", env)
    pre = {"status": "pass", "verified": 1899, "manifest_sha256": MG.sha(root / "cases.json"),
           "environment_fingerprint": env["fingerprint"], "canonical_inputs": {}}
    MG.atomic_json(root / "preflight.json", pre)
    with pytest.raises(ValueError, match="nothing to supersede"):
        MG.supersede(root, "reason", "decision")
    code.write_text("code changed after the freeze")
    with pytest.raises(ValueError, match="code changed after preflight"):
        MG.load_contract(root)
    with pytest.raises(ValueError, match="needs its reason and decision"):
        MG.supersede(root, "  ", "decision")
    with pytest.raises(ValueError, match="needs its reason and decision"):
        MG.supersede(root, "reason", "")
    record = MG.supersede(root, "class-level gate", "the recorded decision")
    assert record["changed"] == ["code.py"] and record["live_code"]["code.py"] == MG.sha(code)
    assert record["live_code"]["other.py"] == MG.sha(other)
    assert record["frozen_code"] == env["identity"]["code"] and record["environment_fingerprint"] == env["fingerprint"]
    assert json.loads((root / MG.SUPERSESSION).read_text()) == record
    assert len(MG.load_contract(root)[0]) == 1899
    with pytest.raises(ValueError, match="already superseded"):
        MG.supersede(root, "again", "decision")
    # A further change after the supersession is refused again; one record per root.
    code.write_text("changed twice")
    with pytest.raises(ValueError, match="after its recorded supersession"):
        MG.load_contract(root)
    code.write_text("code changed after the freeze")
    # A record for another experiment, or an inconsistent one, is refused.
    saved = (root / MG.SUPERSESSION).read_text()
    forged = json.loads(saved)
    forged["environment_fingerprint"] = "other"
    MG.atomic_json(root / MG.SUPERSESSION, forged)
    with pytest.raises(ValueError, match="does not belong"):
        MG.load_contract(root)
    forged = json.loads(saved)
    forged["changed"] = []
    MG.atomic_json(root / MG.SUPERSESSION, forged)
    with pytest.raises(ValueError, match="inconsistent"):
        MG.load_contract(root)
    forged = json.loads(saved)
    del forged["reason"]
    MG.atomic_json(root / MG.SUPERSESSION, forged)
    with pytest.raises(ValueError, match="malformed"):
        MG.load_contract(root)
    (root / MG.SUPERSESSION).write_text(saved)
    assert len(MG.load_contract(root)[0]) == 1899


def test_model_grid_qc_and_completion_carry_the_code_supersession(tmp_path, monkeypatch):
    """Cases run and analyses published after a supersession say so; without the record the same
    live code is a changed environment and the case is refused."""
    import json
    root, stage, row, pre, env, calls = _model_grid_case_fixture(tmp_path, monkeypatch)
    frozen = {"model_grid_audit.py": "a" * 64}
    live = {"model_grid_audit.py": "b" * 64}
    env["identity"]["code"] = frozen
    env["fingerprint"] = MG.identity(env["identity"])
    monkeypatch.setattr(MG, "environment_record", lambda path: {**env["identity"], "code": live})
    root.mkdir()
    MG.atomic_json(root / MG.SUPERSESSION, {
        "schema": 1, "environment_fingerprint": env["fingerprint"], "frozen_code": frozen, "live_code": live,
        "changed": ["model_grid_audit.py"], "reason": "class-level gate", "decision": "the recorded decision",
        "recorded_unix": 0})
    MG.run_case(root, 0, stage)
    qc = json.loads((MG.case_folder(root, row) / "qc.json").read_text())
    assert qc["status"] == "pass" and qc["code_supersession"] == MG.sha(root / MG.SUPERSESSION)
    monkeypatch.setattr(MG, "EXPECTED_NATIVE", {"direct": 1})
    MG.analyze(root, 30, 11)
    completion = json.loads((root / "analysis_status.json").read_text())
    assert completion["code_supersession"]["changed"] == ["model_grid_audit.py"]
    assert completion["code_supersession"]["reason"] == "class-level gate"
    (root / MG.SUPERSESSION).unlink()
    (MG.case_folder(root, row) / "qc.json").unlink()
    with pytest.raises(ValueError, match="changed"):
        MG.run_case(root, 0, stage)


def test_model_grid_source_patch_has_exactly_one_inert_capture_expression(monkeypatch, tmp_path):
    import ast
    source = '''def pipeline(img_pred, task_name, output_task_name, resample, save_lowres, crop, roi_subset, remove_small_blobs, test):
        img_pred = nib.load(tmp_dir / "s01.nii.gz")
        if resample is not None and not save_lowres:
            project(img_pred)
'''
    monkeypatch.setattr(MG, "PRISTINE_SHA256", hashlib.sha256(source.encode()).hexdigest())
    patched = MG.patched_source(source)
    tree = ast.parse(patched)
    expression = tree.body[0].body.pop(1)
    assert expression.value.func.id == "_ivcs_capture_model_grid"
    assert ast.dump(tree, include_attributes=False) == ast.dump(ast.parse(source), include_attributes=False)
    events = []
    # Execute the complete small producer: capture must precede projection, with the
    # exact same object and global class map. No AST approximation of the hook call.
    image, mapping = object(), {1: "inferior_vena_cava"}
    from types import SimpleNamespace
    namespace = {"nib": SimpleNamespace(load=lambda path: image), "tmp_dir": tmp_path,
                 "class_map": {"total": mapping},
                 "project": lambda img: events.append(("project", img)),
                 "_ivcs_capture_model_grid": lambda img, ctx: events.append(("capture", img, ctx["class_map"]))}
    exec(compile(patched, "fixture.py", "exec"), namespace)
    namespace["pipeline"](None, "total", "total", 1.5, False, None, None, False, 0)
    assert events == [("capture", image, {"total": mapping}), ("project", image)]


def test_model_grid_worker_refuses_cpu_fallback(tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(MG.importlib.metadata, "version", lambda name: "2.17.0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="no CPU fallback"):
        MG.inference_worker(tmp_path / "ct.nii.gz", tmp_path / "native.nii.gz", tmp_path / "internal.nii.gz")
    assert not (tmp_path / "internal.nii.gz").exists()


@pytest.mark.needs_metadata
def test_saros_participant_counts_do_not_treat_empty_scan_as_a_second_person():
    meta = ST.load_meta("saros")
    assert ST.participant_key("saros", "case_283", meta) == ST.participant_key(
        "saros", "case_812", meta
    )
    row = ST.cohort_table1_row(
        "saros",
        [("case_812", 12, 8, 12, "OK")],
        {"empty_ids": ["case_283"], "empty": 1},
        meta,
        {},
    )
    assert row["n_analysis_units"] == 2
    assert row["n_participants"] == row["sex_available_n"] == 1
    with pytest.raises(ValueError, match="missing SAROS participant"):
        ST.participant_key("saros", "unmapped", meta)


def test_tcia_patient_identity_is_collection_scoped(tmp_path):
    path = tmp_path / "meta.csv"
    path.write_text(
        "id,tcia_collection,tcia_case_id,gender\ncase_1,A,007,female\ncase_2,B,007,male\n"
    )
    meta = ST._meta_saros(path)
    assert ST.participant_key("saros", "case_1", meta) != ST.participant_key(
        "saros", "case_2", meta
    )


def test_icc_preserves_negative_anova_estimates_and_undefined_constants():
    # Both people have mean 1; all spread is within-person. In a balanced
    # two-visit design MS_between=0 and MS_within=2, giving ICC=-1.
    assert FS.one_way_icc([np.array([0.0, 2.0]), np.array([1.0, 1.0])]) == -1
    assert np.isnan(FS.one_way_icc([np.zeros(2), np.zeros(2)]))
    assert FS.one_way_icc([np.array([1.0, 1.0]), np.array([5.0, 5.0])]) == 1


def test_degenerate_icc_bootstrap_is_an_explicit_failure(monkeypatch):
    monkeypatch.setattr(FS, "N_BOOT", 1)

    class DegenerateDraw:
        def integers(self, low, high, size):
            return np.zeros(size, dtype=int)

    monkeypatch.setattr(FS.np.random, "default_rng", lambda seed: DegenerateDraw())
    d = pd.DataFrame(
        {
            "cohort": ["X"] * 7,
            "participant": ["a", "a", "b", "b", "c", "c", "d"],
            "gap_mm": [0.0, 0.0, 2.0, 4.0, 5.0, 7.0, 99.0],
        }
    )
    # Resampling only the zero-mean person gives an undefined ICC, never a
    # fabricated zero or a quietly discarded replicate.
    with pytest.raises(ValueError, match="undefined ICC bootstrap"):
        FS.repeat_summary(d)


def test_raw_topology_survives_a_failed_resplit_and_requires_original_mask_hashes(
    tmp_path, monkeypatch
):
    seg = tmp_path / "SAROS" / "case_001"
    seg.mkdir(parents=True)
    records = []
    for cls, slices in (
        ("ivc", slice(5, 7)),
        ("lciv", slice(0, 2)),
        ("rciv", slice(2, 5)),
    ):
        a = np.zeros((7, 3, 3), np.uint8)
        a[slices, 1, 1] = 1
        path = seg / RC.FILES[cls]
        sitk.WriteImage(sitk.GetImageFromArray(a), str(path))
        records.append(
            {
                "dataset": "SAROS",
                "case": "case_001",
                "structure": path.name.removesuffix(".nii.gz"),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    pd.DataFrame(records).to_csv(tmp_path / "MANIFEST.csv", index=False)
    monkeypatch.setattr(ST, "LAYOUT", {"saros": ("SAROS", "saros", "SAROS.csv")})
    monkeypatch.setattr(ST, "validate_result_roster", lambda *a: None)
    monkeypatch.setattr(
        ST,
        "load_fullres",
        lambda *a: ([("case_001", 3, 3, 4, "OK")], {"metric": "surface(v2)"}),
    )
    monkeypatch.setattr(
        ST, "load_collapsed", lambda *a: ([], {"unparsed": 0, "failed": 1})
    )
    out = tmp_path / "raw.csv"
    RC.recover(tmp_path, out)
    row = pd.read_csv(out).iloc[0]
    assert row.union_connected == 1 and row.direct_connected == 0
    with (seg / RC.FILES["rciv"]).open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        RC.recover(tmp_path, out)


def test_models_cluster_by_participant_and_match_unweighted_point_estimates():
    data = pd.read_csv(FS.OUT / "analysis_cases.csv")
    coef, _, partition = FS.model_results(data)
    x = np.column_stack(
        [
            np.ones(len(data)),
            data.cohort == "SAROS",
            data.cohort == "autoPET",
            data.slice_mm,
        ]
    ).astype(float)
    expected = np.linalg.lstsq(x, data.gap_mm.to_numpy(), rcond=None)[0]
    observed = coef[(coef.model == "pooled") & (coef.endpoint == "gap_mm")]
    assert np.allclose(observed.estimate, expected)
    assert set(observed.n_participants) == {1759}  # 493 + 366 + 900, not 1873 scans
    non_intercepts = coef.term != "Intercept"
    assert non_intercepts.sum() == 46
    assert set(coef.multiplicity_family_n) == {46}
    assert coef.loc[~non_intercepts, ["q_bh", "q_by"]].isna().all().all()
    for column, method in (("q_bh", "fdr_bh"), ("q_by", "fdr_by")):
        expected_q = FS.multipletests(coef.loc[non_intercepts, "p"], method=method)[1]
        assert np.allclose(coef.loc[non_intercepts, column], expected_q)
    categorical = coef[coef.model == "pooled_spacing_categorical"]
    assert set(categorical.n) == {1873, 1467}
    assert set(categorical[categorical.term.str.startswith("C(spacing_category)")].term) == {
        "C(spacing_category)[T.gt1p5_le3]", "C(spacing_category)[T.gt3]"}
    # Independent Boolean design verifies the intended closed cut points without pd.cut.
    xcat = np.column_stack([np.ones(len(data)),
        (data.slice_mm > 1.5) & (data.slice_mm <= 3), data.slice_mm > 3,
        data.cohort == "SAROS", data.cohort == "autoPET"]).astype(float)
    expected_cat = np.linalg.lstsq(xcat, data.gap_mm.to_numpy(), rcond=None)[0]
    observed_cat = categorical[categorical.endpoint == "gap_mm"]
    assert np.allclose(observed_cat.estimate, expected_cat)
    p = partition.iloc[0]
    assert p.unique_cohort_r2 + p.unique_spacing_r2 + p.shared_r2 == pytest.approx(
        p.full_r2
    )


# Topology endpoints, reference-archive acquisition, grid-resampling and mask-export checks.
import io
import zipfile

import export_relevant_masks as EM
import measure_collapsed_gap as MC
import nibabel as nib


def test_binary_agreement_collapses_label_swaps_but_preserves_actual_disconnection(
    monkeypatch,
):
    monkeypatch.setattr(FS, "N_BOOT", 500)

    class FixedDraw:
        def integers(self, low, high, size):
            return np.tile(np.arange(high), (size[0], 1))

    monkeypatch.setattr(FS.np.random, "default_rng", lambda seed: FixedDraw())
    frame = pd.DataFrame(
        {
            "participant": ["a", "a", "b", "b"],
            "class26": [
                "direct",
                "right_iliac_mediated",
                "disconnected",
                "disconnected",
            ],
        }
    )
    binary, _ = FS.longitudinal_agreement(
        frame, 2, {"direct": 0, "right_iliac_mediated": 0, "disconnected": 1}
    )
    assert binary.set_index("metric").loc["agreement", "estimate"] == 1


def test_participant_balancing_differs_from_pooling_all_visit_pairs(monkeypatch):
    monkeypatch.setattr(FS, "N_BOOT", 500)
    frame = pd.DataFrame(
        {
            "participant": ["a", "a", "b", "b", "b"],
            "class26": ["direct", "disconnected", "direct", "direct", "disconnected"],
        }
    )
    result, detail = FS.longitudinal_agreement(
        frame, 2, {"direct": 0, "right_iliac_mediated": 0, "disconnected": 1}
    )
    assert result.set_index("metric").loc["agreement", "estimate"] == pytest.approx(
        1 / 6
    )
    assert detail.pairs.sum() == 4
    assert np.average(detail.pair_agreement, weights=detail.pairs) == pytest.approx(
        1 / 4
    )


def test_partial_zip_entry_is_verified_and_corrupt_cached_bytes_are_replaced(
    tmp_path, monkeypatch
):
    payload = b"reference label bytes" * 100
    name = "s0001/segmentations/inferior_vena_cava.nii.gz"
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, payload)
    raw = stream.getvalue()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        info = archive.getinfo(name)
    entry = {
        "name": name,
        "offset": info.header_offset,
        "compressed_size": info.compress_size,
        "uncompressed_size": info.file_size,
        "crc32": info.CRC,
        "method": info.compress_type,
    }
    monkeypatch.setattr(
        EM,
        "get_range",
        lambda start, end: raw[start : end + 1] + bytes(max(0, end + 1 - len(raw))),
    )
    EM.fetch_group((tmp_path, [entry]))
    assert (tmp_path / name).read_bytes() == payload
    (tmp_path / name).write_bytes(b"corrupt cached label")
    EM.fetch_group((tmp_path, [entry]))
    assert (tmp_path / name).read_bytes() == payload
    corrupted = dict(entry, crc32=0)
    with pytest.raises(ValueError, match="CRC32"):
        EM.fetch_group((tmp_path, [corrupted]))


def test_public_reference_empty_labels_are_excluded_and_spacing_matches_array_axes(
    tmp_path,
):
    folder = tmp_path / "s0001/segmentations"
    folder.mkdir(parents=True)
    masks = [np.zeros((10, 4, 4), np.uint8) for _ in range(3)]
    masks[0][1:3, 1, 1] = 1
    masks[1][5:7, 1, 1] = 1
    # Native X spacing is 2 mm: two empty X voxels imply a 4-mm surface gap.
    affine = np.diag([2.0, 1.0, 3.0, 1.0])
    for name, mask in zip(MC.REFERENCE_VEIN_FILES, masks):
        nib.save(nib.Nifti1Image(mask, affine), folder / f"{name}.nii.gz")
    result = MC.audit_reference_case((tmp_path, "s0001"))
    assert result["status"] == "measurable"
    assert result["gap_mm"] == 4.0
    assert result["class26"] == "disconnected"
    sheared = affine.copy()
    sheared[0, 1] = 0.2
    for name, mask in zip(MC.REFERENCE_VEIN_FILES, masks):
        nib.save(nib.Nifti1Image(mask, sheared), folder / f"{name}.nii.gz")
    result = MC.audit_reference_case((tmp_path, "s0001"))
    assert result["status"] == "measurable"
    assert result["class26"] == "disconnected"
    assert result["distance_status"] == "unsupported_nonorthonormal_grid"
    assert np.isnan(result["gap_mm"])
    nib.save(
        nib.Nifti1Image(np.zeros_like(masks[0]), sheared),
        folder / f"{MC.REFERENCE_VEIN_FILES[0]}.nii.gz",
    )
    assert MC.audit_reference_case((tmp_path, "s0001"))["status"] == "empty_primary_label"


def test_server_ignoring_range_is_rejected_before_reading_archive(monkeypatch):
    class Response:
        status = 200
        headers = None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, *args):
            raise AssertionError("must never read full archive")

    monkeypatch.setattr(EM.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="full archive"):
        EM.get_range(10, 20)


def empty():
    return [np.zeros((15, 15, 15), bool) for _ in range(3)]


def test_stray_island_does_not_disconnect_main_trunks():
    a, b, c = empty()
    a[1:5, 2, 2] = True
    b[5:9, 2, 2] = True
    c[12, 12, 12] = True
    r = MC.classify_topology([a, b, c])
    assert r["ncomp26"] == 2
    assert r["class26"] == "direct"


def test_tiny_cross_class_contact_does_not_connect_disconnected_main_trunks():
    a, b, c = empty()
    a[1:5, 2, 2] = True
    b[9:13, 2, 2] = True
    b[5, 2, 2] = True  # minor LCIV island touches IVC
    r = MC.classify_topology([a, b, c])
    assert r["any_path26"] == 1
    assert r["main_path26"] == 0
    assert r["class26"] == "disconnected"


def test_diagonal_contact_is_accepted_only_by_26_connectivity():
    a, b, c = empty()
    a[1, 1, 1] = True
    b[2, 2, 2] = True
    r = MC.classify_topology([a, b, c])
    assert r["class26"] == "direct"
    assert r["class6"] == "disconnected"


def test_right_iliac_mediation_requires_a_path_in_the_three_vein_union():
    a, b, c = empty()
    a[1:4, 2, 2] = True
    b[7:10, 2, 2] = True
    c[4:7, 2, 2] = True
    r = MC.classify_topology([a, b, c])
    assert r["class26"] == r["class6"] == "right_iliac_mediated"


def test_empty_primary_is_not_counted_as_disconnection():
    a, b, c = empty()
    a[1, 1, 1] = True
    with pytest.raises(ValueError, match="empty primary"):
        MC.classify_topology([a, b, c])


@pytest.mark.parametrize("left_start, expected", [(4, "direct"), (9, "disconnected")])
def test_empty_right_label_precludes_only_mediation(left_start, expected):
    a, b, c = empty()
    a[1:4, 2, 2] = True
    b[left_start:left_start + 3, 2, 2] = True
    r = MC.classify_topology([a, b, c])
    assert r["class26"] == r["class6"] == expected


@pytest.mark.parametrize("point, touches", [((0, 3, 3), 1), ((6, 3, 3), 1),
    ((3, 0, 3), 1), ((3, 6, 3), 1), ((3, 3, 0), 1), ((3, 3, 6), 1),
    ((0, 0, 0), 1), ((3, 3, 3), 0), (None, 0)])
def test_reference_boundary_rule_includes_every_face_and_minor_fragments(tmp_path, point, touches):
    folder = tmp_path / "s0001/segmentations"
    folder.mkdir(parents=True)
    masks = [np.zeros((7, 7, 7), np.uint8) for _ in range(3)]
    masks[0][1:3, 2, 2] = 1
    if point is not None:
        masks[1][3:5, 2, 2] = 1  # dominant left component, away from every face
        masks[1][point] = 1  # even a minor foreground fragment counts
    for name, mask in zip(MC.REFERENCE_VEIN_FILES, masks):
        nib.save(nib.Nifti1Image(mask, np.eye(4)), folder / f"{name}.nii.gz")
    row = MC.audit_reference_case((tmp_path, "s0001"))
    assert row["lciv_touches_image_boundary"] == touches
    assert row["status"] == ("empty_primary_label" if point is None else "measurable")


def test_annotation_prevalence_uses_split_specific_wilson_intervals_and_same_boundary_rule():
    reference = pd.DataFrame({"status": ["measurable"] * 4 + ["empty_primary_label"],
        "split": ["train", "train", "val", "test", "test"],
        "lciv_touches_image_boundary": [0, 1, 0, 1, 0],
        "class26": ["direct", "disconnected", "disconnected", "direct", None],
        "class6": ["direct", "disconnected", "disconnected", "direct", None]})
    rows = FS.reference_summaries(reference)
    disconnected = rows[(rows.connectivity == 26) & (rows["class"] == "disconnected")].set_index("stratum")
    all_ = disconnected.loc["reference_all"]
    assert (all_.events, all_.n) == (2, 4)
    assert np.allclose([all_.ci_lo, all_.ci_hi], FS.S.wilson_ci(2, 4, FS.CFG.ALPHA))
    train = disconnected.loc["reference_train_no_lciv_image_boundary_contact"]
    assert (train.events, train.n) == (0, 1) and train.ci_hi > 0
    val = disconnected.loc["reference_val_no_lciv_image_boundary_contact"]
    assert (val.events, val.n) == (1, 1) and val.ci_lo < 1
    absent = disconnected.loc["reference_test_no_lciv_image_boundary_contact"]
    assert absent.n == 0 and np.isnan(absent.proportion) and np.isnan(absent.ci_lo)
    assert rows.participants.isna().all()  # no invented released-patient linkage


def test_missing_label_bounds_apply_to_processed_examinations_only():
    cases = pd.DataFrame({"cohort": ["X"] * 3, "class26": ["direct", "disconnected", "disconnected"]})
    flow = pd.DataFrame({"cohort": ["X"], "processed": [5], "measurable": [3], "empty": [2],
                         "earlier_coverage_exclusions": [100]})
    row = FS.add_missing_label_bounds(cases, flow).iloc[0]
    assert (row.no_path_lower_n, row.no_path_upper_n, row.no_path_bound_n) == (2, 4, 5)
    assert row.no_path_lower_prop == .4 and row.no_path_upper_prop == .8
    with pytest.raises(ValueError, match="processed frame"):
        FS.add_missing_label_bounds(cases.iloc[:2], flow)


@pytest.mark.needs_metadata
def test_earliest_selection_is_chronological_and_followup_uses_consecutive_dates():
    data = pd.read_csv(FS.OUT / "analysis_cases.csv")
    d = FS.add_examination_dates(data.sample(frac=1, random_state=7))
    pet = d[d.cohort == "autoPET"]
    assert pet.earliest_examination.sum() == pet.participant.nunique() == 900
    for _, group in pet.groupby("participant"):
        chosen = group[group.earliest_examination == 1].iloc[0]
        assert (chosen.study_date, chosen.case) == min(zip(group.study_date, group.case))
    bad_meta = pd.read_csv(ST.META_DIR / "autoPET.csv")
    bad_meta["Study Date"] = "1900-01-01"
    with pytest.raises(ValueError, match="released CT study metadata"):
        FS.add_examination_dates(d, bad_meta)
    repeated = pd.DataFrame({"cohort": ["autoPET"] * 3, "participant": ["p"] * 3,
                             "study_date": ["2000-01-31", "2000-01-01", "2000-01-11"]})
    detail, summary = FS.add_followup_intervals(repeated, pd.DataFrame({"participant": ["p"]}),
                                               pd.DataFrame({"cohort": ["autoPET"]}))
    assert FS.json.loads(detail.iloc[0].consecutive_intervals_days) == [10, 20]
    assert summary.iloc[0].consecutive_interval_n == 2
    assert summary.iloc[0].consecutive_interval_median_days == 15


@pytest.mark.needs_metadata
def test_any_path_endpoint_sensitivity_reconciles_every_cohort_and_changed_case(monkeypatch):
    data = pd.read_csv(FS.OUT / "analysis_cases.csv")
    # Keep this accounting test fast; confidence interval algorithms are tested separately.
    original = FS.proportion_intervals
    monkeypatch.setattr(FS, "proportion_intervals", lambda frame, selector:
                        original(frame, selector, n_boot=100))
    data = FS.add_examination_dates(data)
    table = FS.topology_summaries(data)
    changed = data[data.any_path26 != data.main_path26]
    zero_not_direct = data[(data.gap_mm == 0) & (data.class26 != "direct")]
    assert len(changed) == 33
    assert set(zip(zero_not_direct.cohort, zero_not_direct.case)) == set(
        zip(changed[changed.gap_mm == 0].cohort, changed[changed.gap_mm == 0].case))
    assert len(zero_not_direct) == 22
    assert ((changed.gap_mm > 0) & (changed.any_path26 == 1)).sum() == 11
    row = table[(table.stratum == "all__any_endpoint26") & (table["class"] == "disconnected")].iloc[0]
    assert (row.events, row.n) == (859, 1873)
    for cohort, frame in data.groupby("cohort"):
        for suffix, selected in [("__thin_le_1p5mm", frame[frame.slice_mm <= 1.5]),
                                  ("__no_lciv_image_boundary_contact", frame[frame.lciv_touches_image_boundary == 0]),
                                  ("__earliest_examination", frame[frame.earliest_examination == 1])]:
            r=table[(table.stratum==cohort+suffix)&(table.connectivity==26)&(table["class"]=="disconnected")].iloc[0]
            assert r.n == len(selected)
            assert r.events == (selected.class26 == "disconnected").sum()


def test_class_specific_agreement_uses_participant_balanced_joint_distribution(monkeypatch):
    monkeypatch.setattr(FS, "N_BOOT", 500)
    frame = pd.DataFrame({"participant": ["a"] * 2 + ["b"] * 3,
        "class26": ["direct", "disconnected", "direct", "direct", "disconnected"]})
    result, _ = FS.longitudinal_agreement(frame, 2,
        {"direct": 0, "right_iliac_mediated": 0, "disconnected": 1})
    result = result.set_index("metric")
    assert result.loc["specific_agreement_connected", "estimate"] == pytest.approx(2 / 7)
    assert result.loc["specific_agreement_disconnected", "estimate"] == 0
    reordered, _ = FS.longitudinal_agreement(frame.iloc[::-1], 2,
        {"direct": 0, "right_iliac_mediated": 0, "disconnected": 1})
    pd.testing.assert_frame_equal(result, reordered.set_index("metric"))


def test_bootstrap_point_estimate_is_scan_weighted_and_blocks_preserve_repeats():
    d = pd.DataFrame({"cohort": ["X"] * 4, "participant": ["a", "a", "b", "b"]})
    result = FS.proportion_intervals(d, [1, 0, 1, 0], n_boot=100)
    # Every participant block has the same 1/2 proportion: a row bootstrap would vary.
    assert result["proportion"] == result["ci_lo"] == result["ci_hi"] == 0.5
    assert result["participants"] == 2


def test_wilson_independent_small_cells_and_zero_events_are_non_degenerate():
    frame = pd.DataFrame({"cohort": ["AMOS"] * 17, "participant": range(17)})
    result = FS.proportion_intervals(frame, [1] * 15 + [0] * 2)
    # Direct score inversion, independent of the production helper.
    z = FS.scipy.stats.norm.ppf(.975)
    p, n = 15/17, 17
    center = (p + z*z/(2*n))/(1+z*z/n)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    assert result["ci_lo"] == pytest.approx(center-half)
    assert result["ci_hi"] == pytest.approx(center+half)
    zero = FS.proportion_intervals(frame, np.zeros(17))
    assert zero["ci_hi"] > zero["ci_lo"]


def test_disconnection_logistic_and_centered_interaction_match_independent_designs():
    data = pd.read_csv(FS.OUT / "analysis_cases.csv")
    coef, _, _ = FS.model_results(data)
    x = np.column_stack([np.ones(len(data)),
        (data.slice_mm > 1.5) & (data.slice_mm <= 3), data.slice_mm > 3,
        data.cohort == "SAROS", data.cohort == "autoPET"]).astype(float)
    y = (data.class26 == "disconnected").to_numpy(float)
    beta = np.zeros(x.shape[1])
    for _ in range(30):
        mu = FS.scipy.special.expit(x @ beta)
        step = np.linalg.solve((x.T*(mu*(1-mu))) @ x, x.T @ (y-mu))
        beta += step
        if np.max(abs(step)) < 1e-12:
            break
    observed = coef[coef.model == "disconnection_spacing_categorical"]
    assert np.allclose(observed.estimate, beta, atol=1e-9)
    residual_scores = x * (y-FS.scipy.special.expit(x @ beta))[:,None]
    group_scores = pd.DataFrame(residual_scores).groupby(data.participant).sum().to_numpy()
    bread = np.linalg.inv((x.T*(mu*(1-mu))) @ x)
    groups, n, k = data.participant.nunique(), len(data), x.shape[1]
    covariance = bread @ (group_scores.T @ group_scores) @ bread
    covariance *= groups/(groups-1) * (n-1)/(n-k)
    half = FS.scipy.stats.t.ppf(.975, groups-1)*np.sqrt(np.diag(covariance))
    assert np.allclose(observed.ci_lo, beta-half, atol=1e-8)
    assert np.allclose(observed.ci_hi, beta+half, atol=1e-8)
    raw = FS.smf.ols("gap_mm ~ slice_mm * C(cohort)", data).fit()
    centered = coef[(coef.model == "pooled_spacing_interaction") & (coef.endpoint == "gap_mm")].set_index("term")
    for cohort in ("SAROS", "autoPET"):
        term = f"C(cohort)[T.{cohort}]"
        assert centered.loc[term, "estimate"] == pytest.approx(
            raw.params[term] + 2.5*raw.params[f"slice_mm:{term}"])


def test_distance_bootstrap_preserves_block_degrees_of_freedom(monkeypatch):
    monkeypatch.setattr(FS, "N_BOOT", 400)
    frame = pd.DataFrame({"cohort": ["X"]*9,
        "participant": ["a"]*2+["b"]*3+["c"]*4,
        "gap_mm": [0.,2.,1.,3.,5.,2.,3.,4.,7.]})
    _, result = FS.repeat_summary(frame)
    sizes = np.array([2,3,4])
    arrays = [g.gap_mm.to_numpy() for _,g in frame.groupby("participant")]
    sse = np.array([((a-a.mean())**2).sum() for a in arrays])
    draws = np.random.default_rng(FS.SEED).integers(0,3,(400,3))
    expected = np.sqrt(sse[draws].sum(axis=1)/(sizes[draws]-1).sum(axis=1))
    lo,hi = np.quantile(expected,[.025,.975])
    row = result.iloc[0]
    assert row.pooled_within_subject_sd_mm == pytest.approx(np.sqrt(sse.sum()/6))
    assert row.within_subject_sd_percentile_lo == pytest.approx(lo)
    assert row.within_subject_sd_percentile_hi == pytest.approx(hi)


def test_native_plane_sampling_identity_phases_and_label_loss_are_explicit(tmp_path):
    import nibabel as nib
    indices = FS.nearest_neighbor_plane_indices(17,1.,1.,0.)
    assert np.array_equal(indices,np.arange(17))
    for target in (2.5,3.,5.):
        for phase in (0.,.25,.5,.75):
            planes = FS.nearest_neighbor_plane_indices(17,1.,target,phase)
            assert len(planes) and planes.min() >= 0 and planes.max() < 17
            centers = np.arange(-2,20)*target + phase*target
            centers = centers[(centers >= -.5) & (centers < 16.5)]
            assert np.array_equal(planes,np.floor(centers+.5).astype(int))
    for args in ((17,0.,3.,0.),(17,1.,np.nan,0.),(17,1.,3.,1.)):
        with pytest.raises(ValueError):
            FS.nearest_neighbor_plane_indices(*args)
    arrays=[]
    row={"cohort":"X","case":"a","mask_case":"a","participant":"person","slice_mm":1.}
    folder=tmp_path/'X'/'a';folder.mkdir(parents=True)
    for name,short,zs in zip(FS.MC.REFERENCE_VEIN_FILES,("ivc","lciv","rciv"),
                             (slice(8,17),slice(0,7),slice(0,0))):
        arr=np.zeros((3,3,17),np.uint8);arr[1,1,zs]=1
        path=folder/f'{name}.nii.gz';nib.save(nib.Nifti1Image(arr,np.eye(4)),path)
        row[f'{name}_sha256']=FS.MC.topology_digest(path)
        row[f'{short}_voxels']=int(arr.sum());arrays.append(arr.astype(bool))
    row.update(FS.MC.classify_topology(arrays))
    result=FS._grid_resampling_case((row,tmp_path,[1.,3.,1000.],[0.]))['records']
    assert result[0]['resampled_class26'] == 'disconnected'
    assert result[1]['resampled_class26'] == 'direct'
    assert not any(r['primary_label_lost'] for r in result[:2])
    assert result[2]['primary_label_lost'] == 1
    assert result[2]['resampled_class26'] == 'empty_primary_label'
    case_table = tmp_path/'cases.csv'
    pd.DataFrame([row]).to_csv(case_table,index=False)
    out = tmp_path/'experiment'
    FS.grid_resampling_experiment(case_table,tmp_path,out,[1.,3.,1000.],[0.],workers=1)
    import json
    provenance = json.loads((out/'grid_resampling_manifest.json').read_text())
    assert provenance['case_n'] == 1 and provenance['case_phase_grid_rows'] == 3
    assert not provenance['inference_grid_recovered']
    for item in provenance['outputs']:
        assert FS.MC.topology_digest(out/item['path']) == item['sha256']
    summary = pd.read_csv(out/'grid_resampling_summary.csv')
    lost = summary[(summary.cohort == 'Pooled') & (summary.target_spacing_mm == 1000.) & (summary.endpoint_rule == 'reidentified') & (summary.connectivity == 26)]
    assert lost.set_index('outcome').loc['disconnected_lower_bound','proportion'] == 0
    assert lost.set_index('outcome').loc['disconnected_upper_bound','proportion'] == 1
    # Hash failure must stop the experiment rather than change the denominator.
    with (folder/f'{FS.MC.REFERENCE_VEIN_FILES[0]}.nii.gz').open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):
        FS._grid_resampling_case((row,tmp_path,[3.],[0.]))


def _write_grid_fixture(tmp_path, case="a", cohort="X", shape=(7, 7, 17), affine=None, nested=False):
    import nibabel as nib
    affine = np.eye(4) if affine is None else affine
    arrays = [np.zeros(shape, np.uint8) for _ in range(3)]
    arrays[0][1, 1, 8:] = 1
    arrays[1][1, 1, :7] = 1
    row = {"cohort": cohort, "case": case, "mask_case": case,
           "participant": f"{cohort}:{case}", "slice_mm": float(np.linalg.norm(affine[:3, 2]))}
    folder = tmp_path/cohort/case
    if nested:
        folder = folder/'segmentations_fullres'
    folder.mkdir(parents=True)
    for arr, name, short in zip(arrays, FS.MC.REFERENCE_VEIN_FILES, ("ivc", "lciv", "rciv")):
        path = folder/f'{name}.nii.gz'
        nib.save(nib.Nifti1Image(arr, affine), path)
        row[f'{name}_sha256'] = FS.MC.topology_digest(path)
        row[f'{short}_voxels'] = int(arr.sum())
    row.update(FS.MC.classify_topology([a.astype(bool) for a in arrays]))
    return row, arrays


def test_grid_endpoint_switch_control_has_independent_reachability_oracle(tmp_path):
    # Native main LCIV is long but sparsely sampled; a smaller island becomes
    # dominant on the coarse grid. Fixed and reidentified endpoint sets diverge.
    import nibabel as nib
    row, arrays = _write_grid_fixture(tmp_path)
    arrays[1][:] = 0
    arrays[1][1, 1, :7] = 1
    arrays[1][4:6, 4:6, 8] = 1
    path = tmp_path/'X'/'a'/f'{FS.MC.REFERENCE_VEIN_FILES[1]}.nii.gz'
    nib.save(nib.Nifti1Image(arrays[1], np.eye(4)), path)
    row[f'{FS.MC.REFERENCE_VEIN_FILES[1]}_sha256'] = FS.MC.topology_digest(path)
    row['lciv_voxels'] = int(arrays[1].sum())
    row.update(FS.MC.classify_topology([a.astype(bool) for a in arrays]))
    records = FS._grid_resampling_case((row, tmp_path, [3., 5.], [0., .25, .5, .75]))['records']
    assert any(r['endpoint_sets_changed'] for r in records)
    assert any(r['fixed_class26'] != r['resampled_class26'] for r in records)
    native = [_independent_dominant(a.astype(bool)) for a in arrays[:2]]
    for r in records:
        idx = FS.nearest_neighbor_plane_indices(17, 1., r['target_spacing_mm'], r['phase_fraction'])
        sampled = [a[:, :, idx].astype(bool) for a in arrays]
        endpoints = [a[:, :, idx] for a in native]
        for conn in (26, 6):
            assert r[f'resampled_class{conn}'] == _independent_class(sampled, conn)
            pair = _independent_label(sampled[0] | sampled[1], conn)
            union = _independent_label(np.logical_or.reduce(sampled), conn)
            def shares(lab):
                ids = [{int(v) for v in lab[e].ravel() if v > 0} for e in endpoints]
                return bool(ids[0] & ids[1])
            fixed = 'direct' if shares(pair) else 'right_iliac_mediated' if shares(union) else 'disconnected'
            assert r[f'fixed_class{conn}'] == fixed


def test_grid_preflight_layout_byte_staging_geometry_and_pilot(tmp_path):
    row, _ = _write_grid_fixture(tmp_path, nested=True)
    larger, _ = _write_grid_fixture(tmp_path, case='b', shape=(9, 9, 17))
    second, _ = _write_grid_fixture(tmp_path, cohort='Y', shape=(9, 9, 17))
    frame = pd.DataFrame([row, larger, second])
    pilot, root = FS._grid_preflight(frame, tmp_path, tmp_path/'stage', pilot_cases=1)
    assert set(zip(pilot.cohort, pilot.case)) == {('X', 'b'), ('Y', 'a')}
    for item in pilot.to_dict('records'):
        paths = FS._grid_mask_paths(item, root)
        assert all(p.parent.name == item['mask_case'] for p in paths)
    # Orthonormal rotated/reversed grids preserve physical distances and are accepted.
    angle = np.deg2rad(25)
    affine = np.eye(4)
    affine[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, -1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    rotated, _ = _write_grid_fixture(tmp_path, case='rotated', affine=affine)
    FS._grid_preflight(pd.DataFrame([rotated]), tmp_path)
    affine[0, 1] = .2
    sheared, _ = _write_grid_fixture(tmp_path, case='sheared', affine=affine)
    with pytest.raises(ValueError, match='nonorthogonal'):
        FS._grid_preflight(pd.DataFrame([sheared]), tmp_path)
    corrupt = tmp_path/'X'/'a'/'segmentations_fullres'/f'{FS.MC.REFERENCE_VEIN_FILES[0]}.nii.gz'
    with corrupt.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        FS._grid_preflight(pd.DataFrame([row]), tmp_path)


def test_grid_paired_intervals_retain_visit_and_phase_blocks():
    # Exhaustively enumerate a two-participant block bootstrap (four equally likely draws).
    # One participant has two visits, each with two phases; the other has one visit.
    frame = pd.DataFrame({'cohort': ['X']*6, 'participant': ['a']*4+['b']*2})
    delta = np.array([-1., -1., 0., 0., 1., 1.])
    stats = FS.proportion_intervals(frame, delta, n_boot=10000, independent_wilson=False)
    assert stats['participants'] == 2 and stats['n'] == 6
    assert stats['proportion'] == 0.
    assert stats['ci_lo'] == -.5 and stats['ci_hi'] == 1.
    assert stats['ci_method'] == 'cohort_stratified_participant_percentile'


def test_grid_resume_validate_and_settings_integrity(tmp_path, monkeypatch):
    import json
    import sqlite3
    row, _ = _write_grid_fixture(tmp_path)
    table = tmp_path/'cases.csv'
    pd.DataFrame([row]).to_csv(table, index=False)
    out = tmp_path/'run'
    FS.grid_resampling_experiment(table, tmp_path, out, [3.], [0., .5], workers=1)
    manifest = json.loads((out/'grid_resampling_manifest.json').read_text())
    assert manifest['publication_eligible_frame']
    baseline = {p.name:p.read_bytes() for p in out.glob('*.csv')}
    def forbidden(task):
        raise AssertionError('resume unexpectedly recomputed a completed case')
    monkeypatch.setattr(FS, '_grid_resampling_case', forbidden)
    FS.grid_resampling_experiment(table, tmp_path, out, [3.], [0., .5], workers=1)
    assert baseline == {p.name:p.read_bytes() for p in out.glob('*.csv')}
    FS.grid_resampling_experiment(table, tmp_path, out, [3.], [0., .5], validate_only=True)
    with pytest.raises(ValueError, match='identity changed'):
        FS.grid_resampling_experiment(table, tmp_path, out, [5.], [0., .5])
    # Refuse a corrupted output, then repair it from the committed case checkpoint.
    with (out/'grid_resampling_cases.csv').open('a') as stream:
        stream.write('corrupt')
    with pytest.raises(ValueError, match='output hash mismatch'):
        FS.grid_resampling_experiment(table, tmp_path, out, [3.], [0., .5], validate_only=True)
    FS.grid_resampling_experiment(table, tmp_path, out, [3.], [0., .5])
    # An incomplete run is resumable, but cannot pass publication validation.
    with sqlite3.connect(out/'grid_resampling_checkpoint.sqlite3') as db:
        db.execute('DELETE FROM cases')
    with pytest.raises(ValueError, match='incomplete experiment'):
        FS.grid_resampling_experiment(table, tmp_path, out, [3.], [0., .5], validate_only=True)


def test_grid_preflight_writes_no_results_and_frame_validation_fails_closed(tmp_path):
    row, _ = _write_grid_fixture(tmp_path)
    table = tmp_path/'cases.csv'
    pd.DataFrame([row]).to_csv(table, index=False)
    out = tmp_path/'preflight'
    FS.grid_resampling_experiment(table, tmp_path, out, [3.], [0.], preflight_only=True)
    assert not out.exists()
    for budget in (0., -.5, np.nan):
        with pytest.raises(ValueError, match='worker memory budget'):
            FS.grid_resampling_experiment(table, tmp_path, out, [3.], [0.], worker_memory_gib=budget)
    for spacings, phases in [([], [0.]), ([3., 3.], [0.]), ([3.], [])]:
        with pytest.raises(ValueError, match='nonempty and unique'):
            FS._grid_frame(table, spacings, phases)
    pd.DataFrame([row, row]).to_csv(table, index=False)
    with pytest.raises(ValueError, match='duplicate case'):
        FS._grid_frame(table, [3.], [0.])


def test_grid_parallel_reproducibility_and_pilot_flag(tmp_path):
    import json
    rows = [_write_grid_fixture(tmp_path, case=case)[0] for case in ('a', 'b')]
    table = tmp_path/'cases.csv'; pd.DataFrame(rows).to_csv(table, index=False)
    for workers in (1, 2):
        FS.grid_resampling_experiment(table, tmp_path, tmp_path/f'run{workers}', [3.], [0., .5], workers=workers,
                                      worker_memory_gib=1. if workers == 2 else None)
    scientific = [p.name for p in (tmp_path/'run1').glob('*.csv') if 'resources' not in p.name]
    assert len(scientific) == 4
    for name in scientific:
        assert (tmp_path/'run1'/name).read_bytes() == (tmp_path/'run2'/name).read_bytes()
    FS.grid_resampling_experiment(table, tmp_path, tmp_path/'pilot', [3.], [0.], pilot_cases=1)
    manifest = json.loads((tmp_path/'pilot'/'grid_resampling_manifest.json').read_text())
    assert not manifest['publication_eligible_frame'] and manifest['case_n'] == 1
    assert manifest['status'] == 'completed_resource_pilot_not_publication'


def _write_multilabel_fixture(tmp_path):
    import json
    import nibabel as nib
    row, arrays = _write_grid_fixture(tmp_path/'canonical')
    labels = (13, 42, 99)  # Deliberately not the TotalSegmentator default indices.
    ml = np.zeros_like(arrays[0])
    for arr, label in zip(arrays, labels):
        ml[arr.astype(bool)] = label
    ml[0, 0, 0] = 117  # Unrelated anatomy must not enter the venous union.
    folder = tmp_path/'multilabel'/'X'/'a';folder.mkdir(parents=True)
    nib.save(nib.Nifti1Image(ml, np.eye(4)), folder/'segmentation.nii.gz')
    indices = dict(zip(FS.MC.REFERENCE_VEIN_FILES, labels))
    for i, name in enumerate(FS.EM.STRUCTURES):
        indices.setdefault(name, 70+i)
    report = {'totalsegmentator_version':'2.17.0', 'task':'total',
              'multilabel':True, 'fast':False, 'classes':{str(i):name for name,i in indices.items()}}
    (folder/'report.json').write_text(json.dumps(report))
    return row, folder


def test_multilabel_export_uses_case_map_and_matches_the_flat_export_bytes(tmp_path):
    import json
    import nibabel as nib
    row, folder = _write_multilabel_fixture(tmp_path)
    frame = FS._grid_multilabel_headers(pd.DataFrame([row]), tmp_path/'multilabel', None)
    indexed = frame.iloc[0].to_dict()
    FS._grid_export_multilabel_case((indexed, tmp_path/'multilabel', tmp_path/'stage'))
    stage = tmp_path/'stage'/'X'/'a'
    assert sorted(p.name for p in stage.iterdir()) == sorted(f'{n}.nii.gz' for n in FS.MC.REFERENCE_VEIN_FILES)
    for name in FS.MC.REFERENCE_VEIN_FILES:
        assert FS.MC.topology_digest(stage/f'{name}.nii.gz') == row[f'{name}_sha256']
    # The all-structure exporter is the default and is byte-compatible.
    flat, error = FS.EM.export_case(('X','a',str(tmp_path/'multilabel'),str(tmp_path/'flat'),True,'',None))
    assert error is None and len(flat) == len(FS.EM.STRUCTURES)
    assert all(FS.MC.topology_digest(tmp_path/'flat'/'X'/'a'/f'{n}.nii.gz') == row[f'{n}_sha256']
               for n in FS.MC.REFERENCE_VEIN_FILES)
    # Exact file hashes, rather than equal voxel counts, remain mandatory.
    bad = indexed.copy();bad[f'{FS.MC.REFERENCE_VEIN_FILES[0]}_sha256'] = '0'*64
    with pytest.raises(ValueError, match='canonical export hash mismatch'):
        FS._grid_export_multilabel_case((bad, tmp_path/'multilabel', tmp_path/'badstage'))
    report = json.loads((folder/'report.json').read_text());report['fast'] = True
    (folder/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='fast=true'):
        FS._grid_multilabel_headers(pd.DataFrame([row]), tmp_path/'multilabel', None)
    with pytest.raises(ValueError, match='source changed'):
        FS._grid_export_multilabel_case((indexed, tmp_path/'multilabel', tmp_path/'stage'))
    # Source multilabels still retain unrelated anatomy and have not been rewritten.
    assert np.asanyarray(nib.load(folder/'segmentation.nii.gz').dataobj)[0,0,0] == 117


def test_multilabel_cli_loads_the_phase0_exporter_despite_a_shadowing_module(tmp_path):
    import os
    (tmp_path/'export_relevant_masks.py').write_text("raise RuntimeError('shadowing exporter loaded')\n")
    result = subprocess.run([sys.executable, str(Path(FS.__file__).resolve()), '--help'],
        cwd=tmp_path, env={**os.environ, 'PYTHONPATH':str(tmp_path)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '--multilabel-root' in result.stdout
    assert Path(FS.EM.__file__).resolve() == Path(FS.__file__).resolve().parent.parent/'phase0_data/export_relevant_masks.py'


def test_grid_cli_executes_without_statsmodels_and_regression_requires_it(tmp_path):
    import os
    import json
    row, _ = _write_multilabel_fixture(tmp_path)
    table = tmp_path/'cases.csv';pd.DataFrame([row]).to_csv(table,index=False)
    blocked = tmp_path/'blocked-imports';blocked.mkdir()
    (blocked/'statsmodels.py').write_text(
        "raise ModuleNotFoundError(\"No module named 'statsmodels'\", name='statsmodels')\n")
    env = {**os.environ, 'PYTHONPATH':str(blocked)}
    script = str(Path(FS.__file__).resolve())
    help_result = subprocess.run([sys.executable,script,'--help'],cwd=tmp_path,
                                 env=env,capture_output=True,text=True)
    assert help_result.returncode == 0, help_result.stderr
    out = tmp_path/'grid-results'
    args = [sys.executable,script,'--grid-resampling-only','--case-table',str(table),
            '--multilabel-root',str(tmp_path/'multilabel'),'--stage-mask-root',str(tmp_path/'stage'),
            '--outdir',str(out),'--workers','1','--target-spacings','3','--phase-fractions','0','.5']
    result = subprocess.run(args,cwd=tmp_path,env=env,capture_output=True,text=True)
    assert result.returncode == 0,result.stderr
    manifest=json.loads((out/'grid_resampling_manifest.json').read_text())
    assert manifest['case_phase_grid_rows'] == 2 and manifest['identity_controls_passed'] == 1
    validated = subprocess.run(args+['--grid-validate-only'],cwd=tmp_path,env=env,
                               capture_output=True,text=True)
    assert validated.returncode == 0,validated.stderr
    assert 'validation MATCH' in validated.stdout
    primary = tmp_path/'primary-results'
    regression = subprocess.run([sys.executable,script,'--outdir',str(primary)],cwd=tmp_path,
                                env=env,capture_output=True,text=True)
    assert regression.returncode != 0
    assert 'required to regenerate the regression analyses' in regression.stderr
    assert not primary.exists()


def test_multilabel_grid_end_to_end_is_identical_to_separate_sources(tmp_path):
    import json
    row, folder = _write_multilabel_fixture(tmp_path)
    original = {p.name:p.read_bytes() for p in folder.iterdir()}
    table = tmp_path/'cases.csv';pd.DataFrame([row]).to_csv(table,index=False)
    FS.grid_resampling_experiment(table,tmp_path/'canonical',tmp_path/'binary-run',[3.],[0.,.5],workers=1)
    FS.grid_resampling_experiment(table,None,tmp_path/'ml-run',[3.],[0.,.5],workers=1,
        multilabel_root=tmp_path/'multilabel',stage_root=tmp_path/'stage')
    for p in (tmp_path/'binary-run').glob('*.csv'):
        if 'resources' not in p.name:
            assert p.read_bytes() == (tmp_path/'ml-run'/p.name).read_bytes()
    manifest=json.loads((tmp_path/'ml-run'/'grid_resampling_manifest.json').read_text())
    assert manifest['source_format'] == 'multilabel_with_canonical_binary_verification'
    assert manifest['source_case_keys'][0]['multilabel_sha256'] == FS.MC.topology_digest(folder/'segmentation.nii.gz')
    assert manifest['source_case_keys'][0]['report_sha256'] == FS.MC.topology_digest(folder/'report.json')
    FS.grid_resampling_experiment(table,None,tmp_path/'ml-run',[3.],[0.,.5],workers=1,
        multilabel_root=tmp_path/'multilabel',stage_root=tmp_path/'stage',validate_only=True)
    assert original == {p.name:p.read_bytes() for p in folder.iterdir()}
    # Preflight may write temporary exports, but never a scientific result directory.
    FS.grid_resampling_experiment(table,None,tmp_path/'preflight-out',[3.],[0.],workers=1,
        multilabel_root=tmp_path/'multilabel',stage_root=tmp_path/'preflight-stage',preflight_only=True)
    assert not (tmp_path/'preflight-out').exists()
    with pytest.raises(ValueError,match='distinct temporary'):
        FS.grid_resampling_experiment(table,None,tmp_path/'missingstage',[3.],[0.],
            multilabel_root=tmp_path/'multilabel')
    with pytest.raises(ValueError,match='outside the original'):
        FS.grid_resampling_experiment(table,None,tmp_path/'unsafestage',[3.],[0.],
            multilabel_root=tmp_path/'multilabel',stage_root=tmp_path/'multilabel'/'exports')


# Independent component engine (SimpleITK), an oracle for the canonical publication cases.
def _independent_label(mask, conn):
    image = sitk.GetImageFromArray(mask.astype(np.uint8))
    return sitk.GetArrayFromImage(sitk.ConnectedComponent(image, conn == 26))


def _independent_dominant(mask):
    lab = _independent_label(mask, 26)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return lab == counts.argmax()


def _independent_class(masks, connectivity):
    a, b = _independent_dominant(masks[0]), _independent_dominant(masks[1])
    union_lab = _independent_label(np.logical_or.reduce(masks), connectivity)
    pair_lab = _independent_label(masks[0] | masks[1], connectivity)

    def connected(lab):
        return bool(set(lab[a].tolist()) & set(lab[b].tolist()) - {0})

    return MC.TOPOLOGY_CLASSES[0 if connected(pair_lab) else 1 if connected(union_lab) else 2]


def test_model_grid_preflight_refuses_without_a_study_root(tmp_path, capsys):
    """The study root has no default (a site-specific absolute path is not portable); the CLI refuses a preflight without it."""
    argv = ["--model-grid-mode", "preflight", "--model-grid-root", str(tmp_path / "grid"),
            "--multilabel-root", str(tmp_path / "ml"), "--model-grid-stage-root", str(tmp_path / "stage")]
    with pytest.raises(SystemExit) as stop:
        FS.main(argv)
    assert stop.value.code == 2
    assert "model-grid preflight requires --multilabel-root, --model-grid-stage-root and --model-grid-study-root" in capsys.readouterr().err
    assert not (tmp_path / "grid").exists(), "the refusal comes before any experiment directory is made"
