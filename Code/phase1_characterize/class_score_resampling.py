#!/usr/bin/env python3
"""Class-score resampling experiment for the iliocaval connectivity audit.

The question: TotalSegmentator returns labels to the native grid by copying the nearest 1.5-mm label
plane. What happens to iliocaval connectivity if the class scores are resampled along every axis first
and the label is decided afterwards?

Each examination in MODEL_GRID_ROOT/cases.json is inferred again with the pinned 2.17.0 pipeline, on a
GPU, with one addition: the softmax of the task-293 (cardiac) submodel, which predicts every vessel
label, is saved. Four label maps are then classified with the study's own rule (classify_topology):

  inference  the captured 1.5-mm prediction
  nearest    the exported native-grid labels (nearest plane), the comparator from the SAME run
  linear     log-softmax interpolated linearly along every axis to the native grid, then the argmax
  slab       as linear, but along each axis coarser than 1.5 mm the score is averaged over the native
             slice thickness first, so that no 1.5-mm plane is skipped

The log of the softmax differs from the logits by a per-voxel constant shared by all classes, so the
argmax of interpolated log-softmax is the argmax of interpolated logits. Voxels that a later submodel
(muscles, ribs) labels in the merged map are not venous on the 1.5-mm grid, as in the merged output.

Controls, per examination, before anything is reported:
  - the CT and the archived native labels match their recorded SHA-256;
  - scipy.ndimage.zoom(order=0, mode="nearest") of the captured labels reproduces the exported native
    labels exactly, which fixes the grid mapping every arm uses;
  - the argmax of the saved scores equals the captured labels on the 1.5-mm grid, up to ties
    (TIE_TOLERANCE_VOXELS), which fixes the axis order of the score array.

    python class_score_resampling.py --check     ... (login node: inputs, versions, weights)
    python class_score_resampling.py --all --workers W ... (every unfinished examination, W at a time, one GPU)
    python class_score_resampling.py --index N   ... (one examination, GPU)
    python class_score_resampling.py --summarize ... (login node: tables from the case records)
"""
from __future__ import annotations

import argparse
import collections
import importlib.machinery
import importlib.metadata
import importlib.util
import fcntl
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

for _folder in sorted(Path(__file__).resolve().parents[1].iterdir()):
    if _folder.is_dir() and _folder.name[0] not in "._" and str(_folder) not in sys.path:
        sys.path.append(str(_folder))

import model_grid_audit as MG
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

SCORES = "cardiac_scores.npz"
EPS = 1e-6                     # floor of the softmax before the log; both sides of a sharp edge saturate alike
SLAB_SAMPLES = 15              # positions averaged across one native slice
SCORE_THRESHOLD = 0.01         # a voxel enters the working box if the three veins sum to at least this
MARGIN_PLANES = 4              # planes added around the working box
TIE_TOLERANCE_VOXELS = 10      # argmax ties between the score array and the label export
LATER_TASKS = (294, 295)       # muscles and ribs, written over the cardiac labels in the merged map
ARMS = ("inference", "nearest", "linear", "slab")
MASKS = {"linear": "linear_venous.nii.gz", "slab": "slab_venous.nii.gz"}
SPACING_TOLERANCE_MM = 0.0005


# ------------------------------------------------------------------------------ resampling
def zoom_coordinates(n_in, n_out):
    """Input-plane position of each output plane under scipy.ndimage.zoom with grid_mode=False."""
    if n_out < 2 or n_in < 2:
        raise ValueError("a grid axis holds fewer than two planes")
    return np.arange(n_out) * ((n_in - 1) / (n_out - 1))


def along_axis(values, coordinates, axis):
    """Linear interpolation of values along one axis at fractional plane positions."""
    last = values.shape[axis] - 1
    position = np.clip(coordinates, 0, last)
    low = np.floor(position).astype(int)
    high = np.minimum(low + 1, last)
    shape = [1] * values.ndim
    shape[axis] = -1
    weight = (position - low).reshape(shape).astype(values.dtype)
    return (1 - weight) * np.take(values, low, axis=axis) + weight * np.take(values, high, axis=axis)


def slab_mean(values, coordinates, step, axis):
    """Mean of the linearly interpolated values over one native slice (step input planes thick)."""
    offsets = ((np.arange(SLAB_SAMPLES) + 0.5) / SLAB_SAMPLES - 0.5) * step
    total = 0
    for offset in offsets:
        total = total + along_axis(values, coordinates + offset, axis)
    return total / SLAB_SAMPLES


def resample_argmax(scores, coordinates, steps, slab):
    """Argmax after resampling channel-first scores along the three spatial axes."""
    result = scores
    for axis in range(3):
        if slab and steps[axis] > 1 + 1e-9:
            result = slab_mean(result, coordinates[axis], steps[axis], axis + 1)
        else:
            result = along_axis(result, coordinates[axis], axis + 1)
    return np.argmax(result, axis=0).astype(np.uint8)


def working_box(mask, margins):
    """Inclusive plane range of the mask along each axis, widened by the margin of that axis."""
    box = []
    for axis in range(3):
        other = tuple(j for j in range(3) if j != axis)
        hits = np.flatnonzero(mask.any(axis=other))
        if hits.size == 0:
            return None
        box.append((max(0, int(hits[0]) - margins[axis]),
                    min(mask.shape[axis] - 1, int(hits[-1]) + margins[axis])))
    return box


def bind_axes(predicted, merged, cardiac, later_ids):
    """The axis order under which the scores' argmax equals the captured 1.5-mm labels."""
    lut = np.zeros(len(cardiac) + 1, np.int32)
    for local, item in cardiac.items():
        lut[int(local)] = int(item["total_id"])
    is_later, is_cardiac = np.zeros(256, bool), np.zeros(256, bool)
    is_later[list(later_ids)] = True
    is_cardiac[lut[1:]] = True
    found = []
    for order in itertools.permutations(range(3)):
        if tuple(predicted.shape[i] for i in order) != merged.shape:
            continue
        expected = lut[np.transpose(predicted, order)]
        foreground = expected != 0
        differs = foreground & (merged != expected)
        overwritten = differs & is_later[merged]
        mismatch = int((differs & ~overwritten).sum() + (~foreground & is_cardiac[merged]).sum())
        found.append((mismatch, order, int(overwritten.sum())))
    if not found:
        raise ValueError(f"score grid {predicted.shape} cannot match the captured grid {merged.shape}; "
                         "TotalSegmentator may have split a very large image")
    mismatch, order, overwritten = min(found)
    if mismatch > TIE_TOLERANCE_VOXELS:
        raise ValueError(f"the saved scores do not reproduce the captured labels ({mismatch} voxels differ)")
    return order, mismatch, overwritten


def later_submodel_ids(mapping):
    """Total-map ids of the submodels written after the cardiac one."""
    from totalsegmentator import map_to_binary as parts
    table = getattr(parts, "map_taskid_to_partname_ct", None) or parts.map_taskid_to_partname
    names = set()
    for task in LATER_TASKS:
        names |= set(parts.class_map_5_parts[table[task]].values())
    return sorted(int(k) for k, v in mapping.items() if v in names)


def analyze(stage, folder, later_ids=None):
    """Controls, the two class-score arms and the classes of all four label maps for one examination."""
    stage, folder = Path(stage), Path(folder)
    sidecar = json.loads((stage / MG.INTERNAL_LABELS).read_text())
    mapping = {int(k): v for k, v in sidecar["total_class_map"].items()}
    cardiac = sidecar["cardiac_submodel"]["classes"]
    venous_total = MG.venous_ids(mapping)
    later = later_submodel_ids(mapping) if later_ids is None else list(later_ids)
    local_of_total = {int(v["total_id"]): int(k) for k, v in cardiac.items()}
    channels = [local_of_total[t] for t in venous_total]

    internal = np.asanyarray(nib.load(stage / MG.INTERNAL_MULTILABEL).dataobj).astype(np.uint8)
    native_image = nib.as_closest_canonical(nib.load(stage / MG.CASE_OUTPUTS[0]))
    native = np.asanyarray(native_image.dataobj).astype(np.uint8)
    factors = np.array(native.shape) / np.array(internal.shape)
    control = ndi.zoom(internal, factors, mode="nearest", order=0)
    if control.shape != native.shape or not np.array_equal(control, native):
        raise ValueError("nearest-plane zoom of the captured labels does not reproduce the exported native labels")
    del control

    with np.load(stage / SCORES) as saved:
        probabilities = saved["probabilities"]
    if probabilities.ndim != 4 or probabilities.shape[0] != len(cardiac) + 1:
        raise ValueError(f"unexpected score array {probabilities.shape}")
    dtype = str(probabilities.dtype)
    order, mismatch, overwritten = bind_axes(np.argmax(probabilities, axis=0), internal, cardiac, later)
    probabilities = np.transpose(probabilities, (0, *[i + 1 for i in order]))

    n_in, n_out = internal.shape, native.shape
    steps = [(n_in[a] - 1) / (n_out[a] - 1) for a in range(3)]
    is_venous, is_later = np.zeros(256, bool), np.zeros(256, bool)
    is_venous[venous_total] = True
    is_later[later] = True
    mask = (probabilities[channels].sum(axis=0) >= SCORE_THRESHOLD) | is_venous[internal]
    box = working_box(mask, [MARGIN_PLANES + math.ceil(s) for s in steps])
    del mask
    maps = {}
    if box is None:
        maps = {arm: np.zeros(native.shape, np.uint8) for arm in MASKS}
    else:
        cut = tuple(slice(lo, hi + 1) for lo, hi in box)
        scores = np.log(np.maximum(probabilities[(slice(None), *cut)].astype(np.float32), EPS))
        del probabilities
        floor = np.float32(np.log(EPS))
        later_here = is_later[internal[cut]]
        for channel in channels:
            scores[channel][later_here] = floor
        code = np.zeros(scores.shape[0], np.uint8)
        for value, channel in enumerate(channels, 1):
            code[channel] = value
        out_cut, coordinates = [], []
        for axis, (lo, hi) in enumerate(box):
            first = max(0, math.ceil(lo / steps[axis]))
            last = min(n_out[axis] - 1, math.floor(hi / steps[axis]))
            out_cut.append(slice(first, last + 1))
            coordinates.append(zoom_coordinates(n_in[axis], n_out[axis])[first:last + 1] - lo)
        for arm in MASKS:
            volume = np.zeros(native.shape, np.uint8)
            volume[tuple(out_cut)] = code[resample_argmax(scores, coordinates, steps, slab=arm == "slab")]
            maps[arm] = volume
    maps["nearest"] = MG.compact_venous(native, venous_total)
    classes = {arm: MG.topology(maps[arm], (1, 2, 3)) for arm in ("nearest", *MASKS)}
    classes["inference"] = MG.topology(MG.compact_venous(internal, venous_total), (1, 2, 3))
    folder.mkdir(parents=True, exist_ok=True)
    for arm, name in MASKS.items():
        partial = folder / f"{arm}.partial.nii.gz"
        nib.save(nib.Nifti1Image(maps[arm], native_image.affine), partial)
        partial.replace(folder / name)
    return {"classes": classes, "axis_order": list(order), "tie_mismatch_voxels": mismatch,
            "venous_voxels_overwritten_by_later_submodels": overwritten, "score_dtype": dtype,
            "native_shape": list(n_out), "internal_shape": list(n_in),
            "native_spacing_mm": [float(x) for x in np.linalg.norm(native_image.affine[:3, :3], axis=0)]}


# ----------------------------------------------------------------------------- GPU worker
def capture_scores(module, target):
    """Ask the instrumented module to save the softmax of the cardiac submodel, and only of it."""
    original = module.nnUNetv2_predict
    seen = []

    def predict(dir_in, dir_out, task_id, *args, **kwargs):
        if task_id == MG.CARDIAC_TASK:
            if kwargs.get("save_probabilities_path") is not None:
                raise ValueError("the pipeline already requests probabilities")
            kwargs["save_probabilities_path"] = Path(target)
            seen.append(task_id)
        return original(dir_in, dir_out, task_id, *args, **kwargs)

    module.nnUNetv2_predict = predict
    return seen


def score_worker(ct, native, internal, scores):
    """model_grid_audit.inference_worker with the cardiac softmax saved; the inference call is the same."""
    import torch
    if importlib.metadata.version("TotalSegmentator") != MG.VERSION or not torch.cuda.is_available():
        raise ValueError("TotalSegmentator 2.17.0 and an allocated CUDA GPU are required; no CPU fallback")
    print("IVCS_GPU_CONTEXT", torch.cuda.get_device_name(0), flush=True)
    name = "totalsegmentator.nnunet"
    if name in sys.modules:
        raise ValueError("nnunet was imported before capture instrumentation")
    source = Path(importlib.util.find_spec(name).origin)
    instrumented = MG.patched_source(source.read_text())

    class CaptureLoader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            return compile(instrumented, str(source), "exec")

    spec = importlib.util.spec_from_file_location(name, source, loader=CaptureLoader(name, str(source)))
    module = importlib.util.module_from_spec(spec)
    module.__dict__["_ivcs_capture_model_grid"] = MG.capture_internal
    sys.modules[name] = module
    spec.loader.exec_module(module)
    setattr(sys.modules["totalsegmentator"], "nnunet", module)
    seen = capture_scores(module, scores)
    os.environ["TS_AUDIT_CAPTURE"] = str(internal)
    from totalsegmentator.config import setup_totalseg, set_config_key
    setup_totalseg()
    set_config_key("send_usage_stats", False)
    set_config_key("statistics_disclaimer_shown", True)
    from totalsegmentator.python_api import totalsegmentator
    totalsegmentator(str(ct), str(native), ml=True, task="total", device="gpu",
                    fast=False, fastest=False, roi_subset=None, body_seg=False,
                    higher_order_resampling=False, higher_order_resampling_LEGACY=False,
                    resampling_order=1, nr_thr_resamp=1, nr_thr_saving=2,
                    remove_small_blobs=False, save_lowres=False)
    if seen != [MG.CARDIAC_TASK] or not Path(scores).is_file():
        raise ValueError("the cardiac submodel's scores were not saved")
    if MG.sha(source) != MG.PRISTINE_SHA256:
        raise ValueError("installed source changed during inference")


# --------------------------------------------------------------------------- one examination
def translate(path, recorded_root, study_root):
    """A path recorded under the original study root, read under this cluster's study root."""
    text, root = str(path), str(recorded_root).rstrip("/")
    if not text.startswith(root + "/"):
        raise ValueError(f"not under the recorded study root {root}: {text}")
    return Path(study_root) / text[len(root) + 1:]


def verified_sources(row, recorded_root, study_root):
    paths = {key: translate(row[key], recorded_root, study_root) for key in ("ct_path", "frozen_native")}
    for key, path in paths.items():
        if MG.sha(path) != row["source_hashes"][key]:
            raise ValueError(f"{key} does not match its recorded SHA-256: {path.name}")
    return paths


def run_case(args, rows):
    if not 0 <= args.index < len(rows):
        raise ValueError("zero-based case index out of range")
    row = rows[args.index]
    folder = MG.case_folder(args.out, row)
    folder.mkdir(parents=True, exist_ok=True)
    record = folder / "qc.json"
    with (folder / ".run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if record.is_file() and json.loads(record.read_text()).get("status") == "pass":
            print("Class scores already complete", row["cohort"], row["case"], flush=True)
            return 0
        qc = {"status": "started", **{k: row[k] for k in ("cohort", "case", "mask_case", "participant")},
              "archive_class26": row["expected_class26"], "archive_class6": row["expected_class6"],
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID")}
        MG.atomic_json(record, qc)
        started = time.time()
        try:
            sources = verified_sources(row, args.recorded_root, args.study_root)
            with tempfile.TemporaryDirectory(prefix="ivcs-class-scores-", dir=args.stage_root) as stage:
                stage = Path(stage)
                ct = stage / ("CT.nii.gz" if str(sources["ct_path"]).endswith(".nii.gz") else "CT.nii")
                shutil.copyfile(sources["ct_path"], ct)
                native, internal = stage / MG.CASE_OUTPUTS[0], stage / MG.INTERNAL_VENOUS
                # Its own TotalSegmentator home: workers running side by side must not rewrite one config file.
                home = stage / "totalseg_home"
                home.mkdir()
                environment = dict(os.environ, TMPDIR=str(stage), TOTALSEG_WEIGHTS_PATH=str(args.weights_root),
                                   TOTALSEG_HOME_DIR=str(home), TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(ct), str(native),
                           str(internal), str(stage / SCORES)]
                log = folder / "inference.log"
                with log.open("w") as stream:
                    done = subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                if done.returncode != 0:
                    raise ValueError("inference failed; see inference.log in the case folder")
                qc["inference_seconds"] = round(time.time() - started, 1)
                qc.update(analyze(stage, folder))
            qc.update(status="pass", total_seconds=round(time.time() - started, 1))
            MG.atomic_json(record, qc)
            print("Class scores complete", row["cohort"], row["case"], qc["total_seconds"], "s", flush=True)
            return 0
        except Exception:
            qc.update(status="failed", error=traceback.format_exc(limit=3))
            MG.atomic_json(record, qc)
            print("Class scores FAILED", row["cohort"], row["case"], qc["error"], flush=True)
            return 1


# ----------------------------------------------------------------- many examinations, one GPU
def _unfinished(args, rows):
    """Indices of the examinations without a passing record, in cases.json order."""
    indices = []
    for index, row in enumerate(rows):
        path = MG.case_folder(args.out, row) / "qc.json"
        try:
            finished = json.loads(path.read_text()).get("status") == "pass"
        except (OSError, ValueError):
            finished = False
        if not finished:
            indices.append(index)
    return indices


def _launch_case(args, index):
    """One examination in its own process; they all see the job's one GPU."""
    command = [sys.executable, str(Path(__file__).resolve()), "--index", str(index),
               "--model-grid-root", str(args.model_grid_root), "--recorded-root", str(args.recorded_root),
               "--study-root", str(args.study_root), "--out", str(args.out),
               "--stage-root", str(args.stage_root), "--weights-root", str(args.weights_root)]
    return subprocess.run(command).returncode


def run_all(args, rows, launch=_launch_case):
    """Run unfinished examinations, args.workers at a time, stopping new launches near the time limit."""
    waiting = collections.deque(_unfinished(args, rows))
    finished_before = len(rows) - len(waiting)
    if args.max_cases is not None:
        waiting = collections.deque(list(waiting)[:args.max_cases])
    start, lock, results = time.monotonic(), threading.Lock(), {}
    total = len(waiting)
    print(f"{finished_before} examinations already complete; {total} to run, {args.workers} at a time", flush=True)

    def work():
        while True:
            with lock:
                expired = (args.stop_launching_after is not None
                           and time.monotonic() - start >= args.stop_launching_after)
                if expired or not waiting:
                    return
                index = waiting.popleft()
            code = launch(args, index)
            with lock:
                results[index] = code
                print(f"[{len(results)}/{total}] index {index} exit {code}; "
                      f"{round((time.monotonic() - start) / 60, 1)} min elapsed", flush=True)

    threads = [threading.Thread(target=work) for _ in range(args.workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failed = sorted(index for index, code in results.items() if code != 0)
    print(f"{len(results) - len(failed)} passed, {len(failed)} failed, {len(waiting)} not started."
          + (" Submit the job again to finish them." if failed or waiting else ""), flush=True)
    return 1 if failed else 0


# ------------------------------------------------------------------- checks and final tables
def check(args, rows):
    """Everything that can be verified on a login node, without a GPU."""
    problems = []
    if len(rows) != 1899:
        problems.append(f"cases.json holds {len(rows)} examinations, not 1,899")
    version = importlib.metadata.version("TotalSegmentator")
    if version != MG.VERSION:
        problems.append(f"TotalSegmentator is {version}, not {MG.VERSION}")
    else:
        source = Path(importlib.util.find_spec("totalsegmentator.nnunet").origin)
        if MG.sha(source) != MG.PRISTINE_SHA256:
            problems.append("installed nnunet.py differs from the reviewed 2.17.0 source")
    for task in (291, 292, MG.CARDIAC_TASK, *LATER_TASKS):
        if args.weights_root is None or not any(Path(args.weights_root).glob(f"Dataset{task}*")):
            problems.append(f"the task-{task} weights were not found under --weights-root")
    args.out.mkdir(parents=True, exist_ok=True)
    probe = args.out / ".writable"
    probe.touch()
    probe.unlink()
    checked = 0
    for row in rows[:args.limit]:
        try:
            verified_sources(row, args.recorded_root, args.study_root)
            checked += 1
        except Exception as error:
            problems.append(f"{row['cohort']}/{row['case']}: {error}")
    print(json.dumps({"examinations": len(rows), "inputs_verified": checked, "problems": problems}, indent=2))
    return 1 if problems else 0


def _publish(path, text):
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text)
    partial.replace(path)


def _scrub(text):
    """Error text without cluster paths, for a record that may leave the cluster."""
    return re.sub(r"(/[^\s/\"']+){2,}", "<path>", str(text))


def _stratum(spacing):
    if spacing <= 1.5 + SPACING_TOLERANCE_MM:
        return "spacing_le_1p5mm"
    return "spacing_gt1p5_to3mm" if spacing <= 3.0 + SPACING_TOLERANCE_MM else "spacing_gt3mm"


def summarize(args, rows):
    cases, incomplete = [], []
    for row in rows:
        path = MG.case_folder(args.out, row) / "qc.json"
        qc = json.loads(path.read_text()) if path.is_file() else {"status": "missing"}
        if qc.get("status") != "pass":
            incomplete.append({"cohort": row["cohort"], "case": row["case"], "status": qc.get("status"),
                               "error": _scrub(qc.get("error", ""))})
            continue
        entry = {key: qc[key] for key in ("cohort", "case", "mask_case", "participant", "archive_class26", "archive_class6")}
        entry["native_spacing_z_mm"] = qc["native_spacing_mm"][2]
        entry["stratum"] = _stratum(entry["native_spacing_z_mm"])
        for arm in ARMS:
            entry[f"{arm}_class26"] = qc["classes"][arm]["class26"]
            entry[f"{arm}_class6"] = qc["classes"][arm]["class6"]
            entry[f"{arm}_venous_voxels"] = sum(qc["classes"][arm]["voxels"])
        entry["tie_mismatch_voxels"] = qc["tie_mismatch_voxels"]
        entry["venous_voxels_overwritten_by_later_submodels"] = qc["venous_voxels_overwritten_by_later_submodels"]
        entry["total_seconds"] = qc["total_seconds"]
        cases.append(entry)
    frame = pd.DataFrame(cases)
    summary = []
    groups = [("all", frame)] + ([(s, frame[frame.stratum == s]) for s in sorted(frame.stratum.unique())] if len(frame) else [])
    for name, group in groups:
        for rule in ("26", "6"):
            measurable = group[group[f"nearest_class{rule}"] != "nonmeasurable"]
            base = measurable[f"nearest_class{rule}"] == "disconnected"
            row_out = {"group": name, "adjacency": rule, "n_measurable": len(measurable)}
            for arm in ARMS:
                gone = measurable[f"{arm}_class{rule}"] == "disconnected"
                row_out[f"{arm}_disconnected"] = int(gone.sum())
                row_out[f"{arm}_pct"] = round(100 * gone.mean(), 1) if len(measurable) else float("nan")
                if arm in MASKS:
                    row_out[f"{arm}_closed_vs_nearest"] = int((base & ~gone).sum())
                    row_out[f"{arm}_opened_vs_nearest"] = int((~base & gone).sum())
                    row_out[f"{arm}_nonmeasurable"] = int((measurable[f"{arm}_class{rule}"] == "nonmeasurable").sum())
            summary.append(row_out)
    _publish(args.out / "class_score_cases.csv", frame.to_csv(index=False))
    _publish(args.out / "class_score_summary.csv", pd.DataFrame(summary).to_csv(index=False))
    record = {"producer_sha256": MG.sha(Path(__file__)), "cases_json_sha256": MG.sha(args.model_grid_root / "cases.json"),
              "complete": len(cases), "expected": len(rows), "incomplete": incomplete,
              "epsilon": EPS, "slab_samples": SLAB_SAMPLES, "versions": _versions()}
    _publish(args.out / "run_record.json", json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"{len(cases)} of {len(rows)} examinations complete; {len(incomplete)} incomplete; tables in {args.out.name}")
    return 0


def _versions():
    found = {}
    for package in ("TotalSegmentator", "nnunetv2", "torch", "numpy", "scipy", "nibabel"):
        try:
            found[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            found[package] = None
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--index", type=int)
    mode.add_argument("--all", action="store_true", help="every unfinished examination, --workers at a time")
    mode.add_argument("--summarize", action="store_true")
    parser.add_argument("--model-grid-root", type=Path, required=True)
    parser.add_argument("--recorded-root", required=True)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path)
    parser.add_argument("--weights-root", type=Path)
    parser.add_argument("--limit", type=int, default=3, help="--check: examinations whose inputs are hashed")
    parser.add_argument("--workers", type=int, default=4, help="--all: examinations run side by side")
    parser.add_argument("--max-cases", type=int, help="--all: at most this many unfinished examinations (a pilot)")
    parser.add_argument("--stop-launching-after", type=float,
                        help="--all: seconds after which no new examination is started")
    args = parser.parse_args(argv)
    rows = json.loads((args.model_grid_root / "cases.json").read_bytes())
    if args.check:
        return check(args, rows)
    if args.summarize:
        return summarize(args, rows)
    if args.stage_root is None or args.weights_root is None:
        parser.error("--index and --all need --stage-root and --weights-root")
    if args.all:
        if args.workers < 1:
            parser.error("--workers must be at least 1")
        return run_all(args, rows)
    return run_case(args, rows)


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "--worker":
        score_worker(*map(Path, sys.argv[2:]))
    else:
        sys.exit(main())
