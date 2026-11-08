"""Full processed-frame internal/native-grid experiment, dispatched by connectivity_statistics.

The installed environment is read-only: the pinned nnunet module is instrumented
once in a fresh inference subprocess, without editing site-packages or weights.
"""
from __future__ import annotations

import difflib
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

import stats as ST
import measure_collapsed_gap as MC

VERSION = "2.17.0"
UPSTREAM_COMMIT = "a5d9b3d599c0b662aace59670a9f751d675eff4e"
PRISTINE_SHA256 = "f1b40e66bbc43661dfac4f76f769dc82e7acbab957c9761694d8d607ed752cf5"
ANCHOR = "        if resample is not None and not save_lowres:\n"
HOOK = ("        # IVCS model-grid capture: passive, before native projection.\n"
        "        _ivcs_capture_model_grid(img_pred, dict(locals(), class_map=class_map))\n")
EXPECTED = {"AMOS": 493, "SAROS": 392, "autoPET": 1014}
EXPECTED_NATIVE = {"direct": 384, "right_iliac_mediated": 597,
                   "disconnected": 892, "nonmeasurable": 26}
NAMES = MC.REFERENCE_VEIN_FILES
STATES = (*MC.TOPOLOGY_CLASSES, "nonmeasurable")
INTERNAL_VENOUS = "internal_venous.nii.gz"
INTERNAL_MULTILABEL = "internal_multilabel.nii.gz"
INTERNAL_LABELS = "internal_labels.json"
INTERNAL_OUTPUTS = (INTERNAL_VENOUS, INTERNAL_MULTILABEL, INTERNAL_LABELS)
CASE_OUTPUTS = ("native_multilabel.nii.gz", *INTERNAL_OUTPUTS, "inference.log")
CARDIAC_TASK = 293       # the TotalSegmentator sub-model that predicts every vessel label used here
CARDIAC_CLASSES = 18     # its class count in the pinned 2.17.0 map (heart ... iliac_vena_right)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def component(value):
    if not isinstance(value, str) or value in ("", ".", "..") or any(x in value for x in ("/", "\\", "\0")):
        raise ValueError("unsafe cohort/case identifier")
    return value


def case_folder(root, row):
    root = Path(root).resolve()
    dest = root / component(row["cohort"]) / component(row["case"])
    if not dest.resolve().is_relative_to(root):
        raise ValueError("case output escapes experiment root")
    return dest


def geometry(img):
    if len(img.shape) != 3 or not np.isfinite(img.affine).all():
        raise ValueError("non-3D or nonfinite NIfTI geometry")
    axes = img.affine[:3, :3]
    spacing = np.linalg.norm(axes, axis=0)
    if (spacing <= 0).any() or not np.allclose((axes / spacing).T @ (axes / spacing), np.eye(3), atol=1e-5):
        raise ValueError("singular or sheared NIfTI geometry")
    if not np.allclose(spacing, img.header.get_zooms()[:3], atol=1e-4, rtol=0):
        raise ValueError("header and affine spacing disagree")
    return {"shape": list(img.shape), "affine": img.affine.tolist(),
            "spacing_mm": spacing.tolist(), "axis2_si_alignment": float(abs(axes[2, 2] / spacing[2]))}


def same_geometry(a, b):
    geometry(a)
    geometry(b)
    return a.shape == b.shape and np.allclose(a.affine, b.affine, atol=1e-3, rtol=0)


def topology(data, labels):
    masks = [data == index for index in labels]
    counts = [int(x.sum()) for x in masks]
    if not counts[0] or not counts[1]:
        return {"class26": "nonmeasurable", "class6": "nonmeasurable", "voxels": counts}
    union = masks[0] | masks[1] | masks[2]
    bounds = [np.flatnonzero(union.any(axis=tuple(j for j in range(3) if j != i))) for i in range(3)]
    box = tuple(slice(int(x[0]), int(x[-1]) + 1) for x in bounds)
    measured = MC.classify_topology([x[box] for x in masks])
    return {"class26": measured["class26"], "class6": measured["class6"], "voxels": counts}


def integer_volume(img, compact=False):
    geometry(img)
    if not np.issubdtype(img.get_data_dtype(), np.integer):
        raise ValueError("segmentation storage must be integer")
    data = np.asanyarray(img.dataobj)
    if not np.issubdtype(data.dtype, np.integer) or (data < 0).any():
        raise ValueError("scaled or negative segmentation labels")
    if compact and not np.isin(data, [0, 1, 2, 3]).all():
        raise ValueError("foreign compact capture label")
    return data


def patched_source(before):
    if hashlib.sha256(before.encode()).hexdigest() != PRISTINE_SHA256:
        raise ValueError("installed nnunet.py differs from the reviewed v2.17.0 source; capture refused")
    if before.count(ANCHOR) != 1 or before.index(ANCHOR) < before.index('img_pred = nib.load(tmp_dir / "s01.nii.gz")'):
        raise ValueError("ambiguous or misplaced capture anchor")
    after = before.replace(ANCHOR, HOOK + ANCHOR)
    compile(after, "ivcs-instrumented-nnunet.py", "exec")
    return after


def compact_venous(values, ids):
    """Compact 1/2/3 venous codes from a total-map volume: the published venous representation."""
    values = np.asanyarray(values)
    compact = np.zeros(values.shape, np.uint8)
    for code, label in enumerate(ids, 1):
        compact[values == label] = code
    return compact


def venous_ids(mapping):
    ids = []
    for name in NAMES:
        matches = [int(k) for k, v in mapping.items() if v == name]
        if len(matches) != 1:
            raise ValueError("missing/ambiguous global venous label")
        ids.append(matches[0])
    return ids


def cardiac_submodel(mapping):
    """The task-293 cardiac part of the installed class map, bound to the total-map ids.

    The part is taken from the installed package's own task-to-part table, so the sidecar
    records what the running inference used rather than a list typed here.
    """
    from totalsegmentator import map_to_binary as parts_module
    table = (getattr(parts_module, "map_taskid_to_partname_ct", None)
             or getattr(parts_module, "map_taskid_to_partname", None))
    if not table or CARDIAC_TASK not in table:
        raise ValueError("installed class map does not name the task-293 cardiac part")
    part = table[CARDIAC_TASK]
    classes = parts_module.class_map_5_parts[part]
    if len(classes) != CARDIAC_CLASSES:
        raise ValueError(f"cardiac part {part} does not hold {CARDIAC_CLASSES} classes")
    by_name = {}
    for total_id, name in mapping.items():
        by_name.setdefault(name, []).append(int(total_id))
    bound = {}
    for local_id, name in classes.items():
        ids = by_name.get(name, [])
        if len(ids) != 1:
            raise ValueError(f"cardiac class {name} is missing or ambiguous in the total class map")
        bound[str(int(local_id))] = {"name": name, "total_id": ids[0]}
    return {"task_id": CARDIAC_TASK, "part": part, "classes": bound}


def capture_internal(img, context):
    """Called exactly once by the instrumented total-task module; never changes img.

    Writes, from one in-memory array: the complete internal label map, the compact venous
    volume and the label sidecar (class map, cardiac sub-model, grid, voxel counts). All three
    are published or none is.
    """
    forbidden = [context.get(x) for x in ("crop", "roi_subset", "save_lowres", "remove_small_blobs", "test")]
    if (context.get("task_name") != "total" or context.get("output_task_name") != "total"
            or any(x is not None and x is not False and x != 0 for x in forbidden)
            or not np.allclose(context.get("resample"), 1.5, atol=1e-8, rtol=0)):
        raise ValueError("capture requires uncropped full-resolution total without optional postprocessing")
    grid = geometry(img)
    if not np.allclose(grid["spacing_mm"], [1.5] * 3, atol=1e-4, rtol=0):
        raise ValueError("capture is not the internal 1.5-mm grid")
    target = Path(os.environ["TS_AUDIT_CAPTURE"])
    outputs = {INTERNAL_VENOUS: target, INTERNAL_MULTILABEL: target.with_name(INTERNAL_MULTILABEL),
               INTERNAL_LABELS: target.with_name(INTERNAL_LABELS)}
    if any(path.exists() for path in outputs.values()):
        raise ValueError("multiple capture calls or stale capture output")
    mapping = {int(k): v for k, v in context["class_map"]["total"].items()}
    cardiac = cardiac_submodel(mapping)
    ids = venous_ids(mapping)
    values = np.asanyarray(img.dataobj)
    stored_dtype = str(values.dtype)
    if not np.issubdtype(values.dtype, np.integer):
        # A scaled header can hand back floats; integral values are a label map, anything else is not.
        if not np.isfinite(values).all() or not np.array_equal(values, np.round(values)):
            raise ValueError("internal prediction is not an integral label volume")
        values = values.astype(np.int64)
    if values.min() < 0 or values.max() > 255:
        raise ValueError("internal prediction is not an unsigned 8-bit label volume")
    counts = np.bincount(values.ravel(), minlength=256)   # no widened copy of a whole-body volume
    present = [label for label in range(1, 256) if counts[label]]
    foreign = [label for label in present if label not in mapping]
    if foreign:
        raise ValueError(f"internal prediction carries labels outside the class map: {foreign}")
    header = img.header.copy()
    header.set_data_dtype(np.uint8)
    sidecar = {"schema": 1, "grid": grid, "stored_dtype": stored_dtype,
               "total_class_map": {str(k): v for k, v in sorted(mapping.items())},
               "cardiac_submodel": cardiac,
               "venous_compact": {str(code): name for code, name in enumerate(NAMES, 1)},
               "voxel_counts": {mapping[label]: int(counts[label]) for label in present}}
    pending_full = outputs[INTERNAL_MULTILABEL].with_name("internal_multilabel.partial.nii.gz")
    nib.save(nib.Nifti1Image(values.astype(np.uint8, copy=False), img.affine, header), pending_full)
    pending_labels = outputs[INTERNAL_LABELS].with_name(INTERNAL_LABELS + ".partial")
    pending_labels.write_text(json.dumps(sidecar, sort_keys=True, indent=2, allow_nan=False) + "\n")
    pending_venous = target.with_name("internal.partial.nii.gz")
    nib.save(nib.Nifti1Image(compact_venous(values, ids), img.affine, header), pending_venous)
    pending_full.replace(outputs[INTERNAL_MULTILABEL])
    pending_labels.replace(outputs[INTERNAL_LABELS])
    pending_venous.replace(target)
    print("IVCS_MODEL_GRID_CAPTURED", target, grid, flush=True)


def _distribution_versions():
    return {p: importlib.metadata.version(p) for p in
            ["TotalSegmentator", "torch", "nnunetv2", "numpy", "scipy", "pandas", "nibabel"]}


def environment_record(weights_root=None):
    versions = _distribution_versions()
    if versions["TotalSegmentator"] != VERSION:
        raise ValueError("model-grid inference requires installed TotalSegmentator 2.17.0")
    import torch
    from totalsegmentator.config import get_weights_dir
    from totalsegmentator.python_api import get_task_config
    config = get_task_config("total", fast=False, fastest=False, quiet=True)
    if (config["task_id"] != [291, 292, 293, 294, 295] or config["resample"] != 1.5
            or config["trainer"] != "nnUNetTrainerNoMirroring" or config["folds"] != [0]
            or config["model"] != "3d_fullres" or config.get("crop") or config.get("cascade")):
        raise ValueError("installed total-task configuration differs from the reviewed full-resolution model")
    spec = importlib.util.find_spec("totalsegmentator.nnunet")
    source = Path(spec.origin).resolve()
    before = source.read_text()
    after = patched_source(before)
    package = source.parent
    root = Path(weights_root or get_weights_dir()).expanduser().resolve()
    # Pin only the five full-resolution models actually used, not unrelated cache tasks.
    weights = {}
    for task in range(291, 296):
        candidates = sorted(x for x in root.glob(f"Dataset{task}_*") if x.is_dir())
        if len(candidates) != 1:
            raise ValueError(f"one preinstalled Dataset{task} required in {root}; no downloads in array jobs")
        folder = candidates[0]
        checkpoints = list(folder.glob("nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"))
        if len(checkpoints) != 1:
            raise ValueError(f"full-resolution fold-0 checkpoint missing/ambiguous: {folder}")
        for required in ["plans.json", "dataset.json"]:
            if not list(folder.rglob(required)):
                raise ValueError(f"missing inference metadata {required}: {folder}")
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix in (".pth", ".json", ".pkl", ".npz"):
                weights[str(path.relative_to(root))] = sha(path)
    from connectivity_statistics import EM
    import gap_config
    import statlib
    code = [Path(__file__), Path(__file__).with_name("connectivity_statistics.py"), Path(MC.__file__),
            Path(ST.__file__), Path(EM.__file__), Path(gap_config.__file__), Path(statlib.__file__)]
    return {"schema": 1, "upstream_commit": UPSTREAM_COMMIT, "source_path": str(source),
            "pristine_nnunet_sha256": PRISTINE_SHA256, "instrumented_nnunet_sha256": hashlib.sha256(after.encode()).hexdigest(),
            "instrumentation": "process-local source loader; installed files unchanged",
            "patch_diff": "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile="nnunet.py", tofile="instrumented_nnunet.py")),
            "installed_source_files": {str(p.relative_to(package)): sha(p) for p in sorted(package.rglob("*.py"))},
            "versions": versions, "python": platform.python_version(), "python_executable": sys.executable,
            "installed_distributions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]},
            "torch_cuda": torch.version.cuda, "weights_root": str(root), "weight_files": weights,
            "resolved_total_task_configuration": config,
            "code": {str(p.resolve().relative_to(ST.REPO_ROOT.resolve())): sha(p) for p in code},
            "settings": {"task": "total", "fast": False, "fastest": False, "ml": True,
                         "roi_subset": None, "body_seg": False, "higher_order_resampling": False,
                         "resampling_order": 1, "nr_thr_resamp": 1, "nr_thr_saving": 2,
                         "remove_small_blobs": False, "device": "gpu", "tta": False,
                         "torch_force_no_weights_only_load": True, "send_usage_stats": False}}


def prepare(root, weights_root=None):
    root = Path(root)
    if root.resolve() in {ST.REPO_ROOT.resolve(), ST.RESULTS.resolve(), (ST.RESULTS / "connectivity").resolve()}:
        raise ValueError("a dedicated experiment root is required")
    value = environment_record(weights_root)
    record = {"status": "prepared_not_inferred", "identity": value, "fingerprint": identity(value)}
    path = root / "environment.json"
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError("experiment environment changed; use a new experiment root")
    atomic_json(path, record)
    return record


def canonical_frame(case_table, participant_table, phase0=None):
    phase0 = Path(phase0 or ST.PHASE0)
    cases = pd.read_csv(case_table).fillna("")
    people = pd.read_csv(participant_table).fillna("")
    if cases.duplicated(["cohort", "case"]).any() or people.duplicated(["cohort", "case"]).any():
        raise ValueError("duplicate canonical case")
    if Counter(people.cohort) != EXPECTED or len(cases) != 1873:
        raise ValueError("canonical processed/measurable frame counts disagree")
    for cohort, roster in [("AMOS", "amos_ct_cases.txt"), ("SAROS", "saros_processed_cases.txt"), ("autoPET", "autopet_cases.txt")]:
        names = [x.split()[0] for x in (phase0 / roster).read_text().splitlines() if x.split()]
        if len(names) != EXPECTED[cohort] or len(set(names)) != len(names) or set(names) != set(people.loc[people.cohort == cohort, "case"]):
            raise ValueError("processed roster identity mismatch")
    crosswalk = pd.read_csv(phase0 / "autopet_ts217_crosswalk.csv").set_index("phase1_case", verify_integrity=True)
    if crosswalk.baseline_case.duplicated().any() or len(crosswalk) != 1014:
        raise ValueError("autoPET crosswalk must be one-to-one for all 1,014 examinations")
    by_case = cases.set_index(["cohort", "case"], verify_integrity=True)
    rows = []
    for row in people.sort_values(["cohort", "case"]).to_dict("records"):
        cohort, case = component(row["cohort"]), component(row["case"])
        key = (cohort, case)
        measurable = key in by_case.index
        if int(row["gap_measurable"]) != measurable:
            raise ValueError("participant mapping disagrees with measurable roster")
        participant = f"{cohort}:{row['participant']}"
        base = {"cohort": cohort, "case": case, "participant": participant,
                "mask_case": crosswalk.loc[case, "baseline_case"] if cohort == "autoPET" else case,
                "expected_class26": "nonmeasurable", "expected_class6": "nonmeasurable"}
        if measurable:
            ref = by_case.loc[key]
            if participant != ref.participant or base["mask_case"] != ref.mask_case:
                raise ValueError("canonical participant/case crosswalk mismatch")
            base.update({"expected_class26": ref.class26, "expected_class6": ref.class6})
            for name in NAMES:
                base[name + "_sha256"] = ref[name + "_sha256"]
        rows.append(base)
    if len(set(cases.set_index(["cohort", "case"]).index) - set(zip(people.cohort, people.case))):
        raise ValueError("foreign measurable case")
    if Counter(x["expected_class26"] for x in rows) != EXPECTED_NATIVE or len({x["participant"] for x in rows}) != 1784:
        raise ValueError("canonical classifications or participant count mismatch")
    return rows


def ct_from_report(report, study_root):
    original = report.get("input")
    if not isinstance(original, str) or not Path(original).is_absolute():
        raise ValueError("baseline report lacks an absolute original CT input")
    path = Path(original)
    if path.is_file():
        return path.resolve()
    # Rebind the confirmed study Data tree only, retaining the complete report suffix.
    if original.count("/Data/") == 1:
        candidate = Path(study_root) / "Data" / original.split("/Data/", 1)[1]
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"original CT input unavailable: {original}; no directory-scan substitution")


def _source_hashes(row):
    return {k: sha(row[k]) for k in ["ct_path", "frozen_native", "frozen_report"]}


def _safe_sources(root, row):
    root = Path(root).resolve()
    for key in ["ct_path", "frozen_native", "frozen_report"]:
        p = Path(row[key]).resolve(strict=True)
        if p.is_relative_to(root) or root.is_relative_to(p.parent):
            raise ValueError("experiment outputs overlap an original input tree")


def preflight(root, case_table, participant_table, multilabel_root, study_root, stage_root, weights_root=None):
    """Verify the entire processed frame on CPU; publish the frozen manifest last."""
    from connectivity_statistics import EM
    root = Path(root).resolve()
    frozen = Path(multilabel_root).resolve()
    if root.is_relative_to(frozen) or frozen.is_relative_to(root):
        raise ValueError("experiment root must be separate from archived predictions")
    if not Path(stage_root).is_dir():
        raise ValueError("a writable node-local staging directory is required")
    provenance_paths = [case_table, participant_table, ST.PHASE0 / "autopet_ts217_crosswalk.csv", *ST.EXACT_ROSTERS.values()]
    canonical_hashes = {str(p): sha(p) for p in provenance_paths}
    record = prepare(root, weights_root)
    rows = canonical_frame(case_table, participant_table)
    verified = []
    problems = []
    for i, row in enumerate(rows):
        try:
            folder = Path(multilabel_root) / row["cohort"] / row["mask_case"]
            report = json.loads((folder / "report.json").read_text())
            mapping = EM.baseline_class_index(folder, NAMES)
            if any(report.get(k) for k in ["fastest", "roi_subset", "save_lowres", "body_seg"]):
                raise ValueError("baseline report indicates a different inference pathway")
            row.update({"ct_path": str(ct_from_report(report, study_root)),
                        "frozen_native": str((folder / "segmentation.nii.gz").resolve()),
                        "frozen_report": str((folder / "report.json").resolve()),
                        "label_indices": [mapping[n] for n in NAMES]})
            _safe_sources(root, row)
            before = _source_hashes(row)
            ct = nib.load(row["ct_path"])
            native = nib.load(row["frozen_native"])
            if not same_geometry(ct, native):
                raise ValueError("CT and archived native geometry differ")
            measured = topology(integer_volume(native), row["label_indices"])
            for conn in [26, 6]:
                if measured[f"class{conn}"] != row[f"expected_class{conn}"]:
                    raise ValueError("case-level native topology differs from canonical roster")
            # The study's binary writer (export_relevant_masks.export_case); hashes verify actual voxels,
            # not counts alone.
            if row["expected_class26"] != "nonmeasurable":
                with tempfile.TemporaryDirectory(prefix="ivcs-native-check-", dir=stage_root) as stage:
                    outputs, error = EM.export_case((row["cohort"], row["mask_case"], str(multilabel_root), stage, True, "", None), structures=NAMES)
                    if error or {x["structure"]: x["sha256"] for x in outputs} != {n: row[n + "_sha256"] for n in NAMES}:
                        raise ValueError(f"archived multilabel does not reproduce canonical binary hashes: {error}")
            if _source_hashes(row) != before:
                raise ValueError("source changed during preflight")
            row.update({"source_hashes": before, "native_geometry": geometry(native)})
            verified.append(row)
        except Exception as exc:
            problems.append({"cohort": row["cohort"], "case": row["case"], "error": str(exc)})
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            print("Model-grid preflight", i + 1, "/", len(rows), "problems", len(problems), flush=True)
    if {str(p): sha(p) for p in provenance_paths} != canonical_hashes:
        raise ValueError("canonical inputs changed during full-frame preflight")
    if len({r["ct_path"] for r in verified}) != len(verified):
        problems.append({"error": "multiple canonical examinations resolve to the same CT path"})
    payload = {"schema": 1, "status": "failed" if problems else "pass", "processed": len(rows),
               "verified": len(verified), "counts": EXPECTED, "problems": problems,
               "environment_fingerprint": record["fingerprint"],
               "canonical_inputs": canonical_hashes,
               "axis2_orientation_warnings": [{"cohort": r["cohort"], "case": r["case"], "alignment": r["native_geometry"]["axis2_si_alignment"]} for r in verified if r["native_geometry"]["axis2_si_alignment"] < .95]}
    if problems:
        atomic_json(root / "preflight.failed.json", payload)
        raise ValueError(f"full-frame preflight failed for {len(problems)} cases; no cases removed")
    manifest = root / "cases.json"
    if manifest.exists() and json.loads(manifest.read_text()) != verified:
        raise ValueError("manifest changed; use a new experiment root")
    atomic_json(manifest, verified)
    payload["manifest_sha256"] = sha(manifest)
    atomic_json(root / "preflight.json", payload)
    print("Model-grid preflight MATCH: 1899/1899, including all 26 nonmeasurable examinations", flush=True)


SUPERSESSION = "code_supersession.json"


def live_code_hashes(env):
    return {rel: sha(ST.REPO_ROOT / rel) for rel in env["identity"].get("code", {})}


def read_supersession(root, env):
    """The experiment root's recorded code supersession, or None when there is none.

    The environment record freezes the hashes of the analysis and launcher code at preflight and
    the contract refuses any later change. A supersession is the one recorded way past that: it
    names each changed file with its frozen and live hash, the reason and the decision, binds to
    the root's environment fingerprint and frozen hashes (a record for another root, or against
    other frozen code, is refused), and is written once per root. Editing environment.json would
    erase the evidence that the code changed; this keeps it.
    """
    path = Path(root) / SUPERSESSION
    if not path.is_file():
        return None
    sup = json.loads(path.read_text())
    required = {"schema", "environment_fingerprint", "frozen_code", "live_code", "changed", "reason",
                "decision", "recorded_unix"}
    if (not isinstance(sup, dict) or sup.get("schema") != 1 or not required <= set(sup)
            or not str(sup["reason"]).strip() or not str(sup["decision"]).strip()):
        raise ValueError("code supersession record is malformed")
    frozen = env["identity"].get("code", {})
    if sup["environment_fingerprint"] != env["fingerprint"] or sup["frozen_code"] != frozen:
        raise ValueError("code supersession does not belong to this experiment's frozen environment")
    changed = sorted(rel for rel in frozen if sup["live_code"].get(rel) != frozen[rel])
    if set(sup["live_code"]) != set(frozen) or sorted(sup["changed"]) != changed or not changed:
        raise ValueError("code supersession record is inconsistent with its own hashes")
    return sup


def expected_code(root, env):
    """The code hashes the live tree must carry: the frozen ones, or a recorded supersession's."""
    sup = read_supersession(root, env)
    return (sup["live_code"] if sup else env["identity"].get("code", {})), sup


def environment_unchanged(root, env):
    """The live environment equals the frozen record, code hashes taken from a supersession if any."""
    code, sup = expected_code(root, env)
    expected = dict(env["identity"])
    if sup:
        expected["code"] = code
    return environment_record(env["identity"]["weights_root"]) == expected


def supersede(root, reason, decision):
    """Record that the analysis/launcher code differs from the frozen environment record; once per root."""
    root = Path(root)
    if not str(reason or "").strip() or not str(decision or "").strip():
        raise ValueError("a code supersession needs its reason and decision")
    env = json.loads((root / "environment.json").read_text())
    if (root / SUPERSESSION).exists():
        raise ValueError("this experiment root is already superseded once; a further code change needs a new experiment root")
    frozen = env["identity"].get("code", {})
    live = live_code_hashes(env)
    changed = sorted(rel for rel in frozen if live[rel] != frozen[rel])
    if not changed:
        raise ValueError("nothing to supersede: the live code equals the frozen record")
    record = {"schema": 1, "environment_fingerprint": env["fingerprint"], "frozen_code": frozen, "live_code": live,
              "changed": changed, "reason": str(reason).strip(), "decision": str(decision).strip(),
              "recorded_unix": time.time(), "python_executable": sys.executable}
    atomic_json(root / SUPERSESSION, record)
    print("Model-grid code supersession recorded for", ", ".join(changed), flush=True)
    return record


def load_contract(root, recheck_environment=False):
    root = Path(root)
    pre = json.loads((root / "preflight.json").read_text())
    env = json.loads((root / "environment.json").read_text())
    if pre["status"] != "pass" or pre["verified"] != 1899 or sha(root / "cases.json") != pre["manifest_sha256"]:
        raise ValueError("missing, changed or unsuccessful full-frame preflight")
    if env["fingerprint"] != identity(env["identity"]) or pre["environment_fingerprint"] != env["fingerprint"]:
        raise ValueError("environment record identity mismatch")
    for path, digest in pre["canonical_inputs"].items():
        if sha(path) != digest:
            raise ValueError("canonical roster changed after preflight")
    code, sup = expected_code(root, env)
    if live_code_hashes(env) != code:
        raise ValueError("analysis/launcher code changed after preflight"
                         + (" and after its recorded supersession" if sup else ""))
    if recheck_environment and not environment_unchanged(root, env):
        raise ValueError("live source, packages or weights changed after environment freeze")
    rows = json.loads((root / "cases.json").read_text())
    if len(rows) != 1899 or Counter(r["cohort"] for r in rows) != EXPECTED or len({(r["cohort"], r["case"]) for r in rows}) != 1899:
        raise ValueError("incomplete or duplicated processed frame")
    return rows, pre, env


def verify_qc(root, row, pre, env, recheck_sources=True):
    folder = case_folder(root, row)
    qc = json.loads((folder / "qc.json").read_text())
    unequal = qc.get("unequal_voxels")
    if (qc["status"] not in ("pass", "mismatch") or qc["case_identity"] != identity(row)
            or qc["manifest_sha256"] != pre["manifest_sha256"]
            or qc["environment_fingerprint"] != env["fingerprint"]
            or qc["source_hashes"] != row["source_hashes"]
            or not isinstance(unequal, dict) or set(unequal) != set(NAMES)
            or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in unequal.values())
            or (qc["status"] == "pass") != (not any(unequal.values()))):
        raise ValueError("case lacks an identity-bound native reproduction record")
    # The record must carry both rerun classes so the analysis can report them.
    topology_record = qc.get("native_topology", {})
    if not all(isinstance(topology_record.get(f"class{conn}"), str) for conn in (26, 6)):
        raise ValueError("case lacks the rerun native topology record")
    if qc.get("internal_capture", {}).get("cardiac_classes") != CARDIAC_CLASSES:
        raise ValueError("case lacks the complete internal capture record")
    for name, digest in qc["outputs"].items():
        if name not in CASE_OUTPUTS or sha(folder / name) != digest:
            raise ValueError("saved case output changed or disappeared")
    if set(qc["outputs"]) != set(CASE_OUTPUTS):
        raise ValueError("incomplete case output hash inventory")
    if recheck_sources and _source_hashes(row) != row["source_hashes"]:
        raise ValueError("original input changed since preflight")
    return qc


def inference_worker(ct, native, internal):
    """Execute the reviewed module exactly once in a fresh process, then the stock API."""
    import importlib.machinery
    import torch
    if importlib.metadata.version("TotalSegmentator") != VERSION or not torch.cuda.is_available():
        raise ValueError("TotalSegmentator 2.17.0 and an allocated CUDA GPU are required; no CPU fallback")
    properties = torch.cuda.get_device_properties(0)
    print("IVCS_GPU_CONTEXT", {"device": torch.cuda.get_device_name(0),
          "capability": torch.cuda.get_device_capability(0), "cuda": torch.version.cuda,
          "cudnn": torch.backends.cudnn.version(),
          "multiprocessors": properties.multi_processor_count,
          "memory_bytes": properties.total_memory,
          "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
          "slurm_gpus": os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS")}, flush=True)
    # A MIG instance is identified above (its multiprocessor count and memory); nvidia-smi is
    # informational and must not abort a case when the instance's driver view differs.
    try:
        subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,uuid", "--format=csv,noheader"],
                       check=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        print("IVCS_GPU_CONTEXT nvidia-smi unavailable:", exc, flush=True)
    name = "totalsegmentator.nnunet"
    if name in sys.modules:
        raise ValueError("nnunet was imported before capture instrumentation")
    original = importlib.util.find_spec(name)
    source = Path(original.origin)
    instrumented = patched_source(source.read_text())

    class CaptureLoader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            return compile(instrumented, str(source), "exec")

    spec = importlib.util.spec_from_file_location(name, source, loader=CaptureLoader(name, str(source)))
    module = importlib.util.module_from_spec(spec)
    module.__dict__["_ivcs_capture_model_grid"] = capture_internal
    sys.modules[name] = module
    spec.loader.exec_module(module)
    setattr(sys.modules["totalsegmentator"], "nnunet", module)
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
    if sha(source) != PRISTINE_SHA256:
        raise ValueError("installed source changed during inference")


def run_case(root, index, stage_root):
    root = Path(root).resolve()
    rows, pre, env = load_contract(root, recheck_environment=True)
    if not 0 <= index < len(rows):
        raise ValueError("zero-based manifest index out of range")
    row = rows[index]
    _safe_sources(root, row)
    folder = case_folder(root, row)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prior = folder / "qc.json"
        if prior.exists() and json.loads(prior.read_text()).get("status") in ("pass", "mismatch"):
            verify_qc(root, row, pre, env)
            print("Model-grid verified resume", row["cohort"], row["case"], flush=True)
            return
        qc = {"schema": 1, "status": "started", "cohort": row["cohort"], "case": row["case"],
              "participant": row["participant"], "case_identity": identity(row),
              "manifest_sha256": pre["manifest_sha256"], "environment_fingerprint": env["fingerprint"],
              "source_hashes": row["source_hashes"], "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "code_supersession": sha(root / SUPERSESSION) if (root / SUPERSESSION).is_file() else None,
              "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"), "started_unix": time.time()}
        atomic_json(prior, qc)
        try:
            if _source_hashes(row) != row["source_hashes"]:
                raise ValueError("input changed before inference")
            with tempfile.TemporaryDirectory(prefix="ivcs-model-grid-", dir=stage_root) as stage:
                stage = Path(stage)
                ct = stage / "CT.nii.gz"
                # Source NIfTI may be uncompressed. Retain its actual suffix for byte staging.
                if not row["ct_path"].endswith(".nii.gz"):
                    ct = stage / "CT.nii"
                import shutil
                shutil.copyfile(row["ct_path"], ct)
                if sha(ct) != row["source_hashes"]["ct_path"]:
                    raise ValueError("staged CT bytes differ")
                native, internal, complete, sidecar_path, log = [stage / n for n in CASE_OUTPUTS]
                command = [sys.executable, str(Path(__file__).resolve()), "--inference-worker", str(ct), str(native), str(internal)]
                child_env = os.environ.copy()
                child_env["TOTALSEG_WEIGHTS_PATH"] = env["identity"]["weights_root"]
                child_env["TMPDIR"] = str(stage)
                child_env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
                with log.open("w") as stream:
                    result = subprocess.run(command, env=child_env, stdout=stream, stderr=subprocess.STDOUT)
                # Retain failure logs on persistent storage, including nonzero subprocess exits.
                shutil.copyfile(log, folder / "inference.log")
                qc["command"] = command
                qc["inference_exit_code"] = result.returncode
                if result.returncode != 0:
                    raise ValueError("inference failed; inspect persistent inference.log")
                missing = [p.name for p in (internal, complete, sidecar_path) if not p.is_file()]
                if missing:
                    raise ValueError(f"complete internal capture missing: {missing}")
                original = nib.load(row["frozen_native"])
                rerun = nib.load(native)
                model = nib.load(internal)
                full = nib.load(complete)
                if not same_geometry(original, rerun) or not same_geometry(nib.load(ct), rerun):
                    raise ValueError("rerun native geometry differs from source CT/archive")
                if not np.allclose(geometry(model)["spacing_mm"], [1.5] * 3, atol=1e-4, rtol=0):
                    raise ValueError("internal capture spacing differs from 1.5 mm")
                if not same_geometry(model, full):
                    raise ValueError("internal venous and complete captures differ in geometry")
                baseline_data, native_data = integer_volume(original), integer_volume(rerun)
                from totalsegmentator.map_to_binary import class_map
                installed = {int(k): v for k, v in class_map["total"].items()}
                actual_ids = venous_ids(installed)
                qc["unequal_voxels"] = {name: int(np.count_nonzero((baseline_data == old) != (native_data == new)))
                                        for name, old, new in zip(NAMES, row["label_indices"], actual_ids)}
                compact_data, complete_data = integer_volume(model, compact=True), integer_volume(full)
                if not np.array_equal(compact_data, compact_venous(complete_data, actual_ids)):
                    raise ValueError("internal venous capture disagrees with the complete internal label map")
                sidecar = json.loads(sidecar_path.read_text())
                if {int(k): v for k, v in sidecar["total_class_map"].items()} != installed:
                    raise ValueError("internal label sidecar disagrees with the installed class map")
                cardiac = sidecar["cardiac_submodel"]
                if (cardiac["task_id"] != CARDIAC_TASK or len(cardiac["classes"]) != CARDIAC_CLASSES
                        or not np.allclose(sidecar["grid"]["affine"], full.affine, atol=1e-6, rtol=0)
                        or list(sidecar["grid"]["shape"]) != list(full.shape)):
                    raise ValueError("internal label sidecar lacks the complete cardiac sub-model or its grid")
                native_classes = topology(native_data, actual_ids)
                internal_classes = topology(compact_data, [1, 2, 3])
                qc.update({"native_topology": native_classes, "internal_topology": internal_classes,
                           "native_geometry": geometry(rerun), "internal_geometry": geometry(model),
                           "internal_capture": {"cardiac_classes": len(cardiac["classes"]),
                                                "labels_present": len(sidecar["voxel_counts"]),
                                                "complete_map_voxels": int(np.count_nonzero(complete_data))}})
                if _source_hashes(row) != row["source_hashes"] or not environment_unchanged(root, env):
                    raise ValueError("inputs, installation or weights changed during inference")
                baseline_classes = topology(baseline_data, row["label_indices"])
                for conn in [26, 6]:
                    if baseline_classes[f"class{conn}"] != row[f"expected_class{conn}"]:
                        raise ValueError("archived per-case class does not match the canonical roster")
                qc["class_reproduced"] = all(native_classes[f"class{conn}"] == row[f"expected_class{conn}"] for conn in [26, 6])
                qc["status"] = "pass" if not any(qc["unequal_voxels"].values()) else "mismatch"
                for output in [native, internal, complete, sidecar_path]:
                    pending = folder / (output.name + ".partial")
                    shutil.copyfile(output, pending)
                    pending.replace(folder / output.name)
                qc["outputs"] = {p.name: sha(folder / p.name) for p in [native, internal, complete, sidecar_path, log]}
            qc["finished_unix"] = time.time()
            atomic_json(prior, qc)
            if qc["status"] == "pass":
                print("Model-grid case MATCH", index, row["cohort"], row["case"], flush=True)
            elif qc["class_reproduced"]:
                print("Model-grid case BOUNDARY", index, row["cohort"], row["case"], "unequal voxels",
                      json.dumps(qc["unequal_voxels"], sort_keys=True), flush=True)
            else:
                print("Model-grid case DIVERGENT", index, row["cohort"], row["case"], "archived class",
                      row["expected_class26"], row["expected_class6"], "rerun", native_classes["class26"],
                      native_classes["class6"], "unequal voxels", json.dumps(qc["unequal_voxels"], sort_keys=True), flush=True)
        except Exception as exc:
            if qc["status"] != "mismatch":
                qc["status"] = "failed"
            qc["error"] = str(exc)
            qc["finished_unix"] = time.time()
            atomic_json(prior, qc)
            raise
        finally:
            with (folder / "attempts.jsonl").open("a") as stream:
                stream.write(json.dumps(qc, sort_keys=True) + "\n")


def status(root):
    rows, pre, env = load_contract(root)
    inventory = []
    for row in rows:
        folder = case_folder(root, row)
        state, reason = "not_run", ""
        qc = {}
        if (folder / "qc.json").exists():
            try:
                qc = json.loads((folder / "qc.json").read_text())
                state = qc.get("status", "invalid")
                reason = qc.get("error", "")
                if state in ("pass", "mismatch"):
                    verify_qc(root, row, pre, env)
            except Exception as exc:
                state, reason = "invalid", str(exc)
        reproduced = None
        if state in ("pass", "mismatch"):
            # From the record's own topology, so a record without the class_reproduced field counts too.
            reproduced = all(qc["native_topology"][f"class{conn}"] == row[f"expected_class{conn}"] for conn in (26, 6))
        inventory.append({"cohort": row["cohort"], "case": row["case"], "participant": row["participant"],
                          "status": state, "reason": reason, "unequal_voxels": qc.get("unequal_voxels", {}),
                          "class_reproduced": reproduced,
                          "inference_exit_code": qc.get("inference_exit_code")})
    counts = dict(Counter(r["status"] for r in inventory))
    reproduction = {"exact": sum(r["status"] == "pass" for r in inventory),
                    "boundary": sum(r["status"] == "mismatch" and bool(r["class_reproduced"]) for r in inventory),
                    "class_divergent": sum(r["status"] in ("pass", "mismatch") and not r["class_reproduced"] for r in inventory)}
    atomic_json(Path(root) / "qc_status.json", {
        "status": "complete" if all(r["status"] in ("pass", "mismatch") for r in inventory) else "blocked",
        "gate": "every case complete with an identity-bound record; voxel identity and class agreement with the archive are reported, not required",
        "counts": counts, "reproduction": reproduction, "cases": inventory})
    return inventory


def analyze(root, n_boot=10000, seed=20261025):
    from connectivity_statistics import model_grid_summaries
    root = Path(root)
    (root / "paired_connectivity.summary.json").unlink(missing_ok=True)
    atomic_json(root / "analysis_status.json", {"status": "validating_not_complete"})
    inventory = status(root)
    if any(r["status"] not in ("pass", "mismatch") for r in inventory):
        atomic_json(root / "analysis_status.json", {"status": "blocked", "reason": "all 1899 cases must complete inference with an identity-bound record", "counts": dict(Counter(r["status"] for r in inventory))})
        raise ValueError("projection-specific analysis blocked; inspect qc_status.json; no complete-case selection")
    rows, pre, env = load_contract(root, recheck_environment=True)
    _, supersession = expected_code(root, env)
    paired, fidelity = [], []
    for index, row in enumerate(rows):
        qc = verify_qc(root, row, pre, env, recheck_sources=False)
        folder = case_folder(root, row)
        native = nib.load(folder / "native_multilabel.nii.gz")
        internal = nib.load(folder / INTERNAL_VENOUS)
        complete = nib.load(folder / INTERNAL_MULTILABEL)
        frozen = nib.load(row["frozen_native"])
        if not same_geometry(native, frozen) or not np.allclose(geometry(internal)["spacing_mm"], [1.5] * 3, atol=1e-4, rtol=0):
            raise ValueError("analysis geometry does not reproduce the native/archive or internal grid")
        from totalsegmentator.map_to_binary import class_map
        ids = venous_ids({int(k): v for k, v in class_map["total"].items()})
        native_data, frozen_data = integer_volume(native), integer_volume(frozen)
        # Voxel fidelity is re-derived here and must equal the QC record; it is reported, not required.
        unequal = {name: int(np.count_nonzero((native_data == new) != (frozen_data == old)))
                   for name, new, old in zip(NAMES, ids, row["label_indices"])}
        if unequal != qc["unequal_voxels"]:
            raise ValueError("analysis independently detected native voxel differences the QC record does not carry")
        compact_data = integer_volume(internal, compact=True)
        if not same_geometry(internal, complete) or not np.array_equal(compact_data, compact_venous(integer_volume(complete), ids)):
            raise ValueError("analysis independently detected an internal venous/complete capture disagreement")
        native_classes = topology(native_data, ids)
        internal_classes = topology(compact_data, [1, 2, 3])
        if native_classes != qc["native_topology"] or internal_classes != qc["internal_topology"]:
            raise ValueError("recomputed output topology differs from QC record")
        item = {k: row[k] for k in ["cohort", "case", "participant", "mask_case"]}
        item["native_spacing_mm"] = row["native_geometry"]["spacing_mm"][2]
        for conn in [26, 6]:
            item[f"native_class{conn}"] = native_classes[f"class{conn}"]
            item[f"internal_class{conn}"] = internal_classes[f"class{conn}"]
            item[f"archive_class{conn}"] = row[f"expected_class{conn}"]
        paired.append(item)
        class_agrees = all(native_classes[f"class{conn}"] == row[f"expected_class{conn}"] for conn in [26, 6])
        fidelity.append({**{k: row[k] for k in ["cohort", "case", "participant"]},
                         "reproduction": "exact" if qc["status"] == "pass" else ("boundary" if class_agrees else "class_divergent"),
                         **{f"unequal_voxels_{name}": unequal[name] for name in NAMES},
                         **{f"archive_voxels_{name}": int(np.count_nonzero(frozen_data == old)) for name, old in zip(NAMES, row["label_indices"])},
                         **{f"rerun_voxels_{name}": int(np.count_nonzero(native_data == new)) for name, new in zip(NAMES, ids)},
                         "archive_class26": row["expected_class26"], "rerun_native_class26": native_classes["class26"],
                         "archive_class6": row["expected_class6"], "rerun_native_class6": native_classes["class6"]})
        if (index + 1) % 100 == 0:
            print("Model-grid paired classification", index + 1, "/", len(rows), flush=True)
    frame = pd.DataFrame(paired)
    if Counter(frame.archive_class26) != EXPECTED_NATIVE:
        raise ValueError("secondary native aggregate check failed")
    frames = model_grid_summaries(frame, n_boot, seed)
    archive_frame = frame.drop(columns=["native_class26", "native_class6"]).rename(
        columns={"archive_class26": "native_class26", "archive_class6": "native_class6"})
    frames.update({name.replace(".csv", "_archive_class.csv"): data
                   for name, data in model_grid_summaries(archive_frame, n_boot, seed).items()})
    frames["paired_connectivity.csv"] = frame
    frames["reproduction_fidelity.csv"] = pd.DataFrame(fidelity)
    divergent = [{"cohort": f["cohort"], "case": f["case"], "participant": f["participant"], "adjacency": str(conn),
                  "archive": f[f"archive_class{conn}"], "rerun": f[f"rerun_native_class{conn}"]}
                 for f in fidelity for conn in (26, 6) if f[f"archive_class{conn}"] != f[f"rerun_native_class{conn}"]]
    for name, data in frames.items():
        pending = root / (name + ".partial")
        data.to_csv(pending, index=False, float_format="%.10g")
        pending.replace(root / name)
    # Publication is last. Detect modifications during classification/bootstrap as
    # well as before it; a failure leaves analysis_status explicitly incomplete.
    for row in rows:
        verify_qc(root, row, pre, env)
    if not environment_unchanged(root, env):
        raise ValueError("environment changed during paired analysis")
    completion = {"status": "complete_projection_representation_audit_not_anatomical_accuracy", "processed": len(frame),
                  "native_class26_counts": EXPECTED_NATIVE, "bootstrap_resamples": n_boot, "seed": seed,
                  "environment_fingerprint": env["fingerprint"], "manifest_sha256": pre["manifest_sha256"],
                  "difference_direction": "internal minus returned native", "intervals": "pointwise cohort-stratified paired participant percentile",
                  "internal_prediction_scope": "captured rerun; exact native equality does not prove the archived run's internal prediction was the same",
                  "internal_capture": "complete total class map with venous compact codes and label sidecar",
                  "native_reproduction": {"gate": "every case complete with an identity-bound record; voxel identity and class agreement with the archive reported, not required",
                                          "exact_cases": sum(f["reproduction"] == "exact" for f in fidelity),
                                          "boundary_cases": sum(f["reproduction"] == "boundary" for f in fidelity),
                                          "class_divergent_cases_n": sum(f["reproduction"] == "class_divergent" for f in fidelity),
                                          "max_unequal_voxels": {name: max(f[f"unequal_voxels_{name}"] for f in fidelity) for name in NAMES},
                                          "class_agreement": f"{sum(f['reproduction'] != 'class_divergent' for f in fidelity)}/{len(rows)}",
                                          "class_agreement26": f"{sum(f['archive_class26'] == f['rerun_native_class26'] for f in fidelity)}/{len(rows)}",
                                          "class_agreement6": f"{sum(f['archive_class6'] == f['rerun_native_class6'] for f in fidelity)}/{len(rows)}",
                                          "class_divergent_cases": divergent,
                                          "archive_native_class26_counts": dict(Counter(frame.archive_class26)),
                                          "rerun_native_class26_counts": dict(Counter(frame.native_class26)),
                                          "sensitivity": "the *_archive_class.csv summaries substitute the archived class for class-divergent cases"},
                  "code_supersession": supersession,
                  "outputs": {name: sha(root / name) for name in frames},
                  "qc_record_identity": identity({f"{r['cohort']}/{r['case']}": sha(case_folder(root, r) / "qc.json") for r in rows})}
    atomic_json(root / "paired_connectivity.summary.json", completion)
    atomic_json(root / "analysis_status.json", completion)
    print("Model-grid analysis MATCH: all 1899 cases complete; reproduction fidelity reported", flush=True)


def dispatch(args):
    root = args.model_grid_root
    if root is None:
        raise ValueError("--model-grid-root is required; frozen publication outputs are never a default destination")
    mode = args.model_grid_mode
    root.mkdir(parents=True, exist_ok=True)
    if mode != "run":
        with (root / ".experiment.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return _dispatch_unlocked(args)
    return _dispatch_unlocked(args)


def _dispatch_unlocked(args):
    root, mode = args.model_grid_root, args.model_grid_mode
    if mode == "prepare":
        prepare(root, args.model_grid_weights_root)
    elif mode == "preflight":
        preflight(root, args.case_table, args.model_grid_participants, args.multilabel_root,
                  args.model_grid_study_root, args.model_grid_stage_root, args.model_grid_weights_root)
    elif mode == "run":
        if args.model_grid_index is None or args.model_grid_stage_root is None:
            raise ValueError("run requires --model-grid-index and --model-grid-stage-root")
        run_case(root, args.model_grid_index, args.model_grid_stage_root)
    elif mode == "status":
        print(Counter(r["status"] for r in status(root)))
    elif mode == "supersede":
        supersede(root, args.model_grid_supersede_reason, args.model_grid_supersede_decision)
    elif mode == "analyze":
        analyze(root, args.model_grid_bootstrap, args.model_grid_seed)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--inference-worker":
        inference_worker(*map(Path, sys.argv[2:]))
    else:
        raise SystemExit("Use connectivity_statistics.py --model-grid-mode; this module is the private inference worker")
