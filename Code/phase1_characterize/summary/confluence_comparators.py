#!/usr/bin/env python3
"""    python confluence_comparators.py --model-grid-root ROOT --recorded-root RECORDED \
        --study-root STUDY --out OUT [--workers N] [--limit N] [--validate-only]

For every examination in ROOT/cases.json it reads the archived TotalSegmentator 2.17.0 multilabel
(native grid), the model-grid capture (1.5-mm inference grid) and the CT, each checked against the
SHA-256 recorded for it (paths recorded under RECORDED are read under STUDY), and writes to OUT:

  comparators_cases.csv    reachability between the largest 26-connected components of the IVC and the
                           right iliac vein, and of the aorta and each common iliac artery, through the
                           union of the pair under 26- and 6-connectivity, on both grids; no image values
  attenuation_cases.csv    the median CT number over each whole venous label on the native grid, per
                           examination (it is not released; only its SHA-256 is recorded)
  attenuation_summary.csv  aggregates only: quartiles by cohort and native class, and examination counts
                           by cohort, class and attenuation quartile
  manifest.json            arguments, inputs, outputs with their SHA-256, and failures by case

Before anything is written for an examination, the paper's own IVC-to-left-iliac classes are recomputed
on each grid and must equal the classes recorded for it. No examination is selected, excluded or ordered
by CT number; the attenuation is a covariate only.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib as _pl
import sys as _sys

for _p in sorted(_pl.Path(__file__).resolve().parents[2].iterdir()):
    if _p.is_dir() and _p.name[0] not in "._" and str(_p) not in _sys.path:
        _sys.path.append(str(_p))  # append, never shadow stdlib

import gap_content_analysis as GC
import measure_collapsed_gap as MCG
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

PAIRS = {"ivc_rciv": ("inferior_vena_cava", "iliac_vena_right"),
         "aorta_iliac_artery_left": ("aorta", "iliac_artery_left"),
         "aorta_iliac_artery_right": ("aorta", "iliac_artery_right")}
VENOUS = {"ivc": "inferior_vena_cava", "lciv": "iliac_vena_left", "rciv": "iliac_vena_right"}
ADJACENCY = {"26": np.ones((3, 3, 3), bool), "6": ndi.generate_binary_structure(3, 1)}
CASE_COLUMNS = ["cohort", "case", "mask_case", "participant"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def translate(path, recorded_root, study_root):
    """A path recorded under the original study root, read under the local study root."""
    text, root = str(path), str(recorded_root).rstrip("/")
    if not text.startswith(root + "/"):
        raise ValueError(f"not under the recorded study root {root}: {text}")
    return _pl.Path(study_root) / text[len(root) + 1:]


def reach(volume, index, first, second):
    """Whether the largest 26-connected components of two labels share a component of their union."""
    a, b = volume == index[first], volume == index[second]
    if not a.any() or not b.any():
        return {rule: "nonmeasurable" for rule in ADJACENCY}
    head, _, _ = MCG.largest_component26(a)
    tail, _, _ = MCG.largest_component26(b)
    out = {}
    for rule, structure in ADJACENCY.items():
        labels, _ = ndi.label(a | b, structure)
        out[rule] = "connected" if MCG.share_component(labels, head.astype(bool), tail.astype(bool)) else "disconnected"
    return out


def venous_classes(volume, index):
    """The paper's IVC-to-left-iliac classes (measure_collapsed_gap.classify_topology)."""
    masks = tuple(volume == index[VENOUS[key]] for key in ("ivc", "lciv", "rciv"))
    if not masks[0].any() or not masks[1].any():
        return {"26": "nonmeasurable", "6": "nonmeasurable"}
    result = MCG.classify_topology(masks)
    return {"26": result["class26"], "6": result["class6"]}


def attenuation(volume, index, ct):
    """Median CT number over each whole venous label and over their union."""
    out, union = {}, np.zeros(volume.shape, bool)
    for key, name in VENOUS.items():
        mask = volume == index[name]
        union |= mask
        out[f"voxels_{key}"] = int(mask.sum())
        out[f"median_hu_{key}"] = float(np.median(ct[mask])) if mask.any() else float("nan")
    out["median_hu_venous"] = float(np.median(ct[union])) if union.any() else float("nan")
    return out


def _labels(loaded):
    return np.asanyarray(loaded["image"].dataobj).astype(np.int32, copy=False), loaded["index"]


def process_case(row, root, recorded_root, study_root):
    """One examination: verified inputs, the venous-class control, reachability and attenuation."""
    paths = {key: translate(row[key], recorded_root, study_root) for key in ("frozen_native", "frozen_report", "ct_path")}
    for key, path in paths.items():
        if sha256(path) != row["source_hashes"][key]:
            raise ValueError(f"{key} does not match its recorded SHA-256: {path}")
    native = GC.load_native(paths["frozen_native"].parent)
    inference = GC.load_internal(_pl.Path(root) / row["cohort"] / row["case"])
    grids = {"native": _labels(native), "inference": _labels(inference)}
    recorded = {"native": {"26": row["expected_class26"], "6": row["expected_class6"]},
                "inference": {"26": inference["binding"]["class26"], "6": inference["binding"]["class6"]}}
    out = {key: row[key] for key in CASE_COLUMNS}
    for grid, (volume, index) in grids.items():
        classes = venous_classes(volume, index)
        if classes != recorded[grid]:
            raise ValueError(f"{grid} grid: recomputed classes {classes} differ from the recorded {recorded[grid]}")
        for pair, (first, second) in PAIRS.items():
            for rule, value in reach(volume, index, first, second).items():
                out[f"{grid}_{pair}_{rule}"] = value
    ct = nib.load(str(paths["ct_path"]))
    volume, index = grids["native"]
    if ct.shape[:3] != volume.shape or not np.allclose(ct.affine, native["image"].affine, atol=1e-3):
        raise ValueError("the CT and the native label map do not share a grid")
    hu = attenuation(volume, index, np.asanyarray(ct.dataobj))
    return out, {**{key: row[key] for key in CASE_COLUMNS}, "native_class26": recorded["native"]["26"], **hu}


def _worker(task):
    row, root, recorded_root, study_root = task
    try:
        reach_row, hu_row = process_case(row, root, recorded_root, study_root)
        return {"case": row["case"], "reach": reach_row, "hu": hu_row}
    except Exception as error:   # recorded by case and counted; the run never hides a failure
        return {"case": row["case"], "error": f"{type(error).__name__}: {error}"}


def summarize(hu):
    """Aggregates only: quartiles by cohort and native class, counts by attenuation quartile."""
    rows = []
    measurable = hu[hu.native_class26 != "nonmeasurable"].copy()
    for cohort, part in [("pooled", measurable)] + list(measurable.groupby("cohort", sort=True)):
        for label, group in [("all", part)] + list(part.groupby("native_class26", sort=True)):
            for column in ("median_hu_venous", "median_hu_ivc", "median_hu_lciv", "median_hu_rciv"):
                values = group[column].dropna()
                if len(values):
                    q25, q50, q75 = np.percentile(values, [25, 50, 75])
                    rows.append({"kind": "quartiles", "cohort": cohort, "native_class26": label, "measure": column,
                                 "n": int(len(values)), "q25": float(q25), "median": float(q50), "q75": float(q75)})
    cuts = np.percentile(measurable.median_hu_venous.dropna(), [25, 50, 75])
    measurable["quartile"] = np.searchsorted(cuts, measurable.median_hu_venous, side="right") + 1
    for (cohort, quartile, label), group in measurable.groupby(["cohort", "quartile", "native_class26"], sort=True):
        rows.append({"kind": "count_by_quartile", "cohort": cohort, "native_class26": label, "measure": "median_hu_venous",
                     "quartile": int(quartile), "n": int(len(group))})
    rows += [{"kind": "quartile_cut", "measure": "median_hu_venous", "quartile": i + 1, "median": float(c)}
             for i, c in enumerate(cuts)]
    return pd.DataFrame(rows)


def _publish(path, data):
    part = path.with_name(path.name + ".part")
    part.write_bytes(data)
    os.replace(part, path)
    return hashlib.sha256(data).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-grid-root", type=_pl.Path, required=True)
    parser.add_argument("--recorded-root", required=True)
    parser.add_argument("--study-root", type=_pl.Path, required=True)
    parser.add_argument("--out", type=_pl.Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--validate-only", action="store_true",
                        help="process the first --limit examinations (default 3) and write nothing but a report")
    args = parser.parse_args(argv)
    cases_path = args.model_grid_root / "cases.json"
    cases = json.loads(cases_path.read_bytes())
    limit = args.limit if args.limit is not None else (3 if args.validate_only else None)
    rows = cases[:limit] if limit is not None else cases
    tasks = [(row, args.model_grid_root, args.recorded_root, args.study_root) for row in rows]
    if args.workers > 1:
        with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(_worker, tasks, chunksize=4))
    else:
        results = [_worker(task) for task in tasks]
    failures = {r["case"]: r["error"] for r in results if "error" in r}
    if args.validate_only:
        print(json.dumps({"processed": len(results), "failures": failures}, indent=2))
        return 1 if failures else 0
    args.out.mkdir(parents=True, exist_ok=True)
    reach_frame = pd.DataFrame([r["reach"] for r in results if "reach" in r])
    hu_frame = pd.DataFrame([r["hu"] for r in results if "hu" in r])
    outputs = {"comparators_cases.csv": reach_frame.to_csv(index=False).encode("utf-8"),
               "attenuation_cases.csv": hu_frame.to_csv(index=False, float_format="%.6g").encode("utf-8"),
               "attenuation_summary.csv": summarize(hu_frame).to_csv(index=False, float_format="%.6g").encode("utf-8")}
    manifest = {"producer_sha256": sha256(__file__), "arguments": {k: str(v) for k, v in vars(args).items()},
                "cases_json_sha256": sha256(cases_path), "examinations": len(rows), "failures": failures,
                "outputs": {name: _publish(args.out / name, data) for name, data in outputs.items()}}
    _publish(args.out / "manifest.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(f"{len(rows) - len(failures)} examinations written, {len(failures)} failed; outputs in {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
