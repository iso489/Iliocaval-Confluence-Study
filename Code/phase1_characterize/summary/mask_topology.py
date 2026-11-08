#!/usr/bin/env python3
"""    python Code/phase1_characterize/summary/mask_topology.py --masks MASKS \
        --reference-cache VENOUS_CACHE [--workers 2]

MASKS is the archived TotalSegmentator 2.17.0 native-grid export (a folder per cohort and mask case
holding the IVC, both iliac veins, the vertebrae T12 to S1 and the sacrum, with MANIFEST.csv);
VENOUS_CACHE is the venous audit's cache of the released annotations. For each measurable examination
(connectivity/analysis_cases.csv) it reads the three venous masks, each checked against the
SHA-256 the analysis recorded, and computes the Betti numbers (b0, b1, b2) of their union: components,
tunnels and cavities, with a 26-connected foreground and a 6-connected background
(overlap_metrics.betti_numbers). The anatomical confluence, one tube dividing in two, is (1, 0, 0).
b0 must equal the component count the analysis recorded (`ncomp26`).

For each disconnected examination it then places the corridor centroid (gap_content_native: the corridor
between the dominant components) on a vertebral ruler built from the examination's own vertebra masks,
each checked against MANIFEST.csv, by measure_bifurcation's rule: every vertebra anchored at its
craniocaudal midpoint, the sacrum's top at the L5/S1 disc when S1 is empty, linear between anchors, and
flagged when the anchors are out of order or skip a level. Before the table's position is used, the
caudal end of the dominant IVC component computed here must equal the table's. The released annotations
the venous audit measured get the same Betti numbers, and b0 must equal that audit's `ncomp26`. Every
mask is read once and parsed from the bytes it hashed.

Outputs, in Results/phase1_characterize/summary/: mask_topology_cases.csv,
reference_topology_cases.csv and mask_topology_run_record.json; summary_tables.py computes the rates
and intervals from them.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import pathlib as _pl
import platform
import sys as _sys
from concurrent.futures import ProcessPoolExecutor

for _p in sorted(_pl.Path(__file__).resolve().parents[2].iterdir()):
    if _p.is_dir() and _p.name[0] not in "._" and str(_p) not in _sys.path:
        _sys.path.append(str(_p))  # append, never shadow stdlib

import gap_content_analysis as GC
import measure_bifurcation as MB
import measure_collapsed_gap as MCG
import numpy as np
import overlap_metrics as OM
import pandas as pd

PRODUCER = _pl.Path(__file__).resolve()
PRODUCER_SHA256 = hashlib.sha256(PRODUCER.read_bytes()).hexdigest()   # stamped when the run starts
ROOT = PRODUCER.parents[3]
RESULTS = ROOT / "Results" / "phase1_characterize"
OUT = RESULTS / "summary"
INPUTS = {"analysis_cases": RESULTS / "connectivity" / "analysis_cases.csv",
          "gap_native": RESULTS / "gap_content_native" / "gap_content_cases.csv",
          "reference_cases": RESULTS / "connectivity" / "reference_cases.csv",
          "reference_source_files": RESULTS / "connectivity" / "reference_source_files.csv"}
VEINS = ("inferior_vena_cava", "iliac_vena_left", "iliac_vena_right")


def read_mask(path, expected):
    """A binary mask parsed from the very bytes whose SHA-256 must equal the recorded one."""
    data, digest = GC.read_hashed(path)
    if digest != expected:
        raise ValueError(f"{path} is not the file its record names")
    image = GC.nifti_from_bytes(data, path.name)
    arr = np.asanyarray(image.dataobj)
    if arr.dtype.kind not in "bui" or arr.min() < 0 or arr.max() > 1:
        raise ValueError(f"{path} is not a binary integer mask")
    return image, arr.astype(bool)


def box(mask):
    """The bounding box of a non-empty mask, as slices."""
    support = [np.flatnonzero(mask.any(axis=tuple(j for j in range(3) if j != i))) for i in range(3)]
    return tuple(slice(int(s[0]), int(s[-1]) + 1) for s in support)


def betti(mask):
    """(b0, b1, b2) of a mask under 26-connectivity, on its bounding box (a crop changes none of them)."""
    if not mask.any():
        return (0, 0, 0)
    return OM.betti_numbers(mask[box(mask)], 26)


def ras_z(image, mask, offset=(0, 0, 0)):
    """Craniocaudal extent (min, max), in mm, of a mask's voxel centres (RAS and LPS share +z); `offset`
    is the index of the mask's first voxel when it is a crop."""
    index = np.argwhere(mask) + np.asarray(offset)
    z = index @ np.asarray(image.affine)[2, :3] + np.asarray(image.affine)[2, 3]
    return float(z.min()), float(z.max())


def union_of(paths):
    """The union of three masks on one grid, with the first mask's image and the masks themselves."""
    first, union, masks = None, None, []
    for path, expected in paths:
        image, mask = read_mask(path, expected)
        if first is None:
            first, union = image, mask.copy()
        elif image.shape != first.shape or not np.allclose(image.affine, first.affine):
            raise ValueError(f"{path} is not on the grid of {paths[0][0].name}")
        else:
            union |= mask
        masks.append(mask)
    return first, union, masks


def vertebra_anchors(case_dir, digests):
    """measure_bifurcation.read_vertebra_anchors on hashed bytes: [(level, anchor_z_mm), ...]."""
    anchors = []
    for name, level in MB.VERT_LEVELS:
        image, mask = read_mask(case_dir / f"{name}.nii.gz", digests[name])
        if mask.any():
            crop = box(mask)
            low, high = ras_z(image, mask[crop], [s.start for s in crop])
            anchors.append((level, 0.5 * (low + high)))
    if not any(level == 6.0 for level, _ in anchors):        # 6.0 is S1 in VERT_LEVELS
        image, mask = read_mask(case_dir / "sacrum.nii.gz", digests["sacrum"])
        if mask.any():
            crop = box(mask)
            anchors.append((5.5, ras_z(image, mask[crop], [s.start for s in crop])[1]))   # sacrum top = the L5/S1 disc
    return anchors


def measure_case(task):
    """Betti numbers of one examination's venous union, and the vertebral level of its corridor if disconnected."""
    masks_root, row, digests, gap = task
    case_dir = masks_root / row["cohort"] / row["mask_case"]
    image, union, (ivc, _, _) = union_of([(case_dir / f"{name}.nii.gz", row[f"{name}_sha256"]) for name in VEINS])
    b0, b1, b2 = betti(union)
    if b0 != int(float(row["ncomp26"])):     # stored as "2" or "2.0"
        raise ValueError(f"{row['cohort']} {row['case']}: b0 {b0} differs from the recorded ncomp26 {row['ncomp26']}")
    out = {"cohort": row["cohort"], "case": row["case"], "class26": row["class26"], "b0": b0, "b1": b1, "b2": b2,
           "gap_z_mm": math.nan, "ruler_qc": "", "gap_level": math.nan, "gap_in_range": 0,
           "gap_level_bin": "", "gap_level_label": ""}
    if row["class26"] != "disconnected":
        return out
    crop = box(ivc)
    dominant, _, _ = MCG.largest_component26(ivc[crop])     # a crop keeps the array-order tie rule
    caudal, offset = float(gap["ivc_caudal_z_mm"]), float(gap["corridor_centroid_z_minus_ivc_caudal_mm"])
    if abs(ras_z(image, dominant, [s.start for s in crop])[0] - caudal) > 1e-3:
        raise ValueError(f"{row['cohort']} {row['case']}: the gap-content table is not bound to these masks")
    ruler = MB.build_vert_ruler(vertebra_anchors(case_dir, digests))
    out["ruler_qc"] = ruler[3]
    if math.isfinite(offset):
        level, inside = MB.z_to_vert_level(ruler, caudal + offset)
        out.update({"gap_z_mm": caudal + offset, "gap_level": level, "gap_in_range": int(inside)})
        if math.isfinite(level):
            out.update({"gap_level_bin": MB.LEVEL_NAME.get(MB.level_bin(level), str(MB.level_bin(level))),
                        "gap_level_label": MB._level_label(level)})
    return out


def measure_reference(task):
    """Betti numbers of one released annotation's venous union."""
    cache, case, digests, ncomp = task
    _, union, _ = union_of([(cache / case / "segmentations" / f"{name}.nii.gz", digests[name]) for name in VEINS])
    b0, b1, b2 = betti(union)
    if b0 != ncomp:
        raise ValueError(f"{case}: b0 {b0} differs from the venous audit's ncomp26 {ncomp}")
    return {"case": case, "b0": b0, "b1": b1, "b2": b2}


def _publish(path, data):
    part = path.with_name(path.name + ".part")
    with open(part, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(part, path)
    return hashlib.sha256(data).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--masks", type=_pl.Path, required=True)
    parser.add_argument("--reference-cache", type=_pl.Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", type=_pl.Path, default=OUT)
    args = parser.parse_args(argv)
    raw = {name: path.read_bytes() for name, path in INPUTS.items()}
    manifest_bytes = (args.masks / "MANIFEST.csv").read_bytes()
    digests = {name: hashlib.sha256(data).hexdigest() for name, data in raw.items()}
    frames = {name: pd.read_csv(io.BytesIO(data), keep_default_na=False) for name, data in raw.items()}
    manifest = pd.read_csv(io.BytesIO(manifest_bytes), keep_default_na=False)
    cases = frames["analysis_cases"]
    if len(cases) != 1873 or int(cases.class26.eq("disconnected").sum()) != 892:
        raise ValueError("the analysis frame is not the 1,873 measurable examinations with 892 disconnected")
    gap = frames["gap_native"].set_index(["cohort", "case"])
    structures = {key: dict(zip(g.structure, g.sha256)) for key, g in manifest.groupby(["dataset", "case"])}
    tasks = [(args.masks, r, structures[(r["cohort"], r["mask_case"])],
              gap.loc[(r["cohort"], r["case"])] if r["class26"] == "disconnected" else None)
             for r in cases.to_dict("records")]
    reference = frames["reference_cases"]
    measured = reference[reference.status == "measurable"]
    if len(measured) != 600:
        raise ValueError("the venous audit measured 600 released cases")
    sources = dict(zip(frames["reference_source_files"].name, frames["reference_source_files"].sha256))
    ref_tasks = [(args.reference_cache, c, {n: sources[f"{c}/segmentations/{n}.nii.gz"] for n in VEINS}, int(float(n)))
                 for c, n in zip(measured.case, measured.ncomp26)]
    rows, refs = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(measure_case, tasks, chunksize=4), 1):
            rows.append(row)
            if i % 200 == 0 or i == len(tasks):
                print(f"examinations {i}/{len(tasks)}", flush=True)
        for i, row in enumerate(pool.map(measure_reference, ref_tasks, chunksize=8), 1):
            refs.append(row)
            if i % 200 == 0 or i == len(ref_tasks):
                print(f"released annotations {i}/{len(ref_tasks)}", flush=True)
    table = pd.DataFrame(rows)
    ref_table = pd.DataFrame(refs).merge(measured[["case", "split"]], on="case", validate="one_to_one")[
        ["case", "split", "b0", "b1", "b2"]]
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = {"mask_topology_cases.csv": _publish(args.out / "mask_topology_cases.csv",
                                                   table.to_csv(index=False, float_format="%.6f").encode("utf-8")),
               "reference_topology_cases.csv": _publish(args.out / "reference_topology_cases.csv",
                                                        ref_table.to_csv(index=False).encode("utf-8"))}
    record = {"producer": PRODUCER.relative_to(ROOT).as_posix(), "producer_sha256": PRODUCER_SHA256,
              "betti_rule": "overlap_metrics.betti_numbers(mask, 26): 26-connected foreground, 6-connected background",
              "ruler_rule": "measure_bifurcation.VERT_LEVELS, build_vert_ruler, z_to_vert_level, level_bin, _level_label",
              "python": platform.python_version(),
              "inputs": {**{INPUTS[n].relative_to(ROOT).as_posix(): d for n, d in digests.items()},
                         "masks/MANIFEST.csv": hashlib.sha256(manifest_bytes).hexdigest()},
              "examinations": len(table), "released_annotations": len(ref_table), "outputs": outputs}
    _publish(args.out / "mask_topology_run_record.json", (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(table.groupby("class26")[["b0", "b1", "b2"]].describe().to_string())


if __name__ == "__main__":
    main()
