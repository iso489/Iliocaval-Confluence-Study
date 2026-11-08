#!/usr/bin/env python3
"""Gap content, spacing-aware separation and path bottleneck for the Phase-1 connectivity audit.

For every measurable examination this measures, on one label source and in physical millimetres:

* the closest pair between the dominant IVC and left iliac components and between the whole
  labels, under the published closed-box metric, with the per-axis decomposition of the gap
  (through-plane, in-plane, oblique or contact), the number of missing planes and the
  separation in units of the examination's own through-plane spacing;
* the corridor through which every near-shortest connection between the two dominant
  components must pass, and the labels that occupy it (background, right iliac vein, venous
  fragments, iliac arteries, aorta, any other class), with the position of that corridor
  relative to the venous confluence;
* for connected examinations, the inscribed radius at the narrowest point of the widest
  connecting route (the bottleneck radius).

Three label sources share one measurement: the archived 2.17.0 native multilabel, the model-grid
experiment's complete internal capture, or the three binary venous exports (which cannot see
arteries). Every native-grid case is bound to the canonical case table before it is measured: voxel
counts, both connectivity classes and the published separation must be reproduced. There is no
published separation on the inference grid to reproduce.

Labels found in a corridor are observations, not causes: an arterial label inside a gap does
not establish that the network would otherwise have called those voxels venous. Nothing here
adjudicates anatomy.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pathlib as _pl
import platform

# --- run-from-anywhere: put sibling phase dirs on sys.path (tests use Code/conftest.py) ---
import sys as _sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _p in sorted(_pl.Path(__file__).resolve().parent.parent.iterdir()):
    if _p.is_dir() and _p.name[0] not in "._" and str(_p) not in _sys.path:
        _sys.path.append(str(_p))  # append, never shadow stdlib

import connectivity_statistics as FS
import measure_collapsed_gap as MC
import model_grid_audit as MG
import nibabel as nib
import numpy as np
import pandas as pd
import statlib as S
import stats as ST
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

SOURCES = ("native", "internal", "venous")
CATEGORIES = ("background", "right_iliac_vein", "ivc_fragment", "lciv_fragment",
              "iliac_artery_left", "iliac_artery_right", "aorta", "other_label")
ARTERIAL = ("iliac_artery_left", "iliac_artery_right", "aorta")
ORIENTATIONS = ("contact", "through_plane", "in_plane", "oblique", "mixed")
DOMINANCE_LEVELS = (1.0, 0.9, 0.5)
THROUGH_PLANE_AXIS = 2            # native NIfTI index axis 2, as in the grid-sampling experiment
SLICE_TABLE_ROUNDING_MM = 0.0005  # half a unit in the canonical table's third decimal of slice_mm
SLACK_VOXEL_DIAGONALS = 1.0       # corridor slack, in units of the voxel diagonal
GAP_ROUNDING_MM = 0.0051          # the canonical table carries two decimals
VENOUS_INDEX = {name: code for code, name in enumerate(MC.REFERENCE_VEIN_FILES, 1)}
VENOUS_KEYS = ("ivc", "lciv", "rciv")
WORKER_PEAK_BYTES = 6 * 1024**3   # conservative: one whole-body uint8 volume, three masks, EDT crops
OUTPUT_NAMES = ("gap_content_cases.csv", "gap_content_summary.csv", "gap_content_report.md")
STATUS = "completed_gap_content_measurement_not_anatomical_adjudication"
REQUIRED_COLUMNS = ("cohort", "case", "mask_case", "participant", "class26", "class6", "gap_mm",
                    "slice_mm", "ivc_voxels", "lciv_voxels", "rciv_voxels")


# --- hashing and reading: one read per file, the digest is of the bytes parsed --------------

def read_hashed(path):
    data = Path(path).read_bytes()
    return data, hashlib.sha256(data).hexdigest()


def nifti_from_bytes(data, name):
    raw = gzip.decompress(data) if str(name).endswith(".gz") else data
    return nib.Nifti1Image.from_bytes(raw)


def read_hashed_nifti(path):
    data, digest = read_hashed(path)
    return nifti_from_bytes(data, Path(path).name), digest


def spacing_from_affine(affine):
    affine = np.asarray(affine, float)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("affine must be a finite 4x4 matrix")
    axes = affine[:3, :3]
    spacing = np.linalg.norm(axes, axis=0)
    if (spacing <= 0).any() or not np.allclose((axes / spacing).T @ (axes / spacing), np.eye(3), atol=1e-5):
        raise ValueError("label grid is not orthogonal; index-scaled distances would be wrong")
    return spacing


# --- geometry primitives ---------------------------------------------------------------------

def dominant_components(ivc, lciv):
    a, av, _ = MC.largest_component26(np.asarray(ivc, bool))
    b, bv, _ = MC.largest_component26(np.asarray(lciv, bool))
    if not av or not bv:
        raise ValueError("empty primary mask in measurable frame")
    return a, b


def _surface(mask):
    """Voxels with a face neighbour outside the mask or on the array edge."""
    return mask & ~ndi.binary_erosion(mask, ndi.generate_binary_structure(3, 1), border_value=0)


def closest_pairs(a, b, spacing):
    """Closest pairs between two voxel sets under the published closed-box metric.

    Every box minimizer has a centre distance of at most d_center + |spacing|, because the box
    distance of a pair is never more than one voxel diagonal below its centre distance, so the
    exact box evaluation over that ball finds all of them. Surface voxels suffice for both
    minima: stepping an interior voxel one face toward its partner never increases either
    distance, and the last voxel before leaving the set is a surface voxel. At zero box distance
    (contact) an interior voxel can tie with the surface voxels, so the minimizer count there is a
    lower bound; the corridor is empty at contact by construction.
    """
    spacing = np.asarray(spacing, float)
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    if a.shape != b.shape or (a & b).any():
        raise ValueError("closest pairs need two disjoint masks on one grid")
    pa = np.argwhere(_surface(a))
    pb = np.argwhere(_surface(b))
    if not len(pa) or not len(pb):
        raise ValueError("closest pairs need two nonempty masks")
    tree = cKDTree(pa * spacing)
    nearest, _ = tree.query(pb * spacing)
    d_center = float(nearest.min())
    radius = d_center + float(np.linalg.norm(spacing)) + 1e-9
    neighbours = tree.query_ball_point(pb * spacing, r=radius)
    j_index = np.repeat(np.arange(len(pb)), [len(x) for x in neighbours])
    i_index = np.fromiter((i for x in neighbours for i in x), int, count=len(j_index))
    delta = np.abs(pa[i_index] - pb[j_index])
    gaps = np.maximum(delta - 1, 0)
    box = np.sqrt(((gaps * spacing) ** 2).sum(axis=1))
    d_box = float(box.min())
    keep = box <= d_box + 1e-9
    i_index, j_index, gaps = i_index[keep], j_index[keep], gaps[keep]
    centers = np.sqrt((((pa[i_index] - pb[j_index]) * spacing) ** 2).sum(axis=1))
    order = np.lexsort((i_index, j_index, np.round(centers, 9)))
    i_index, j_index, gaps, centers = i_index[order], j_index[order], gaps[order], centers[order]
    minimizers = np.stack([pa[i_index], pb[j_index]], axis=1)
    representative = minimizers[0]
    return {"d_center": d_center, "d_box": d_box, "minimizers": minimizers,
            "axis_gap_voxels": gaps.astype(int), "n_minimizers": len(minimizers),
            "representative": representative,
            "displacement_index": (representative[1] - representative[0]).astype(int),
            "d_center_representative": float(centers[0])}


def orientation_of(axis_gap_voxels):
    gaps = np.atleast_2d(np.asarray(axis_gap_voxels, int))
    through = gaps[:, THROUGH_PLANE_AXIS] > 0
    in_plane = (gaps[:, [k for k in range(3) if k != THROUGH_PLANE_AXIS]] > 0).any(axis=1)
    out = np.where(through & in_plane, "oblique",
                   np.where(through, "through_plane", np.where(in_plane, "in_plane", "contact")))
    return [str(x) for x in out]


def orientation_summary(axis_gap_voxels):
    gaps = np.atleast_2d(np.asarray(axis_gap_voxels, int))
    classes = orientation_of(gaps)
    counts = {name: classes.count(name) for name in ORIENTATIONS[:-1] if classes.count(name)}
    return {"orientation": classes[0] if len(counts) == 1 else "mixed", "counts": counts,
            "missing_planes": int(gaps[:, THROUGH_PLANE_AXIS].min())}


def corridor(a, b, spacing, slack_mm=None):
    """The lens of voxels whose summed centre distances to A and to B stay within d_center + slack.

    Every near-shortest path between the components passes through it. Touching components
    (26-adjacent, i.e. zero box separation) have no corridor.
    """
    spacing = np.asarray(spacing, float)
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    if slack_mm is None:
        slack_mm = SLACK_VOXEL_DIAGONALS * float(np.linalg.norm(spacing))
    distance_a = ndi.distance_transform_edt(~a, sampling=spacing)
    d_center = float(distance_a[b].min())
    if (ndi.binary_dilation(a, np.ones((3, 3, 3), bool)) & b).any():
        return np.zeros(a.shape, bool), d_center
    distance_b = ndi.distance_transform_edt(~b, sampling=spacing)
    lens = (distance_a + distance_b <= d_center + slack_mm + 1e-9) & ~a & ~b
    return lens, d_center


def segment_corridor(minimizers, a, b, samples_per_voxel=2):
    """Voxels whose boxes a straight segment between a minimizing pair crosses with positive length.

    Each minimizing pair contributes the segment between its voxel centres. Candidate voxels are
    the 3x3x3 neighbourhoods of the voxels nearest to midpoint samples taken at least twice per
    voxel along the longest axis, so the segment cannot leave the candidate tube between two
    samples; every candidate is then tested exactly against its open box (slab test), and a voxel
    joins only when the segment runs a positive length inside it. A box the segment merely touches
    at a face, edge or corner is excluded, so two voxels in contact enclose their whole segment and
    leave no corridor. The set depends only on the geometry, not on the direction of traversal.
    The endpoint components are excluded.
    """
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    minimizers = np.asarray(minimizers, float)
    mask = np.zeros(a.shape, bool)
    if not len(minimizers):
        return mask
    shape = np.asarray(a.shape)
    start, stop = minimizers[:, 0, :], minimizers[:, 1, :]
    delta = stop - start
    steps = int(np.ceil(samples_per_voxel * np.abs(delta).max())) + 1
    offsets = np.array(list(np.ndindex(3, 3, 3))) - 1
    eps = 1e-9
    for m in range(steps):
        points = start + ((m + 0.5) / steps) * delta
        nearest = np.floor(points + 0.5).astype(int)
        candidates = (nearest[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
        seg_start = np.repeat(start, len(offsets), axis=0)
        seg_delta = np.repeat(delta, len(offsets), axis=0)
        inside = ((candidates >= 0) & (candidates < shape)).all(axis=1)
        candidates, seg_start, seg_delta = candidates[inside], seg_start[inside], seg_delta[inside]
        if not len(candidates):
            continue
        low, high = candidates - 0.5, candidates + 0.5
        moving = seg_delta != 0
        safe = np.where(moving, seg_delta, 1.0)
        t1, t2 = (low - seg_start) / safe, (high - seg_start) / safe
        between = (seg_start > low) & (seg_start < high)
        t_low = np.where(moving, np.minimum(t1, t2), np.where(between, -np.inf, np.inf))
        t_high = np.where(moving, np.maximum(t1, t2), np.where(between, np.inf, -np.inf))
        t_enter = np.maximum(t_low.max(axis=1), 0.0)
        t_exit = np.minimum(t_high.min(axis=1), 1.0)
        hit = candidates[t_exit - t_enter > eps]
        mask[hit[:, 0], hit[:, 1], hit[:, 2]] = True
    return mask & ~a & ~b

def category_of(name):
    if name is None:
        return "background"
    if name == "iliac_vena_right":
        return "right_iliac_vein"
    if name == "inferior_vena_cava":
        return "ivc_fragment"
    if name == "iliac_vena_left":
        return "lciv_fragment"
    if name in ARTERIAL:
        return name
    return "other_label"


def corridor_content(corridor_mask, volume, index, a, b):
    """Tabulate the labels inside a corridor; venous voxels there are fragments by construction."""
    inverse = {int(v): k for k, v in index.items()}
    values = np.asarray(volume)[np.asarray(corridor_mask, bool) & ~np.asarray(a, bool) & ~np.asarray(b, bool)]
    n = len(values)
    counts = np.bincount(values.astype(np.int64), minlength=1) if n else np.zeros(1, int)
    foreign = [label for label in np.flatnonzero(counts) if label and label not in inverse]
    if foreign:
        raise ValueError(f"corridor carries labels outside the class map: {foreign}")
    counts_by_name = {inverse[label]: int(counts[label]) for label in np.flatnonzero(counts) if label}
    categories = {name: 0 for name in CATEGORIES}
    categories["background"] = int(counts[0]) if len(counts) else 0
    other = {}
    for name, count in counts_by_name.items():
        category = category_of(name)
        categories[category] += count
        if category == "other_label":
            other[name] = count
    if n:
        dominant = max(CATEGORIES, key=lambda c: (categories[c], -CATEGORIES.index(c)))
        fraction = categories[dominant] / n
    else:
        dominant, fraction = "", float("nan")
    return {"n_voxels": n, "counts_by_name": counts_by_name,
            "category_counts": {k: v for k, v in categories.items() if v},
            "dominant_category": dominant, "dominant_fraction": fraction,
            "pure": bool(n and fraction == 1.0),
            "arterial_voxels": int(sum(categories[name] for name in ARTERIAL)),
            "other_labels": json.dumps(dict(sorted(other.items())))}


def bottleneck_radius(a, b, union, spacing, connectivity=26):
    """Largest r such that A and B stay connected inside the union restricted to inside-distance >= r.

    The inside distance of a union voxel is the distance from its centre to the nearest centre
    outside the union (the array border counts as outside). Bisection over the distinct values
    uses one labelling per step; NaN when the components are not connected at all.
    """
    if connectivity not in (26, 6):
        raise ValueError("connectivity must be 26 or 6")
    structure = ndi.generate_binary_structure(3, 3 if connectivity == 26 else 1)
    spacing = np.asarray(spacing, float)
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    union = np.asarray(union, bool)
    if not (a <= union).all() or not (b <= union).all():
        raise ValueError("the endpoints must lie inside the union")
    padded = np.pad(union, 1)
    inside = ndi.distance_transform_edt(padded, sampling=spacing)[1:-1, 1:-1, 1:-1]
    levels = np.unique(inside[union])

    def connected(level):
        labels, _ = ndi.label(union & (inside >= level - 1e-9), structure)
        return MC.share_component(labels, a, b)

    if not len(levels) or not connected(levels[0]):
        return float("nan")
    low, high = 0, len(levels) - 1
    while low < high:
        middle = (low + high + 1) // 2
        if connected(levels[middle]):
            low = middle
        else:
            high = middle - 1
    return float(levels[low])


# --- per-case measurement ----------------------------------------------------------------------

def _bounding_slices(mask, pad, shape):
    support = [np.flatnonzero(mask.any(axis=tuple(j for j in range(3) if j != i))) for i in range(3)]
    return tuple(slice(max(int(x[0]) - p, 0), min(int(x[-1]) + 1 + p, n))
                 for x, p, n in zip(support, pad, shape))


def _ras(affine, indices):
    return np.asarray(indices, float) @ affine[:3, :3].T + affine[:3, 3]


def measure_case(volume, index, affine, row, label_scope="total", binding=None):
    """One examination: bind to the canonical row (or, for an internal capture, to its own model-grid
    record passed as `binding`), then measure every quantity of the design."""
    started = time.monotonic()
    affine = np.asarray(affine, float)
    spacing = spacing_from_affine(affine)
    case_name = f"{row['cohort']}/{row['case']}"
    if binding is None:
        # The canonical table carries slice_mm to at most three decimals (SAROS/case_137 reads 1.991
        # for a header spacing of 1.9907 mm), so the header may sit up to half a unit in the third
        # decimal from the table value; the next unit is refused.
        if abs(spacing[THROUGH_PLANE_AXIS] - float(row["slice_mm"])) > SLICE_TABLE_ROUNDING_MM + 1e-9:
            raise ValueError(f"slice spacing disagrees with the canonical table for {case_name}: "
                             f"axis-2 spacing {spacing[THROUGH_PLANE_AXIS]:.4f} mm versus {row['slice_mm']}")
        expected_counts = {key: int(row[f"{key}_voxels"]) for key in VENOUS_KEYS}
        expected_class = {conn: row[f"class{conn}"] for conn in (26, 6)}
        record_name = "canonical table"
    else:
        # An internal capture: the inference grid, bound to the frozen model-grid record of this case.
        # The record stores the header spacing exactly, so the comparison is exact to rounding.
        recorded = np.asarray(binding["spacing_mm"], float)
        if recorded.shape != (3,) or not np.allclose(spacing, recorded, atol=1e-6, rtol=0):
            raise ValueError(f"grid spacing disagrees with the model-grid record for {case_name}: "
                             f"header {np.round(spacing, 4).tolist()} versus recorded {recorded.tolist()}")
        expected_counts = dict(zip(VENOUS_KEYS, (int(x) for x in binding["voxels"])))
        expected_class = {conn: binding[f"class{conn}"] for conn in (26, 6)}
        record_name = "model-grid record"
    volume = np.asanyarray(volume)
    if not np.issubdtype(volume.dtype, np.integer) or volume.ndim != 3 or (volume < 0).any():
        raise ValueError("label volume must be a 3-D non-negative integer array")
    try:
        ids = [int(index[name]) for name in MC.REFERENCE_VEIN_FILES]
    except KeyError as exc:
        raise ValueError(f"class map lacks a venous label: {exc}") from exc
    masks = [volume == label for label in ids]
    for key, mask in zip(VENOUS_KEYS, masks):
        if int(mask.sum()) != expected_counts[key]:
            raise ValueError(f"voxel count disagrees with the {record_name}: {key} in {case_name}")
    if not masks[0].any() or not masks[1].any():
        raise ValueError("empty primary mask in measurable frame")
    union = masks[0] | masks[1] | masks[2]
    tight = _bounding_slices(union, (0, 0, 0), volume.shape)
    crops = [m[tight] for m in masks]
    topology = MC.classify_topology(crops)
    for conn in (26, 6):
        if topology[f"class{conn}"] != expected_class[conn]:
            if binding is None:
                raise ValueError(f"canonical class{conn} not reproduced for {case_name}")
            raise ValueError(f"class{conn} disagrees with the model-grid record for {case_name}: "
                             f"measured {topology[f'class{conn}']}, recorded {expected_class[conn]}")
    whole = closest_pairs(crops[0], crops[1], spacing)
    # The published separation is a native-grid fact; on the inference grid there is none to reproduce.
    if binding is None and abs(whole["d_box"] - float(row["gap_mm"])) > GAP_ROUNDING_MM:
        raise ValueError(f"published separation not reproduced for {row['cohort']}/{row['case']}: "
                         f"{whole['d_box']:.4f} versus {row['gap_mm']}")
    a, b = dominant_components(crops[0], crops[1])
    dominant = closest_pairs(a, b, spacing)
    whole_summary = orientation_summary(whole["axis_gap_voxels"])
    dominant_summary = orientation_summary(dominant["axis_gap_voxels"])
    # The lens is widest halfway along the gap; pad the crop so it is never clipped.
    slack = SLACK_VOXEL_DIAGONALS * float(np.linalg.norm(spacing))
    half_width = float(np.sqrt(slack * dominant["d_center"] / 2 + slack ** 2 / 4))
    pad = tuple(int(np.ceil((half_width + slack) / s)) + 1 for s in spacing)
    box = _bounding_slices(union, pad, volume.shape)
    shift = np.array([t.start - w.start for t, w in zip(tight, box)])
    crops = [m[box] for m in masks]
    a_box = np.zeros(crops[0].shape, bool)
    a_box[tuple(slice(s, s + n) for s, n in zip(shift, a.shape))] = a
    b_box = np.zeros(crops[0].shape, bool)
    b_box[tuple(slice(s, s + n) for s, n in zip(shift, b.shape))] = b
    a, b = a_box, b_box
    crop_volume = volume[box]
    corridor_mask = (segment_corridor(dominant["minimizers"] + shift, a, b) if dominant["d_box"] > 0
                     else np.zeros(a.shape, bool))
    content = corridor_content(corridor_mask, crop_volume, index, a, b)
    lens, d_center = corridor(a, b, spacing, slack)
    if abs(d_center - dominant["d_center"]) > 1e-6:
        raise AssertionError("distance transform and KD-tree disagree on the centre distance")
    lens_content = corridor_content(lens, crop_volume, index, a, b)
    offset = np.array([s.start for s in box])
    voxel_mm3 = float(np.prod(spacing))
    displacement_mm = dominant["displacement_index"] * spacing
    displacement_ras = affine[:3, :3] @ dominant["displacement_index"]
    norm = float(np.linalg.norm(displacement_mm))
    ras_a = _ras(affine, np.argwhere(a) + offset)
    ras_b = _ras(affine, np.argwhere(b) + offset)
    ivc_caudal = float(ras_a[:, 2].min())
    lciv_cranial = float(ras_b[:, 2].max())
    caudal_plane = ras_a[ras_a[:, 2] <= ivc_caudal + spacing.min() / 2]
    rciv_cranial = float("nan")
    if crops[2].any():
        rciv_dominant, _, _ = MC.largest_component26(crops[2])
        rciv_cranial = float(_ras(affine, np.argwhere(rciv_dominant) + offset)[:, 2].max())
    centroid = np.full(3, np.nan)
    inplane_offset = float("nan")
    if content["n_voxels"]:
        centroid = _ras(affine, np.argwhere(corridor_mask) + offset).mean(axis=0)
        inplane_offset = float(np.hypot(*(centroid[:2] - caudal_plane[:, :2].mean(axis=0))))
    arterial_in_corridor, nearest_label, nearest_distance = float("nan"), "", float("nan")
    if label_scope == "total":
        arterial_in_corridor = float(content["arterial_voxels"] > 0)
        if content["n_voxels"]:
            points = (np.argwhere(corridor_mask) + offset) * spacing
            for name in ARTERIAL:
                label = index.get(name)
                if label is None:
                    continue
                artery = np.argwhere(volume == label)
                if not len(artery):
                    continue
                distance = float(cKDTree(artery * spacing).query(points)[0].min())
                if not (distance >= nearest_distance):
                    nearest_label, nearest_distance = name, distance
    radius = float("nan")
    if expected_class[26] != "disconnected":
        radius = bottleneck_radius(a, b, crops[0] | crops[1] | crops[2], spacing)
    result = {key: row[key] for key in ("cohort", "case", "mask_case", "participant", "class26", "class6",
                                         "gap_mm", "slice_mm")}
    if binding is not None:
        # The internal classes select the subsets; the native classes travel with the row; gap_mm and
        # slice_mm stay the canonical native values (the strata are native slice spacing on every source).
        result.update({"class26": expected_class[26], "class6": expected_class[6],
                       "native_class26": row["class26"], "native_class6": row["class6"], "grid": "internal"})
    result.update({
        "label_scope": label_scope,
        "spacing_mm_0": float(spacing[0]), "spacing_mm_1": float(spacing[1]), "spacing_mm_2": float(spacing[2]),
        "ivc_voxels": int(masks[0].sum()), "lciv_voxels": int(masks[1].sum()), "rciv_voxels": int(masks[2].sum()),
        "ivc_dominant_voxels": int(a.sum()), "lciv_dominant_voxels": int(b.sum()),
        "whole_d_box_mm": whole["d_box"], "whole_d_center_mm": whole["d_center"],
        "whole_orientation": whole_summary["orientation"], "whole_orientation_counts": json.dumps(whole_summary["counts"]),
        "whole_missing_planes": whole_summary["missing_planes"], "whole_n_minimizers": whole["n_minimizers"],
        "whole_axis_gap_mm_0": float(whole["axis_gap_voxels"][0, 0] * spacing[0]),
        "whole_axis_gap_mm_1": float(whole["axis_gap_voxels"][0, 1] * spacing[1]),
        "whole_axis_gap_mm_2": float(whole["axis_gap_voxels"][0, 2] * spacing[2]),
        "whole_separation_in_slice_units": whole["d_box"] / float(spacing[THROUGH_PLANE_AXIS]),
        "dominant_d_box_mm": dominant["d_box"], "dominant_d_center_mm": dominant["d_center"],
        "dominant_orientation": dominant_summary["orientation"],
        "dominant_orientation_counts": json.dumps(dominant_summary["counts"]),
        "dominant_missing_planes": dominant_summary["missing_planes"], "dominant_n_minimizers": dominant["n_minimizers"],
        "displacement_index_0": int(dominant["displacement_index"][0]),
        "displacement_index_1": int(dominant["displacement_index"][1]),
        "displacement_index_2": int(dominant["displacement_index"][2]),
        "displacement_mm_x": float(displacement_ras[0]), "displacement_mm_y": float(displacement_ras[1]),
        "displacement_mm_z": float(displacement_ras[2]),
        "displacement_through_plane_fraction": abs(float(displacement_mm[THROUGH_PLANE_AXIS])) / norm if norm else float("nan"),
        "corridor_voxels": content["n_voxels"], "corridor_volume_mm3": content["n_voxels"] * voxel_mm3,
        **{f"corridor_{name}_voxels": int(content["category_counts"].get(name, 0)) for name in CATEGORIES},
        "corridor_other_labels": content["other_labels"],
        "corridor_dominant_category": content["dominant_category"],
        "corridor_dominant_fraction": content["dominant_fraction"], "corridor_pure": int(content["pure"]),
        "arterial_in_corridor": arterial_in_corridor, "nearest_arterial_label": nearest_label,
        "nearest_arterial_distance_mm": nearest_distance,
        "lens_slack_mm": slack, "lens_voxels": lens_content["n_voxels"],
        "lens_volume_mm3": lens_content["n_voxels"] * voxel_mm3,
        "lens_dominant_category": lens_content["dominant_category"],
        "lens_dominant_fraction": lens_content["dominant_fraction"],
        "lens_arterial_voxels": lens_content["arterial_voxels"],
        "lens_right_iliac_vein_voxels": int(lens_content["category_counts"].get("right_iliac_vein", 0)),
        "lens_ivc_fragment_voxels": int(lens_content["category_counts"].get("ivc_fragment", 0)),
        "lens_lciv_fragment_voxels": int(lens_content["category_counts"].get("lciv_fragment", 0)),
        "ivc_caudal_z_mm": ivc_caudal, "lciv_cranial_z_mm": lciv_cranial, "rciv_cranial_z_mm": rciv_cranial,
        "si_overlap_mm": lciv_cranial - ivc_caudal,
        "corridor_centroid_z_minus_ivc_caudal_mm": float(centroid[2] - ivc_caudal),
        "corridor_centroid_z_minus_rciv_cranial_mm": float(centroid[2] - rciv_cranial),
        "corridor_inplane_offset_mm": inplane_offset,
        "bottleneck_radius_mm": radius, "bottleneck_radius_min_spacing_units": radius / float(spacing.min()),
        "elapsed_seconds": time.monotonic() - started})
    return result

# --- label sources --------------------------------------------------------------------------------

def load_native(folder):
    """The archived 2.17.0 multilabel and its report; the report is validated by the exporter."""
    folder = Path(folder)
    report_bytes, report_digest = read_hashed(folder / "report.json")
    index = FS.EM.validate_baseline_report(json.loads(report_bytes), MC.REFERENCE_VEIN_FILES)
    image, digest = read_hashed_nifti(folder / "segmentation.nii.gz")
    if not np.issubdtype(image.get_data_dtype(), np.integer):
        raise ValueError("multilabel volume must have integer storage")
    return {"image": image, "index": index, "label_scope": "total",
            "source_hashes": {"segmentation.nii.gz": digest, "report.json": report_digest}}


def load_venous(mask_root, row):
    """The three hash-bound binary exports, in either the flat or the full-resolution layout."""
    folder = Path(mask_root) / row["cohort"] / row["mask_case"]
    arrays, affine, hashes = [], None, {}
    for name in MC.REFERENCE_VEIN_FILES:
        candidates = [folder / f"{name}.nii.gz", folder / "segmentations_fullres" / f"{name}.nii.gz"]
        present = [p for p in candidates if p.is_file()]
        if not present:
            raise FileNotFoundError(f"missing venous mask; tried {candidates}")
        image, digest = read_hashed_nifti(present[0])
        if digest != row[f"{name}_sha256"]:
            raise ValueError(f"mask hash mismatch: {row['cohort']}/{row['case']}/{name}")
        data = np.asanyarray(image.dataobj)
        if data.min() < 0 or data.max() > 1:
            raise ValueError("venous export is not binary")
        if affine is not None and (data.shape != arrays[0].shape or not np.allclose(image.affine, affine, atol=1e-5)):
            raise ValueError(f"grid mismatch: {row['cohort']}/{row['case']}")
        affine = image.affine
        arrays.append(data.astype(bool))
        hashes[f"{name}.nii.gz"] = digest
    compact = np.zeros(arrays[0].shape, np.uint8)
    for code, mask in enumerate(arrays, 1):
        compact[mask] = code
    return {"image": nib.Nifti1Image(compact, affine), "index": dict(VENOUS_INDEX),
            "label_scope": "venous_only", "source_hashes": hashes}


def load_internal(folder):
    """The model-grid experiment's complete internal capture for one passed case."""
    folder = Path(folder)
    qc_bytes, qc_digest = read_hashed(folder / "qc.json")
    qc = json.loads(qc_bytes)
    if qc.get("status") not in ("pass", "mismatch"):
        raise ValueError(f"internal capture is only read from a complete model-grid case (qc.json pass or mismatch): {folder}")
    outputs = qc.get("outputs", {})
    hashes = {"qc.json": qc_digest}
    payload = {}
    for name in (MG.INTERNAL_MULTILABEL, MG.INTERNAL_LABELS):
        data, digest = read_hashed(folder / name)
        if outputs.get(name) != digest:
            raise ValueError(f"internal output hash mismatch: {folder / name}")
        payload[name], hashes[name] = data, digest
    sidecar = json.loads(payload[MG.INTERNAL_LABELS])
    index = {name: int(i) for i, name in sidecar["total_class_map"].items()}
    image = nifti_from_bytes(payload[MG.INTERNAL_MULTILABEL], MG.INTERNAL_MULTILABEL)
    # The record's own account of the internal grid and its venous topology is what the volume is
    # bound to (the canonical table describes the native grid). run_case writes both at capture.
    geometry, topology = qc.get("internal_geometry"), qc.get("internal_topology")
    if (not isinstance(geometry, dict) or not isinstance(topology, dict) or "spacing_mm" not in geometry
            or not {"class26", "class6", "voxels"} <= set(topology)):
        raise ValueError(f"internal record lacks its geometry or topology: {folder / 'qc.json'}")
    binding = {"spacing_mm": [float(x) for x in geometry["spacing_mm"]], "voxels": [int(x) for x in topology["voxels"]],
               "class26": topology["class26"], "class6": topology["class6"]}
    return {"image": image, "index": index, "label_scope": "total", "source_hashes": hashes, "binding": binding}


def load_source(source, root, row):
    if source == "native":
        return load_native(Path(root) / row["cohort"] / row["mask_case"])
    if source == "venous":
        return load_venous(root, row)
    if source == "internal":
        return load_internal(MG.case_folder(root, row))
    raise ValueError(f"unknown label source {source!r}")


# --- the run --------------------------------------------------------------------------------------

def _frame(case_table, source, pilot_cases=None):
    data = pd.read_csv(case_table)
    required = set(REQUIRED_COLUMNS)
    if source == "venous":
        required |= {f"{name}_sha256" for name in MC.REFERENCE_VEIN_FILES}
    missing = sorted(required - set(data.columns))
    if missing or data[sorted(required)].isna().any().any():
        raise ValueError(f"case table lacks the canonical schema: missing {missing}")
    if data.duplicated(["cohort", "case"]).any():
        raise ValueError("duplicate case in the case table")
    for column in ("cohort", "case", "mask_case"):
        for value in data[column]:
            MG.component(value)
    if not data.class26.isin(MC.TOPOLOGY_CLASSES).all() or not data.class6.isin(MC.TOPOLOGY_CLASSES).all():
        raise ValueError("unknown canonical topology class")
    data = data.sort_values(["cohort", "case"]).reset_index(drop=True)
    if pilot_cases is not None:
        if pilot_cases < 1:
            raise ValueError("pilot cases per cohort must be positive")
        data = data.groupby("cohort", sort=True).head(pilot_cases).reset_index(drop=True)
    return data


def _code_hashes():
    import gap_config
    files = [Path(__file__), Path(MC.__file__), Path(MG.__file__), Path(FS.__file__), Path(ST.__file__),
             Path(FS.EM.__file__), Path(__file__).with_name("measure_ivc_lciv_gap.py"),
             Path(S.__file__), Path(gap_config.__file__)]
    return {str(p.resolve().relative_to(ST.REPO_ROOT.resolve())): read_hashed(p)[1] for p in files}


def _worker(task):
    row, source, root, fingerprint = task
    loaded = load_source(source, root, row)
    affine = loaded["image"].affine
    result = measure_case(np.asanyarray(loaded["image"].dataobj), loaded["index"], affine, row, loaded["label_scope"],
                          binding=loaded.get("binding"))
    result["source_hashes"] = json.dumps(loaded["source_hashes"], sort_keys=True)
    return {"fingerprint": fingerprint, "row": result}


def _case_file(cases_dir, row):
    folder = MG.case_folder(cases_dir, row)
    return folder.with_name(folder.name + ".json")   # not with_suffix: case ids may contain dots


def _atomic_text(path, text):
    pending = path.with_name(path.name + ".partial")
    pending.write_text(text)
    pending.replace(path)


IDENTITY_FIRST = ("cohort", "case", "mask_case", "participant", "class26", "class6", "gap_mm", "slice_mm",
                  "label_scope")
#: Present only on internal-grid rows, right after the identity columns; native rows do not carry them.
INTERNAL_IDENTITY = ("grid", "native_class26", "native_class6")


def _cases_frame(rows_by_key, rows):
    """One column order whether the rows come from memory or from the checkpoint files."""
    frame = pd.DataFrame([rows_by_key[(row["cohort"], row["case"])] for row in rows])
    leading = [c for c in IDENTITY_FIRST + INTERNAL_IDENTITY if c in frame.columns]
    ordered = leading + sorted(c for c in frame.columns if c not in leading)
    return frame[ordered]


def _write_frame(path, frame):
    pending = path.with_name(path.name + ".partial")
    frame.to_csv(pending, index=False, float_format="%.10g")
    pending.replace(path)


def run(case_table, source, root, outdir, workers=1, validate_only=False, pilot_cases=None,
        n_boot=10000, seed=20261025):
    from multiprocessing import get_context

    import hpc_resources as HR
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}")
    if workers < 1:
        raise ValueError("positive workers required")
    case_table, root, outdir = Path(case_table), Path(root), Path(outdir)
    if outdir.resolve() == case_table.resolve().parent or outdir.resolve() == root.resolve():
        raise ValueError("outputs must be separate from the canonical tables and the label source")
    frame = _frame(case_table, source, pilot_cases)
    if source == "internal":
        MG.load_contract(root)
    identity = {"schema": 1, "source": source, "case_table_sha256": read_hashed(case_table)[1],
                "code": _code_hashes(), "pilot_cases": pilot_cases,
                "parameters": {"slack_voxel_diagonals": SLACK_VOXEL_DIAGONALS, "through_plane_axis": THROUGH_PLANE_AXIS,
                               "dominance_levels": list(DOMINANCE_LEVELS), "gap_rounding_mm": GAP_ROUNDING_MM,
                               "bootstrap_resamples": n_boot, "seed": seed},
                "cases": len(frame)}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    outdir.mkdir(parents=True, exist_ok=True)
    cases_dir = outdir / "cases"
    rows = frame.to_dict("records")
    done = {}
    for row in rows:
        path = _case_file(cases_dir, row)
        if path.is_file():
            payload = json.loads(path.read_text())
            if payload.get("fingerprint") == fingerprint:
                done[(row["cohort"], row["case"])] = payload["row"]
    pending = [row for row in rows if (row["cohort"], row["case"]) not in done]
    manifest_path = outdir / "gap_content_manifest.json"
    if validate_only:
        if pending or not manifest_path.is_file():
            raise ValueError("incomplete experiment cannot be validated")
    else:
        memory = HR.available_memory_bytes()
        cap = max(1, (memory - 4 * 1024**3) // WORKER_PEAK_BYTES) if memory else HR.UNKNOWN_MEMORY_WORKER_CAP
        resolved = max(1, min(workers, HR.visible_cpu_count(), cap, max(len(pending), 1)))
        print(f"Gap content: {len(frame)} cases ({len(done)} checkpointed, {len(pending)} pending); "
              f"source={source}; workers requested={workers}, resolved={resolved}", flush=True)
        tasks = [(row, source, str(root), fingerprint) for row in pending]
        if resolved == 1:
            results = map(_worker, tasks)
        else:
            HR.limit_blas_threads(1)
            pool = ProcessPoolExecutor(max_workers=resolved, mp_context=get_context("spawn"),
                                       initializer=HR.limit_blas_threads, initargs=(1,))
            results = pool.map(_worker, tasks)
        for row, payload in zip(pending, results):
            path = _case_file(cases_dir, row)
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_text(path, json.dumps(payload, sort_keys=True, allow_nan=True) + "\n")
            done[(row["cohort"], row["case"])] = payload["row"]
            print("Gap content progress:", len(done), "of", len(frame), "case", row["cohort"], row["case"],
                  "seconds:", round(payload["row"]["elapsed_seconds"], 1), flush=True)
        if resolved > 1:
            pool.shutdown()
    cases = _cases_frame(done, rows)
    summary = summaries(cases, n_boot=n_boot, seed=seed)
    report = render_report(summary, cases, identity)
    frames = {"gap_content_cases.csv": cases, "gap_content_summary.csv": summary}
    if validate_only:
        manifest = json.loads(manifest_path.read_text())
        if manifest["fingerprint"] != fingerprint or manifest["cases"] != len(frame):
            raise ValueError("result manifest identity mismatch")
        recorded = {item["path"]: item["sha256"] for item in manifest["outputs"]}
        if set(recorded) != set(OUTPUT_NAMES):
            raise ValueError("manifest output inventory is incomplete or foreign")
        for name, digest in recorded.items():
            if read_hashed(outdir / name)[1] != digest:
                raise ValueError(f"output hash mismatch: {name}")
        for name, expected in frames.items():
            # An empty label string round-trips through CSV as a missing value.
            comparable = expected.reset_index(drop=True).replace({"": np.nan})
            pd.testing.assert_frame_equal(pd.read_csv(outdir / name), comparable,
                                          check_dtype=False, rtol=1e-8, atol=1e-9)
        if (outdir / "gap_content_report.md").read_text() != report:
            raise ValueError("report text does not re-derive from the case files")
        print("Gap content validation MATCH (fingerprint, case files, output hashes, re-derived summaries)", flush=True)
        return
    manifest_path.unlink(missing_ok=True)
    for name, data in frames.items():
        _write_frame(outdir / name, data)
    _atomic_text(outdir / "gap_content_report.md", report)
    manifest = {**identity, "fingerprint": fingerprint, "status": STATUS,
                "label_scope": sorted(set(cases.label_scope)),
                "python": platform.python_version(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "outputs": [{"path": name, "sha256": read_hashed(outdir / name)[1]} for name in OUTPUT_NAMES]}
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Gap content completed: {len(cases)} cases -> {outdir}", flush=True)


# --- summaries and report ----------------------------------------------------------------------

def _spacing_strata(cases):
    return {"spacing_le_1p5mm": cases[cases.slice_mm <= 1.5],
            "spacing_gt1p5_to3mm": cases[(cases.slice_mm > 1.5) & (cases.slice_mm <= 3)],
            "spacing_gt3mm": cases[cases.slice_mm > 3]}


def summaries(cases, n_boot=10000, seed=20261025):
    """Stratified proportions and medians with the project's participant-aware intervals."""
    if cases.empty or cases.duplicated(["cohort", "case"]).any():
        raise ValueError("case frame is empty or duplicated")
    strata = {"all": cases, **{c: d for c, d in cases.groupby("cohort", sort=True)}, **_spacing_strata(cases)}
    rows = []

    def proportion(stratum, subset, frame, outcome, selector):
        stats = FS.proportion_intervals(frame, np.asarray(selector, bool), n_boot, True, seed)
        rows.append({"stratum": stratum, "subset": subset, "outcome": outcome, "n": stats["n"],
                     "events": stats["events"], "estimate": stats["proportion"], "proportion": stats["proportion"],
                     "ci_lo": stats["ci_lo"], "ci_hi": stats["ci_hi"], "participants": stats["participants"],
                     "ci_method": stats["ci_method"]})

    def median(stratum, subset, frame, column):
        values = frame[column].to_numpy(float)
        lo, hi, method = (FS.median_interval(values, frame.participant.tolist(), n_boot=n_boot, seed=seed)
                          if len(values) else (np.nan, np.nan, "not_estimable"))
        rows.append({"stratum": stratum, "subset": subset, "outcome": f"median:{column}", "n": len(frame),
                     "events": np.nan, "estimate": float(np.median(values)) if len(values) else np.nan,
                     "proportion": np.nan, "ci_lo": lo, "ci_hi": hi,
                     "participants": frame.participant.nunique(), "ci_method": method})

    for stratum, frame in strata.items():
        if frame.empty:
            continue
        subsets = {"disconnected": frame[frame.class26 == "disconnected"],
                   "mediated": frame[frame.class26 == "right_iliac_mediated"],
                   "connected": frame[frame.class26 != "disconnected"],
                   "positive_separation": frame[frame.whole_d_box_mm > 0]}
        for subset in ("disconnected", "mediated"):
            part = subsets[subset]
            if part.empty:
                continue
            for name in ORIENTATIONS:
                proportion(stratum, subset, part, f"whole_orientation={name}", part.whole_orientation == name)
            proportion(stratum, subset, part, "exactly_one_missing_plane",
                       (part.whole_orientation == "through_plane") & (part.whole_missing_planes == 1))
            proportion(stratum, subset, part, "whole_separation_in_slice_units<=1", part.whole_separation_in_slice_units <= 1 + 1e-9)
            proportion(stratum, subset, part, "whole_separation_in_slice_units>2", part.whole_separation_in_slice_units > 2 + 1e-9)
            for category in CATEGORIES:
                for level in DOMINANCE_LEVELS:
                    proportion(stratum, subset, part, f"corridor_dominant_category={category};dominance>={level:.1f}",
                               (part.corridor_dominant_category == category) & (part.corridor_dominant_fraction >= level - 1e-12))
            proportion(stratum, subset, part, "corridor_contains_right_iliac_vein", part.corridor_right_iliac_vein_voxels > 0)
            proportion(stratum, subset, part, "corridor_contains_venous_fragment",
                       (part.corridor_ivc_fragment_voxels + part.corridor_lciv_fragment_voxels) > 0)
            known = part[part.arterial_in_corridor.notna()]
            if not known.empty:
                proportion(stratum, subset, known, "arterial_in_corridor", known.arterial_in_corridor == 1)
                proportion(stratum, subset, known, "lens_arterial_in_corridor", known.lens_arterial_voxels > 0)
        connected = subsets["connected"]
        if not connected.empty:
            minimum = connected[["spacing_mm_0", "spacing_mm_1", "spacing_mm_2"]].min(axis=1)
            proportion(stratum, "connected", connected, "bottleneck_radius<=min_spacing",
                       connected.bottleneck_radius_mm <= minimum + 1e-9)
            proportion(stratum, "connected", connected, "class6=disconnected", connected.class6 == "disconnected")
            median(stratum, "connected", connected, "bottleneck_radius_mm")
        positive = subsets["positive_separation"]
        if not positive.empty:
            median(stratum, "positive_separation", positive, "whole_separation_in_slice_units")
            median(stratum, "positive_separation", positive, "whole_d_box_mm")
    return pd.DataFrame(rows)


def _ci(lo, hi, nd=1):
    return S.format_ci(lo, hi, nd=nd, sep=", ") or "NA"


def render_report(summary, cases, identity):
    lines = ["# Gap content, spacing-aware separation and path bottleneck", "",
             f"Source: {identity['source']}; cases: {len(cases)}; label scope: {', '.join(sorted(set(cases.label_scope)))}.",
             *(["Internal source: class26 and class6 are the classes measured on the internal inference grid, bound to",
                "each case's model-grid record (grid spacing, venous voxel counts, both classes); gap_mm and slice_mm",
                "are the canonical native values, so the spacing strata are native slice spacing, and the published",
                "native separation is not a target on this grid."] if identity.get("source") == "internal" else []),
             "Intervals are pointwise 95% intervals: cohort-stratified participant percentile resamples where",
             "participants repeat or strata pool cohorts, Wilson scores for independent single-cohort strata,",
             "BCa for medians. Labels found in a corridor are observations, not causes: an arterial label",
             "inside a gap does not establish that the network would otherwise have called those voxels venous.", ""]
    pooled = summary[summary.stratum == "all"]
    for subset, title in (("disconnected", "Disconnected examinations"), ("connected", "Connected examinations"),
                          ("positive_separation", "Positive published separation")):
        part = pooled[pooled.subset == subset]
        if part.empty:
            continue
        lines += [f"## {title} (pooled)", "", "| Outcome | n | events | estimate | 95% CI | method |", "|---|---:|---:|---:|---|---|"]
        for row in part.itertuples():
            method = row.ci_method
            if row.outcome.startswith("median:"):
                lines.append(f"| {row.outcome} | {row.n} | | {row.estimate:.3f} | {_ci(row.ci_lo, row.ci_hi)} | {method} |")
            elif row.events or not row.outcome.startswith("corridor_dominant_category="):
                lines.append(f"| {row.outcome} | {row.n} | {int(row.events)} | {100 * row.estimate:.1f}% | {_ci(row.ci_lo, row.ci_hi)} | {method} |")
        lines.append("")
    lines += ["## Per-cohort disconnected orientation", "", "| Cohort | n | through-plane | in-plane | oblique | mixed |", "|---|---:|---:|---:|---:|---:|"]
    for cohort in sorted(set(cases.cohort)):
        part = summary[(summary.stratum == cohort) & (summary.subset == "disconnected")].set_index("outcome")
        if part.empty:
            continue
        cells = [f"{int(part.loc[f'whole_orientation={name}', 'events'])}" for name in ("through_plane", "in_plane", "oblique", "mixed")]
        lines.append(f"| {cohort} | {int(part.iloc[0].n)} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=SOURCES, required=True, help="Which label volume is measured")
    ap.add_argument("--case-table", type=Path, default=FS.OUT / "analysis_cases.csv",
                    help="Canonical case identities, classes, counts and separations")
    ap.add_argument("--multilabel-root", type=Path, help="2.17.0 baseline cohort/mask_case/{segmentation.nii.gz,report.json}")
    ap.add_argument("--model-grid-root", type=Path, help="Completed model-grid experiment root (internal source)")
    ap.add_argument("--mask-root", type=Path, default=ST.REPO_ROOT / "Local masks",
                    help="Hash-bound cohort/mask_case/structure.nii.gz exports (venous source)")
    ap.add_argument("--outdir", type=Path, required=True, help="Dedicated output directory")
    ap.add_argument("--workers", type=int, default=1, help="Requested case workers; capped by the allocation")
    ap.add_argument("--validate-only", action="store_true", help="Re-derive and verify a completed run")
    ap.add_argument("--pilot-cases", type=int, help="First N cases per cohort only (a resource pilot)")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261025)
    args = ap.parse_args(argv)
    roots = {"native": args.multilabel_root, "internal": args.model_grid_root, "venous": args.mask_root}
    root = roots[args.source]
    if root is None:
        ap.error(f"--source {args.source} requires its root argument")
    run(args.case_table, args.source, root, args.outdir, workers=args.workers, validate_only=args.validate_only,
        pilot_cases=args.pilot_cases, n_boot=args.bootstrap, seed=args.seed)


if __name__ == "__main__":
    main()
