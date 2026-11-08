#!/usr/bin/env python3
"""Connectivity classification of the predicted venous labels.

`classify_topology` takes the inferior vena cava (IVC), left iliac and right iliac masks and reports
whether the largest 26-connected IVC and left iliac components are joined by a path through venous
voxels: directly (within the IVC and left iliac labels), only through the right iliac label, or not at
all. `read_topology_case` and `audit_reference_case` apply it to one predicted examination and to one
released annotation case; `largest_component26`, `share_component` and `topology_digest` are the
helpers they share with the other measurement modules.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# --- run-from-anywhere: put sibling phase dirs on sys.path (tests use Code/conftest.py) ---
import sys as _sys, pathlib as _pl
for _p in sorted(_pl.Path(__file__).resolve().parent.parent.iterdir()):
    if _p.is_dir() and _p.name[0] not in "._":
        if str(_p) not in _sys.path:
            _sys.path.append(str(_p))  # append, never shadow stdlib
import gap_config as C
import overlap_metrics as OM
from measure_ivc_lciv_gap import surface_gap

# class -> mask filename
FILES = {"ivc": "inferior_vena_cava.nii.gz",
         "lciv": "iliac_vena_left.nii.gz",
         "rciv": "iliac_vena_right.nii.gz"}
CLASSES = ("ivc", "lciv", "rciv")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_masks(seg_dir: Path):
    """Read the three vein masks from a segmentation dir. Returns
    (masks: {cls: bool ndarray or None}, ref_img: sitk.Image or None, spacing_zyx or None).
    A class whose file is absent maps to None (the confluence helpers tolerate a missing
    right iliac). ref_img is the first mask read (for the affine/direction); spacing is
    reordered from sitk (sx,sy,sz) to the (sz,sy,sx) every array-axis-0=z helper expects."""
    seg_dir = Path(seg_dir)
    masks, ref = {}, None
    for cls in CLASSES:
        p = seg_dir / FILES[cls]
        if p.exists():
            img = sitk.ReadImage(str(p))
            if ref is not None:
                same_grid = (img.GetSize() == ref.GetSize()
                             and np.allclose(img.GetSpacing(), ref.GetSpacing())
                             and np.allclose(img.GetOrigin(), ref.GetOrigin())
                             and np.allclose(img.GetDirection(), ref.GetDirection()))
                if not same_grid:
                    raise ValueError(f"GRID_MISMATCH:{cls}")
            masks[cls] = sitk.GetArrayFromImage(img).astype(bool)
            if ref is None:
                ref = img
        else:
            masks[cls] = None
    if ref is None:
        return masks, None, None
    sx, sy, sz = ref.GetSpacing()
    return masks, ref, (float(sz), float(sy), float(sx))


# Publication topology audit: dominant endpoints, fixed across adjacency rules.

def venous_union(masks):
    """Boolean OR of the venous classes present; None when no class is present."""
    u = None
    for cls in CLASSES:
        if cls in masks and masks[cls] is not None:
            a = np.asarray(masks[cls], bool)
            u = a if u is None else (u | a)
    return u


def union_connectivity(masks, connectivity=None):
    """Connected components of the raw venous union, without any bridging.

    Returns (n_components, ivc_lciv_connected): the number of connected components of the union
    (26-connectivity by default, gap_config.OVERLAP_CONNECTIVITY) and whether the IVC and left iliac
    masks fall in a shared component of the union, that is, whether a continuous venous-labelled path
    spans the confluence. Returns (count, False) when the IVC or left iliac mask is missing or empty.
    An unsupported `connectivity` raises (overlap_metrics._structure accepts 6, 18 or 26 only).
    """
    from scipy.ndimage import label
    conn = C.OVERLAP_CONNECTIVITY if connectivity is None else connectivity
    struct = OM._structure(conn, 3)
    u = venous_union(masks)
    if u is None or not u.any():
        return 0, False
    # Crop to the union's bounding box before labelling: the components, and which of them the IVC
    # and left iliac voxels fall in, are unchanged, and a whole-body scan needs far less memory.
    zz, yy, xx = np.nonzero(u)
    sl = (slice(int(zz.min()), int(zz.max()) + 1),
          slice(int(yy.min()), int(yy.max()) + 1),
          slice(int(xx.min()), int(xx.max()) + 1))
    lab, n = label(u[sl], struct)
    ivc = masks.get("ivc")
    lciv = masks.get("lciv")
    if ivc is None or lciv is None \
            or not np.asarray(ivc, bool).any() or not np.asarray(lciv, bool).any():
        return int(n), False
    ivc_labels = set(np.unique(lab[np.asarray(ivc, bool)[sl]]).tolist()) - {0}
    lciv_labels = set(np.unique(lab[np.asarray(lciv, bool)[sl]]).tolist()) - {0}
    return int(n), bool(ivc_labels & lciv_labels)

import hashlib

import nibabel as nib
from scipy import ndimage as ndi

TOPOLOGY_CLASSES = ("direct", "right_iliac_mediated", "disconnected")
REFERENCE_VEIN_FILES = tuple(filename.removesuffix(".nii.gz") for filename in FILES.values())

def topology_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def largest_component26(mask):
    labels, count = ndi.label(mask, np.ones((3, 3, 3), bool))
    if count == 0:
        return np.zeros_like(mask), 0, 0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    index = int(np.argmax(sizes))  # deterministic array-order tie rule
    return labels == index, int(sizes[index]), count


def share_component(labels, a, b):
    x = np.unique(labels[a])
    y = np.unique(labels[b])
    return bool(np.intersect1d(x[x > 0], y[y > 0]).size)


def classify_topology(masks):
    """Fix endpoints to largest 26-components; vary only the allowed adjacency.

    The union retains all original vein voxels. Stray disconnected islands cannot
    create a failed path between the dominant IVC and LCIV components. Keeping
    these endpoint masks fixed makes no-path at 26 a subset of no-path at 6.
    """
    a, av, an = largest_component26(masks[0])
    b, bv, bn = largest_component26(masks[1])
    if not av or not bv:
        raise ValueError("empty primary mask in measurable frame")
    union = masks[0] | masks[1] | masks[2]
    pair = masks[0] | masks[1]
    result = {
        "ivc_main_voxels": av,
        "lciv_main_voxels": bv,
        "ivc_ncomp26": an,
        "lciv_ncomp26": bn,
    }
    for connectivity in (26, 6):
        structure = ndi.generate_binary_structure(3, 3 if connectivity == 26 else 1)
        lab, nc = ndi.label(union, structure)
        union_ok = share_component(lab, a, b)
        any_ok = share_component(lab, masks[0], masks[1])
        direct_lab, _ = ndi.label(pair, structure)
        direct_ok = share_component(direct_lab, a, b)
        result.update(
            {
                f"ncomp{connectivity}": nc,
                f"any_path{connectivity}": int(any_ok),
                f"main_path{connectivity}": int(union_ok),
                f"class{connectivity}": TOPOLOGY_CLASSES[
                    0 if direct_ok else 1 if union_ok else 2
                ],
            }
        )
    if result["main_path6"] > result["main_path26"]:
        raise AssertionError("6-connected path without a 26-connected path")
    return result


def read_topology_case(task):
    cohort, case, mask_case, folder, expected = task
    crops, boxes, affine, shape = [], [], None, None
    hashes = {}
    counts = []
    boundary = []
    for name, wanted, wanted_voxels in expected:
        path = Path(folder) / cohort / mask_case / f"{name}.nii.gz"
        actual = topology_digest(path)
        if actual != wanted:
            raise ValueError(f"mask hash mismatch: {cohort}/{case}/{name}")
        img = nib.load(path)
        if affine is not None and (
            img.shape != shape or not np.allclose(img.affine, affine, atol=1e-5)
        ):
            raise ValueError(f"grid mismatch: {cohort}/{case}")
        affine, shape = img.affine, img.shape
        arr = np.asanyarray(img.dataobj)
        if arr.ndim != 3 or arr.min() < 0 or arr.max() > 1:
            raise ValueError("mask must be binary 3D")
        voxels = int(np.count_nonzero(arr))
        if voxels != wanted_voxels:
            raise ValueError("manifest voxel count mismatch")
        counts.append(voxels)
        if voxels:
            support = [
                np.flatnonzero(arr.any(axis=tuple(j for j in range(3) if j != i)))
                for i in range(3)
            ]
            box = tuple(slice(int(x[0]), int(x[-1]) + 1) for x in support)
        else:
            box = None
        boxes.append(box)
        crops.append(arr[box].astype(bool) if box else None)
        boundary.append(
            bool(box and any(s.start == 0 or s.stop == n for s, n in zip(box, shape)))
        )
        hashes[name] = actual
        del arr, img
    present = [box for box in boxes if box]
    lo = np.min([[s.start for s in box] for box in present], axis=0)
    hi = np.max([[s.stop for s in box] for box in present], axis=0)
    masks = []
    for crop, box in zip(crops, boxes):
        mask = np.zeros(tuple(hi - lo), bool)
        if box:
            sl = tuple(slice(s.start - x, s.stop - x) for s, x in zip(box, lo))
            mask[sl] = crop
        masks.append(mask)
    result = {
        "cohort": cohort,
        "case": case,
        "mask_case": mask_case,
        **classify_topology(masks),
        "lciv_touches_image_boundary": int(boundary[1]),
        "ivc_voxels": counts[0],
        "lciv_voxels": counts[1],
        "rciv_voxels": counts[2],
    }
    result.update({f"{name}_sha256": h for name, h in hashes.items()})
    return result


def audit_reference_case(task):
    cache, case = task
    arrays, boxes, shape, affine, spacing = [], [], None, None, None
    orthonormal = True
    for name in REFERENCE_VEIN_FILES:
        image = nib.load(cache / case / "segmentations" / f"{name}.nii.gz")
        if shape is not None and (
            shape != image.shape or not np.allclose(affine, image.affine)
        ):
            raise ValueError("reference mask grid mismatch")
        shape, affine = image.shape, image.affine
        spacing = np.asarray(image.header.get_zooms()[:3])
        direction = affine[:3, :3] / spacing
        if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
            orthonormal = False
        arr = np.asanyarray(image.dataobj)
        if arr.dtype.kind not in "bui" or arr.min() < 0 or arr.max() > 1:
            raise ValueError("reference mask is not binary integer")
        if arr.any():
            support = [
                np.flatnonzero(arr.any(axis=tuple(j for j in range(3) if j != i)))
                for i in range(3)
            ]
            box = tuple(slice(int(x[0]), int(x[-1]) + 1) for x in support)
            arrays.append(arr[box].astype(bool))
        else:
            box = None
            arrays.append(None)
        boxes.append(box)
        del arr, image
    row = {
        "case": case,
        "ivc_present": int(boxes[0] is not None),
        "lciv_present": int(boxes[1] is not None),
        "rciv_present": int(boxes[2] is not None),
        "lciv_touches_image_boundary": int(
            boxes[1] is not None
            and any(s.start == 0 or s.stop == n for s, n in zip(boxes[1], shape))
        ),
    }
    if boxes[0] is None or boxes[1] is None:
        return {**row, "status": "empty_primary_label"}
    lo = np.min([[s.start for s in b] for b in boxes if b], axis=0)
    hi = np.max([[s.stop for s in b] for b in boxes if b], axis=0)
    masks = []
    for a, b in zip(arrays, boxes):
        m = np.zeros(tuple(hi - lo), bool)
        if b:
            m[tuple(slice(s.start - x, s.stop - x) for s, x in zip(b, lo))] = a
        masks.append(m)
    # Topology is invariant to XYZ/ZYX axis order. Distance scale matches XYZ arrays.
    gap = surface_gap(masks[0], masks[1], spacing)[0] if orthonormal else np.nan
    return {
        **row,
        "status": "measurable",
        "gap_mm": gap,
        "distance_status": "computed" if orthonormal else "unsupported_nonorthonormal_grid",
        "slice_mm": float(spacing[2]),
        **classify_topology(masks),
    }

