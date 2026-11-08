"""Measure the 3D gap between IVC and LCIV mask edges.

By default reads from <case>/segmentations/. Use --segdir to read from
a parallel directory (e.g. segmentations_fullres/), for example to compare
the fast and full-resolution passes.
"""
import argparse
import sys
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
from pathlib import Path

# --- run-from-anywhere: put sibling phase dirs on sys.path (tests use Code/conftest.py) ---
import sys as _sys, pathlib as _pl
for _p in sorted(_pl.Path(__file__).resolve().parent.parent.iterdir()):
    if _p.is_dir() and _p.name[0] not in "._":
        if str(_p) not in _sys.path:
            _sys.path.append(str(_p))  # append, never shadow stdlib
import gap_config as C


def _same_grid(a, b, atol=1e-5):
    """Require identical discrete size and physical image geometry."""
    return (a.GetSize() == b.GetSize()
            and np.allclose(a.GetSpacing(), b.GetSpacing(), rtol=0.0, atol=atol)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), rtol=0.0, atol=atol)
            and np.allclose(a.GetDirection(), b.GetDirection(), rtol=0.0, atol=atol))


def surface_gap(ivc_arr, lciv_arr, spacing_zyx):
    """Nearest distance between two binary masks, returned as
    (surface_to_surface_mm, center_to_center_mm).

    distance_transform_edt(~ivc) gives, at each background voxel CENTER, the Euclidean (mm) distance
    to the nearest IVC voxel CENTER; minimized over the LCIV voxels this is the center-to-center
    distance `c2c`.

    Surface-to-surface models each voxel as an axis-aligned box of full side (sz,sy,sx).
    Face-adjacent masks -> 0; a one-voxel void -> one voxel; surface <= c2c.

    Implemented with a KD-tree over voxel coordinates (~10^4-10^5 per mask) rather
    than a full-volume EDT with feature transform, which would allocate three
    full-volume index arrays per worker. c2c = nearest IVC voxel center; surface = nearest center of
    the DILATED IVC (which equals the nearest box surface, exact over all pairs).
    """
    sz, sy, sx = spacing_zyx
    scale = np.array([sz, sy, sx], float)
    lciv_pts = np.argwhere(lciv_arr) * scale
    ivc_pts = np.argwhere(ivc_arr) * scale
    if lciv_pts.size == 0 or ivc_pts.size == 0:
        return float("nan"), float("nan")
    c2c = float(cKDTree(ivc_pts).query(lciv_pts)[0].min())
    ivc_dil = binary_dilation(ivc_arr, structure=np.ones((3, 3, 3), bool))
    dil_pts = np.argwhere(ivc_dil) * scale
    surface = float(cKDTree(dil_pts).query(lciv_pts)[0].min())
    return surface, c2c


def surface_nearest_pair(ivc_arr, lciv_arr, spacing_zyx):
    """The voxel pair that ACHIEVES :func:`surface_gap`'s surface distance.

    Returns `(ivc_index, lciv_index, surface_mm)`; `(None, None, nan)` if either mask is empty.

    It sits beside the metric so that a localisation uses the pair the gap comes from. On an
    anisotropic grid they genuinely disagree: at spacing (5.0, 1.0, 1.0), a pair three voxels apart
    in x within one slice is 3.0 mm centre-to-centre and **2.0 mm** surface, while a pair one slice
    and one voxel apart is 5.10 mm centre-to-centre and **0.0 mm** surface. The centre-minimising
    search picks the first; the reported gap is the second.

    EXACTNESS. The axis-aligned box separation is `sqrt(sum_k (s_k * max(|di_k| - 1, 0))^2)` for
    an integer index offset `di`. The dilated-IVC KD-tree that :func:`surface_gap` already uses
    attains that minimum exactly: dilating by a 3x3x3 box offers every `delta in {-1,0,1}^3`, the
    per-axis choice is independent, and the optimal `delta_k` always moves the IVC voxel TOWARD
    the LCIV voxel, so it can never be clipped away at an array edge. That gives the winning LCIV
    voxel; its partner is then recovered by an exact vectorised box-to-box scan over the IVC
    voxels, which is O(|IVC|) once.

    Ties are broken the same way `surface_gap`'s own `.min()` is -- by first occurrence in
    `np.argwhere` (z, then y, then x) -- so the pair is deterministic.
    """
    scale = np.array(spacing_zyx, float)
    lciv_idx = np.argwhere(lciv_arr)
    ivc_idx = np.argwhere(ivc_arr)
    if lciv_idx.size == 0 or ivc_idx.size == 0:
        return None, None, float("nan")
    ivc_dil = binary_dilation(ivc_arr, structure=np.ones((3, 3, 3), bool))
    dil_pts = np.argwhere(ivc_dil) * scale
    dist, _ = cKDTree(dil_pts).query(lciv_idx * scale)
    j = int(dist.argmin())
    l = lciv_idx[j]
    sep = np.maximum(np.abs(ivc_idx - l).astype(float) * scale - scale, 0.0)
    box = np.sqrt((sep ** 2).sum(axis=1))
    i = int(box.argmin())
    return ivc_idx[i], l, float(box[i])


def box_to_box_mm(index_a, index_b, spacing_zyx):
    """Exact axis-aligned separation (mm) between the voxel BOXES at two integer indices.

    Zero for face-, edge- or corner-adjacent voxels and for a voxel with itself; otherwise the
    Euclidean norm of the per-axis gaps. Exposed so a caller (and a test) can check a reported
    pair against a reported gap without re-deriving the convention.
    """
    d = np.abs(np.asarray(index_a, float) - np.asarray(index_b, float))
    scale = np.asarray(spacing_zyx, float)
    return float(np.sqrt((np.maximum(d * scale - scale, 0.0) ** 2).sum()))


def lciv_quality(lciv_arr):
    """Segmentation-quality proxy from the LCIV mask alone (Phase-1 'segmentation
    confidence' correlate): (largest_connected_component_fraction, n_components) with
    full 3-D connectivity. A clean single vessel gives (1.0, 1); a fragmented mask
    gives (<1.0, >1). This may reflect anatomy, truncation, or segmentation error;
    true per-voxel softmax
    uncertainty would need the network probabilities; this is a defensible mask-only
    surrogate."""
    from scipy.ndimage import label, generate_binary_structure
    lab, n = label(np.asarray(lciv_arr, bool), generate_binary_structure(3, 3))
    if n == 0:
        return float("nan"), 0
    sizes = np.bincount(lab.ravel())[1:]
    return float(sizes.max() / sizes.sum()), int(n)


def body_csa(ct_slice, sx, sy, hu_thresh=-500.0):
    """Body cross-sectional area (mm^2) of one axial CT slice: the largest connected
    component of (HU > hu_thresh), so air and a detached scanner table are excluded.
    A patient-size proxy (Phase-1 'patient size' correlate). NaN if no body voxels."""
    from scipy.ndimage import label
    body = np.asarray(ct_slice, float) > hu_thresh
    # 8-connectivity (full 3x3 structure) so a pinched body silhouette joined only
    # at a diagonal isn't split into two components and under-counted.
    lab, n = label(body, structure=np.ones((3, 3), int))
    if n == 0:
        return float("nan")
    largest = int(np.bincount(lab.ravel())[1:].max())
    return float(largest * float(sx) * float(sy))


def case_aux_metrics(case_dir: Path, segdir_name: str = "segmentations_fullres") -> dict:
    """Per-case Phase-1 auxiliary correlates: LCIV segmentation quality (always) and a
    patient-size proxy = body CSA at the LCIV-top slice (only when a CT is present in
    the case dir; NaN otherwise, e.g. when the CT is not kept beside the masks). Graceful on any
    missing input."""
    case_dir = Path(case_dir)
    out = {"lciv_lcc_frac": float("nan"), "lciv_n_components": 0,
           "body_csa_mm2": float("nan")}
    lp = case_dir / segdir_name / "iliac_vena_left.nii.gz"
    if not lp.exists():
        return out
    lciv = sitk.ReadImage(str(lp))
    larr = sitk.GetArrayFromImage(lciv).astype(bool)
    out["lciv_lcc_frac"], out["lciv_n_components"] = lciv_quality(larr)
    # patient size: body CSA at the LCIV-top (most-cranial) slice, if a CT is here
    ct_path = next((case_dir / n for n in ("ct.nii.gz", "ct.nii") if (case_dir / n).exists()),
                   None)
    if ct_path is not None and larr.any():
        ct = sitk.ReadImage(str(ct_path))
        if not _same_grid(lciv, ct):
            raise ValueError(f"CT grid does not match LCIV mask for {case_dir.name}")
        ct_arr = sitk.GetArrayFromImage(ct).astype(float)        # (z, y, x) HU
        sx, sy, _sz = ct.GetSpacing()
        # most-cranial LCIV slice by the PHYSICAL cranial direction (consistent with
        # measure_one): the present voxel with the largest physical z. Picking the
        # raw highest array index (ks.max()) silently assumes higher index = cranial,
        # which is the most-CAUDAL slice when D[2,2] < 0 (reversed z) and can be wrong
        # for any direction where physical z is not monotonic in k (tilted gantry).
        # We take the k-slice of the present LCIV voxel with the largest physical z.
        ztop = _most_cranial_slice(lciv, larr)
        if 0 <= ztop < ct_arr.shape[0]:
            out["body_csa_mm2"] = body_csa(ct_arr[ztop], sx, sy)
    return out


def _most_cranial_slice(img, arr) -> int:
    """Array index k of the present voxel of `arr` (a (z,y,x) mask) with the largest PHYSICAL z. In
    SimpleITK's LPS world frame the third physical axis is ALWAYS Superior (cranial+), regardless of
    the direction matrix, so the most-cranial voxel is simply the one with maximal physical z — and
    for a tilted gantry that voxel can lie at an interior (i,j), which the per-voxel physical-z
    evaluation below handles. A negative D[2,2] only reverses which ARRAY INDEX is cranial, not the
    physical-z meaning."""
    D = np.asarray(img.GetDirection(), float).reshape(3, 3)
    ks, js, iis = np.nonzero(arr)                              # (z, y, x) indices
    # physical z = origin_z + D[2,:] . (i*sx, j*sy, k*sz). Vectorized over voxels.
    ox, oy, oz = img.GetOrigin()
    sx, sy, sz = img.GetSpacing()
    phys_z = (oz + D[2, 0] * (iis * sx) + D[2, 1] * (js * sy) + D[2, 2] * (ks * sz))
    return int(ks[np.argmax(phys_z)])


def measure_one(case_dir: Path, segdir_name: str) -> dict:
    seg = case_dir / segdir_name
    ivc_path = seg / "inferior_vena_cava.nii.gz"
    lciv_path = seg / "iliac_vena_left.nii.gz"
    if not (ivc_path.exists() and lciv_path.exists()):
        return {"case": case_dir.name, "error": f"missing files in {seg}"}

    ivc = sitk.ReadImage(str(ivc_path))
    lciv = sitk.ReadImage(str(lciv_path))
    if not _same_grid(ivc, lciv):
        return {"case": case_dir.name, "error": "IVC/LCIV grid mismatch"}
    ivc_arr = sitk.GetArrayFromImage(ivc).astype(bool)
    lciv_arr = sitk.GetArrayFromImage(lciv).astype(bool)
    sx, sy, sz = ivc.GetSpacing()

    def z_extent_lps(img, arr):
        # physical-z extent of the mask, computed over EVERY present voxel (vectorized). For an
        # axis-aligned/axis-flipped direction physical z = k*sz+origin, independent of (i,j),
        # matching measure_bifurcation._z_extent.
        ks, js, iis = np.nonzero(arr)                 # (z,y,x) indices, ALL present voxels
        if ks.size == 0:
            return float('nan'), float('nan')
        D = np.asarray(img.GetDirection(), float).reshape(3, 3)
        ox, oy, oz = img.GetOrigin()
        sxx, syy, szz = img.GetSpacing()
        phys_z = (oz + D[2, 0] * (iis * sxx) + D[2, 1] * (js * syy)
                  + D[2, 2] * (ks * szz))
        return float(phys_z.min()), float(phys_z.max())

    if not ivc_arr.any() or not lciv_arr.any():
        return {"case": case_dir.name, "error": "empty mask"}

    ivc_lo, ivc_hi = z_extent_lps(ivc, ivc_arr)
    lciv_lo, lciv_hi = z_extent_lps(lciv, lciv_arr)
    # Cranial = LARGER physical z, ALWAYS. In SimpleITK's LPS world frame the third physical axis is
    # Superior (cranial+) independent of the direction matrix; a negative D[2,2] only reverses which
    # ARRAY INDEX k is cranial, not the physical-z meaning. z_extent_lps returns (lo, hi) physical z
    # (lo <= hi), so a mask's caudal-most extreme is `lo` and its cranial-most extreme is `hi`.
    # Reversed-z storage is present in the SAROS masks and is handled exactly; oblique orthonormal
    # grids are measured and flagged OBLIQUE, and only non-orthonormal directions fail closed below.
    # raw_z_gap_mm keeps the SIGN: negative == the IVC and iliac labels
    # overlap in z. It is the cranial separation from the IVC's most-CAUDAL extreme (ivc_lo) to the
    # LCIV's most-CRANIAL extreme (lciv_hi); the IVC sits cranial of the LCIV. z_gap_mm clamps it at
    # 0 (the convention the gap analysis uses), which left-censors the variable; emitting the raw
    # value makes the censoring recoverable.
    raw_z_gap_mm = float(ivc_lo - lciv_hi)
    z_gap_mm = max(0.0, raw_z_gap_mm)

    # Index*spacing distances are exact under ANY finite orthonormal direction, not only signed
    # axis permutations. If p = D S i and D.T D = I, then
    # ||D S(i-j)||_2 == ||S(i-j)||_2; the same rigid transform applies to the voxel boxes used by
    # surface_gap. A rotation cannot change the primary Euclidean gap, so tilted-gantry grids are
    # measured; fail closed only for malformed non-orthonormal directions (scale/shear), which
    # SimpleITK geometry should not contain and for which the spacing-only metric would indeed be
    # wrong.
    D_signed = np.asarray(ivc.GetDirection(), float).reshape(3, 3)
    orthonormal = (bool(np.isfinite(D_signed).all())
                   and bool(np.allclose(D_signed.T @ D_signed, np.eye(3),
                                        rtol=0.0, atol=1e-6))
                   and bool(np.isclose(abs(np.linalg.det(D_signed)), 1.0,
                                       rtol=0.0, atol=1e-6)))
    if not orthonormal:
        return {"case": case_dir.name, "error": "NONORTHONORMAL_DIRECTION"}

    # surface-to-surface is the PRIMARY magnitude; center-to-center (c2c) is kept
    # for the Pythagorean geometry/invariant (d^2 = z^2 + lat^2 holds for c2c).
    surf_gap, c2c_gap = surface_gap(ivc_arr, lciv_arr, (sz, sy, sx))
    min_3d_gap = surf_gap

    # QC: flag degenerate (tiny) masks at the source so a bad segmentation does
    # not silently enter the gap distribution.
    ivc_vox, lciv_vox = int(ivc_arr.sum()), int(lciv_arr.sum())
    qc = []
    if ivc_vox < C.QC_MIN_VOXELS:
        qc.append("SMALL_IVC")
    if lciv_vox < C.QC_MIN_VOXELS:
        qc.append("SMALL_LCIV")
    if int((ivc_arr & lciv_arr).sum()) > 0:
        qc.append("OVERLAP")
    # surface_gap / c2c use index*spacing voxel coordinates, whose pairwise Euclidean distances
    # are exact for every orthonormal direction, including genuine rotations and reflections.
    # REVERSED_Z: the direction z-column is negative (D[2,2] < 0), i.e. the array
    # index k runs caudal-to-cranial. Physical z is still Superior, so the gap above
    # is computed correctly regardless (cranial = max physical z); this is a PURELY
    # INFORMATIONAL QC flag marking non-canonical index ordering, not an error.
    if float(np.sign(D_signed[2, 2])) < 0:
        qc.append("REVERSED_Z")
    # OBLIQUE: the direction is orthonormal but is NOT a signed axis permutation -- a genuinely
    # tilted (oblique) acquisition. Like REVERSED_Z above this is PURELY INFORMATIONAL, never a
    # refusal, and for the same reason: the gap is unaffected. A rigid rotation is an isometry, so
    # the box-to-box minimum is identical, which is why the endpoint measures these grids instead of
    # failing closed.
    _near1 = np.abs(np.abs(D_signed) - 1.0) < 1e-6
    axis_aligned = (bool(np.all((np.abs(D_signed) < 1e-6) | _near1))
                    and bool(np.all(_near1.sum(axis=0) == 1))
                    and bool(np.all(_near1.sum(axis=1) == 1)))
    if not axis_aligned:
        qc.append("OBLIQUE")
    # LARGE_GAP_IMPLAUSIBLE: a surface gap above QC_MAX_GAP_MM is physiologically impossible (a
    # normal confluence spans ~15-30 mm), so it almost always signals a mis-registration or a
    # wrong-case pairing. Uses the PRIMARY surface gap to match that count. A FLAG, not a drop: the
    # row is retained for the analysis-side sensitivity check to exclude.
    if min_3d_gap > C.QC_MAX_GAP_MM:
        qc.append("LARGE_GAP_IMPLAUSIBLE")

    return {
        "case": case_dir.name,
        "ivc_z_ext": float(ivc_hi - ivc_lo),
        "lciv_z_ext": float(lciv_hi - lciv_lo),
        "ivc_voxels": ivc_vox,
        "lciv_voxels": lciv_vox,
        "z_gap_mm": float(z_gap_mm),
        "raw_z_gap_mm": raw_z_gap_mm,
        "min_3d_gap_mm": float(min_3d_gap),      # surface-to-surface (PRIMARY)
        "c2c_3d_gap_mm": float(c2c_gap),         # center-to-center (geometry)
        "sx_mm": float(sx), "sy_mm": float(sy), "sz_mm": float(sz),
        "qc_flag": "+".join(qc) if qc else "OK",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cases", nargs="+", help="Case names")
    ap.add_argument("--data-dir", type=Path, default=Path("test_data"))
    ap.add_argument("--segdir", default="segmentations",
                    help="Segmentation subdirectory name "
                         "(default: segmentations; or segmentations_fullres)")
    ap.add_argument("--compare", action="store_true",
                    help="Compare segmentations/ vs segmentations_fullres/ "
                         "side-by-side for each case")
    args = ap.parse_args()

    if args.compare:
        print(f"{'case':<8} "
              f"{'fast_3d':>9} {'full_3d':>9} {'delta_mm':>9} "
              f"{'fast_vox':>9} {'full_vox':>9}")
        print('-' * 65)
        for c in args.cases:
            case_dir = args.data_dir / c
            fast = measure_one(case_dir, "segmentations")
            full = measure_one(case_dir, "segmentations_fullres")
            if "error" in fast or "error" in full:
                msg = fast.get("error") or full.get("error")
                print(f"{c:<8} {msg}")
                continue
            delta = full['min_3d_gap_mm'] - fast['min_3d_gap_mm']
            print(f"{c:<8} "
                  f"{fast['min_3d_gap_mm']:9.1f} {full['min_3d_gap_mm']:9.1f} "
                  f"{delta:+9.1f} "
                  f"{fast['ivc_voxels']:9d} {full['ivc_voxels']:9d}")
        return 0

    print(f"{'case':<8} {'ivc_z':>6} {'lciv_z':>7} "
          f"{'z_gap':>7} {'surf_3d':>8} {'raw_gap':>8} {'c2c_3d':>7} {'qc':>10}  "
          f"(segdir={args.segdir})")
    print('-' * 72)
    for c in args.cases:
        case_dir = args.data_dir / c
        r = measure_one(case_dir, args.segdir)
        if "error" in r:
            print(f"{c:<8} ERROR: {r['error']}")
            continue
        print(f"{c:<8} {r['ivc_z_ext']:6.0f} {r['lciv_z_ext']:7.0f} "
              f"{r['z_gap_mm']:7.2f} {r['min_3d_gap_mm']:8.2f} "
              f"{r['raw_z_gap_mm']:+8.1f} {r['c2c_3d_gap_mm']:7.2f} "
              f"{r['qc_flag']:>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
