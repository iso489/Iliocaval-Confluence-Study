#!/usr/bin/env python3
"""Locate the IVC-LCIV gap relative to the iliocaval confluence (an automated PROXY).

TotalSegmentator leaves the confluence UNLABELED (a gap between the IVC label and the iliac-vein
labels), so the confluence cannot be read directly off the segmentation. We estimate its level from
the two common iliac vein (CIV) masks ALONE (independent of the IVC mask — no circular reasoning)
and express the gap's position relative to it. This z is an automated PROXY for the confluence, NOT
a new clinical definition: the literature localizes the confluence visually (most caudal single-IVC
slice / most cranial two-CIV slice) or by vertebral level (typically L5; Moussallem 2012 65.75%).
An optional --reference CSV of confluence z values can be compared with the apex proxy; the study
collects no such reference. The wall-contact / surface-tangency idea itself has precedent in the
VMTK/Antiga bifurcation reference systems (apex = where the two vessel surfaces meet); here it is
adapted to stacked 2-D segmentation cross-sections with extrapolation across the unlabeled gap and
bootstrap-CI propagation.

The PRIMARY apex ('wallgap', gap_config.BIF_APEX_METHOD) extrapolates the per-slice minimum
wall-to-wall distance between the two CIV masks to zero, with no circular-lumen assumption; the
equivalent-circle 'tangency' contact is the comparison. The wall-gap estimand preserves
elliptical/compressed CIV shape, the whole point of a May-Thurner study (the left CIV is
characteristically flattened where the right common iliac artery crosses it). Both are reported per
case (z_bif primary, z_bif_alt comparison) and cross-checked in the summary. A centroid-coincidence
estimand (sep -> 0) would sit about one vein diameter cranial of wall contact and manufacture a
spurious offset.

The apex carries a BCa-bootstrap 95% CI (a ratio of OLS coefficients needs the second-order
interval).

The confluence is reported as a POINT (z_bif + CI) AND as a ZONE: the gap bracket
[iliac_top, ivc_bottom] is the unlabeled transition region; its median height is
compared to the literature junction height (~15.5 mm, Pirro 2005).

All distances are physical mm along z; larger z = cranial (LPS superior).

A run in which no case was measured (every row MISSING, GRID_MISMATCH, EMPTY or
ERROR:), or in which no case yields an apex (every row NO_RIGHT, FEW_SLICES or failed,
z_bif NA throughout), publishes nothing, exits non-zero and leaves any previous table
in place; a partial failure rate is data, and is published. --reference is read
before anything is published, so a malformed reference also leaves it in place."""
import argparse
import collections
import csv
import math
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# --- run-from-anywhere: put sibling phase dirs on sys.path (tests use Code/conftest.py) ---
import sys as _sys, pathlib as _pl
for _p in sorted(_pl.Path(__file__).resolve().parent.parent.iterdir()):
    if _p.is_dir() and _p.name[0] not in "._":
        if str(_p) not in _sys.path:
            _sys.path.append(str(_p))  # append, never shadow stdlib
import hpc_resources as HR  # SLURM/cgroup-aware workers + thread pinning
import gap_config as C
import statlib as S
from measure_all import load_expected_roster
from tableio import AtomicTable, results_path  # atomic + completion-marked

DATA = None      # set in main()/_init_worker (Pool initializer; fork AND spawn)
SEGDIR = "segmentations_fullres"

IVC = "inferior_vena_cava.nii.gz"
LIL = "iliac_vena_left.nii.gz"
RIL = "iliac_vena_right.nii.gz"

COLUMNS = ["case", "ivc_bottom", "iliac_top", "z_gap", "z_bif",
           "z_bif_ci_lo", "z_bif_ci_hi", "z_bif_alt", "apex_method",
           "slope", "slope_se", "ivc_overshoot", "iliac_undershoot",
           "gap_center_offset", "frac_above", "apex_in_gap", "iliac_r_mm",
           "alpha1_deg", "alpha2_deg", "fit_n", "window_fallback",
           "vert_n", "gap_top_level", "gap_bottom_level", "apex_level", "vert_qc",
           "flag",
           "window_stability_mm", "conv_angle_deg", "oblique",
           "curvature_status", "curvature_error_type"]


# ---------------------------------------------------------------------------
# Pure statistical core (no SimpleITK; unit-tested in test_bifurcation.py)
# ---------------------------------------------------------------------------
def apex_relative(ivc_bottom, iliac_top, z_bif):
    """Apex-relative quantities (z increases cranially; the labeled gap occupies
    the z interval [iliac_top, ivc_bottom]).

    frac_above = overshoot/z_gap is a genuine fraction in [0,1] ONLY when the apex
    falls inside the gap; outside it becomes a signed unbounded ratio with no
    "fraction" meaning. We therefore also return apex_in_gap (1 if z_bif is within
    [iliac_top, ivc_bottom], else 0; NaN when z_gap<=0), and the summary leads with
    apex_in_gap + raw overshoot/undershoot/offset. frac_above stays NaN for
    z_gap<=0 (excluded, not zeroed).

    NO-GAP cases (z_gap <= 0, i.e. the IVC and iliac labels OVERLAP or abut): there
    is no unlabeled transition region to be centered on, so overshoot / undershoot /
    gap_center_offset / apex_in_gap are ALL undefined (NaN), not finite numbers that
    would silently enter the centeredness aggregates. z_gap itself is kept (it
    is the negative/zero overlap extent, reported as-is)."""
    z_gap = ivc_bottom - iliac_top
    if z_gap > 0:
        overshoot = ivc_bottom - z_bif
        undershoot = z_bif - iliac_top
        offset = 0.5 * (ivc_bottom + iliac_top) - z_bif
        frac_above = overshoot / z_gap
        apex_in_gap = 1.0 if (iliac_top <= z_bif <= ivc_bottom) else 0.0
    else:
        # no unlabeled gap -> every gap-relative quantity is undefined
        overshoot = undershoot = offset = float("nan")
        frac_above = apex_in_gap = float("nan")
    return {"z_gap": z_gap, "ivc_overshoot": overshoot,
            "iliac_undershoot": undershoot, "gap_center_offset": offset,
            "frac_above": frac_above, "apex_in_gap": apex_in_gap}


def decide_flag(slope, unstable, ci_lo, ci_hi, z_gap, n_shared,
                variant=False, nonlinear=False, curvature_assessable=True,
                min_slices=None, slope_tol=None, wide_ci_mm=None):
    min_slices = C.BIF_MIN_SLICES if min_slices is None else min_slices
    slope_tol = C.BIF_SLOPE_TOL if slope_tol is None else slope_tol
    wide_ci_mm = C.BIF_WIDE_CI_MM if wide_ci_mm is None else wide_ci_mm
    if n_shared < min_slices:
        return "FEW_SLICES"
    if variant:
        return "VARIANT"
    if unstable or slope >= -slope_tol:
        return "NONCONVERGING"
    if not curvature_assessable:
        return "CURVATURE_UNASSESSABLE"
    if nonlinear:
        return "NONLINEAR"
    hw = 0.5 * (ci_hi - ci_lo)
    if not math.isfinite(hw) or hw > wide_ci_mm:
        return "WIDE_CI"
    return "OK_OVERLAP" if z_gap <= 0 else "OK"


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def _quad_apex_shift(z, sep, R, z_lin):
    """Materiality of curvature: how far the apex moves if sep(z) is modeled as
    quadratic instead of linear. Fit sep = a z^2 + b z + c, solve sep = R, take the
    real root nearest the linear apex z_lin, return |z_quad - z_lin|.

    This is MATERIAL curvature, not "uncomputable" — caller treats it as an apex move larger than
    any finite CI half-width. NaN — genuinely not computable (too few points, flat quadratic, or
    degenerate inputs); curvature materiality is undetermined."""
    z = np.asarray(z, float)
    sep = np.asarray(sep, float)
    if z.size < 4 or not (math.isfinite(R) and math.isfinite(z_lin)):
        return float("nan")
    try:
        a, b, c = np.polyfit(z, sep, 2)
    except Exception:
        return float("nan")
    if abs(a) < 1e-12:
        return float("nan")
    disc = b * b - 4.0 * a * (c - R)
    if disc < 0:
        # well-formed quadratic that never reaches contact level R: the apex shift
        # is effectively infinite (no quadratic apex exists) -> material curvature.
        return float("inf")
    s = math.sqrt(disc)
    roots = [(-b + s) / (2.0 * a), (-b - s) / (2.0 * a)]
    z_quad = min(roots, key=lambda t: abs(t - z_lin))
    return abs(z_quad - z_lin)


def _nonlinear_flag(curv_significant, shift, ci_hw):
    """    shift == +inf -> NONLINEAR UNCONDITIONALLY. _quad_apex_shift returns +inf when the quadratic
    sep-fit has NO real root at the contact level R (the curve never reaches contact), so a linear
    extrapolation TO R is unbounded-ly wrong — untrustworthy regardless of the quadratic-coefficient
    p-value. NaN shift -> uncomputable (<4 pts / flat quadratic); not flagged on its own."""
    if shift == float("inf"):
        return True
    finite_material = (math.isfinite(shift) and math.isfinite(ci_hw) and shift > ci_hw)
    return bool(curv_significant and finite_material)


def _point_apex(zz, sep, contact, slope_tol=None):
    """Fast point-estimate apex (no bootstrap): OLS root of sep(z) = median(contact).
    Used for the window-size sensitivity sweep, so it is cheap to evaluate at
    several windows per case."""
    slope_tol = C.BIF_SLOPE_TOL if slope_tol is None else slope_tol
    zz = np.asarray(zz, float)
    if zz.size < 2 or np.ptp(zz) == 0:
        return float("nan")
    slope, inter = np.polyfit(zz, np.asarray(sep, float), 1)
    if abs(slope) <= slope_tol:
        return float("nan")
    return (float(np.median(contact)) - inter) / slope


def _cc_count(mask2d):
    """Number of connected components in a 2-D boolean mask using 8-CONNECTIVITY (full 3x3 structuring
    element)."""
    from scipy.ndimage import label as cc_label, generate_binary_structure
    return int(cc_label(np.asarray(mask2d, bool),
                        generate_binary_structure(2, 2))[1])


def _principal_axis_3d(pts):
    """Unit 3-D direction of the best-fit line through points (first principal axis
    via SVD). `pts` is (N, 3) in physical (x, y, z) mm. None if < 2 points or
    degenerate. The principal axis IS a least-squares 3-D secant of the centerline."""
    pts = np.asarray(pts, float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 2:
        return None
    c = pts - pts.mean(axis=0)
    if not np.any(c):
        return None
    _u, _s, vt = np.linalg.svd(c, full_matrices=False)
    d = vt[0]
    n = np.linalg.norm(d)
    return d / n if n > 0 else None


def vessel_axis_3d(centroids, cranial):
    """Principal-axis unit vector of a vessel's per-slice centroids (N x 3, (x,y,z) mm),
    oriented cranially (z-component > 0) when `cranial` else caudally (z-component < 0).
    None if undefined."""
    d = _principal_axis_3d(centroids)
    if d is None:
        return None
    if (d[2] > 0) != cranial:        # flip to the requested cranio-caudal orientation
        d = -d
    return d


def bifurcation_angle_deg(ivc_centroids, civ_centroids):
    """**alpha2** (left CIV) / **alpha1** (right CIV): the iliocaval bifurcation angle, defined per
    Chen H-W et al. PRESERVE L/R: alpha2 is the LEFT (larger) angle, alpha1 the RIGHT — swapping
    them is a real error.

    SECANT-3D construction (a 3-D estimand, vs the literature's 2-D coronal
    projection): the angle between the CIV's travel direction and the IVC's **caudal
    axis** (the IVC's straight-down continuation). It is 0° when the CIV is a straight
    caudal continuation of the IVC and grows to ~90° as the vein splays laterally —
    matching the literature semantics. Each vessel's direction is the first principal
    axis (a least-squares 3-D secant) of its per-slice centroids over the confluence
    window. Translation-invariant; NaN when either axis is undefined.

    NOTE: this is a true 3-D angle and the cited papers report a 2-D coronal projection, so it is a
    measurement of the same anatomical quantity but not the same number, and should be compared as
    such. For two directions at EQUAL elevation phi above the projection plane with in-plane angle
    psi, cos(theta_3) - cos(theta_2) = (1 - cos psi) * sin^2(phi) >= 0, so theta_3 <= theta_2: in
    the near-symmetric case the 3-D angle is at MOST the projected one. The direction of any
    discrepancy against Chen 2023 is therefore not predictable from geometry alone and must not be
    presented as if it were."""
    d_ivc = vessel_axis_3d(ivc_centroids, cranial=True)     # IVC points cranial
    d_civ = vessel_axis_3d(civ_centroids, cranial=False)    # CIV travels caudal
    if d_ivc is None or d_civ is None:
        return float("nan")
    cosang = float(np.dot(d_civ, -d_ivc))                   # vs the IVC's caudal axis
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))


def _conv_angle_deg(slope):
    """Apparent FULL interiliac convergence angle (the angle BETWEEN the two common
    iliac veins, i.e. the Pirro interiliac / Chen alpha1+alpha2 analogue — NOT the
    single left-vein angle alpha2), computed as 2*atan(|slope|/2). Reported for face
    validity against the full-angle literature range (Pirro 2005 interiliac ~69 deg
    [28-107]; Chen 2023 alpha1+alpha2 ~83 deg [45-130]) and never enters inference. Do
    NOT read it against alpha2 (~49 deg) alone.

    What the number is, and the SIGN of its error:

      1. It is NOT a coronal projection. `slope` is fitted to `sep`, and `sep` is
         `np.linalg.norm(cl - cr)` over PHYSICAL (x, y, z) centroids, i.e. the full 3-D
         distance, so anterior-posterior splay is included and the literature comparison's
         projection premise does not apply to this number.
      2. Orthogonal projection can enlarge an angle as well as shrink it: u = (1,0,10),
         v = (-1,0,10) have a 3-D angle of arccos(99/101) = 11.42 deg and project onto z = 0
         at exactly 180 deg, and over isotropic random directions the sign is a coin flip
         (49.9% / 50.1%). In the configuration two SYMMETRIC iliac veins occupy (equal
         elevation phi above the plane, in-plane angle psi),
         cos(theta_3) - cos(theta_2) = (1 - cos psi) * sin^2(phi) >= 0, hence
         theta_3 <= theta_2 always: a projection reads at or ABOVE the 3-D angle.
      3. "Each vein approaches the midline at |slope|/2 per unit z" is an ASSUMPTION of
         symmetric convergence, not a derivation. `slope` is d(sep)/dz, the SUM of the two
         per-vein rates a and b, and one number cannot identify two directions. The true
         inter-vein angle is atan(a) + atan(b), and because atan is strictly concave on
         (0, inf), Jensen gives 2*atan((a+b)/2) >= atan(a) + atan(b) with equality iff
         a = b. So this formula OVERSTATES an asymmetric confluence and never understates
         it. Over slope 0.2-2.0 and asymmetry a/b 1-9 the worst overstatement is
         **+17.74 deg** (a true 72.26 deg reported as 90.00 deg, 24.6% relative). The
         iliocaval confluence is characteristically asymmetric (Chen 2023 gives alpha2
         48.8 deg against alpha1 34.1 deg), so this is the expected regime, not a corner
         case.

    NO REPAIR OF THIS FORMULA RECOVERS THE TRUE ANGLE, because the information is not in
    its input. `bifurcation_angle_deg` builds a per-vein 3-D principal axis for each side,
    and its alpha1 + alpha2 is the asymmetry-free full angle; this function is the
    slope-derived cross-check, with its assumption and the SIGN of its error stated."""
    if not (isinstance(slope, float) and math.isfinite(slope)):
        return float("nan")
    return math.degrees(2.0 * math.atan(abs(slope) / 2.0))


def summarize(rows, seed=None):
    """NaN-safe aggregation over OK-like rows, plus the wallgap-vs-tangency apex
    cross-check, the confluence-zone height, and selection-bias accounting."""
    seed = C.SEED if seed is None else seed
    flags = {}
    for r in rows:
        f = r.get("flag", "?")
        flags[f] = flags.get(f, 0) + 1
    ok = [r for r in rows if r.get("flag") in ("OK", "OK_OVERLAP")]

    def col(key, predicate=None):
        vals = [r[key] for r in ok
                if key in r and _finite(r[key])
                and (predicate is None or predicate(r))]
        return np.array(vals, float)

    frac = col("frac_above", predicate=lambda r: r.get("apex_in_gap") == 1.0)
    overshoot = col("ivc_overshoot")
    undershoot = col("iliac_undershoot")
    offset = col("gap_center_offset")
    apex_in = col("apex_in_gap")
    radii = col("iliac_r_mm")
    zgap_ok = col("z_gap")

    def med_ci(a):
        if a.size == 0:
            return float("nan"), (float("nan"), float("nan"))
        lo, hi = S.median_ci_auto(a, C.CI_METHOD, C.ALPHA, C.N_BOOT, seed)
        return float(np.median(a)), (lo, hi)

    m_frac, ci_frac = med_ci(frac)
    m_over, ci_over = med_ci(overshoot)
    m_under, ci_under = med_ci(undershoot)
    m_off, ci_off = med_ci(offset)
    m_zone, ci_zone = med_ci(zgap_ok[zgap_ok > 0]) if zgap_ok.size else (float("nan"), (float("nan"),) * 2)
    n_apex_in = int(np.nansum(apex_in)) if apex_in.size else 0

    # The apex_in_gap rate above is k of the accepted fits (flag OK), but the cases it speaks for
    # are every case with z_gap > 0, whatever its flag, and an outcome exists only where the fit was
    # accepted (apex_relative gives one for every accepted record). So the rate is reported with its
    # worst-case (Manski) range over that frame, which assumes nothing about the fits that failed. A
    # row with no finite z_gap (SAROS's EMPTY rows) may or may not belong to the frame, so a second
    # range counts each as a possible member. Both refuse, with the reason, when an evaluated row
    # lies outside the frame (the printed rate would not be a rate over it), when an accepted row in
    # the frame has no outcome (apex_relative always gives one), or when an outcome is not 0 or 1
    # (it would not be a proportion's). Most rows the gate withholds still carry a fitted z_bif
    # (NONLINEAR, WIDE_CI, VARIANT: an apex estimated and rejected), so where those rejected
    # estimates fall is counted too, as a diagnostic of whether the gate is related to the outcome,
    # never as a rate.
    frame = [r for r in rows if _finite(r.get("z_gap")) and r["z_gap"] > 0]
    n_frame_unknown = sum(1 for r in rows if not _finite(r.get("z_gap")))
    frame_in = frame_eval = rejected_fitted = rejected_inside = accepted_missing = 0
    unevaluated = collections.Counter()
    for r in frame:
        if r.get("flag") in ("OK", "OK_OVERLAP"):
            if _finite(r.get("apex_in_gap")):
                frame_eval += 1
                frame_in += int(r["apex_in_gap"] == 1.0)
            else:
                accepted_missing += 1
            continue
        unevaluated[str(r.get("flag", "?"))] += 1
        if all(_finite(r.get(k)) for k in ("z_bif", "iliac_top", "ivc_bottom")):
            rejected_fitted += 1
            rejected_inside += int(r["iliac_top"] <= r["z_bif"] <= r["ivc_bottom"])
    if frame_eval != int(apex_in.size):
        refused = "outside_frame"
    elif accepted_missing:
        refused = "accepted_without_outcome"
    elif not all(v in (0.0, 1.0) for v in apex_in):
        refused = "not_binary"
    else:
        refused = None
    if refused is None:
        bounds = S.worst_case_proportion_bounds(frame_in, frame_eval, len(frame))
        bounds_unknown = S.worst_case_proportion_bounds(frame_in, frame_eval, len(frame),
                                                        n_frame_unknown)
    else:
        bounds = bounds_unknown = (float("nan"), float("nan"))

    # wallgap-vs-tangency apex cross-check. The bias and LoA are unaffected by referencing to
    # iliac_top (it cancels in the per-case difference) and are the headline.
    base = [(r["z_bif"], r["z_bif_alt"], r["iliac_top"]) for r in ok
            if _finite(r.get("z_bif")) and _finite(r.get("z_bif_alt"))
            and _finite(r.get("iliac_top"))]
    if len(base) >= 2:
        pa = np.array([b[0] - b[2] for b in base], float)   # wallgap apex vs iliac_top
        pb = np.array([b[1] - b[2] for b in base], float)   # tangency apex vs iliac_top
        cross = S.method_agreement(pa, pb, C.ALPHA, tol=C.BIF_VALID_TOL_MM)
    else:
        cross = None
    win_stab = col("window_stability_mm")
    # window-stability split: under window_fallback the sweep collapses onto the
    # SAME cranial-most min_slices points at every window, so the spread is ~0 BY
    # CONSTRUCTION and tells us nothing. Report non-fallback and fallback separately.
    win_stab_nf = col("window_stability_mm", predicate=lambda r: not r.get("window_fallback"))
    win_stab_fb = col("window_stability_mm", predicate=lambda r: bool(r.get("window_fallback")))
    conv_ang = col("conv_angle_deg")
    # alpha2 (LCIV-IVC) / alpha1 (RCIV-IVC) are independent of the apex fit, so they
    # aggregate over EVERY case with a finite angle (not just OK-like).
    alpha2_all = np.array([r["alpha2_deg"] for r in rows
                           if "alpha2_deg" in r and _finite(r["alpha2_deg"])], float)
    alpha1_all = np.array([r["alpha1_deg"] for r in rows
                           if "alpha1_deg" in r and _finite(r["alpha1_deg"])], float)
    m_a2, ci_a2 = med_ci(alpha2_all)

    def med_zgap(pred):
        vals = [r["z_gap"] for r in rows
                if pred(r) and "z_gap" in r and _finite(r["z_gap"])]
        return float(np.median(vals)) if vals else float("nan")
    excl_flags = ("NO_RIGHT", "GRID_MISMATCH", "FEW_SLICES", "VARIANT",
                  "NONCONVERGING", "CURVATURE_UNASSESSABLE", "NONLINEAR", "WIDE_CI")
    n_wf = sum(1 for r in ok if r.get("window_fallback"))
    return {
        "flags": flags,
        "n_total": len(rows),
        "n_ok_like": len(ok),
        "n_window_fallback": n_wf,
        "n_frac_above": int(frac.size),
        "median_frac_above": m_frac, "ci_frac_above": ci_frac,
        "n_apex_in_gap": n_apex_in, "n_apex_eval": int(apex_in.size),
        "n_apex_frame": len(frame), "n_apex_frame_unknown": n_frame_unknown,
        "apex_unevaluated_flags": dict(unevaluated),
        "apex_in_gap_bounds": bounds, "apex_in_gap_bounds_unknown": bounds_unknown,
        "apex_in_gap_bounds_refused": refused,
        "n_apex_rejected_fitted": rejected_fitted, "n_apex_rejected_inside": rejected_inside,
        "median_overshoot": m_over, "ci_overshoot": ci_over,
        "median_undershoot": m_under, "ci_undershoot": ci_under,
        "median_offset": m_off, "ci_offset": ci_off,
        "n_offset_eval": int(offset.size),   # gap>0 cases only (no-gap offsets are NaN)
        "median_zone_mm": m_zone, "ci_zone": ci_zone,
        "median_iliac_r_mm": float(np.median(radii)) if radii.size else float("nan"),
        "median_window_stability_mm": float(np.median(win_stab)) if win_stab.size else float("nan"),
        "median_window_stability_nonfallback_mm":
            float(np.median(win_stab_nf)) if win_stab_nf.size else float("nan"),
        "n_window_stability_nonfallback": int(win_stab_nf.size),
        "median_window_stability_fallback_mm":
            float(np.median(win_stab_fb)) if win_stab_fb.size else float("nan"),
        "n_window_stability_fallback": int(win_stab_fb.size),
        "window_fallback_rate": (n_wf / len(ok)) if ok else float("nan"),
        "median_conv_angle_deg": float(np.median(conv_ang)) if conv_ang.size else float("nan"),
        "median_alpha2_deg": m_a2, "ci_alpha2": ci_a2, "n_alpha2": int(alpha2_all.size),
        "median_alpha1_deg": float(np.median(alpha1_all)) if alpha1_all.size else float("nan"),
        "cross_check": cross,
        "med_zgap_ok": med_zgap(lambda r: r.get("flag") in ("OK", "OK_OVERLAP")),
        "med_zgap_excluded": med_zgap(lambda r: r.get("flag") in excl_flags),
    }


# ---------------------------------------------------------------------------
# Image I/O (SimpleITK / scipy imported lazily so the math above is testable)
# ---------------------------------------------------------------------------
def _load(path):
    import SimpleITK as sitk
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(bool)   # (z, y, x)
    return img, arr


def _same_grid(a, b, atol=1e-5):
    """Strict shared-grid contract for masks combined voxelwise or slice-paired."""
    return (a.GetSize() == b.GetSize()
            and np.allclose(a.GetSpacing(), b.GetSpacing(), rtol=0.0, atol=atol)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), rtol=0.0, atol=atol)
            and np.allclose(a.GetDirection(), b.GetDirection(), rtol=0.0, atol=atol))


def _phys_z(img, k):
    return img.TransformContinuousIndexToPhysicalPoint((0.0, 0.0, float(k)))[2]


def is_oblique(img, tol=1e-6):
    """True when the image direction matrix is NOT a signed permutation (a genuine
    rotation/shear, e.g. a tilted gantry), within `tol`. Mirrors the OBLIQUE test in
    measure_ivc_lciv_gap.py: axis FLIPS/PERMUTATIONS (the LPS/RAS
    flips that are near-universal for CT) preserve voxel-pair distances and physical-z
    = k*sz+origin independent of (i,j), so they are NOT oblique; only a non-signed-
    permutation direction couples physical z to the in-plane (i,j)."""
    D = np.abs(np.asarray(img.GetDirection(), float).reshape(3, 3))
    near1 = np.abs(D - 1.0) < tol
    axis_aligned = (bool(np.all((D < tol) | near1))
                    and bool(np.all(near1.sum(axis=0) == 1))
                    and bool(np.all(near1.sum(axis=1) == 1)))
    return not axis_aligned


def _z_extent(img, arr):
    """Physical-z extent (min, max) of the mask, plus the present-z slice indices.

    For an axis-aligned/axis-flipped direction physical z = k*sz+origin, independent of (i,j)."""
    ks_all, js, iis = np.nonzero(arr)             # (z,y,x) indices, ALL present voxels
    ks = np.where(arr.any(axis=(1, 2)))[0]        # present z-slice indices (returned)
    D = np.asarray(img.GetDirection(), float).reshape(3, 3)
    ox, oy, oz = img.GetOrigin()
    sx, sy, sz = img.GetSpacing()
    phys_z = oz + D[2, 0] * (iis * sx) + D[2, 1] * (js * sy) + D[2, 2] * (ks_all * sz)
    return float(phys_z.min()), float(phys_z.max()), ks


def _largest_cc_slice(sl):
    """The LARGEST 8-connected component of a 2-D boolean slice (None if empty).

    This is the single speck-hardening definition shared by the centroid, d_min and
    the equivalent-circle radii: all three components of
    contact_actual = sep - d_min must be hardened CONSISTENTLY, or a detached speck
    that cannot drag the largest-CC centroid can still collapse d_min (a single
    contralateral voxel can take d_min from 5.00 to 0.00, inflating that slice's wallgap
    contact source by one lumen width). 8-connectivity matches _cc_count."""
    sl = np.asarray(sl, bool)
    if not sl.any():
        return None
    from scipy.ndimage import label as cc_label, generate_binary_structure
    lab, n = cc_label(sl, generate_binary_structure(2, 2))   # 8-connectivity (matches _cc_count)
    if n > 1:
        sizes = np.bincount(lab.ravel())[1:]                 # component sizes (skip background)
        return lab == (int(np.argmax(sizes)) + 1)            # keep only the largest component
    return sl


def _centroid_phys(img, arr, k):
    """Physical (x, y, z) centroid of slice k's LARGEST connected component.

    Averaging ALL nonzero voxels lets a detached segmentation speck (present on a
    minority of slices, so it does NOT trip VARIANT which needs >1 component on >=50%
    of window slices) drag the centroid, sep, the OLS slope and the apex (an apex shift of
    about 1 mm). Restricting to the largest 8-connected component per slice (the SAME
    connectivity _cc_count uses for component counting) makes the centroid robust to
    such specks while leaving a clean single-lumen slice unchanged."""
    sl = _largest_cc_slice(arr[k])
    if sl is None:
        return None
    ys, xs = np.nonzero(sl)
    return img.TransformContinuousIndexToPhysicalPoint(
        (float(xs.mean()), float(ys.mean()), float(k)))


def _axis_centroids(img, arr, ks, anchor_z, window_mm, cranial):
    """Per-slice physical centroids (x, y, z mm) for the mask slices within
    `window_mm` of `anchor_z` on the confluence side: `cranial=True` keeps
    z ∈ [anchor, anchor+window] (the IVC, whose confluence end is caudal); else
    z ∈ [anchor-window, anchor] (a CIV, whose confluence end is cranial). Feeds the
    alpha2 secant fit. Returns an (M, 3) array (possibly empty)."""
    out = []
    for k in ks:
        c = _centroid_phys(img, arr, int(k))
        if c is None:
            continue
        z = float(c[2])
        if cranial:
            if not (anchor_z <= z <= anchor_z + window_mm):
                continue
        elif not (anchor_z - window_mm <= z <= anchor_z):
            continue
        out.append((c[0], c[1], c[2]))
    return np.array(out, float) if out else np.empty((0, 3))


# --------------------------------------------------------------------------- Vertebral-level
# localization of the confluence (literature-standard anchor)
# --------------------------------------------------------------------------- The literature defines
# the iliocaval confluence by VERTEBRAL LEVEL (Moussallem 2012: L5 in 65.75%; Pirro 2005: L4-S1,
# most often L5). TotalSegmentator's default 'total' task labels the lumbar vertebrae + sacrum in
# the SAME image as the veins, so we can express iliac_top / ivc_bottom / the apex z_bif as a
# CONTINUOUS vertebral level — a clinically-meaningful, per-patient-normalized anchor (it also
# cancels the ~0.6 m between-patient table-position variance that otherwise inflates absolute-z CCC
# to ~1). It is REPORTED DISTRIBUTIONALLY (level histogram vs Moussallem/Pirro), never fed back as a
# fake per-case reference standard. With no vertebra masks (veins-only runs) levels read NA and
# vert_qc=NO_VERT.
VERT_LEVELS = [("vertebrae_T12", 0.0), ("vertebrae_L1", 1.0), ("vertebrae_L2", 2.0),
               ("vertebrae_L3", 3.0), ("vertebrae_L4", 4.0), ("vertebrae_L5", 5.0),
               ("vertebrae_S1", 6.0)]
LEVEL_NAME = {-1: "T11", 0: "T12", 1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5",
              6: "S1", 7: "S2"}


def level_bin(level):
    """Continuous vertebral level -> the integer bin whose interval [k-0.5, k+0.5) holds it.

    This is the same half-open rule as the at-L5 rate printed beside the histogram,
    `(ax >= 4.5) & (ax < 5.5)`, the figure compared with the literature (Moussallem 2012, 65.75 %),
    so level 4.5 is L5 in both the histogram and the rate. Python's `round()` rounds half to even
    and would disagree with that rule at every `k.5` with `k` even.
    """
    return int(math.floor(float(level) + 0.5))


def _level_label(level):
    """Continuous vertebral level -> coarse label (integer=body 'L4', half=disc 'L4/L5')."""
    if level is None or (isinstance(level, float) and not math.isfinite(level)):
        return "NA"
    lo = int(math.floor(level + 0.25))
    hi = int(math.floor(level + 0.75))
    if lo == hi:
        return LEVEL_NAME.get(lo, f"?{lo}")
    return f"{LEVEL_NAME.get(lo, '?')}/{LEVEL_NAME.get(hi, '?')}"


def read_vertebra_anchors(seg_dir):
    """[(level_index, anchor_z_mm), ...] for the lumbar vertebrae (+ sacrum) present in
    seg_dir, anchored at each vertebra's z-extent midpoint; the sacrum is anchored at
    its TOP (~S1 promontory) only when vertebrae_S1 is absent. Robust to a missing or
    unreadable mask (skipped). Empty list when no vertebra masks exist."""
    anchors = []
    for name, lvl in VERT_LEVELS:
        p = seg_dir / f"{name}.nii.gz"
        if not p.exists():
            continue
        try:
            img, arr = _load(p)
            if not arr.any():
                continue
            z_lo, z_hi, _ = _z_extent(img, arr)
        except Exception:
            continue
        anchors.append((lvl, 0.5 * (z_lo + z_hi)))
    # have_s1 comes from the anchors actually ADDED: an empty or unreadable vertebrae_S1 adds none,
    # so the sacrum fallback still fires, as the docstring promises ("robust to a missing or
    # unreadable mask").
    have_s1 = any(lvl == 6.0 for lvl, _ in anchors)      # 6.0 is S1 in VERT_LEVELS
    if not have_s1 and (seg_dir / "sacrum.nii.gz").exists():
        try:
            img, arr = _load(seg_dir / "sacrum.nii.gz")
            if arr.any():
                z_lo, z_hi, _ = _z_extent(img, arr)
                # The sacrum TOP is the S1 superior endplate = the L5/S1 disc, i.e. level 5.5 in the
                # body-MIDPOINT convention this function uses for L1..L5 (L5 body centre = 5.0, S1
                # body centre = 6.0, their boundary = 5.5). Fallback only fires when vertebrae_S1 is
                # absent.
                anchors.append((5.5, z_hi))      # sacrum top = L5/S1 disc (midpoint-consistent)
        except Exception:
            pass
    return anchors


def build_vert_ruler(anchors):
    """(interp_fn|None, z_min, z_max, qc) from [(level, anchor_z)]. interp maps a
    physical z to a continuous vertebral level (piecewise-linear on the anchors).
    qc: OK | NO_VERT | FEW_VERT | NONMONOTONIC. Requires z to DECREASE with level
    (LPS +z cranial: L1 highest z -> S1 lowest); a non-monotone order signals a
    TotalSegmentator level mislabel and is flagged rather than trusted."""
    if not anchors:
        return None, float("nan"), float("nan"), "NO_VERT"
    if len(anchors) < 2:
        return None, float("nan"), float("nan"), "FEW_VERT"
    a = sorted(anchors, key=lambda t: t[0])              # by level ascending
    levels = np.array([t[0] for t in a], float)
    zs = np.array([t[1] for t in a], float)
    if not np.all(np.diff(zs) < 0):                      # z must strictly decrease
        return None, float("nan"), float("nan"), "NONMONOTONIC"
    xp = zs[::-1]                                         # ascending z for np.interp
    fp = levels[::-1]

    def interp(z):
        if not (isinstance(z, (int, float)) and math.isfinite(z)):
            return float("nan")
        return float(np.interp(z, xp, fp))
    # A dropped MID-chain vertebra (e.g. L5 absent between L4 and S1 — a realistic TotalSegmentator
    # per-vertebra failure) is otherwise interpolated ACROSS the ~2-body span, misplacing a
    # continuous level by up to a full body, yet would still read qc='OK' and feed the published
    # at-L5 rate. Flag it SPARSE_VERT: the interp is still returned (callers can localize) but
    # summarize()'s vert_qc=='OK' filter then excludes it from the rate rather than trusting it.
    qc = "SPARSE_VERT" if float(np.max(np.diff(levels))) > 1.0 + 1e-6 else "OK"
    return interp, float(zs.min()), float(zs.max()), qc


def z_to_vert_level(ruler, z):
    """(level, in_range_bool) for a physical z given build_vert_ruler() output."""
    interp, z_min, z_max, _ = ruler
    if interp is None or not (isinstance(z, (int, float)) and math.isfinite(z)):
        return float("nan"), False
    return interp(z), bool(z_min <= z <= z_max)


SUPPORTED_APEX_METHODS = ("wallgap", "tangency")


def require_apex_method(apex_method):
    if apex_method not in SUPPORTED_APEX_METHODS:
        raise ValueError(
            f"apex_method {apex_method!r} is not an implemented apex method; expected one of "
            f"{', '.join(repr(m) for m in SUPPORTED_APEX_METHODS)} "
            f"(i.e. 'wallgap' or 'tangency').")
    return apex_method


def alt_apex_or_nan(z_alt, z_lo, z_hi):
    """The ALTERNATE apex, or NaN when it ran outside the physically plausible band.

    The primary CI is clamped and then forced to `WIDE_CI` when it overran, so the case is
    excluded rather than leaking a clamped-but-real-looking interval. The alternate apex has no
    overrun determination of its own (`ci_overran` is computed only from the primary interval), so
    clamping it would pull an out-of-band value to the bound and into the wallgap-vs-tangency
    agreement statistic, where it would pull bias/LoA/CCC toward the bound as if it were an
    estimate.

    NaN is how the rest of this module says "not usable", and `summarize`'s own `_finite` filter
    drops the row from the cross-check. A value ON the bound is inside the band and is kept.
    """
    if not (isinstance(z_alt, float) and math.isfinite(z_alt)):
        return float("nan")
    return float(z_alt) if z_lo <= z_alt <= z_hi else float("nan")


def measure_bif_one(case_dir, segdir=SEGDIR, window_mm=None, min_slices=None,
                    apex_method=None):
    window_mm = C.BIF_WINDOW_MM if window_mm is None else window_mm
    min_slices = C.BIF_MIN_SLICES if min_slices is None else min_slices
    apex_method = require_apex_method(
        C.BIF_APEX_METHOD if apex_method is None else apex_method)
    from scipy.spatial import cKDTree
    from scipy.ndimage import binary_dilation

    seg = case_dir / segdir
    ivc_p, l_p, r_p = seg / IVC, seg / LIL, seg / RIL
    if not (ivc_p.exists() and l_p.exists()):
        return {"case": case_dir.name, "flag": "MISSING"}

    ivc_img, ivc = _load(ivc_p)
    l_img, larr = _load(l_p)
    if not _same_grid(ivc_img, l_img):
        return {"case": case_dir.name, "flag": "GRID_MISMATCH"}
    # OBLIQUE QC: flag a non-signed-permutation direction matrix (tilted
    # gantry), where physical-z is coupled to the in-plane (i,j); consistent with the
    # OBLIQUE flag in measure_ivc_lciv_gap.py. Surfaced as the per-case `oblique` field.
    oblique = 1 if is_oblique(ivc_img) else 0
    if not ivc.any() or not larr.any():
        return {"case": case_dir.name, "flag": "EMPTY", "oblique": oblique}

    ivc_bottom, _, ivc_ks = _z_extent(ivc_img, ivc)
    _, l_top, l_ks = _z_extent(l_img, larr)

    has_r = r_p.exists()
    rarr = r_img = r_ks = r_top = None
    if has_r:
        r_img, rarr = _load(r_p)
        if not _same_grid(ivc_img, r_img):
            return {"case": case_dir.name, "flag": "GRID_MISMATCH",
                    "oblique": oblique}
        if rarr.any():
            _, r_top, r_ks = _z_extent(r_img, rarr)
            # iliac_top (defines z_gap/apex_in_gap) is the cranial-most of EITHER
            # vein; the fit below uses only slices shared by BOTH veins. If one
            # vein extends a few slices beyond the other those top slices set
            # iliac_top but are not in the fit support — an intentional asymmetry.
            iliac_top = max(l_top, r_top)
        else:
            has_r = False
            iliac_top = l_top
    else:
        iliac_top = l_top

    out = {"case": case_dir.name, "ivc_bottom": ivc_bottom,
           "iliac_top": iliac_top, "z_gap": ivc_bottom - iliac_top,
           "apex_method": apex_method, "oblique": oblique}

    # alpha2 (LCIV-IVC) and alpha1 (RCIV-IVC) bifurcation angles (secant-3D), measured
    # over the confluence window: the IVC's caudal-most window vs each CIV's cranial-most
    # window. Independent of the apex fit, so emitted for every case (incl. NO_RIGHT).
    ivc_cent = _axis_centroids(ivc_img, ivc, ivc_ks, ivc_bottom, window_mm, cranial=True)
    out["alpha2_deg"] = bifurcation_angle_deg(
        ivc_cent, _axis_centroids(l_img, larr, l_ks, l_top, window_mm, cranial=False))
    out["alpha1_deg"] = (bifurcation_angle_deg(
        ivc_cent, _axis_centroids(r_img, rarr, r_ks, r_top, window_mm, cranial=False))
        if has_r else float("nan"))

    if not has_r:
        out["flag"] = "NO_RIGHT"
        return out

    sx, sy, _sz = l_img.GetSpacing()
    vox_area = float(sx) * float(sy)
    both = sorted(set(l_ks.tolist()) & set(r_ks.tolist()))
    # contact_actual = sep - d_min is the centroid separation at which the two walls WOULD TOUCH:
    # d_min reaches 0 exactly at contact, where contact_actual = sep. Shape- and orientation-aware,
    # NO circular assumption: the 'wallgap' (PRIMARY) contact distance. It is not a reach along one
    # direction: on voxelized oblique rectangles at 1 mm, two parallel 45-degree lumens offset along
    # x read 23.029 mm against a directional reach of 34.000 (-32.3%), and a 60/20-degree pair
    # reads 27.917 against 34.000 (-17.9%). "When do the walls first touch" is a minimum over all
    # directions, which R = median(sep - d_min) solves for (statlib.wall_contact_apex). multi = 1 if
    # either vein has >1 connected mask component.
    pts = []
    for k in both:
        # The FULL masks still feed _cc_count below: multi-component topology is exactly what the
        # VARIANT signal must see, so it must NOT be hardened.
        l_cc = _largest_cc_slice(larr[k])
        r_cc = _largest_cc_slice(rarr[k])
        if l_cc is None or r_cc is None:
            continue
        lys, lxs = np.nonzero(l_cc)
        rys, rxs = np.nonzero(r_cc)
        cl, cr = _centroid_phys(l_img, larr, k), _centroid_phys(r_img, rarr, k)
        if not (cl and cr):
            continue
        # Use full physical 3-D distance. On a tilted acquisition the two centroids
        # from one array-k plane can have different physical z; truncating to x/y
        # would understate both centroid and wall distances.
        sep = float(np.linalg.norm(np.asarray(cl, float) - np.asarray(cr, float)))
        r_l = math.sqrt(len(lxs) * vox_area / math.pi)
        r_r = math.sqrt(len(rxs) * vox_area / math.pi)
        # d_min and sep MUST share one coordinate convention: sep is built from _centroid_phys
        # (TransformContinuousIndexToPhysicalPoint -> physical x,y, so direction-cosine aware), so
        # contact_actual = sep - d_min only makes sense if d_min is in the SAME physical in-plane
        # frame. Map each boundary voxel through the same physical transform and use all (x,y,z)
        # components. For axis-aligned (identity-direction) images physical = origin + index*spacing
        # and the origin cancels in cKDTree differences, so the result equals the index*spacing form
        # there (verified in tests); on oblique or rotated volumes only the physical form is
        # correct. Dilating one cross-section by a full 3x3 voxel
        # neighbourhood converts nearest voxel-centre distance to nearest voxel-BOX surface
        # distance: face-adjacent masks are 0, one empty voxel is one spacing unit, and a diagonal
        # gap uses the Euclidean corner distance. This matches the Phase-1 surface_gap
        # convention and avoids calling centre distances "walls".
        l_dil = binary_dilation(l_cc, structure=np.ones((3, 3), bool))
        ldys, ldxs = np.nonzero(l_dil)
        lc = np.array([l_img.TransformContinuousIndexToPhysicalPoint(
            (float(x), float(y), float(k))) for y, x in zip(ldys, ldxs)], float)
        rc = np.array([r_img.TransformContinuousIndexToPhysicalPoint(
            (float(x), float(y), float(k))) for y, x in zip(rys, rxs)], float)
        dmin = float(cKDTree(lc).query(rc)[0].min())
        multi = 1.0 if (_cc_count(larr[k]) > 1 or _cc_count(rarr[k]) > 1) else 0.0
        pts.append((0.5 * (cl[2] + cr[2]), sep, r_l + r_r, sep - dmin, multi))

    if len(pts) < min_slices:
        out["flag"] = "FEW_SLICES"
        out["fit_n"] = len(pts)
        return out

    pts = np.array(pts)
    z_top = pts[:, 0].max()
    cranial = pts[pts[:, 0] >= z_top - window_mm]
    window_fallback = len(cranial) < min_slices
    if window_fallback:
        # cranial-most min_slices points (apex-relevant; never caudal wide lumens)
        win = pts[np.argsort(-pts[:, 0])][:min_slices]
    else:
        win = cranial
    zz, sep_w = win[:, 0], win[:, 1]
    variant = bool(np.mean(win[:, 4]) >= C.BIF_VARIANT_FRAC)

    tang = S.wall_contact_apex(zz, sep_w, win[:, 2], alpha=C.ALPHA,
                               n_boot=C.BIF_NBOOT, seed=C.SEED,
                               slope_tol=C.BIF_SLOPE_TOL, ci_method=C.CI_METHOD,
                               block=C.BIF_BLOCK_SLICES)
    wall = S.wall_contact_apex(zz, sep_w, win[:, 3], alpha=C.ALPHA,
                               n_boot=C.BIF_NBOOT, seed=C.SEED,
                               slope_tol=C.BIF_SLOPE_TOL, ci_method=C.CI_METHOD,
                               block=C.BIF_BLOCK_SLICES)
    primary, alt = (tang, wall) if apex_method == "tangency" else (wall, tang)

    curv = S.quadratic_significance(zz, sep_w, C.BIF_CURV_ALPHA)
    ci_hw = 0.5 * (primary["ci_hi"] - primary["ci_lo"])
    shift = _quad_apex_shift(zz, sep_w, primary.get("median_r", float("nan")),
                             primary["z_bif"])
    # NONLINEAR materiality. _quad_apex_shift returns +inf when the quadratic fit has NO real root
    # at the contact level R (the curve never reaches contact) — the WORST-case curvature, where
    # linear extrapolation TO R is unbounded-ly wrong. A FINITE shift is material only when
    # curvature is ALSO statistically significant AND the quadratic apex leaves the linear apex's
    # own CI (mild significant curvature that does not move the apex past its CI is not material). A
    # NaN shift (uncomputable: <4 pts / flat quadratic) leaves materiality undetermined and is not
    # flagged on its own.
    curvature_assessable = curv.get("status") == "ok"
    nonlinear = _nonlinear_flag(curv["significant"], shift, ci_hw)

    # window-size sensitivity: cheap point-apex (no bootstrap) at 15/20/25/30 mm,
    # so the reader can see how stable z_bif is to the (fixed) window choice.
    contact_col = 2 if apex_method == "tangency" else 3
    sweep = []
    for w in (15.0, 20.0, 25.0, 30.0):
        cw = pts[pts[:, 0] >= z_top - w]
        if len(cw) < min_slices:
            cw = pts[np.argsort(-pts[:, 0])][:min_slices]
        sweep.append(_point_apex(cw[:, 0], cw[:, 1], cw[:, contact_col]))
    sweep = [v for v in sweep if math.isfinite(v)]
    window_stability = (max(sweep) - min(sweep)) if len(sweep) >= 2 else float("nan")

    z_gap = ivc_bottom - iliac_top

    # When the bootstrap slope is near zero the apex is a ratio with a near-zero denominator, so the
    # BCa endpoints can blow up to absurd values (an endpoint can run to thousands of
    # millimetres). Bound everything to the case's shared-slice z-extent padded by
    # one fit window; a CI that ran past that bound was never trustworthy, so FORCE WIDE_CI on it so
    # the case is excluded from apex stats rather than leaking a clamped-but-real-looking interval.
    # (z_bif itself is intentionally NOT clamped here — it is gated by the flag; only the reported
    # CI/alt bounds are bounded.)
    z_lo_b = float(pts[:, 0].min()) - window_mm
    z_hi_b = float(pts[:, 0].max()) + window_mm

    def _clamp(v):
        if not (isinstance(v, float) and math.isfinite(v)):
            return v
        return min(z_hi_b, max(z_lo_b, v))
    raw_lo, raw_hi = primary["ci_lo"], primary["ci_hi"]
    ci_lo_c, ci_hi_c = _clamp(raw_lo), _clamp(raw_hi)
    # It is refused (NaN), not clamped: a clamped alternate apex is indistinguishable from a real
    # one in the published table -- the bound is a per-case quantity no output column records --
    # and it feeds the wallgap-vs-tangency agreement statistic, which a bound-valued point pulls
    # toward the bound.
    z_bif_alt_c = alt_apex_or_nan(alt["z_bif"], z_lo_b, z_hi_b)
    # a finite raw bound outside [z_lo_b, z_hi_b] means the un-clamped CI was
    # impossible -> the apex is untrustworthy, force exclusion via WIDE_CI.
    ci_overran = ((math.isfinite(raw_lo) and (raw_lo < z_lo_b or raw_lo > z_hi_b)) or
                  (math.isfinite(raw_hi) and (raw_hi < z_lo_b or raw_hi > z_hi_b)))

    flag = decide_flag(primary["slope"], primary["unstable"],
                       ci_lo_c, ci_hi_c, z_gap, len(win),
                       variant=variant, nonlinear=nonlinear,
                       curvature_assessable=curvature_assessable)
    if ci_overran and flag in ("OK", "OK_OVERLAP"):
        flag = "WIDE_CI"
    out.update(slope=primary["slope"], slope_se=primary["slope_se"],
               fit_n=int(len(win)),
               z_bif=primary["z_bif"], z_bif_ci_lo=ci_lo_c,
               z_bif_ci_hi=ci_hi_c, z_bif_alt=z_bif_alt_c,
               # iliac_r_mm always comes from the tangency fit, in wallgap runs too.
               iliac_r_mm=0.5 * tang["median_r"],
               window_fallback=bool(window_fallback),
               window_stability_mm=window_stability,
               conv_angle_deg=_conv_angle_deg(primary["slope"]),
               curvature_status=curv.get("status", "unknown"),
               curvature_error_type=curv.get("error_type", ""),
               flag=flag)
    if flag in ("OK", "OK_OVERLAP"):
        out.update(apex_relative(ivc_bottom, iliac_top, primary["z_bif"]))
    # vertebral-level localization (literature anchor; absent vertebrae -> NA columns,
    # vert_qc=NO_VERT). gap_top/gap_bottom levels are computed for every case with a
    # ruler; apex_level only when the apex itself is usable (OK/OK_OVERLAP).
    anchors = read_vertebra_anchors(seg)
    ruler = build_vert_ruler(anchors)
    _, _, _, vqc = ruler
    # The gap spans [iliac_top, ivc_bottom] with z increasing CRANIALLY, so ivc_bottom is its roof
    # (top) and iliac_top its floor (bottom). The OUT_OF_FOV check below is symmetric and so is
    # unaffected.
    gap_top_lvl, in_top = z_to_vert_level(ruler, ivc_bottom)
    gap_bot_lvl, in_bot = z_to_vert_level(ruler, iliac_top)
    # `np.interp` CLAMPS, and the OUT_OF_FOV test below looks only at the two gap endpoints. The
    # apex is an EXTRAPOLATION beyond the cranial-most CIV slice and can sit outside the ruler while
    # both gap endpoints sit inside it, so its level is REFUSED (NaN), NOT CLAMPED, the same rule the
    # alternate apex follows above. `summarize` filters `apex_level` on `_finite`, so the case
    # leaves the at-L5 rate; `vert_qc` is deliberately left OK, because the two GAP levels are in
    # range and remain valid for that case, and `flag` stays OK, so `flag == OK` with a non-finite
    # `apex_level` is this condition's signature.
    apex_lvl = float("nan")
    if flag in ("OK", "OK_OVERLAP"):
        apex_lvl, in_apex = z_to_vert_level(ruler, primary["z_bif"])
        if not in_apex:
            apex_lvl = float("nan")
    if vqc == "OK" and not (in_top and in_bot):
        vqc = "OUT_OF_FOV"          # confluence z's fall outside the labeled vertebrae
    out.update(vert_n=len(anchors), gap_top_level=gap_top_lvl,
               gap_bottom_level=gap_bot_lvl, apex_level=apex_lvl, vert_qc=vqc)
    return out


def _init_worker(data_dir):
    """Relying on the inherited global only works under fork; spawn re-imports the module with
    DATA=None and every case TypeErrors."""
    # Pin BLAS/OpenMP to one thread PER WORKER. It has to happen here rather than at import: a
    # forked worker already has numpy loaded, so only a runtime limiter reaches the library that is
    # actually running.
    HR.limit_blas_threads(1)
    global DATA
    DATA = Path(data_dir)


def work(case):
    try:
        return measure_bif_one(DATA / case)
    except Exception as e:                               # never kill the batch
        traceback.print_exc(file=sys.stderr)            # keep the trace for debug
        return {"case": case, "flag": f"ERROR:{type(e).__name__}", "err": str(e)}


# ---------------------------------------------------------------------------
# Output formatting + summary
# ---------------------------------------------------------------------------
def _fmt(r):
    def g(k, nd=1):
        # math.isfinite, not a NaN test: an infinity formats as text (`f"{float('-inf'):.1f}"` is
        # `'-inf'`) that float() parses back into a number, so it is written NA, matching `_finite`.
        v = r.get(k)
        if isinstance(v, (int, float)) and math.isfinite(v):
            return f"{v:.{nd}f}"
        return "NA"
    wf = str(int(r["window_fallback"])) if "window_fallback" in r else "NA"
    ob = str(int(r["oblique"])) if "oblique" in r else "NA"
    return " ".join([
        r["case"], g("ivc_bottom"), g("iliac_top"), g("z_gap"), g("z_bif"),
        g("z_bif_ci_lo"), g("z_bif_ci_hi"), g("z_bif_alt"),
        str(r.get("apex_method", "NA")), g("slope", 3), g("slope_se", 3),
        g("ivc_overshoot"), g("iliac_undershoot"), g("gap_center_offset"),
        g("frac_above", 2), g("apex_in_gap", 0), g("iliac_r_mm", 1),
        g("alpha1_deg", 1), g("alpha2_deg", 1),
        str(r.get("fit_n", "NA")), wf,
        g("vert_n", 0), g("gap_top_level", 2), g("gap_bottom_level", 2),
        g("apex_level", 2), str(r.get("vert_qc", "NA")),
        r.get("flag", "NA"),
        # trailing fields; order matches COLUMNS
        g("window_stability_mm", 2), g("conv_angle_deg", 1), ob,
        str(r.get("curvature_status", "NA")),
        str(r.get("curvature_error_type", "") or "NA")])


def _print_apex_bounds(s):
    """The apex-in-gap rate's worst-case range and the rejected estimates' diagnostic."""
    lo, hi = s["apex_in_gap_bounds"]
    why = {"outside_frame": "a row with an apex_in_gap outcome has no positive z_gap, so the rate "
                            "above is not a rate over that frame",
           "accepted_without_outcome": "an accepted z_gap>0 row has no apex_in_gap outcome, which "
                                       "apex_relative always gives, so the counts disagree",
           "not_binary": "an apex_in_gap outcome is not 0 or 1, so the rate above is not a "
                         "proportion's"}.get(s["apex_in_gap_bounds_refused"])
    if why is None and math.isnan(lo):
        why = "no case has z_gap > 0"
    if why is not None:
        print(f"     worst-case range over every z_gap>0 case: not computed ({why})")
        return
    n_frame, n_eval = s["n_apex_frame"], s["n_apex_eval"]
    without = ", ".join(f"{k}={v}" for k, v in sorted(s["apex_unevaluated_flags"].items()))
    print(f"     over all {n_frame} z_gap>0 cases, assuming nothing about the "
          f"{n_frame - n_eval} without an accepted fit ({without or 'none'}): "
          f"{100*lo:.1f}% to {100*hi:.1f}% (worst-case bound: an identification range, "
          f"not a confidence interval)")
    u = s["n_apex_frame_unknown"]
    if u:
        lo_u, hi_u = s["apex_in_gap_bounds_unknown"]
        print(f"     with the {u} case{'s that have' if u != 1 else ' that has'} no z_gap, whose "
              f"membership of the z_gap>0 frame is unknown, counted as possible members: "
              f"{100*lo_u:.1f}% to {100*hi_u:.1f}%")
    fitted = s["n_apex_rejected_fitted"]
    if fitted and n_eval:
        inside = s["n_apex_rejected_inside"]
        print(f"     diagnostic, not a rate: the apex the gate rejected as unreliable falls in the "
              f"gap in {inside} of {fitted} ({100*inside/fitted:.1f}%) cases with a fitted apex, "
              f"against {s['n_apex_in_gap']} of {n_eval} "
              f"({100*s['n_apex_in_gap']/n_eval:.1f}%) accepted fits")


def print_summary(rows):
    s = summarize(rows)
    n = s["n_total"]
    print("\n=== confluence-proxy summary ===")
    print(f"cases: {n}   flags: " +
          ", ".join(f"{k}={v}" for k, v in sorted(s["flags"].items())))
    ok = s["n_ok_like"]
    if ok == 0:
        print("no OK cases — check that right-iliac masks exist and converge.")
        return

    def _ci(lo, hi, nd, **kw):
        # Every interval this summary prints goes through `statlib.format_ci`, so an interval
        # collapsed to a point (a median on an atom of whole-slice values) prints as the token; a
        # real interval keeps this site's text.
        return S.format_ci(lo, hi, nd, **kw) or "NA"

    def line(label, m, ci, unit="mm"):
        if math.isnan(m):
            print(f"  {label}: NA")
        else:
            print(f"  {label}: {m:+.1f} {unit} (95%CI {_ci(ci[0], ci[1], 1)})")

    print(f"  apex recovered (OK/OK_OVERLAP): {ok}/{n} ({100*ok/n:.0f}%)   "
          f"PRIMARY estimand: {C.BIF_APEX_METHOD} "
          f"({'shape-agnostic mask-to-mask' if C.BIF_APEX_METHOD=='wallgap' else 'equiv-circle tangency'}); "
          f"median lumen r={s['median_iliac_r_mm']:.1f} mm; CI={C.CI_METHOD}")
    print("  NOTE: the apex z is a geometry-only proxy, not an expert-referenced "
          "definition. The study collects no reference for it, so no apex agreement "
          "figure is reported; the optional `--reference` comparison applies only to a "
          "reference CSV the user supplies.")
    excl = ", ".join(f"{k}={v}" for k, v in sorted(s["flags"].items())
                     if k not in ("OK", "OK_OVERLAP"))
    print(f"  excluded: {excl or 'none'}")
    # HEADLINE: lead with the SIGNED offset of the unlabeled gap relative to the
    # apex PROXY and the apex-in-gap rate, NOT a reassuring "centered %". The apex is
    # an automated, externally-UNVALIDATED proxy, so we report where the gap actually
    # sits relative to it (sign + CI) and let the reader judge.
    m_off, ci_off = s["median_offset"], s["ci_offset"]
    # The direction is derived ONCE, here, and every sentence below that names it uses this
    # variable.
    gap_direction = (None if math.isnan(m_off) else
                     "cranial" if m_off > 0 else "caudal" if m_off < 0 else "level with")
    if not math.isnan(m_off):
        # offset = gap-center - apex; z increases CRANIALLY (see apex_relative),
        # so offset>0 => gap-center is at LARGER z => CRANIAL of the apex (the apex sits
        # caudal/below the gap-center); offset<0 => CAUDAL. State the direction in words.
        print(f"  SIGNED gap offset vs apex PROXY: median {m_off:+.1f} mm "
              f"(95%CI {_ci(ci_off[0], ci_off[1], 1, sep=',', signed=True)}); the unlabeled gap sits a "
              f"median ~{abs(m_off):.1f} mm {gap_direction} of the apex.")
        print(f"     apex is an automated PROXY and is not externally validated; the study "
              f"collects no reference for it. "
              f"(+offset = gap-center cranial of apex, -offset = caudal). "
              f"n={s.get('n_offset_eval', 0)} z_gap>0 cases whose apex fit was accepted (flag OK).")
    else:
        print("  SIGNED gap offset vs apex PROXY: NA (no z_gap>0 case with an accepted apex fit)")
    if s["n_apex_eval"]:
        if gap_direction in ("cranial", "caudal"):
            where = (f"in THESE data the gap centre sits a median {abs(m_off):.1f} mm "
                     f"{gap_direction} of the apex proxy, not centred on it")
        else:
            where = ("the offset is not defined on this run, so no direction is asserted "
                     "here")
        print(f"  apex falls INSIDE the labeled gap: {s['n_apex_in_gap']}/{s['n_apex_eval']} "
              f"({100*s['n_apex_in_gap']/s['n_apex_eval']:.0f}%) of the z_gap>0 cases whose apex "
              f"fit was accepted (flag OK; apex_in_gap rate; {where})")
    elif s["n_apex_frame"]:
        print(f"  apex falls INSIDE the labeled gap: not estimable, none of the "
              f"{s['n_apex_frame']} z_gap>0 cases has an accepted apex fit (flag OK)")
    if s["n_apex_frame"] or s["n_apex_eval"]:
        _print_apex_bounds(s)
    # supporting (raw, gap>0 only): overshoot / undershoot magnitudes
    line("median IVC overshoot above apex", s["median_overshoot"], s["ci_overshoot"])
    line("median iliac undershoot below apex", s["median_undershoot"], s["ci_undershoot"])
    # confluence-as-ZONE (literature treats it as a region, not a point)
    if not math.isnan(s["median_zone_mm"]):
        print(f"  confluence transition zone (unlabeled gap) height: median "
              f"{s['median_zone_mm']:.1f} mm (95%CI {_ci(s['ci_zone'][0], s['ci_zone'][1], 1)}); "
              f"literature junction height ~{C.BIF_LIT_ZONE_MM:.1f} mm (Pirro 2005)")
    # VERTEBRAL-LEVEL localization (literature anchor; Moussallem L5 65.75% / Pirro L4-S1).
    # Distributional comparison only — NEVER a per-case reference standard. Over OK cases whose
    # vertebral ruler is valid (vert_qc=OK).
    from collections import Counter
    vrows = [r for r in rows if r.get("flag") in ("OK", "OK_OVERLAP")
             and r.get("vert_qc") == "OK"]
    ax = np.array([r["apex_level"] for r in vrows if _finite(r.get("apex_level"))], float)
    gt = np.array([r["gap_top_level"] for r in vrows if _finite(r.get("gap_top_level"))], float)
    gb = np.array([r["gap_bottom_level"] for r in vrows if _finite(r.get("gap_bottom_level"))], float)
    nv_ok = sum(1 for r in rows if r.get("vert_qc") == "OK")
    nv_bad = sum(1 for r in rows if r.get("vert_qc") in
                 ("NONMONOTONIC", "OUT_OF_FOV", "FEW_VERT"))
    nv_none = sum(1 for r in rows if r.get("vert_qc") in (None, "NO_VERT"))
    if ax.size:
        med = float(np.median(ax))
        hist = Counter(level_bin(v) for v in ax)  # same bin edges as pct_l5
        dist = "  ".join(f"{LEVEL_NAME.get(k, k)}:{100*hist[k]/ax.size:.0f}%"
                         for k in sorted(hist))
        pct_l5 = 100.0 * float(np.mean((ax >= 4.5) & (ax < 5.5)))
        print(f"  VERTEBRAL-LEVEL apex (anatomical anchor, n={ax.size} vert_qc=OK): "
              f"median {med:.1f} = {_level_label(med)};  level distribution -> {dist}")
        print(f"     at-L5: {pct_l5:.0f}%  cf. confluence ~L5 (Moussallem 2012 65.75%; "
              f"Pirro 2005 L4-S1). COARSE anchor (~half a vertebral body ~{C.BIF_VALID_TOL_MM:.0f}mm)"
              f" -> a REGION-level comparison with the literature, not a validation of the sub-cm apex offset; automated, not blinded-expert.")
        if gt.size and gb.size:
            print(f"     unlabeled gap bracket (median): top {_level_label(float(np.median(gt)))}"
                  f" -> bottom {_level_label(float(np.median(gb)))}")
        if nv_bad:
            print(f"     vert QC excluded: {nv_bad} (NONMONOTONIC mislabel / OUT_OF_FOV / FEW_VERT)")
    elif nv_none and not nv_ok:
        print("  VERTEBRAL-LEVEL: unavailable (no vertebra masks). Re-run TotalSegmentator "
              "with vertebrae_L1..L5 + sacrum in the roi_subset to enable this anchor.")
    # wallgap-vs-tangency internal cross-check (bias/LoA are the headline; CCC/ICC
    # are on the landmark-RELATIVE apex, not absolute z which inflates them to ~1)
    cc = s.get("cross_check")
    if cc and cc.get("n"):
        print(f"  apex method cross-check (wallgap vs tangency, n={cc['n']}): "
              f"bias={cc['bias']:+.1f} mm  LoA[{cc['loa_lo']:+.1f},{cc['loa_hi']:+.1f}] "
              f"(the headline: circular-tangency offset)  "
              f"CCC={cc['ccc']:.2f} ICC={cc['icc_a1']:.2f} (on apex vs iliac_top; "
              f"absolute-z CCC is ~1.0 and uninformative)")
    # apex window-size sensitivity: the 15-30mm sweep is VACUOUS for
    # window_fallback cases (every window collapses onto the same cranial-most
    # min_slices points, so the spread is ~0 BY CONSTRUCTION, not by stability).
    # Report the fallback RATE and split the median spread by fallback status.
    wf_rate = s.get("window_fallback_rate", float("nan"))
    n_wf = s.get("n_window_fallback", 0)
    if not math.isnan(wf_rate):
        print(f"  window-fallback rate: {n_wf}/{ok} ({100*wf_rate:.0f}%) "
              f"(cranial window < {C.BIF_MIN_SLICES} slices -> sweep collapses to a "
              f"single point set; its spread is uninformative)")
    nf_n = s.get("n_window_stability_nonfallback", 0)
    fb_n = s.get("n_window_stability_fallback", 0)
    if nf_n:
        print(f"  window sensitivity (NON-fallback, n={nf_n}): median apex spread "
              f"across 15-30mm windows = "
              f"{s['median_window_stability_nonfallback_mm']:.1f} mm")
    if fb_n:
        print(f"  window sensitivity (fallback, n={fb_n}): median spread "
              f"{s['median_window_stability_fallback_mm']:.1f} mm "
              f"(UNINFORMATIVE — ~0 by construction; reported for completeness only)")
    if not math.isnan(s.get("median_alpha2_deg", float("nan"))):
        print(f"  alpha2 (LCIV-IVC bifurcation angle, SECANT-3D): median "
              f"{s['median_alpha2_deg']:.0f} deg "
              f"(95%CI {_ci(s['ci_alpha2'][0], s['ci_alpha2'][1], 0)}), n={s['n_alpha2']}; "
              f"alpha1 (RCIV-IVC) median {s['median_alpha1_deg']:.0f} deg. "
              f"cf. Chen 2023 alpha2 48.8 [25-90], alpha1 34.1 [18-65] (2-D coronal "
              f"projection; these are 3-D secants, so the same anatomy but not the same "
              f"number -- the SIGN of the difference is not predictable from geometry)")
    if not math.isnan(s.get("median_conv_angle_deg", float("nan"))):
        print(f"  median apparent FULL interiliac convergence angle "
              f"(angle between both CIVs, NOT alpha2 alone): "
              f"{s['median_conv_angle_deg']:.0f} deg — BIASED HIGH, not low, and not a "
              f"coronal projection: it is 2*atan(slope/2) on a "
              f"FULL 3-D separation, which assumes the two veins diverge symmetrically "
              f"and over-reads when they do not (up to +17.7 deg at 9:1 asymmetry). "
              f"cf. full-angle range Pirro 2005 ~69 deg [28-107], Chen 2023 "
              f"alpha1+alpha2 ~83 deg [45-130] — face validity only; the "
              f"asymmetry-free full angle is alpha1+alpha2 above)")
    if not math.isnan(s["median_frac_above"]):
        print(f"  (secondary) frac of gap above apex among apex-in-gap cases only: "
              f"{s['median_frac_above']:.2f} "
              f"(95%CI {_ci(s['ci_frac_above'][0], s['ci_frac_above'][1], 2)}); "
              f"n={s['n_frac_above']} — a fraction only when apex_in_gap=1")
    if not (math.isnan(s["med_zgap_ok"]) or math.isnan(s["med_zgap_excluded"])):
        print(f"  covariate comparison (describes the selection, bounds nothing): "
              f"median z_gap OK={s['med_zgap_ok']:.1f}mm vs "
              f"excluded={s['med_zgap_excluded']:.1f}mm "
              f"(a divergence says the OK subset differs, not how far an OK-only rate is off)")
    print("  NOTE: medians over OK cases only; VARIANT/NONLINEAR/NONCONVERGING/"
          "WIDE_CI/FEW_SLICES/NO_RIGHT are excluded and counted above. The apex is an "
          "extrapolated, externally-UNVALIDATED proxy; the headline is the SIGNED "
          "gap offset + apex-in-gap rate (not a 'centered %'), and all gap-relative "
          "quantities are conditional on z_gap>0 and on an accepted apex fit (flag OK); the "
          "apex-in-gap rate's worst-case range over every z_gap>0 case is printed beside it. "
          "The apex is not validated against an external reference; the study collects no "
          "reference for it.")


# --------------------------------------------------------------------------- Optional reference
# comparison: apex proxy vs a user-supplied reference (the study collects no reference for the
# apex).
def load_reference(path):
    """Read a reference CSV with columns: case, ref_z [, vert_level]. ref_z is a reference
    confluence z in the SAME LPS frame as the masks (mm).

    Refused: a duplicate case (two values for one case is not a reference, and picking one by
    file order is not a declared rule), and a file without the case/ref_z columns (an empty dict
    would read downstream as "insufficient matched reference cases", indistinguishable from a
    small reference set). Dropped and reported with a count: empty case ids, and non-finite
    ref_z (`float("nan")` and `float("inf")` both parse, so a ValueError handler alone would
    admit them).

    A duplicate RAISES: it is a defect in the input that no downstream default can repair.
    Non-finite and empty-id rows are dropped and REPORTED with a count, because a reference
    file legitimately may not have a value for every case, and silence about how many were
    dropped is what turns a partial reference into an apparently complete one.
    """
    ref, seen_dupes, dropped = {}, [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        missing = [c for c in ("case", "ref_z") if c not in cols]
        if missing:
            raise ValueError(
                f"reference CSV {path} is missing required column(s) {missing}; "
                f"found {cols}. An empty reference is reported as 'insufficient matched "
                f"cases', which reads as a small reference rather than the wrong file.")
        for row in reader:
            cid = (row.get("case") or "").strip()
            if not cid:
                dropped.append("<empty case id>")
                continue
            try:
                z = float(row["ref_z"])
            except (KeyError, TypeError, ValueError):
                dropped.append(cid)
                continue
            if not math.isfinite(z):
                dropped.append(cid)
                continue
            if cid in ref:
                seen_dupes.append(cid)
                continue
            ref[cid] = z
    if seen_dupes:
        raise ValueError(
            f"reference CSV {path} has more than one row for {sorted(set(seen_dupes))}. "
            f"A reference standard cannot hold two values for one case, and taking the last "
            f"row is a rule nobody declared.")
    if dropped:
        print(f"  [reference] dropped {len(dropped)} row(s) with a missing or non-finite "
              f"ref_z: {sorted(set(dropped))[:10]}")
    return ref


def validate_against_reference(rows, ref, tol_mm=None):
    """Agreement of the apex proxy z_bif vs a user-supplied reference over the
    cases present in both. Returns the agreement bundle (bias, LoA, CCC, ICC,
    within-tolerance fraction) or None if too few matched.

    For this to be an EXTERNAL comparison the reference must be produced blinded to the
    algorithm's output."""
    tol_mm = C.BIF_VALID_TOL_MM if tol_mm is None else tol_mm
    # normalize both by iliac_top so CCC/ICC reflect the few-mm agreement, not the
    # ~0.6 m between-patient table-position variance; bias/LoA are unaffected (the
    # landmark cancels in the per-case difference) and within_tol is on |Δz| in mm.
    # require a finite iliac_top so the landmark base is CONSISTENT across cases:
    # mixing landmark-relative (real iliac_top) and absolute-z (base 0.0) cases
    # corrupts the CCC/ICC. Real OK rows always carry iliac_top; this only
    # drops pathological rows, matching the _finite(iliac_top) guard in summarize().
    pairs = [(r["z_bif"], ref[r["case"]], r["iliac_top"])
             for r in rows if r.get("flag") in ("OK", "OK_OVERLAP")
             and _finite(r.get("z_bif")) and _finite(r.get("iliac_top"))
             and r.get("case") in ref]
    if len(pairs) < 2:
        return None
    base = np.array([p[2] for p in pairs], float)
    est = np.array([p[0] for p in pairs], float) - base
    gt = np.array([p[1] for p in pairs], float) - base
    out = S.method_agreement(est, gt, C.ALPHA, tol=tol_mm)
    # load_reference drops non-finite values, so the two counts should agree; this checks it rather
    # than assuming it, because the reference is not the only way a pair can go non-finite.
    out["n_matched"] = int(out.get("n", len(pairs)))
    if out["n_matched"] != len(pairs):
        print(f"  [reference] {len(pairs) - out['n_matched']} matched pair(s) were dropped as "
              f"non-finite before the agreement statistics; reporting the count actually used")
    return out


def print_validation(val):
    if val is None:
        print("\n=== external validation ===\n  insufficient matched reference "
              "cases (need >=2 OK cases with a reference z).")
        return
    print("\n=== external validation: apex proxy vs reference ===")
    print(f"  matched n={val['n_matched']}")
    print(f"  bias (proxy - ref): {val['bias']:+.1f} mm   "
          f"95% LoA param[{val['loa_lo']:+.1f},{val['loa_hi']:+.1f}] "
          f"non-param[{val['loa_lo_np']:+.1f},{val['loa_hi_np']:+.1f}]")
    print(f"  Lin's CCC={val['ccc']:.3f}   ICC(A,1)={val['icc_a1']:.3f}   "
          f"within {val.get('tol', float('nan')):.0f} mm: "
          f"{100*val.get('within_tol_frac', float('nan')):.0f}%")
    # No source establishes such a tolerance for a venous landmark, and the nearby real convention
    # (automated vertebra LABELING accuracy "within one vertebral level") answers a different
    # question.
    print(f"  (tolerance {C.BIF_VALID_TOL_MM:.0f} mm is this project's own uncalibrated\n"
          f"   choice, not a field standard; it gates nothing.)")


def main():
    global DATA
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True,
                    help="cohort dir, e.g. amos-data or saros-dataset")
    ap.add_argument("--prefix", required=True, help="output prefix, e.g. amos")
    ap.add_argument(
        "--jobs", type=int, default=None,
        help="worker processes. Default: resolved from the SLURM/cgroup CPU and MEMORY allocation, capped by the measured ~5.57 GiB per-worker peak. An explicit value is obeyed exactly.")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N cases (quick test)")
    ap.add_argument("--roster", type=Path, default=None,
                    help="exhaustive roster; defaults for prefix amos/saros/autopet")
    ap.add_argument("--reference", default=None,
                    help="optional CSV (case,ref_z[,vert]) of a reference confluence z to "
                         "compare with the apex proxy; none is supplied with this study")
    ap.add_argument("--ref-tol", type=float, default=C.BIF_VALID_TOL_MM,
                    help="within-tolerance (mm) for the validation hit-rate")
    args = ap.parse_args()
    DATA = Path(args.data_dir)
    if not DATA.is_dir():
        ap.error(f"cohort data directory does not exist: {DATA}")

    available = sorted(p.parent.parent.name for p in DATA.glob(f"*/{SEGDIR}/{IVC}"))
    expected = load_expected_roster(args.prefix, args.roster)
    cases = list(expected) if expected is not None else available
    if expected is not None:
        extras = sorted(set(available) - set(expected))
        if extras:
            print(f"ignoring {len(extras)} out-of-roster case dir(s): {extras[:10]}",
                  file=sys.stderr, flush=True)
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        sys.exit(f"no cases with {SEGDIR}/{IVC} under {DATA}")
    ref = load_reference(args.reference) if args.reference else None
    print(f"measuring confluence proxy for {len(cases)} cases in {DATA}",
          flush=True)

    # The limit is carried in the FILENAME, not merely in a marker inside the file, so the
    # artifact is recognisable as partial to the roster validator and a human running `ls`.
    out_path = results_path(args.prefix, "bifurcation", args.limit)
    results = []
    # AtomicTable publishes on success only, so an interrupted run leaves the previous complete
    # table untouched instead of a truncated one that looks like a smaller cohort.
    with AtomicTable(out_path) as table:
        table.header(" ".join(COLUMNS))   # a column header is not a data row
        # Resolve BEFORE the Pool: `Pool(None)` falls back to `os.cpu_count()`, which is the node,
        # not the allocation, and ignores memory entirely -- the exact oversubscription this
        # resolves. `--jobs N` is obeyed exactly; the default is the min of the visible CPU
        # allocation and what the memory budget affords at the measured per-worker peak.
        jobs = HR.resolve_worker_count(requested=args.jobs)
        print(f"[resources] {HR.describe()} -> {jobs} worker(s)", flush=True)
        with Pool(jobs, initializer=_init_worker,
                  initargs=(str(DATA),)) as pool:
            for i, r in enumerate(pool.imap(work, cases), 1):
                results.append(r)
                table.write(_fmt(r))
                if i % 25 == 0:
                    table.flush()
                    print(f"  {i}/{len(cases)}", flush=True)
        # POSTCONDITIONS, raised INSIDE the AtomicTable block so no `.part` is published
        # and any previous table stands. Every outcome of `measure_bif_one` that measured
        # anything sets `z_gap` (`NO_RIGHT` and `FEW_SLICES` included), and every outcome that
        # measured nothing (`MISSING`, `GRID_MISMATCH`, `EMPTY`, an `ERROR:` from `work`) omits
        # it, so the count does not depend on the flag vocabulary. The table exists to locate
        # the confluence, so a run in which no case yields an apex (`z_bif`) is refused too: one
        # absent right-iliac file per case is enough to produce it. Only an all-failure table is
        # refused: a partial failure rate is data.
        flags = collections.Counter(str(r.get("flag", "NA")) for r in results)
        tally = ", ".join(f"{f}={k}" for f, k in sorted(flags.items()))
        if not any(_finite(r.get("z_gap")) for r in results):
            hint = (f" This driver reads <case>/{SEGDIR}/ only, so a flat 2.17.0 export reads as "
                    f"MISSING for every case." if set(flags) == {"MISSING"} else "")
            raise SystemExit(
                f"{out_path}: nothing was measured -- none of the {len(results)} rows carries a "
                f"z_gap (flags: {tally}). The table was NOT published; any previous one "
                f"stands.{hint}")
        if not any(_finite(r.get("z_bif")) for r in results):
            raise SystemExit(
                f"{out_path}: no apex was estimated -- none of the {len(results)} rows carries a "
                f"z_bif (flags: {tally}), so the table holds no confluence estimate, which is what "
                f"it is for. The table was NOT published; any previous one stands. Check that each "
                f"case holds both iliac masks (NO_RIGHT) over enough shared slices (FEW_SLICES).")
    print(f"done -> {out_path}")
    print_summary(results)
    if ref is not None:
        print_validation(validate_against_reference(results, ref, args.ref_tol))


if __name__ == "__main__":
    main()
