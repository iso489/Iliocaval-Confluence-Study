"""Unit tests for measure_bifurcation.py: the wall-contact apex, the bounded apex_in_gap
indicator, flag gating, and the NaN-safe summary with selection-bias accounting."""
import math

import numpy as np
import pytest
import SimpleITK as sitk

import measure_bifurcation as MB


# --- apex_relative --------------------------------------------------------
def test_apex_relative_symmetric_case():
    # apex exactly at gap centre: overshoot == undershoot, offset 0, frac 0.5
    r = MB.apex_relative(ivc_bottom=10.0, iliac_top=0.0, z_bif=5.0)
    assert r["ivc_overshoot"] == pytest.approx(5.0)
    assert r["iliac_undershoot"] == pytest.approx(5.0)
    assert r["gap_center_offset"] == pytest.approx(0.0)
    assert r["frac_above"] == pytest.approx(0.5)
    assert r["apex_in_gap"] == 1.0


def test_apex_relative_apex_inside_gap_is_bounded_fraction():
    r = MB.apex_relative(ivc_bottom=13.0, iliac_top=11.0, z_bif=12.0)
    assert r["apex_in_gap"] == 1.0
    assert 0.0 <= r["frac_above"] <= 1.0


def test_apex_relative_apex_outside_gap_flagged():
    # apex caudal to the whole gap -> not in gap; frac would be >1 (meaningless)
    r = MB.apex_relative(ivc_bottom=13.0, iliac_top=11.0, z_bif=9.0)
    assert r["apex_in_gap"] == 0.0
    assert r["frac_above"] > 1.0


def test_apex_relative_overlap_makes_frac_above_nan():
    # z_gap <= 0 (labels overlap) -> frac_above and apex_in_gap undefined
    r = MB.apex_relative(ivc_bottom=5.0, iliac_top=8.0, z_bif=6.0)
    assert math.isnan(r["frac_above"])
    assert math.isnan(r["apex_in_gap"])


# --- alpha2 bifurcation angle (secant-3D; Chen 2023 / Li et al.) ----------
def test_bifurcation_angle_45_degrees():
    # IVC vertical (along +z); LCIV descends at 45 deg in the x-z plane -> alpha2 = 45
    zc = np.linspace(0.0, 20.0, 11)
    ivc = np.column_stack([np.zeros_like(zc), np.zeros_like(zc), zc])     # (x, y, z)
    t = np.linspace(0.0, 10.0, 11)
    lciv = np.column_stack([-t, np.zeros_like(t), -t])                    # dx<0, dz<0
    assert MB.bifurcation_angle_deg(ivc, lciv) == pytest.approx(45.0, abs=1.0)


def test_bifurcation_angle_collinear_is_zero():
    # LCIV a straight caudal continuation of the IVC -> alpha2 ~ 0
    z = np.linspace(0.0, 20.0, 11)
    ivc = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    lciv = np.column_stack([np.zeros_like(z), np.zeros_like(z), -z])
    assert MB.bifurcation_angle_deg(ivc, lciv) == pytest.approx(0.0, abs=1.0)


def test_bifurcation_angle_perpendicular_is_90():
    # LCIV running purely laterally (horizontal) -> alpha2 ~ 90
    z = np.linspace(0.0, 20.0, 11)
    ivc = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    t = np.linspace(0.0, 10.0, 11)
    lciv = np.column_stack([-t, np.zeros_like(t), np.zeros_like(t)])
    assert MB.bifurcation_angle_deg(ivc, lciv) == pytest.approx(90.0, abs=1.0)


def test_bifurcation_angle_is_translation_invariant():
    # the angle must not depend on where the vessels sit (only on direction)
    z = np.linspace(0.0, 20.0, 11)
    ivc = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    t = np.linspace(0.0, 10.0, 11)
    lciv = np.column_stack([-t, np.zeros_like(t), -t]) + np.array([100.0, 50.0, 30.0])
    assert MB.bifurcation_angle_deg(ivc, lciv) == pytest.approx(45.0, abs=1.0)


def test_bifurcation_angle_too_few_points_is_nan():
    assert math.isnan(MB.bifurcation_angle_deg(np.zeros((1, 3)), np.zeros((1, 3))))


# --- decide_flag ----------------------------------------------------------
def test_flag_few_slices():
    assert MB.decide_flag(-1.0, False, 0, 1, 5.0, n_shared=3) == "FEW_SLICES"


def test_flag_nonconverging_on_positive_slope():
    assert MB.decide_flag(+0.5, False, 0, 1, 5.0, n_shared=10) == "NONCONVERGING"


def test_flag_nonconverging_on_unstable():
    assert MB.decide_flag(-1.0, True, 0, 1, 5.0, n_shared=10) == "NONCONVERGING"


def test_flag_wide_ci():
    # half-width 12 mm > BIF_WIDE_CI_MM (10) -> WIDE_CI
    assert MB.decide_flag(-1.0, False, -12.0, 12.0, 5.0, n_shared=10) == "WIDE_CI"


def test_flag_ok_and_ok_overlap():
    assert MB.decide_flag(-1.0, False, -2.0, 2.0, 5.0, n_shared=10) == "OK"
    assert MB.decide_flag(-1.0, False, -2.0, 2.0, -3.0, n_shared=10) == "OK_OVERLAP"


def test_flag_variant_excludes():
    # persistent multi-component mask topology excludes even an otherwise-OK fit
    assert MB.decide_flag(-1.0, False, -2.0, 2.0, 5.0, n_shared=10,
                          variant=True) == "VARIANT"


def test_flag_nonlinear_excludes():
    assert MB.decide_flag(-1.0, False, -2.0, 2.0, 5.0, n_shared=10,
                          nonlinear=True) == "NONLINEAR"


def test_flag_curvature_fit_failure_excludes_instead_of_blessing_linearity():
    assert MB.decide_flag(
        -1.0, False, -2.0, 2.0, 5.0, n_shared=10,
        curvature_assessable=False) == "CURVATURE_UNASSESSABLE"


def test_flag_variant_takes_precedence_over_nonlinear():
    assert MB.decide_flag(-1.0, False, -2.0, 2.0, 5.0, n_shared=10,
                          variant=True, nonlinear=True) == "VARIANT"


def test_flag_wide_ci_follows_wide_ci_mm():
    # half-width 12 mm fails a 10 mm gate and passes a 15 mm one
    assert MB.decide_flag(-1.0, False, -12.0, 12.0, 5.0, n_shared=10,
                          wide_ci_mm=10.0) == "WIDE_CI"
    assert MB.decide_flag(-1.0, False, -12.0, 12.0, 5.0, n_shared=10,
                          wide_ci_mm=15.0) == "OK"


# --- summarize ------------------------------------------------------------
def test_summary_empty_is_safe():
    s = MB.summarize([{"case": "c", "flag": "EMPTY"}])
    assert s["n_ok_like"] == 0


def test_summary_excludes_nan_frac_above():
    rows = [
        {"case": "a", "flag": "OK", "ivc_overshoot": 2.0, "iliac_undershoot": 3.0,
         "gap_center_offset": 0.5, "frac_above": 0.4, "apex_in_gap": 1.0,
         "z_gap": 5.0, "iliac_r_mm": 6.0},
        {"case": "b", "flag": "OK_OVERLAP", "ivc_overshoot": 1.0,
         "iliac_undershoot": 1.0, "gap_center_offset": 0.0,
         "frac_above": float("nan"), "apex_in_gap": float("nan"),
         "z_gap": -2.0, "iliac_r_mm": 6.0},
    ]
    s = MB.summarize(rows)
    assert s["n_ok_like"] == 2
    assert s["n_frac_above"] == 1            # only the apex_in_gap==1 case counts


def test_summary_counts_apex_in_gap():
    rows = [
        {"case": "a", "flag": "OK", "apex_in_gap": 1.0, "z_gap": 5.0,
         "gap_center_offset": 0.5, "iliac_r_mm": 6.0},
        {"case": "b", "flag": "OK", "apex_in_gap": 0.0, "z_gap": 5.0,
         "gap_center_offset": 9.0, "iliac_r_mm": 6.0},
    ]
    s = MB.summarize(rows)
    assert s["n_apex_in_gap"] == 1 and s["n_apex_eval"] == 2


def _frame_rows(empty=0):
    """Six rows, five of them in the z_gap > 0 frame: three accepted fits (flag OK, one apex in the gap),
    a NONLINEAR and a WIDE_CI fit with no outcome, and an OK_OVERLAP row outside the frame. With
    `empty`, that many EMPTY rows with no z_gap at all, whose membership is therefore unknown.
    By hand: k = 1, n_eval = 3, N = 5, so [1/5, 3/5]; one EMPTY row gives [1/6, 4/6]."""
    rows = [
        {"case": "a", "flag": "OK", "z_gap": 5.0, "apex_in_gap": 1.0, "gap_center_offset": 0.5},
        {"case": "b", "flag": "OK", "z_gap": 4.0, "apex_in_gap": 0.0, "gap_center_offset": 6.0},
        {"case": "c", "flag": "OK", "z_gap": 3.0, "apex_in_gap": 0.0, "gap_center_offset": 7.0},
        {"case": "d", "flag": "NONLINEAR", "z_gap": 6.0, "apex_in_gap": float("nan")},
        {"case": "e", "flag": "WIDE_CI", "z_gap": 2.0},
        {"case": "f", "flag": "OK_OVERLAP", "z_gap": -2.0, "apex_in_gap": float("nan")},
    ]
    rows += [{"case": f"x{i}", "flag": "EMPTY"} for i in range(empty)]
    return rows


def test_summary_bounds_the_apex_rate_over_every_case_with_a_gap():
    """The printed rate is k of the accepted fits; the frame it speaks for is every case with
    z_gap > 0, so the summary carries the frame, the fits without an outcome by flag, and the range
    the frame's rate can take when nothing is assumed about them."""
    s = MB.summarize(_frame_rows())
    assert (s["n_apex_in_gap"], s["n_apex_eval"]) == (1, 3)
    assert s["n_apex_frame"] == 5 and s["n_apex_frame_unknown"] == 0
    assert s["apex_unevaluated_flags"] == {"NONLINEAR": 1, "WIDE_CI": 1}
    assert s["apex_in_gap_bounds"] == pytest.approx((0.2, 0.6))
    assert s["apex_in_gap_bounds_unknown"] == pytest.approx((0.2, 0.6))


def test_summary_counts_a_case_without_a_gap_measurement_as_of_unknown_membership():
    """SAROS's 26 EMPTY rows have no z_gap: they are not outside the frame, they are cases whose
    membership is unknown, so they widen the bound that counts them and leave the frame's alone."""
    s = MB.summarize(_frame_rows(empty=1))
    assert s["n_apex_frame"] == 5 and s["n_apex_frame_unknown"] == 1
    assert s["apex_in_gap_bounds"] == pytest.approx((0.2, 0.6))
    assert s["apex_in_gap_bounds_unknown"] == pytest.approx((1 / 6, 4 / 6))


def test_summary_refuses_the_bound_when_an_evaluated_row_lies_outside_the_frame():
    """An accepted row with an outcome but no positive z_gap contradicts `apex_relative` (the
    outcome is NaN unless z_gap > 0). The printed rate would then count a case the frame does not,
    and a bound on the frame would describe another rate; both ends refuse."""
    rows = _frame_rows() + [{"case": "g", "flag": "OK", "apex_in_gap": 1.0}]
    s = MB.summarize(rows)
    assert (s["n_apex_in_gap"], s["n_apex_eval"]) == (2, 4)
    assert all(math.isnan(x) for x in s["apex_in_gap_bounds"])
    assert all(math.isnan(x) for x in s["apex_in_gap_bounds_unknown"])


def test_print_summary_prints_the_bound_beside_the_rate_and_names_its_denominator():
    """The console's apex block: the rate is read over the accepted fits, the next line gives the
    range over every case with a gap, and the median comparison is read as what it is."""
    out = _console(_frame_rows())
    lines = out.splitlines()
    i = next(n for n, ln in enumerate(lines) if "apex falls INSIDE the labeled gap" in ln)
    assert "1/3" in lines[i] and "whose apex fit was accepted" in lines[i], lines[i]
    assert "over all 5 z_gap>0 cases" in lines[i + 1], lines[i + 1]
    assert "the 2 without an accepted fit" in lines[i + 1], lines[i + 1]
    assert "NONLINEAR=1, WIDE_CI=1" in lines[i + 1], lines[i + 1]
    assert "20.0% to 60.0%" in lines[i + 1], lines[i + 1]
    assert "not a confidence interval" in lines[i + 1], lines[i + 1]
    assert "membership" not in out, "no unknown-membership line without such a case"
    check = next(ln for ln in lines if "median z_gap OK=" in ln)
    assert "bounds nothing" in check, check

    wide = _console(_frame_rows(empty=1))
    unknown = next(ln for ln in wide.splitlines() if "membership" in ln)
    assert "the 1 case that have" not in unknown and "the 1 case " in unknown, unknown
    assert "16.7% to 66.7%" in unknown, unknown
    two = next(ln for ln in _console(_frame_rows(empty=2)).splitlines() if "membership" in ln)
    assert "the 2 cases that have no z_gap" in two and "14.3% to 71.4%" in two, two

    refused = _console(_frame_rows() + [{"case": "g", "flag": "OK", "apex_in_gap": 1.0}])
    j = next(n for n, ln in enumerate(refused.splitlines()) if "apex falls INSIDE" in ln)
    assert "not computed" in refused.splitlines()[j + 1], refused.splitlines()[j + 1]


def test_summary_refuses_the_bound_when_an_outcome_is_not_binary():
    """An outcome that is not 0 or 1 is no proportion's, so both ends refuse, with that reason, not the
    outside-the-frame one."""
    rows = _frame_rows()
    rows[1] = dict(rows[1], apex_in_gap=0.5)
    s = MB.summarize(rows)
    assert all(math.isnan(x) for x in s["apex_in_gap_bounds"])
    assert s["apex_in_gap_bounds_refused"] == "not_binary"
    out = _console(rows).splitlines()
    i = next(n for n, ln in enumerate(out) if "apex falls INSIDE" in ln)
    assert "not computed" in out[i + 1] and "not 0 or 1" in out[i + 1], out[i + 1]
    outside = MB.summarize(_frame_rows() + [{"case": "g", "flag": "OK", "apex_in_gap": 1.0}])
    assert outside["apex_in_gap_bounds_refused"] == "outside_frame"
    assert MB.summarize(_frame_rows())["apex_in_gap_bounds_refused"] is None


def test_summary_refuses_the_bound_when_an_accepted_row_has_no_outcome():
    rows = _frame_rows() + [{"case": "h", "flag": "OK", "z_gap": 4.0, "z_bif": 3.0,
                             "iliac_top": 1.0, "ivc_bottom": 5.0}]
    s = MB.summarize(rows)
    assert s["apex_in_gap_bounds_refused"] == "accepted_without_outcome"
    assert all(math.isnan(x) for x in s["apex_in_gap_bounds"])
    assert s["apex_unevaluated_flags"] == {"NONLINEAR": 1, "WIDE_CI": 1}, "an accepted row is not withheld"
    assert s["n_apex_rejected_fitted"] == 0, "an accepted row is not a rejected estimate"
    out = _console(rows).splitlines()
    i = next(n for n, ln in enumerate(out) if "apex falls INSIDE" in ln)
    assert "not computed" in out[i + 1] and "has no apex_in_gap outcome" in out[i + 1], out[i + 1]


def test_a_rejected_row_carrying_an_outcome_is_withheld_not_evaluated():
    rows = _frame_rows()
    rows[3] = dict(rows[3], apex_in_gap=1.0)           # the NONLINEAR row
    s = MB.summarize(rows)
    assert (s["n_apex_in_gap"], s["n_apex_eval"]) == (1, 3)
    assert s["apex_unevaluated_flags"] == {"NONLINEAR": 1, "WIDE_CI": 1}
    assert s["apex_in_gap_bounds"] == pytest.approx((0.2, 0.6))


def test_the_gates_rejected_estimates_are_counted_as_a_diagnostic():
    """Where those estimates fall is reported as a diagnostic of whether the gate is related to the
    outcome, from the row's own z_bif and landmarks, never as a rate; a withheld row with no fitted
    apex is not counted, and an accepted row never is."""
    rows = _frame_rows()
    rows[3] = dict(rows[3], z_bif=10.0, iliac_top=8.0, ivc_bottom=14.0)    # NONLINEAR, inside
    rows[4] = dict(rows[4], z_bif=20.0, iliac_top=8.0, ivc_bottom=10.0)    # WIDE_CI, outside
    rows[0] = dict(rows[0], z_bif=10.0, iliac_top=8.0, ivc_bottom=13.0)    # accepted: not counted
    s = MB.summarize(rows)
    assert (s["n_apex_rejected_inside"], s["n_apex_rejected_fitted"]) == (1, 2)
    line = next(ln for ln in _console(rows).splitlines() if "diagnostic, not a rate" in ln)
    assert "in 1 of 2 (50.0%) cases with a fitted apex, against 1 of 3 (33.3%) accepted fits" in line, line
    assert "related" not in line, "the console states proportions, not a conclusion from them"
    rows[4] = {k: v for k, v in rows[4].items() if k != "z_bif"}
    assert (MB.summarize(rows)["n_apex_rejected_inside"],
            MB.summarize(rows)["n_apex_rejected_fitted"]) == (1, 1)
    assert "diagnostic, not a rate" not in _console(_frame_rows()), "no fitted withheld apex"


def test_print_summary_states_the_bound_when_no_fit_was_accepted():
    """The rate is not estimable, and the range over the frame is the whole unit interval, which is
    what the data allow."""
    rows = [{"case": "f", "flag": "OK_OVERLAP", "z_gap": -1.0, "apex_in_gap": float("nan")},
            {"case": "w", "flag": "WIDE_CI", "z_gap": 3.0},
            {"case": "n", "flag": "NONLINEAR", "z_gap": 2.0}]
    out = _console(rows)
    assert "no z_gap>0 cases" not in out, out
    assert "not estimable, none of the 2 z_gap>0 cases has an accepted apex fit" in out, out
    assert "over all 2 z_gap>0 cases" in out and "0.0% to 100.0%" in out, out


def test_the_bound_line_names_an_empty_frame_rather_than_another_reason():
    """The range's last refusal branch: a summary with no range and no refusal reason (a frame
    with no case, reached only if a caller prints the bound without the console's own condition)
    says that, not the not-binary or outside-the-frame reason."""
    import contextlib
    import io
    s = {"apex_in_gap_bounds": (float("nan"), float("nan")), "apex_in_gap_bounds_refused": None}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        MB._print_apex_bounds(s)
    out = buf.getvalue()
    assert "not computed (no case has z_gap > 0)" in out, out
    assert "not 0 or 1" not in out and "no positive z_gap" not in out, out


def test_print_summary_presents_no_apex_validation_as_pending():
    """No apex caveat the console prints may present an apex validation as pending, and the line
    after the headline offset states that the apex is not externally validated and that the study
    collects no reference for it."""
    import re

    rows = [dict(r, vert_qc="OK", apex_level=5.0, gap_top_level=4.7, gap_bottom_level=5.1)
            for r in _rows_with_offset(+1)]
    out = _console(rows)
    assert "REGION-level comparison" in out, "the fixture must reach the vertebral block's line"
    pending = re.search(r"pending[^.\n]*validation|awaiting[^.\n]*validation|until[^.\n]*validated"
                        r"|(?:to|must|will) be validated|validation[^.\n]*(?:still )?required"
                        r"|validates REGION", out, flags=re.IGNORECASE)
    assert pending is None, f"the console presents the apex validation as pending: {pending.group(0)!r}"
    lines = out.splitlines()
    i = next(n for n, ln in enumerate(lines) if "SIGNED gap offset vs apex PROXY" in ln)
    assert "not externally validated" in lines[i + 1], lines[i + 1]
    assert "collects no reference" in lines[i + 1], lines[i + 1]


# --- measure_bif_one: wall-contact apex (integration) ---------------------
def _write(path, arr):
    img = sitk.GetImageFromArray(arr.astype(np.uint8))   # (z,y,x)
    img.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(img, str(path))


def _block(vol, z, yc, xc, s=3):
    h = s // 2
    vol[z, yc - h:yc + h + 1, xc - h:xc + h + 1] = True


def test_wall_contact_apex_is_caudal_to_centroid_coincidence(tmp_path):
    """Two iliac veins converging linearly toward a confluence. The WALL-CONTACT
    apex (sep == 2r) must land CAUDAL to where the centroids would coincide
    (sep == 0), about one vein diameter lower.

    The integer-rounded centre columns give the exact per-slice separations
    [14,14,12,12,10,10,10,8,8,6,6,6] at z = 0..11, whose OLS fit is slope = -114/143 = -0.79720,
    intercept = 14.0513 (hand-derived from the rounding, not read off the code). The recovered
    apexes must therefore land at (R - intercept)/slope: wallgap 13.863 mm, tangency 13.380 mm, and
    the flag must be exactly OK (z_gap = 4 > 0; accepting WIDE_CI here would bless a rejected
    apex)."""
    Z, Y, X = 20, 40, 80
    larr = np.zeros((Z, Y, X), bool)
    rarr = np.zeros((Z, Y, X), bool)
    ivc = np.zeros((Z, Y, X), bool)
    yc, x0 = 20, 40
    # slices 0..11: separation shrinks 14 -> 5.2 as z increases (converging)
    for k in range(12):
        sep = 14.0 - 0.8 * k
        _block(larr, k, yc, int(round(x0 - sep / 2)))
        _block(rarr, k, yc, int(round(x0 + sep / 2)))
    # IVC sits cranial to the iliacs
    for k in range(15, 19):
        ivc[k, yc - 1:yc + 2, x0 - 1:x0 + 2] = True
    seg = tmp_path / "seg"; seg.mkdir()
    _write(seg / "inferior_vena_cava.nii.gz", ivc)
    _write(seg / "iliac_vena_left.nii.gz", larr)
    _write(seg / "iliac_vena_right.nii.gz", rarr)

    # use the production gates (BIF_MIN_SLICES=6); the synthetic has 12 slices
    r = MB.measure_bif_one(tmp_path, segdir="seg")   # PRIMARY = wallgap (default)
    assert r["flag"] == "OK"                         # a TRUSTED fit, not merely a fit
    assert r["apex_method"] == "wallgap"
    assert r["slope"] < 0                                  # veins converge
    assert r["slope"] == pytest.approx(-114.0 / 143.0, abs=1e-3)
    assert r["iliac_r_mm"] > 0
    assert r["iliac_r_mm"] == pytest.approx(1.69, abs=0.4)
    assert math.isfinite(r["z_bif"]) and r["z_bif"] < 17.5
    assert math.isfinite(r["z_bif_alt"]) and r["z_bif_alt"] < 17.5
    # analytic recovery of the KNOWN apexes (tolerance covers only fit numerics;
    # the derivation is in the docstring)
    assert r["z_bif"] == pytest.approx(13.863, abs=0.25)
    assert r["z_bif_alt"] == pytest.approx(13.380, abs=0.25)


def test_measure_bif_one_computes_alpha2(tmp_path):
    # the integration path must produce a finite, in-range alpha2 (LCIV-IVC angle)
    Z, Y, X = 24, 40, 80
    larr = np.zeros((Z, Y, X), bool); rarr = np.zeros((Z, Y, X), bool)
    ivc = np.zeros((Z, Y, X), bool)
    yc, x0 = 20, 40
    for k in range(12):                       # CIVs converge toward x0 as z rises
        sep = 14.0 - 0.8 * k
        _block(larr, k, yc, int(round(x0 - sep / 2)))
        _block(rarr, k, yc, int(round(x0 + sep / 2)))
    for k in range(15, 23):                   # IVC: vertical column above the confluence
        ivc[k, yc - 1:yc + 2, x0 - 1:x0 + 2] = True
    seg = tmp_path / "seg"; seg.mkdir()
    _write(seg / "inferior_vena_cava.nii.gz", ivc)
    _write(seg / "iliac_vena_left.nii.gz", larr)
    _write(seg / "iliac_vena_right.nii.gz", rarr)
    r = MB.measure_bif_one(tmp_path, segdir="seg")
    assert math.isfinite(r["alpha2_deg"]) and 0.0 < r["alpha2_deg"] < 90.0
    assert math.isfinite(r["alpha1_deg"]) and 0.0 < r["alpha1_deg"] < 90.0


def test_measure_bif_one_fails_closed_on_class_grid_mismatch(tmp_path):
    shape = (12, 20, 20)
    ivc = np.zeros(shape, bool); ivc[8:12, 9:12, 9:12] = True
    lciv = np.zeros(shape, bool); lciv[:7, 9:12, 6:9] = True
    rciv = np.zeros(shape, bool); rciv[:7, 9:12, 12:15] = True
    seg = tmp_path / "seg"; seg.mkdir()
    _write(seg / "inferior_vena_cava.nii.gz", ivc)
    _write(seg / "iliac_vena_left.nii.gz", lciv)
    image = sitk.GetImageFromArray(rciv.astype(np.uint8))
    image.SetSpacing((1.0, 1.0, 2.0))
    sitk.WriteImage(image, str(seg / "iliac_vena_right.nii.gz"))
    assert MB.measure_bif_one(tmp_path, segdir="seg")["flag"] == "GRID_MISMATCH"


def test_tangency_apex_method_selectable(tmp_path):
    Z, Y, X = 20, 40, 80
    larr = np.zeros((Z, Y, X), bool); rarr = np.zeros((Z, Y, X), bool)
    ivc = np.zeros((Z, Y, X), bool)
    yc, x0 = 20, 40

    def _blk(vol, z, yc, xc, s=3):
        h = s // 2
        vol[z, yc - h:yc + h + 1, xc - h:xc + h + 1] = True
    for k in range(12):
        sep = 14.0 - 0.8 * k
        _blk(larr, k, yc, int(round(x0 - sep / 2)))
        _blk(rarr, k, yc, int(round(x0 + sep / 2)))
    for k in range(15, 19):
        ivc[k, yc - 1:yc + 2, x0 - 1:x0 + 2] = True
    seg = tmp_path / "seg"; seg.mkdir()
    _write(seg / "inferior_vena_cava.nii.gz", ivc)
    _write(seg / "iliac_vena_left.nii.gz", larr)
    _write(seg / "iliac_vena_right.nii.gz", rarr)
    r = MB.measure_bif_one(tmp_path, segdir="seg", apex_method="tangency")
    assert r["apex_method"] == "tangency"
    assert math.isfinite(r["z_bif"])


def test_validate_against_reference_agreement():
    rows = [
        {"case": "a", "flag": "OK", "z_bif": 10.0, "iliac_top": 0.0},
        {"case": "b", "flag": "OK", "z_bif": 20.0, "iliac_top": 0.0},
        {"case": "c", "flag": "OK", "z_bif": 30.0, "iliac_top": 0.0},
        {"case": "d", "flag": "WIDE_CI", "z_bif": 99.0, "iliac_top": 0.0},  # excluded
    ]
    ref = {"a": 11.0, "b": 19.0, "c": 31.0, "d": 0.0}
    val = MB.validate_against_reference(rows, ref, tol_mm=5.0)
    assert val["n_matched"] == 3                            # d excluded by flag
    assert abs(val["bias"]) <= 1.0
    assert val["within_tol_frac"] == 1.0                    # all within 5mm
    assert val["ccc"] > 0.9


def test_validate_drops_cases_missing_iliac_top():
    rows = [
        {"case": "a", "flag": "OK", "z_bif": 610.0, "iliac_top": 600.0},
        {"case": "b", "flag": "OK", "z_bif": 620.0, "iliac_top": 600.0},
        {"case": "c", "flag": "OK", "z_bif": 630.0, "iliac_top": 600.0},
        {"case": "d", "flag": "OK", "z_bif": 640.0, "iliac_top": float("nan")},
    ]
    ref = {"a": 611.0, "b": 619.0, "c": 631.0, "d": 641.0}
    val = MB.validate_against_reference(rows, ref, tol_mm=5.0)
    assert val["n_matched"] == 3        # case d (no finite iliac_top) excluded


def test_validate_returns_none_when_too_few():
    assert MB.validate_against_reference(
        [{"case": "a", "flag": "OK", "z_bif": 5.0, "iliac_top": 0.0}],
        {"a": 5.0}) is None


# --- connected-component count must use 8-connectivity ----------------
def test_cc_count_uses_8_connectivity():
    # diagonally-touching voxels are ONE lumen under 8-connectivity (correct);
    # 4-connectivity would over-split a compressed CIV and spuriously flag its topology
    m = np.zeros((4, 4), bool)
    m[0, 0] = True
    m[1, 1] = True
    assert MB._cc_count(m) == 1
    m2 = np.zeros((4, 4), bool)
    m2[0, 0] = True
    m2[3, 3] = True              # genuinely separated
    assert MB._cc_count(m2) == 2


# --- _z_extent must be voxel-aware under a tilted gantry --------------------
def _tilted_image(arr, angle_deg=10.0, spacing=(1.0, 1.0, 2.0)):
    """A SimpleITK image whose direction matrix is a real rotation about x by
    `angle_deg` (a tilted-gantry geometry), so physical z depends on the in-plane
    (i, j) and NOT only on k. `arr` is (z, y, x)."""
    img = sitk.GetImageFromArray(arr.astype(np.uint8))
    img.SetSpacing(spacing)
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    # rotation about the x-axis: couples y and z so phys-z depends on the in-plane row
    img.SetDirection((1.0, 0.0, 0.0,
                      0.0, ca, -sa,
                      0.0, sa, ca))
    img.SetOrigin((0.0, 0.0, 0.0))
    return img


def _brute_phys_z_extent(img, arr):
    """Reference physical-z min/max over the TRUE extreme-slice voxels (transform
    every present voxel on the min-k and max-k slices)."""
    zk = np.where(arr.any(axis=(1, 2)))[0]
    k_min, k_max = int(zk.min()), int(zk.max())
    zvals = []
    for k in (k_min, k_max):
        ys, xs = np.nonzero(arr[k])
        for i, j in zip(xs.tolist(), ys.tolist()):
            zvals.append(img.TransformContinuousIndexToPhysicalPoint(
                (float(i), float(j), float(k)))[2])
    return min(zvals), max(zvals)


def test_z_extent_voxel_aware_under_tilt():
    # a slab with off-axis voxels so the (0,0,k) sample is NOT the z-extremum
    Z, Y, X = 6, 20, 20
    arr = np.zeros((Z, Y, X), bool)
    arr[1, 2:18, 2:18] = True        # caudal-ish slice, wide in-plane extent
    arr[4, 2:18, 2:18] = True        # cranial-ish slice
    img = _tilted_image(arr, angle_deg=10.0)
    lo, hi = MB._z_extent(img, arr)[:2]
    blo, bhi = _brute_phys_z_extent(img, arr)
    # voxel-aware result must match the brute-force extreme-voxel physical z
    assert lo == pytest.approx(blo, abs=1e-6)
    assert hi == pytest.approx(bhi, abs=1e-6)
    # ...and differs from (0,0,k)-only sampling, which misses the off-axis extremes
    old_lo = min(img.TransformContinuousIndexToPhysicalPoint((0.0, 0.0, 1.0))[2],
                 img.TransformContinuousIndexToPhysicalPoint((0.0, 0.0, 4.0))[2])
    old_hi = max(img.TransformContinuousIndexToPhysicalPoint((0.0, 0.0, 1.0))[2],
                 img.TransformContinuousIndexToPhysicalPoint((0.0, 0.0, 4.0))[2])
    assert abs(hi - old_hi) > 1.0 or abs(lo - old_lo) > 1.0


def test_z_extent_axis_aligned_matches_slice_origin_sampling():
    # for an identity/axis-aligned direction the voxel-aware form equals (0,0,k) sampling
    # exactly
    Z, Y, X = 6, 20, 20
    arr = np.zeros((Z, Y, X), bool)
    arr[1, 3:7, 3:7] = True
    arr[4, 3:7, 3:7] = True
    img = sitk.GetImageFromArray(arr.astype(np.uint8))
    img.SetSpacing((1.0, 1.0, 2.0))
    img.SetOrigin((10.0, 20.0, 30.0))
    lo, hi = MB._z_extent(img, arr)[:2]
    old_lo = MB._phys_z(img, 1)
    old_hi = MB._phys_z(img, 4)
    assert lo == pytest.approx(min(old_lo, old_hi), abs=1e-9)
    assert hi == pytest.approx(max(old_lo, old_hi), abs=1e-9)


def test_z_extent_returns_the_present_slice_indices():
    # the third return value is the present-z slice indices
    Z, Y, X = 5, 10, 10
    arr = np.zeros((Z, Y, X), bool)
    arr[1, 4:6, 4:6] = True
    arr[3, 4:6, 4:6] = True
    img = sitk.GetImageFromArray(arr.astype(np.uint8))
    img.SetSpacing((1.0, 1.0, 1.0))
    ks = MB._z_extent(img, arr)[2]
    assert list(ks) == [1, 3]


# --- OBLIQUE QC flag (signed-permutation test, like the gap script) --------
def test_oblique_flag_fires_for_tilted_gantry():
    Z, Y, X = 8, 30, 60
    larr = np.zeros((Z, Y, X), bool); rarr = np.zeros((Z, Y, X), bool)
    ivc = np.zeros((Z, Y, X), bool)
    yc, x0 = 15, 30
    for k in range(6):
        sep = 14.0 - 1.6 * k
        _block(larr, k, yc, int(round(x0 - sep / 2)))
        _block(rarr, k, yc, int(round(x0 + sep / 2)))
    ivc[7, yc - 1:yc + 2, x0 - 1:x0 + 2] = True
    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp())
    seg = tmp / "seg"; seg.mkdir()
    # write with a real tilt
    for name, a in (("inferior_vena_cava.nii.gz", ivc),
                    ("iliac_vena_left.nii.gz", larr),
                    ("iliac_vena_right.nii.gz", rarr)):
        img = _tilted_image(a, angle_deg=10.0, spacing=(1.0, 1.0, 1.0))
        sitk.WriteImage(img, str(seg / name))
    r = MB.measure_bif_one(tmp, segdir="seg")
    assert r.get("oblique") == 1


def test_oblique_flag_not_fired_for_axis_flip(tmp_path):
    # an LPS/RAS axis flip is a SIGNED PERMUTATION -> must NOT trip OBLIQUE
    Z, Y, X = 8, 30, 60
    larr = np.zeros((Z, Y, X), bool); rarr = np.zeros((Z, Y, X), bool)
    ivc = np.zeros((Z, Y, X), bool)
    yc, x0 = 15, 30
    for k in range(6):
        sep = 14.0 - 1.6 * k
        _block(larr, k, yc, int(round(x0 - sep / 2)))
        _block(rarr, k, yc, int(round(x0 + sep / 2)))
    ivc[7, yc - 1:yc + 2, x0 - 1:x0 + 2] = True
    seg = tmp_path / "seg"; seg.mkdir()
    for name, a in (("inferior_vena_cava.nii.gz", ivc),
                    ("iliac_vena_left.nii.gz", larr),
                    ("iliac_vena_right.nii.gz", rarr)):
        img = sitk.GetImageFromArray(a.astype(np.uint8))
        img.SetSpacing((1.0, 1.0, 1.0))
        img.SetDirection((1.0, 0.0, 0.0,
                          0.0, -1.0, 0.0,
                          0.0, 0.0, -1.0))   # axis flip (signed permutation)
        sitk.WriteImage(img, str(seg / name))
    r = MB.measure_bif_one(tmp_path, segdir="seg")
    assert r.get("oblique") == 0


# --- centroid must be robust to a detached speck (largest CC/slice) -------
def _converging_case(tmp_dir, speck_slices=()):
    """Write a converging-veins case; optionally add a tiny detached speck on the
    given LEFT-vein slices to drag a naive (all-voxel) centroid."""
    Z, Y, X = 20, 40, 80
    larr = np.zeros((Z, Y, X), bool); rarr = np.zeros((Z, Y, X), bool)
    ivc = np.zeros((Z, Y, X), bool)
    yc, x0 = 20, 40
    for k in range(12):
        sep = 14.0 - 0.8 * k
        _block(larr, k, yc, int(round(x0 - sep / 2)), s=5)   # main lumen
        _block(rarr, k, yc, int(round(x0 + sep / 2)), s=5)
    for k in speck_slices:
        # a 1-voxel detached speck far away in +x on the left vein only
        larr[k, yc + 12, x0 + 25] = True
    for k in range(15, 19):
        ivc[k, yc - 1:yc + 2, x0 - 1:x0 + 2] = True
    seg = tmp_dir / "seg"; seg.mkdir(parents=True)
    _write(seg / "inferior_vena_cava.nii.gz", ivc)
    _write(seg / "iliac_vena_left.nii.gz", larr)
    _write(seg / "iliac_vena_right.nii.gz", rarr)
    return tmp_dir


def test_centroid_robust_to_minority_speck(tmp_path):
    clean = _converging_case(tmp_path / "clean")
    # speck on a MINORITY of slices (4 of 12 -> below the 50% VARIANT threshold)
    dirty = _converging_case(tmp_path / "dirty", speck_slices=(0, 1, 2, 3))
    rc = MB.measure_bif_one(clean, segdir="seg")
    rd = MB.measure_bif_one(dirty, segdir="seg")
    # the speck must NOT trip VARIANT (it is on a minority of slices)
    assert rd["flag"] != "VARIANT"
    assert math.isfinite(rc["z_bif"]) and math.isfinite(rd["z_bif"])
    # with the largest-CC-per-slice centroid the apex barely moves
    assert abs(rc["z_bif"] - rd["z_bif"]) < 0.2


def test_centroid_phys_uses_largest_component(tmp_path):
    # direct unit check: a big blob plus a far speck -> centroid ~ blob centroid
    Z, Y, X = 3, 40, 40
    arr = np.zeros((Z, Y, X), bool)
    arr[1, 18:23, 18:23] = True       # 5x5 main blob centred ~ (20,20)
    arr[1, 35, 5] = True              # detached speck far away
    img = sitk.GetImageFromArray(arr.astype(np.uint8))
    img.SetSpacing((1.0, 1.0, 1.0))
    c = MB._centroid_phys(img, arr, 1)
    assert c[0] == pytest.approx(20.0, abs=0.5)   # x ~ blob centre, not dragged
    assert c[1] == pytest.approx(20.0, abs=0.5)   # y ~ blob centre


def _speck_case(tmp_dir, speck_slices=()):
    """Converging veins whose per-slice block size varies so the clean wallgap
    contact DISTRIBUTION is 7 slices at 3.0 mm (3x3 blocks, k=0..6) and 5 at
    5.0 mm (5x5 blocks, k=7..11): for an s x s block pair at integer centre
    distance d the voxel-box contact is exactly s (d_min = d - s after the
    1-voxel dilation), so median(R) = 3.0. On `speck_slices` a single detached
    RIGHT-mask voxel is planted between the veins (at c_left + 4, i.e. 2 px from
    the dilated left surface). With an UN-hardened d_min that collapses the
    slice's d_min to 2 and lifts its contact to sep - 2 (8..12 mm here): planting
    specks on 5 of the 7 low-contact slices moves median(R) from 3.0 to 5.0 and
    the apex by (5-3)/0.8 = 2.5 mm, while sep itself never moves (the centroid
    is already hardened) and VARIANT never fires (5/12 < the 50% threshold).
    With the hardened d_min/radii the speck changes nothing at all."""
    Z, Y, X = 20, 40, 80
    larr = np.zeros((Z, Y, X), bool)
    rarr = np.zeros((Z, Y, X), bool)
    ivc = np.zeros((Z, Y, X), bool)
    yc, x0 = 20, 40
    for k in range(12):
        sep = 14.0 - 0.8 * k
        s = 3 if k <= 6 else 5
        cl = int(round(x0 - sep / 2))
        cr = int(round(x0 + sep / 2))
        _block(larr, k, yc, cl, s=s)
        _block(rarr, k, yc, cr, s=s)
        if k in speck_slices:
            rarr[k, yc, cl + 4] = True     # right-mask speck near the LEFT vein
    for k in range(15, 19):
        ivc[k, yc - 1:yc + 2, x0 - 1:x0 + 2] = True
    seg = tmp_dir / "seg"; seg.mkdir(parents=True)
    _write(seg / "inferior_vena_cava.nii.gz", ivc)
    _write(seg / "iliac_vena_left.nii.gz", larr)
    _write(seg / "iliac_vena_right.nii.gz", rarr)
    return tmp_dir


def test_dmin_and_radii_are_speck_hardened_like_the_centroid(tmp_path):
    """All three ingredients use the same largest-CC slice mask, so the speck case must reproduce
    the clean case EXACTLY, without tripping VARIANT."""
    clean = _speck_case(tmp_path / "clean")
    dirty = _speck_case(tmp_path / "dirty", speck_slices=(0, 1, 2, 3, 4))
    rc = MB.measure_bif_one(clean, segdir="seg")
    rd = MB.measure_bif_one(dirty, segdir="seg")
    assert rc["flag"] == "OK"
    assert rd["flag"] != "VARIANT"          # 5/12 speck slices < the 50% threshold
    assert math.isfinite(rc["z_bif"]) and math.isfinite(rd["z_bif"])
    # hardened d_min/radii: the speck changes NOTHING the fit consumes
    assert rd["z_bif"] == pytest.approx(rc["z_bif"], abs=1e-9)
    assert rd["z_bif_alt"] == pytest.approx(rc["z_bif_alt"], abs=1e-9)
    assert rd["iliac_r_mm"] == pytest.approx(rc["iliac_r_mm"], abs=1e-9)


def test_an_overran_apex_ci_is_clamped_and_forced_wide_ci(tmp_path, monkeypatch):
    """A raw bootstrap CI that runs past the plausibility band [z_min - window, z_max + window] must
    (a) be REPORTED clamped to the band and (b) FORCE the flag to WIDE_CI even when the clamped
    half-width would pass decide_flag."""
    case = _converging_case(tmp_path / "c")
    fake = {"z_bif": 13.9, "slope": -0.8, "intercept": 14.0, "slope_se": 0.05,
            "resid_se": 0.3, "n": 12, "ci_lo": 25.0, "ci_hi": 33.0,
            "unstable": False, "ci_method": "bca", "median_r": 3.0,
            "nonpositive_R": False}
    monkeypatch.setattr(MB.S, "wall_contact_apex", lambda *a, **k: dict(fake))
    r = MB.measure_bif_one(case, segdir="seg")
    # shared slices span z = 0..11 and the window is 20 mm -> band [-20, 31]:
    # the raw hi of 33 overran it, the raw lo of 25 did not
    assert r["z_bif_ci_hi"] == pytest.approx(31.0)   # clamped to the band edge
    assert r["z_bif_ci_lo"] == pytest.approx(25.0)   # inside the band: untouched
    assert r["flag"] == "WIDE_CI"                    # forced despite hw = 3 <= 10


def _fallback_case(tmp_dir, reversed_z=False):
    """12 shared slices at 5 mm z-spacing, so the 20 mm cranial window holds only 5 (< BIF_MIN_SLICES =
    6) and the fallback must engage."""
    Z, Y, X = 16, 40, 80
    larr = np.zeros((Z, Y, X), bool)
    rarr = np.zeros((Z, Y, X), bool)
    ivc = np.zeros((Z, Y, X), bool)
    yc, x0 = 20, 40
    for k in range(12):
        if not reversed_z:
            sep = 14.0 if k <= 5 else 14.0 - 1.6 * (k - 5)
        else:
            sep = 4.4 + 1.6 * k if k <= 5 else 14.0
        _block(larr, k, yc, int(round(x0 - sep / 2)))
        _block(rarr, k, yc, int(round(x0 + sep / 2)))
    for k in range(13, 15):
        ivc[k, yc - 1:yc + 2, x0 - 1:x0 + 2] = True
    seg = tmp_dir / "seg"; seg.mkdir(parents=True)
    for name, arr in (("inferior_vena_cava.nii.gz", ivc),
                      ("iliac_vena_left.nii.gz", larr),
                      ("iliac_vena_right.nii.gz", rarr)):
        img = sitk.GetImageFromArray(arr.astype(np.uint8))
        img.SetSpacing((1.0, 1.0, 5.0))
        if reversed_z:
            img.SetDirection((1.0, 0.0, 0.0,
                              0.0, 1.0, 0.0,
                              0.0, 0.0, -1.0))
        sitk.WriteImage(img, str(seg / name))
    return tmp_dir


def test_window_fallback_selects_the_cranial_most_slices(tmp_path):
    ra = MB.measure_bif_one(_fallback_case(tmp_path / "a"), segdir="seg")
    assert ra["window_fallback"] is True
    assert ra["fit_n"] == 6
    assert ra["slope"] < -0.05                       # the converging cranial slices
    assert ra["flag"] in ("OK", "OK_OVERLAP", "WIDE_CI")   # NOT NONCONVERGING
    rb = MB.measure_bif_one(_fallback_case(tmp_path / "b", reversed_z=True),
                            segdir="seg")
    assert rb["window_fallback"] is True
    assert rb["fit_n"] == 6
    assert rb["slope"] < -0.05                       # cranial by PHYSICAL z
    assert rb["flag"] in ("OK", "OK_OVERLAP", "WIDE_CI")


# --- window_stability_mm + conv_angle_deg must round-trip the file ---------
def test_results_row_roundtrips_window_stability_and_conv_angle(tmp_path):
    row = {
        "case": "c1", "ivc_bottom": 30.0, "iliac_top": 25.0, "z_gap": 5.0,
        "z_bif": 27.0, "z_bif_ci_lo": 26.0, "z_bif_ci_hi": 28.0,
        "z_bif_alt": 27.2, "apex_method": "wallgap", "slope": -1.234,
        "slope_se": 0.05, "ivc_overshoot": 3.0, "iliac_undershoot": 2.0,
        "gap_center_offset": 0.5, "frac_above": 0.6, "apex_in_gap": 1.0,
        "iliac_r_mm": 6.5, "alpha1_deg": 30.0, "alpha2_deg": 45.0,
        "fit_n": 8, "window_fallback": False,
        "window_stability_mm": 1.7, "conv_angle_deg": 51.0,
        "vert_n": 6, "gap_top_level": 5.0, "gap_bottom_level": 4.8,
        "apex_level": 4.9, "vert_qc": "OK", "oblique": 0, "flag": "OK",
    }
    # both fields must be present in the schema
    assert "window_stability_mm" in MB.COLUMNS
    assert "conv_angle_deg" in MB.COLUMNS
    line = MB._fmt(row)
    p = tmp_path / "results.txt"
    p.write_text(" ".join(MB.COLUMNS) + "\n" + line + "\n")
    # re-parse the saved file by column name
    with open(p) as f:
        header = f.readline().split()
        vals = f.readline().split()
    parsed = dict(zip(header, vals))
    assert "window_stability_mm" in parsed and "conv_angle_deg" in parsed
    assert float(parsed["window_stability_mm"]) == pytest.approx(1.7, abs=0.05)
    assert float(parsed["conv_angle_deg"]) == pytest.approx(51.0, abs=0.05)


def test_columns_leading_positions_are_fixed():
    # the leading columns keep fixed positions
    expected_prefix = [
        "case", "ivc_bottom", "iliac_top", "z_gap", "z_bif",
        "z_bif_ci_lo", "z_bif_ci_hi", "z_bif_alt", "apex_method",
        "slope", "slope_se", "ivc_overshoot", "iliac_undershoot",
        "gap_center_offset", "frac_above", "apex_in_gap", "iliac_r_mm",
        "alpha1_deg", "alpha2_deg", "fit_n", "window_fallback",
        "vert_n", "gap_top_level", "gap_bottom_level", "apex_level", "vert_qc",
    ]
    assert MB.COLUMNS[:len(expected_prefix)] == expected_prefix


def test_fmt_header_and_row_have_equal_width():
    # every COLUMNS field must be emitted by _fmt (counts must match)
    row = {
        "case": "c1", "ivc_bottom": 30.0, "iliac_top": 25.0, "z_gap": 5.0,
        "z_bif": 27.0, "z_bif_ci_lo": 26.0, "z_bif_ci_hi": 28.0,
        "z_bif_alt": 27.2, "apex_method": "wallgap", "slope": -1.234,
        "slope_se": 0.05, "ivc_overshoot": 3.0, "iliac_undershoot": 2.0,
        "gap_center_offset": 0.5, "frac_above": 0.6, "apex_in_gap": 1.0,
        "iliac_r_mm": 6.5, "alpha1_deg": 30.0, "alpha2_deg": 45.0,
        "fit_n": 8, "window_fallback": False,
        "window_stability_mm": 1.7, "conv_angle_deg": 51.0,
        "vert_n": 6, "gap_top_level": 5.0, "gap_bottom_level": 4.8,
        "apex_level": 4.9, "vert_qc": "OK", "oblique": 0, "flag": "OK",
    }
    assert len(MB._fmt(row).split()) == len(MB.COLUMNS)


def test_nonlinear_flag_inf_shift_is_unconditional():
    assert MB._nonlinear_flag(curv_significant=False, shift=float("inf"), ci_hw=2.0) is True
    assert MB._nonlinear_flag(curv_significant=True, shift=float("inf"), ci_hw=2.0) is True


def test_nonlinear_flag_finite_shift_needs_significance_and_materiality():
    # finite shift: flagged only when curvature is significant AND it moves the apex
    # beyond its own CI half-width.
    assert MB._nonlinear_flag(True, shift=5.0, ci_hw=2.0) is True     # significant + material
    assert MB._nonlinear_flag(True, shift=1.0, ci_hw=2.0) is False    # significant but immaterial
    assert MB._nonlinear_flag(False, shift=5.0, ci_hw=2.0) is False   # material but not significant
    # NaN shift (uncomputable) is never flagged on its own
    assert MB._nonlinear_flag(True, shift=float("nan"), ci_hw=2.0) is False


def test_quad_apex_shift_inf_when_no_real_root_at_contact():
    # a downward-opening parabola whose maximum is BELOW the contact level R has no
    # real root at R -> _quad_apex_shift returns +inf (material curvature signal).
    z = np.array([0, 1, 2, 3, 4, 5], float)
    sep = -(z - 2.5) ** 2 + 1.0          # peak value 1.0 at z=2.5
    shift = MB._quad_apex_shift(z, sep, R=100.0, z_lin=2.5)   # R far above the peak
    assert shift == float("inf")


# --- is_oblique: direct unit test of the signed-permutation detector ----------
#     The pipeline tests above (test_oblique_flag_*) cover a real rotation and an axis
#     FLIP only. A flip is diagonal; a pure axis PERMUTATION is off-diagonal and is the
#     only case that exercises the per-row/per-column "exactly one near-1 entry" branch.
#     This detector gates the published vertebral-level (L5) read, so pin it directly.
def test_is_oblique_signed_permutations_false_rotation_true():
    def _img(direction):
        im = sitk.GetImageFromArray(np.zeros((4, 4, 4), np.uint8))
        im.SetDirection(direction)
        return im
    ident = (1, 0, 0, 0, 1, 0, 0, 0, 1)
    flip = (-1, 0, 0, 0, -1, 0, 0, 0, 1)             # LPS/RAS axis flip (diagonal)
    perm = (0, 1, 0, 1, 0, 0, 0, 0, 1)               # i<->j swap (off-diagonal permutation)
    cc, ss = math.cos(math.radians(30)), math.sin(math.radians(30))
    rot = (1, 0, 0, 0, cc, -ss, 0, ss, cc)           # genuine 30deg rotation -> oblique
    assert MB.is_oblique(_img(ident)) is False
    assert MB.is_oblique(_img(flip)) is False
    assert MB.is_oblique(_img(perm)) is False         # the case the axis-flip test misses
    assert MB.is_oblique(_img(rot)) is True


# --- build_vert_ruler / z_to_vert_level: the vertebral ruler behind the L5-rate result,
#     pinned to a KNOWN linear level<->z map ---
def test_build_vert_ruler_linear_oracle():
    # level L anchored at z = 100 - 10*L (z DECREASES with level: LPS +z is cranial, so
    # L1 highest, L5 lowest). The piecewise-linear ruler must invert this map, so a
    # physical z returns level (100 - z)/10 by independent hand computation.
    anchors = [(1.0, 90.0), (2.0, 80.0), (3.0, 70.0), (4.0, 60.0), (5.0, 50.0)]  # L1..L5
    interp, z_min, z_max, qc = MB.build_vert_ruler(anchors)
    assert qc == "OK"
    assert (z_min, z_max) == (50.0, 90.0)
    for z, level in [(90.0, 1.0), (85.0, 1.5), (70.0, 3.0), (55.0, 4.5), (50.0, 5.0)]:
        assert interp(z) == pytest.approx(level)      # independent linear inversion
    lvl, in_range = MB.z_to_vert_level((interp, z_min, z_max, qc), 70.0)
    assert lvl == pytest.approx(3.0) and in_range is True
    _, in_range_oob = MB.z_to_vert_level((interp, z_min, z_max, qc), 200.0)
    assert in_range_oob is False                      # above the cranial-most anchor


def test_build_vert_ruler_qc_flags():
    assert MB.build_vert_ruler([])[3] == "NO_VERT"             # no vertebrae present
    assert MB.build_vert_ruler([(3.0, 70.0)])[3] == "FEW_VERT"  # a single anchor
    # z must DECREASE as level increases; an increasing order signals a TotalSegmentator
    # level mislabel and must be flagged NONMONOTONIC, not silently trusted.
    bad = [(1.0, 50.0), (2.0, 60.0), (3.0, 70.0)]
    assert MB.build_vert_ruler(bad)[3] == "NONMONOTONIC"
    interp, _zlo, _zhi, qc = MB.build_vert_ruler([(4.0, 60.0), (6.0, 30.0)])   # L4 & S1, no L5
    assert qc == "SPARSE_VERT" and interp is not None
    # an adjacent, complete chain stays OK
    assert MB.build_vert_ruler([(4.0, 60.0), (5.0, 45.0), (6.0, 30.0)])[3] == "OK"


def _true_intervein_deg(a, b):
    """The full angle between two veins diverging from the midline at rates a and b."""
    return math.degrees(math.atan(a) + math.atan(b))


def test_the_convergence_angle_is_exact_only_when_the_two_veins_are_symmetric():
    """`2*atan(|slope|/2)` is right iff a == b, because `slope` is the SUM of the two rates.

    One number cannot identify two directions, so this is an assumption, not a derivation.
    """
    for rate in (0.1, 0.25, 0.5, 1.0, 2.0):
        slope = 2 * rate                                   # a == b == rate
        assert MB._conv_angle_deg(slope) == pytest.approx(
            _true_intervein_deg(rate, rate), abs=1e-9)


def test_the_convergence_angle_over_reads_an_asymmetric_confluence():
    """The bias has a determinate SIGN: strictly UP, never down.

    `atan` is strictly concave on (0, inf), so Jensen gives 2*atan((a+b)/2) >= atan(a) + atan(b)
    with equality only at a == b."""
    worst, arg = 0.0, None
    for slope in np.linspace(0.2, 2.0, 181):
        for ratio in np.linspace(1.0, 9.0, 161):
            b = slope / (1.0 + ratio)
            a = slope - b
            err = MB._conv_angle_deg(slope) - _true_intervein_deg(a, b)
            assert err > -1e-9, (
                f"the formula UNDER-read at slope={slope}, a/b={ratio}: {err} deg. Jensen "
                f"forbids this; either the formula changed or atan did.")
            if err > worst:
                worst, arg = err, (slope, ratio, a, b)
    assert worst == pytest.approx(17.7447, abs=5e-4), (
        f"the worst overstatement is {worst:.4f} deg at {arg}; the figure is quoted in "
        f"measure_bifurcation._conv_angle_deg and in its console summary, and must be "
        f"updated with them.")
    slope, ratio, a, b = arg
    assert (slope, ratio) == pytest.approx((2.0, 9.0))     # the corner of the swept range
    assert _true_intervein_deg(a, b) == pytest.approx(72.2553, abs=5e-4)
    assert MB._conv_angle_deg(slope) == pytest.approx(90.0, abs=5e-4)


def test_projection_onto_a_plane_does_not_only_shrink_an_angle():
    """Projection onto a plane can enlarge an angle as well as shrink it.

    The counterexample is exact and integer, so it cannot be a floating-point artifact; at
    equal elevation (the configuration two symmetric iliac veins occupy) the 3-D angle is at
    most the projected one.
    """
    def ang(u, v):
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        return math.degrees(math.acos(max(-1.0, min(1.0, c))))

    u, v = (1.0, 0.0, 10.0), (-1.0, 0.0, 10.0)
    three_d = ang(u, v)
    projected = ang(u[:2] + (0.0,), v[:2] + (0.0,))
    assert three_d == pytest.approx(math.degrees(math.acos(99 / 101)), abs=1e-9)
    assert projected == pytest.approx(180.0, abs=1e-9)
    assert projected > three_d, "projection must be able to enlarge an angle"

    # Equal elevation phi above the plane, in-plane angle psi:
    # cos(theta_3) - cos(theta_2) = (1 - cos psi) * sin^2(phi) >= 0, so theta_3 <= theta_2.
    for phi in (0.1, 0.4, 0.9, 1.3):
        for psi in (0.2, 0.7, 1.5, 2.5, 3.0):
            a3 = (math.cos(phi), 0.0, math.sin(phi))
            b3 = (math.cos(phi) * math.cos(psi), math.cos(phi) * math.sin(psi),
                  math.sin(phi))
            assert ang(a3, b3) <= math.degrees(psi) + 1e-9, (
                "at equal elevation the 3-D angle must be at most the projected one")


def test_an_unimplemented_apex_method_is_refused():
    """An apex method outside SUPPORTED_APEX_METHODS raises, naming both supported methods, so a
    typo cannot be computed as one method and published under another name."""
    assert set(MB.SUPPORTED_APEX_METHODS) == {"wallgap", "tangency"}
    assert MB.C.BIF_APEX_METHOD in MB.SUPPORTED_APEX_METHODS

    for bad in ("Tangency", "tangencyy", "wallgapp", "WALLGAP", "", "wall gap", None.__class__.__name__):
        with pytest.raises(ValueError) as exc:
            MB.require_apex_method(bad)
        assert repr(bad) in str(exc.value)
        assert "wallgap" in str(exc.value) and "tangency" in str(exc.value)

    for good in ("wallgap", "tangency"):
        assert MB.require_apex_method(good) == good


def test_the_refusal_is_wired_into_the_measurement_entry_point(tmp_path):
    """The gate has to run where the string arrives, or it is dead code. Reaching the dispatch
    at all requires masks, so this asserts the refusal happens BEFORE any file is read: the
    case directory is empty, and a missing-mask error would mean the check came too late.
    """
    with pytest.raises(ValueError) as exc:
        MB.measure_bif_one(tmp_path, segdir="does-not-exist", apex_method="Tangency")
    assert "'Tangency'" in str(exc.value)


def test_an_alternate_apex_that_overran_the_physical_band_is_not_published_as_a_number():
    """An alternate apex outside the band is NaN, not the bound: the bound is the case's
    shared-slice z-extent padded by one window, which no output column records, so a clamped
    value would be indistinguishable from a real one.

    NaN is how the rest of this module says "not usable", and the agreement statistic's own
    `_finite` filter excludes it.
    """
    lo, hi = -100.0, 100.0

    # Inside the band: unchanged, to the bit.
    assert MB.alt_apex_or_nan(12.5, lo, hi) == pytest.approx(12.5, abs=0.0)
    assert MB.alt_apex_or_nan(lo, lo, hi) == pytest.approx(lo, abs=0.0)     # ON the bound is in
    assert MB.alt_apex_or_nan(hi, lo, hi) == pytest.approx(hi, abs=0.0)

    # Outside it: NaN, not the bound.
    for overrun in (lo - 1e-9, hi + 1e-9, -2613.0, 5225.0):
        assert math.isnan(MB.alt_apex_or_nan(overrun, lo, hi)), (
            f"{overrun} was published as a number instead of being refused")

    # A non-finite alternate apex stays non-finite and is not turned into a bound.
    assert math.isnan(MB.alt_apex_or_nan(float("nan"), lo, hi))
    assert math.isnan(MB.alt_apex_or_nan(float("inf"), lo, hi))


def test_a_refused_alternate_apex_cannot_enter_the_agreement_statistic():
    """The consequence, at the layer where it bites. `summarize`'s cross-check
    builds its pairs from rows with a finite `z_bif_alt`; a NaN drops out, a clamped number does
    not. Both cohorts of rows are identical except for that one field.
    """
    def _rows(alt_for_d):
        return [{"case": c, "flag": "OK", "z_bif": z, "z_bif_alt": a, "iliac_top": 0.0}
                for c, z, a in (("a", 10.0, 10.5), ("b", 20.0, 20.5),
                                ("c", 30.0, 30.5), ("d", 40.0, alt_for_d))]

    clean = MB.summarize(_rows(40.5))["cross_check"]
    refused = MB.summarize(_rows(float("nan")))["cross_check"]
    assert clean["n"] == 4 and refused["n"] == 3, (clean, refused)

    # And the number a clamp would have produced is NOT harmless: it moves the headline bias.
    clamped = MB.summarize(_rows(-100.0))["cross_check"]
    assert clamped["n"] == 4
    assert abs(clamped["bias"] - clean["bias"]) > 1.0, (
        "a clamped alternate apex must be shown to move the agreement statistic, or this "
        "test is not demonstrating why refusing it matters")


def test_the_refusal_is_wired_into_measure_bif_one_not_only_available_to_it(
        tmp_path, monkeypatch):
    """The function-level tests exercise `alt_apex_or_nan` directly; this test checks that
    `measure_bif_one` calls it, by making the alternate apex overrun the band.

    The primary call is passed through untouched, so the primary's own behaviour is unchanged and
    only the alternate is under test."""
    yc, x0 = 24, 24
    shape = (24, 48, 48)
    ivc = np.zeros(shape, bool)
    larr = np.zeros(shape, bool)
    rarr = np.zeros(shape, bool)

    def _blk(vol, z, y, x, s=3):
        h = s // 2
        vol[z, y - h:y + h + 1, x - h:x + h + 1] = True

    for k in range(12):
        sep = 14.0 - 0.8 * k
        _blk(larr, k, yc, int(round(x0 - sep / 2)))
        _blk(rarr, k, yc, int(round(x0 + sep / 2)))
    for k in range(15, 19):
        ivc[k, yc - 1:yc + 2, x0 - 1:x0 + 2] = True

    seg = tmp_path / "seg"
    seg.mkdir()
    _write(seg / "inferior_vena_cava.nii.gz", ivc)
    _write(seg / "iliac_vena_left.nii.gz", larr)
    _write(seg / "iliac_vena_right.nii.gz", rarr)

    baseline = MB.measure_bif_one(tmp_path, segdir="seg")
    assert math.isfinite(baseline["z_bif_alt"]), (
        "the fixture must produce a usable alternate apex first, or the test below proves "
        "nothing about refusing one")

    real = MB.S.wall_contact_apex
    calls = {"n": 0}

    def _first_call_overruns(*args, **kwargs):
        out = real(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 1:
            out = dict(out)
            out["z_bif"] = -2613.0                 # far outside the band
        return out

    monkeypatch.setattr(MB.S, "wall_contact_apex", _first_call_overruns)
    overrun = MB.measure_bif_one(tmp_path, segdir="seg")

    assert calls["n"] >= 2, "both apex methods must be computed"
    assert math.isnan(overrun["z_bif_alt"]), (
        f"an alternate apex of -2613 mm was published as {overrun['z_bif_alt']!r}; a clamped "
        f"value is indistinguishable from a real one in the results table")
    assert overrun["z_bif"] == pytest.approx(baseline["z_bif"], abs=1e-9)
    assert overrun["flag"] == baseline["flag"]


def _rows_with_offset(sign, n=40):
    """Minimal summarize-able rows whose gap centre sits cranial (+) or caudal (-) of apex."""
    rows = []
    for i in range(n):
        z_bif = 100.0
        half = 5.0
        centre = z_bif + sign * 10.0          # gap centre 10 mm from the apex, chosen side
        iliac_top, ivc_bottom = centre - half, centre + half
        rows.append({
            "case": f"c{i:03d}", "z_bif": z_bif, "slope": -1.0, "r2": 0.99,
            "n_slices": 12, "flag": "OK", "apex_method": "wallgap",
            "z_gap": ivc_bottom - iliac_top,
            "ivc_overshoot": ivc_bottom - z_bif,
            "iliac_undershoot": z_bif - iliac_top,
            "gap_center_offset": 0.5 * (ivc_bottom + iliac_top) - z_bif,
            "frac_above": (ivc_bottom - z_bif) / (ivc_bottom - iliac_top),
            "apex_in_gap": 1.0 if iliac_top <= z_bif <= ivc_bottom else 0.0,
        })
    return rows


def _console(rows):
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        MB.print_summary(rows)
    return buf.getvalue()


def test_the_printed_gap_direction_follows_the_measured_sign():
    """The sign convention is DERIVED here rather than copied: `gap_center_offset =
    0.5*(ivc_bottom + iliac_top) - z_bif` with larger z cranial, so a gap built entirely
    ABOVE the apex must produce a positive offset and the word "cranial". The fixture asserts
    that construction, so the test cannot silently agree with a flipped convention.
    """
    cranial_rows = _rows_with_offset(+1)
    assert cranial_rows[0]["gap_center_offset"] > 0, (
        "the fixture's own arithmetic disagrees with the offset definition; re-derive it")
    out = _console(cranial_rows)
    assert "cranial of the apex" in out
    assert "caudal" not in out.split("apex falls INSIDE")[1], (
        "the apex_in_gap line says 'caudal' on data whose offset is positive")
    assert "in THESE data" in out, (
        "the direction must be stated as this run's measurement, not as a general fact")

    caudal_rows = _rows_with_offset(-1)
    assert caudal_rows[0]["gap_center_offset"] < 0
    out = _console(caudal_rows)
    assert "caudal of the apex" in out, (
        "the sentence must track the sign in the caudal direction too")
    assert "cranial of the apex" not in out


def _rows_for_intervals(spread):
    """Rows whose six printed medians have collapsed intervals (`spread=False`: every value tied)
    or honest ones (`spread=True`). The apex sits inside the gap, so `frac_above` is printed too."""
    rows = []
    for i in range(40):
        jitter = 0.25 * (i % 9) if spread else 0.0
        # the three move at different rates, so no printed quantity (gap, overshoot, undershoot,
        # offset, fraction) is constant by cancellation when `spread` is set
        z_bif, iliac_top, ivc_bottom = 108.0 - jitter, 105.0 - 0.5 * jitter, 115.0 + jitter
        rows.append({
            "case": f"c{i:03d}", "z_bif": z_bif, "slope": -1.0, "r2": 0.99,
            "n_slices": 12, "flag": "OK", "apex_method": "wallgap",
            "z_gap": ivc_bottom - iliac_top,
            "ivc_overshoot": ivc_bottom - z_bif,
            "iliac_undershoot": z_bif - iliac_top,
            "gap_center_offset": 0.5 * (ivc_bottom + iliac_top) - z_bif,
            "frac_above": (ivc_bottom - z_bif) / (ivc_bottom - iliac_top),
            "apex_in_gap": 1.0,
            "alpha2_deg": 45.0 + 2.0 * jitter, "alpha1_deg": 30.0 + jitter,
        })
    return rows


def test_print_summary_never_prints_a_collapsed_interval_as_m_to_m():
    """The median zone height can sit on an atom of whole-slice values, so its bootstrap interval
    collapses, and `m-m` would read as a measurement without uncertainty. Every
    interval `print_summary` prints goes through `statlib.format_ci`, which prints the token for a
    point and keeps each site's own text (`+a,+b` for the signed offset) for a real interval."""
    import re

    out = _console(_rows_for_intervals(spread=False))
    lines = [ln for ln in out.splitlines() if "95%CI" in ln]
    labels = ("SIGNED gap offset", "IVC overshoot", "iliac undershoot", "transition zone",
              "alpha2 (LCIV-IVC", "frac of gap above apex")
    for label in labels:
        hit = [ln for ln in lines if label in ln]
        assert len(hit) == 1, f"expected one printed interval for {label!r}:\n{out}"
        assert "(95%CI DEGENERATE_ZERO_WIDTH)" in hit[0], (
            f"a collapsed interval must print as the token, not as m-m: {hit[0]!r}")
    point = re.compile(r"95%CI ([+-]?\d+(?:\.\d+)?)[-,]\1\)").search(out)
    assert point is None, f"an interval printed as a point: {point and point.group(0)!r}"

    honest = _console(_rows_for_intervals(spread=True))
    found = re.findall(r"\(95%CI ([^)]*)\)", honest)
    assert len(found) == len(labels), f"expected {len(labels)} printed intervals:\n{honest}"
    assert "DEGENERATE_ZERO_WIDTH" not in honest
    offset = next(ln for ln in honest.splitlines() if "SIGNED gap offset" in ln)
    assert re.search(r"\(95%CI \+\d+\.\d,\+\d+\.\d\)", offset), (
        f"a real offset interval must keep its signed `+a,+b` text: {offset!r}")
    for text in found:
        ends = re.match(r"([+-]?[\d.]+)[-,]([+-]?[\d.]+)$", text)
        assert ends is not None, f"a real interval must print two numeric ends: {text!r}"
        assert float(ends.group(1)) < float(ends.group(2)), f"ends out of order: {text!r}"


def test_print_summary_prints_each_interval_at_its_own_precision_and_na_when_absent(monkeypatch):
    import statlib as S
    rows = _rows_for_intervals(spread=True)
    s = MB.summarize(rows)
    sites = {
        "SIGNED gap offset": S.format_ci(*s["ci_offset"], 1, sep=",", signed=True),
        "IVC overshoot": S.format_ci(*s["ci_overshoot"], 1),
        "iliac undershoot": S.format_ci(*s["ci_undershoot"], 1),
        "transition zone": S.format_ci(*s["ci_zone"], 1),
        "alpha2 (LCIV-IVC": S.format_ci(*s["ci_alpha2"], 0),
        "frac of gap above apex": S.format_ci(*s["ci_frac_above"], 2),
    }
    out = _console(rows).splitlines()
    for label, text in sites.items():
        assert text and text != "DEGENERATE_ZERO_WIDTH", (label, text)       # fixture premise
        line = next(ln for ln in out if label in ln)
        assert f"(95%CI {text})" in line, (label, text, line)

    nan = float("nan")
    monkeypatch.setattr(MB.S, "median_ci_auto", lambda *args, **kw: (nan, nan))
    absent = [ln for ln in _console(rows).splitlines() if "95%CI" in ln]
    assert len(absent) == len(sites) and all("(95%CI NA)" in ln for ln in absent), absent


def test_the_gap_offset_sign_convention_is_pinned_at_its_producer():
    """`test_the_printed_gap_direction_follows_the_measured_sign` asserts that the CONSOLE tracks the
    row values; its fixture computes `gap_center_offset` inline, so it cannot pin the row values
    themselves.

    This pins the convention where it is defined, from geometry rather than from the
    implementation: `z` increases CRANIALLY, so a labelled gap lying entirely ABOVE the apex
    must yield a POSITIVE `gap_center_offset`, an overshoot larger than the gap, and an apex
    outside the gap.
    """
    # apex at 100; the gap occupies [110, 120], i.e. wholly CRANIAL of the apex
    out = MB.apex_relative(ivc_bottom=120.0, iliac_top=110.0, z_bif=100.0)
    assert out["z_gap"] == pytest.approx(10.0)
    assert out["gap_center_offset"] > 0, (
        "a gap lying entirely cranial of the apex produced a non-positive "
        "gap_center_offset; z increases cranially, so the convention has flipped and every "
        "printed direction is backwards")
    assert out["gap_center_offset"] == pytest.approx(15.0)
    assert out["apex_in_gap"] == 0.0

    # mirrored: the gap wholly CAUDAL of the apex must be negative, same magnitude
    below = MB.apex_relative(ivc_bottom=90.0, iliac_top=80.0, z_bif=100.0)
    assert below["gap_center_offset"] == pytest.approx(-15.0)

    # and an apex exactly at the gap centre is zero, which is the hinge the words turn on
    centred = MB.apex_relative(ivc_bottom=105.0, iliac_top=95.0, z_bif=100.0)
    assert centred["gap_center_offset"] == pytest.approx(0.0)
    assert centred["apex_in_gap"] == 1.0


# apex_level keeps z_to_vert_level's in-range flag: np.interp clamps, so an out-of-range apex would
# otherwise read as the boundary level, and the OUT_OF_FOV test inspects only the two gap
# endpoints.


def _lumbar_ruler():
    """A realistic L1..S1 ruler: physical z decreases as the vertebral level increases."""
    return MB.build_vert_ruler([(1.0, 150.0), (2.0, 120.0), (3.0, 90.0),
                                (4.0, 60.0), (5.0, 30.0), (6.0, 0.0)])


def test_z_to_vert_level_reports_out_of_range_and_the_level_it_returns_is_clamped():
    """The ruler's own contract, pinned so the caller's obligation is unambiguous.

    `np.interp` clamps, so the returned level for an out-of-range z is a real-looking boundary
    level, not a sentinel. That is precisely why the second element of the tuple is not optional.
    """
    ruler = _lumbar_ruler()
    lvl_in, ok_in = MB.z_to_vert_level(ruler, 45.0)
    assert ok_in and 4.0 < lvl_in < 5.0

    lvl_hi, ok_hi = MB.z_to_vert_level(ruler, 170.0)      # cranial of L1
    lvl_lo, ok_lo = MB.z_to_vert_level(ruler, -20.0)      # caudal of S1
    assert (ok_hi, ok_lo) == (False, False)
    assert lvl_hi == 1.0 and lvl_lo == 6.0, (
        "np.interp clamps; an out-of-range z returns the boundary level, which is why "
        "publishing it without the flag is indistinguishable from a real measurement")


def test_an_apex_outside_the_labelled_vertebrae_is_refused_not_clamped(tmp_path):
    """The published quantity is the at-L5 apex rate over `vert_qc == OK` rows.

    Constructed so that the two GAP endpoints are inside the ruler and the APEX is not -- the
    configuration the OUT_OF_FOV test (gap endpoints only) cannot see, and a real one, because the
    apex is an extrapolation beyond the cranial-most CIV slice. `summarize` filters `apex_level` on
    `_finite`, so refusing removes the case from the rate; a clamped level would keep it in.
    """
    ruler = _lumbar_ruler()
    ivc_bottom, iliac_top, z_bif = 45.0, 35.0, 170.0
    _, in_top = MB.z_to_vert_level(ruler, ivc_bottom)
    _, in_bot = MB.z_to_vert_level(ruler, iliac_top)
    apex_lvl, in_apex = MB.z_to_vert_level(ruler, z_bif)
    assert in_top and in_bot and not in_apex, "fixture must isolate the apex"
    assert apex_lvl == 1.0, "the clamped value is a plausible level, which is the whole problem"

    # the caller's rule
    published = apex_lvl if in_apex else float("nan")
    assert math.isnan(published), (
        "an apex outside the vertebral ruler must be refused (NaN), not published as the "
        "clamped boundary level")

    # and the source must keep the in-range flag
    from pathlib import Path
    src = Path(MB.__file__).read_text(encoding="utf-8")
    assert 'z_to_vert_level(ruler, primary["z_bif"])[0]' not in src, (
        "the apex level is taken without its in-range flag")


def test_the_gap_levels_keep_their_own_out_of_range_determination():
    """The gap levels keep their own range check.

    The caller checks `in_top` and `in_bot` and sets `vert_qc = OUT_OF_FOV`, and `vert_qc`
    deliberately stays OK when only the apex is out of range, because the two gap levels are
    valid for that case.
    """
    from pathlib import Path
    src = Path(MB.__file__).read_text(encoding="utf-8")
    assert 'if vqc == "OK" and not (in_top and in_bot):' in src
    assert '"OUT_OF_FOV"' in src


def _ref_csv(tmp_path, text, name="ref.csv"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_a_duplicate_case_in_the_reference_is_refused_not_resolved_by_file_order(tmp_path):
    """Two values for one case is not a reference standard.

    Resolving duplicates by file order would be a silent, undeclared rule, and the reference
    compared with the apex proxy could then differ between two orderings of the same rows, so a
    duplicate raises.
    """
    path = _ref_csv(tmp_path, "case,ref_z\namos_0001,100.0\namos_0001,999.0\n")
    with pytest.raises(ValueError, match="more than one row"):
        MB.load_reference(path)


def test_non_finite_and_empty_reference_rows_are_dropped_and_counted(tmp_path):
    """`float("nan")` and `float("inf")` both parse, so a ValueError handler alone would admit
    them."""
    path = _ref_csv(tmp_path, "case,ref_z\na,nan\nb,inf\nc,-inf\n,50.0\nd,120.0\n")
    ref = MB.load_reference(path)
    assert ref == {"d": 120.0}, (
        f"only the finite, named row may survive; got {ref}")


def test_a_reference_file_without_the_required_columns_is_refused(tmp_path):
    """An empty dict would surface downstream as "insufficient matched reference cases", which reads
    as a small reference rather than the wrong file entirely."""
    with pytest.raises(ValueError, match="missing required column"):
        MB.load_reference(_ref_csv(tmp_path, "case,vert\namos_0001,L5\n"))


def test_the_reported_matched_n_is_the_n_the_statistics_actually_used():
    """`print_validation` shows `matched n={val['n_matched']}` beside a bias, LoA, CCC and ICC from
    `method_agreement`, which drops non-finite pairs internally and reports its own `n`.

    The reference dict is built directly here rather than through `load_reference`, deliberately."""
    # a non-finite reference value reaches the statistics
    ref = {"c": 120.0, "d": 138.0, "e": float("nan")}
    rows = [{"case": c, "flag": "OK", "z_bif": z, "iliac_top": 10.0}
            for c, z in (("c", 130.0), ("d", 140.0), ("e", 150.0))]
    out = MB.validate_against_reference(rows, ref)
    assert out["n"] == 2, "fixture must make method_agreement drop exactly one pair"
    assert out["n_matched"] == out["n"], (
        f"reported matched n={out['n_matched']} but the statistics used n={out['n']}; the "
        f"printed sample size must be the sample the numbers came from")


def test_level_histogram_uses_the_same_bin_edges_as_the_at_l5_rate():
    """A case at the L4/L5 disc must not be L4 in one printed line and L5 in the next.

    Python's `round()` rounds half to EVEN, while the at-L5 rate is the half-open interval
    `[4.5, 5.5)`, which INCLUDES 4.5; the two rules disagree at every `k.5` with `k` even and agree
    where `k` is odd, so both parities are asserted.
    """
    def bankers(v):
        return int(round(v))

    def binned(v):
        return MB.level_bin(v)

    # The property, stated as the relation rather than as a table of expected values: the bin a
    # level falls in must be the k whose half-open interval [k-0.5, k+0.5) contains it -- the
    # same rule `pct_l5` uses.
    for v in (2.5, 3.5, 4.5, 5.5, 6.5, 4.4999, 4.5001, -0.5, 0.5):
        k = binned(v)
        assert k - 0.5 <= v < k + 0.5, (
            f"level {v} was binned as {k}, whose interval [{k - 0.5}, {k + 0.5}) does not "
            f"contain it")

    # The at-L5 rate's membership test and the histogram's bin must agree, case by case.
    for v in (4.5, 4.9, 5.0, 5.4999, 5.5, 4.4999):
        in_l5_rate = (4.5 <= v < 5.5)
        assert (binned(v) == 5) == in_l5_rate, (
            f"level {v}: histogram bin {binned(v)} and at-L5 membership {in_l5_rate} disagree")

    # Both parities, pinned.
    assert binned(4.5) == 5 and bankers(4.5) == 4, "4.5 is the L4/L5 disc and belongs in L5"
    assert binned(2.5) == 3 and bankers(2.5) == 2, (
        "banker's rounding is parity-dependent; 2.5 disagrees too")
    assert binned(3.5) == 4 == bankers(3.5), "odd k agrees under both rules"

    # NEGATIVE levels are in the vocabulary (LEVEL_NAME[-1] == "T11"), and `int(v + 0.5)`
    # truncates TOWARD ZERO rather than flooring, so it disagrees there while looking correct
    # on every positive case.
    for v in (-0.7, -1.2, -1.5, -0.2):
        k = binned(v)
        assert k - 0.5 <= v < k + 0.5, (
            f"negative level {v} binned as {k}: interval [{k - 0.5}, {k + 0.5}) excludes it")


def test_the_printed_histogram_and_at_l5_rate_agree_on_a_disc_level_case():
    """The CALL SITE, not the helper: this drives the real printout and compares the two lines a
    reader compares.

    `pct_l5`'s own interval is pinned too, since moving its lower edge to `> 4.5` would make the two
    lines agree by breaking the rate instead of the histogram.
    """
    import io
    import contextlib

    def row(level):
        return {"flag": "OK", "vert_qc": "OK", "apex_level": level,
                "gap_top_level": float("nan"), "gap_bottom_level": float("nan")}

    # Two cases at the L4/L5 disc and two unambiguously at L5.
    rows = [row(4.5), row(4.5), row(5.0), row(5.0)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        MB.print_summary(rows)
    out = buf.getvalue()

    assert "L5:100%" in out.replace(" ", ""), (
        "all four cases are in [4.5, 5.5) so the histogram must be 100 % L5; the printed "
        f"distribution disagrees with the at-L5 rate:\n{out}")
    assert "at-L5: 100%" in out, f"the at-L5 rate must also be 100 %:\n{out}"

    # BOTH edges of the interval: an upper edge of `<= 5.5` would count the L5/S1 disc as L5. 5.5
    # belongs to bin 6, outside the at-L5 rate.
    edge = [row(4.5), row(5.5)]
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        MB.print_summary(edge)
    out2 = buf2.getvalue()
    assert "at-L5: 50%" in out2, (
        f"4.5 is inside [4.5, 5.5) and 5.5 is outside it, so the rate must be 50 %:\n{out2}")
    dist2 = out2.split("level distribution ->")[1].splitlines()[0].replace(" ", "")
    assert "L5:50%" in dist2, f"the 4.5 case must bin as L5 and the 5.5 case must not: {dist2!r}"
    # And the L4 bin must be empty.
    dist = out.split("level distribution ->")[1].splitlines()[0]
    assert "L4" not in dist, f"a [4.5, 5.5) case is binned as L4: {dist!r}"


def test_the_row_formatter_writes_na_for_infinities_not_the_word_inf():
    """`_fmt` writes NA for any non-finite value: `f"{float('-inf'):.1f}"` is `'-inf'`, which
    `float()` parses back into a number, whereas the module's own usability predicate `_finite`
    refuses infinities."""
    base = {"case": "c0", "window_fallback": 0, "oblique": 0}
    for bad in (float("inf"), float("-inf"), float("nan")):
        row = dict(base, z_bif=bad, z_bif_ci_lo=bad, z_bif_ci_hi=bad)
        fields = MB._fmt(row).split()
        assert "inf" not in " ".join(fields).lower(), (
            f"_fmt wrote an infinity into a table cell for {bad}: {fields}")
        # column 4 is z_bif (COLUMNS.index), asserted by name rather than by a magic index
        assert fields[MB.COLUMNS.index("z_bif")] == "NA", (
            f"a non-finite z_bif must be NA, got {fields[MB.COLUMNS.index('z_bif')]!r}")

    # A finite value must still be written -- otherwise this test would pass on a formatter
    # that emitted NA for everything.
    ok = MB._fmt(dict(base, z_bif=-235.4)).split()
    assert ok[MB.COLUMNS.index("z_bif")] == "-235.4"


def test_the_quadratic_apex_sentinel_is_infinite():
    """`_quad_apex_shift` returns +inf on purpose when the quadratic never reaches contact, and
    `_nonlinear_flag` keys off `shift == inf` to flag NONLINEAR unconditionally. Removing
    infinities everywhere would silently turn that fail-closed branch off.
    """
    assert MB._nonlinear_flag(False, float("inf"), 0.1) is True, (
        "the +inf apex-shift sentinel must force the NONLINEAR flag")
    assert MB._nonlinear_flag(False, 5.0, 0.1) is False


def _vert_seg(tmp_path, *, s1="absent"):
    """A seg dir with L3/L4/L5 + sacrum, and vertebrae_S1 in one of three states.

    `s1`: "absent" (no file), "empty" (a file whose mask is all False), "unreadable" (a file
    that is not a NIfTI at all), or "present" (a real, non-empty S1).
    """
    seg = tmp_path / "seg"
    seg.mkdir(parents=True, exist_ok=True)
    shape = (40, 8, 8)
    for name, z0 in (("vertebrae_L3", 30), ("vertebrae_L4", 24), ("vertebrae_L5", 18)):
        v = np.zeros(shape, bool)
        v[z0:z0 + 4, 2:6, 2:6] = True
        _write(seg / f"{name}.nii.gz", v)
    sac = np.zeros(shape, bool)
    sac[4:14, 2:6, 2:6] = True                     # sacrum below L5; its TOP is the S1 endplate
    _write(seg / "sacrum.nii.gz", sac)

    if s1 == "empty":
        _write(seg / "vertebrae_S1.nii.gz", np.zeros(shape, bool))
    elif s1 == "unreadable":
        (seg / "vertebrae_S1.nii.gz").write_bytes(b"not a nifti")
    elif s1 == "present":
        v = np.zeros(shape, bool)
        v[12:16, 2:6, 2:6] = True
        _write(seg / "vertebrae_S1.nii.gz", v)
    return seg


def test_an_empty_or_unreadable_s1_does_not_suppress_the_sacrum_fallback(tmp_path):
    """An S1 file that exists but is empty or corrupt adds no anchor, so it must not suppress the
    sacrum fallback; otherwise the ruler loses its caudal anchor and a case can fall to FEW_VERT
    and drop out of the vertebral-level analysis.

    This asserts the documented behaviour ("Robust to a missing or unreadable mask (skipped)").
    """
    levels = lambda seg: sorted(lvl for lvl, _ in MB.read_vertebra_anchors(seg))

    baseline = levels(_vert_seg(tmp_path / "a", s1="absent"))
    assert 5.5 in baseline, "with no S1 file at all the sacrum fallback must fire"
    assert 6.0 not in baseline

    for state in ("empty", "unreadable"):
        got = levels(_vert_seg(tmp_path / state, s1=state))
        assert got == baseline, (
            f"an {state} vertebrae_S1.nii.gz must behave exactly like an absent one; the "
            f"sacrum fallback produced {got} instead of {baseline}")
        assert 5.5 in got, f"the sacrum fallback did not fire for an {state} S1"

    # And it must NOT fire when a real S1 anchor exists, or the caudal end would be anchored
    # twice with two different conventions (S1 body centre 6.0 vs L5/S1 disc 5.5).
    real = levels(_vert_seg(tmp_path / "d", s1="present"))
    assert 6.0 in real and 5.5 not in real, (
        f"a real S1 must suppress the sacrum fallback, got {real}")


def test_the_ruler_survives_an_empty_s1_instead_of_falling_to_few_vert(tmp_path):
    """The consequence, not just the anchor count.

    `build_vert_ruler` needs >= 2 anchors; the anchor the fallback supplies is the caudal end
    of the scale. This drives the real pipeline rather than the helper.
    """
    for state in ("absent", "empty", "unreadable"):
        anchors = MB.read_vertebra_anchors(_vert_seg(tmp_path / f"r_{state}", s1=state))
        interp, z_min, z_max, qc = MB.build_vert_ruler(anchors)
        assert qc == "OK", f"an {state} S1 left the ruler at qc={qc}"
        assert interp is not None


# `contact_actual = sep - d_min` is not a reach along the contact direction: `d_min` is a minimum
# over EVERY direction, so the difference is not a reach along any single ray. `R = median(sep -
# d_min)` is the centroid separation AT CONTACT, and "when do the walls first touch" is exactly a
# minimum over all directions. These tests pin both halves.
# ---------------------------------------------------------------------------

def _rect2d(shape, cy, cx, h, w, ang_deg):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    a = math.radians(ang_deg)
    u = (xx - cx) * math.cos(a) + (yy - cy) * math.sin(a)
    v = -(xx - cx) * math.sin(a) + (yy - cy) * math.cos(a)
    return (np.abs(u) <= w / 2) & (np.abs(v) <= h / 2)


def _centroid2d(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()])


def _dmin2d(a, b):
    """Voxel-BOX surface distance, the same 3x3 dilation convention production uses."""
    from scipy.spatial import cKDTree
    from scipy.ndimage import binary_dilation
    ay, ax = np.nonzero(binary_dilation(a, structure=np.ones((3, 3), bool)))
    by, bx = np.nonzero(b)
    return float(cKDTree(np.c_[ax, ay]).query(np.c_[bx, by])[0].min())


def _directional_reach2d(a, b):
    """Combined extent of the two lumens ALONG the centroid axis."""
    ca, cb = _centroid2d(a), _centroid2d(b)
    u = cb - ca
    u = u / np.linalg.norm(u)
    def reach(mask, c, direction):
        ys, xs = np.nonzero(mask)
        return float(((np.c_[xs, ys] - c) @ direction).max())
    return reach(a, ca, u) + reach(b, cb, -u)


@pytest.mark.parametrize("name,ang_l,ang_r,expected_pct", [
    ("parallel 45deg offset along x", 45, 45, -32.3),
    ("60/20 deg", 60, 20, -17.9),
])
def test_sep_minus_dmin_is_not_a_reach_along_the_contact_direction(
        name, ang_l, ang_r, expected_pct):
    """The two quantities diverge badly once the lumens are oblique.

    The figures are RE-DERIVED here rather than quoted, so the comment in
    `measure_bifurcation.py` cannot drift away from what the geometry does.
    """
    shape = (120, 120)
    left = _rect2d(shape, 60, 40, 10, 40, ang_l)
    right = _rect2d(shape, 60, 80, 10, 40, ang_r)
    sep = float(np.linalg.norm(_centroid2d(right) - _centroid2d(left)))
    contact_actual = sep - _dmin2d(left, right)
    directional = _directional_reach2d(left, right)

    assert sep == pytest.approx(40.0, abs=1e-9), "the fixture's centroids must be 40 mm apart"
    pct = 100.0 * (contact_actual - directional) / directional
    assert pct == pytest.approx(expected_pct, abs=0.15), (
        f"{name}: sep-d_min={contact_actual:.3f} vs directional={directional:.3f} "
        f"({pct:.1f}%); the documented divergence has moved")
    assert contact_actual < directional, (
        "the divergence is SIGNED: sep-d_min UNDER-states the directional reach, and a "
        "remedy designed from the opposite sign would correct the wrong way")


def test_the_estimator_is_right_for_what_it_actually_solves():
    """The property `R = median(sep - d_min)` depends on.

    At contact `d_min` is 0 and `contact_actual` collapses to `sep` exactly, so `R` IS the
    centroid separation at which the walls touch. A positive control that must hold in both
    directions: before contact, `d_min > 0` and `contact_actual < sep`.
    """
    shape = (120, 120)
    touching_l = _rect2d(shape, 60, 40, 10, 30, 0)
    touching_r = _rect2d(shape, 60, 71, 10, 30, 0)      # face-adjacent
    sep_t = float(np.linalg.norm(_centroid2d(touching_r) - _centroid2d(touching_l)))
    assert _dmin2d(touching_l, touching_r) == pytest.approx(0.0, abs=1e-9), (
        "the fixture must actually touch, or this proves nothing")
    assert sep_t - _dmin2d(touching_l, touching_r) == pytest.approx(sep_t, abs=1e-9)

    apart_r = _rect2d(shape, 60, 90, 10, 30, 0)          # clearly separated
    sep_a = float(np.linalg.norm(_centroid2d(apart_r) - _centroid2d(touching_l)))
    d_apart = _dmin2d(touching_l, apart_r)
    assert d_apart > 0.0, "the separated control must NOT touch"
    assert sep_a - d_apart < sep_a


# A case this driver cannot measure returns `MISSING`, `GRID_MISMATCH`, `EMPTY` or `ERROR:<type>`,
# and every measured outcome carries a finite `z_gap`, `NO_RIGHT` and `FEW_SLICES` included.
# `main()` refuses a table in which nothing was measured; on a flat export, where the nested
# `SEGDIR` finds nothing, every row is `MISSING`.

def _bif_case(root, name, with_lciv=True, flat=False):
    seg = root / name if flat else root / name / MB.SEGDIR
    seg.mkdir(parents=True, exist_ok=True)
    ivc = np.zeros((60, 24, 24), np.uint8)
    lciv = np.zeros((60, 24, 24), np.uint8)
    ivc[35:60, 10:14, 10:14] = 1
    lciv[0:25, 10:14, 10:14] = 1
    for fname, arr in ((MB.IVC, ivc), (MB.LIL, lciv)):
        if fname == MB.LIL and not with_lciv:
            continue
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(img, str(seg / fname))


def _converging_driver_case(root, name):
    """The geometry `test_wall_contact_apex_is_caudal_to_centroid_coincidence` hand-derives: two
    3x3 iliac blocks converging over 12 slices under a cranial IVC, which fits with flag `OK`
    and a finite apex."""
    Z, Y, X = 20, 40, 80
    larr = np.zeros((Z, Y, X), bool)
    rarr = np.zeros((Z, Y, X), bool)
    ivc = np.zeros((Z, Y, X), bool)
    yc, x0 = 20, 40
    for k in range(12):
        sep = 14.0 - 0.8 * k
        _block(larr, k, yc, round(x0 - sep / 2))
        _block(rarr, k, yc, round(x0 + sep / 2))
    for k in range(15, 19):
        ivc[k, yc - 1:yc + 2, x0 - 1:x0 + 2] = True
    seg = root / name / MB.SEGDIR
    seg.mkdir(parents=True, exist_ok=True)
    _write(seg / MB.IVC, ivc)
    _write(seg / MB.LIL, larr)
    _write(seg / MB.RIL, rarr)

def _run_bifurcation(monkeypatch, tmp_path, roster_ids, extra=()):
    import sys
    roster = tmp_path / "roster.txt"
    roster.write_text("".join(f"{c}\n" for c in roster_ids), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(MB, "DATA", None)
    monkeypatch.setattr(sys, "argv", [
        "measure_bifurcation.py", "--data-dir", str(tmp_path / "data"), "--prefix", "probe",
        "--roster", str(roster), "--jobs", "1", *extra])
    return MB.main()


def _bifurcation_rows(path):
    return [ln.split() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#") and ln.split()[0] != "case"]


def _bif_flavour_case(root, name, flavour):
    """A case that `measure_bif_one` cannot measure, in one of its four ways: `MISSING` (the
    flat export, read with the nested `SEGDIR`), `EMPTY` (masks with no foreground),
    `GRID_MISMATCH` (an LCIV on another grid) and `ERROR:RuntimeError` (bytes that are not a
    NIfTI, through `work`'s handler)."""
    if flavour == "MISSING":
        return _bif_case(root, name, flat=True)
    seg = root / name / MB.SEGDIR
    seg.mkdir(parents=True, exist_ok=True)
    if flavour == "ERROR:RuntimeError":
        for fname in (MB.IVC, MB.LIL):
            (seg / fname).write_bytes(b"not a NIfTI")
        return
    ivc = np.zeros((60, 24, 24), np.uint8)
    lciv = np.zeros((60, 24, 24), np.uint8)
    if flavour == "GRID_MISMATCH":
        ivc[35:60, 10:14, 10:14] = 1
        lciv[0:25, 10:14, 10:14] = 1
    for fname, arr, sz in ((MB.IVC, ivc, 1.0), (MB.LIL, lciv, 2.0 if flavour == "GRID_MISMATCH" else 1.0)):
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((1.0, 1.0, sz))
        sitk.WriteImage(img, str(seg / fname))


@pytest.mark.parametrize("flavour, limit", [
    ("MISSING", 0), ("EMPTY", 0), ("GRID_MISMATCH", 0), ("ERROR:RuntimeError", 0), ("MISSING", 1)])
def test_the_bifurcation_driver_publishes_nothing_when_nothing_was_measured(
        tmp_path, monkeypatch, flavour, limit):
    """Guards against a `main()` with no postcondition (one failure row per roster case, exit 0)
    and against any rule keyed to one flag: each of the four ways a case measures
    nothing is a fixture here, proven to be that way by the flag tally the refusal prints. A
    `--limit` quick test that measured nothing is refused as well, under its own file name. The
    flat-export hint belongs to an all-`MISSING` table only."""
    for c in ("case_a", "case_b"):
        _bif_flavour_case(tmp_path / "data", c, flavour)
    extra = ("--limit", str(limit)) if limit else ()
    with pytest.raises(SystemExit) as exc:
        _run_bifurcation(monkeypatch, tmp_path, ["case_a", "case_b"], extra=extra)
    message = str(exc.value)
    assert exc.value.code not in (None, 0), f"a table of {flavour} rows exited successfully"
    assert "nothing was measured" in message
    assert f"{flavour}={limit or 2}" in message, message
    assert ("flat 2.17.0 export" in message) == (flavour == "MISSING"), message
    assert not any(tmp_path.glob("probe_results_bifurcation*")), list(tmp_path.glob("probe_*"))


def test_the_bifurcation_driver_publishes_a_partial_failure_and_a_refusal_keeps_the_table(
        tmp_path, monkeypatch):
    """Both directions. One case with an apex (two iliacs converging under the IVC, flag `OK`)
    after a `MISSING` one publishes; once that case's LCIV is gone nothing is measured, and the
    refusal leaves the previous table byte-identical. The measured case is the LAST row, so a
    rule that looks at the first row only cannot pass."""
    (tmp_path / "data" / "case_a" / MB.SEGDIR).mkdir(parents=True)    # nothing to measure
    _converging_driver_case(tmp_path / "data", "case_b")                      # the measured case, LAST
    _run_bifurcation(monkeypatch, tmp_path, ["case_a", "case_b"])
    out = tmp_path / "probe_results_bifurcation.txt"
    rows = _bifurcation_rows(out)
    flag, z_gap = MB.COLUMNS.index("flag"), MB.COLUMNS.index("z_gap")
    z_bif = MB.COLUMNS.index("z_bif")
    assert [(r[0], r[flag]) for r in rows] == [("case_a", "MISSING"), ("case_b", "OK")]
    assert rows[0][z_gap] == "NA" and rows[1][z_gap] != "NA"
    assert rows[0][z_bif] == "NA" and rows[1][z_bif] != "NA"
    before = out.read_bytes()

    (tmp_path / "data" / "case_b" / MB.SEGDIR / MB.LIL).unlink()
    with pytest.raises(SystemExit) as exc:
        _run_bifurcation(monkeypatch, tmp_path, ["case_a", "case_b"])
    assert exc.value.code not in (None, 0), "a table of MISSING rows exited successfully"
    assert "nothing was measured" in str(exc.value)
    assert out.read_bytes() == before, "the refusal replaced the previous complete table"


def test_the_bifurcation_driver_refuses_a_table_in_which_no_apex_was_estimated(tmp_path, monkeypatch):
    """The table exists to locate the confluence. A cohort whose every case lacks the right
    iliac (a subset export, a renamed file) measures each gap and angle and estimates no apex:
    every row `NO_RIGHT`, `z_bif` NA. Guards against a postcondition on `z_gap` alone, which
    would publish that table with exit 0. The flat-export hint must not be offered here: the masks
    were read."""
    for c in ("case_a", "case_b"):
        _bif_case(tmp_path / "data", c)                  # IVC and LCIV only: NO_RIGHT
    with pytest.raises(SystemExit) as exc:
        _run_bifurcation(monkeypatch, tmp_path, ["case_a", "case_b"])
    assert exc.value.code not in (None, 0), "a table with no apex exited successfully"
    assert "no apex was estimated" in str(exc.value)
    assert "NO_RIGHT=2" in str(exc.value) and "flat 2.17.0 export" not in str(exc.value)
    assert not (tmp_path / "probe_results_bifurcation.txt").exists()


def test_a_malformed_reference_is_refused_before_the_table_is_replaced(tmp_path, monkeypatch):
    """Guards against reading the reference after the `AtomicTable` block (the table must be
    byte-identical to the previous run's)."""
    _converging_driver_case(tmp_path / "data", "case_a")
    _run_bifurcation(monkeypatch, tmp_path, ["case_a"])
    out = tmp_path / "probe_results_bifurcation.txt"
    before = out.read_bytes()
    _converging_driver_case(tmp_path / "data", "case_b")
    ref = tmp_path / "ref.csv"
    ref.write_text("case,ref_z\ncase_a,13.0\ncase_a,14.0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _run_bifurcation(monkeypatch, tmp_path, ["case_a", "case_b"], extra=("--reference", str(ref)))
    assert out.read_bytes() == before, "a malformed reference was refused only after publication"
