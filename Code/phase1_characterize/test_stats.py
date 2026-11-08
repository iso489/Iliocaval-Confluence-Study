import ast
import csv
import re
from pathlib import Path

import numpy as np
import pytest

import gap_config as C
import statlib as S
import stats as ST


ROOT = Path(__file__).resolve().parents[2]


V2_HEADER_COLS = "case ivc_z lciv_z z_gap surf_3d raw_z_gap c2c_3d qc"


def _write(tmp_path, name, lines):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return p


def test_analysis_input_identity_is_stable_to_generated_outputs(tmp_path):
    code = tmp_path / "Code" / "common"
    phase1 = tmp_path / "Code" / "phase1_characterize"
    phase0 = tmp_path / "Code" / "phase0_data"
    metadata = tmp_path / "METADATA"
    cohort = tmp_path / "Results" / "phase1_characterize" / "AMOS"
    for directory in (code, phase1, phase0, metadata, cohort):
        directory.mkdir(parents=True, exist_ok=True)
    (code / "gap_config.py").write_text("PRESENT_MM = 0.5\n", encoding="utf-8")
    (phase1 / "stats.py").write_text("# analysis\n", encoding="utf-8")
    (phase0 / "amos_ct_cases.txt").write_text("amos_0001 imagesTr\n", encoding="utf-8")
    (metadata / "AMOS.csv").write_text("amos_id\n1\n", encoding="utf-8")
    raw = cohort / "amos_results_fullres.txt"
    raw.write_text("amos_0001 1 1 0 0 0 1 OK\n", encoding="utf-8")
    generated = cohort / "provenance.txt"
    generated.write_text("generated_utc: first\n", encoding="utf-8")

    first = ST._analysis_input_identity(tmp_path)
    generated.write_text("generated_utc: second\n", encoding="utf-8")
    assert ST._analysis_input_identity(tmp_path) == first

    raw.write_text("amos_0001 1 1 0 2 0 3 OK\n", encoding="utf-8")
    assert ST._analysis_input_identity(tmp_path)[0] != first[0]


# --- load_fullres: v2 format ----------------------------------------------
def test_load_fullres_v2(tmp_path):
    p = _write(tmp_path, "fr.txt", [
        "case_000 210 185 15.00 12.53 +15.00 15.04 OK",
        "case_001 ERROR: empty mask",
        "case_002 165 3 9.00 11.80 +9.00 13.40 SMALL_LCIV",
    ])
    rows, counts = ST.load_fullres(p)
    assert counts["measurable"] == 2 and counts["empty"] == 1
    assert counts["qc_small"] == 1 and counts["metric"] == "surface(v2)"
    case, z, surf, c2c, qc = rows[0]
    assert (case, z, surf, c2c) == ("case_000", 15.0, 12.53, 15.04)


def test_load_fullres_counts_overlap(tmp_path):
    # OVERLAP cases (3D label overlap, surf forced 0) must be counted so they are
    # not silently biasing the median/present-rate
    p = _write(tmp_path, "fr.txt", [
        "case_000 210 185 15.00 12.53 +15.00 15.04 OK",
        "case_003 200 180 0.00 0.00 -5.00 0.00 OVERLAP",
        "case_004 200 180 0.00 0.00 -5.00 0.00 SMALL_IVC+OVERLAP",
    ])
    rows, counts = ST.load_fullres(p)
    assert counts["qc_overlap"] == 2
    assert counts["qc_small"] == 1          # the combined-flag row also counts SMALL


def test_load_fullres_reads_the_v1_format(tmp_path):
    p = _write(tmp_path, "fr.txt", [
        "case_000 210 185 15.0 15.0 +15.0 OK",
        "case_002 215 175 5.0 5.0 +5.0 OK",
    ])
    rows, counts = ST.load_fullres(p)
    assert counts["metric"] == "c2c(v1)"
    _case, z, surf, c2c, qc = rows[0]
    assert surf == c2c == 15.0                       # v1: c2c falls back to col4


def test_total_accounting_is_closed(tmp_path):
    p = _write(tmp_path, "fr.txt", [
        "case_000 210 185 15.00 12.53 +15.00 15.04 OK",
        "case_001 ERROR: empty mask",
        "case_002 1 1 nan 12.0 +1.0 13.0 OK",        # non-finite -> unparsed
    ])
    rows, c = ST.load_fullres(p)
    assert c["measurable"] + c["empty"] + c["unparsed"] == 3
    assert c["unparsed"] == 1


def test_parse_does_not_silently_drop_bad_measurable_rows(tmp_path):
    p = _write(tmp_path, "fr.txt", [
        "case_000 210 185 15.00 12.53 +15.00 15.04 OK",
        "case_007 garbage row here",
    ])
    _rows, c = ST.load_fullres(p)
    assert c["unparsed"] == 1                          # counted, not dropped


def test_load_compare(tmp_path):
    p = _write(tmp_path, "cmp.txt", [
        "case_000 17.50 15.00 -2.50 49923 49405 OK",
        "case_001 empty mask",
    ])
    rows, counts = ST.load_compare(p)
    assert counts["measurable"] == 1 and counts["empty"] == 1
    assert rows[0] == ("case_000", 17.5, 15.0)


def test_load_spacing_drops_na(tmp_path):
    p = _write(tmp_path, "sp.txt", [
        "case_000 2.000 0.800 0.800",
        "case_001 NA NA NA",
        "case_002 3.000 0.900 0.900",
    ])
    d = ST.load_spacing(p)
    assert d == {"case_000": 2.0, "case_002": 3.0}


def test_load_spacing_missing_file_is_empty(tmp_path):
    assert ST.load_spacing(tmp_path / "nope.txt") == {}


def test_load_aux_and_inject_phase1_correlates(tmp_path):
    # aux file columns: case sz sy sx lcc n_comp body_csa
    lines = ["case_000 2.0 0.8 0.8 1.00 1 30000",
             "case_001 2.0 0.8 0.8 0.80 3 20000",       # fragmented + small
             "case_002 2.0 0.8 0.8 1.00 1 40000"]       # clean + large
    lines += [f"case_{i:03d} 2.0 0.8 0.8 1.00 1 {25000 + i*100}" for i in range(3, 33)]
    p = _write(tmp_path, "aux.txt", lines)
    aux = ST.load_aux(p)
    assert aux["case_001"]["n_comp"] == 3 and aux["case_001"]["lcc"] == pytest.approx(0.80)
    meta = {f"case_{i:03d}": dict(sex=None, age=None, scanner=None, contrast=None, region=None)
            for i in range(33)}
    ST._inject_aux_correlates(meta, aux)
    assert meta["case_001"]["seg_quality"] == "fragmented"   # n_comp>1
    assert meta["case_000"]["seg_quality"] == "clean"
    assert meta["case_002"]["size_band"] == "large"          # max CSA
    assert meta["case_001"]["size_band"] == "small"          # min CSA


def _aux_rows(n, csa):
    return {f"case_{i:03d}": {"sz": 2.0, "lcc": 1.0, "n_comp": 1.0, "csa": csa(i)} for i in range(n)}


def test_patient_size_availability_follows_the_injection_rule():
    """`_inject_aux_correlates` bands body-CSA tertiles only from `3 * MIN_GROUP_N` finite areas, and
    the area is read off the CT image, so a mask-only run has none. The provenance line must state the
    outcome by the SAME rule, so it is checked on both sides of the boundary against what the
    injection actually did."""
    need = 3 * C.MIN_GROUP_N
    nan = float("nan")
    assert ST.aux_correlate_availability(_aux_rows(1014, lambda i: nan)) == (
        f"patient_size (body-CSA tertiles) NOT COMPUTED: 0 of 1014 rows carry a finite body CSA and "
        f"banding needs {need}; the body CSA is read from the CT image (measure_ivc_lciv_gap.body_csa), "
        f"so a run without the CT leaves it NA")
    for n_finite, banded in ((need - 1, False), (need, True)):
        aux = _aux_rows(need + 5, lambda i: 20000.0 + 100.0 * i if i < n_finite else nan)
        meta = {cid: {} for cid in aux}
        ST._inject_aux_correlates(meta, aux)
        assert any("size_band" in rec for rec in meta.values()) is banded, n_finite
        expected = (
            f"patient_size (body-CSA tertiles) banded from {n_finite} of {need + 5} rows with a finite "
            f"body CSA") if banded else (
            f"patient_size (body-CSA tertiles) NOT COMPUTED: {n_finite} of {need + 5} rows carry a finite "
            f"body CSA and banding needs {need}; the body CSA is read from the CT image "
            f"(measure_ivc_lciv_gap.body_csa), so a run without the CT leaves it NA")
        assert ST.aux_correlate_availability(aux) == expected
    assert ST.aux_correlate_availability({}) == "no aux rows: seg_quality and patient_size were not computed"


def test_write_provenance_carries_the_aux_correlate_line(tmp_path):
    aux = _aux_rows(40, lambda i: float("nan"))
    path = tmp_path / "provenance.txt"
    ST.write_provenance(path, "autopet", aux=aux)
    assert [line for line in path.read_text().splitlines() if line.startswith("aux_correlates")] == [
        "aux_correlates: " + ST.aux_correlate_availability(aux)]
    ST.write_provenance(path, "autopet")
    assert not any(line.startswith("aux_correlates") for line in path.read_text().splitlines()), (
        "a caller that supplies no aux table must not be told whether a correlate was computed")


def test_load_fullres_header_line_does_not_flip_metric(tmp_path):
    p = _write(tmp_path, "fr.txt", [
        "case_id ivc_z lciv_z z_gap surf_3d raw_z_gap c2c_3d qc",   # header has '_'
        "amos_0001 100 90 5.0 5.0 5.5 6.2 OK",
        "amos_0002 100 88 7.0 7.0 7.5 8.1 OK",
    ])
    rows, counts = ST.load_fullres(p)
    assert counts["metric"] == "surface(v2)"          # NOT flipped to c2c(v1)
    assert rows[0][2] != rows[0][3]                    # surf (5.0) != c2c (6.2)


# --- empty compare file must not crash the cohort ------------------------
def test_stat_row_empty_v_is_placeholder():
    r = ST._stat_row("3D surface (compare)", "cmp.txt", np.array([]),
                     {"measurable": 0, "empty": 2, "unparsed": 0})
    assert r["n_measurable"] == 0
    assert r["median_mm"] == "" and r["gap_present_pct"] == ""


def test_all_stats_empty_compare_does_not_crash():
    # empty fast/full (compare file all-empty) with a populated fullres file
    # must produce 5 rows, the two compare rows being n=0 placeholders, not a crash.
    z = np.array([5.0, 7.0, 9.0])
    surf = np.array([4.5, 6.8, 8.9])
    c2c = np.array([5.0, 7.1, 9.2])
    fr_counts = {"measurable": 3, "empty": 0, "unparsed": 0, "qc_small": 0}
    cmp_counts = {"measurable": 0, "empty": 2, "unparsed": 0}
    rows = ST.all_stats(z, surf, c2c, fr_counts,
                        np.array([]), np.array([]), cmp_counts,
                        "fr.txt", "cmp.txt", censored=0.0)
    assert len(rows) == 5
    cmp_rows = [r for r in rows if "compare" in r["metric"]]
    assert len(cmp_rows) == 2 and all(r["n_measurable"] == 0 for r in cmp_rows)


def test_classify_absent_when_no_gap():
    # primary <= PRESENT_MM means no gap
    assert ST.classify_gap(0.0, 0.2, 0.2) == "absent"
    # centroids far apart, primary gap zero
    assert ST.classify_gap(0.0, 42.0, 0.0) == "absent"


def test_classify_purely_lateral_when_no_axial_extent():
    assert ST.classify_gap(0.0, 8.0, 8.0) == "lateral"


def test_classify_small_lateral_is_axial_at_any_gap_size():
    # large gap that is essentially axial (lateral leg < 1 mm)
    assert ST.classify_gap(50.0, 50.005, 50.005) == "axial"


def test_classify_real_lateral_not_axial_even_at_large_gap():
    assert ST.classify_gap(30.0, 42.0, 42.0) == "mixed"


def test_classify_gap_cannot_be_called_without_a_primary_metric():
    """`primary` is positional and required: presence read off the centre-to-centre separation,
    which for two tubular structures is never within 0.5 mm, would make `absent` unreachable.
    """
    with pytest.raises(TypeError):
        ST.classify_gap(0.0, 0.2)


def test_presence_and_direction_are_taken_from_different_measurands():
    """The two measurands, and why they are not the same one.

    Presence comes from the primary surface gap; direction from the centre-to-centre pair."""
    assert ST.classify_gap(6.0, 40.0, 0.0) == "absent"
    # and the same geometry with a real primary gap is classified, not suppressed
    assert ST.classify_gap(6.0, 40.0, 6.0) == "mixed"
    # presence is on `primary` ALONE: a large `d` cannot rescue a sub-threshold primary
    assert ST.classify_gap(0.0, 1000.0, 0.4) == "absent"


def test_lateral_residual_clamps_and_computes():
    assert ST.lateral_residual(3.0, 5.0) == pytest.approx(4.0)   # 3-4-5
    assert ST.lateral_residual(5.0, 3.0) == pytest.approx(0.0)   # clamped


def test_censored_fraction_at_zero():
    z = np.array([0.0, 0.0, 5.0, 10.0])
    assert ST.censored_fraction(z) == pytest.approx(50.0)


# --- quantization diagnostics ---------------------------------------------
def test_quantization_frac_detects_heaping():
    v = np.array([5.0, 10.0, 15.0, 7.3, 11.8])       # 3/5 on 5mm multiples
    assert ST.quantization_frac(v, step=5.0, tol=0.05) == pytest.approx(0.6)


def test_quantization_frac_empty_is_nan():
    import math
    assert math.isnan(ST.quantization_frac(np.array([])))


def test_correlates_splits_biological_and_technical_fdr_families():
    # the biological correlate (sex) and the CONFOUNDED technical stratifier
    # (scanner) must be in SEPARATE BH families. Here each family has a single test,
    # so its BH q-value must equal its p-value (no inflation from the other family).
    rng = np.random.default_rng(0)
    rows, meta = [], {}
    for i in range(40):
        cid = f"case_{i:03d}"
        sex = "M" if i % 2 == 0 else "F"           # modest biological effect
        scanner = "A" if i < 20 else "B"           # independent of gap
        g = 5.0 + (0.7 if sex == "M" else 0.0) + rng.normal(0, 1.0)
        rows.append((cid, 0.0, g, g, "OK"))
        meta[cid] = dict(sex=sex, age=None, scanner=scanner, contrast=None, region=None)
    out, _panels, _ct = ST.correlates("X", rows, meta)
    sex_rows = [r for r in out if r["variable"] == "sex"]
    scn_rows = [r for r in out if r["variable"] == "scanner"]
    assert sex_rows and scn_rows
    # single-member family -> BH q == p (would be inflated if pooled across families)
    assert sex_rows[0]["qvalue"] == sex_rows[0]["pvalue"]
    assert scn_rows[0]["qvalue"] == scn_rows[0]["pvalue"]


def test_eps_mag_thresholds():
    assert ST._eps_mag(0.0) == "negligible"
    assert ST._eps_mag(0.03) == "small"
    assert ST._eps_mag(0.10) == "medium"
    assert ST._eps_mag(0.20) == "large"
    assert ST._eps_mag(None) == "NA"


# --- cover the two functions that PRODUCE the slice-normalized headline -----
def test_slice_normalized_basic_and_skips_bad_sz():
    surf = np.array([5.0, 10.0, np.nan, 6.0])
    sz = np.array([2.5, 5.0, 2.0, 0.0])          # nan-surf and sz=0 must be skipped
    vals, n = ST.slice_normalized(surf, sz)
    assert n == 2
    assert list(np.round(vals, 6)) == [2.0, 2.0]   # 5/2.5, 10/5


def test_slice_normalized_median_pins_voxel_headline():
    # The function that PRODUCES the "~1.2-1.8 slice-thickness-unit" headline: a cohort whose surface
    # gap is 1.4x its own slice thickness must normalize to a median of 1.4 units.
    # A regression in slice_normalized would move this off 1.4 and be caught.
    rng = np.random.default_rng(0)
    sz = rng.uniform(1.0, 5.0, size=200)
    surf = 1.4 * sz
    vals, n = ST.slice_normalized(surf, sz)
    assert n == 200
    assert float(np.median(vals)) == pytest.approx(1.4, abs=1e-9)


def test_quantization_frac_percase_uses_each_cases_own_step():
    v = np.array([5.0, 10.0, 7.3])
    steps = np.array([2.5, 5.0, 2.0])            # 5/2.5=2, 10/5=2 heaped; 7.3/2=3.65 not
    assert ST.quantization_frac_percase(v, steps, tol=0.05) == pytest.approx(2 / 3)


def test_quantization_frac_percase_skips_nonfinite_step():
    v = np.array([5.0, 9.9])
    steps = np.array([2.5, np.nan])              # only the first case has a usable step
    assert ST.quantization_frac_percase(v, steps, tol=0.05) == pytest.approx(1.0)


# --- guard the headline "sex-invariant" claim with a null + positive control ---
def _sex_cohort(effect_mm, seed=20240611, n=300):
    rng = np.random.default_rng(seed)
    rows, meta = [], {}
    for i in range(n):
        cid = f"case_{i:03d}"
        sex = "M" if i % 2 == 0 else "F"
        g = 5.0 + (effect_mm if sex == "M" else 0.0) + rng.normal(0, 1.0)
        g = max(0.0, g)
        rows.append((cid, 0.0, g, g, "OK"))
        meta[cid] = dict(sex=sex, age=None, scanner=None, contrast=None, region=None)
    return rows, meta


def test_sex_invariance_null_cohort_reports_negligible_effect():
    # With NO sex effect, the sex KW eps^2 must read negligible (<0.01); a regression that
    # made sex spuriously significant would fail here. (single-member bio family -> q == p)
    rows, meta = _sex_cohort(effect_mm=0.0)
    out, _panels, _ct = ST.correlates("Xnull", rows, meta)
    sex_rows = [r for r in out if r["variable"] == "sex"]
    assert sex_rows
    assert float(sex_rows[0]["eps_squared"]) < 0.01
    assert sex_rows[0]["qvalue"] == sex_rows[0]["pvalue"]


def test_sex_effect_is_detected_when_present():
    # Positive control: a strong injected sex effect must NOT read negligible, so the null
    # result above is a real finding, not a test that can never detect anything.
    rows, meta = _sex_cohort(effect_mm=4.0)
    out, _panels, _ct = ST.correlates("Xpos", rows, meta)
    sex_rows = [r for r in out if r["variable"] == "sex"]
    assert sex_rows
    assert float(sex_rows[0]["eps_squared"]) > 0.06   # at least 'medium'; clearly not null


# --- metadata parsers (duplicate-id blanking and contrast parse, on synthetic CSVs) ---
def test_meta_amos_conflicting_duplicate_blanked_order_independent(tmp_path):
    # amos_id 500 appears twice with DIFFERENT demographics (two distinct patients). The unique id
    # is unaffected. Reversing the two conflicting rows must give an IDENTICAL result.
    hdr = "amos_id,Patient's Sex,Patient's Age,Manufacturer"
    a = "500,M,050Y,Philips"
    b = "500,F,070Y,Siemens"
    uniq = "501,F,060Y,Siemens"
    p_ab = _write(tmp_path, "amos_ab.csv", [hdr, a, b, uniq])
    p_ba = _write(tmp_path, "amos_ba.csv", [hdr, b, a, uniq])   # conflicting rows reversed
    m_ab = ST._meta_amos(str(p_ab))
    m_ba = ST._meta_amos(str(p_ba))
    blank = dict(sex=None, age=None, scanner=None, contrast=None, region=None)
    assert m_ab["amos_0500"] == blank                  # conflicting id -> blanked
    assert m_ba["amos_0500"] == blank                  # ...identically under either order
    assert m_ab["amos_0501"] == dict(sex="F", age=60, scanner="Siemens",
                                     contrast=None, region=None)   # unique id intact
    assert m_ab["amos_0501"] == m_ba["amos_0501"]


def test_result_roster_validation_rejects_wrong_modality_substitution(tmp_path, monkeypatch):
    roster = _write(tmp_path, "amos_roster.txt", [
        "amos_0001 imagesTr", "amos_0002 imagesTs"])
    result = _write(tmp_path, "amos_results.txt", [
        "amos_0001 1 1 0 0 0 1 OK",
        "amos_0600 ERROR: empty mask",
    ])
    monkeypatch.setitem(ST.EXACT_ROSTERS, "amos", roster)

    with pytest.raises(RuntimeError, match="missing=.*amos_0002.*unexpected=.*amos_0600"):
        ST.validate_result_roster("amos", result)


def test_amos_adult_sensitivity_selects_before_measurability(monkeypatch):
    monkeypatch.setattr(C, "N_BOOT", 40)
    # The 2.17.0 roster holds no participant under 18, so six synthetic minors are marked here; what
    # is under test is the selection rule, not the cohort's age mix.
    roster = [ln.split()[0] for ln in ST.EXACT_ROSTERS["amos"].read_text().splitlines()
              if ln.split()]
    minors = set(roster[:6])
    meta = {cid: {"age": (16 if cid in minors else 40)} for cid in roster}
    # One adult case (the last) is empty. All six minors remain measurable; selection must
    # produce 487 adult eligible and 486 adult measurable, not replace the empty.
    rows = [(cid, 0.0, 2.0, 2.0, "OK") for cid in roster[:-1]]
    result = ST.amos_adult_sensitivity(
        rows, {"empty": 1, "unparsed": 0}, meta)
    assert result["primary_ct_frame_n"] == len(roster) == 493
    assert result["adult_eligible_n"] == 487
    assert result["adult_measurable_n"] == 486
    assert result["adult_missing_ids"] == roster[-1] == "amos_0500"
    assert result["excluded_minor_n"] == 6


def test_saros_extreme_tail_sensitivity_keeps_primary_and_discloses_exclusion(monkeypatch):
    monkeypatch.setattr(C, "N_BOOT", 40)
    rows = [("case_001", 0.0, 5.0, 5.0, "OK"),
            ("case_002", 0.0, 6.0, 6.0, "OK"),
            ("case_698", 0.0, 202.53, 202.53, "OK")]
    out = ST.saros_extreme_tail_sensitivity(rows)
    assert out[0]["n"] == 3 and out[0]["excluded_n"] == 0
    assert out[1]["n"] == 2 and out[1]["excluded_ids"] == "case_698"
    assert out[1]["max_gap_mm"] == 6.0


def test_collapsed_median_ci_is_not_overwritten_by_topology_ci(monkeypatch):
    """The multicomponent Wilson CI must not overwrite the post-merge median's BCa interval in CSV
    or console output."""
    monkeypatch.setattr(C, "N_BOOT", 200)
    collapsed = []
    for i in range(30):
        orig = float(4 + (i % 5))
        post = float(1 + (i % 4))
        o_present = orig > C.PRESENT_MM
        c_present = post > C.PRESENT_MM if i % 3 else False
        topology = "MULTI_COMPONENT" if i % 2 else "OK"
        collapsed.append((f"case_{i:03d}", orig, post, o_present, c_present,
                          post - orig, bool(i % 4), topology))

    rows, _ = ST.collapsed_class_stats("TEST", "test", collapsed)
    median = float(rows[0]["median_collapsed_mm"])
    lo, hi = (float(x) for x in rows[0]["median_collapsed_ci"].split("-"))
    assert lo <= median <= hi
    assert hi < 10.0  # a gap CI in mm, not a topology percentage interval


def test_meta_autopet_contrast_parse_and_subject_key(tmp_path):
    # 'mit KM' and the portal-venous phase ('p.v.'/'portalven') are contrast-enhanced -> with;
    # 'nativ'/'ohne KM' -> without. Patient-level fields (sex/age/scanner/region) stay
    # subject-shared; a full study-folder id still normalizes to the bare subject for those.
    hdr = ("Subject ID,Modality,Manufacturer,sex,age,diagnosis,Study Date,"
           "Study Description,Series Description")
    rows = [   # subject ids are hex hashes, exactly like the real PETCT_<hash> subject keys
        "PETCT_abc123,CT,Siemens,F,060Y,lymphoma,1/2/2020,Ganzkoerper mit KM,venous",
        "PETCT_def456,CT,Siemens,M,050Y,melanoma,3/4/2019,nativ low-dose,wholebody",
        "PETCT_beef01,CT,Siemens,F,045Y,melanoma,5/6/2021,CT p.v.,portalven phase",
        # Mixed study descriptor, but the selected series is explicitly portal-venous.
        "PETCT_feed01,CT,Siemens,F,045Y,melanoma,6/7/2021,"
        "PET-CT Ganzkoerper nativ u. mit KM,GK p.v.3",
        # SAME subject, TWO studies of DIFFERENT contrast on different dates: each study must
        # keep its own label rather than share one order-dependent value.
        "PETCT_c0ffee,CT,Siemens,M,055Y,melanoma,1/1/2018,Ganzkoerper mit KM,venous",
        "PETCT_c0ffee,CT,Siemens,M,055Y,melanoma,2/2/2018,nativ,wholebody",
    ]
    p = _write(tmp_path, "autopet.csv", [hdr, *rows])
    m = ST._meta_autopet(str(p))
    # full study-folder case-ids (embed subject + MM-DD-YYYY study date) resolve per study
    assert m["PETCT_abc123_01-02-2020-NA-PET-CT-1"]["contrast"] == "with"      # 'mit KM'
    assert m["PETCT_def456_03-04-2019-NA-PET-CT-2"]["contrast"] == "without"   # 'nativ'
    assert m["PETCT_beef01_05-06-2021-NA-PET-CT-3"]["contrast"] == "with"      # portal-venous
    assert m["PETCT_feed01_06-07-2021-NA-PET-CT-4"]["contrast"] == "with"      # series wins
    # the two studies of one subject keep DIFFERENT contrast (per-study, not collapsed)
    assert m["PETCT_c0ffee_01-01-2018-NA-PET-CT-a"]["contrast"] == "with"
    assert m["PETCT_c0ffee_02-02-2018-NA-PET-CT-b"]["contrast"] == "without"
    # patient-level fields are subject-shared; a full study-folder id normalizes to subject
    assert m["PETCT_abc123_01-02-2020-NA-PET-CT-1"]["sex"] == "F"
    assert "PETCT_abc123_01-02-2020-NA-PET-CT-1" in m and "PETCT_abc123" in m


def test_autopet_subject_sensitivity_selects_earliest_study_deterministically():
    rows = [
        ("PETCT_abc123_08-12-2005-NA-later", 0.0, 8.0, 8.0, "OK"),
        ("PETCT_def456_01-01-2004-NA-only", 0.0, 6.0, 6.0, "OK"),
        ("PETCT_abc123_03-23-2003-NA-earliest", 0.0, 3.0, 3.0, "OK"),
    ]
    selected = ST.autopet_one_study_per_subject(rows)
    assert [r[0] for r in selected] == [
        "PETCT_abc123_03-23-2003-NA-earliest",
        "PETCT_def456_01-01-2004-NA-only",
    ]


def test_autopet_subject_sensitivity_does_not_replace_empty_earliest_study():
    later = ("PETCT_abc123_08-12-2005-NA-later", 0.0, 8.0, 8.0, "OK")
    earliest_empty = "PETCT_abc123_03-23-2003-NA-earliest"
    # Selection sees the empty earliest id before measurability filtering. The later
    # row must not be substituted, which would create availability-dependent selection.
    assert ST.autopet_one_study_per_subject(
        [later], all_case_ids=[later[0], earliest_empty]) == []


def test_autopet_correlate_rows_disclose_analysis_unit_and_scope(monkeypatch):
    monkeypatch.setattr(C, "N_BOOT", 40)
    rows = [(f"PETCT_{i:06x}_01-01-2000-NA-x", 0.0, float(i + 1),
             float(i + 1), "OK") for i in range(20)]
    meta = ST._SubjectKeyedMeta()
    for i in range(20):
        meta[f"PETCT_{i:06x}"] = dict(sex=None, age=None, scanner=None,
                                       contrast=None, region=None)
    study_out, _p, _t = ST.correlates("autoPET", rows, meta)
    assert study_out[0]["analysis_unit"] == "study"
    assert study_out[0]["inference_scope"] == (
        "descriptive_repeated_studies_not_cluster_adjusted")

    subject_out, _p, _t = ST.correlates(
        "autoPET", rows, meta, analysis_unit="subject",
        inference_scope="subject_level_sensitivity_earliest_study")
    assert subject_out[0]["analysis_unit"] == "subject"
    assert subject_out[0]["inference_scope"] == (
        "subject_level_sensitivity_earliest_study")


def test_load_fullres_scans_status_field_not_case_id(tmp_path):
    p = tmp_path / "fr.txt"
    p.write_text(
        "case ivc_z lciv_z z_gap surf raw_z c2c qc\n"
        "PETCT_emptylung_x 210 185 15 12.5 15 15.0 OK\n"   # 'empty' in the ID -> still measurable
        "case_002 200 180 10 8.0 10 10.5 OK\n"
        "case_003 ERROR: empty mask\n")                     # a genuine empty (status field)
    rows, counts = ST.load_fullres(str(p))
    ids = [r[0] for r in rows]
    assert "PETCT_emptylung_x" in ids and "case_002" in ids   # 2 measurable
    assert counts["measurable"] == 2 and counts["empty"] == 1  # only the real ERROR row drops


def test_a_missing_vein_variant_data_row_is_measurable_not_empty(tmp_path):
    """Data rows are identified structurally (numeric gaps at t[1]/t[2]) and free-form columns are
    never substring-scanned, while genuine failure rows keep their exact empty/failed
    classification."""
    # delta = orig - collapsed (+ = the merge shrank it).
    p = _write(tmp_path, "collapsed.txt", [
        "# mode=labels",
        "amos_0001 4.00 0.00 +4.00 1 0 2 0 MISSING_VEIN OK 1 1 1",   # MISSING_VEIN data row
        "amos_0002 3.50 1.25 +2.25 1 1 2 0 MULTI_COMPONENT OK 1 1 0",  # control
        "amos_0003 ERROR: EMPTY_RESPLIT_VESSEL",                     # algorithmic
        "amos_0004 ERROR: missing/empty IVC or LCIV",                # missingness
    ])
    rows, counts = ST.load_collapsed(p)
    assert counts["measurable"] == 2
    assert counts["empty"] == 1                       # only the missing-input row
    assert counts["failed"] == 1                      # only the re-split failure
    assert counts["unparsed"] == 0
    variants = {r[0]: r[7] for r in rows}
    assert variants["amos_0001"] == "MISSING_VEIN"    # payload preserved, not booked
    assert variants["amos_0002"] == "MULTI_COMPONENT"
    assert "EMPTY_RESPLIT_VESSEL" in counts["failure_reasons"]


def test_fullres_loader_splits_missingness_from_producer_errors(tmp_path):
    """load_fullres splits missingness (`empty`) from read/producer failure (`failed`), each with
    ids and reasons, and the accounting closes over every processed line."""
    p = _write(tmp_path, "fullres.txt", [
        V2_HEADER_COLS,
        "amos_0001 10.0 5.0 5.00 4.20 5.00 5.60 OK",
        "amos_0002 ERROR: NO_MASK",                    # total failure: missingness
        "amos_0003 ERROR: empty mask",                 # empty mask: missingness
        "amos_0004 ERROR: RuntimeError",               # producer exception: FAILED
        "amos_0005 not a parseable row at all",        # neither: unparsed
    ])
    rows, counts = ST.load_fullres(p)
    assert counts["measurable"] == 1 and len(rows) == 1
    assert counts["empty"] == 2
    assert counts["failed"] == 1
    assert counts["unparsed"] == 1
    assert counts["empty_ids"] == ["amos_0002", "amos_0003"]
    assert counts["failed_ids"] == ["amos_0004"]
    assert counts["failure_reasons"] == {"NO_MASK": 1, "EMPTY_MASK": 1,
                                         "RUNTIMEERROR": 1}
    total = (counts["measurable"] + counts["empty"] + counts["failed"]
             + counts["unparsed"])
    assert total == 5                                  # closed over processed lines


def test_bh_families_are_monotone_against_an_independent_oracle():
    """This fixture (rng=101, measured, frozen) yields a THREE-member technical family whose sorted
    p-values invert the raw m*p/rank sequence (3*p_(2) > 2*p_(3), asserted as a precondition so the
    fixture cannot rot into a case where the cumulative minimum is a no-op). The pipeline q-values
    must equal a test-local independent BH implementation and be monotone in p."""
    rng = np.random.default_rng(101)
    rows, meta = [], {}
    for i in range(60):
        cid = f"case_{i:03d}"
        scanner = "A" if i % 2 == 0 else "B"
        contrast = "contrast" if (i // 2) % 2 == 0 else "non_contrast"
        region = "abdomen" if (i // 4) % 2 == 0 else "thorax"
        g = (5.0 + (1.0 if scanner == "A" else 0.0)
             + (1.0 if contrast == "contrast" else 0.0) + rng.normal(0, 1.0))
        rows.append((cid, 0.0, g, g, "OK"))
        meta[cid] = dict(sex=("M" if (i // 8) % 2 == 0 else "F"), age=None,
                         scanner=scanner, contrast=contrast, region=region)
    out, _panels, _ct = ST.correlates("X", rows, meta)
    tech, bio = {}, {}
    for r in out:
        if r["pvalue"] == "" or r["variable"] in tech or r["variable"] in bio:
            continue
        (tech if r["variable"] in C.CONFOUNDED_VARIABLES else bio)[r["variable"]] = (
            float(r["pvalue"]), float(r["qvalue"]))
    assert set(tech) == {"scanner", "contrast", "region"}     # three-member family
    ps = sorted(p for p, _ in tech.values())
    m = len(ps)
    # PRECONDITION: the raw m*p/rank sequence inverts between ranks 2 and 3, so
    # the cumulative minimum is load-bearing (without it q would be non-monotone)
    assert ps[1] * m / 2 > ps[2] * m / 3, "fixture rotted: no BH inversion"
    # independent oracle: sort ascending, q_(i) = min over j>=i of m*p_(j)/j
    order = sorted(tech, key=lambda k: tech[k][0])
    oracle, running = {}, float("inf")
    for rank in range(m, 0, -1):
        var = order[rank - 1]
        running = min(running, tech[var][0] * m / rank)
        oracle[var] = min(running, 1.0)
    for var in tech:
        assert tech[var][1] == pytest.approx(oracle[var], abs=5e-4), var
    # monotone: q ordering follows p ordering across the family
    qs_in_p_order = [tech[v][1] for v in order]
    assert all(a <= b + 1e-12 for a, b in zip(qs_in_p_order, qs_in_p_order[1:]))
    # the bio family (sex alone) is untouched by the technical family: m=1 -> q == p
    assert bio["sex"][1] == bio["sex"][0]


def test_reported_bca_ci_matches_scipy_and_is_seed_locked():
    """(a) ENGINE, independently derived: on this n=12 continuous fixture the bootstrap-median
    distribution is discrete enough that the 95% BCa endpoints snap to the same order statistics for
    any healthy implementation: statlib's interval must EQUAL scipy.stats.bootstrap(method='BCa')
    computed live in this test (an independent implementation, not a copied constant). (b) SEED
    reproducibility lock: on an n=200 continuous fixture the endpoints
    are seed-sensitive (asserted as a precondition), and the interval produced through the
    pipeline's own median_ci_auto(C.CI_METHOD, C.ALPHA, C.N_BOOT, C.SEED) is pinned to the exact
    endpoints recorded at SEED=20240611. The pin freezes the published-CI reproduction chain; its
    correctness is carried by half (a)."""
    from scipy.stats import bootstrap
    v = np.array([2.3, 3.1, 4.7, 5.2, 5.9, 6.4, 7.8, 8.1, 9.6, 11.2, 12.5, 14.1])
    lo, hi = ST.S.median_ci_auto(v, C.CI_METHOD, C.ALPHA, C.N_BOOT, C.SEED)
    res = bootstrap((v,), np.median, confidence_level=1 - C.ALPHA,
                    n_resamples=9999, method="BCa",
                    random_state=np.random.default_rng(0))
    assert lo == pytest.approx(res.confidence_interval.low, abs=1e-9)
    assert hi == pytest.approx(res.confidence_interval.high, abs=1e-9)
    # (b) the seed lock
    rng = np.random.default_rng(42)
    w = np.round(rng.lognormal(1.6, 0.55, 200), 4)
    ci_seed = ST.S.median_ci_auto(w, C.CI_METHOD, C.ALPHA, C.N_BOOT, C.SEED)
    ci_next = ST.S.median_ci_auto(w, C.CI_METHOD, C.ALPHA, C.N_BOOT, C.SEED + 1)
    assert ci_seed != ci_next, "fixture is not seed-sensitive"
    assert ci_seed[0] == pytest.approx(4.4238, abs=1e-9)
    assert ci_seed[1] == pytest.approx(5.382022565990001, abs=1e-9)


# Every assertion below is derived from the module's own declared contract, not copied from the
# implementation: `stats.py` documents `c2c >= z` as "center-based, must hold" and `surf <= c2c` as
# the de-biasing property, gaps are distances so they cannot be negative, and the compare file's
# column 3 is by construction column 2 minus column 1.
# ===========================================================================

# The per-case tables print every distance to two decimals, so two renderings of
# ONE measurement can differ by at most one printed ulp; on the study's tables the
# paired full/surf values are equal to the last digit.
_PRINTED_ULP_MM = 0.01


@pytest.mark.parametrize("row,why", [
    ("bad_001 220 140 25.00 -7.50 +25.00 25.62 OK",  "negative surface gap"),
    ("bad_002 220 140 -5.00 3.00 -5.00 5.00 OK",     "negative axial gap"),
    ("bad_003 220 140 5.00 99.00 +5.00 5.00 OK",     "surf > c2c breaks de-biasing"),
    ("bad_004 220 140 99.00 3.00 +99.00 5.00 OK",    "z > c2c breaks the hard invariant"),
    ("bad_005 220 140 25.00 1000000 +25.00 25.62 OK", "1,000,000 mm surface gap"),
    # A negative c2c is NOT redundant with the invariant clauses below it: with z and surf
    # both near zero, `z - c2c` and `surf - c2c` stay inside INVARIANT_SLACK_MM and the
    # row slips through.
    ("bad_006 220 140 0.00 0.00 +0.00 -0.10 OK",      "negative centre-to-centre distance"),
])
def test_fullres_refuses_physically_impossible_rows(tmp_path, row, why):
    """A row that cannot describe a real measurement must not reach a statistic.

    It is COUNTED (never silently dropped -- that would move the denominator), so
    the accounting stays closed, but it must not appear among the measurable rows.
    """
    p = _write(tmp_path, "fr.txt", [
        "good_000 210 185 15.00 12.53 +15.00 15.04 OK",
        row,
    ])
    rows, counts = ST.load_fullres(p)
    assert [r[0] for r in rows] == ["good_000"], f"invalid row accepted: {why}"
    assert counts["measurable"] == 1
    assert counts["unparsed"] == 1, why
    assert counts["measurable"] + counts["empty"] + counts["failed"] \
        + counts["unparsed"] == 2                       # accounting still closed
    assert "bad_" in " ".join(counts["unparsed_ids"])


def test_fullres_keeps_rows_that_only_graze_the_invariant_slack(tmp_path):
    """The refusal must use the module's own INVARIANT_SLACK_MM, not zero.

    Rounding at two decimals can put `surf` a hair above `c2c` on a genuinely
    touching case; that is the reason the slack constant exists, and refusing
    those rows would drop real measurements.
    """
    eps = C.INVARIANT_SLACK_MM / 2.0
    p = _write(tmp_path, "fr.txt", [
        f"case_000 210 185 15.00 {15.04 + eps:.5f} +15.00 15.04 OK",
        f"case_001 210 185 {15.04 + eps:.5f} 12.53 +15.00 15.04 OK",
    ])
    _rows, counts = ST.load_fullres(p)
    assert counts["measurable"] == 2 and counts["unparsed"] == 0


@pytest.mark.parametrize("row,which", [
    ("case_001 -3.00 15.00 +18.00 49923 49405 OK", "fast (column 1)"),
    ("case_001 15.00 -3.00 -18.00 49923 49405 OK", "full (column 2)"),
])
def test_compare_refuses_negative_distances(tmp_path, row, which):
    """BOTH value columns, not just the first: a one-sided test of a two-sided guard protects one
    side.
    """
    p = _write(tmp_path, "cmp.txt", [
        "case_000 17.50 15.00 -2.50 49923 49405 OK",
        row,
    ])
    rows, counts = ST.load_compare(p)
    assert [r[0] for r in rows] == ["case_000"], f"a negative {which} was accepted"
    assert counts["unparsed"] == 1


def test_compare_row_is_checked_against_its_own_delta_column(tmp_path):
    """Column 3 IS column 2 minus column 1. A row that contradicts itself is not a
    row whose first two columns can be trusted."""
    p = _write(tmp_path, "cmp.txt", [
        "case_000 17.50 15.00 -2.50 49923 49405 OK",     # consistent
        "case_001 10.00 20.00 -999.00 49923 49405 OK",   # delta contradicts the pair
    ])
    rows, counts = ST.load_compare(p)
    assert [r[0] for r in rows] == ["case_000"]
    assert counts["unparsed"] == 1


def test_compare_delta_tolerance_admits_independent_rounding(tmp_path):
    """Both operands and the delta are printed to 2 dp from unrounded values, so a 0.01 disagreement is
    expected and must not be refused."""
    p = _write(tmp_path, "cmp.txt", ["case_000 20.96 20.59 -0.38 34946 34769 OK"])
    _rows, counts = ST.load_compare(p)
    assert counts["measurable"] == 1 and counts["unparsed"] == 0


def test_redundant_same_case_values_are_equality_checked(tmp_path):
    """The compare file's `full` column and the fullres file's `surf` column are the SAME measurement
    of the same case."""
    fr = [("case_000", 15.0, 12.53, 15.04, "OK"), ("case_001", 9.0, 7.00, 9.10, "OK")]
    ok = [("case_000", 17.5, 12.53), ("case_001", 11.0, 7.00)]
    ST.assert_compare_matches_fullres(fr, ok, "fr.txt", "cmp.txt")   # must not raise
    drifted = [("case_000", 17.5, 12.53), ("case_001", 11.0, 8.05)]  # 1.05 mm drift
    with pytest.raises(RuntimeError, match="case_001"):
        ST.assert_compare_matches_fullres(fr, drifted, "fr.txt", "cmp.txt")


def test_redundant_value_tolerance_is_one_printed_ulp():
    fr = [("case_000", 15.0, 12.53, 15.04, "OK")]
    ST.assert_compare_matches_fullres(
        fr, [("case_000", 17.5, 12.53 + _PRINTED_ULP_MM * 0.9)], "fr.txt", "cmp.txt")
    with pytest.raises(RuntimeError):
        ST.assert_compare_matches_fullres(
            fr, [("case_000", 17.5, 12.53 + _PRINTED_ULP_MM * 10)], "fr.txt", "cmp.txt")


def test_invariants_are_evaluated_before_the_csv_is_written():
    """`summary()` prints `invariant c2c>=z: PASS/FAIL` after `all_stats.csv` and `provenance.txt`
    are written, so the guard must run before them, or a corrupt table is published first and
    questioned second.

    Two assertions. The first is behavioural: the guard refuses a violating cohort and
    admits a clean one. The second is STRUCTURAL, by `ast`, because ordering is the
    property under test and a grep for the call name would also match a comment about it.
    """
    # hard invariant c2c >= z: violated far beyond the slack
    with pytest.raises(RuntimeError, match="c2c"):
        ST._publication_invariants_hold(np.array([25.0]), np.array([20.59]),
                                        np.array([0.0]), "amos_results_fullres.txt")
    # de-biasing invariant surf <= c2c
    with pytest.raises(RuntimeError, match="surf"):
        ST._publication_invariants_hold(np.array([1.0]), np.array([99.0]),
                                        np.array([25.62]), "amos_results_fullres.txt")
    # and the passing direction, so the guard is not vacuously strict
    ST._publication_invariants_hold(np.array([25.0, 10.0]), np.array([20.59, 9.57]),
                                    np.array([25.62, 13.56]), "amos_results_fullres.txt")

    import ast
    tree = ast.parse(Path(ST.__file__).read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run_dataset")
    guard = [n.lineno for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "_publication_invariants_hold"]
    publish = [n.lineno for n in ast.walk(fn)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr in ("to_csv", "write_provenance")]
    publish += [n.lineno for n in ast.walk(fn)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "write_provenance"]
    assert guard, "run_dataset does not call the publication invariant guard at all"
    assert publish, "run_dataset publishes nothing -- the ordering test is looking at the wrong function"
    assert min(guard) < min(publish), (
        f"the invariant guard runs at line {min(guard)} but publication starts at "
        f"line {min(publish)}: a corrupt table is written before it is questioned")


def test_mixed_schema_table_does_not_misparse_the_later_rows(tmp_path):
    """Each row is read by its own shape: in a v1-first file, a later v2 row must keep its own c2c
    from column 6 rather than take `c2c := surf`, or the geometry pair (z, c2c) would carry the
    SURFACE value and the d>=z invariant, the axial/lateral split and `median_axial_fraction` would
    be computed against the wrong second coordinate.
    """
    p = _write(tmp_path, "fr.txt", [
        "case_000 210 185 15.0 15.0 +15.0 OK",                  # v1: 7 tokens
        "case_001 210 185 15.00 12.53 +15.00 19.04 OK",         # v2: 8 tokens
    ])
    rows, counts = ST.load_fullres(p)
    assert counts["measurable"] == 2
    by = {r[0]: r for r in rows}
    # v1 has no separate surface column, so surf and c2c are the same number
    assert by["case_000"][2] == by["case_000"][3] == 15.0
    # v2 must keep its OWN c2c from column 6, not inherit surf
    assert by["case_001"][2] == 12.53 and by["case_001"][3] == 19.04


def test_mixed_schema_table_is_labelled_mixed_not_silently_v2(tmp_path):
    """A file holding both schemas has no single honest cohort metric: v1's column 4 is a
    centre-to-centre measurement and v2's is a surface measurement, so the pooled `surf` array would
    mix two different measurands."""
    p = _write(tmp_path, "fr.txt", [
        "case_000 210 185 15.0 15.0 +15.0 OK",
        "case_001 210 185 15.00 12.53 +15.00 19.04 OK",
    ])
    _rows, counts = ST.load_fullres(p)
    assert counts["metric"] == "mixed(v1+v2)"
    assert counts["metric"] != "surface(v2)"        # so cmp_stale is True downstream


def test_uniform_tables_are_labelled_by_their_schema(tmp_path):
    """Per-row detection must agree with first-row detection on a uniform file."""
    v2 = _write(tmp_path, "v2.txt", [
        "case_000 210 185 15.00 12.53 +15.00 15.04 OK",
        "case_001 215 175 5.00 4.00 +5.00 5.20 OK",
    ])
    v1 = _write(tmp_path, "v1.txt", [
        "case_000 210 185 15.0 15.0 +15.0 OK",
        "case_002 215 175 5.0 5.0 +5.0 OK",
    ])
    assert ST.load_fullres(v2)[1]["metric"] == "surface(v2)"
    assert ST.load_fullres(v1)[1]["metric"] == "c2c(v1)"


def test_absent_compare_file_does_not_discard_the_primary_analysis(tmp_path, monkeypatch):
    """A missing SECONDARY fast-vs-full file disables only the agreement statistics (reported as
    `fast-vs-full agreement: NOT RUN`); the PRIMARY median, its BCa interval and the present-rate are
    still computed."""
    rdir = tmp_path / "Results" / "phase1_characterize" / "AMOS"
    rdir.mkdir(parents=True)
    (rdir / "amos_results_fullres.txt").write_text(
        "\n".join(f"amos_{i:04d} 220 140 {5 + i % 4}.00 {4 + i % 4}.20 +5.00 {9 + i % 4}.62 OK"
                  for i in range(1, 41)) + "\n")
    # deliberately NO amos_results_compare.txt
    monkeypatch.setattr(ST, "RESULTS", tmp_path / "Results" / "phase1_characterize")
    monkeypatch.setattr(ST, "validate_result_roster", lambda *a, **k: None)
    # run_dataset loads metadata regardless of --no-correlates, and amos_adult_sensitivity refuses
    # this fixture (it needs the exact roster table), so metadata loading is stubbed out.
    monkeypatch.setattr(ST, "load_meta", lambda ds: {})
    out = tmp_path / "out"
    res = ST.run_dataset("amos", do_correlates=False, do_figures=False, outdir=out)
    assert res is not None, "the cohort was discarded for want of a secondary file"
    csv_path = out / "AMOS" / "all_stats.csv"
    assert csv_path.exists(), "the primary statistics were never written"
    text = csv_path.read_text()
    assert "3D surface-to-surface (PRIMARY)" in text
    with open(csv_path, newline="") as fh:
        primary = next(r for r in csv.DictReader(fh)
                       if r["metric"].startswith("3D surface-to-surface (PRIMARY)"))
    assert int(primary["n_measurable"]) == 40
    assert primary["median_mm"] not in ("", None)


def test_the_no_compare_path_survives_the_figure_renderers(tmp_path, monkeypatch):
    """The no-compare branch hands downstream code a `cmp_counts` dict built by hand, and the FIGURE
    path is the consumer the primary test does not exercise.

    Every consumer reads `cmp_counts` through `.get()` with a default, and the two
    bracket-indexed reads in the module (`counts["failure_reasons"]` at summary(),
    `fr_counts["empty_ids"]`) are guarded by their own `.get` or are explicitly the FULLRES
    counts; this test pins that.
    """
    import matplotlib
    matplotlib.use("Agg")
    rdir = tmp_path / "Results" / "phase1_characterize" / "AMOS"
    rdir.mkdir(parents=True)
    (rdir / "amos_results_fullres.txt").write_text("\n".join(
        f"amos_{i:04d} 220 140 {5 + i % 4}.00 {4 + i % 4}.20 +5.00 {9 + i % 4}.62 OK"
        for i in range(1, 61)) + "\n")
    monkeypatch.setattr(ST, "RESULTS", tmp_path / "Results" / "phase1_characterize")
    monkeypatch.setattr(ST, "validate_result_roster", lambda *a, **k: None)
    # run_dataset loads metadata regardless of --no-correlates, and amos_adult_sensitivity refuses
    # this fixture (it needs the exact roster table), so metadata loading is stubbed out.
    monkeypatch.setattr(ST, "load_meta", lambda ds: {})
    out = tmp_path / "out"
    res = ST.run_dataset("amos", do_correlates=False, do_figures=True, outdir=out)
    assert res is not None
    figures = sorted(p.name for p in (out / "AMOS" / "figures").glob("*.png"))
    assert figures, "the figure renderers produced nothing on the no-compare path"
    # the fast-vs-full Bland-Altman is the ONE figure that genuinely needs the compare
    # file; every other figure must still be drawn rather than the whole cohort skipped
    assert "bland_altman_fast_vs_full.png" not in figures
    assert "distribution.png" in figures and "ecdf.png" in figures


def test_absent_fullres_file_still_skips_the_cohort(tmp_path, monkeypatch):
    rdir = tmp_path / "Results" / "phase1_characterize" / "AMOS"
    rdir.mkdir(parents=True)
    (rdir / "amos_results_compare.txt").write_text(
        "amos_0001 20.96 20.59 -0.37 34946 34769 OK\n")
    monkeypatch.setattr(ST, "RESULTS", tmp_path / "Results" / "phase1_characterize")
    monkeypatch.setattr(ST, "validate_result_roster", lambda *a, **k: None)
    assert ST.run_dataset("amos", do_correlates=False, do_figures=False,
                          outdir=tmp_path / "out") is None


def test_acquisition_note_figures_track_the_data_not_a_literal():
    """Feed a cohort set whose spacings and gaps are nothing like the committed ones
    and confirm every figure in the sentence moves with them. A literal cannot pass this."""
    rng = np.random.default_rng(0)
    payload = [
        {"dir": "SYNTH_A", "surf": np.full(40, 8.0), "sz": np.full(40, 4.0)},   # 2.0 units
        {"dir": "SYNTH_B", "surf": np.full(40, 30.0), "sz": np.full(40, 4.0)},  # 7.5 units
    ]
    note = ST._acquisition_scaling_note(payload, mm_eps=0.20, slice_eps=0.02)
    assert "~2.0-7.5 slice-thickness units" in note
    assert "matched 4.0mm" in note                      # the modal spacing, derived
    assert "SYNTH_A 8.0" in note and "SYNTH_B 30.0" in note
    assert "2.67" not in note and "3.75" not in note    # no committed-cohort literal appears
    assert "1.2-1.8" not in note
    _ = rng


def test_acquisition_note_omits_a_claim_it_cannot_support():
    """There is no thin-slice stratum when every cohort sits at the matched reference,
    so the 'thin-slice gaps remain N-M units' clause must disappear rather than be
    invented."""
    payload = [
        {"dir": "SYNTH_A", "surf": np.full(40, 5.0), "sz": np.full(40, 2.5)},
        {"dir": "SYNTH_B", "surf": np.full(40, 7.5), "sz": np.full(40, 2.5)},
    ]
    note = ST._acquisition_scaling_note(payload, mm_eps=0.20, slice_eps=0.02)
    assert "thin-slice" not in note
    assert "matched 2.5mm" in note


# Three paths reach lciv_n_components = 0: an absent LCIV mask file (case_aux_metrics returns its
# default), an empty mask (lciv_quality returns (nan, 0)), and an exception caught by
# measure_all.py. A count of 0 must leave seg_quality unset rather than read as clean.
# ===========================================================================

@pytest.mark.parametrize("n_comp,lcc", [(0, float("nan")), (0, 1.0), (-1, float("nan"))])
def test_unmeasured_component_count_is_not_reported_as_clean(n_comp, lcc):
    """A component count of zero cannot describe a mask that was measured: a real mask has at least one
    component."""
    meta = {"case_000": dict(sex=None, age=None, scanner=None, contrast=None, region=None)}
    ST._inject_aux_correlates(meta, {"case_000": {"n_comp": n_comp, "lcc": lcc,
                                                  "csa": float("nan")}})
    assert "seg_quality" not in meta["case_000"], (
        "an unmeasured LCIV mask was labelled as a segmentation-quality observation")


def test_measured_component_counts_classify():
    """The guard must not swallow real observations: one component is clean, more than
    one is fragmented, and a single component that is not the whole mask is fragmented."""
    meta = {c: dict(sex=None) for c in ("clean", "frag_n", "frag_lcc")}
    ST._inject_aux_correlates(meta, {
        "clean":    {"n_comp": 1, "lcc": 1.00, "csa": float("nan")},
        "frag_n":   {"n_comp": 3, "lcc": 0.80, "csa": float("nan")},
        "frag_lcc": {"n_comp": 1, "lcc": 0.90, "csa": float("nan")},
    })
    assert meta["clean"]["seg_quality"] == "clean"
    assert meta["frag_n"]["seg_quality"] == "fragmented"
    assert meta["frag_lcc"]["seg_quality"] == "fragmented"


def test_aux_writer_emits_na_rather_than_a_count_it_did_not_measure(tmp_path):
    """The producer writes NA for an unmeasured component count, matching its two neighbouring
    columns, so the aux table never claims zero components and an operator reading it sees 'not
    measured' rather than a number."""
    import measure_all as MA
    assert MA._aux_component_token(1) == "1"
    assert MA._aux_component_token(4) == "4"
    assert MA._aux_component_token(0) == "NA"        # absent mask / empty mask / exception
    assert MA._aux_component_token(-1) == "NA"
    assert MA._aux_component_token(float("nan")) == "NA"
    assert MA._aux_component_token(None) == "NA"


def test_the_aux_ROW_uses_the_token_and_not_the_raw_count():
    """Binds the aux ROW to the helper structurally, so neither a bare subscript nor a wrapped one
    (e.g. `_aux_component_token(max(1, ...))`, which would write a false '1') can pass."""
    import ast
    import pathlib
    import measure_all as MA

    tree = ast.parse(pathlib.Path(MA.__file__).read_text(encoding="utf-8"))

    def _is_the_count(node):
        return (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                and node.value.id == "aux"
                and isinstance(node.slice, ast.Constant)
                and node.slice.value == "lciv_n_components")

    guarded = bare = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.FormattedValue):
            continue
        expr = node.value
        if _is_the_count(expr):
            bare += 1
        elif (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name)
              and expr.func.id == "_aux_component_token"
              and len(expr.args) == 1 and not expr.keywords
              and _is_the_count(expr.args[0])):
            guarded += 1

    assert bare == 0, (
        f"{bare} interpolation(s) of aux['lciv_n_components'] bypass _aux_component_token, "
        f"so the aux TABLE asserts a component count the producer never measured.")
    assert guarded == 1, (
        f"expected exactly one guarded interpolation of the component count, found "
        f"{guarded}; the argument must be the subscript itself, not an expression built "
        f"from it.")


def test_the_collapsed_parser_refuses_impossible_rows(tmp_path):
    """`load_collapsed` refuses every impossible row below.

    Each refusal is COUNTED, never dropped: the denominator is what these diagnostics are
    about, so a row that vanishes silently is worse than one that is refused loudly.
    """
    p = _write(tmp_path, "collapsed.txt", [
        "# mode=labels",
        "good_0001 6.25 3.75 +2.50 1 1 1 1 OK OK 1 1 1",        # a clean control
        "neg_0001 -6.25 3.75 -10.00 1 1 1 1 OK OK 1 1 1",       # negative ORIGINAL gap
        "neg_0002 6.25 -3.75 +10.00 1 1 1 1 OK OK 1 1 1",       # negative COLLAPSED gap
        "dlt_0001 6.25 3.75 -999.00 1 1 1 1 OK OK 1 1 1",       # delta contradicts columns
        "flg_0001 6.25 3.75 +2.50 7 1 1 1 OK OK 1 1 1",         # present flag = 7
        "flg_0002 6.25 3.75 +2.50 1 1 1 5 OK OK 1 1 1",         # connected flag = 5
        "flg_0003 6.25 3.75 +2.50 1 1 1 1 OK OK 1 1 3",         # resolved flag = 3
    ])
    rows, counts = ST.load_collapsed(p)
    assert counts["measurable"] == 1, "only the clean control may survive"
    assert [r[0] for r in rows] == ["good_0001"]
    assert counts["unparsed"] == 6, "every refusal must be counted, not dropped"
    assert counts["empty"] == 0 and counts["failed"] == 0
    reasons = counts["failure_reasons"]
    assert reasons["NEGATIVE_GAP"] == 2
    assert reasons["DELTA_CONTRADICTS_VALUES"] == 1
    assert reasons["NONBINARY_FLAG"] == 3


def test_the_collapsed_delta_convention_is_orig_minus_collapsed(tmp_path):
    """The SIGN is the load-bearing detail, and getting it backwards would refuse essentially
    every real row while looking like a working guard.

    `parse_gap_table` checks `(b - a)` for the fullres/compare tables; the collapsed table
    uses the OPPOSITE order, delta = orig - collapsed (+ = the merge shrank it), and the study's
    data confirm it: across all 1,855 paired rows, max|delta - (orig - coll)| = 0.01 (one printed
    ULP) against max|delta - (coll - orig)| = 122.14.
    """
    p = _write(tmp_path, "collapsed.txt", [
        "# mode=labels",
        "right_001 9.00 4.00 +5.00 1 1 1 1 OK OK 1 1 1",    # orig - coll = +5.00
        "wrong_001 9.00 4.00 -5.00 1 1 1 1 OK OK 1 1 1",    # the reversed convention
    ])
    rows, counts = ST.load_collapsed(p)
    assert [r[0] for r in rows] == ["right_001"]
    assert counts["failure_reasons"]["DELTA_CONTRADICTS_VALUES"] == 1


def test_the_collapsed_delta_guard_tolerates_exactly_the_printed_rounding(tmp_path):
    """Both columns are printed to 2 dp, so the delta can legitimately disagree with their
    difference by the accumulated rounding of three printed numbers -- which is what
    `DELTA_ROUNDING_BOUND_MM` is. A tolerance of zero would refuse real rows; a loose one
    would admit the -999.00 case above. This pins both edges.
    """
    assert ST.DELTA_ROUNDING_BOUND_MM == pytest.approx(3 * (ST.PRINTED_ULP_MM / 2))
    inside = ST.DELTA_ROUNDING_BOUND_MM * 0.9
    outside = ST.DELTA_ROUNDING_BOUND_MM * 10
    p = _write(tmp_path, "collapsed.txt", [
        "# mode=labels",
        f"in_0001 9.00 4.00 {5.00 + inside:+.5f} 1 1 1 1 OK OK 1 1 1",
        f"out_001 9.00 4.00 {5.00 + outside:+.5f} 1 1 1 1 OK OK 1 1 1",
    ])
    rows, counts = ST.load_collapsed(p)
    assert [r[0] for r in rows] == ["in_0001"]
    assert counts["failure_reasons"]["DELTA_CONTRADICTS_VALUES"] == 1


def _ci_method_bypasses(src):
    """`{enclosing function: [line, ...]}` for every median-CI call that does not pass the
    method through. Read with `ast`, so a comment or string that names the call is not counted."""
    routed = {"method", "ci_method"}
    targets = {"median_ci_bca", "paired_median_diff_ci"}
    out = {}
    for fn in ast.walk(ast.parse(src)):
        if not isinstance(fn, ast.FunctionDef):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in targets
                    and not ({k.arg for k in node.keywords} & routed)):
                out.setdefault(fn.name, []).append(node.lineno)
    return out


def test_no_median_ci_call_site_bypasses_CI_METHOD():
    """No median-CI call site in stats.py may bypass gap_config.CI_METHOD.

    None is allowed anywhere, so a bypass cannot hide inside one named function.
    """
    src = (ROOT / "Code" / "phase1_characterize" / "stats.py").read_text(encoding="utf-8")
    bypass = _ci_method_bypasses(src)
    assert not bypass, (
        f"`gap_config.CI_METHOD` is documented as the confidence-interval method for medians "
        f"and is stamped into every `provenance.txt`, but these call sites do not pass it: "
        f"{bypass}. A sensitivity run set to 'percentile' would record percentile for the whole "
        f"cohort while these stayed on BCa. Route them through "
        f"`median_ci_auto(..., C.CI_METHOD)`; do not relax this guard.")


def test_the_bypass_detector_sees_a_planted_bypass():
    """The positive control, and the reason the test above is not vacuous.

    An empty result is what a CORRECT module and a BROKEN detector look like alike. This feeds
    the detector a module that definitely contains one and asserts it is found, so the zero
    above means "none present" rather than "none findable".
    """
    planted = (
        "import statlib as S\n"
        "def innocent():\n"
        "    return S.median_ci_auto(v, C.CI_METHOD, C.ALPHA)\n"
        "def guilty():\n"
        "    lo, hi = S.median_ci_bca(v, alpha=C.ALPHA, n_boot=C.N_BOOT, seed=C.SEED)\n"
        "    return S.paired_median_diff_ci(a, b, alpha=C.ALPHA)\n")
    found = _ci_method_bypasses(planted)
    assert set(found) == {"guilty"}, f"the detector missed a planted bypass: {found}"
    assert len(found["guilty"]) == 2, found


def test_the_missingness_note_does_not_claim_mnar_or_assert_a_mechanism():
    """The branch fires when empties concentrate in one level of a stratifier. What it may say is
    that empties concentrate in a level of an OBSERVED stratifier. What it may not say is MNAR
    (which is dependence on the UNOBSERVED value; concentration in a recorded variable is MAR
    *given* that variable), nor a field-of-view mechanism, nor a direction for the conditional
    median.

    The mechanism clause is checked in BOTH directions: for a `region` stratifier it may offer a
    pelvic-FOV hypothesis; for `scanner`, which is what AMOS and autoPET use, it must not.
    """
    import io
    import contextlib
    import stats as ST
    import gap_config as C

    def run(ds, strat_name, top, n_empty=20):
        # The cohort is chosen to MATCH the stratifier, because `summary` also prints that
        # cohort's COHORT_ELIGIBLE note and SAROS's legitimately mentions a pelvic-FOV
        # hypothesis. Asserting on the whole buffer with the wrong cohort tests the note,
        # not the mechanism clause.
        strata = {top: int(0.95 * n_empty), "other": n_empty - int(0.95 * n_empty)}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ST.summary(ds, ds.upper(), np.array([1.0, 2.0, 3.0]),
                       np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]),
                       {"empty": n_empty, "unparsed": 0, "failed": 0,
                        "metric": "surface(v2)"},
                       [], empty_strata=strata, strat_name=strat_name)
        return buf.getvalue()

    assert ST.STRATIFIER["saros"] == "region" and ST.STRATIFIER["amos"] == "scanner", (
        "the STRATIFIER map changed; this test's two cases are chosen to match it")

    assert C.MIN_GROUP_N <= 20, "the fixture must clear MIN_GROUP_N for the branch to fire"

    region_out = run("saros", "region", "abdomen")
    assert "NON-RANDOM MISSINGNESS" in region_out, "the branch did not fire on the fixture"
    for banned in ("INFORMATIVE (MNAR)", "MNAR censoring", "bounded below"):
        assert banned not in region_out, (
            f"the missingness note asserts {banned!r}")
    assert "does NOT identify MNAR" in region_out
    assert "CONDITIONAL ON MEASURABILITY" in region_out
    assert "NOT identified" in region_out
    assert "PLAUSIBLE mechanism" in region_out and "hypothesis" in region_out

    scanner_out = run("amos", "scanner", "Philips")
    assert "NON-RANDOM MISSINGNESS" in scanner_out
    assert "PLAUSIBLE mechanism" not in scanner_out and "FOV" not in scanner_out, (
        "a SCANNER concentration is being given a field-of-view mechanism; AMOS and autoPET "
        "stratify on scanner")
    assert "not a coverage variable" in scanner_out


def test_the_direction_percentages_are_computed_among_present_cases_only():
    """Touching masks have an exactly-zero PRIMARY gap (no gap at all), and they must not enter the
    axial/lateral/mixed denominator, as they would if presence were decided on the centre-to-centre
    separation, which is never sub-millimetre for two tubular veins. Constructed here so the
    arithmetic is checkable by hand: 4 cases, 2 with no primary gap.
    """
    z = np.array([0.0, 6.0, 0.0, 6.0])
    c2c = np.array([40.0, 40.0, 8.0, 8.0])
    surf = np.array([0.0, 6.0, 0.0, 6.0])        # cases 0 and 2 have NO primary gap
    counts = {"empty": 0, "unparsed": 0}
    row = ST._stat_row("axial z_gap", "fr.txt", z, counts, z=z, d=c2c, primary=surf)

    assert row["n_measurable"] == 4
    # two present cases; the percentages must be over 2, not over 4
    labels = [ST.classify_gap(a, b, c) for a, b, c in zip(z, c2c, surf)]
    assert labels.count("absent") == 2
    total = row["axial_pct"] + row["lateral_pct"] + row["mixed_pct"]
    assert total == pytest.approx(100.0), (
        f"the three direction percentages must exhaust the PRESENT cases; they sum to {total}, "
        f"which means the denominator includes the absent ones")


def test_a_row_with_no_present_case_reports_blank_rather_than_dividing_by_zero():
    """The present-only denominator must not become a crash when nothing is present."""
    z = np.array([0.0, 0.0])
    row = ST._stat_row("axial z_gap", "fr.txt", z, {"empty": 0, "unparsed": 0},
                       z=z, d=np.array([40.0, 8.0]), primary=np.array([0.0, 0.0]))
    assert row["axial_pct"] == "" and row["lateral_pct"] == "" and row["mixed_pct"] == ""
    assert row["median_lateral_residual_mm"] == ""


def test_the_decomposition_row_cannot_be_built_without_the_primary_metric():
    """`_stat_row` must refuse rather than fall back to `d`."""
    z = np.array([0.0, 6.0])
    with pytest.raises(ValueError, match="primary"):
        ST._stat_row("axial z_gap", "fr.txt", z, {"empty": 0, "unparsed": 0},
                     z=z, d=np.array([40.0, 8.0]))


def test_absent_is_reachable_on_the_shipped_classifier():
    """A label that cannot occur is not a category, and a reader of `axial_pct` could not tell that
    a missing fourth share had been redistributed across the other three.
    """
    assert ST.classify_gap(0.0, 40.0, 0.0) == "absent", (
        "a touching mask with distant centroids must be `absent`; if this returns a direction, "
        "presence is being read off the secondary metric")


def test_the_quantization_fingerprint_can_exclude_the_exact_zero_atom():
    """`v = 0` is within tol of a multiple of EVERY step, so the left-censored spike is scored as
    slice-thickness heaping when it is nothing of the kind.

    Built so the two answers are exact: 4 values at a 2.5 mm step, two of them exact zeros and
    one a genuine multiple. Inclusive -> 3/4 = 0.75; zero-excluded -> 1/2 = 0.50.
    """
    v = np.array([0.0, 0.0, 5.0, 1.3])
    steps = np.full(4, 2.5)
    assert ST.quantization_frac_percase(v, steps, tol=0.05) == pytest.approx(0.75)
    assert ST.quantization_frac_percase(v, steps, tol=0.05, exclude_zero=True) == pytest.approx(0.5)


def test_the_zero_atom_inflation_is_real_at_cohort_scale():
    """A 21.5 % zero atom on top of values that are deliberately NOT slice multiples must move the
    fraction by about the size of the atom: that is the whole mechanism.
    """
    rng = np.random.default_rng(0)
    n = 2000
    nz = int(0.215 * n)
    v = np.concatenate([np.zeros(nz), rng.uniform(0.2, 0.9, n - nz) + 3 * 2.5])
    steps = np.full(n, 2.5)
    inclusive = ST.quantization_frac_percase(v, steps, tol=0.05)
    excluded = ST.quantization_frac_percase(v, steps, tol=0.05, exclude_zero=True)
    assert inclusive - excluded == pytest.approx(0.215, abs=0.01), (
        f"the zero atom must account for its own share of the inclusive figure; measured "
        f"{inclusive:.3f} - {excluded:.3f}")
    assert excluded == pytest.approx(0.0, abs=1e-9)


def _spacing_cohort(gaps, sz, sy, sx, name="X", n_empty=0):
    """A results-dict of the shape `cross_cohort` consumes."""
    n = len(gaps)
    return {"dir": name,
            "surf": np.asarray(gaps, float),
            "sz": np.full(n, float(sz)),
            "sy": np.full(n, float(sy)),
            "sx": np.full(n, float(sx)),
            "n_empty": n_empty}


def test_the_normalizer_sensitivity_reports_every_divisor_not_just_the_flattering_one():
    """Two cohorts constructed so the divisors disagree BY CONSTRUCTION.

    Cohort A has coarse z and fine in-plane spacing; cohort B the reverse, with gaps scaled so
    that dividing by z brings them together while dividing by the in-plane geometric mean drives
    them apart. If only one divisor were reported, the same data would support either
    conclusion -- which is the finding.
    """
    a = _spacing_cohort([10.0, 11.0, 12.0, 13.0, 14.0], sz=5.0, sy=0.5, sx=0.5, name="A")
    b = _spacing_cohort([2.0, 2.2, 2.4, 2.6, 2.8], sz=1.0, sy=2.5, sx=2.5, name="B")

    rows = ST.normalizer_sensitivity([a, b])
    labels = [r[0] for r in rows]
    assert labels == ["raw mm", "/ z spacing", "/ 3-D geometric-mean spacing",
                      "/ in-plane geometric-mean spacing"]
    by = {r[0]: r[1] for r in rows}
    # Dividing by z makes the two cohorts identical (10/5 == 2/1, ...), so heterogeneity dies.
    assert by["/ z spacing"] < by["raw mm"]
    # Dividing by the in-plane mean pushes them further apart than the raw values are.
    assert by["/ in-plane geometric-mean spacing"] >= by["raw mm"]


def test_the_sensitivity_reports_nothing_rather_than_a_subset_when_spacing_is_missing():
    """A partial answer here is worse than none: three divisors out of four, silently, would
    look like a complete sensitivity analysis while omitting the one that reverses it."""
    a = _spacing_cohort([1.0, 2.0, 3.0], sz=1.0, sy=1.0, sx=1.0, name="A")
    b = _spacing_cohort([2.0, 3.0, 4.0], sz=2.0, sy=1.0, sx=1.0, name="B")
    b.pop("sy")
    assert ST.normalizer_sensitivity([a, b]) == []


def test_worst_case_median_bounds_assume_nothing_about_the_unmeasured():
    """Manski bounds, on a hand-checkable case.

    Five observed values 1..5 with two unmeasured. Lower: [0,0,1,2,3,4,5] -> median 2.
    Upper: [1,2,3,4,5,inf,inf] -> median 4. Both derived by hand, not from the function.
    """
    low, high, tipping = ST.worst_case_median_bounds([1, 2, 3, 4, 5], 2)
    assert (low, high) == (2.0, 4.0)
    assert tipping == 2

    # No unmeasured case: point-identified, and the tipping count is zero rather than blank.
    low, high, tipping = ST.worst_case_median_bounds([1, 2, 3, 4, 5], 0)
    assert (low, high, tipping) == (3.0, 3.0, 0)

    # No observed value at all: nothing to bound, and NOT (0, 0).
    assert ST.worst_case_median_bounds([], 4) == (None, None, None)


def test_the_three_dimensional_divisor_uses_all_three_spacings():
    """The study's cohorts are isotropic IN-PLANE (sy == sx on every case), so a divisor written as
    `sz*sy*sy` instead of `sz*sy*sx` would be invisible on real data and on any fixture with that
    symmetry; this one deliberately breaks it."""
    a = _spacing_cohort([4.0, 5.0, 6.0], sz=1.0, sy=8.0, sx=0.5, name="A")
    b = _spacing_cohort([4.0, 5.0, 6.0], sz=1.0, sy=0.5, sx=8.0, name="B")
    rows = {r[0]: r[1] for r in ST.normalizer_sensitivity([a, b])}
    # sz*sy*sx is 4.0 for BOTH cohorts, so a divisor using all three collapses the
    # heterogeneity to nothing; one using sy twice does not.
    assert rows["/ 3-D geometric-mean spacing"] == pytest.approx(rows["raw mm"], abs=1e-9)


def test_a_cohort_with_no_in_plane_spacing_voids_the_whole_sensitivity():
    """Three cohorts, one without in-plane spacing.

    With three, skipping one still leaves a computable (and wrong) answer, so only refusing outright
    survives."""
    a = _spacing_cohort([1.0, 2.0, 3.0], sz=1.0, sy=1.0, sx=1.0, name="A")
    b = _spacing_cohort([2.0, 3.0, 4.0], sz=2.0, sy=1.0, sx=1.0, name="B")
    c = _spacing_cohort([5.0, 6.0, 7.0], sz=4.0, sy=1.0, sx=1.0, name="C")
    c.pop("sx")
    assert ST.normalizer_sensitivity([a, b, c]) == [], (
        "a sensitivity computed over a SUBSET of cohorts is not the sensitivity, and is more "
        "misleading than none because it looks complete")


def test_the_worst_case_lower_bound_places_the_unmeasured_at_zero_not_at_the_observed_minimum():
    """On the study's cohorts the observed minimum IS 0 mm (the gap is floored there), so
    substituting `min(observed)` for `0` is invisible. A cohort whose smallest gap is well above
    zero separates them, and the distinction is the whole content of a Manski bound: the unmeasured
    cases are unconstrained, not drawn from the observed support."""
    values = [10.0, 11.0, 12.0, 13.0, 14.0]          # nothing near zero
    low, _high, _tip = ST.worst_case_median_bounds(values, 3)
    # Placing three unmeasured at 0: [0,0,0,10,11,12,13,14] -> median 10.5. Placing them at
    # the observed minimum 10.0 would give [10,10,10,10,11,12,13,14] -> median 10.5 as well,
    # so the median alone cannot separate them; the LOWER END of the support can.
    assert low == pytest.approx(10.5)
    low2, _h, _t = ST.worst_case_median_bounds(values, 6)
    # With six unmeasured at 0 the median falls into the zeros; at the observed minimum it
    # could not fall below 10.
    assert low2 == pytest.approx(0.0), (
        "the lower bound must be reachable from BELOW the observed support, or it is not a "
        "worst case")


def test_the_tipping_count_is_measured_against_the_observed_median_not_the_upper_bound():
    """The two candidate definitions coincide on the real cohorts, so they need cases built to separate
    them. Both below are hand-derived.

    CASE A -- observed 1,2,3,4,5 with 4 unmeasured. Observed median 3.
      k=0  [0,0,0,0,1,2,3,4,5]        median 1
      k=2  [0,0,1,2,3,4,5,inf,inf]    median 3   equal to observed, NOT above it
      k=3  [0,1,2,3,4,5,inf,inf,inf]  median 4   first above  -> observed-median rule gives 3
      k=4  [1,2,3,4,5,inf,inf,inf,inf] median 5 = the upper bound -> upper-bound rule gives 4

    CASE B -- observed 1,1,1,1,9,9,9,9,9 with 4 unmeasured. Observed median 9, and the upper
    bound is ALSO 9, so the median can never rise above it: the observed-median rule gives 4
    (the whole budget, i.e. "it never moves up"), while the upper-bound rule gives 2. The two
    rules disagree in OPPOSITE directions across the two cases, so neither can be mistaken for
    the other by a test that only ever sees one.
    """
    _lo, hi, tip = ST.worst_case_median_bounds([1, 2, 3, 4, 5], 4)
    assert (hi, tip) == (5.0, 3), "case A: the upper-bound rule would give 4"

    _lo, hi, tip = ST.worst_case_median_bounds([1, 1, 1, 1, 9, 9, 9, 9, 9], 4)
    assert hi == 9.0
    assert tip == 4, "case B: the upper-bound rule would give 2"


def test_the_upper_bound_is_unbounded_when_the_unmeasured_could_outnumber_the_observed():
    """Padding with the observed MAXIMUM instead of +inf is invisible whenever the median stays inside
    the observed values, which is every real cohort here (n_empty << n_measured). It becomes visible
    exactly when the unmeasured cases are numerous enough to carry the median into the padded
    region, and then the two answers are different in kind: `+inf` (the median is not bounded above
    by anything the data contain) versus the largest observed gap (a bound the data cannot support).

    A Manski bound assumes nothing about the unobserved values, so the observed maximum is
    not an upper limit on them.
    """
    _lo, high, _tip = ST.worst_case_median_bounds([1, 2, 3], 3)
    assert high == float("inf"), (
        "with as many unmeasured cases as observed ones the median is unbounded above; "
        "reporting the observed maximum would smuggle in a support assumption")

    # And it stays finite in the ordinary case, so the sentinel is not a blanket.
    _lo, high, _tip = ST.worst_case_median_bounds([1, 2, 3, 4, 5], 2)
    assert high == 4.0


def test_the_bound_rows_carry_each_cohort_s_own_unmeasured_count():
    """The estimator can be right while the WIRING drops `n_empty`.

    Two cohorts, one with unmeasured cases and one without. If the caller passed a constant
    zero, both would come back point-identified and the published table would say so.
    """
    a = _spacing_cohort([1.0, 2.0, 3.0, 4.0, 5.0], sz=1.0, sy=1.0, sx=1.0, name="A", n_empty=4)
    b = _spacing_cohort([1.0, 2.0, 3.0, 4.0, 5.0], sz=1.0, sy=1.0, sx=1.0, name="B", n_empty=0)
    rows = {r[0]: r for r in ST.worst_case_rows([a, b])}

    assert rows["A"][2] == 4 and rows["B"][2] == 0
    assert rows["A"][4] < rows["A"][3] < rows["A"][5], "A must be bounded, not point-identified"
    assert rows["B"][4] == rows["B"][3] == rows["B"][5], "B has nothing to bound"
    # The observed median is the same in both; only the missingness differs, so a caller that
    # dropped n_empty would make the two rows identical.
    assert rows["A"] != rows["B"]


def test_the_eligible_frame_rows_use_each_cohort_s_own_eligible_denominator():
    import gap_config as C

    rows = {r[0]: r for r in ST.eligible_frame_worst_case_rows([
        {"dir": "SAROS", "ds": "saros", "surf": np.array([1.0, 2.0, 3.0, 4.0, 5.0]), "n_empty": 2},
        {"dir": "autoPET", "ds": "autopet", "surf": np.array([1.0, 2.0, 3.0, 4.0, 5.0]), "n_empty": 0}])}
    assert set(rows) == {"SAROS", "autoPET"}
    for name, key in (("SAROS", "saros"), ("autoPET", "autopet")):
        n_elig = C.COHORT_ELIGIBLE[key][1]
        assert rows[name][2] == n_elig, f"{name}: the row does not carry its own eligible denominator"
        assert rows[name][4] == n_elig - 5, f"{name}: unmeasured is not eligible minus measured"
    # identical data, different denominators: a wiring that dropped `ds` would make the rows equal
    assert rows["SAROS"][6] < rows["SAROS"][5] < rows["SAROS"][7]
    assert rows["SAROS"][6:9] != rows["autoPET"][6:9]

    # A denominator smaller than the measured count is a contradiction, not a zero.
    with pytest.raises(RuntimeError, match="eligible"):
        ST.eligible_frame_worst_case_rows(
            [{"dir": "SAROS", "ds": "saros", "surf": np.arange(500.0), "n_empty": 0}])


def test_ci_formatter_requires_both_endpoints_to_be_finite():
    """`_ci` must return blank when either end is non-finite or None.

    A test that fed only a bad `lo` would miss a formatter that ignores `hi` (which would print
    `5.0-nan`, `-inf-5.0` or `5.0-inf`).
    """
    nan, inf = float("nan"), float("inf")
    assert ST._ci(5.0, 7.0) == "5.0-7.0", "a finite interval must still format"
    for lo, hi in ((5.0, nan), (nan, 5.0), (-inf, 5.0), (5.0, inf), (inf, inf), (nan, nan)):
        assert ST._ci(lo, hi) == "", (
            f"_ci({lo}, {hi}) must be blank, got {ST._ci(lo, hi)!r}")
    assert ST._ci(None, 5.0) == "" and ST._ci(5.0, None) == ""


def test_ci_publishes_a_point_as_the_token_never_as_an_interval():
    """`_ci` formats every interval this module publishes, and a zero-width interval is not one: a
    point prints as the token DEGENERATE_ZERO_WIDTH, while a finite interval and the absent cases
    keep their formats (the test above)."""
    assert ST._ci(2.5, 2.5, 2) == "DEGENERATE_ZERO_WIDTH"
    assert ST._ci(0.0, 0.0, 2) == "DEGENERATE_ZERO_WIDTH"
    assert ST._ci(-1.0, -1.0) == "DEGENERATE_ZERO_WIDTH"
    assert ST._ci(5.0, 7.0) == "5.0-7.0"                       # positive control
    assert ST._ci(float("nan"), float("nan"), 2) == ""          # absent stays blank


def test_ci_prints_the_ends_to_the_precision_that_separates_them():
    """At one decimal the interval [2.50, 2.55] would print as `2.5-2.5` (2.55 is stored as
    2.5499...), a real interval that reads as a point. The ends get the decimals that separate them,
    and ends that already differ print at the requested precision."""
    assert ST._ci(2.5, 2.55, 1) == "2.50-2.55"
    assert ST._ci(2.49, 2.51, 1) == "2.49-2.51"
    assert ST._ci(2.5, 2.5004, 2) == "2.5000-2.5004"
    assert ST._ci(3.0, 4.0, 1) == "3.0-4.0" and ST._ci(5.24, 5.94, 2) == "5.24-5.94"
    # the printed VALUES decide, not the strings: -0.00 and 0.00 are one point
    assert ST._ci(-0.001, 0.001, 2) == "-0.001-0.001"
    assert ST._ci(1.0, 1.000001, 1) == "1.000000-1.000001"
    assert ST._ci(1.0, 1.0000001, 1) == "1.0-1.0000001"
    assert ST._ci(1.0, 1.0 + 1e-12, 2) == "1.0-1.000000000001"


def test_every_interval_the_module_prints_goes_through_the_one_formatter():
    """Every `95%CI {` site in stats.py formats its interval through `_ci`."""
    src = (ROOT / "Code" / "phase1_characterize" / "stats.py").read_text(encoding="utf-8")
    sites = [m.start() for m in re.finditer(r"95%CI \{", src)]
    # six sites: _desc_line, the slice-normalized line, the per-stratum lines, cross_cohort and
    # the two zero-decimal Wilson lines
    assert len(sites) >= 6, f"only {len(sites)} '95%CI {{' sites found; the guard has lost its target"
    bypass = [src[i:i + 60] for i in sites if not src.startswith("95%CI {_ci(", i)]
    assert not bypass, "intervals printed without `_ci`:\n  " + "\n  ".join(bypass)


def test_the_cluster_robust_console_line_prints_the_tables_own_ends_in_order():
    """The ends are the CSV's own, already rounded to four decimals, so the console says what the table
    says."""
    r = {"metric": "3D surface-to-surface (PRIMARY) median", "iid_lo": 2.5, "iid_hi": 2.955,
         "cluster_lo": 2.5, "cluster_hi": 2.97, "icc": 0.8199, "design_effect": 1.2603,
         "width_ratio": 1.033}
    assert ST._cluster_robust_console(r) == (
        "    3D surface-to-surface (PRIMARY) median iid [2.5000,2.9550] -> cluster [2.5000,2.9700]"
        "  ICC=0.8199 DEFF=1.2603 width x1.033")
    tied = dict(r, cluster_lo=2.97)
    assert "cluster [DEGENERATE_ZERO_WIDTH]" in ST._cluster_robust_console(tied)
    blank = dict(r, iid_lo=float("nan"))
    assert "iid [NA]" in ST._cluster_robust_console(blank)


def test_the_distribution_annotation_prints_both_intervals_with_its_en_dash():
    """Driven here: a real interval keeps the en dash and its order; a collapsed median prints the
    token, never `m\u2013m`; the IQR, a range of the data, keeps its own format."""
    text = ST._distribution_annotation(pk=85, n=100, pp=85.0, plo=76.71, phi=90.72, med=5.2,
                                       mlo=5.0, mhi=5.94, p25=2.5, p75=8.1, mean=6.3, sd=4.4)
    assert text == ("present 85/100 (85%, 95% CI 77\u201391)\n"
                    "median 5.2 mm (95% CI 5.0\u20135.9; IQR 2.5\u20138.1)\n"
                    "mean 6.3 \u00b1 4.4 mm (SD)")
    tied = ST._distribution_annotation(85, 100, 85.0, 76.71, 90.72, 5.0, 5.0, 5.0, 2.5, 8.1, 6.3, 4.4)
    assert "median 5.0 mm (95% CI DEGENERATE_ZERO_WIDTH; IQR" in tied


def test_the_cluster_and_annotation_intervals_print_na_for_an_absent_end():
    """An absent end reaches these helpers as NaN from an estimator or as a blank cell read back from
    the CSV; both are driven."""
    nan = float("nan")
    r = {"metric": "m", "iid_lo": 2.5, "iid_hi": 2.955, "cluster_lo": nan, "cluster_hi": 2.97,
         "icc": 0.82, "design_effect": 1.26, "width_ratio": 1.03}
    assert "-> cluster [NA]" in ST._cluster_robust_console(r)
    assert "-> cluster [NA]" in ST._cluster_robust_console(dict(r, cluster_lo="", cluster_hi=""))
    text = ST._distribution_annotation(0, 0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan)
    assert "95% CI NA)" in text and "(95% CI NA;" in text, text


def test_the_distribution_figure_hands_the_annotation_its_intervals_in_order(tmp_path, monkeypatch):
    """The real call is made here, on a sample with no atom, through a spy that records what it passes:
    the present rate's and the median's intervals are the estimators' own, lower end first."""
    seen = []
    real = ST._distribution_annotation

    def spy(*args):
        seen.append(args)
        return real(*args)

    monkeypatch.setattr(ST, "_distribution_annotation", spy)
    d = _continuous_gap()
    ST.figures("AMOS", d.copy(), d, [], tmp_path)
    assert len(seen) == 1, f"the annotation was built {len(seen)} times"
    pk = int((d > C.PRESENT_MM).sum())
    pp, plo, phi = S.proportion_pct_ci(pk, d.size, C.ALPHA)
    mlo, mhi = S.median_ci_auto(d, C.CI_METHOD, C.ALPHA, C.N_BOOT, C.SEED)
    expected = (pk, d.size, pp, plo, phi, np.median(d), mlo, mhi, np.percentile(d, 25),
                np.percentile(d, 75), d.mean(), S.sample_sd(d))
    assert seen[0] == expected, (seen[0], expected)
    assert plo < phi and mlo < mhi and np.percentile(d, 25) < np.median(d) < np.percentile(d, 75)

def test_the_present_and_band_rates_print_their_own_wilson_intervals_in_order():
    """Present 30 of 40 and band 12 of 40 differ, so the two sites cannot be exchanged either."""
    import contextlib
    import io
    surf = np.array([0.0] * 10 + [5.0] * 12 + [12.0] * 18)
    assert C.PRESENT_MM < 5.0 and C.BAND_LO_MM <= 5.0 <= C.BAND_HI_MM < 12.0   # fixture premises
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ST.summary("amos", "AMOS", surf.copy(), surf, surf.copy(),
                   {"empty": 0, "unparsed": 0, "failed": 0, "metric": "surface(v2)"}, [])
    out = buf.getvalue()
    for k, text in ((30, "(75%, 95%CI "), (12, "30% (95%CI ")):
        _pct, lo, hi = S.proportion_pct_ci(k, 40, C.ALPHA)
        assert lo < hi
        expected = f"{text}{ST._ci(lo, hi, 0)})"
        assert expected in out, f"expected {expected!r} in:\n{out}"
        assert ST._ci(lo, hi, 0) == f"{lo:.0f}-{hi:.0f}", "the ends must print in order"

def _lattice_z():
    """A z_gap-like lattice of whole 2.5-mm slices: 30 % at 0, an atom of 40 % at 2.5 mm that
    straddles the middle, 30 % at 5.0 mm. Measured at the default method and seed: every resample's
    median is 2.5, so BCa and percentile both collapse to [2.5, 2.5]."""
    z = np.array([0.0] * 120 + [2.5] * 160 + [5.0] * 120)
    assert np.median(z) == 2.5 and np.mean(z == 2.5) == 0.40 and np.mean(z < 2.5) == 0.30
    return z


def _continuous_gap():
    v = np.random.default_rng(7).gamma(2.0, 2.0, 400)
    assert np.unique(v).size == v.size                          # no atom anywhere
    return v


def test_the_all_stats_row_publishes_a_collapsed_median_interval_as_the_token():
    """A continuous sample keeps a real interval."""
    row = ST._stat_row("axial z_gap", "fr.txt", _lattice_z(), {"empty": 0, "unparsed": 0})
    assert row["median_mm"] == 2.5
    assert row["median_ci"] == "DEGENERATE_ZERO_WIDTH", row["median_ci"]
    row = ST._stat_row("3D surface", "fr.txt", _continuous_gap(), {"empty": 0, "unparsed": 0})
    lo, hi = (float(x) for x in row["median_ci"].split("-"))
    assert lo < row["median_mm"] < hi


def test_the_console_summary_prints_a_collapsed_median_interval_as_the_token():
    """`_desc_line` formats its interval through `_ci`, so a collapsed median prints the token; a
    continuous sample keeps a real interval."""
    line = ST._desc_line(_lattice_z())
    assert "(95%CI DEGENERATE_ZERO_WIDTH)" in line and "2.5-2.5" not in line, line
    line = ST._desc_line(_continuous_gap())
    assert re.search(r"\(95%CI \d+\.\d+-\d+\.\d+\)", line), line


def test_the_cross_cohort_record_prints_a_collapsed_median_interval_as_the_token(tmp_path):
    """A lattice cohort must read as the token, beside a continuous cohort that keeps its interval."""
    sz = np.full(400, 2.5)
    results = [{"surf": _lattice_z(), "counts": {"empty": 0}, "dir": "AMOS", "ds": "amos", "sz": sz},
               {"surf": _continuous_gap(), "counts": {"empty": 0}, "dir": "SAROS", "ds": "saros",
                "sz": sz}]
    ST.cross_cohort(results, outdir=tmp_path)
    text = (tmp_path / "_cross_cohort.txt").read_text(encoding="utf-8")
    amos = next(ln for ln in text.splitlines() if ln.startswith("  AMOS ") and "95%CI" in ln)
    saros = next(ln for ln in text.splitlines() if ln.startswith("  SAROS ") and "95%CI" in ln)
    assert "(95%CI DEGENERATE_ZERO_WIDTH)" in amos, amos
    assert re.search(r"\(95%CI \d+\.\d+-\d+\.\d+\)", saros), saros


_PRINTED_CI = re.compile(r"95%CI (-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)\)")


def _printed_in_order(text):
    """Every `95%CI a-b)` printed in `text`, with a < b required of each."""
    pairs = [(float(a), float(b)) for a, b in _PRINTED_CI.findall(text)]
    assert pairs and all(a < b for a, b in pairs), f"an interval is printed out of order in {text!r}"
    return pairs


def test_the_console_and_record_intervals_print_their_estimators_ends_in_order(tmp_path, capsys):
    """The two tests above check that a collapsed interval reads as the token and that a real one
    matches `\\d+-\\d+`, which a reversed interval also matches, so each site is driven here on
    samples with no atom: the printed interval is the estimator's own at the site's precision (one
    end passed twice, or another precision, fails) and its lower end is printed first (a swap
    fails)."""
    def est(x):
        return S.median_ci_auto(x, C.CI_METHOD, C.ALPHA, C.N_BOOT, C.SEED)

    v = _continuous_gap()
    lo, hi = est(v)
    assert lo < hi                                             # the fixture's premise: a real interval

    dsc = S.describe(v, alpha=C.ALPHA, n_boot=C.N_BOOT, seed=C.SEED, ci_method=C.CI_METHOD)
    line = ST._desc_line(v)                                    # the summary's median line
    assert f"(95%CI {ST._ci(dsc['median_ci_lo'], dsc['median_ci_hi'], 1)})" in line, line
    _printed_in_order(line)

    sz = np.full(v.size, 2.5)                                  # the slice-normalized line
    ST.summary("amos", "AMOS", v.copy(), v, v.copy(),
               {"empty": 0, "unparsed": 0, "failed": 0, "metric": "surface(v2)"}, [], sz=sz)
    out = capsys.readouterr().out
    su, n_su = ST.slice_normalized(v, sz)
    slo, shi = est(su)
    norm = next(ln for ln in out.splitlines() if "slice-normalized gap" in ln)
    assert f"(95%CI {ST._ci(slo, shi, 2)}), n={n_su}" in norm, norm
    _printed_in_order(norm)

    rows, meta = [], {}                                        # the per-stratum lines
    for i, g in enumerate(v):
        cid = f"amos_{i:04d}"
        rows.append((cid, 0.0, float(g), float(g), "OK"))
        meta[cid] = {"scanner": ("A", "B", "C")[i % 3]}
    ST.stratified_primary("AMOS", "amos", rows, meta)
    out = capsys.readouterr().out
    for k in ("A", "B", "C"):
        a = np.array([float(g) for i, g in enumerate(v) if ("A", "B", "C")[i % 3] == k])
        klo, khi = est(a)
        line = next(ln for ln in out.splitlines() if ln.strip().startswith(f"{k} "))
        assert f"(95%CI {ST._ci(klo, khi, 1)}) " in line, line
        _printed_in_order(line)

    w = np.random.default_rng(11).gamma(3.0, 1.5, 300)        # the cross-cohort record
    results = [{"surf": v, "counts": {"empty": 0}, "dir": "AMOS", "ds": "amos", "sz": sz},
               {"surf": w, "counts": {"empty": 0}, "dir": "SAROS", "ds": "saros",
                "sz": np.full(w.size, 2.5)}]
    ST.cross_cohort(results, outdir=tmp_path)
    text = (tmp_path / "_cross_cohort.txt").read_text(encoding="utf-8")
    for name, x in (("AMOS", v), ("SAROS", w)):
        xlo, xhi = est(x)
        line = next(ln for ln in text.splitlines() if ln.startswith(f"  {name} ") and "95%CI" in ln)
        assert f"(95%CI {ST._ci(xlo, xhi, 2)})" in line, line
        _printed_in_order(line)


def test_each_correlate_console_line_prints_its_table_eps2_interval(capsys):
    """Each variable's cell must appear, bracketed, on its console line."""
    rng = np.random.default_rng(3)
    rows, meta = [], {}
    for i in range(90):
        cid = f"case_{i:03d}"
        sex, scanner = ("M", "F")[i % 2], ("A", "B", "C")[i % 3]
        g = 4.0 + (0.8 if sex == "M" else 0.0) + (0.5 * (i % 3)) + rng.normal(0, 1.0)
        rows.append((cid, 0.0, g, g, "OK"))
        meta[cid] = dict(sex=sex, age=None, scanner=scanner, contrast=None, region=None)
    out_rows, _panels, _ct = ST.correlates("X", rows, meta)
    out = capsys.readouterr().out
    cells = {r["variable"]: r["eps2_ci"] for r in out_rows if r.get("eps2_ci")}   # not `overall`
    assert {"sex", "scanner"} <= set(cells), cells
    for var, cell in cells.items():
        line = next(ln for ln in out.splitlines() if ln.strip().startswith(f"{var} ") and "eps^2=" in ln)
        assert f"[{cell}]" in line, (var, cell, line)
        lo, hi = (float(x) for x in cell.split("-"))
        assert lo < hi, cell


def test_the_correlate_level_prints_a_narrow_interval_to_the_precision_that_separates_it(
        monkeypatch):
    """The estimator is pinned to that interval so that only the formatting is under test."""
    monkeypatch.setattr(S, "median_ci_auto", lambda *a, **k: (2.5, 2.55))
    assert ST._level_stats([2.0, 2.5, 2.5, 3.0])["median_ci"] == "2.50-2.55"


def test_pfmt_rejects_a_non_finite_p_value():
    """An infinite or NaN p/q-value is not a p-value."""
    assert ST._pfmt(float("inf")) == "" and ST._pfmt(float("-inf")) == ""
    assert ST._pfmt(float("nan")) == "" and ST._pfmt(None) == ""
    assert ST._pfmt(0.03) == "0.0300", "a real p-value must still format"
    assert ST._pfmt(1.2e-18) == "1.20e-18", "the tiny-but-nonzero path must survive"


def test_the_manski_worst_case_bound_may_be_infinite():
    """`worst_case_median_bounds` returns `high = inf` when enough unmeasured cases could push the
    median above every observed value -- an unbounded worst case is the honest answer there, and
    it is printed by its own emitter rather than through `_ci`. A cleanup that made every
    infinity vanish would silently convert "unbounded" into a finite-looking number.
    """
    import numpy as np
    low, high, tipping = ST.worst_case_median_bounds(np.array([1.0, 2.0, 3.0]), 10)
    assert high == float("inf"), (
        f"ten unmeasured cases against three observed must leave the worst-case median "
        f"unbounded, got {high}")
    assert np.isfinite(low), "the lower bound is still identified"


# qc_flagged is blank where no QC rule applies (rows 3-5) and a count where one does. The column
# cannot be split: the 32-col / 5-row shape is contractual and `collapsed_class_stats` derives from
# it. So "not computed" is spelled the way this schema spells not-applicable everywhere else --
# blank. The blank/zero pair is tested in BOTH directions, because a sentinel whose absent branch is
# the only one exercised passes when the fixture's expected answer IS the degenerate answer.

def _qc_rows(qc_small, big_surface_gaps=0):
    """Five all_stats rows with a controllable qc_small and outlier count."""
    surf = np.array([4.5, 6.8, 8.9] + [C.QC_MAX_GAP_MM + 50.0] * big_surface_gaps)
    z = np.array([5.0, 7.0, 9.0] + [10.0] * big_surface_gaps)
    c2c = np.array([5.0, 7.1, 9.2] + [11.0] * big_surface_gaps)
    fr_counts = {"measurable": surf.size, "empty": 0, "unparsed": 0, "qc_small": qc_small}
    cmp_counts = {"measurable": surf.size, "empty": 0, "unparsed": 0}
    return ST.all_stats(z, surf, c2c, fr_counts, surf.copy(), surf.copy(), cmp_counts,
                        "fr.txt", "cmp.txt", censored=0.0)


def test_a_row_whose_qc_was_never_computed_publishes_blank_not_zero():
    """Rows 3-5 have no QC rule at all. Publishing `0` asserts that QC ran and found
    nothing, which is a claim the producer never made."""
    rows = _qc_rows(qc_small=0)
    never_computed = rows[2:]
    assert len(never_computed) == 3
    for r in never_computed:
        assert r["qc_flagged"] == "", (
            f"{r['metric']!r} publishes qc_flagged={r['qc_flagged']!r}; no QC count is "
            f"computed for this row, so it must be blank. A literal 0 is indistinguishable "
            f"from a real zero count.")


def test_a_row_whose_qc_ran_and_found_none_publishes_zero_not_blank():
    """The other direction, and the one that normally ships untested. Blank must mean
    'never asked'; a genuine zero must survive as 0."""
    rows = _qc_rows(qc_small=0, big_surface_gaps=0)
    assert rows[0]["qc_flagged"] == 0, (
        "the axial row's qc_small WAS computed and is genuinely 0; blanking it would lose "
        "the computed/not-computed distinction")
    assert rows[1]["qc_flagged"] == 0, (
        "the primary row's outlier count WAS computed and is genuinely 0")


def test_the_two_computed_rows_do_not_carry_each_others_count():
    """The column means different things in rows 1 and 2. If they were ever swapped the
    totals would still look plausible, so pin which count lands where."""
    rows = _qc_rows(qc_small=7, big_surface_gaps=2)
    assert rows[0]["qc_flagged"] == 7, (
        f"row 1 (axial z_gap) must carry qc_small (degenerate masks); got "
        f"{rows[0]['qc_flagged']!r}")
    assert rows[1]["qc_flagged"] == 2, (
        f"row 2 (3D surface, PRIMARY) must carry the count of gaps above QC_MAX_GAP_MM "
        f"({C.QC_MAX_GAP_MM} mm); got {rows[1]['qc_flagged']!r}")
