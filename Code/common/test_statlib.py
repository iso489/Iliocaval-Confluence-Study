import importlib
import math
import sys
from decimal import Decimal
from fractions import Fraction
from unittest import mock

import numpy as np
import pytest

import statlib as S


# --- dispersion -----------------------------------------------------------
def test_sample_sd_textbook():
    # SD of 1..5 with ddof=1 is sqrt(2.5)
    assert S.sample_sd([1, 2, 3, 4, 5]) == pytest.approx(math.sqrt(2.5))


def test_sample_sd_differs_from_population():
    v = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    assert S.sample_sd(v) == pytest.approx(np.std(v, ddof=1))
    assert S.sample_sd(v) != pytest.approx(np.std(v, ddof=0))


def test_sample_sd_singleton_is_nan():
    assert math.isnan(S.sample_sd([3.0]))


def test_skewness_symmetric_near_zero():
    assert abs(S.skewness([-2, -1, 0, 1, 2])) < 1e-9


def test_skewness_right_tail_positive():
    assert S.skewness([0, 0, 0, 1, 10]) > 0


def test_skewness_matches_scipy_bias_true():
    from scipy.stats import skew
    rng = np.random.default_rng(1)
    v = rng.gamma(2, 2, 200)
    assert S.skewness(v) == pytest.approx(float(skew(v, bias=True)))


# --- Wilson proportion CI -------------------------------------------------
def test_wilson_ci_8_of_10():
    lo, hi = S.wilson_ci(8, 10)
    assert 0.49 < lo < 0.50 and 0.94 < hi < 0.95


def test_wilson_ci_brackets_phat_and_is_ordered():
    lo, hi = S.wilson_ci(42, 100)
    assert lo < 0.42 < hi


def test_wilson_ci_full_successes():
    lo, hi = S.wilson_ci(10, 10)
    assert hi == pytest.approx(1.0) and lo < 1.0   # never degenerate to [1,1]


def test_wilson_ci_zero_successes_stays_in_unit_interval():
    lo, hi = S.wilson_ci(0, 10)
    assert lo == pytest.approx(0.0) and 0.0 < hi < 1.0


def test_wilson_ci_n_zero_is_nan():
    lo, hi = S.wilson_ci(0, 0)
    assert math.isnan(lo) and math.isnan(hi)


def test_wilson_ci_brackets_its_estimate_at_the_extremes():
    """At k = 0 the Wilson centre and half-width are equal, and at k = n they sum to one, so the bounds
    meet the point estimate exactly in arithmetic but not in floating point: unclamped,
    `wilson_ci(0, 17)` gives a lower bound of 1.39e-17 and `wilson_ci(4, 4)` an upper bound of
    0.9999999999999999. The two tests above use `pytest.approx`, which accepts both, and an error
    bar drawn from such an interval would be negative. The interval must contain its own estimate,
    exactly."""
    for n in (4, 9, 15, 17, 21, 29, 33, 38, 40, 1014):
        lo, hi = S.wilson_ci(0, n)
        assert lo == 0.0 and 0.0 < hi < 1.0, (n, lo, hi)
        lo, hi = S.wilson_ci(n, n)
        assert hi == 1.0 and 0.0 < lo < 1.0, (n, lo, hi)
    for n in range(1, 300):
        for k in sorted({0, 1, n // 2, n - 1, n}):
            lo, hi = S.wilson_ci(k, n)
            assert lo <= k / n <= hi, (k, n, lo, hi)


# --- McNemar paired-binary test -------------------------------------------
def _sm_mcnemar(b, c, exact, correction=True):
    """statsmodels oracle: mcnemar takes the full 2x2 [[a,b],[c,d]]; a,d cancel."""
    from statsmodels.stats.contingency_tables import mcnemar
    res = mcnemar([[0, b], [c, 0]], exact=exact, correction=correction)
    return float(res.pvalue)


def test_mcnemar_exact_matches_statsmodels():
    for b, c in [(10, 0), (3, 7), (1, 8), (12, 5)]:
        r = S.mcnemar_test(b, c, exact=True)
        assert r["p"] == pytest.approx(_sm_mcnemar(b, c, exact=True), abs=1e-9)
        assert r["method"] == "exact_binomial"
        assert r["n_discordant"] == b + c


def test_mcnemar_chi2_continuity_matches_statsmodels():
    for b, c in [(30, 10), (100, 60), (40, 25)]:
        r = S.mcnemar_test(b, c, exact=False)
        assert r["p"] == pytest.approx(_sm_mcnemar(b, c, exact=False, correction=True),
                                       rel=1e-6, abs=1e-9)
        assert r["method"] == "chi2_continuity"


def test_mcnemar_chi2_extreme_tail_is_nonzero_and_matches_scipy():
    """The actual collapsed-class imbalance must not underflow to an exact zero p."""
    from scipy.stats import chi2 as scipy_chi2

    b, c = 177, 1
    r = S.mcnemar_test(b, c, exact=False)
    expected = float(scipy_chi2.sf(r["chi2"], 1))

    assert 0.0 < r["p"] < 1e-30
    assert r["p"] == pytest.approx(expected, rel=1e-12)


def test_mcnemar_symmetric_split_is_nonsignificant():
    # b == c: no evidence the marginal rates differ. Exact gives exactly p = 1.0;
    # the Yates correction is floored so chi2 = 0 (not the spurious 1/n).
    r = S.mcnemar_test(6, 6, exact=True)
    assert r["p"] == pytest.approx(1.0)
    assert not r["significant"]
    rc = S.mcnemar_test(50, 50, exact=False)
    assert rc["chi2"] == pytest.approx(0.0)
    assert rc["p"] == pytest.approx(1.0)


def test_mcnemar_no_discordant_pairs_is_p1_method_none():
    r = S.mcnemar_test(0, 0)
    assert r["p"] == 1.0 and r["method"] == "none" and r["n_discordant"] == 0
    assert not r["significant"]


def test_mcnemar_auto_switches_exact_vs_chi2_at_25():
    assert S.mcnemar_test(12, 12, exact=None)["method"] == "exact_binomial"   # n=24 < 25
    assert S.mcnemar_test(13, 12, exact=None)["method"] == "chi2_continuity"  # n=25


def test_mcnemar_all_discordant_one_side_is_significant():
    r = S.mcnemar_test(15, 0, exact=True)   # every discordant pair went one way
    assert r["p"] < 1e-4 and r["significant"]


def test_mcnemar_negative_or_nonfinite_is_nan_not_raised():
    assert math.isnan(S.mcnemar_test(-1, 3)["p"])
    assert S.mcnemar_test(-1, 3)["method"] == "invalid"
    assert math.isnan(S.mcnemar_test(float("nan"), 3)["p"])


# --- paired median-difference CI ------------------------------------------
def test_paired_median_diff_ci_recovers_shift():
    rng = np.random.default_rng(0)
    y = rng.gamma(2, 2, 300)
    x = y + 1.5                                   # constant +1.5 paired shift
    m, lo, hi = S.paired_median_diff_ci(x, y)
    assert m == pytest.approx(1.5, abs=1e-9)      # exact-difference vector
    assert lo <= 1.5 <= hi


def test_paired_median_diff_ci_identical_is_zero_width():
    v = [1.0, 2.0, 3.0, 4.0, 5.0]
    m, lo, hi = S.paired_median_diff_ci(v, v)     # collapsed == original everywhere
    assert m == pytest.approx(0.0) and lo == pytest.approx(0.0) and hi == pytest.approx(0.0)


def test_paired_median_diff_ci_empty_is_nan():
    m, lo, hi = S.paired_median_diff_ci([], [])
    assert math.isnan(m) and math.isnan(lo) and math.isnan(hi)


def test_paired_median_diff_ci_drops_nonfinite_pairs():
    x = [1.0, 2.0, np.nan, 4.0]
    y = [0.0, 1.0, 1.0, 3.0]
    m, lo, hi = S.paired_median_diff_ci(x, y)
    assert m == pytest.approx(1.0)                # diffs {1,1,1} after dropping the NaN pair


def test_the_degenerate_interval_token_is_spelled_as_published():
    """The token is a cell value in the published tables, so its spelling is a contract with every
    reader of them; it is written out here rather than read back from the module."""
    assert S.DEGENERATE_INTERVAL == "DEGENERATE_ZERO_WIDTH"


@pytest.mark.parametrize("lo, hi, expected", [
    (2.5, 2.5, True),                          # autoPET's axial z_gap: a lattice atom
    (0.0, 0.0, True),                          # the collapsed-class paired difference
    (-1.0, -1.0, True),                        # a negative point is still a point
    (np.float64(5.0), np.float64(5.0), True),  # what the estimators actually return
    (3, 3, True),
    (2.5, 2.55, False),
    (0.0, 0.6041666666666667, False),          # a 51 % atom with the rest above: honest
    (5.0, 4.0, False),                         # reversed ends are not a point
    (float("nan"), float("nan"), False),       # absent, not degenerate: `_ci` blanks it
    (5.0, float("nan"), False),
    (float("inf"), float("inf"), False),       # inf == inf, but an infinite end is no point
    (None, None, False),
    (None, 5.0, False),
    (5.0, None, False),                        # an absent upper end is no point
    (1.0, 1.0 + 1e-12, False),                 # however narrow: no tolerance makes it a point
    (-0.0, 0.0, True),                         # signed zeros are one point
    (np.float32(2.5), np.float32(2.5), True),  # numpy scalars are real numbers
    (np.int64(3), np.int64(3), True),
    (True, True, False),                       # a bool is not an interval end
    (1, True, False),                          # 1 == True in Python: the upper end's gate decides
    (True, 1.0, False),
    (5.0, Decimal("5"), False),                # equal, but a Decimal is no `numbers.Real` end
    (Decimal("5"), 5.0, False),
    (np.float32(0.1), 0.1, False),             # NumPy 2 says equal; the ends differ in float64
    (np.float32(2.5), 2.5, True),              # exactly representable: one point
    (2**53, 2**53 + 1, False),                 # distinct integers that share a float64
    (np.int64(2**53 + 1), np.float64(2**53), False),
    (np.uint64(2**64 - 1), float(2**64), False),
    (np.int64(3), 3.0, True),                  # an integer and a float that are one number
    (0.1, np.float32(0.1), False),             # the float32 pair in either order
    ("5", "5", False),                         # nor is a string that spells one
])
def test_interval_is_degenerate_exactly_for_a_finite_point(lo, hi, expected):
    assert S.interval_is_degenerate(lo, hi) is expected


@pytest.mark.parametrize("x, expected", [
    (2.5, True), (0, True), (np.float32(1.5), True), (np.int64(7), True), (-0.0, True),
    (float("nan"), False), (float("inf"), False), (-float("inf"), False), (None, False),
    (True, False), (np.bool_(True), False), ("2.5", False), (np.float64("nan"), False),
    (10**400, False),                                         # past float64: not printable
])
def test_is_finite_real_is_the_one_gate_for_an_interval_end(x, expected):
    assert S.is_finite_real(x) is expected


TOKEN = "DEGENERATE_ZERO_WIDTH"


@pytest.mark.parametrize("args, kw, expected", [
    ((2.5, 2.5, 2), {}, TOKEN),                                # a point: the token, never m-m
    ((-0.0, 0.0, 2), {}, TOKEN),
    ((2.5, 2.5, 1), {"sep": "–"}, TOKEN),                 # whatever the site's separator
    ((2.5, 2.55, 1), {}, "2.50-2.55"),
    ((1.0, 1.000001, 1), {}, "1.000000-1.000001"),             # six decimals, the cap
    ((1.0, 1.0000001, 1), {}, "1.0-1.0000001"),                # past it: shortest round-trip form
    ((1e6, 1e6 + 1e-7, 2), {}, "1000000.0-1000000.0000001"),   # narrow is never printed as a point
    ((2**53, 2**53 + 1, 0), {}, "9007199254740992-9007199254740993"),
    ((1.0, 2.0, 7), {}, "1.0000000-2.0000000"),                # nd above the cap is honoured
    ((np.float32(2.5), np.float32(2.75), 2), {}, "2.50-2.75"),
    ((np.int64(2), np.int64(3), 1), {}, "2.0-3.0"),
    ((True, False, 1), {}, ""), (("5", "6", 1), {}, ""),       # not ends: blank
    ((5.0, None, 1), {}, ""), ((None, 5.0, 1), {}, ""),
    ((float("nan"), 5.0, 1), {}, ""), ((5.0, float("inf"), 1), {}, ""),
    ((3.0, 5.0, 1), {"sep": ",", "signed": True}, "+3.0,+5.0"),  # a site's own text is kept
    ((-5.0, -3.0, 1), {"sep": ",", "signed": True}, "-5.0,-3.0"),
    ((5.0, 5.94, 1), {"sep": ", "}, "5.0, 5.9"),
    ((10.0, 10.0000001, 1), {"sep": ",", "signed": True}, "+10.0,+10.0000001"),  # sign kept
    ((Fraction(5, 2), Fraction(5, 2) + Fraction(1, 10**20), 1), {},
     "5/2-250000000000000000001/100000000000000000000"),        # exact, never `2-2`
    ((10**400, 10**400 + 1, 0), {}, ""),
])
def test_format_ci_prints_a_point_as_the_token_and_a_real_interval_as_two_ends(args, kw, expected):
    assert S.format_ci(*args, **kw) == expected


@pytest.mark.parametrize("lo, hi", [
    (2.0, 1.0), (2.55, 2.5), (np.float32(2.5), 2.0), (0.1, -0.1), (Fraction(3, 2), 1),
    (np.int64(2**53 + 1), np.float64(2**53)),   # NumPy calls these equal; they are not
])
def test_format_ci_refuses_reversed_ends(lo, hi):
    """No estimator here returns a lower end above the upper, so a reversed pair is a caller's swap,
    and fails loudly. Equal ends, signed zeros included, are a point and print the token."""
    with pytest.raises(ValueError, match="above upper end"):
        S.format_ci(lo, hi, 1)
    assert S.format_ci(0.0, -0.0, 2) == TOKEN and S.format_ci(-0.0, 0.0, 2) == TOKEN


def _printed_ends(text):
    """The two numbers a `lo-hi` cell prints, splitting at the `-` that separates them (not a sign
    and not an exponent's)."""
    for i, ch in enumerate(text):
        if ch == "-" and i > 0 and text[i - 1] not in "eE-":
            try:
                return float(text[:i]), float(text[i + 1:])
            except ValueError:
                continue
    raise AssertionError(f"not two printed ends: {text!r}")


def test_format_ci_prints_the_token_exactly_when_the_predicate_says_point():
    """Swept over ends, gaps from zero to 0.05 and every precision a table uses: the token appears
    exactly for a point, and every other cell prints two different numbers."""
    for base in (0.0, 1.0, 2.5, -3.2, 1e6, 123456.789):
        for gap in (0.0, 1e-12, 1e-9, 1e-7, 1e-3, 0.05):
            for nd in range(8):
                lo, hi = base, base + gap
                text = S.format_ci(lo, hi, nd)
                assert (text == TOKEN) is S.interval_is_degenerate(lo, hi), (lo, hi, nd, text)
                if text != TOKEN:
                    left, right = _printed_ends(text)
                    assert left != right, f"{(lo, hi, nd)} printed as a point: {text!r}"
    for lo, hi in ((0.1, np.float32(0.1)), (np.float32(2.3), 2.3), (np.float32(2.5), 2.5),
                   (np.float64(2**53), np.int64(2**53 + 1)), (np.int64(3), 3.0)):
        for nd in range(8):
            text = S.format_ci(lo, hi, nd)
            assert (text == TOKEN) is S.interval_is_degenerate(lo, hi), (lo, hi, nd, text)


def test_the_estimator_returns_the_zero_width_interval_it_computes():
    """The refusal is a publication rule, separate from the estimator's numeric contract: the
    straddling 40 % atom (25 % below it) returns the point it collapses to, for the caller to
    refuse. Measured at the default method and seed; a different seed could move this."""
    d = np.array([-1.0] * 25 + [0.0] * 40 + [1.0] * 35)
    assert np.mean(d == 0.0) == 0.40 and np.mean(d < 0.0) == 0.25
    m, lo, hi = S.paired_median_diff_ci(d, np.zeros_like(d))
    assert (m, lo, hi) == (0.0, 0.0, 0.0)
    assert S.interval_is_degenerate(lo, hi)


# --- BH-FDR ---------------------------------------------------------------
def test_bh_fdr_handworked():
    q = S.bh_fdr([0.001, 0.008, 0.039, 0.041, 0.9])
    assert q[0] == pytest.approx(0.005)
    assert q[4] == pytest.approx(0.9)


def test_bh_fdr_qvalues_never_below_pvalues():
    p = [0.001, 0.02, 0.03, 0.5]
    q = S.bh_fdr(p)
    assert all(qi >= pi - 1e-12 for pi, qi in zip(p, q))


def test_bh_fdr_preserves_input_order():
    p = [0.5, 0.001, 0.2]
    q = S.bh_fdr(p)
    assert q[1] < q[0] and q[1] < q[2]


def test_bh_fdr_passes_through_none():
    q = S.bh_fdr([0.01, None, 0.02])
    assert q[1] is None and q[0] is not None and q[2] is not None


def test_bh_fdr_all_none():
    assert S.bh_fdr([None, None]) == [None, None]


def test_bh_fdr_matches_statsmodels_if_available():
    p = [0.001, 0.008, 0.039, 0.041, 0.9, 0.5]
    q = S.bh_fdr(p)
    try:
        from statsmodels.stats.multitest import multipletests
        oq = multipletests(p, method="fdr_bh")[1]
        assert np.allclose(q, oq)
    except ImportError:
        pytest.skip("statsmodels not installed")


# --- Kruskal-Wallis -------------------------------------------------------
def test_kruskal_matches_scipy_and_eps_formula():
    from scipy.stats import kruskal
    rng = np.random.default_rng(2)
    g = [rng.normal(m, 2, n) for m, n in [(5, 30), (6, 35), (5.5, 25)]]
    kt = S.kruskal_test(*g)
    H, p = kruskal(*g)
    n = 90
    assert kt["H"] == pytest.approx(H)
    assert kt["p"] == pytest.approx(p)
    assert kt["eps_squared"] == pytest.approx(H / (n - 1))
    # eta is the bias-adjusted variant clamped at 0 for reporting; here H is large
    # so the clamp is inert and the value equals the raw formula.
    assert kt["eta_squared"] == pytest.approx(max(0.0, (H - 3 + 1) / (n - 3)))


def test_kruskal_eta_squared_clamped_nonnegative_near_null():
    # Three nearly-identical interleaved groups give a tiny H (here H=0.8 < k-1=2),
    # so the raw bias-adjusted eta^2 = (H-k+1)/(n-k) = -0.2 would be NEGATIVE.
    # For reporting it must be clamped to exactly 0.0 (analogous to adjusted R^2).
    g1, g2, g3 = [1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]
    kt = S.kruskal_test(g1, g2, g3)
    n, k = 9, 3
    H = kt["H"]
    assert (H - k + 1) / (n - k) < 0.0          # precondition: raw formula is negative
    assert kt["eta_squared"] == 0.0             # clamped, not negative
    # eps^2 must remain a bounded, non-negative effect size in [0, 1]
    assert 0.0 <= kt["eps_squared"] <= 1.0


def test_kruskal_eta_squared_nan_when_n_le_k():
    # n <= k must still yield NaN eta (not 0.0): there is no effect size to report.
    # Two singleton groups -> n=2, k=2, n<=k -> base dict with eta_squared None.
    kt = S.kruskal_test([1.0], [2.0])
    assert kt["eta_squared"] is None


def test_kruskal_too_few_groups_returns_none():
    assert S.kruskal_test([1, 2, 3])["p"] is None


# --- agreement statistics -------------------------------------------------
def test_lin_ccc_identical_is_one():
    v = [1.0, 2.0, 3.0, 4.0]
    assert S.lin_ccc(v, v) == pytest.approx(1.0)


def test_lin_ccc_scaled():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 100)
    y = x + rng.normal(0, 0.2, 100)
    c = S.lin_ccc(x, y)
    assert 0.8 < c < 1.0


def test_icc_a1_perfect():
    v = [1.0, 5.0, 9.0, 2.0]
    assert S.icc_a1(v, v) == pytest.approx(1.0)


def test_icc_a1_constant_offset_penalized():
    # absolute-agreement ICC must DROP when a constant bias is added
    v = np.array([1.0, 5.0, 9.0, 2.0, 7.0])
    assert S.icc_a1(v, v + 3.0) < S.icc_a1(v, v)


def test_icc_a1_matches_pingouin_oracle():
    # pin ICC(A,1) to an INDEPENDENT oracle (pingouin ICC2 == two-way random,
    # single, absolute agreement) so a future algebra regression is caught.
    a = [9.0, 6.0, 8.0, 7.0, 10.0, 6.0]
    b = [8.0, 5.0, 8.0, 6.0, 9.0, 7.0]
    assert S.icc_a1(a, b) == pytest.approx(0.8322147, abs=1e-6)   # pingouin ICC(A,1)


def test_lin_ccc_matches_rho_times_cb_oracle():
    # Lin's CCC == Pearson r * bias-correction Cb (independent decomposition).
    a = np.array([9.0, 6.0, 8.0, 7.0, 10.0, 6.0])
    b = np.array([8.0, 5.0, 8.0, 6.0, 9.0, 7.0])
    mx, my = a.mean(), b.mean()
    vx = ((a - mx) ** 2).mean()
    vy = ((b - my) ** 2).mean()
    cov = ((a - mx) * (b - my)).mean()
    r = cov / math.sqrt(vx * vy)
    sx, sy = math.sqrt(vx), math.sqrt(vy)
    cb = 2.0 / (sx / sy + sy / sx + (mx - my) ** 2 / (sx * sy))
    assert S.lin_ccc(a, b) == pytest.approx(r * cb, abs=1e-12)


def test_bland_altman_basic():
    x = np.array([10.0, 12.0, 14.0, 16.0])
    y = np.array([11.0, 11.0, 15.0, 15.0])
    ba = S.bland_altman(x, y)
    assert ba["bias"] == pytest.approx(np.mean(x - y))
    assert ba["sd_diff"] == pytest.approx(np.std(x - y, ddof=1))
    assert ba["loa_lo"] < ba["bias"] < ba["loa_hi"]


def test_bland_altman_nonparametric_and_diagnostics():
    rng = np.random.default_rng(4)
    a = rng.gamma(2, 3, 200)
    b = a + rng.normal(0, 1, 200)
    ba = S.bland_altman(a, b)
    # np LoA bracket the central 95% of differences
    d = a - b
    assert ba["loa_lo_np"] == pytest.approx(np.percentile(d, 2.5))
    assert ba["loa_hi_np"] == pytest.approx(np.percentile(d, 97.5))
    assert ba["diff_normal_p"] is not None
    assert -1.0 <= ba["prop_bias_r"] <= 1.0


# These tests bind the LoA rule (statlib.primary_loa) to code.

def test_primary_loa_selects_the_distribution_free_pair_when_differences_are_non_normal():
    rng = np.random.default_rng(11)
    a = rng.gamma(1.2, 4.0, 400)          # strongly skewed -> Shapiro rejects
    b = a + rng.gamma(1.2, 1.0, 400)
    ba = S.bland_altman(a, b)
    assert ba["diff_normal_p"] < 0.05, "fixture must actually be non-normal"

    sel = S.primary_loa(ba)
    assert sel["kind"] == "nonparametric"
    # the selected endpoints must BE the empirical pair, not merely be labelled so
    assert sel["lo"] == ba["loa_lo_np"]
    assert sel["hi"] == ba["loa_hi_np"]
    assert (sel["lo"], sel["hi"]) != (ba["loa_lo"], ba["loa_hi"])
    assert "Shapiro" in sel["reason"]


def test_primary_loa_keeps_the_parametric_pair_when_normality_holds():
    rng = np.random.default_rng(12)
    a = rng.normal(50.0, 5.0, 300)
    b = a + rng.normal(0.0, 1.0, 300)
    ba = S.bland_altman(a, b)
    assert ba["diff_normal_p"] >= 0.05, "fixture must actually look normal"

    sel = S.primary_loa(ba)
    assert sel["kind"] == "parametric"
    assert (sel["lo"], sel["hi"]) == (ba["loa_lo"], ba["loa_hi"])


def test_primary_loa_reports_rather_than_hides_an_untestable_case():
    """No silent fallback: an untested assumption must be visible, not defaulted away."""
    sel = S.primary_loa({"diff_normal_p": None, "loa_lo": -1.0, "loa_hi": 1.0,
                         "loa_lo_np": -0.5, "loa_hi_np": 0.5})
    assert sel["kind"] == "undetermined"
    assert "unvalidated" in sel["reason"]


def test_primary_loa_does_not_threshold_proportional_bias():
    """The rule is the rule as written. A prop-bias cutoff would be an uncalibrated constant, so
    the selection must depend on the normality p ALONE."""
    base = {"loa_lo": -2.0, "loa_hi": 2.0, "loa_lo_np": -1.0, "loa_hi_np": 1.0}
    for r in (-0.99, -0.41, 0.0, 0.41, 0.99):
        assert S.primary_loa({**base, "diff_normal_p": 1e-20,
                              "prop_bias_r": r})["kind"] == "nonparametric"
        assert S.primary_loa({**base, "diff_normal_p": 0.9,
                              "prop_bias_r": r})["kind"] == "parametric"


def test_primary_loa_honours_the_alpha_it_is_given():
    ba = {"diff_normal_p": 0.03, "loa_lo": -2.0, "loa_hi": 2.0,
          "loa_lo_np": -1.0, "loa_hi_np": 1.0}
    assert S.primary_loa(ba, alpha=0.05)["kind"] == "nonparametric"
    assert S.primary_loa(ba, alpha=0.01)["kind"] == "parametric"


# --- multi-sample BCa acceleration ----------------------------------------
# The multi-sample acceleration is Efron & Tibshirani eq (15.36): each sample centred on ITS
# OWN jackknife mean and scaled by ITS OWN n_j, not the single-sample formula (eq 14.15) applied
# to a concatenated jackknife. These tests are anchored to a
# published example with a pinned reference value rather than to the repository's own
# output, because a self-referential pin cannot tell a correct formula from a wrong one.

# Efron & Tibshirani's mouse data (An Introduction to the Bootstrap, ch. 2): survival in
# days, treatment n=7 vs control n=9. Deliberately unequal, which is what makes it
# discriminating -- at equal n the missing 1/n_j scaling partly cancels.
ET_MOUSE_TREATMENT = [16.0, 23.0, 38.0, 94.0, 99.0, 141.0, 197.0]
ET_MOUSE_CONTROL = [10.0, 27.0, 31.0, 40.0, 46.0, 50.0, 52.0, 104.0, 146.0]
# scipy/stats/tests/test_resampling.py::test_BCa_acceleration_against_reference, which
# derives it from eq (15.36) for this exact example.
ET_MOUSE_A_HAT = 0.011008228344026734


def _mouse_jackknives():
    """Per-sample jackknife of the difference of means, as (15.36) requires."""
    z = np.asarray(ET_MOUSE_TREATMENT, float)
    y = np.asarray(ET_MOUSE_CONTROL, float)

    def stat(a, b):
        return a.mean() - b.mean()

    jz = np.array([stat(np.delete(z, i), y) for i in range(z.size)])
    jy = np.array([stat(z, np.delete(y, i)) for i in range(y.size)])
    return [jz, jy]


def test_multisample_acceleration_matches_the_published_reference_value():
    """The citation-anchored oracle. The pooled-jackknife form returns +10.42 % of this."""
    a = S.bca_acceleration(_mouse_jackknives())
    assert a == pytest.approx(ET_MOUSE_A_HAT, rel=1e-12)


def test_multisample_acceleration_matches_scipy_on_the_same_example():
    """Second opinion from an independent implementation, not from our own arithmetic."""
    scipy_bootstrap = pytest.importorskip("scipy.stats").bootstrap
    res = scipy_bootstrap(
        (np.asarray(ET_MOUSE_TREATMENT, float), np.asarray(ET_MOUSE_CONTROL, float)),
        lambda a, b, axis=-1: a.mean(axis=axis) - b.mean(axis=axis),
        method="BCa", n_resamples=99, rng=np.random.default_rng(0))
    assert res is not None                       # the interval itself is stream-dependent
    assert S.bca_acceleration(_mouse_jackknives()) == pytest.approx(
        ET_MOUSE_A_HAT, rel=1e-12)


def test_pooling_the_jackknife_is_not_the_same_as_the_multisample_formula():
    """Pin the SIZE of the pooled form's error, so a regression cannot be waved off as rounding."""
    jz, jy = _mouse_jackknives()
    pooled = np.concatenate([jz, jy])
    d = pooled.mean() - pooled
    wrong = float((d ** 3).sum() / (6.0 * float((d ** 2).sum()) ** 1.5))
    assert wrong == pytest.approx(0.012155468206762, rel=1e-9)
    assert abs(wrong / ET_MOUSE_A_HAT - 1.0) > 0.10, (
        "the pooled form is supposed to be ~10 % off on this example; if it is not, the "
        "fixture does not show the pooled form's error")


def test_single_sample_acceleration_reduces_to_the_single_sample_formula():
    """J=1 of eq (15.36) IS eq (14.15): a single-sample caller gets the single-sample
    acceleration exactly, asserted over 200 random configurations."""
    rng = np.random.default_rng(3)
    for _ in range(200):
        n = int(rng.integers(3, 200))
        jack = rng.normal(size=n) * rng.uniform(0.1, 10.0)
        d = jack.mean() - jack
        denom = 6.0 * float((d ** 2).sum()) ** 1.5
        expected = float((d ** 3).sum() / denom) if denom else 0.0
        assert S.bca_acceleration([jack]) == pytest.approx(expected, abs=1e-15)
        assert S.bca_acceleration(jack) == pytest.approx(expected, abs=1e-15)


def test_kruskal_eps2_ci_keeps_its_groups_separate_in_the_acceleration():
    """The multi-sample caller, at an imbalance the study data reach.

    Groups of 6 and 30. Pooling the same jackknife (discarding the per-group structure) must
    give a materially different acceleration, otherwise this fixture cannot discriminate and the
    test is decorative.
    """
    rng = np.random.default_rng(5)
    groups = [rng.lognormal(0.0, 1.1, 6), rng.exponential(1.0, 30) + 0.5]
    e0, lo, hi = S.kruskal_eps2_ci(groups, n_boot=800, seed=11)
    assert math.isfinite(e0) and math.isfinite(lo) and math.isfinite(hi)
    assert 0.0 <= lo <= hi <= 1.0

    clean = [np.asarray(g, float) for g in groups]
    per_group = []
    for gi, g in enumerate(clean):
        vals = []
        for i in range(g.size):
            red = [np.delete(g, i) if j == gi else clean[j] for j in range(len(clean))]
            vals.append(S.kruskal_test(*red)["eps_squared"])
        per_group.append(np.asarray(vals, float))

    correct = S.bca_acceleration(per_group)
    pooled = S.bca_acceleration(np.concatenate(per_group))
    assert abs(correct - pooled) > 0.02, (
        f"fixture cannot discriminate: grouped {correct:.6f} vs pooled {pooled:.6f} are too close to "
        "distinguish the two formulas")


# --- clustered inference -----------------------------------------

def test_design_effect_recovers_a_known_icc():
    """Oracle: data built with a known variance split must report that ICC back.

    sigma_b^2 = 1, sigma_w^2 = 1 -> ICC = 0.5 by construction. Equal clusters of size 4, so
    the Kish mean is exactly 4 and DEFF = 1 + 3*0.5 = 2.5.
    """
    rng = np.random.default_rng(1)
    k, m = 800, 4
    b = rng.normal(0.0, 1.0, k)
    vals, labs = [], []
    for j in range(k):
        vals.extend(b[j] + rng.normal(0.0, 1.0, m))
        labs.extend([j] * m)

    icc, m_kish, deff = S.design_effect(vals, labs)
    assert icc == pytest.approx(0.5, abs=0.05)
    assert m_kish == pytest.approx(4.0, abs=1e-12)
    assert deff == pytest.approx(1.0 + (m_kish - 1) * icc, rel=1e-12)


def test_design_effect_uses_the_kish_mean_not_the_arithmetic_mean():
    """Unequal clusters inflate DEFF more than the plain average implies; that is the whole
    reason Kish's formula exists, and getting it wrong understates the correction."""
    sizes = [1] * 90 + [10] * 10          # arithmetic mean 1.9, Kish mean much larger
    rng = np.random.default_rng(2)
    vals, labs = [], []
    for j, n in enumerate(sizes):
        vals.extend(rng.normal(j, 1.0, n))
        labs.extend([j] * n)
    _, m_kish, _ = S.design_effect(vals, labs)
    n = np.array(sizes, float)
    assert m_kish == pytest.approx((n ** 2).sum() / n.sum(), rel=1e-12)
    assert m_kish > n.mean() + 1.0, "Kish mean must exceed the arithmetic mean here"


def test_design_effect_is_one_when_there_is_no_clustering():
    rng = np.random.default_rng(3)
    vals = rng.normal(0.0, 1.0, 500)
    icc, m_kish, deff = S.design_effect(vals, np.arange(500))   # every row its own cluster
    assert m_kish == pytest.approx(1.0)
    assert deff == pytest.approx(1.0)
    # The ICC is genuinely unestimable with no within-cluster replication, and saying so is
    # the honest answer; DEFF is still exactly 1, because m_kish = 1 makes it so for any ICC.
    assert math.isnan(icc)


def test_cluster_bca_ci_is_wider_than_the_iid_interval_under_real_clustering():
    """The property this estimator exists for: iid intervals are too narrow when rows
    repeat within a subject. Deliberately strong clustering so the comparison is not noise."""
    rng = np.random.default_rng(4)
    k, m = 120, 5
    b = rng.normal(0.0, 3.0, k)                 # large between-cluster variance
    vals, labs = [], []
    for j in range(k):
        vals.extend(b[j] + rng.normal(0.0, 1.0, m))
        labs.extend([j] * m)

    iid_lo, iid_hi = S.bca_ci(vals, np.mean, n_boot=800, seed=11)
    cl_lo, cl_hi = S.cluster_bca_ci(vals, labs, np.mean, n_boot=800, seed=11)
    icc, _, deff = S.design_effect(vals, labs)

    assert icc > 0.5 and deff > 2.0, "fixture is not strongly clustered"
    assert (cl_hi - cl_lo) > (iid_hi - iid_lo), (
        f"cluster interval {cl_hi - cl_lo:.4f} is not wider than iid {iid_hi - iid_lo:.4f}")
    assert (cl_hi - cl_lo) / (iid_hi - iid_lo) == pytest.approx(math.sqrt(deff), rel=0.35)


def test_cluster_bca_ci_reduces_to_the_iid_interval_when_every_cluster_is_a_singleton():
    """No clustering must cost nothing. Same seed, same stream, same answer."""
    rng = np.random.default_rng(5)
    vals = rng.lognormal(0.0, 0.8, 200)
    iid = S.bca_ci(vals, np.median, n_boot=500, seed=7)
    clustered = S.cluster_bca_ci(vals, np.arange(vals.size), np.median, n_boot=500, seed=7)
    assert clustered == pytest.approx(iid, abs=1e-12)


def test_cluster_bca_ci_resamples_clusters_not_rows():
    """An implementation that resampled rows would pass the width test on some fixtures. This one
    it cannot: with two perfectly homogeneous clusters, every cluster-level resample yields
    one of three possible datasets, so the interval endpoints must be drawn from that tiny
    set -- a row-level resample would produce values between them."""
    vals = [0.0] * 25 + [10.0] * 25
    labs = [0] * 25 + [1] * 25
    lo, hi = S.cluster_bca_ci(vals, labs, np.mean, n_boot=400, seed=3)
    assert set(np.round([lo, hi], 9)) <= {0.0, 5.0, 10.0}, (
        f"({lo}, {hi}) contains a value only row-level resampling could produce")


def test_cluster_bca_ci_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="differ in length"):
        S.cluster_bca_ci([1.0, 2.0, 3.0], [0, 1], np.mean)


# --- bootstrap CIs (percentile + BCa) -------------------------------------
def test_bootstrap_ci_deterministic():
    v = list(range(50))
    assert S.median_ci(v, seed=1) == S.median_ci(v, seed=1)


def test_bootstrap_ci_constant_array_is_degenerate():
    assert S.median_ci([4.0] * 20) == (4.0, 4.0)


def test_bootstrap_ci_empty_is_nan():
    lo, hi = S.median_ci([])
    assert math.isnan(lo) and math.isnan(hi)


def test_bca_brackets_point_estimate_and_is_ordered():
    rng = np.random.default_rng(5)
    v = rng.gamma(2, 3, 300)
    lo, hi = S.median_ci_bca(v)
    assert lo <= np.median(v) <= hi and lo < hi


def test_bca_degenerate_inputs():
    assert S.bca_ci([7.0] * 10) == (7.0, 7.0)
    lo, hi = S.bca_ci([])
    assert math.isnan(lo) and math.isnan(hi)


def test_median_ci_auto_dispatch():
    v = list(np.random.default_rng(6).gamma(2, 2, 100))
    assert S.median_ci_auto(v, "percentile") == S.median_ci(v)
    assert S.median_ci_auto(v, "bca") == S.median_ci_bca(v)


# --- wall-contact apex (radius-aware) -------------------------------------
def test_wall_contact_apex_solves_sep_equals_R():
    # sep = 20 - z, sum-of-radii constant 6 -> wall contact (sep=6) at z=14
    z = np.linspace(0, 10, 11)
    sep = 20.0 - z
    sumr = np.full_like(z, 6.0)
    fit = S.wall_contact_apex(z, sep, sumr)
    assert fit["slope"] == pytest.approx(-1.0)
    assert fit["median_r"] == pytest.approx(6.0)
    assert fit["z_bif"] == pytest.approx(14.0)


def test_wall_contact_apex_refuses_a_nonpositive_contact_level():
    """With sep = 14 - 0.8*z (sep=0 at z=17.5) and R = -3 an unguarded solve gives
    z = (-3-14)/(-0.8) = 21.25, CRANIAL of 17.5, with a finite tight CI, which could pass the flag
    gate. The guard must refuse: unstable=True, z_bif=NaN, no CI,
    nonpositive_R=True."""
    z = np.arange(12, dtype=float)
    sep = 14.0 - 0.8 * z                # exact line: slope -0.8, perfectly estimable
    bad = S.wall_contact_apex(z, sep, np.full(12, -3.0), seed=5)
    assert bad["nonpositive_R"] is True
    assert bad["unstable"] is True
    assert math.isnan(bad["z_bif"])
    assert math.isnan(bad["ci_lo"]) and math.isnan(bad["ci_hi"])
    # R = 0 is the boundary: point-lumens touching AT centroid coincidence is still
    # not a wall contact; the guard is <= 0, not < 0
    zero = S.wall_contact_apex(z, sep, np.zeros(12), seed=5)
    assert zero["nonpositive_R"] is True and zero["unstable"] is True
    # the DISCRIMINATING half: identical geometry with a physical R solves normally,
    # caudal of centroid coincidence
    good = S.wall_contact_apex(z, sep, np.full(12, 3.0), seed=5)
    assert good["nonpositive_R"] is False
    assert good["unstable"] is False
    assert good["z_bif"] == pytest.approx((3.0 - 14.0) / -0.8)   # 13.75 < 17.5


def test_the_apex_slope_screen_is_two_sided_and_stricter_than_nominal():
    """Combined, the effective one-sided convergence level is alpha/2 = 2.5%, and this fixture pins the
    consequence at the n=6 floor with EXACT OLS algebra: the residual pattern p is orthogonal to [1,
    z], so the fitted slope is exactly +/-1, the residual SD is exactly c, and t = sqrt(17.5)/c."""
    z = np.arange(6, dtype=float)                    # n=6 -> df=4, the BIF floor
    p = np.array([1.0, -1.0, 0.0, 0.0, -1.0, 1.0])   # sum p = 0, sum p*z = 0
    assert abs(p.sum()) < 1e-12 and abs((p * z).sum()) < 1e-12
    sxx = float(((z - z.mean()) ** 2).sum())         # 17.5
    t_one = S.t_ppf(0.95, 4)                         # one-sided 5% bar
    t_two = S.t_crit(0.05, 4)                        # two-sided 5% bar (the gate's)
    assert t_one == pytest.approx(2.1318, abs=1e-3)
    assert t_two == pytest.approx(2.7764, abs=1e-3)
    rr = np.full(6, 3.0)

    def fit(slope_sign, t_target):
        c = math.sqrt(sxx) / t_target                # resid SD = c, SE = c/sqrt(sxx)
        return S.wall_contact_apex(z, slope_sign * z + 14.0 + c * p, rr, seed=7)

    # |t| = 2.4 sits in the one-sided band (2.13 < t < 2.78): one-sided p is between
    # 2.5% and 5%, and the case is EXCLUDED -> the gate is stricter than a nominal
    # one-sided 5% test, exactly the alpha/2 = 2.5% effective level
    inband = fit(-1.0, 2.4)
    assert inband["unstable"] is True
    assert math.isnan(inband["z_bif"])
    # |t| = 3.0 clears the two-sided bar: admitted
    strong = fit(-1.0, 3.0)
    assert strong["unstable"] is False and math.isfinite(strong["z_bif"])
    # same magnitude, POSITIVE slope: the screen itself is sign-blind; the sign is
    # rejected downstream by decide_flag, whose
    # positive-slope test lives in test_bifurcation.py
    diverging = fit(+1.0, 3.0)
    assert diverging["unstable"] is False
    assert diverging["slope"] == pytest.approx(1.0)


def test_boot_indices_are_circular_contiguous_blocks():
    """The i.i.d. fallbacks (block None/<=1/>=n) must still return n in-range indices."""
    n, block = 10, 3
    wrapped = False
    for seed in range(8):                       # several draws so wrapping occurs
        idx = S._boot_indices(n, block, np.random.default_rng(seed))
        assert idx.shape == (n,)
        assert idx.min() >= 0 and idx.max() < n
        for b in range(n // block + 1):
            chunk = idx[b * block:(b + 1) * block]
            if chunk.size >= 2:
                assert ((np.diff(chunk) % n) == 1).all(), (seed, b, chunk)
                if int(chunk.max()) - int(chunk.min()) > block:   # e.g. [9, 0, 1]
                    wrapped = True
    assert wrapped                              # the circular wrap was exercised
    for degenerate in (None, 0, 1, n, n + 5):
        idx = S._boot_indices(n, degenerate, np.random.default_rng(0))
        assert idx.shape == (n,) and idx.min() >= 0 and idx.max() < n


def test_jackknife_delete_block_is_circular_and_falls_back():
    n, block = 8, 3
    reps = list(S._jackknife_delete_block(n, block))
    assert len(reps) == n
    for s, kept in enumerate(reps):
        removed = set(range(n)) - set(kept.tolist())
        assert removed == {(s + j) % n for j in range(block)}   # incl. wrap at s=6,7
        assert kept.size == n - block
    for bad in (None, 0, 1, n, n + 3):          # delete-one fallbacks
        reps1 = list(S._jackknife_delete_block(n, bad))
        assert len(reps1) == n
        assert all(r.size == n - 1 for r in reps1)
        assert {next(iter(set(range(n)) - set(r.tolist()))) for r in reps1} == set(range(n))
    reps2 = list(S._jackknife_delete_block(4, 3))   # n - block = 1 < 2 -> delete-one
    assert all(r.size == 3 for r in reps2)


def test_apex_ci_coverage_smoke_bound():
    """The floor catches a resampler or BCa regression that collapses coverage (to ~0-50%) while
    sitting far enough below nominal coverage not to flake. Truth: sep = 14 - 0.8 z + N(0, 0.5), R
    = 3, apex at z = 13.75; n = 12 slices, block = 3, 40 seeded draws."""
    rng = np.random.default_rng(20260809)
    z = np.arange(12, dtype=float)
    truth = (3.0 - 14.0) / -0.8                      # 13.75
    hits = usable = 0
    for _ in range(40):
        sep = 14.0 - 0.8 * z + rng.normal(0.0, 0.5, 12)
        fit = S.wall_contact_apex(z, sep, np.full(12, 3.0), n_boot=199,
                                  seed=int(rng.integers(0, 2 ** 31 - 1)), block=3)
        if fit["unstable"] or not (math.isfinite(fit["ci_lo"])
                                   and math.isfinite(fit["ci_hi"])):
            continue
        usable += 1
        hits += 1 if fit["ci_lo"] <= truth <= fit["ci_hi"] else 0
    assert usable >= 30                              # the design is comfortably estimable
    assert hits / usable >= 0.6                      # gross-regression floor (docstring)


def test_bca_midp_on_discrete_sample_brackets_median():
    # discrete sample: many bootstrap medians tie theta; mid-p must keep the CI
    # sane (bracketing the median, not shifted off it)
    v = [1, 2, 2, 3, 3, 3, 4, 4, 5]
    lo, hi = S.median_ci_bca(v, seed=3)
    assert lo <= np.median(v) <= hi


# --- curvature / linearity diagnostic -------------------------------------
def test_quadratic_significance_flags_curve():
    x = np.linspace(0, 10, 30)
    y = 2.0 * x ** 2 - 3.0 * x + 1.0          # strongly quadratic
    r = S.quadratic_significance(x, y)
    assert r["significant"] is True
    assert r["quad_coef"] == pytest.approx(2.0, abs=0.1)


def test_quadratic_significance_passes_line():
    # an exact line has zero residual -> quad term is not significant (deterministic)
    x = np.linspace(0, 10, 30)
    assert S.quadratic_significance(x, -1.5 * x + 4.0)["significant"] is False
    # a linear+noise signal is flagged only at ~the alpha false-positive rate,
    # NOT always (a t-test on a null quadratic term rejects ~5% by construction)
    rng = np.random.default_rng(0)
    hits = sum(S.quadratic_significance(x, -1.5 * x + 4.0 + rng.normal(0, 0.5, 30))["significant"]
               for _ in range(40))
    assert hits <= 6


def test_quadratic_significance_too_few_points():
    r = S.quadratic_significance([0, 1, 2], [0, 1, 4])
    assert r["significant"] is None and r["p"] is None
    assert r["status"] == "insufficient_data"


def test_quadratic_significance_fit_failure_is_not_reported_as_no_curvature(monkeypatch):
    monkeypatch.setattr(
        np, "polyfit",
        lambda *args, **kwargs: (_ for _ in ()).throw(np.linalg.LinAlgError("singular")))
    r = S.quadratic_significance(np.arange(6.0), np.arange(6.0) ** 2)
    assert r["significant"] is None
    assert r["status"] == "fit_failed"
    assert r["error_type"] == "LinAlgError"


def test_quadratic_significance_excludes_infinite_observations():
    r = S.quadratic_significance(
        [0, 1, 2, 3, 4, 5, np.inf],
        [0, 1, 4, 9, 16, 25, 99],
    )

    assert r["n"] == 6
    assert r["status"] == "ok"
    assert r["significant"] is True


# --- method / reference agreement -----------------------------------------
def test_method_agreement_bundle():
    a = np.array([10.0, 20.0, 30.0, 40.0])
    b = a + 1.0                                   # constant +1 offset
    out = S.method_agreement(a, b, tol=2.0)
    assert out["bias"] == pytest.approx(-1.0)     # a - b
    assert out["within_tol_frac"] == 1.0
    assert out["ccc"] > 0.9


def test_method_agreement_within_tol_partial():
    a = np.array([0.0, 0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 10.0, 10.0])
    out = S.method_agreement(a, b, tol=5.0)
    assert out["within_tol_frac"] == pytest.approx(0.5)


# --- Student-t distribution (pure stdlib; no scipy required) ---------------
def test_t_sf_matches_scipy_across_df():
    from scipy.stats import t as tdist
    for df in (1, 2, 3, 4, 5, 8, 12, 30, 100, 500):
        for tv in (0.3, 1.0, 2.0, 2.776, 5.0):
            assert S.t_sf(tv, df) == pytest.approx(float(tdist.sf(tv, df)), abs=1e-9)
            assert S.t_sf(-tv, df) == pytest.approx(float(tdist.sf(-tv, df)), abs=1e-9)


def test_t_ppf_matches_scipy_across_df():
    from scipy.stats import t as tdist
    for df in (1, 2, 3, 4, 5, 8, 12, 30, 100, 500):
        for p in (0.9, 0.95, 0.975, 0.995):
            assert S.t_ppf(p, df) == pytest.approx(float(tdist.ppf(p, df)), abs=1e-7)


def test_t_crit_is_student_t_not_normal_at_small_df():
    # at small df the two-sided critical value must be the
    # heavier-tailed t value, NOT the normal quantile z=1.95996
    assert S.t_crit(0.05, 4) == pytest.approx(2.7764451, abs=1e-4)
    assert S.t_crit(0.05, 4) > S.z_crit(0.05) + 0.5      # clearly heavier than z
    # and it must equal the pure-stdlib t quantile (no scipy fallback to z)
    assert S.t_crit(0.05, 4) == pytest.approx(S.t_ppf(0.975, 4), abs=1e-10)


def test_t_crit_approaches_z_at_large_df():
    assert S.t_crit(0.05, 100000) == pytest.approx(S.z_crit(0.05), abs=1e-3)


def test_t_sf_heavier_tailed_than_normal_at_low_df():
    # the normal tail understates the p-value at low df, so a curvature test must use
    # the t tail
    from statistics import NormalDist
    normal_two_sided = 2.0 * (1.0 - NormalDist().cdf(2.5))
    t_two_sided = 2.0 * S.t_sf(2.5, 4)
    assert t_two_sided > 2.0 * normal_two_sided    # t tail is much fatter at df=4


# --- quadratic_significance must use the t tail, not the normal tail ---
def test_quadratic_significance_pvalue_is_t_based():
    rng = np.random.default_rng(21)
    x = np.linspace(0, 6, 9)                    # n=9 -> df = n-3 = 6
    y = 0.25 * x ** 2 - 1.0 * x + 2.0 + rng.normal(0, 0.4, 9)
    r = S.quadratic_significance(x, y)
    assert r["p"] == pytest.approx(2.0 * S.t_sf(abs(r["t"]), r["n"] - 3), abs=1e-9)


# --- scipy-absent fallback regression (these paths never run with scipy installed) -
def test_scipy_absent_fallbacks_match_scipy_enabled_values():
    """Characterization/regression test for the pure-Python paths that are never
    exercised when scipy is installed. statlib imports scipy LAZILY inside the few
    functions that need it (shapiro_p, kruskal_test), so blocking scipy in sys.modules
    forces `from scipy.stats import ...` to raise ImportError at call time — directly
    activating the scipy-absent branches without any module reload. The functions
    checked here (t_sf/t_ppf/t_crit, percentile + BCa bootstrap, wilson_ci, bh_fdr,
    describe) are pure stdlib and must return finite results IDENTICAL to the
    scipy-enabled run within ~1e-6. Reference is computed first (scipy present), then
    the fallback under the patch."""
    rng = np.random.default_rng(123)
    v = rng.gamma(2.0, 3.0, 80)                    # skewed sample for BCa/describe
    pvals = [0.001, 0.008, 0.039, 0.041, 0.9, 0.5]

    # --- reference values WITH scipy importable ---
    ref_t_sf = [S.t_sf(tv, df) for tv in (0.3, 1.0, 2.776, 5.0) for df in (1, 4, 30)]
    ref_t_ppf = [S.t_ppf(p, df) for p in (0.9, 0.975, 0.995) for df in (1, 4, 30)]
    ref_t_crit = [S.t_crit(0.05, df) for df in (1, 4, 30, 100)]
    ref_pct = S.bootstrap_ci(v, np.median, seed=7)
    ref_bca = S.bca_ci(v, np.median, seed=7)
    ref_wilson = S.wilson_ci(8, 10)
    ref_fdr = S.bh_fdr(pvals)
    ref_desc = S.describe(v, seed=7)

    assert all(math.isfinite(x) for x in ref_t_sf + ref_t_ppf + ref_t_crit)
    assert all(math.isfinite(x) for x in ref_pct) and all(math.isfinite(x) for x in ref_bca)

    # --- same calls with scipy made un-importable -> exercises the fallback paths ---
    with mock.patch.dict(sys.modules, {"scipy": None, "scipy.stats": None}):
        # sanity: scipy really is blocked now
        with pytest.raises(ImportError):
            __import__("scipy.stats")
        # scipy-dependent helpers must degrade gracefully (documents the contract)
        assert S.shapiro_p(v) is None
        assert S.kruskal_test([1, 2, 3, 4], [5, 6, 7, 8], [2, 3, 4, 5])["p"] is None

        fb_t_sf = [S.t_sf(tv, df) for tv in (0.3, 1.0, 2.776, 5.0) for df in (1, 4, 30)]
        fb_t_ppf = [S.t_ppf(p, df) for p in (0.9, 0.975, 0.995) for df in (1, 4, 30)]
        fb_t_crit = [S.t_crit(0.05, df) for df in (1, 4, 30, 100)]
        fb_pct = S.bootstrap_ci(v, np.median, seed=7)
        fb_bca = S.bca_ci(v, np.median, seed=7)
        fb_wilson = S.wilson_ci(8, 10)
        fb_fdr = S.bh_fdr(pvals)
        fb_desc = S.describe(v, seed=7)

    # finite + matches scipy-enabled reference within tolerance
    for a, b in zip(fb_t_sf, ref_t_sf):
        assert math.isfinite(a) and a == pytest.approx(b, abs=1e-6)
    for a, b in zip(fb_t_ppf, ref_t_ppf):
        assert math.isfinite(a) and a == pytest.approx(b, abs=1e-6)
    for a, b in zip(fb_t_crit, ref_t_crit):
        assert math.isfinite(a) and a == pytest.approx(b, abs=1e-6)
    assert fb_pct == pytest.approx(ref_pct, abs=1e-6)
    assert fb_bca == pytest.approx(ref_bca, abs=1e-6)
    assert fb_wilson == pytest.approx(ref_wilson, abs=1e-6)
    assert fb_fdr == pytest.approx(ref_fdr, abs=1e-6)
    # describe: every numeric field matches except shapiro_p, which is legitimately
    # None without scipy (it is the one describe() field that depends on scipy).
    assert set(fb_desc) == set(ref_desc)
    assert fb_desc["shapiro_p"] is None and ref_desc["shapiro_p"] is not None
    for key, val in ref_desc.items():
        if key in ("shapiro_p", "median_ci_method"):
            continue
        assert math.isfinite(fb_desc[key]) and fb_desc[key] == pytest.approx(val, abs=1e-6)

    # restore module state explicitly so any import-time caching cannot leak into
    # later tests (the patch already auto-restores sys.modules; reload is belt-and-braces).
    importlib.reload(S)
    assert S.shapiro_p(v) is not None              # scipy is back -> non-None p-value


# --- BCa pinned to scipy's BCa (the bracket+order tests above
#     would ALSO pass for a plain percentile interval; this catches a dropped z0 bias
#     correction or a sign-flipped acceleration, which a percentile CI cannot) ------
def test_bca_ci_matches_scipy_bca_oracle():
    from scipy.stats import bootstrap as scipy_bootstrap
    rng = np.random.default_rng(20240611)
    samp = rng.gamma(2.0, 3.0, 300)                # skewed -> z0 and accel both non-zero
    # independent RNGs (ours seeded; scipy random_state=1) -> not bit-identical, but a
    # correct BCa agrees closely while a broken z0/accel diverges far beyond tol.
    for stat, tol in ((np.mean, 0.15), (np.median, 0.40)):
        mine = S.bca_ci(samp, stat, alpha=0.05, n_boot=6000, seed=20240611)
        sp = scipy_bootstrap((samp,), stat, method="BCa", n_resamples=6000,
                             confidence_level=0.95, random_state=1)
        assert mine[0] == pytest.approx(float(sp.confidence_interval.low), abs=tol)
        assert mine[1] == pytest.approx(float(sp.confidence_interval.high), abs=tol)

    # On a small, heavily-skewed sample the bias-correction/acceleration shift the interval well
    # beyond the percentile one, so require BCa to MATCH scipy's BCa oracle within a tolerance that
    # the plain percentile interval FAILS. A regression that dropped z0 or flipped the acceleration
    # sign (collapsing BCa toward percentile) then fails here even though it passes the large-n
    # tolerance checks above.
    small = np.random.default_rng(7).gamma(1.2, 2.0, 28)     # heavy skew, small n
    bca = S.bca_ci(small, np.mean, alpha=0.05, n_boot=8000, seed=20240611)
    pct = S.bootstrap_ci(small, np.mean, alpha=0.05, n_boot=8000, seed=20240611)
    sp = scipy_bootstrap((small,), np.mean, method="BCa", n_resamples=8000,
                         confidence_level=0.95, random_state=1)
    lo_o, hi_o = float(sp.confidence_interval.low), float(sp.confidence_interval.high)
    disc = 0.15
    assert abs(bca[0] - lo_o) <= disc and abs(bca[1] - hi_o) <= disc     # BCa matches oracle
    assert abs(pct[0] - lo_o) > disc or abs(pct[1] - hi_o) > disc        # percentile does NOT
    assert abs(bca[0] - pct[0]) > 0.02 or abs(bca[1] - pct[1]) > 0.02    # BCa != percentile


# --- kruskal_eps2_ci: the bootstrap CI for the KW effect size (the eps^2 POINT
#     estimate is pinned via kruskal_test elsewhere) ---------------------------------
def test_kruskal_eps2_ci_point_matches_scipy_and_brackets():
    from scipy.stats import kruskal
    rng = np.random.default_rng(2)
    groups = [rng.normal(m, 2, n) for m, n in [(5, 30), (6, 35), (5.5, 25)]]
    eps2, lo, hi = S.kruskal_eps2_ci(groups, seed=20240611)
    H, _ = kruskal(*groups)
    n = sum(len(g) for g in groups)
    assert eps2 == pytest.approx(H / (n - 1))      # point estimate pinned to scipy's H
    assert lo <= eps2 <= hi and lo < hi            # stratified bootstrap CI brackets it
    assert 0.0 <= lo and hi <= 1.0                 # eps^2 stays a bounded effect size


def test_kruskal_eps2_ci_deterministic_and_degenerate():
    rng = np.random.default_rng(3)
    g = [rng.normal(0.0, 1.0, 20), rng.normal(0.5, 1.0, 20)]
    assert S.kruskal_eps2_ci(g, seed=7) == S.kruskal_eps2_ci(g, seed=7)   # seeded -> reproducible
    out = S.kruskal_eps2_ci([[1.0, 2.0, 3.0]])     # k < 2 groups -> no effect size
    assert all(math.isnan(x) for x in out)


def test_kruskal_eps2_ci_covers_zero_exactly_when_the_test_does_not_reject():
    """The lower end is 0 exactly when the Kruskal-Wallis test of the same groups does not reject
    at the interval's own alpha, so on null data the coverage of 0 is the test's non-rejection rate,
    replicate by replicate."""
    rng = np.random.default_rng(9020)
    g = [rng.lognormal(0, 1, 100) for _ in range(3)]
    e0, lo, hi = S.kruskal_eps2_ci(g)
    assert S.kruskal_test(*g)["p"] > 0.99 and e0 == pytest.approx(0.000041, abs=5e-7)
    assert lo == 0.0 and lo <= e0 <= hi
    r = np.random.default_rng(1)
    pvals = []
    for _ in range(30):                                   # continuous null data: no replicate H is 0
        gs = [r.lognormal(0, 1, 20) for _ in range(3)]
        e0, lo, hi = S.kruskal_eps2_ci(gs, n_boot=200, seed=int(r.integers(1 << 30)))
        p = S.kruskal_test(*gs)["p"]
        pvals.append(p)
        assert (lo == 0.0) == (p >= 0.05), (p, lo)
        assert lo <= e0 <= hi
    assert min(pvals) < 0.05 <= max(pvals)                # both branches were exercised
    # the rule follows the interval's own alpha, not a fixed 0.05
    mid = [gs for gs in ([r.lognormal(0, 1, 20) + d for d in (0.0, 0.15, 0.3)] for _ in range(200))
           if 0.06 < S.kruskal_test(*gs)["p"] < 0.4][0]
    assert S.kruskal_eps2_ci(mid, alpha=0.05, n_boot=200, seed=3)[1] == 0.0
    assert S.kruskal_eps2_ci(mid, alpha=0.5, n_boot=200, seed=3)[1] > 0.0


def test_kruskal_eps2_ci_contains_its_estimate_whatever_the_bca_ends(monkeypatch):
    """The interval is widened to contain its estimate. Driven here through a substituted
    `_bca_endpoints` returning ends on either side, on data the test rejects, so the lower-end rule
    cannot mask the guard."""
    rng = np.random.default_rng(2)
    groups = [rng.normal(m, 2, n) for m, n in [(5, 30), (6, 35), (5.5, 25)]]
    kt = S.kruskal_test(*groups)
    e0 = kt["eps_squared"]
    assert kt["p"] < 0.05
    for ends in ((e0 + 0.05, e0 + 0.10), (e0 - 0.10, e0 - 0.05)):
        monkeypatch.setattr(S, "_bca_endpoints", lambda *_a, ends=ends: ends)
        _, lo, hi = S.kruskal_eps2_ci(groups, n_boot=50, seed=1)
        assert (lo, hi) == (min(ends[0], e0), max(ends[1], e0)) and lo <= e0 <= hi


def test_kruskal_eps2_ci_the_percentile_fallback_obeys_both_rules(monkeypatch):
    """On this null dataset the percentile interval sits wholly above the estimate (replicates
    average the null bias, 0.0067, against an estimate of 0.000041), so both rules show: the lower
    end is 0 because the test does not reject, and the estimate is inside."""
    rng = np.random.default_rng(9020)
    g = [rng.lognormal(0, 1, 100) for _ in range(3)]
    monkeypatch.setattr(S, "_bca_endpoints", lambda *_a: None)
    e0, lo, hi = S.kruskal_eps2_ci(g, n_boot=400, seed=5)
    assert lo == 0.0 and lo <= e0 <= hi and hi > 0.005           # the percentile upper end, kept


def test_kruskal_eps2_ci_does_not_reject_at_p_equal_to_alpha():
    """Driven by setting alpha to the data's own p, on data where the BCa lower end is positive just
    past it (on continuous p no other test reaches p == alpha, so only this one separates `>=`
    from `>`)."""
    rng = np.random.default_rng(2)
    groups = [rng.normal(m, 2, n) for m, n in [(5, 30), (6, 35), (5.5, 25)]]
    p = S.kruskal_test(*groups)["p"]
    assert S.kruskal_eps2_ci(groups, alpha=p, n_boot=400, seed=1)[1] == 0.0
    assert S.kruskal_eps2_ci(groups, alpha=p * 1.001, n_boot=400, seed=1)[1] > 0.0


def test_kruskal_eps2_ci_leaves_an_undefined_interval_undefined(monkeypatch):
    """Ends that are not finite return as they came, so a non-rejecting test cannot turn an undefined
    interval into the half-interval (0, nan)."""
    rng = np.random.default_rng(9020)
    g = [rng.lognormal(0, 1, 100) for _ in range(3)]
    monkeypatch.setattr(S, "_bca_endpoints", lambda *_a: (float("nan"), float("nan")))
    e0, lo, hi = S.kruskal_eps2_ci(g, n_boot=100, seed=5)
    assert math.isfinite(e0) and math.isnan(lo) and math.isnan(hi)


def test_kruskal_eps2_ci_keeps_the_bca_upper_end_of_a_non_rejecting_interval(monkeypatch):
    """The upper end is the BCa upper end whenever that lies above the estimate; the real
    `_bca_endpoints` is wrapped, not replaced, so its own answer is the reference."""
    r = np.random.default_rng(1)
    mid = next(gs for gs in ([r.lognormal(0, 1, 20) + d for d in (0.0, 0.15, 0.3)] for _ in range(200))
               if 0.06 < S.kruskal_test(*gs)["p"] < 0.4)
    seen, real = [], S._bca_endpoints
    monkeypatch.setattr(S, "_bca_endpoints", lambda *a: seen.append(real(*a)) or seen[-1])
    e0, lo, hi = S.kruskal_eps2_ci(mid, n_boot=400, seed=3)
    assert lo == 0.0 and len(seen) == 1 and seen[0][1] > e0 and hi == seen[0][1]


def _eps2_of(*gs):
    """Test-local eps^2 = tie-corrected H / (n-1), straight from scipy."""
    from scipy.stats import kruskal
    H, _ = kruskal(*gs)
    return float(H) / (sum(len(g) for g in gs) - 1)


def test_kruskal_eps2_ci_is_bca_shifted_below_its_own_percentile_interval():
    """The plain percentile interval inherits the upward shift of the eps^2 bootstrap; the BCa z0
    term must remove it. This test reproduces the function's exact bootstrap stream and asserts the
    returned endpoints sit strictly BELOW the plain 2.5/97.5 percentiles of that same stream, so a
    percentile interval fails it."""
    rng = np.random.default_rng(2)
    groups = [rng.normal(m, 2, n) for m, n in [(5, 30), (6, 35), (5.5, 25)]]
    e0, lo, hi = S.kruskal_eps2_ci(groups, seed=20240611)

    # the mis-centring: the point sits near the 43rd percentile of its own bootstrap;
    # the boot mean exceeds the point by ~+0.019
    brng = np.random.default_rng(20240611)
    boot = np.array([_eps2_of(*[g[brng.integers(0, g.size, g.size)] for g in groups])
                     for _ in range(2000)])
    assert boot.mean() > e0 + 0.01                      # upward-shifted bootstrap
    p_lo, p_hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
    assert lo < p_lo - 0.005                            # BCa lower sits below percentile
    assert hi < p_hi - 0.015                            # BCa upper sits below percentile
    assert lo <= e0 <= hi                               # still brackets the point

    # seeded reproducibility pin: correctness from the oracles, chain from the pin. The pinned
    # values were derived independently of this module, by rebuilding z0 and the acceleration from
    # Efron & Tibshirani eq (15.36) and running them against the same bootstrap stream
    # (acceleration a = -0.018312861629).
    assert S.kruskal_eps2_ci(groups, seed=20240612)[1:] != (lo, hi)  # seed-sensitive
    assert e0 == pytest.approx(0.1198523353, abs=1e-9)   # point estimate
    assert lo == pytest.approx(0.0173223223, abs=1e-9)
    assert hi == pytest.approx(0.2573719487, abs=1e-9)


def test_kruskal_eps2_ci_matches_scipy_bootstrap_bca_oracle():
    """Endpoints must agree within bootstrap stream noise (measured: |d_lo| ~= 0.002, |d_hi| ~= 0.015
    at n_boot = 2000; tolerance 0.025 covers the seed spread). A percentile implementation fails:
    its endpoints sit (+0.013, +0.031) above the BCa pair, and the hi gap alone is ~2x the
    tolerance."""
    from scipy.stats import bootstrap
    rng = np.random.default_rng(2)
    groups = [rng.normal(m, 2, n) for m, n in [(5, 30), (6, 35), (5.5, 25)]]

    def eps2_vec(*gs, axis=-1):
        gs = [np.asarray(g) for g in gs]
        if gs[0].ndim == 1:
            return _eps2_of(*gs)
        return np.array([_eps2_of(*[g[i] for g in gs]) for i in range(gs[0].shape[0])])

    sp = bootstrap(groups, eps2_vec, method="BCa", n_resamples=2000,
                   vectorized=True, random_state=np.random.default_rng(7))
    _, lo, hi = S.kruskal_eps2_ci(groups, seed=20240611)
    assert lo == pytest.approx(sp.confidence_interval.low, abs=0.025)
    assert hi == pytest.approx(sp.confidence_interval.high, abs=0.025)


def test_eps_squared_is_clamped_to_the_unit_interval():
    """The clamp must return exactly 1.0, and the CI on such data must stay in [0,1] deterministically
    (every replicate resamples constant groups, so the whole interval collapses to 1.0)."""
    for reps in (3, 10):
        kt = S.kruskal_test([1.0] * reps, [2.0] * reps, [3.0] * reps)
        assert kt["H"] > (3 * reps - 1) * kt["eps_squared"] - 1e-12  # H genuinely at/over n-1
        assert kt["eps_squared"] == 1.0                              # clamped exactly
    e0, lo, hi = S.kruskal_eps2_ci([[1.0] * 3, [2.0] * 3, [3.0] * 3], seed=1)
    assert (e0, lo, hi) == (1.0, 1.0, 1.0)


def test_paired_statistics_reject_misaligned_inputs_instead_of_truncating():
    """Unequal lengths must raise: head-truncating to the shorter series would turn a misalignment
    into PERFECT agreement (icc_a1([1,2,3,4,5],[1,2,3]) and lin_ccc([1,2,3,4,100],[1,2,3,4]) would
    both be 1.0)."""
    for fn in (S.lin_ccc, S.icc_a1):
        with pytest.raises(ValueError, match="differ in length"):
            fn([1.0, 2.0, 3.0, 4.0, 100.0], [1.0, 2.0, 3.0, 4.0])


def test_aligned_paired_inputs_are_accepted():
    assert S.lin_ccc([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.0)


def test_bca_midp_z0_is_pinned_against_an_analytic_oracle_and_strict_lt_fails_it():
    """`_bca_endpoints` computes the bias correction as
    `prop = mean(boot < theta) + 0.5 * mean(boot == theta)` (mid-p, the SciPy convention), which is
    correct and necessary and which the other tests cannot tell from the strict-`<` alternative:
    `test_bca_midp_on_discrete_sample_brackets_median` asserts only that the CI brackets the median,
    and on its own sample [1,2,2,3,3,3,4,4,5] BOTH conventions bracket (strict-< gives (2,3), mid-p
    gives (2,4)), while the scipy oracle test uses continuous gamma samples where ties are
    essentially absent and the two conventions coincide. Agreement with SciPy is checked by
    `test_the_midp_z0_convention_agrees_with_the_pinned_scipy`.

    Part 1 pins z0 against an ANALYTIC oracle, on a hand-built bootstrap distribution with a known
    tie composition, so the expected value is derived from the BCa definition with the mid-p tie
    count rather than read off the implementation. Part 2 shows what the difference costs on the
    study's own data shape.
    """
    from statistics import NormalDist

    # --- Part 1: analytic oracle on a known tie composition ---------------------------------
    # 100 replicates: 10 strictly below theta, 60 tied at theta, 30 strictly above.
    theta = 2.0
    boot = np.array([1.0] * 10 + [2.0] * 60 + [3.0] * 30)
    jack = np.array([2.0] * 8)                      # symmetric -> acceleration a = 0 exactly
    N = NormalDist()

    #   mid-p    : prop = 0.10 + 0.5*0.60 = 0.40   -> z0 = Phi^-1(0.40) = -0.253347...
    #   strict-< : prop = 0.10                     -> z0 = Phi^-1(0.10) = -1.281552...
    z0_midp = N.inv_cdf(0.40)
    z0_strict = N.inv_cdf(0.10)
    assert z0_midp == pytest.approx(-0.2533471, abs=1e-6)
    assert z0_strict == pytest.approx(-1.2815516, abs=1e-6)

    # With a = 0 the BCa percentile positions collapse to Phi(2*z0 + z_q).
    def expected_ci(z0):
        lo = N.cdf(2 * z0 + N.inv_cdf(0.025))
        hi = N.cdf(2 * z0 + N.inv_cdf(0.975))
        return (float(np.percentile(boot, 100 * lo)), float(np.percentile(boot, 100 * hi)))

    got = S._bca_endpoints(theta, boot, jack, 0.05)
    assert got == pytest.approx(expected_ci(z0_midp)), (
        "the bias correction is not counting half the ties (mid-p)")
    assert got != pytest.approx(expected_ci(z0_strict)), (
        "mid-p and strict-< agree here, so this fixture cannot discriminate -- fix the fixture")

    # --- Part 2: what strict-< would cost on slice-quantized, zero-inflated gaps -------------
    # The manuscript's gaps are explicitly slice-thickness-quantized and zero-inflated, and their
    # median CIs flow through exactly this path. On such a sample ~95% of bootstrap medians tie
    # theta, so strict-< drives z0 to ~-2.5 and the interval collapses onto the zero spike:
    #     strict-< : prop=0.0055  z0=-2.5427  -> CI (0.0, 0.0)   <- degenerate, WRONG LOCATION
    #     mid-p    : prop=0.4790  z0=-0.0527  -> CI (2.0, 3.0)   <- what statlib returns
    # A degenerate zero-width median-gap CI at 0 mm would go straight into the manuscript.
    quantized = np.concatenate([np.zeros(12), np.full(14, 2.0), np.full(8, 4.0),
                                np.full(4, 6.0), np.full(2, 8.0)])
    lo, hi = S.median_ci_bca(quantized, seed=7)
    assert (lo, hi) == pytest.approx((2.0, 3.0))
    assert lo > 0.0, "the CI collapsed onto the zero spike -- the strict-< failure mode"
    assert hi > lo, "degenerate zero-width median CI"
    assert lo <= float(np.median(quantized)) <= hi


def test_the_midp_z0_convention_agrees_with_the_pinned_scipy():
    """`scipy.stats._resampling._percentile_of_score`, the function `_bca_interval` uses to
    compute z0, returns

        ((a < score).sum() + (a <= score).sum()) / (2 * B)

    which is algebraically `mean(a < score) + 0.5 * mean(a == score)`: mid-p, the convention
    `statlib._bca_endpoints` uses. This test checks that agreement against the SciPy actually
    installed, so a change of convention in SciPy fails here and says so. The version is pinned in
    `environment.yml` (`scipy=1.16.2`); the test reads the running one.

    The mid-p choice itself is pinned independently of SciPy by
    `test_bca_midp_z0_is_pinned_against_an_analytic_oracle_and_strict_lt_fails_it`, against an
    ANALYTIC oracle derived from the BCa definition, which stays valid whatever SciPy does.
    """
    scipy = pytest.importorskip("scipy")
    from scipy.stats import _resampling

    # 100 replicates: 10 strictly below theta, 60 tied at theta, 30 above -- the same tie
    # composition as the analytic-oracle test above, so the two are directly comparable.
    theta = 2.0
    boot = np.array([1.0] * 10 + [2.0] * 60 + [3.0] * 30)

    scipy_prop = float(_resampling._percentile_of_score(boot, theta, axis=-1))
    midp_prop = float(np.mean(boot < theta) + 0.5 * np.mean(boot == theta))
    strict_prop = float(np.mean(boot < theta))

    assert scipy_prop == pytest.approx(midp_prop), (
        f"scipy {scipy.__version__} does not compute the z0 percentile with mid-p "
        f"(got {scipy_prop}, mid-p is {midp_prop}). The rationale in statlib is written "
        f"relative to a SciPy that does; re-derive it before changing statlib.")
    assert scipy_prop != pytest.approx(strict_prop), (
        "this fixture has no ties, so it cannot tell mid-p from strict-< -- fix the fixture")
    assert scipy_prop == pytest.approx(0.40)      # 0.10 + 0.5*0.60, by hand

    # And the module's own endpoint routine uses that same proportion, so "statlib agrees with
    # scipy" is a statement about the executed path and not only about two formulas on paper.
    jack = np.array([2.0] * 8)                    # symmetric -> acceleration a = 0 exactly
    from statistics import NormalDist
    N = NormalDist()
    z0 = N.inv_cdf(scipy_prop)
    expected = (float(np.percentile(boot, 100 * N.cdf(2 * z0 + N.inv_cdf(0.025)))),
                float(np.percentile(boot, 100 * N.cdf(2 * z0 + N.inv_cdf(0.975)))))
    assert S._bca_endpoints(theta, boot, jack, 0.05) == pytest.approx(expected)


# --------------------------------------------------------------------------
# Input validation: gates in FRONT of the estimators (Wilson, the two-family BH
# including monotonicity). No estimator arithmetic is under test here; every one of
# these asserts what happens to input an estimator should never be handed.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [-0.5, -1e-12, 1.5, 1.0 + 1e-9, float("nan"),
                                 float("inf"), float("-inf")])
def test_bh_fdr_refuses_a_pvalue_outside_the_unit_interval(bad):
    with pytest.raises(ValueError, match=r"not in \[0, 1\]"):
        S.bh_fdr([bad, 0.2])


def test_bh_fdr_refuses_a_pvalue_that_is_not_a_number():
    with pytest.raises(ValueError, match="not a real number"):
        S.bh_fdr(["0.05", 0.2])          # a string that LOOKS numeric is still not a p-value
    with pytest.raises(ValueError, match="not a real number"):
        S.bh_fdr([object(), 0.2])
    with pytest.raises(ValueError, match="not a real number"):
        S.bh_fdr([None, "0.05"])         # not rescued by another entry being legitimately None


def test_bh_fdr_refuses_numeric_strings_because_the_step_up_ranks_by_the_raw_value():
    """This failure gives WRONG NUMBERS rather than invalid ones. `bh_fdr` ranks with
    `sorted(idx, key=lambda i: pvals[i])` -- the RAW value, not a coerced float -- so numeric
    strings would be ordered LEXICOGRAPHICALLY:

        bh_fdr(["1e-3", "0.05"])  would give  [0.001, 0.001]
        bh_fdr([1e-3, 0.05])      gives       [0.002, 0.05]     <- correct

    because `"0.05" < "1e-3"` as text while `0.001 < 0.05` as numbers. The second q-value would
    be 50x too small, yet finite, in [0, 1] and entirely plausible in a results table. Any p-value
    rendered in scientific notation -- which is exactly how small, interesting p-values are
    written -- is at risk, and no range check can catch it.

    The remedy is a TYPE check, not a coercion: `float("1e-3")` succeeds, so accepting anything
    `float()` can parse would keep the ordering error. `numbers.Real` admits Python
    floats and ints, numpy scalars and Fractions -- everything a real caller passes -- and
    excludes str and bytes.
    """
    with pytest.raises(ValueError, match="ordered as text"):
        S.bh_fdr(["1e-3", "0.05"])
    # the numeric form of the same family is the correct answer, hand-checked:
    # ranks 1,2 of m=2 -> q = [0.001*2/1, 0.05*2/2] = [0.002, 0.05]
    assert S.bh_fdr([1e-3, 0.05]) == pytest.approx([0.002, 0.05])
    # every numeric type a caller supplies is accepted
    from fractions import Fraction
    assert S.bh_fdr([np.float64(0.001), np.float32(0.05)]) == pytest.approx([0.002, 0.05],
                                                                           rel=1e-6)
    assert S.bh_fdr([0, 1]) == pytest.approx([0.0, 1.0])            # ints are Real
    assert S.bh_fdr([Fraction(1, 1000), Fraction(1, 20)]) == pytest.approx([0.002, 0.05])


def test_bh_fdr_refuses_bool_because_a_flag_column_yields_plausible_qvalues():
    """`bool` is `numbers.Real`, so the type check excludes it by name.

    Passing a `significant` flag column where a p-value column was meant is a realistic mix-up, and
    `True`/`False` coerce to 1.0/0.0 -- both inside [0, 1], both producing entirely plausible
    q-values with nothing to mark them as wrong."""
    with pytest.raises(ValueError, match="not a real number"):
        S.bh_fdr([True, False])
    with pytest.raises(ValueError, match="not a real number"):
        S.bh_fdr([0.01, True])
    with pytest.raises(ValueError, match="not a real number"):
        S.bh_fdr([np.bool_(True), 0.02])


def test_mcnemar_refuses_non_numeric_counts_and_never_raises_on_a_huge_one():
    """A count must BE a number rather than convert to one, for the same reason `bh_fdr` refuses a
    string p-value: a string count means the caller handed over unparsed data.

    And `mcnemar_test(10**400, 3)` must not raise `OverflowError` from `float()` (its docstring
    says "Never raises (module contract)"); it is reported through the same 'invalid' state as
    every other unusable count.
    """
    for b, c in [("3", 7), (3, "7"), (None, 7), (object(), 7), (True, 7), (3, False)]:
        assert S.mcnemar_test(b, c)["method"] == "invalid", f"({b!r}, {c!r}) was accepted"
    huge = S.mcnemar_test(10 ** 400, 3)          # must not raise OverflowError
    assert huge["method"] == "invalid" and math.isnan(huge["p"])
    # a plain large-but-representable count is a count and is accepted
    assert S.mcnemar_test(10 ** 6, 3)["method"] == "chi2_continuity"


def test_bh_fdr_does_not_hand_a_nan_pvalue_a_neighbours_qvalue():
    """Pinned by symptom rather than mechanism.

    Every comparison with NaN is False, so in an ungated step-up `min(running_min, nan)` returns
    `running_min` and the NEIGHBOURING test's q is carried into the invalid slot (`[nan, 0.2]`
    would give `[0.2, 0.2]`, a finite, significant-looking q-value belonging to a different test);
    a negative p would propagate a negative q (`[-0.5, 0.2]` -> `[-1.0, 0.2]`).

    `pytest.raises` also rules out a version that silently drops the bad entry and corrects the
    rest: it would RETURN rather than raise, and every case below would fail.
    """
    for bad in (float("nan"), -0.5, 1.5):
        with pytest.raises(ValueError):
            S.bh_fdr([bad, 0.2])
    # the valid neighbour on its own is untouched, so the refusal is about the invalid entry
    assert S.bh_fdr([0.2]) == pytest.approx([0.2])


def test_bh_fdr_keeps_every_legitimate_degenerate_case():
    """The module's never-raise contract covers DEGENERATE data.

    `None` means "this test was not run" and keeps its meaning; an empty family and an all-`None`
    family are answerable and are answered. Only genuinely invalid numbers are refused.
    """
    assert S.bh_fdr([]) == []
    assert S.bh_fdr([None, None]) == [None, None]
    q = S.bh_fdr([0.01, None, 0.02])
    assert q[1] is None and q[0] is not None and q[2] is not None
    # the exact boundary values are legitimate p-values, not edge cases to reject
    assert S.bh_fdr([0.0, 1.0]) == pytest.approx([0.0, 1.0])


def test_bh_fdr_why_silently_excluding_an_invalid_p_would_be_worse_than_raising():
    """The reason `bh_fdr` raises instead of dropping the bad entry, made arithmetic.

    Dropping an invalid p shrinks the family size m, and m multiplies every OTHER q in the
    family. So the error would not land on the invalid test -- it would land on the valid ones,
    which is precisely where nobody would look for it. Hand-derived: for p = 0.04 at rank 1 of
    m = 4, q = 0.04*4/1 = 0.16; at rank 1 of m = 3 it is 0.04*3/1 = 0.12, a 25% relative change
    in a published q-value caused by a test the reader never sees.

    Only the caller knows whether a test that produced a non-number should count toward m, which
    is why the decision is pushed to the caller. A caller that excludes it says so by mapping a
    non-finite p to None before the call (`p if np.isfinite(p) else None`).
    """
    four = S.bh_fdr([0.04, 0.5, 0.6, 0.7])
    three = S.bh_fdr([0.04, 0.5, 0.6])
    assert four[0] == pytest.approx(0.16)
    assert three[0] == pytest.approx(0.12)
    assert four[0] != pytest.approx(three[0])


@pytest.mark.parametrize("b,c", [(1.4, 2.6), (0.4, 0.4), (2.5, 2.5), (1, 0.5)])
def test_mcnemar_refuses_fractional_discordant_counts(b, c):
    r = S.mcnemar_test(b, c)
    assert r["method"] == "invalid"
    assert math.isnan(r["p"]) and not r["significant"]


def test_mcnemar_fractional_input_is_distinguishable_from_a_rounded_one():
    """Rounding with `int(round(...))` would make two different inputs one result.

    `mcnemar_test(1.4, 2.6)` would return EXACTLY what `mcnemar_test(1, 3)` returns -- same p
    (0.6249999999999994), same method, same n_discordant -- so a caller who had accidentally
    passed rates, weights or per-case means would get a fully formed test back. And
    `mcnemar_test(0.4, 0.4)`, 0.8 discordant "pairs", would round to (0, 0) and be reported as
    method 'none', i.e. as no discordant pairs at all, which reads as "no evidence of change"
    rather than "this input is not a pair count".
    """
    fractional = S.mcnemar_test(1.4, 2.6)
    rounded = S.mcnemar_test(1, 3)
    assert rounded["method"] == "exact_binomial"
    assert rounded["p"] == pytest.approx(0.625)
    assert fractional["method"] != rounded["method"], (
        "a fractional pair count is being rounded into a valid test")
    assert S.mcnemar_test(0.4, 0.4)["method"] == "invalid", (
        "0.8 discordant pairs is being reported as 'no discordant pairs'")


def test_mcnemar_accepts_whole_valued_counts_in_every_form_a_caller_supplies():
    """Refusing fractions must not refuse the numpy scalars every real caller passes."""
    ref = S.mcnemar_test(3, 7)
    assert ref["method"] == "exact_binomial"
    for b, c in [(3.0, 7.0), (np.int64(3), np.int64(7)), (np.float64(3.0), np.float64(7.0))]:
        r = S.mcnemar_test(b, c)
        assert r["method"] == ref["method"]
        assert r["p"] == pytest.approx(ref["p"])
        assert r["n_discordant"] == ref["n_discordant"]
    # and the 'none' and 'invalid' states are as specified
    assert S.mcnemar_test(0, 0)["method"] == "none"
    assert S.mcnemar_test(-1, 3)["method"] == "invalid"
    assert S.mcnemar_test(float("nan"), 3)["method"] == "invalid"


@pytest.mark.parametrize("k,n", [(12, 10), (-3, 10), (11, 10), (1, 0), (0, -5)])
def test_proportion_pct_ci_refuses_a_count_pair_its_own_interval_refuses(k, n):
    """A count pair that `wilson_ci` refuses is refused in all three positions: otherwise
    `proportion_pct_ci(12, 10)` would report an impossible 120% beside a NaN interval (and
    `(-3, 10)` a negative percentage), which in a results table reads as a real rate whose CI
    failed to compute.
    """
    pct, lo, hi = S.proportion_pct_ci(k, n)
    assert math.isnan(pct) and math.isnan(lo) and math.isnan(hi)


def test_proportion_pct_ci_definedness_is_all_or_nothing():
    """All three values defined, or all three NaN; never a mix.

    A range test `n <= 0 or k < 0 or k > n` alone cannot enforce this: every comparison is False
    against NaN, and the clamp `max(0.0, nan), min(1.0, nan)` returns `(0.0, 1.0)` (Python's
    max/min return the non-NaN operand), so a NaN count would get a FULLY DEFINED, maximally wide
    interval that an `isnan(lo)` guard cannot see, and `proportion_pct_ci(3, inf)` would read as a
    confident 0% with a zero-width interval.

    Both halves call `_valid_count_pair` directly, so the invariant is a property of the code
    rather than a coincidence of two guards happening to agree. The sweep below is the property
    itself, run over finite, non-finite and fractional inputs.
    """
    from itertools import product
    values = [0, 1, 5, 10, -3, 12, 0.5, 2.5, float("nan"), float("inf"), float("-inf")]
    checked = 0
    for k, n in product(values, values):
        pct, lo, hi = S.proportion_pct_ci(k, n)
        defined = {not math.isnan(x) for x in (pct, lo, hi)}
        assert len(defined) == 1, (
            f"proportion_pct_ci({k}, {n}) reported a mix of defined and NaN: {(pct, lo, hi)}")
        checked += 1
    assert checked == len(values) ** 2 == 121

    # and the same invariant one level down, in wilson_ci
    for k, n in product(values, values):
        lo, hi = S.wilson_ci(k, n)
        assert math.isnan(lo) == math.isnan(hi), f"wilson_ci({k}, {n}) = {(lo, hi)}"


@pytest.mark.parametrize("k,n", [(float("nan"), 10), (5, float("nan")), (3, float("inf")),
                                 (float("-inf"), 10), (0.5, 10), (1, 2.5), (2.5, 5),
                                 ("3", 10), (3, "10"), (True, 10)])
def test_wilson_and_proportion_refuse_non_finite_fractional_and_non_numeric_counts(k, n):
    """Inputs a three-clause range test cannot see.

    Non-finite is the dangerous one (a range test alone would give `(0.0, 1.0)`, a real-looking
    interval). Fractional is refused for the same reason `mcnemar_test` refuses fractional
    discordant counts: a fractional success tally means the caller computed a rate, a weight or a
    mean, and the binomial interval does not describe it.
    Strings and bools are refused because a count must BE a number, not convert to one --
    the consistency `bh_fdr`'s type check demands.
    """
    assert all(math.isnan(x) for x in S.proportion_pct_ci(k, n))
    assert all(math.isnan(x) for x in S.wilson_ci(k, n))


def test_proportion_pct_ci_point_estimate_lies_inside_its_own_interval():
    """A whole-range property check on the pair: the point estimate lies inside its own interval."""
    for n in range(1, 40):
        for k in range(0, n + 1):
            pct, lo, hi = S.proportion_pct_ci(k, n)
            assert lo - 1e-9 <= pct <= hi + 1e-9, f"{k}/{n}: {pct} outside ({lo}, {hi})"
            assert 0.0 <= lo <= hi <= 100.0


# --- the worst-case (Manski) bound on a proportion with missing outcomes -------------
def test_worst_case_proportion_bounds_are_the_hand_derived_cohort_values():
    """The three cohorts' apex-in-gap figures, the ends derived by hand (and by the exhaustive
    enumeration in the next test): with k in the gap
    among n_eval accepted fits (flag OK) out of N cases with z_gap > 0, and U cases of unknown membership,
    the range is [k/(N+U), (k + N - n_eval + U)/(N+U)]."""
    cases = {
        (12, 316, 419, 0): (12 / 419, 115 / 419),      # AMOS
        (7, 195, 236, 0): (7 / 236, 12 / 59),          # SAROS, the measured frame
        (7, 195, 236, 26): (7 / 262, 37 / 131),        # SAROS, its 26 EMPTY rows possibly in it
        (9, 369, 501, 0): (3 / 167, 47 / 167),         # autoPET
        (3, 10, 10, 0): (0.3, 0.3),                    # nothing missing: point-identified
        (0, 0, 3, 0): (0.0, 1.0),                      # no outcome at all: the whole unit interval
        (0, 0, 0, 2): (0.0, 1.0),                      # only members of unknown membership
    }
    for (k, n_eval, n_frame, n_unknown), want in cases.items():
        got = S.worst_case_proportion_bounds(k, n_eval, n_frame, n_unknown)
        assert got == pytest.approx(want, abs=1e-15), ((k, n_eval, n_frame, n_unknown), got)
    assert S.worst_case_proportion_bounds(12, 316, 419) == pytest.approx((12 / 419, 115 / 419)), (
        "n_unknown defaults to 0")


def test_worst_case_proportion_bounds_are_the_extremes_over_every_completion():
    """The definition, enumerated: every frame member without an outcome takes 0 or 1, every unit of
    unknown membership is out of the frame or in it with 0 or 1, and the range is the least and
    greatest rate over the completions with a non-empty frame. An end that is only sometimes the
    extreme (the lower end computed at e = 0 rather than e = U, say) fails here."""
    import itertools
    from fractions import Fraction

    for n_frame in range(0, 6):
        for n_eval in range(0, n_frame + 1):
            for k in range(0, n_eval + 1):
                for n_unknown in range(0, 3):
                    rates = set()
                    for miss in itertools.product((0, 1), repeat=n_frame - n_eval):
                        for unk in itertools.product((None, 0, 1), repeat=n_unknown):
                            size = n_frame + sum(u is not None for u in unk)
                            if size:
                                rates.add(Fraction(k + sum(miss) + sum(u == 1 for u in unk), size))
                    got = S.worst_case_proportion_bounds(k, n_eval, n_frame, n_unknown)
                    if not rates:
                        assert all(math.isnan(x) for x in got), (k, n_eval, n_frame, n_unknown)
                        continue
                    want = (float(min(rates)), float(max(rates)))
                    assert got == pytest.approx(want, abs=1e-15), (k, n_eval, n_frame, n_unknown)


@pytest.mark.parametrize("args", [
    (5, 4, 10, 0),                       # more successes than evaluated outcomes
    (3, 11, 10, 0),                      # more evaluated than the frame holds
    (-1, 4, 10, 0), (1, -4, 10, 0), (1, 4, -10, 0), (1, 4, 10, -1),
    (0, 0, 0, 0),                        # an empty frame: no proportion exists
    (float("nan"), 4, 10, 0), (1, 4, float("inf"), 0), (1, 4, 10, float("nan")),
    (1.5, 4, 10, 0), (1, 4, 10, 0.5),    # a fractional tally is a rate, not a count
    (True, 4, 10, 0), ("1", 4, 10, 0), (1, 4, None, 0),
])
def test_worst_case_proportion_bounds_refuse_an_invalid_count(args):
    """The module's count contract (`_valid_count_pair`): never raise, and never return a range
    built from a miscounted, non-finite, fractional or non-numeric tally. A NaN in one end only
    would read as half a bound, so both ends refuse together."""
    lo, hi = S.worst_case_proportion_bounds(*args)
    assert math.isnan(lo) and math.isnan(hi), (args, lo, hi)


def test_worst_case_proportion_bounds_never_raise_and_return_floats():
    """The contract is NaN, never an exception, and the ends are floats whatever whole-number type the
    counts arrive as."""
    from fractions import Fraction
    huge = 10 ** 400
    assert all(math.isnan(x) for x in S.worst_case_proportion_bounds(huge, huge, huge))
    for counts in ((Fraction(1), Fraction(3), Fraction(5)), (np.int64(1), np.int64(3), np.int64(5)),
                   (1.0, 3.0, 5.0)):
        lo, hi = S.worst_case_proportion_bounds(*counts)
        assert type(lo) is float and type(hi) is float, (counts, type(lo), type(hi))
        assert (lo, hi) == pytest.approx((0.2, 0.6)), counts


def test_describe_validates_the_method_before_the_empty_input_shortcut():
    """An empty cohort must not be the one place a typo survives.

    `describe` returns `{"n": 0}` on empty input before it would ever reach `median_ci_auto`, so
    delegating the check would catch a typo for a populated cohort and let it pass for an
    empty one -- the worst place to lose it, since an empty cohort is exactly when nobody is
    reading the numbers closely.
    """
    with pytest.raises(ValueError, match="not a confidence-interval method"):
        S.describe([], ci_method="bcaa")
    assert S.describe([]) == {"n": 0}


def test_the_allowed_ci_methods_are_exactly_the_two_that_are_implemented():
    """The validator's allowed set must not drift away from what the branches implement, and the
    reported name must match the estimator that ran.

    Checking that `S.median_ci_auto(v, name)` is not None cannot fail (it always returns a tuple,
    `(nan, nan)` at worst), and checking `describe(v, ci_method=name)["median_ci_method"] == name`
    is `x == x`, because `describe` assigns that key from the argument; on `[1,2,3,4,9,2.5,3.5]`
    the BCa and percentile median CIs are both `(2.0, 4.0)`. A `describe` that always ran BCa while
    echoing `ci_method` would report an estimator that never ran. Validating the NAME does not
    connect the name to the ESTIMATOR; only comparing the returned interval against the
    independently computed one does.
    """
    assert S.CI_METHODS == ("bca", "percentile")

    # A fixture on which the two estimators genuinely differ, asserted first: on a
    # fixture where they agree the rest of this test would be untestable in substance.
    v = np.round(np.random.default_rng(7).gamma(2.0, 2.0, size=60), 1)
    bca, pct = S.median_ci_bca(v), S.median_ci(v)
    assert bca != pct, "fixture cannot distinguish the two estimators -- fix the fixture"

    # The reported name must match the estimator that ACTUALLY RAN, for every allowed
    # name.
    expected = {"bca": bca, "percentile": pct}
    for name in S.CI_METHODS:
        assert S.median_ci_auto(v, name) == pytest.approx(expected[name])
        d = S.describe(v, ci_method=name)
        assert d["median_ci_method"] == name
        assert (d["median_ci_lo"], d["median_ci_hi"]) == pytest.approx(expected[name]), (
            f"describe reported median_ci_method={name!r} but returned the interval of "
            f"the other estimator -- the label and the computation have come apart")


def test_the_block_jackknife_docstring_does_not_promise_coverage():
    """Matching the acceleration's jackknife to the bootstrap's resampling removes a definite
    internal contradiction. It does not establish that the resulting interval has correct
    coverage under dependence (no theory does), and the recognised route is studentization
    (Goetze & Kuensch 1996), which this module does not implement.
    """
    doc = " ".join(S._jackknife_delete_block.__doc__.split())
    for token in ("does NOT buy a coverage guarantee", "no theory", "STUDENTIZATION",
                  "Goetze & Kuensch 1996", "coherent HEURISTIC", "Liu & Singh 1992",
                  "Kuensch 1989", "percentile-t"):
        assert token in doc, (
            f"_jackknife_delete_block's scope statement does not say {token!r}; a coherent "
            f"acceleration is not a guaranteed one")


# --- what the McNemar rule means ------------------------------------------------------------
def test_the_mcnemar_rule_admits_it_cites_against_current_guidance():
    """Both branches are the two variants Fagerland et al. 2013 advise against.

    The rule has one consumer, the discordance there is extreme, and every variant returns the
    same conclusion. That invariance is what makes this a disclosure, so the test re-derives it
    from the study's discordant counts: a near-balanced split would make the p-values disagree,
    and this fires.
    """
    doc = S.mcnemar_test.__doc__
    for token in ("Fagerland", "23848987", "did not perform well for any of the",
                  "exact conditional", "continuity correction", "mid-p",
                  "NEAR-BALANCED"):
        assert token in doc, (
            f"the McNemar rule's guidance disclosure does not state {token!r}")

    # The study's discordant counts, and the invariance the disclosure claims.
    worsts = []
    for b, c in ((177, 1), (47, 0), (51, 0)):
        n = b + c
        exact = S.mcnemar_test(b, c, exact=True)["p"]
        yates = S.mcnemar_test(b, c, exact=False)["p"]
        midp = max(0.0, exact - math.comb(n, min(b, c)) * 0.5 ** n)
        chi_nocc = (b - c) ** 2 / n
        asym = math.erfc(math.sqrt(chi_nocc / 2))
        assert n >= 25, "the AUTO rule would not take the Yates branch on this split"
        assert S.mcnemar_test(b, c)["method"] == "chi2_continuity"
        worst = max(exact, yates, midp, asym)
        assert worst < 1e-10, (
            f"the four McNemar variants disagree at b={b}, c={c}: "
            f"exact={exact:.3e} yates={yates:.3e} midp={midp:.3e} asym={asym:.3e}. The "
            f"disclosure's claim that the conclusion is invariant to the test choice does not "
            f"hold for these counts")
        worsts.append(worst)
    # The single largest p over 3 cohorts x 4 methods, which the docstring quotes: AMOS/Yates.
    assert max(worsts) == pytest.approx(1.95e-11, rel=1e-2), (
        f"the worst-case McNemar p over all cohorts and methods is {max(worsts):.3e}, "
        f"not the 1.95e-11 the disclosure states")


