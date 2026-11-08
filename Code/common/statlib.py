"""Design rules: * Every estimator that a referee would scrutinize lives here and is unit-tested
against hand-computed / textbook / scipy-oracle values (see test_statlib.py). * numpy is required.
scipy is OPTIONAL: only kruskal_test() AND kruskal_eps2_ci() need it (the latter calls kruskal_test
for the point estimate and every bootstrap replicate), plus shapiro_p()'s normality p-value;
everything else works without scipy. The scipy-dependent paths degrade to None/NaN when scipy is
absent. Normal/t quantiles come from the stdlib `statistics` module so confidence intervals never
require scipy. * Functions are NaN-tolerant by construction (they drop NaNs) and never raise on
degenerate input (empty, singleton, constant) — they return NaN/flags so a batch summary is never
silently corrupted by one bad case. Two exceptions RAISE: - `bh_fdr` refuses an invalid p-value:
there is no honest q-value for an uncorrectable p, and the two silent alternatives are both wrong in
ways that land on the OTHER tests in the family, so the caller must decide (see its docstring). -
the CI-method dispatchers RAISE on an unrecognised method name, which is configuration rather than
data. Everything else honours the contract as written,
including `wilson_ci`, `proportion_pct_ci` and `mcnemar_test`, which report invalid input as
NaN/flags.

Conventions:
  * "sample" statistics use ddof=1 (we are estimating population parameters from
    finite cohorts, not describing a census).
  * Proportions are reported with Wilson score intervals (good small-n coverage,
    never leaves [0,1]).
  * Uncertainty on medians and on the bifurcation x-intercept is obtained by the
    bootstrap (BCa by default, second-order accurate for skewed/ratio statistics WITH A
    CONTINUOUS sampling distribution -- NOT for a lattice-valued median, see `bca_ci`;
    percentile available), made reproducible with an explicit seed. A moving-block
    bootstrap with a matching delete-block jackknife is provided for autocorrelated
    (slice-ordered) data so the BCa acceleration is coherent with the resampling.
"""
from __future__ import annotations

import math
import numbers
from statistics import NormalDist

import numpy as np

# Default seed / n_boot for every bootstrap so published numbers are reproducible. Falls back to the
# literals if gap_config is not importable, so statlib stays usable stand-alone / dependency-light.
try:
    import gap_config as _gc
    DEFAULT_SEED = int(_gc.SEED)
    DEFAULT_NBOOT = int(_gc.N_BOOT)
except Exception:
    DEFAULT_SEED = 20240611
    DEFAULT_NBOOT = 2000


# ---------------------------------------------------------------------------
# Critical values (stdlib, no scipy needed)
# ---------------------------------------------------------------------------
def z_crit(alpha: float = 0.05) -> float:
    """Two-sided normal critical value, e.g. 1.95996 for alpha=0.05."""
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


# --- Student-t distribution from the regularized incomplete beta (stdlib only) ---
# Implemented so the module's confidence intervals and the curvature/flat-slope
# tests never depend on scipy. A normal quantile/tail in place of the t would be
# severely anti-conservative at the small df (n-2, n-3) this code uses (e.g. df=4:
# t_crit 2.776 vs z 1.960). These are exact to ~1e-9 vs scipy and pure stdlib
# (math.lgamma).
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (Lentz's method, NR betacf);
    converges for x < (a+1)/(a+b+2)."""
    MAXIT, EPS, FPMIN = 300, 1e-15, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) in [0, 1] (pure stdlib)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_sf(t: float, df: int) -> float:
    """Upper-tail Student-t survival function P(T > t), df>0 (pure stdlib).
    P(|T|>|t|) = I_x(df/2, 1/2) with x = df/(df+t^2); halve for one tail."""
    if df is None or df <= 0:
        return float("nan")
    t = float(t)
    x = df / (df + t * t)
    ib = 0.5 * _betai(df / 2.0, 0.5, x)          # = P(T > |t|)
    return ib if t >= 0.0 else 1.0 - ib


def t_cdf(t: float, df: int) -> float:
    """Student-t CDF P(T <= t), df>0 (pure stdlib)."""
    s = t_sf(t, df)
    return float("nan") if s != s else 1.0 - s


def t_ppf(p: float, df: int) -> float:
    """Student-t quantile: the t with P(T <= t) = p, df>0 (pure stdlib).
    Inverts the monotone CDF by bisection; ~1e-9 accurate over the df grid used."""
    if df is None or df <= 0 or not (0.0 < p < 1.0):
        return 0.0 if p == 0.5 else float("nan")
    if p == 0.5:
        return 0.0
    if p < 0.5:                                   # use symmetry about 0
        return -t_ppf(1.0 - p, df)
    hi = 4.0                                       # expand until cdf(hi) brackets p
    while t_cdf(hi, df) < p and hi < 1e12:
        hi *= 2.0
    lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def t_crit(alpha: float, df: int) -> float:
    """Two-sided Student-t critical value t_{1-alpha/2, df} (pure stdlib).
    Exact to ~1e-9 vs scipy and does NOT degrade to the normal quantile z at small
    df. Returns NaN for df<=0."""
    if df is None or df <= 0:
        return float("nan")
    return t_ppf(1.0 - alpha / 2.0, df)


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------
def _clean(v) -> np.ndarray:
    a = np.asarray(v, dtype=float).ravel()
    return a[np.isfinite(a)]   # drop NaN AND +/-inf so one bad case never corrupts the batch


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------
CI_METHODS = ("bca", "percentile")


def _check_ci_method(method, argname: str = "ci_method") -> str:
    """Refuse an unrecognised confidence-interval method name.

    A method name is CONFIGURATION, not data: there is no degenerate-data reading of
    `"percentil"`, so the module's never-raise-on-degenerate-input contract does not
    reach it. THREE functions branch on the name with a two-branch `if name ==
    <one option>: ... else: <the other>`, which cannot distinguish "the other option"
    from "not an option at all", and they fall in OPPOSITE directions:
    `median_ci_auto`/`paired_median_diff_ci` test for `"percentile"`, so without this
    check a typo would silently select BCa, while `wall_contact_apex` tests for `"bca"`,
    so the same typo would silently select the percentile interval. `describe` is the
    fourth site and does not branch at all: it delegates to `median_ci_auto` and ECHOES
    the name into `median_ci_method`, which is why it could report an estimator that did
    not run and why it is checked here explicitly.

    That matters more here than a typo usually would, because of where the name comes
    from. In production the name is `gap_config.CI_METHOD` at every site:
    `median_ci_auto` (every Phase-1 median CI that goes through it), `describe`
    (`stats.py` x3), `paired_median_diff_ci` (`stats.py`) and `wall_contact_apex`
    (`measure_bifurcation.py` x2), and that constant is stamped into
    `Results/phase1_characterize/*/provenance.txt`. Without this check a single mistyped
    character would degrade every apex interval from second-order to first-order
    accuracy, leave the median intervals alone, and record the mistyped name in the
    provenance of all three cohorts as though it were a declared choice.
    """
    if method not in CI_METHODS:
        raise ValueError(
            f"{argname}={method!r} is not a confidence-interval method; "
            f"expected one of {CI_METHODS}. (In production this argument usually "
            f"carries gap_config.CI_METHOD, which is stamped into provenance.txt, so "
            f"a typo there would be recorded as a deliberate choice.)")
    return method


# ---------------------------------------------------------------------------
# Dispersion
# ---------------------------------------------------------------------------
def sample_sd(v, ddof: int = 1) -> float:
    """Sample standard deviation (ddof=1). NaN when n <= ddof."""
    a = _clean(v)
    if a.size <= ddof:
        return float("nan")
    return float(np.std(a, ddof=ddof))


def skewness(v) -> float:
    """Population Fisher-Pearson skewness (g1). 0 for a symmetric sample."""
    a = _clean(v)
    if a.size < 2:
        return float("nan")
    m = a.mean()
    m2 = float(((a - m) ** 2).mean())
    if m2 <= 0:
        return 0.0
    m3 = float(((a - m) ** 3).mean())
    return m3 / m2 ** 1.5


def shapiro_p(v):
    """Shapiro-Wilk normality p-value, or None if scipy/sample too small."""
    a = _clean(v)
    if a.size < 3:
        return None
    try:
        from scipy.stats import shapiro
        if a.size > 5000:                      # shapiro unreliable/slow huge n
            a = a[np.linspace(0, a.size - 1, 5000).astype(int)]
        return float(shapiro(a).pvalue)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Proportion confidence interval (Wilson score)
# ---------------------------------------------------------------------------
def _valid_count_pair(k, n) -> bool:
    """Is (k, n) a usable binomial (successes, trials) pair?

    Both must be real numbers (not bool), finite and WHOLE, with 0 <= k <= n and n > 0.
    Finiteness is checked because every ordering test against NaN is False: a range test alone
    would accept NaN, and the clamp `max(0.0, nan), min(1.0, nan)` would return **(0.0, 1.0)**
    (Python's max/min return the non-NaN operand), a fully defined, maximally wide interval,
    while `proportion_pct_ci(3, inf)` would read as a confident 0% with a zero-width CI. Whole
    numbers are required for the same reason `mcnemar_test` refuses fractional discordant counts:
    a fractional success tally means the caller computed a rate, a weight or a mean, and the
    binomial interval does not describe it."""
    for x in (k, n):
        if isinstance(x, bool) or not isinstance(x, numbers.Real):
            return False
        xf = float(x)
        if not math.isfinite(xf) or xf != int(xf):
            return False
    return not (n <= 0 or k < 0 or k > n)


def wilson_ci(k: int, n: int, alpha: float = 0.05):
    """Wilson score interval for a binomial proportion k/n. (lo, hi) in [0,1].
    Wilson (1927), in the form of the NIST/SEMATECH e-Handbook of Statistical Methods, section
    7.2.4.1 (its U.L. and L.L., with z = z_{1-alpha/2}); it assumes independent units.
    Returns (nan, nan) for degenerate or invalid input rather than raising a
    math-domain error — honoring the module's 'never raise on degenerate input'
    contract (a caller that passes k>n, e.g. a miscounted success tally, gets NaN,
    not a crash that aborts the batch). See `_valid_count_pair` for what 'invalid'
    covers (negative, out-of-range, non-finite and fractional counts).

    The bounds are clamped to the point estimate: at
    k = 0 and k = n they meet it exactly in arithmetic but can land a rounding step outside
    it in floating point."""
    if not _valid_count_pair(k, n):
        return float("nan"), float("nan")
    z = z_crit(alpha)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return min(max(0.0, center - margin), phat), max(min(1.0, center + margin), phat)


def proportion_pct_ci(successes: int, n: int, alpha: float = 0.05):
    """(pct, lo_pct, hi_pct) for a proportion, Wilson interval, as percentages.

    A `successes` outside [0, n] means a miscounted tally, and the honest report of a miscounted
    tally is NaN in all three positions, not a confident-looking point estimate with the uncertainty
    silently missing.

    The predicate is called DIRECTLY rather than inferred from `isnan(lo)`: both halves are gated
    on the same predicate, so the invariant "all three defined or all three NaN" is a property of
    the code rather than a coincidence of two guards agreeing.

    Degenerate input returns NaNs rather than raising (module contract).
    """
    if not _valid_count_pair(successes, n):
        return float("nan"), float("nan"), float("nan")
    lo, hi = wilson_ci(successes, n, alpha)
    return 100.0 * successes / n, 100.0 * lo, 100.0 * hi


def _whole_count(x) -> bool:
    """A count as `_valid_count_pair` reads one: a real number, not a bool, finite, whole, >= 0."""
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        return False
    try:
        xf = float(x)
    except OverflowError:                       # an int beyond float range is no count we can use
        return False
    return math.isfinite(xf) and xf == int(xf) and xf >= 0


def worst_case_proportion_bounds(successes, n_evaluated, n_frame, n_unknown=0):
    """(lo, hi) in [0, 1]: the range a frame's proportion can take when nothing is assumed about
    the members whose outcome is missing.

    `successes` of `n_evaluated` members with an outcome, out of `n_frame` members; the other
    `n_frame - n_evaluated` have none. Manski's bound for a missing binary outcome puts each of
    them at 0 for the lower end and at 1 for the upper, so the range is
    [k / N, (k + N - n_evaluated) / N]. `n_unknown` more units may or may not belong to the frame
    (a case whose membership could not be measured); over every e of them joining it, the lower
    end k / (N + e) falls and the upper end (a + e) / (N + e), a = k + N - n_evaluated <= N, rises,
    so the union is [k / (N + U), (k + N - n_evaluated + U) / (N + U)] (verified by exhaustive
    enumeration of completions in
    `test_worst_case_proportion_bounds_are_the_extremes_over_every_completion`). Both forms are
    Manski's: Partial
    Identification of Probability Distributions (Springer 2003), Corollary 1.1.1 (eq. 1.10) and,
    with the unknown units as the jointly missing cell and no outcome observable without
    membership, Corollary 3.8.1 (eq. 3.24).

    This is an identification range, not a confidence interval: it says what the data cannot rule
    out, before sampling error (no confidence interval is computed here, neither for the rate,
    Imbens and Manski, Econometrica 2004, nor for the range, Horowitz and Manski, JASA 2000). It is wide by construction when many outcomes are missing, which is
    its point. It presumes every missing outcome exists and went unobserved, and counts units as
    independent members of a finite frame.

    Refuses (NaN, NaN), never raises, unless every count is whole and non-negative in the sense of
    `_valid_count_pair`, successes <= n_evaluated <= n_frame, and the frame can be non-empty
    (n_frame + n_unknown > 0); otherwise no proportion exists."""
    k, n, frame, unknown = (int(x) if _whole_count(x) else None
                            for x in (successes, n_evaluated, n_frame, n_unknown))
    if None in (k, n, frame, unknown) or not k <= n <= frame or frame + unknown == 0:
        return float("nan"), float("nan")
    size = frame + unknown
    return k / size, (k + frame - n + unknown) / size


# ---------------------------------------------------------------------------
# Paired binary test (McNemar) + paired median-difference CI, for a
# before/after comparison on the SAME cases (the collapsed-class diagnostic:
# gap present/absent, and gap magnitude, BEFORE vs AFTER the venous-class merge
# + nearest-trunk re-split). Only the DISCORDANT cells (b, c) enter McNemar; the
# two marginal Wilson present-rates (proportion_pct_ci) IGNORE the pairing and
# are descriptive only.
# ---------------------------------------------------------------------------
def mcnemar_test(b, c, exact=None, alpha: float = 0.05) -> dict:
    """McNemar test on the discordant cells of a paired binary 2x2.

    b = # pairs positive-before / negative-after; c = # negative-before /
    positive-after (the concordant cells a, d are irrelevant to McNemar). Tests
    H0: the two marginal positive-rates are equal, i.e. the b:c split is 50/50.

    `exact`:
      * True  -> two-sided EXACT binomial p = min(1, 2*P(X <= min(b,c))),
        X ~ Binomial(b+c, 0.5), via the regularized incomplete beta (_betai,
        pure stdlib): P(X<=k) = I_{0.5}(n-k, k+1). No overflow, no scipy.
      * False -> chi-square (1 df) with Yates continuity correction; the
        correction is FLOORED at 0 so b == c gives chi2 = 0 (p = 1), avoiding the
        classic (|b-c|-1)^2 = 1 over-correction at zero difference.
      * None  -> AUTO: exact when the discordant total b+c < 25 (small-sample
        regime where the normal approximation is unreliable), else Yates chi².

    BOTH BRANCHES ARE THE TWO VARIANTS THE CURRENT RECOMMENDATION LITERATURE ADVISES
    AGAINST. Fagerland
    MW, Lydersen S & Laake P, BMC Med Res Methodol 2013;13:91
    (doi:10.1186/1471-2288-13-91, PMID 23848987) compare the mid-p, asymptotic,
    asymptotic-with-continuity-correction, exact-conditional and exact-unconditional
    McNemar tests over 9,595 scenarios and conclude that "the exact conditional test and
    the asymptotic test with continuity correction did not perform well for any of the
    considered scenarios". `exact=True` here IS the exact conditional test and
    `exact=False` IS the continuity-corrected asymptotic one, so the rule above cites
    against current guidance in both directions: the exact branch is over-conservative and
    the Yates branch violates the nominal level. Their recommendation is mid-p (which did
    not violate the nominal level in any of the 9,595 scenarios) or the exact unconditional
    test, with the plain asymptotic test acceptable where small violations are.

    THE CHOICE DOES NOT AFFECT THE REPORTED CONCLUSION, AND THAT IS MEASURED RATHER THAN ASSERTED.
    Recomputing every variant on the study's discordant counts gives p < 1e-10 throughout; the
    single largest p over all three cohorts and all four methods is 1.95e-11 (AMOS, Yates), against
    an autoPET Yates value of 2.64e-39 at the other end. **The reported conclusion is invariant to
    the test choice**, and the marginal rates it feeds are labelled descriptive-only. The choice
    would matter for a consumer with a NEAR-BALANCED discordant split, or with b+c small enough
    that the branch boundary at 25 is in play; mid-p is the recommended test there, and the
    one-sided exact tail it needs is computed below via `_betai`.

    Never raises (module contract): b + c == 0 (no discordant pairs -> no
    information about change) -> chi2 NaN, p 1.0, method 'none'; a negative,
    non-finite or NON-INTEGRAL count -> p NaN, method 'invalid'. Returns
    dict(b, c, n_discordant, chi2, p, method, significant). chi2 is the Yates
    statistic, always reported as a descriptive effect size regardless of which
    p was used; `significant` = p < alpha (False when p is NaN or method 'none').

    NON-INTEGRAL COUNTS ARE REFUSED, not rounded. b and c are counts of
    DISCORDANT PAIRS; a fractional value means the caller computed something that is
    not a pair count (a rate, a weighted tally, a mean over cases) and the test does
    not apply to it. Rounding would turn such input into a fully formed result:
    `mcnemar_test(1.4, 2.6)` would equal `mcnemar_test(1, 3)` (p = 0.625, method
    'exact_binomial'), and `mcnemar_test(0.4, 0.4)`, 0.8 discordant "pairs", would round to
    (0, 0) and read as method 'none', i.e. as *no discordant pairs at all*. A fractional
    count gets the same 'invalid' state as a negative or non-finite one, rather than a
    second failure mode; a whole-valued
    float (3.0, or numpy's int64/float64, which every caller here passes) is a
    count and is accepted.
    """
    invalid = {"b": b, "c": c, "n_discordant": float("nan"),
               "chi2": float("nan"), "p": float("nan"), "method": "invalid",
               "significant": False}
    # `bool` is excluded because True/False are flags, not tallies. OverflowError is caught because
    # `float(10**400)` raises it and the docstring promises never to raise.
    for x in (b, c):
        if isinstance(x, bool) or not isinstance(x, numbers.Real):
            return invalid
    try:
        bf, cf = float(b), float(c)
    except (TypeError, ValueError, OverflowError):
        return invalid
    if not (math.isfinite(bf) and math.isfinite(cf)) or bf < 0 or cf < 0 \
            or bf != int(bf) or cf != int(cf):
        return invalid
    bi, ci = int(bf), int(cf)
    n = bi + ci
    if n == 0:
        return {"b": bi, "c": ci, "n_discordant": 0, "chi2": float("nan"),
                "p": 1.0, "method": "none", "significant": False}
    cc = max(0.0, abs(bi - ci) - 1.0)          # Yates correction, floored at 0
    chi2 = cc * cc / n
    use_exact = (n < 25) if exact is None else bool(exact)
    if use_exact:
        k = min(bi, ci)
        p = min(1.0, 2.0 * _betai(n - k, k + 1, 0.5))     # 2*P(X<=k), Binom(n,0.5)
        method = "exact_binomial"
    else:
        # For one degree of freedom, P(Chi2 >= x) = erfc(sqrt(x/2)).  Computing
        # this as 2*Phi(-sqrt(x)) is algebraically equivalent but NormalDist.cdf
        # loses the extreme lower tail and gives an impossible exact p=0 at the
        # collapsed-class discordance observed in this project (for example
        # b=177, c=1). erfc preserves the representable survival probability.
        p = min(1.0, math.erfc(math.sqrt(chi2 / 2.0)))
        method = "chi2_continuity"
    return {"b": bi, "c": ci, "n_discordant": n, "chi2": float(chi2),
            "p": float(p), "method": method, "significant": bool(p < alpha)}


def paired_median_diff_ci(x, y, alpha: float = 0.05, n_boot: int = DEFAULT_NBOOT,
                          seed: int = DEFAULT_SEED, method: str = "bca"):
    """(median_diff, lo, hi): CI on the MEDIAN of the paired differences d = x - y.

    x[i] and y[i] must already be aligned to the SAME unit (the caller inner-joins
    by case id, never zips positionally). BCa by default; method='percentile' for the
    first-order interval. BCa's second-order accuracy is a result for smooth statistics
    (DiCiccio & Efron 1996, pp.203-205), and a median of differences with an atom is not
    one. The interval is ZERO-WIDTH, (m, m), not only for a constant difference
    vector (e.g. collapsed == original everywhere) but whenever an atom reaches far enough
    past the middle on both sides that almost every resample has its median on it: on the
    collapsed-class pairs, where 63-72 % of differences are exactly
    zero, it is a point in all three cohorts although np.ptp(d) > 0. So a caller
    tests the interval with `interval_is_degenerate`, prints it with `format_ci`, and never
    asserts a real change from the interval alone. An empty pair returns NaNs.

    An unrecognised `method` raises rather than silently selecting BCa.
    """
    _check_ci_method(method, "method")
    pair = _clean_pair(x, y)
    if pair is None:
        return float("nan"), float("nan"), float("nan")
    a, bb = pair
    d = a - bb
    if method == "percentile":
        lo, hi = bootstrap_ci(d, np.median, alpha, n_boot, seed)
    else:
        lo, hi = bca_ci(d, np.median, alpha, n_boot, seed)
    return float(np.median(d)), float(lo), float(hi)


#: What a published interval cell carries in place of `m-m`. The estimators above return
#: the point they compute (their contract, and the docstring of
#: `paired_median_diff_ci`); publication refuses it, so no table prints a point as an interval.
DEGENERATE_INTERVAL = "DEGENERATE_ZERO_WIDTH"


def interval_is_degenerate(lo, hi) -> bool:
    """True exactly when (lo, hi) is a point: two finite real ends that are equal.

    A bootstrap interval for a median collapses to a point when an atom of tied values reaches far
    enough past the middle of the sample on both sides that almost every resample has its median
    on it (the two tail probabilities decide it, not the atom's share). In the Phase-1 tables this
    occurs at atoms of 9.5 % (AMOS's collapsed gap, 5.00 mm), 23.3 %
    (autoPET's axial z_gap, 2.50 mm, one slice) and 63-72 % (the unchanged pairs of the
    collapsed-class paired difference), while a 51 % atom with every other value above it gives an
    honest interval. So the test is the interval itself, never the atom's share. A median's
    bootstrap distribution is discrete in any case (Hesterberg, The American Statistician 2015,
    doi:10.1080/00031305.2015.1089789, section 3.3, "Sample Median").

    Absent (None, NaN) and unbounded (inf) ends are not a point, and stay with the caller's own
    rules (`format_ci` blanks them; the Manski bound prints its own infinity)."""
    if not (is_finite_real(lo) and is_finite_real(hi)):
        return False
    return bool(_exact(lo) == _exact(hi))


def _exact(x):
    """`x` as a value whose comparison with another interval end is exact.

    Python compares int, float and Fraction exactly, so NumPy integers become `int` and NumPy floats
    of at most 64 bits become `float`, which widens them exactly."""
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating) and np.finfo(x).bits <= 64:
        return float(x)
    return x


def is_finite_real(x) -> bool:
    """True when `x` can be an end of a printed interval: a finite real number, not a bool.

    The one gate that `interval_is_degenerate` and `format_ci` share, so the two cannot disagree
    about what an end is: numpy scalars count (`np.float32` and `np.int64` register as
    `numbers.Real`), `True` and `False` do not although Python makes `bool` an `int`, and None, a
    string, NaN and inf do not.
    """
    if isinstance(x, bool) or not isinstance(x, numbers.Real):    # np.bool_ is not Real either
        return False
    try:
        return bool(math.isfinite(x))
    except OverflowError:            # an integer past float64's range cannot be printed as one
        return False


def format_ci(lo, hi, nd: int = 1, sep: str = "-", signed: bool = False) -> str:
    """The one way a Phase-1 confidence interval is printed as text.

    - An end that is not a finite real number (`is_finite_real`) gives `""`. A bootstrap or
    parametric interval with a NaN or infinite end is not an interval. A genuinely unbounded bound,
    the Manski worst-case median, is printed by its own emitter and never comes here. - A point
    (`interval_is_degenerate`) gives `DEGENERATE_INTERVAL`, never `m-m`. - Otherwise the ends get
    `nd` decimals, or more, up to six, when `nd` would print them alike: [2.50, 2.55] at one decimal
    would print `2.5-2.5`. The printed VALUES are compared, so `-0.00` and `0.00` count as alike.

    `sep` and `signed` (a `+` on positive ends) set a call site's separator and sign style; they
    change nothing else.
    """
    if not (is_finite_real(lo) and is_finite_real(hi)):
        return ""
    if _exact(lo) > _exact(hi):
        raise ValueError(f"format_ci: lower end {lo!r} is above upper end {hi!r}; an interval's "
                         f"ends are passed in order")
    if interval_is_degenerate(lo, hi):
        return DEGENERATE_INTERVAL
    sign = "+" if signed else ""
    for digits in range(nd, max(nd, 6) + 1):
        text_lo, text_hi = f"{lo:{sign}.{digits}f}", f"{hi:{sign}.{digits}f}"
        if float(text_lo) != float(text_hi):
            return f"{text_lo}{sep}{text_hi}"
    text_lo, text_hi = repr(float(lo)), repr(float(hi))
    if text_lo == text_hi:           # finer than float64 (big integers, Fraction, longdouble)
        text_lo, text_hi = str(lo), str(hi)
    if signed:
        text_lo, text_hi = (t if t.startswith("-") else "+" + t for t in (text_lo, text_hi))
    return f"{text_lo}{sep}{text_hi}"


# ---------------------------------------------------------------------------
# Multiple-testing correction (Benjamini-Hochberg FDR)
# ---------------------------------------------------------------------------
def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values (q-values).

    Preserves input order. `None` entries are excluded from the family and
    returned as `None`, so callers can keep a stable column of results while
    only correcting the tests that were actually run.

    RAISES `ValueError` on a p-value that is not a real number in [0, 1]: here refusing is
    more honest than flagging (the CI-method check is the module's other refusal). The
    module's no-raise contract covers DEGENERATE data (empty, singleton, constant),
    and those are handled: `[]` returns `[]` and an all-`None` family returns all
    `None`. An out-of-range p is not degenerate data, it is a caller bug, and there
    is no defensible number to return for it:

      * EXCLUDING it (treating it like `None`) silently shrinks the family size m,
        which makes every OTHER q-value in that family anti-conservative — the
        error lands on the valid tests, where nobody would look for it;
      * KEEPING it means inventing a value, and an ungated step-up does worse:
        because every comparison with NaN is False, `min(running_min, nan)` returns
        `running_min`, so a NaN p-value would silently inherit the neighbouring
        test's q (`bh_fdr([nan, 0.2])` would give `[0.2, 0.2]`, a finite,
        significant-looking q-value belonging to a different test), and a negative
        p would propagate a negative q (`[-0.5, 0.2]` -> `[-1.0, 0.2]`).

    Only the caller knows whether a test that produced a non-number should count
    toward m, so the caller decides. The idiom is to map a non-finite p to `None`
    before calling here (`p if np.isfinite(p) else None`), which declares the
    exclusion explicitly instead of letting it happen by accident.

    A p-value must also BE a real number, not merely convert to one. The
    step-up ranks by the RAW value (`sorted(idx, key=lambda i: pvals[i])`), so a
    family of numeric STRINGS would be ordered lexicographically and give wrong
    q-values wherever the two orders differ: `bh_fdr(["1e-3", "0.05"])` would give
    `[0.001, 0.001]` where the correct answer is `[0.002, 0.05]`, a 50-fold
    overstatement of significance for the second test, because `"0.05" < "1e-3"` as
    text. `float()` accepts every one of those strings, so coercion alone cannot catch
    it; the type is refused. numpy scalars, ints and Fractions are `numbers.Real` and
    are accepted.

    The validation is a gate in front of the standard step-up.
    """
    for i, p in enumerate(pvals):
        if p is None:
            continue
        if isinstance(p, bool) or not isinstance(p, numbers.Real):
            raise ValueError(
                f"bh_fdr: p-value at index {i} is not a real number: {p!r}. A value "
                f"that merely converts to a float is not enough — the step-up ranks by "
                f"the raw value, so a numeric string would be ordered as text. `bool` "
                f"is refused too: True/False is a significance FLAG, and passing a "
                f"flag column where a p column was meant yields entirely plausible "
                f"q-values. `Decimal` is not `numbers.Real` and is refused with "
                f"everything else; pass `float(d)` if you mean it. Pass None for a "
                f"test that was not run.")
        pf = float(p)
        if not (math.isfinite(pf) and 0.0 <= pf <= 1.0):
            raise ValueError(
                f"bh_fdr: p-value at index {i} is not in [0, 1]: {pf!r}. A NaN or "
                f"out-of-range p cannot be corrected — decide whether the test "
                f"counts toward the family size m and pass None if it does not.")
    idx = [i for i, p in enumerate(pvals) if p is not None]
    out = [None] * len(pvals)
    m = len(idx)
    if m == 0:
        return out
    order = sorted(idx, key=lambda i: pvals[i])          # ascending p
    running_min = float("inf")
    for rank in range(m, 0, -1):                          # m -> 1
        i = order[rank - 1]
        q = float(pvals[i]) * m / rank
        running_min = min(running_min, q)
        out[i] = min(running_min, 1.0)
    return out


# ---------------------------------------------------------------------------
# Kruskal-Wallis omnibus test + effect size
# ---------------------------------------------------------------------------
def kruskal_test(*groups):
    """Kruskal-Wallis H test with an epsilon-squared effect size.

    Returns dict(H, p, df, eps_squared, eta_squared, n, k). p/H are None when
    fewer than 2 non-empty groups exist or scipy is unavailable. Effect sizes:
      eps_squared = H / (n - 1)              in [0, 1]
      eta_squared = max(0, (H-k+1)/(n-k))    bias-adjusted variant, clamped at 0
    The bias-adjusted eta^2 (analogous to adjusted R^2) can dip slightly below 0
    near the null; it is clamped to 0 for reporting. eta_squared is NaN when n<=k.
    """
    clean = [_clean(g) for g in groups]
    clean = [g for g in clean if g.size > 0]
    n = int(sum(g.size for g in clean))
    k = len(clean)
    base = {"H": None, "p": None, "df": None, "eps_squared": None,
            "eta_squared": None, "n": n, "k": k}
    if k < 2 or n <= k:
        return base
    try:
        from scipy.stats import kruskal
    except ImportError:
        return base                      # scipy absent -> p reported as NA
    try:
        H, p = kruskal(*clean)
    except ValueError:
        return base                      # e.g. all values identical -> no signal
    # any other exception is a real bug and is allowed to propagate
    H = float(H)
    base.update(H=H, p=float(p), df=k - 1,
                # tie-corrected H can exceed n-1 (perfect separation with heavy
                # ties pushes H/(n-1) to 1+eps in float), so eps^2 is clamped to
                # the docstring's [0,1] exactly as eta^2 is clamped at 0
                eps_squared=min(1.0, H / (n - 1)) if n > 1 else float("nan"),
                # bias-adjusted eta^2 (analogous to adjusted R^2) can dip slightly
                # below 0 near the null; clamp to 0 for reporting. NaN when n<=k.
                eta_squared=max(0.0, (H - k + 1) / (n - k)) if n > k else float("nan"))
    return base


def kruskal_eps2_ci(groups, alpha: float = 0.05, n_boot: int = DEFAULT_NBOOT,
                    seed: int = DEFAULT_SEED):
    """Stratified-bootstrap BCa CI for the Kruskal-Wallis epsilon-squared.

    Resamples cases WITHIN each group (stratified, group sizes preserved) and
    recomputes eps^2 = H/(n-1) per replicate, then takes BCa endpoints
    (`_bca_endpoints`): z0 from the mid-p position of the point estimate in its
    own bootstrap, acceleration from a delete-one-case jackknife that respects
    group membership — the grouped jackknife matched to the stratified
    resampling. BCa rather than percentile because resampling a continuous
    sample WITH replacement manufactures ties, scipy's tie correction inflates
    every replicate's H, and the bootstrap distribution is therefore shifted
    upward relative to the point estimate; the plain percentile interval
    inherits that shift wholesale, while the BCa z0 term measures and removes
    the mis-centring (the per-replicate tie inflation itself is a property of
    the resampling and is not further corrected). Returns (eps2, lo,
    hi), all within [0,1] via the kruskal_test clamp. The point is to report a
    magnitude label (negligible/small/medium/large) WITH its uncertainty rather
    than asserting it from a point estimate that sits a few thousandths from a
    Cohen/Tomczak cutpoint.

    Two guarantees the BCa ends do not give near the null: the interval contains its own estimate
    (widened when both ends fall on one side of it), and its lower end is 0 whenever the
    Kruskal-Wallis test of the same groups does not reject at `alpha` (rejection is p < alpha, as
    everywhere here), so the interval then covers eps^2 = 0: on null data with identical group
    distributions the coverage of 0 is the test's non-rejection rate (in null-data simulations,
    94.5-97.0 % with this rule against 0-27.5 % for the plain BCa ends). Two limits: with tied
    data a replicate can reach H = 0, so the lower end can also be 0 beside a rejecting test; and
    with groups that share a centre but differ in spread the test itself is anticonservative when
    the smaller group is the more dispersed, so the coverage of 0 falls with it (and conservative the other way round, where
    it rises). eps^2 sits on its boundary under the null, where (n-1) eps^2 is asymptotically
    chi-square on k-1 degrees of freedom rather than normal, and BCa's theory assumes an
    asymptotically normal standardized statistic (DiCiccio & Efron 1996, p.203). The percentile
    fallback (in `kruskal_eps2_raw`) and the finite-ends guard (in `eps2_interval_rule`) are
    defensive: with the real `_bca_endpoints` neither runs (it returns None only below two
    replicates, where `kruskal_eps2_raw` has already returned). NaN-safe; (nan, nan, nan) if scipy
    is absent or k<2 (no H means no point estimate either); (eps2, nan, nan) when fewer than 2
    replicates are computable."""
    return eps2_interval_rule(*kruskal_eps2_raw(groups, alpha, n_boot, seed), alpha=alpha)


def kruskal_eps2_raw(groups, alpha: float = 0.05, n_boot: int = DEFAULT_NBOOT,
                     seed: int = DEFAULT_SEED):
    """The bootstrap and grouped jackknife behind `kruskal_eps2_ci`, before its interval rule.

    Returns (eps2, lo, hi, p); (nan, nan, nan, nan) for k < 2 or no H; (eps2, nan, nan, p) for
    fewer than 2 computable replicates."""
    clean = [_clean(g) for g in groups]
    clean = [g for g in clean if g.size > 0]
    if len(clean) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    kt = kruskal_test(*clean)
    e0 = kt.get("eps_squared")
    if e0 is None:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        res = [g[rng.integers(0, g.size, g.size)] for g in clean]
        e = kruskal_test(*res).get("eps_squared")
        if e is not None and e == e:
            boot.append(e)
    if len(boot) < 2:
        return float(e0), float("nan"), float("nan"), float(kt["p"])
    # Per-GROUP jackknives, kept separate: each group's delete-one values enter the acceleration
    # as their own sample (see `bca_acceleration`) rather than as one pooled list, which matters
    # because the groups are markedly unequal in the study data (up to 13:1 and 20:1), the regime
    # where pooling is worst.
    jack = []
    for gi, g in enumerate(clean):
        per_group = []
        for i in range(g.size):
            red = [np.delete(g, i) if j == gi else clean[j]
                   for j in range(len(clean))]
            ej = kruskal_test(*red).get("eps_squared")
            if ej is not None and ej == ej:
                per_group.append(ej)
        if len(per_group) >= 2:
            jack.append(np.asarray(per_group, float))
    ep = _bca_endpoints(float(e0), boot, jack, alpha)
    if ep is None:                      # mirrors bca_ci: fall back to percentile
        ep = (float(np.percentile(boot, 100 * alpha / 2)),
              float(np.percentile(boot, 100 * (1 - alpha / 2))))
    return float(e0), float(ep[0]), float(ep[1]), float(kt["p"])


def eps2_interval_rule(e0, lo, hi, p, alpha: float = 0.05):
    """Non-finite ends pass through unchanged (the finite-ends guard)."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return float(e0), float(lo), float(hi)
    #  eps^2 is bounded below by 0 and biased upward under the null (mean (k-1)/(n-1)), and
    # its bootstrap reproduces the bias, so the BCa ends can both miss the estimate and rarely reach 0
    # (on null data they cover 0 in 0-27.5 % of replicates). The interval is widened to contain its
    # estimate, and its lower end is 0 whenever the Kruskal-Wallis test of the same groups does not
    # reject at this alpha, so it then covers eps^2 = 0 (with tied data a replicate can reach H = 0,
    # and the lower end can also be 0 beside a rejecting test).
    lo, hi = min(float(lo), float(e0)), max(float(hi), float(e0))
    if p >= alpha:
        lo = 0.0
    return float(e0), lo, hi


# ---------------------------------------------------------------------------
# Method-agreement statistics (axial vs 3D; fast vs full-res)
# ---------------------------------------------------------------------------
def lin_ccc(x, y) -> float:
    """Lin's concordance correlation coefficient. 1.0 == perfect agreement.

    Lin LI, Biometrics 1989;45:255-268.

    IT IS AN INTERCHANGEABILITY INDEX FOR TWO PROCESSING SETTINGS, NOT A RELIABILITY
    COEFFICIENT, and it is SECONDARY to Bland-Altman here, never the headline. It answers
    "do these two pipelines return the same number on the same case", which does not
    generalise beyond the two settings compared and says nothing about reproducibility
    across scanners, readers or protocols.

    AND IT IS INFLATED BY A WIDE BETWEEN-CASE RANGE -- measured on this study's own data,
    not merely cited. Atkinson G & Nevill AM, Sports Med 1998;26(4):217-38
    (doi:10.2165/00007256-199826040-00002, PMID 9820922) warn that correlation-based
    reliability indices are "highly influenced by the range of measured values" and caution
    against "concluding acceptable relative reliability even if a correlation is above 0.9".
    On the study's autoPET fast-vs-full pairs (TotalSegmentator 2.17.0, n=1014, CCC = 0.771),
    restricting the BETWEEN-CASE range while leaving the measurement process untouched drops CCC
    to 0.764 / 0.711 / 0.631 / 0.535 / 0.471 / 0.386 at full-res gap ceilings of
    30 / 20 / 15 / 10 / 8 / 6 mm -- **while the Bland-Altman SD of differences FALLS as
    well, 4.35 -> 2.60 mm.** The measurement error gets smaller as the index loses half its
    value, which is the whole point: the index is reading case heterogeneity, not agreement.
    Quote it beside the LoA, never instead of them.
    """
    x = _clean_pair(x, y)
    if x is None:
        return float("nan")
    a, b = x
    mx, my = a.mean(), b.mean()
    vx = ((a - mx) ** 2).mean()
    vy = ((b - my) ** 2).mean()
    cov = ((a - mx) * (b - my)).mean()
    denom = vx + vy + (mx - my) ** 2
    return float(2 * cov / denom) if denom > 0 else float("nan")


def icc_a1(x, y) -> float:
    """ICC(A,1): two-way single-measure ABSOLUTE-agreement intraclass corr.

    For two methods (k=2 columns) over n subjects (rows). Penalizes systematic
    bias between methods (unlike consistency ICC). 1.0 == identical readings.

    THE FULL MODEL LABEL, BECAUSE THERE ARE TEN FORMS AND THE NAME ALONE DOES NOT PICK ONE.
    Report it as **ICC(A,1): two-way mixed-effects model, absolute agreement, single measure**
    (McGraw KO & Wong SP, Psychol Methods 1996;1:30-46, Table 4 -- the A,1 entry). Koo TK & Li MY, J
    Chiropr Med 2016;15(2):155-63 (doi:10.1016/j.jcm.2016.02.012, PMID 27330520) is the
    model-selection rationale and is explicit that "researchers should explicitly specify the ICC
    form they used", with best practice reporting model, type and definition.

    SCOPE: the two "raters" are two TotalSegmentator inference paths, so this is agreement
    between two PROCESSING SETTINGS. It is a MIXED-effects model precisely because those two
    settings are the only ones of interest and are not a random draw, so the coefficient does
    NOT generalise to reliability across scanners, readers or protocols. Koo & Li's
    poor/moderate/good/excellent bands (<0.5 / 0.5-0.75 / 0.75-0.9 / >0.9) are for reliability
    studies and are not a licence to grade this number that way.

    RANGE DEPENDENCE APPLIES HERE EXACTLY AS IT DOES TO CCC -- see :func:`lin_ccc`, where the
    measurement is recorded. On the study's autoPET pairs ICC(A,1) tracks CCC to three
    decimals across every range restriction (0.771 down to 0.386), so nothing about the
    intraclass form protects it.
    """
    pair = _clean_pair(x, y)
    if pair is None:
        return float("nan")
    a, b = pair
    M = np.column_stack([a, b])
    n, k = M.shape
    if n < 2:
        return float("nan")
    grand = M.mean()
    row = M.mean(axis=1)
    col = M.mean(axis=0)
    ss_rows = k * ((row - grand) ** 2).sum()
    ss_cols = n * ((col - grand) ** 2).sum()
    ss_total = ((M - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + (k / n) * (msc - mse)
    return float((msr - mse) / denom) if denom != 0 else float("nan")


def bland_altman(x, y, alpha: float = 0.05) -> dict:
    """Bland-Altman agreement: bias (mean difference), SD of differences,
    95% parametric limits of agreement (LoA) with CIs on each (Bland & Altman
    1986/1999), PLUS non-parametric percentile LoA and validity diagnostics.
    Differences are x - y.

    The parametric LoA = bias +/- z*SD assume the DIFFERENCES are ~normal and homoscedastic. For the
    skewed, magnitude-dependent gaps here those assumptions can fail, so we also report: *
    loa_lo_np/loa_hi_np : empirical 2.5/97.5 percentiles of the differences. MARGINAL
    distribution-free limits: they assume nothing about the SHAPE of the difference distribution,
    and that is the whole of what they repair. No constant-width interval CAN be conditionally
    valid under an SD-magnitude relation; a simulation that separates the two failures confirms
    these limits DO repair skew (0.9500 vs the parametric pair's 0.9622 at skewness 7.0) and leave
    heteroscedasticity untouched. * prop_bias_r : the correlation of the differences with the pair
    means (a proportional-bias check). This is the BIAS axis only. Its significance test is
    identical to the test on the slope of difference-on-mean, so a near-zero value says the CENTRE
    does not drift with magnitude and says NOTHING about whether the SPREAD does: corr(M, D) can be
    exactly 0 with SD(D|M) unbounded, and autoPET is this study's instance (r = +0.0096 at
    p = 0.761, SD slope p = 2.05e-41). Nothing here reports a heteroscedasticity
    diagnostic. Mechanics note: LoA themselves use the normal quantile z (BA convention); only the
    CI ON each limit uses t_{n-1}. That is correct and is NOT a z-vs-t bug; the relevant caveat is
    the normality assumption, hence the np LoA above."""
    pair = _clean_pair(x, y)
    if pair is None:
        return {"n": 0, "bias": float("nan"), "sd_diff": float("nan"),
                "loa_lo": float("nan"), "loa_hi": float("nan"),
                "loa_lo_np": float("nan"), "loa_hi_np": float("nan"),
                "diff_normal_p": None, "prop_bias_r": float("nan")}
    a, b = pair
    d = a - b
    n = d.size
    bias = float(d.mean())
    sd = sample_sd(d)
    z = z_crit(alpha)
    loa_lo, loa_hi = bias - z * sd, bias + z * sd
    # non-parametric LoA (distribution-free): central 95% of the differences
    loa_lo_np = float(np.percentile(d, 100 * alpha / 2))
    loa_hi_np = float(np.percentile(d, 100 * (1 - alpha / 2)))
    # proportional-bias check: correlate difference with the mean of the methods
    means = (a + b) / 2.0
    if n >= 3 and np.ptp(d) > 0 and np.ptp(means) > 0:
        prop_bias_r = float(np.corrcoef(means, d)[0, 1])
    else:
        prop_bias_r = float("nan")
    out = {"n": int(n), "bias": bias, "sd_diff": sd,
           "loa_lo": float(loa_lo), "loa_hi": float(loa_hi),
           "loa_lo_np": loa_lo_np, "loa_hi_np": loa_hi_np,
           "diff_normal_p": shapiro_p(d), "prop_bias_r": prop_bias_r}
    if n >= 2 and not math.isnan(sd):
        tc = t_crit(alpha, n - 1)
        se_bias = sd / math.sqrt(n)
        se_loa = sd * math.sqrt(1.0 / n + z * z / (2 * (n - 1)))   # BA 1999
        out.update(bias_ci_lo=bias - tc * se_bias, bias_ci_hi=bias + tc * se_bias,
                   loa_lo_ci_lo=loa_lo - tc * se_loa, loa_lo_ci_hi=loa_lo + tc * se_loa,
                   loa_hi_ci_lo=loa_hi - tc * se_loa, loa_hi_ci_hi=loa_hi + tc * se_loa)
    return out


def binned_empirical_loa(x, y, n_bins: int = 3, alpha: float = 0.05) -> dict:
    """Conditional, distribution-free limits of agreement: empirical percentiles WITHIN bands
    of measurement magnitude. Differences are x - y; bands are quantiles of (x + y) / 2.

    WHY THIS EXISTS, and why neither existing pair can do its job
    -------------------------------------------------------------------
    The parametric pair assumes normal, homoscedastic differences. The marginal non-parametric
    pair drops the normality assumption and keeps the homoscedasticity one -- it is still a
    CONSTANT-WIDTH interval, and no constant-width interval can be
    conditionally valid under an SD-magnitude relation. All three cohorts here have one at
    p from 2e-41 to 1e-68, so this is not a hypothetical.

    Measured by simulation, on a generator fitted to six summary statistics per cohort (300
    replicates, fitted on n and evaluated on 40,000 independent draws, nominal 0.950):

        estimator        overall   small gaps   mid    large gaps
        marginal np      0.943-0.948   1.000   0.941-0.988   0.855-0.888
        this one         0.935-0.943   0.946-0.954   0.927-0.938   0.929-0.938

    Every estimator hits ~0.95 OVERALL -- the marginal figure is the one figure that carries no
    information about the defect. Conditionally, the marginal pair excludes nothing at small
    gaps and under-covers by 6 to 10 points at large ones, and a sweep from n=100 to n=2000
    shows that pattern is FLAT: it is structural, so more cases cannot repair it.

    WHAT IT COSTS
    ------------- * It is a STEP function of magnitude, not a
    smooth one. Bin edges are quantiles of the observed means, so they are data-dependent in
    LOCATION while the RULE (equal-count bins) is fixed in advance -- the same posture as reporting
    tertiles, which this project already does. Below ~500 the bins hold too few points to estimate a
    2.5th percentile and it under-covers. In the study tables autoPET (1014) meets it and AMOS
    (493) and SAROS (362) do not; `min_bin_n` is reported so a reader can see the exposure rather
    than infer it. * `n_bins` defaults to **3** because this project already reports magnitude
    TERTILES and the measurement above was made at 3. It is an argument, not a module constant, so
    no caller inherits it silently.

    Returns `bins` (one dict per band, in increasing magnitude), plus the ENVELOPE across bands
    and `min_bin_n`. The envelope is the union of the per-band intervals: it is what a consumer
    that can only hold two scalars should show, it is never NARROWER than any band, and it must
    NOT be quoted as a constant-width LoA -- that is the defect this function removes.
    """
    pair = _clean_pair(x, y)
    if pair is None:
        return {"n": 0, "n_bins": 0, "bins": [], "envelope_lo": float("nan"),
                "envelope_hi": float("nan"), "min_bin_n": 0}
    a, b = pair
    d, m = a - b, (a + b) / 2.0
    n = d.size
    if n < 2 * n_bins:
        # too few points to band at all: report ONE band, which degenerates to the marginal
        # pair. Degenerating loudly beats banding a handful of points into noise.
        lo, hi = float(np.percentile(d, 100 * alpha / 2)), float(np.percentile(d, 100 * (1 - alpha / 2)))
        return {"n": int(n), "n_bins": 1, "min_bin_n": int(n),
                "bins": [{"index": 0, "m_lo": float(m.min()), "m_hi": float(m.max()),
                          "n": int(n), "loa_lo": lo, "loa_hi": hi}],
                "envelope_lo": lo, "envelope_hi": hi}
    edges = np.quantile(m, np.linspace(0.0, 1.0, n_bins + 1))
    bins = []
    for i in range(n_bins):
        lo_e, hi_e = edges[i], edges[i + 1]
        sel = (m >= lo_e) & (m <= hi_e) if i == n_bins - 1 else (m >= lo_e) & (m < hi_e)
        chunk = d[sel]
        if chunk.size == 0:
            continue
        bins.append({"index": i, "m_lo": float(lo_e), "m_hi": float(hi_e), "n": int(chunk.size),
                     "loa_lo": float(np.percentile(chunk, 100 * alpha / 2)),
                     "loa_hi": float(np.percentile(chunk, 100 * (1 - alpha / 2)))})
    return {"n": int(n), "n_bins": len(bins), "bins": bins,
            "min_bin_n": min(bn["n"] for bn in bins),
            "envelope_lo": min(bn["loa_lo"] for bn in bins),
            "envelope_hi": max(bn["loa_hi"] for bn in bins)}


def primary_loa(ba: dict, alpha: float = 0.05, binned: dict = None) -> dict:
    """Apply the project's declared LoA rule and say WHICH pair it selects.

    `bland_altman` returns both pairs; this function records which one the rule selects. On the
    study data the normality test rejects in every cohort (Shapiro-Wilk p of order 1e-26 to 1e-29)
    and the choice is not cosmetic: on SAROS the parametric band is 75 % wider than the empirical
    one.

    The rule implemented here is exactly the rule as written and nothing more. In
    particular `prop_bias_r` is REPORTED, never thresholded: a proportional-bias cutoff would be
    an uncalibrated constant. A caller that wants to weigh proportional bias has the number.

    The justification is not a p-value but a proof plus a measurement: no constant-width interval
    CAN be conditionally valid under an SD-magnitude relation, and both pairs above are
    constant-width. Making the choice unconditional is deliberate; a rule that switched on a
    heteroscedasticity test would introduce exactly the uncalibrated cutoff the paragraph above
    refuses, and would leave the estimator depending on the power of that test.

    Without a binned bundle the function falls back to the normality rule (the `FALLBACK` reasons
    below); that is a caller omission, not a second rule.

    Returns `kind` ("binned_conditional" | "nonparametric" | "parametric" | "undetermined"),
    `lo`, `hi`, and a `reason` short enough to put in a figure caption or a CSV cell. For the
    binned kind `lo`/`hi` are the ENVELOPE across bands -- never narrower than any band, and
    not to be quoted as a constant-width interval.
    """
    if binned and binned.get("bins"):
        nb, mn = binned["n_bins"], binned["min_bin_n"]
        thin = " min bin n=%d < 100, see the n>=500 caveat" % mn if mn < 100 else ""
        return {"kind": "binned_conditional",
                "lo": binned["envelope_lo"], "hi": binned["envelope_hi"],
                "reason": (f"conditional distribution-free LoA in {nb} equal-count magnitude "
                           f"bands; no constant-width interval can be conditionally valid under "
                           f"an SD-magnitude relation, and all cohorts have one. lo/hi are the "
                           f"ENVELOPE across bands, not a constant-width LoA; per-band limits are "
                           f"in the companion _agreement_binned.csv.{thin}")}
    p = ba.get("diff_normal_p")
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return {"kind": "undetermined", "lo": ba.get("loa_lo", float("nan")),
                "hi": ba.get("loa_hi", float("nan")),
                "reason": "FALLBACK (no binned bundle supplied): normality of the "
                          "differences could not be tested; parametric limits shown, "
                          "unvalidated"}
    if p < alpha:
        return {"kind": "nonparametric", "lo": ba["loa_lo_np"], "hi": ba["loa_hi_np"],
                "reason": f"FALLBACK (no binned bundle supplied): differences non-normal "
                          f"(Shapiro-Wilk p={p:.2g} < {alpha:g}); marginal distribution-free "
                          f"LoA, which is constant-width and so not conditionally valid"}
    return {"kind": "parametric", "lo": ba["loa_lo"], "hi": ba["loa_hi"],
            "reason": f"FALLBACK (no binned bundle supplied): differences consistent with "
                      f"normality (Shapiro-Wilk p={p:.2g} >= {alpha:g}); parametric LoA"}


def _clean_pair(x, y):
    a = np.asarray(x, float).ravel()
    b = np.asarray(y, float).ravel()
    if a.size != b.size:
        # Misalignment is a caller bug and must surface as one, not as a flattering number.
        raise ValueError(
            f"paired inputs differ in length: {a.size} vs {b.size}; the two series must be "
            f"aligned case-by-case before a paired statistic is computed")
    ok = np.isfinite(a) & np.isfinite(b)   # drop NaN AND +/-inf (match _clean) so one
    a, b = a[ok], b[ok]                     # bad case never corrupts a paired summary
    return (a, b) if a.size else None


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------
def bootstrap_ci(v, stat_fn=np.median, alpha: float = 0.05,
                 n_boot: int = DEFAULT_NBOOT, seed: int = DEFAULT_SEED):
    """Percentile bootstrap CI for an arbitrary statistic. Reproducible under
    `seed`. Degenerate (constant) input returns (c, c); empty returns (nan,nan)."""
    a = _clean(v)
    if a.size == 0:
        return float("nan"), float("nan")
    if a.size == 1 or np.ptp(a) == 0:
        return float(a[0]), float(a[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boot = np.array([float(stat_fn(a[row])) for row in idx])
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return lo, hi


def median_ci(v, alpha: float = 0.05, n_boot: int = DEFAULT_NBOOT,
              seed: int = DEFAULT_SEED):
    return bootstrap_ci(v, np.median, alpha, n_boot, seed)


# --------------------------------------------------------------------------- Bias-corrected and
# accelerated (BCa) bootstrap — second-order accurate, the correct choice for skewed statistics
# (medians of skewed gaps) and ratio statistics (the apex x-intercept = -intercept/slope). Falls
# back to the percentile interval when the bias/acceleration terms are undefined. THE
# SECOND-ORDER-ACCURACY CLAIM IS SCOPED, and the scope excludes one of this study's own endpoints.
# It is a theorem about SMOOTH statistics; the median of a LATTICE-VALUED distribution is not
# smooth, and there the interval is worse than the first-order one it improves on (90.0% coverage
# on the study's autoPET population at the even-n midpoint target, below). On the same
# population the plain percentile interval covers 94.7% and the exact binomial order-statistic
# interval 94.9%, so this is a property of BCa on a lattice and not of the lattice. Smooth the
# population and it largely goes away (94.2-94.8% under smoothing). THE MECHANISM IS z0 BELOW,
# FORMED ON A BOOTSTRAP DISTRIBUTION WITH AN ATOM AT THE ESTIMATE. Changing only the tie convention
# in its count (mid-p to strict-<, the guard below kept) moves coverage from 90.0% to 95.3% at the
# even-n midpoint target, and that is two effects. Elsewhere, at most 35% tied, the switch moves
# 95.9% to 97.1%, both above nominal. Conditioning on the atom lowers every interval's coverage;
# what is BCa's is the size of the loss there. So 95.3% is the strict count under this guard, not
# the published strict interval, whose limit there is the point interval [2.50, 2.50]. **That is NOT
# a reason to switch tie rules.** `test_statlib.py`'s Part 2 records what strict-< does on a small
# quantized zero-inflated sample (n=40, ~95% of bootstrap medians tied): z0 goes to -2.5427 and the
# interval COLLAPSES to (0.0, 0.0), degenerate and at the wrong location. Neither tie rule is a
# remedy; the safe alternatives are the ones that never form z0 at all.
def bca_acceleration(jack) -> float:
    """BCa acceleration from a jackknife, correct for one OR several samples.

    Efron & Tibshirani, *An Introduction to the Bootstrap*, eq (15.36), which is what
    `scipy.stats.bootstrap(method="BCa")` implements:

        U_ji = (n_j - 1) * (theta_hat_j_dot - theta_hat_ji)
        a    = (1/6) * sum_j[ sum_i U_ji^3 / n_j^3 ] / ( sum_j[ sum_i U_ji^2 / n_j^2 ] )^{3/2}

    Each sample is centred on ITS OWN jackknife mean and scaled by ITS OWN n_j. Passing a
    concatenated jackknife to the single-sample form (eq 14.15) is wrong twice over: the
    pooled mean is an n_j-weighted blend of the strata means, so it injects a per-stratum
    offset, and the per-stratum scaling is lost. On E&T's own unequal-n mouse-data example
    the pooled form returns 0.012155, against the reference 0.011008 -- 10.4 % high.

    Note there are two published conventions. Davison & Hinkley (R's `boot::bca.ci`) centre
    on the full-data statistic and drop the 1/n_j; they disagree with E&T under unequal n.
    E&T/SciPy is adopted here because it is the one carrying a pinned reference value, and
    because R's own `boot` is internally inconsistent about it (`var.linear` in the same
    package is strata-aware). What both agree on, and where the pooled form is uniquely
    indefensible, is that the centring point must be stratum-neutral.

    `jack` may be a flat array (one sample) or a sequence of per-sample arrays. For J=1 this
    reduces to eq (14.15) exactly -- the (n-1) and 1/n^k factors cancel -- which
    `test_single_sample_acceleration_reduces_to_the_single_sample_formula` pins over 200
    random configurations. Returns 0.0, the no-correction value, when the acceleration is
    undefined.
    """
    # A sequence whose ELEMENTS are themselves sequences is several samples; anything else
    # (a flat list of floats, an ndarray) is one. Deciding this by element shape rather than
    # by container type keeps `bca_acceleration([1.0, 2.0, 3.0])` meaning what a reader
    # expects instead of three one-element strata.
    if (isinstance(jack, (list, tuple)) and jack
            and all(np.ndim(s) >= 1 for s in jack)):
        samples = [np.asarray(s, float).ravel() for s in jack]
    else:
        samples = [np.asarray(jack, float).ravel()]

    num = den = 0.0
    for s in samples:
        s = s[np.isfinite(s)]
        n = s.size
        if n < 2:
            continue
        u = (n - 1) * (s.mean() - s)
        num += float((u ** 3).sum()) / n ** 3
        den += float((u ** 2).sum()) / n ** 2
    if den <= 0.0:
        return 0.0
    return float(num / (6.0 * den ** 1.5))


def _bca_endpoints(theta_hat, boot, jack, alpha):
    """BCa interval endpoints from the bootstrap replicates `boot`, the jackknife
    replicates `jack`, and the point estimate `theta_hat`. Returns (lo, hi) or
    None to signal the caller should fall back to the percentile interval."""
    boot = np.asarray(boot, float)
    boot = boot[np.isfinite(boot)]
    if boot.size < 2:
        return None
    N = NormalDist()
    # bias-correction z0 from the fraction of replicates below the estimate, counting ties at half
    # weight. Many bootstrap replicates of a median over a small/discrete sample equal theta
    # exactly, and a strict `<` undercounts them and biases z0 downward (shifting the whole
    # interval). ATTRIBUTION: IT IS NOT EFRON'S. The BCa bias correction as
    # published (Efron 1987; DiCiccio & Efron 1996) is the strict form z0 = Phi^-1(#{theta* <
    # theta_hat}/B) with no tie rule; the mid-p idea -- crediting half of a discrete probability
    # atom -- comes from discrete p-VALUE theory (Lancaster HO, JASA 1961; 56:223-234) and no source
    # applies it to z0. WHAT IT ACTUALLY IS: SciPy's convention, and that is checkable rather than
    # asserted. `scipy.stats._resampling._percentile_of_score`, which `_bca_interval` uses for
    # exactly this quantity, returns ((a < s).sum() + (a <= s).sum()) / (2B) -- mid-p -- at the
    # pinned 1.16.2, and test_the_midp_z0_convention_agrees_with_the_pinned_scipy pins the
    # agreement. So the rule is neither novel to this project nor Efron's: it matches the
    # reference implementation, and it is named after it here. AND IT IS NOT COSTLESS, which is why
    # it is not called "principled" either. The block above measures 90.0% coverage against 95.3%
    # for the strict count under the guard (the percentile interval wherever no replicate lies below
    # the estimate) on the study's autoPET population, and z0 on a tied median is the mechanism.
    # It is used because neither tie convention is a remedy: strict-< has no value where no
    # replicate lies below the estimate (the guard then gives the percentile interval) and COLLAPSES
    # the interval to (0.0, 0.0) on a small quantized sample where ~95% of replicates tie
    # (test_statlib.py Part 2) -- a worse failure there -- and because the safe alternatives are the
    # ones that never form z0 at all, which is what `CI_METHOD` selects. Directionally the half-ties
    # rule shrinks z0 toward 0, which is the same direction as the "reduced bias correction" of
    # Tibbe TD & Montoya AK, Front Psychol 2022;13:810258 (doi:10.3389/fpsyg.2022.810258, PMID
    # 35712166) -- but theirs is a scalar reduction, not a tie count, and their own conclusion is
    # that where type-I control matters the PERCENTILE interval is the recommended method. Cited as
    # a signpost to that conclusion, never as authority for this formula.
    prop = float(np.mean(boot < theta_hat) + 0.5 * np.mean(boot == theta_hat))
    if prop <= 0.0 or prop >= 1.0:
        z0 = 0.0                      # degenerate -> no bias correction
    else:
        z0 = N.inv_cdf(prop)
    # acceleration a from the jackknife skewness of the estimator. `jack` may be a flat
    # array (one resampling unit set) or a sequence of per-sample jackknives; the two forms
    # coincide for one sample.
    a = bca_acceleration(jack)
    zlo, zhi = N.inv_cdf(alpha / 2), N.inv_cdf(1 - alpha / 2)

    def adj(zq):
        d = 1.0 - a * (z0 + zq)
        if d == 0.0:                      # ratio diverges; clamp to the side's limit
            return 0.0 if zq < 0 else 1.0
        return N.cdf(z0 + (z0 + zq) / d)

    a1, a2 = adj(zlo), adj(zhi)
    a1 = min(max(a1, 0.0), 1.0)
    a2 = min(max(a2, 0.0), 1.0)
    return float(np.percentile(boot, 100 * a1)), float(np.percentile(boot, 100 * a2))


def bca_ci(v, stat_fn=np.median, alpha: float = 0.05,
           n_boot: int = DEFAULT_NBOOT, seed: int = DEFAULT_SEED):
    """BCa bootstrap CI for a 1-D-sample statistic. Degenerate/empty inputs
    behave like bootstrap_ci; falls back to percentile if BCa is undefined."""
    a = _clean(v)
    if a.size == 0:
        return float("nan"), float("nan")
    if a.size == 1 or np.ptp(a) == 0:
        return float(a[0]), float(a[0])
    theta = float(stat_fn(a))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boot = np.array([float(stat_fn(a[row])) for row in idx])
    jack = np.array([float(stat_fn(np.delete(a, i))) for i in range(a.size)])
    ep = _bca_endpoints(theta, boot, jack, alpha)
    if ep is None:
        lo = float(np.percentile(boot, 100 * alpha / 2))
        hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
        return lo, hi
    return ep


def cluster_bca_ci(values, clusters, stat_fn=np.median, alpha: float = 0.05,
                   n_boot: int = DEFAULT_NBOOT, seed: int = DEFAULT_SEED):
    """BCa CI that resamples CLUSTERS, not rows.

    `bca_ci` resamples rows and delete-one-ROW jackknifes, which assumes the rows are
    independent. When several rows come from one subject they are not, and the interval is
    too narrow: the design effect is `1 + (m_kish - 1) * ICC`, so the shortfall is bounded by
    how large the clusters are, not by how strong the correlation is.

    Here the bootstrap draws whole clusters with replacement and the jackknife deletes whole
    clusters, so the resampling unit and the jackknife unit match.

    `clusters` is a sequence of labels parallel to `values`. Rows whose value is non-finite
    are dropped with their label. With one cluster, or fewer than two usable values, this
    degrades exactly like `bca_ci`. Returns (lo, hi).

    NOTE ON THE ESTIMAND: the statistic is computed over the POOLED rows of the resampled
    clusters, so a subject contributing five studies contributes five rows -- the same
    estimand the study-level primary reports, with only the *uncertainty* made
    cluster-aware. Weighting each subject equally instead is a different estimand and a
    scientific choice, not a bug fix; it is deliberately not made here.
    """
    v = np.asarray(values, float).ravel()
    lab = np.asarray(clusters).ravel()
    if v.size != lab.size:
        raise ValueError(f"values and clusters differ in length: {v.size} vs {lab.size}")
    ok = np.isfinite(v)
    v, lab = v[ok], lab[ok]
    if v.size == 0:
        return float("nan"), float("nan")
    if v.size == 1 or np.ptp(v) == 0:
        return float(v[0]), float(v[0])

    groups = [v[lab == u] for u in pd_unique(lab)]
    nc = len(groups)
    if nc < 2:
        return bca_ci(v, stat_fn, alpha, n_boot, seed)

    theta = float(stat_fn(v))
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, nc, nc)
        boot[b] = float(stat_fn(np.concatenate([groups[i] for i in idx])))
    jack = np.array([float(stat_fn(np.concatenate([groups[j] for j in range(nc) if j != i])))
                     for i in range(nc)])
    ep = _bca_endpoints(theta, boot, jack, alpha)
    if ep is None:
        return (float(np.percentile(boot, 100 * alpha / 2)),
                float(np.percentile(boot, 100 * (1 - alpha / 2))))
    return ep


def pd_unique(labels):
    """Order-preserving unique, so cluster order is a property of the data, not of a sort.

    `np.unique` sorts, which would make the bootstrap stream depend on label spelling. This
    keeps first-appearance order so the seeded stream depends only on row order.
    """
    seen, out = set(), []
    for x in labels:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def design_effect(values, clusters):
    """(ICC, Kish mean cluster size, DEFF) for a clustered sample.

    ICC by the one-way random-effects ANOVA estimator; Kish's mean cluster size
    `sum(n_j^2)/sum(n_j)` because unequal clusters inflate the design effect more than the
    arithmetic mean suggests; `DEFF = 1 + (m_kish - 1) * ICC`, the factor by which an iid
    variance understates the clustered one. Reported so the size of the dependency is a
    number in an artifact rather than an adjective in prose. Returns NaNs when undefined.
    """
    v = np.asarray(values, float).ravel()
    lab = np.asarray(clusters).ravel()
    ok = np.isfinite(v)
    v, lab = v[ok], lab[ok]
    groups = [v[lab == u] for u in pd_unique(lab)]
    groups = [g for g in groups if g.size]
    k, n = len(groups), v.size
    nan = float("nan")
    if k < 1:
        return nan, nan, nan
    sizes = np.array([g.size for g in groups], float)
    if k < 2 or n <= k:
        # No within-cluster variance to estimate, so the ICC is genuinely undefined. The
        # Kish mean is not, and when every cluster is a singleton DEFF is exactly 1 for any
        # ICC -- returning NaN there would report "unknown" for a quantity that is known.
        m_kish = float((sizes ** 2).sum() / sizes.sum())
        return nan, m_kish, (1.0 if m_kish == 1.0 else nan)
    grand = v.mean()
    ms_between = float(sum(g.size * (g.mean() - grand) ** 2 for g in groups)) / (k - 1)
    ms_within = float(sum(((g - g.mean()) ** 2).sum() for g in groups)) / (n - k)
    # ANOVA ICC uses n0, the "average" cluster size that makes E[MSB] linear in sigma_b^2
    n0 = (n - (sizes ** 2).sum() / n) / (k - 1)
    denom = ms_between + (n0 - 1) * ms_within
    icc = 0.0 if denom == 0 else (ms_between - ms_within) / denom
    icc = float(min(max(icc, 0.0), 1.0))
    m_kish = float((sizes ** 2).sum() / sizes.sum())
    return icc, m_kish, float(1.0 + (m_kish - 1.0) * icc)


def median_ci_bca(v, alpha: float = 0.05, n_boot: int = DEFAULT_NBOOT,
                  seed: int = DEFAULT_SEED):
    return bca_ci(v, np.median, alpha, n_boot, seed)


def median_ci_auto(v, method: str = "bca", alpha: float = 0.05,
                   n_boot: int = DEFAULT_NBOOT, seed: int = DEFAULT_SEED):
    """Dispatch median CI on method ('bca' | 'percentile').

    An unrecognised `method` raises rather than silently selecting BCa.
    This is the entry point every Phase-1 median CI goes through with
    `gap_config.CI_METHOD`, so it is where a mistyped constant has to stop.
    """
    _check_ci_method(method, "method")
    if method == "percentile":
        return median_ci(v, alpha, n_boot, seed)
    return median_ci_bca(v, alpha, n_boot, seed)


# ---------------------------------------------------------------------------
# Linear-fit x-intercept with bootstrap CI (bifurcation apex)
# ---------------------------------------------------------------------------
def _boot_indices(n: int, block, rng) -> np.ndarray:
    """Bootstrap resampling indices. i.i.d. when `block` is falsy / <=1 / >=n;
    otherwise a CIRCULAR moving-block bootstrap: contiguous blocks of length
    `block` are drawn with replacement from every possible start and concatenated
    to length n. Circular wrapping gives boundary slices the same inclusion
    probability as interior slices while preserving local autocorrelation. Assumes
    the data are ordered (e.g. by z)."""
    if not block or block <= 1 or block >= n:
        return rng.integers(0, n, n)
    n_blocks = -(-n // block)                       # ceil(n / block)
    starts = rng.integers(0, n, n_blocks)
    return np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]


def _jackknife_delete_block(n: int, block):
    """Yield kept-index arrays for a CIRCULAR DELETE-BLOCK jackknife (Kunsch 1989),
    matching the circular moving-block bootstrap: each replicate removes one
    contiguous, possibly wrapping block from the z-ordered data. Falls back to the
    ordinary delete-one jackknife when `block` is falsy / <=1 / >=n.

    Using a delete-one jackknife to estimate the BCa acceleration while the bootstrap
    itself resamples blocks is internally incoherent (the acceleration then assumes
    i.i.d. data, the exact assumption the block bootstrap abandons); the delete-block
    jackknife restores that coherence for autocorrelated slice data.

    Making the acceleration match the resampling scheme removes a definite internal contradiction.
    It does NOT buy a coverage guarantee: BCa's second-order accuracy is a theorem for smooth
    statistics under i.i.d. sampling, and there is no theory establishing that a block-jackknife
    acceleration restores it under dependence. The recognised route to accuracy for dependent data
    is STUDENTIZATION (percentile-t; Goetze & Kuensch 1996), which this module does not implement.
    So the block-BCa acceleration here is a coherent HEURISTIC, and every interval built on it is
    reported as indicative, never as a calibrated 95% interval (see `BIF_BLOCK_SLICES`).
    Construction: Kuensch 1989 for the delete-block jackknife and the
    moving-block bootstrap it matches; Liu & Singh 1992 for the block jackknife's use in
    variance/skewness estimation under dependence.

    When fewer than two observations would remain, the block jackknife cannot fit a
    line; that degenerate regime falls back to ordinary delete-one."""
    full = np.arange(n)
    if not block or block <= 1 or block >= n or (n - block) < 2:
        for i in range(n):
            yield np.delete(full, i)
    else:
        for s in range(n):
            removed = {(s + j) % n for j in range(block)}
            yield np.array([i for i in full if i not in removed], int)


def wall_contact_apex(z, sep, sumr, alpha: float = 0.05,
                      n_boot: int = DEFAULT_NBOOT, seed: int = DEFAULT_SEED,
                      slope_tol: float = 1e-6, ci_method: str = "bca",
                      block: int = None) -> dict:
    """Wall-contact iliac confluence apex: the z where centroid separation sep(z)
    equals the contact distance R = median(`sumr`). Point estimate fits sep~z by OLS
    and solves z_bif = (R - intercept)/slope (R is stable, low-variance).

    NOTE on `sumr` (the contact-distance source array): this engine is GENERIC in R. The parameter
    `sumr` means 'contact distance per slice', not specifically the sum of radii.

    CI is BCa by default (jackknife acceleration over points). Returns dict(z_bif, slope,
    intercept, slope_se, resid_se, n, ci_lo, ci_hi, unstable, ci_method, median_r), plus
    nonpositive_R when the contact level is not positive.

    CONTACT-LEVEL GUARD: walls contact at a POSITIVE separation, so a non-positive contact level R =
    median(sumr) <= 0 is physically impossible (it arises from fragmented/concave masks where
    nearest-surface points sit farther apart than the centroids on >=50% of window slices) and
    solving sep(z) = R there puts the apex on the WRONG (cranial) side of centroid coincidence.
    Bootstrap replicates whose OWN median level is non-positive are dropped exactly like flat-slope
    replicates: a root at an impossible contact level is not a valid draw of the estimand."""
    _check_ci_method(ci_method)
    z = np.asarray(z, float).ravel()
    sep = np.asarray(sep, float).ravel()
    sumr = np.asarray(sumr, float).ravel()
    ok = ~(np.isnan(z) | np.isnan(sep) | np.isnan(sumr))
    z, sep, sumr = z[ok], sep[ok], sumr[ok]
    order = np.argsort(z, kind="stable")            # order by z for block bootstrap
    z, sep, sumr = z[order], sep[order], sumr[order]
    n = z.size
    out = {"z_bif": float("nan"), "slope": float("nan"), "intercept": float("nan"),
           "slope_se": float("nan"), "resid_se": float("nan"), "n": int(n),
           "ci_lo": float("nan"), "ci_hi": float("nan"), "unstable": True,
           "ci_method": ci_method, "median_r": float("nan"),
           "nonpositive_R": False}
    if n < 2 or np.ptp(z) == 0:
        return out

    def _root(zz, ss, rr):
        if np.ptp(zz) == 0:
            return None
        level = float(np.median(rr))
        if level <= 0:                 # impossible contact level (see docstring)
            return None
        slope, intercept = np.polyfit(zz, ss, 1)
        if abs(slope) <= slope_tol:
            return None
        return (level - intercept) / slope

    slope, intercept = np.polyfit(z, sep, 1)
    out["slope"], out["intercept"] = float(slope), float(intercept)
    out["median_r"] = float(np.median(sumr))
    if n > 2:
        resid = sep - (slope * z + intercept)
        sxx = float(((z - z.mean()) ** 2).sum())
        s2 = float((resid ** 2).sum()) / (n - 2)
        out["resid_se"] = math.sqrt(s2)
        out["slope_se"] = math.sqrt(s2 / sxx) if sxx > 0 else float("nan")
    unstable = abs(slope) <= slope_tol
    if out["median_r"] <= 0:
        out["nonpositive_R"] = True
        unstable = True
    if n < 3:                          # 2-point exact fit: no df to test slope significance
        unstable = True
    # flat-slope guard (t_{n-2}). NOTE: this is
    # deliberately a TWO-SIDED "distinguishable from zero" screen (the near-0/0 ratio
    # explodes symmetrically in the slope sign); the caller's decide_flag rejects
    # non-negative slopes separately, so the composite "significantly negative"
    # convergence test runs at an effective ONE-SIDED level of alpha/2 (2.5%), not alpha.
    se = out["slope_se"]
    if (not unstable) and isinstance(se, float) and math.isfinite(se) and se > 0 \
            and abs(slope) / se < t_crit(alpha, n - 2):
        unstable = True
    theta = ((out["median_r"] - intercept) / slope) if not unstable else float("nan")
    out["z_bif"] = theta

    rng = np.random.default_rng(seed)
    roots = []
    for _ in range(n_boot):
        bidx = _boot_indices(n, block, rng)
        r = _root(z[bidx], sep[bidx], sumr[bidx])
        if r is not None:
            roots.append(r)
    roots = np.asarray(roots, float)
    if roots.size >= 2 and not unstable:
        if ci_method == "bca":
            jack = []
            for keep in _jackknife_delete_block(n, block):    # delete-block to match block bootstrap
                r = _root(z[keep], sep[keep], sumr[keep])
                if r is not None:
                    jack.append(r)
            ep = _bca_endpoints(theta, roots, np.asarray(jack, float), alpha)
            if ep is None:
                out["ci_lo"] = float(np.percentile(roots, 100 * alpha / 2))
                out["ci_hi"] = float(np.percentile(roots, 100 * (1 - alpha / 2)))
                out["ci_method"] = "percentile"
            else:
                out["ci_lo"], out["ci_hi"] = ep
        else:
            out["ci_lo"] = float(np.percentile(roots, 100 * alpha / 2))
            out["ci_hi"] = float(np.percentile(roots, 100 * (1 - alpha / 2)))
    out["unstable"] = bool(unstable or roots.size < 2)
    return out


# ---------------------------------------------------------------------------
# Curvature / linearity diagnostic (for the apex convergence fit)
# ---------------------------------------------------------------------------
def quadratic_significance(x, y, alpha: float = 0.05) -> dict:
    """Test whether a quadratic term is needed, i.e. whether y(x) is non-linear.
    Fits y = a*x^2 + b*x + c by OLS and t-tests the quadratic coefficient a.
    Returns dict(quad_coef, quad_se, t, p, n, significant, status, error_type).
    `significant` (p<alpha) means the linear extrapolation the apex relies on is biased.
    It is ``None`` when curvature is not assessable; that state must never be interpreted
    as evidence of linearity."""
    import warnings

    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    # Non-finite coordinates are absent observations. Passing infinity into ``polyfit`` can
    # produce a warning-dependent NaN covariance or platform-specific LAPACK failure; neither
    # is evidence that the relation is linear.
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    out = {"quad_coef": float("nan"), "quad_se": float("nan"), "t": float("nan"),
           "p": None, "n": int(n), "significant": None,
           "status": "insufficient_data", "error_type": ""}
    if n < 5 or np.ptp(x) == 0:
        return out
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            coef, cov = np.polyfit(x, y, 2, cov=True)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError, Warning) as exc:
        out["status"] = "fit_failed"
        out["error_type"] = type(exc).__name__
        return out
    quad = float(coef[0])
    # Roundoff guard: a numerically-flat curve (e.g. an exact line) leaves ~1e-16
    # residuals, so the t-test would run on floating-point noise and spuriously
    # "detect" curvature. If the quadratic term's contribution across the x-span is
    # negligible vs the y-range, there is no curvature to test.
    q_contrib = abs(quad) * float(np.ptp(x)) ** 2
    if q_contrib <= 1e-6 * max(float(np.ptp(y)), 1.0):
        out.update(quad_coef=quad, significant=False, status="ok")
        return out
    var = float(cov[0, 0])
    if not (math.isfinite(var) and var > 0):
        out["status"] = "fit_failed"
        out["error_type"] = "INVALID_COVARIANCE"
        return out
    se = math.sqrt(var)
    t = quad / se
    df = n - 3
    # two-sided p from the Student-t tail (pure stdlib t_sf): the quadratic
    # coefficient's statistic is t_{n-3}. A NORMAL tail would understate p at low df and
    # could spuriously flag curvature.
    p = float(2.0 * t_sf(abs(t), df))
    out.update(quad_coef=quad, quad_se=se, t=float(t), p=p,
               significant=bool(p < alpha), status="ok")
    return out


# ---------------------------------------------------------------------------
# Method / reference agreement bundle (apex cross-check and reference comparison)
# ---------------------------------------------------------------------------
def method_agreement(a, b, alpha: float = 0.05, tol: float = None) -> dict:
    """Agreement of two paired measurements a vs b (e.g. an apex estimate vs a
    vertebral reference, or wallgap vs tangency apex): bias + parametric &
    non-parametric 95% LoA (Bland-Altman), Lin's CCC, ICC(A,1), and — if `tol` is
    given — the fraction within tolerance |a-b|<=tol.

    `tol` is the CALLER'S number and this function attaches no meaning to it. No source establishes
    such a tolerance for a venous landmark, and the nearby real convention — automated vertebra
    LABELING accuracy "within one vertebral level" — measures whether a vertebra was named
    correctly, which is a different question. A generic agreement helper is the wrong place to
    assert an anatomical standard at all."""
    pair = _clean_pair(a, b)
    if pair is None:
        return {"n": 0, "bias": float("nan"), "ccc": float("nan"),
                "icc_a1": float("nan")}
    aa, bb = pair
    ba = bland_altman(aa, bb, alpha)
    out = {"n": int(aa.size), "bias": ba["bias"],
           "loa_lo": ba["loa_lo"], "loa_hi": ba["loa_hi"],
           "loa_lo_np": ba["loa_lo_np"], "loa_hi_np": ba["loa_hi_np"],
           "ccc": lin_ccc(aa, bb), "icc_a1": icc_a1(aa, bb)}
    if tol is not None:
        out["within_tol_frac"] = float(np.mean(np.abs(aa - bb) <= tol))
        out["tol"] = float(tol)
    return out


# ---------------------------------------------------------------------------
# One-stop robust description of a distribution
# ---------------------------------------------------------------------------
def describe(v, alpha: float = 0.05, n_boot: int = DEFAULT_NBOOT,
             seed: int = DEFAULT_SEED, ci_method: str = "bca") -> dict:
    """Robust + parametric summary with a bootstrap CI on the median and a
    normality flag. Leads with median/IQR (appropriate for skewed gaps). The
    median CI defaults to BCa (second-order accurate under skew for a CONTINUOUS
    statistic; see `bca_ci`'s note on lattice-valued medians); pass
    ci_method='percentile' for the first-order interval.

    An unrecognised `ci_method` raises. The check is deliberately made
    BEFORE the empty-input early return, not left to the `median_ci_auto` delegation
    below: otherwise a typo would be caught for a populated cohort and pass silently
    for an empty one, which is the worst possible place to lose it. Note that
    `median_ci_method` is echoed straight into the returned dict, so without the check
    this function could report an estimator that never ran."""
    _check_ci_method(ci_method)
    a = _clean(v)
    n = a.size
    if n == 0:
        return {"n": 0}
    med_lo, med_hi = median_ci_auto(a, ci_method, alpha, n_boot, seed)
    return {
        "n": int(n),
        "median": float(np.median(a)),
        "median_ci_lo": med_lo, "median_ci_hi": med_hi,
        "median_ci_method": ci_method,
        "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)),
        "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
        "mean": float(a.mean()), "sd": sample_sd(a),
        "min": float(a.min()), "max": float(a.max()),
        "skew": skewness(a), "shapiro_p": shapiro_p(a),
    }
