#!/usr/bin/env python3
"""Reads the per-case results produced by measure_all.py / measure_ivc_lciv_gap.py
plus the cohort metadata, and emits, per cohort (paths under Results/phase1_characterize/):

  <DIR>/all_stats.csv                distribution stats (median+95%CI, IQR, mean,
                                     sample-SD, Wilson CIs for proportions, skew,
                                     normality p, censoring + QC accounting)
  <DIR>/<prefix>_correlates.csv      gap vs sex/scanner/age/contrast/region,
                                     Kruskal-Wallis with epsilon-squared effect
                                     size and BH-FDR q-values across the family
  autoPET/autopet_subject_sensitivity.csv
                                     deterministic earliest-study-per-subject
                                     sensitivity analysis (one row/unit; n=900)
  <DIR>/<prefix>_agreement.csv       axial-vs-3D and fast-vs-fullres method
                                     agreement: Lin's CCC, ICC(A,1), Bland-Altman
  <DIR>/provenance.txt               versions, seed, and every version-controlled
                                     threshold used (gap_config.py)
  <DIR>/figures/*.png                distribution, axial-vs-3D, ECDF, correlates,
                                     Bland-Altman
  (console)                          full summary with CIs + cohort-flow
                                     exclusion accounting
  _cross_cohort.txt (all)            between-cohort heterogeneity test

PRIMARY OUTCOME is the 3D nearest SURFACE-to-surface distance. It is invariant under any orthonormal
image direction (signed axis permutations, reversed-z AND genuine tilt: a rigid rotation is an
isometry, so a tilted acquisition yields the same gap) but NOT uncensored: it is floored at 0
(touching/ overlapping masks read 0), so it is zero-inflated / left-censored at the resolution floor
and, for axial-dominated gaps, still quantizes to integer multiples of the per-case slice thickness
— hence the slice-normalized and within-slice-thickness analyses below (the cross-cohort mm
difference attenuates strongly after scaling by slice thickness, without isolating a causal
mechanism). The axial z_gap is a SECONDARY, left-censored descriptor, summarized but never used for
parametric inference. All thresholds live in gap_config.py and are written to provenance.txt for
audit. Bootstrap CIs use a fixed seed (gap_config.SEED) so published numbers are reproducible.

Statistical conventions (see statlib.py, which is unit-tested):
  * dispersion = sample SD (ddof=1); proportions = Wilson score 95% CI;
    medians = BCa bootstrap 95% CI (gap_config.CI_METHOD); multiplicity =
    Benjamini-Hochberg FDR across the per-cohort correlate family (bio/tech families
    kept separate); effect size = epsilon-squared (reported with a bootstrap CI when
    the magnitude label is near a Cohen/Tomczak cutpoint).
  * Per-level pairwise comparisons are NOT performed (omnibus KW only); reading
    the per-level medians as pairwise tests would require Dunn's test with its
    own correction.

Usage:
  python3 stats.py amos
  python3 stats.py all --figures
  python3 stats.py saros --no-correlates

Requires numpy + pandas. scipy and matplotlib are optional (imported lazily for
the KW test/Shapiro/KDE and for --figures); the CSV/console core runs without
them, degrading gracefully (e.g. p-values reported as NA).

Metric definitions (applied identically to every cohort):
  gap_present_pct        % of measurable cases with gap > PRESENT_MM (0.5)
  band_2p5_7p5_pct       % with BAND_LO_MM <= gap <= BAND_HI_MM
  n_gt_30mm              count with gap > LARGE_GAP_MM
  sd_mm                  SAMPLE SD (ddof=1)
  axial_pct/lateral_pct  scale-invariant split on the lateral leg (gap_config)
  median_axial_fraction  median of per-case z_gap/centre-to-centre distance over cases with
                         a resolvable gap (NOT z_gap/surface_gap: the surface gap is a
                         nearest-point distance and can be smaller than the axial extent)
  median_lateral_residual median sqrt(max(0, gap3d^2 - z_gap^2)) (UPPER bound)
"""
import argparse
import hashlib
import math
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# --- run-from-anywhere: put sibling phase dirs on sys.path (tests use Code/conftest.py) ---
import sys as _sys, pathlib as _pl
for _p in sorted(_pl.Path(__file__).resolve().parent.parent.iterdir()):
    if _p.is_dir() and _p.name[0] not in "._":
        if str(_p) not in _sys.path:
            _sys.path.append(str(_p))  # append, never shadow stdlib
import gap_config as C
import statlib as S
import figio as _FIGIO  # deterministic, atomic figure writes

# The repository root is two parents up from Code/phase1_characterize/.
BASE = Path(__file__).resolve().parent
REPO_ROOT = BASE.parent.parent
META_DIR = REPO_ROOT / "METADATA"
RESULTS = REPO_ROOT / "Results" / "phase1_characterize"
PHASE0 = BASE.parent / "phase0_data"

# dataset key -> (results dir, file prefix, metadata csv)
LAYOUT = {
    "amos": ("AMOS", "amos", "AMOS.csv"),
    "saros": ("SAROS", "saros", "SAROS.csv"),
    "autopet": ("autoPET", "autopet", "autoPET.csv"),
}

# Cohorts with an exhaustive study roster. Validate case identity before computing any
# statistic, so a table whose row count matches the roster but whose case ids differ cannot
# pass.
EXACT_ROSTERS = {
    "amos": PHASE0 / "amos_ct_cases.txt",
    "saros": PHASE0 / "saros_processed_cases.txt",
    "autopet": PHASE0 / "autopet_cases.txt",
}


def validate_result_roster(dataset, result_path):
    """Fail closed when an exhaustive result table and its roster disagree.

    Returns a compact accounting dict on success. SAROS uses its exact 392-case
    processed roster. The 36 cases of the 428-case eligible set that are not analysed (30
    processed and then excluded on confluence eligibility, and 6 absent from the mask export)
    are not substituted into result validation.
    """
    roster = EXACT_ROSTERS.get(dataset)
    if roster is None:
        return None
    expected = [ln.split()[0] for ln in roster.read_text().splitlines() if ln.split()]
    actual = []
    for ln in Path(result_path).read_text().splitlines():
        tok = ln.split()
        if tok and _looks_like_case_id(tok[0]):
            actual.append(tok[0])

    dup_expected = sorted({x for x in expected if expected.count(x) > 1})
    dup_actual = sorted({x for x in actual if actual.count(x) > 1})
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if (len(actual) != len(expected) or dup_expected or dup_actual or missing or unexpected):
        raise RuntimeError(
            f"{dataset} result/roster identity mismatch: expected={len(expected)} "
            f"actual={len(actual)} missing={missing[:10]} unexpected={unexpected[:10]} "
            f"duplicate_roster={dup_expected[:10]} duplicate_results={dup_actual[:10]}. "
            "Regenerate the cohort table before reporting statistics."
        )
    return {"expected": len(expected), "actual": len(actual)}

STATS_HEADER = [
    "metric", "source", "n_total", "n_measurable", "n_empty", "n_unparsed",
    "empty_rate_pct", "censored_at_0_pct", "qc_flagged",
    "gap_present_pct", "gap_present_ci",
    "median_mm", "median_ci", "p25_mm", "p75_mm", "iqr_mm",
    "mean_mm", "sd_mm", "skew", "normality_p",
    "p10_mm", "p90_mm", "min_mm", "max_mm",
    "band_2p5_7p5_pct", "band_ci", "n_gt_30mm",
    "axial_pct", "lateral_pct", "mixed_pct",
    "median_axial_fraction", "median_lateral_residual_mm",
]

# Collapsed-class diagnostic CSV (<prefix>_collapsed_diag.csv). One row per cohort; a SEPARATE
# file from all_stats.csv (whose 32-col / 5-row shape is contractual). See collapsed_class_stats.
COLLAPSED_HEADER = [
    "dataset", "mode", "n_paired", "n_empty", "n_failed", "n_unparsed",
    "failure_reasons",
    "median_orig_mm", "median_orig_ci", "median_collapsed_mm", "median_collapsed_ci",
    "median_paired_delta_mm", "paired_delta_ci",
    "paired_delta_atom_pct", "pct_pairs_changed", "pct_pairs_changed_ci",
    "pct_pairs_shrank", "pct_pairs_shrank_ci", "pct_pairs_grew", "pct_pairs_grew_ci",
    "present_orig_pct", "present_orig_ci", "present_collapsed_pct", "present_collapsed_ci",
    "mcnemar_b_fixed", "mcnemar_c_created", "mcnemar_p", "mcnemar_method",
    "union_connected_pct", "union_connected_ci",
    "resolved_endpoint_available_n", "resolved_endpoint_status",
    "resolved_pct", "resolved_ci", "topology_ok_pct",
    "topology_ok_present_n", "topology_ok_resolved_n",
    "topology_ok_resolved_pct", "topology_ok_resolved_ci",
    "multicomponent_present_n", "multicomponent_resolved_n",
    "multicomponent_resolved_pct", "multicomponent_resolved_ci",
]


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def provenance() -> dict:
    try:
        import scipy
        scipy_v = scipy.__version__
    except Exception:
        scipy_v = "absent"
    thresholds = {k: getattr(C, k) for k in dir(C) if k.isupper()}
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy_v,
        "seed": C.SEED, "n_boot": C.N_BOOT, "alpha": C.ALPHA,
        "thresholds": thresholds,
    }


def _analysis_input_paths(root=None):
    """Ordered Phase-1 input files covered by the stable analysis identity."""
    root = Path(root or REPO_ROOT).resolve()
    candidates = []
    for directory in (root / "Code" / "common", root / "Code" / "phase1_characterize"):
        if directory.is_dir():
            candidates.extend(
                p for p in directory.rglob("*")
                if p.is_file() and p.suffix in {".py", ".md"}
            )
    candidates.extend(p for p in (root / "Code" / "phase0_data").glob("*cases.txt")
                      if p.is_file())
    candidates.extend(p for p in (root / "METADATA").glob("*.csv") if p.is_file())
    for cohort in ("AMOS", "SAROS", "autoPET"):
        cohort_dir = root / "Results" / "phase1_characterize" / cohort
        candidates.extend(p for p in cohort_dir.glob("*_results*.txt") if p.is_file())
    env = root / "environment.yml"
    if env.is_file():
        candidates.append(env)

    return sorted(set(candidates), key=lambda p: p.relative_to(root).as_posix())


def _analysis_input_identity(root=None):
    """Content identity for stable Phase-1 analysis inputs, excluding generated outputs.

    Hashing the whole git diff is self-invalidating because ``stats.py`` rewrites timestamped
    provenance and generated result files. This manifest instead covers the code/config,
    cohort rosters/metadata, and committed per-case measurement tables that determine the
    statistics. It deliberately excludes figures, aggregate CSVs, and provenance files.
    """
    root = Path(root or REPO_ROOT).resolve()
    unique = _analysis_input_paths(root)
    digest = hashlib.sha256()
    for path in unique:
        rel = path.relative_to(root).as_posix()
        payload = path.read_bytes()
        digest.update(rel.encode("utf-8") + b"\0")
        digest.update(str(len(payload)).encode("ascii") + b"\0")
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest(), len(unique)


def _git_commit():
    """Commit, dirty state, and stable content identity for the Phase-1 analysis inputs."""
    try:
        import subprocess
        cwd = str(Path(__file__).resolve().parent)
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=5, cwd=cwd, check=True).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            timeout=5, cwd=cwd, check=True).stdout
        digest, n_files = _analysis_input_identity()
        dirty = "yes" if status.strip() else "no"
        return (f"{commit}; dirty={dirty}; analysis_inputs_sha256={digest}; "
                f"analysis_input_files={n_files}")
    except Exception:
        pass
    return "NOT RECORDED — analysis tree is not a git repository"


def _totalseg_provenance_line(ds):
    """The TotalSegmentator version and resolution that produced this cohort's masks:
    totalseg_provenance.txt when present, else gap_config.TOTALSEG_VERSION."""
    if ds and ds in LAYOUT:
        f = RESULTS / LAYOUT[ds][0] / "totalseg_provenance.txt"
        if f.exists():
            return f.read_text().strip().replace("\n", " | ")
    fallback = f"TotalSegmentator {C.TOTALSEG_VERSION}"
    return fallback


def write_provenance(path: Path, ds=None, aux=None):
    p = provenance()
    lines = ["# IVC-LCIV gap analysis provenance",
             f"generated_utc : {p['generated_utc']}",
             f"python        : {p['python']}",
             f"numpy         : {p['numpy']}",
             f"pandas        : {p['pandas']}",
             f"scipy         : {p['scipy']}",
             f"seed          : {p['seed']}   n_boot: {p['n_boot']}   alpha: {p['alpha']}",
             f"primary       : {C.PRIMARY_INFERENCE}",
             "                (gap_config.PRIMARY_INFERENCE)"]
    if ds and ds in C.COHORT_ELIGIBLE:
        label, n_elig, note = C.COHORT_ELIGIBLE[ds]
        lines.append(f"cohort        : eligible {label} = {n_elig}")
        lines.append(f"  selection   : {note}")
    lines.append(f"totalseg      : {_totalseg_provenance_line(ds)}")
    if aux is not None:
        lines.append(f"aux_correlates: {aux_correlate_availability(aux)}")
    if ds:
        lines.append(f"dataset       : {C.DATASET_VERSION.get(ds, 'NOT RECORDED')}")
    lines.append(f"git_commit    : {_git_commit()}")
    lines.append("version-controlled analysis settings (gap_config.py):")
    for k, v in sorted(p["thresholds"].items()):
        lines.append(f"  {k} = {v}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Metadata (normalized to a common, results-keyed schema)
# ---------------------------------------------------------------------------
def _age(s):
    if not s:
        return None
    m = re.match(r"0*(\d+)", str(s).strip())
    return int(m.group(1)) if m else None


def _sex(s):
    if not s:
        return None
    s = str(s).strip().upper()
    return "M" if s.startswith("M") else "F" if s.startswith("F") else None


def _scanner(manufacturer):
    if not manufacturer:
        return None
    return manufacturer.strip().split()[0].title()


def age_band(a):
    if a is None:
        return None
    for cut, label in C.AGE_BANDS:
        if cut is None or a < cut:
            return label
    return C.AGE_BANDS[-1][1]


def _meta_amos(path):
    import csv
    out, skipped, conflicts = {}, 0, set()
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                cid = f"amos_{int(r['amos_id']):04d}"
            except (KeyError, ValueError):
                skipped += 1
                continue
            rec = dict(sex=_sex(r.get("Patient's Sex")),
                       age=_age(r.get("Patient's Age")),
                       scanner=_scanner(r.get("Manufacturer")),
                       contrast=None, region=None)
            if cid not in out:
                out[cid] = rec
            elif out[cid] != rec:
                # Defensive raw-input path: the immutable upstream snapshot has amos_id=500
                # twice for distinct patients (the normalized analysis CSV does not). Last-row wins
                # is CSV-ORDER-DEPENDENT and non-reproducible, so the ambiguous case is
                # BLANKED for the demographic joins (sex/age/scanner) rather than silently
                # mis-joined; its mask-derived seg_quality/size_band still apply. Flagged.
                conflicts.add(cid)
                out[cid] = dict(sex=None, age=None, scanner=None,
                                contrast=None, region=None)
    if skipped:
        print(f"  [meta amos: {skipped} rows had no parseable amos_id -> not joinable]")
    if conflicts:
        print(f"  [meta amos: {len(conflicts)} duplicated amos_id(s) with CONFLICTING "
              f"demographics -> blanked (order-independent, not mis-joined): "
              f"{', '.join(sorted(conflicts))}]")
    return out


def _meta_saros(path):
    import csv
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            cid = (r.get("id") or "").strip()
            if cid:
                if cid in out:
                    raise ValueError(f"duplicate SAROS metadata case: {cid}")
                collection = (r.get("tcia_collection") or "").strip()
                patient = (r.get("tcia_case_id") or "").strip()
                out[cid] = dict(sex=_sex(r.get("gender")), age=None, scanner=None,
                                contrast=None, region=(r.get("anatomic_region") or None),
                                participant=(f"{collection}:{patient}"
                                             if collection and patient else None))
    return out


def participant_key(dataset, case_id, meta):
    """Analysis participant key; fail rather than treat an unknown SAROS patient as unique.

    TCIA subject identifiers are collection-scoped. AMOS has no released cross-case
    patient linkage in this analysis, so its case key remains a declared proxy.
    """
    if dataset == "autopet":
        return _autopet_subject_id(case_id)
    if dataset == "saros":
        participant = meta.get(case_id, {}).get("participant")
        if not participant:
            raise ValueError(f"missing SAROS participant mapping: {case_id}")
        return participant
    if dataset == "amos":
        return case_id
    raise ValueError(f"unknown dataset: {dataset}")


def _is_calendar_date(year, month, day) -> bool:
    """True iff (year, month, day) is a date that exists.

    `datetime.date` is the oracle rather than a hand-written month table, so leap years and the
    30/31-day months are right by construction.
    """
    from datetime import date as _date
    try:
        _date(int(year), int(month), int(day))
    except (ValueError, TypeError, OverflowError):
        return False
    return True


class _SubjectKeyedMeta(dict):
    """autoPET result case-ids are full study-folder names
    (``PETCT_<hash>_<study-date>-NA-...``), but the TCIA manifest keys on the bare
    ``PETCT_<hash>`` Subject ID. This dict stores one PATIENT-level record per subject
    (sex/age/scanner/region) and normalizes every lookup key down to that subject prefix,
    so ``cid in meta`` / ``meta.get(cid)`` join sites accept either id form.

    A lookup returns a COPY of the merged record, so a caller can never mutate the shared subject
    record (this is why per-study writes go through ``set_study``, not the copy)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._study = {}              # {full_study_id: {field: value}} study-level overrides
        self._contrast_by_study = {}  # {(subject, (m, d, y)): 'with'|'without'} from the CSV

    @staticmethod
    def _subj(k):
        # One pattern, defined once below.
        m = _AUTOPET_SUBJECT_RE.match(str(k))
        return m.group(1) if m else str(k)

    @staticmethod
    def _study_date(k):
        """(month, day, year) parsed from a full study-folder id, or None."""
        m = _AUTOPET_CASE_DATE_RE.match(str(k))
        if not m:
            return None
        mm, dd, yyyy = (int(x) for x in m.groups())
        return (mm, dd, yyyy) if _is_calendar_date(yyyy, mm, dd) else None

    def __contains__(self, k):
        return super().__contains__(self._subj(k))

    def get(self, k, default=None):
        base = super().get(self._subj(k), None)
        if base is None:
            return default
        rec = dict(base)                                # copy: never mutate the shared record
        d = self._study_date(k)
        if d is not None:
            c = self._contrast_by_study.get((self._subj(k), d))
            if c is not None:
                rec["contrast"] = c                     # per-study contrast overrides subject
        else:
            # A bare subject id (the earliest-study-per-subject key, as cohort_table1_row
            # uses for demographics) takes the contrast of that subject's earliest study.
            candidates = [(date, contrast)
                          for (subject, date), contrast in self._contrast_by_study.items()
                          if subject == self._subj(k)]
            if candidates:
                rec["contrast"] = min(
                    candidates,
                    key=lambda item: (item[0][2], item[0][0], item[0][1]))[1]
        rec.update(self._study.get(str(k), {}))         # per-study seg_quality / size_band
        return rec

    def __getitem__(self, k):
        r = self.get(k, None)
        if r is None:
            raise KeyError(k)
        return r

    def set_study(self, full_id, field, value):
        """Attach a STUDY-level correlate (seg_quality/size_band) to one study id, so a
        subject's studies keep their own value instead of sharing one order-dependent one."""
        self._study.setdefault(str(full_id), {})[field] = value


# Anchored at both ends: an unanchored match would map `PETCT_00fg11` and `PETCT_00fh22` to the
# same key `PETCT_00f`, merging two subjects.
SUBJECT_PATTERN = r"(?<![A-Za-z0-9])(PETCT_[0-9a-fA-F]+)(?=[_/]|$)"
_AUTOPET_SUBJECT_RE = re.compile(SUBJECT_PATTERN)
_AUTOPET_CASE_DATE_RE = re.compile(
    r"PETCT_[0-9a-fA-F]+_(\d{1,2})-(\d{1,2})-(\d{4})"
)


def _autopet_subject_id(case_id):
    """Subject key for a full autoPET study id; unknown ids remain distinct."""
    m = _AUTOPET_SUBJECT_RE.match(str(case_id))
    return m.group(1) if m else str(case_id)


def cluster_robust_rows(fullres_rows):
    """Cluster-robust companions to the autoPET headline intervals.

    autoPET's 1,014 study rows come from 900 subjects: 81 subjects repeat, 114 excess scans,
    largest cluster 5. The primary interval (`bca_ci`) resamples ROWS and delete-one-ROW
    jackknifes, so it treats them as independent. The one-study-per-subject sensitivity is a
    subject-level analysis of a different estimand (the earliest-study subset, n=900).

    This function gives the same estimand, resampled by SUBJECT, beside the iid interval, with
    the design effect that sizes the dependency.

    The proportion rows carry cluster-robust intervals for the same two proportions whose
    Wilson CIs are reported iid.

    Returns a list of dicts (one per metric), or [] when the rows are not clusterable.
    """
    rows = [(str(r[0]), float(r[2])) for r in fullres_rows
            if r[2] is not None and float(r[2]) == float(r[2])]
    if len(rows) < 2:
        return []
    ids = [c for c, _ in rows]
    gaps = np.array([g for _, g in rows], float)
    subjects = [_autopet_subject_id(c) for c in ids]
    n_subj = len(set(subjects))
    if n_subj < 2 or n_subj == len(rows):
        return []                        # nothing repeats: the iid interval IS the answer
    sizes = np.array([subjects.count(s) for s in sorted(set(subjects))], float)

    present = (gaps > C.PRESENT_MM).astype(float)
    band = ((gaps >= C.BAND_LO_MM) & (gaps <= C.BAND_HI_MM)).astype(float)

    def block(metric, values, stat, iid_ci, unit):
        icc, m_kish, deff = S.design_effect(values, subjects)
        clo, chi = S.cluster_bca_ci(values, subjects, stat, C.ALPHA, C.N_BOOT, C.SEED)
        ilo, ihi = iid_ci
        wr = ((chi - clo) / (ihi - ilo)) if (ihi - ilo) > 0 else float("nan")
        return {
            "metric": metric,
            "unit_of_measurement": unit,
            "resampling_unit_reported_as_primary": "study",
            "resampling_unit_here": "subject",
            "n_rows": len(values),
            "n_subjects": n_subj,
            "max_cluster_size": int(sizes.max()),
            "n_repeating_subjects": int((sizes > 1).sum()),
            "icc": round(icc, 4),
            "kish_mean_cluster_size": round(m_kish, 4),
            "design_effect": round(deff, 4),
            "iid_lo": round(ilo, 4), "iid_hi": round(ihi, 4),
            "cluster_lo": round(clo, 4), "cluster_hi": round(chi, 4),
            "width_ratio": round(wr, 4),
        }

    med_iid = S.median_ci_auto(gaps, C.CI_METHOD, C.ALPHA, C.N_BOOT, C.SEED)
    out = [block("3D surface-to-surface (PRIMARY) median", gaps, np.median, med_iid, "mm")]
    for name, vals in (("gap_present_pct", present), ("band_2p5_7p5_pct", band)):
        lo, hi = S.wilson_ci(int(vals.sum()), vals.size, C.ALPHA)
        out.append(block(name, vals, np.mean, (lo, hi), "proportion"))
    return out


def _autopet_case_date(case_id):
    """Chronological ``(year, month, day)`` key; undated cases sort last."""
    m = _AUTOPET_CASE_DATE_RE.match(str(case_id))
    if not m:
        return (9999, 99, 99)
    month, day, year = (int(v) for v in m.groups())
    return (year, month, day)


def autopet_one_study_per_subject(fullres_rows, all_case_ids=None):
    """Deterministically retain the earliest measurable study for each subject.

    The committed Phase-1 autoPET table has 1014 study rows from 900 subjects. This
    function provides the independent-unit sensitivity subset without changing or
    relabeling the study-level primary artifact. When ``all_case_ids`` includes empty
    or unparsed studies, selection occurs before measurability filtering: an empty
    earliest study is not silently replaced by a later measurable acquisition.
    Sorting includes the full case id as a deterministic same-date tie-breaker.
    """
    by_id = {str(row[0]): row for row in fullres_rows}
    ids = list(by_id) if all_case_ids is None else [str(cid) for cid in all_case_ids]
    ordered_ids = sorted(set(ids),
                         key=lambda cid: (_autopet_subject_id(cid),
                                          _autopet_case_date(cid), cid))
    selected_ids, seen = [], set()
    for cid in ordered_ids:
        subject = _autopet_subject_id(cid)
        if subject in seen:
            continue
        seen.add(subject)
        selected_ids.append(cid)
    return [by_id[cid] for cid in selected_ids if cid in by_id]


def amos_adult_sensitivity(fullres_rows, counts, meta, age_cutoff=18):
    """One-row sensitivity artifact restricted to AMOS CT cases aged >= cutoff.

    The released metadata contain six CT participants younger than 18; they are
    exactly the six ids absent from the mask export, so the 493-case primary frame contains none
    and this artifact coincides with it. Selection
    occurs before measurability filtering and reports closed missingness.
    """
    # The roster is read from EXACT_ROSTERS, as validate_result_roster does, so both
    # preconditions use one cohort definition; the table must account for every roster case,
    # closed missingness included.
    roster_path = EXACT_ROSTERS.get("amos")
    if roster_path is None:
        raise RuntimeError("AMOS adult sensitivity needs the exact AMOS roster; none registered")
    expected = [ln.split()[0] for ln in roster_path.read_text().splitlines() if ln.split()]
    table_total = (len(fullres_rows) + int(counts.get("empty", 0))
                   + int(counts.get("unparsed", 0)))
    if table_total != len(expected):
        raise RuntimeError(f"AMOS adult sensitivity requires the exact {len(expected)}-case "
                           f"table named by {roster_path.name}; found {table_total}")
    adult_ids = [cid for cid in expected
                 if (rec := meta.get(cid)) is not None
                 and isinstance(rec.get("age"), (int, float))
                 and rec["age"] >= age_cutoff]
    unknown = [cid for cid in expected
               if meta.get(cid) is None or meta.get(cid).get("age") is None]
    gaps = {row[0]: float(row[2]) for row in fullres_rows}
    values = np.array([gaps[cid] for cid in adult_ids if cid in gaps], float)
    if not adult_ids or values.size == 0:
        raise RuntimeError("AMOS adult sensitivity has no eligible measurable cases")
    missing = sorted(set(adult_ids) - set(gaps))
    present_n = int((values > C.PRESENT_MM).sum())
    pp, plo, phi = S.proportion_pct_ci(present_n, values.size, C.ALPHA)
    mlo, mhi = S.median_ci_auto(values, C.CI_METHOD, C.ALPHA, C.N_BOOT, C.SEED)
    minor_ids = [cid for cid in expected if cid not in adult_ids and cid not in unknown]
    return {
        "analysis": "AMOS_adult_only_sensitivity",
        "age_cutoff_years": age_cutoff,
        "primary_ct_frame_n": len(expected),
        "adult_eligible_n": len(adult_ids),
        "adult_measurable_n": int(values.size),
        "adult_missing_n": len(missing),
        "adult_missing_ids": ";".join(missing),
        "metadata_age_unknown_n": len(unknown),
        "excluded_minor_n": len(minor_ids),
        "excluded_minor_ids": ";".join(minor_ids),
        "median_gap_mm": round(float(np.median(values)), 2),
        "median_gap_ci": _ci(mlo, mhi, 2),
        "gap_present_n": present_n,
        "gap_present_pct": round(pp, 2),
        "gap_present_ci": _ci(plo, phi, 2),
    }


def saros_extreme_tail_sensitivity(fullres_rows, cutoff_mm=200.0):
    """Two-row descriptive sensitivity with and without values above 200 mm.

    This does not delete a QC-valid observation from the primary table. It makes the
    influence of the extreme right tail on location summaries explicit and durable.
    """
    pairs = [(str(row[0]), float(row[2])) for row in fullres_rows]
    excluded = [cid for cid, value in pairs if value > cutoff_mm]
    out = []
    for label, selected in (
            ("all_measurable", pairs),
            (f"exclude_gt_{cutoff_mm:g}mm", [(c, v) for c, v in pairs if v <= cutoff_mm])):
        values = np.array([v for _, v in selected], float)
        if values.size == 0:
            raise RuntimeError("SAROS tail sensitivity has no measurable cases")
        lo, hi = S.median_ci_auto(values, C.CI_METHOD, C.ALPHA, C.N_BOOT, C.SEED)
        present = int((values > C.PRESENT_MM).sum())
        pp, plo, phi = S.proportion_pct_ci(present, values.size, C.ALPHA)
        out.append({
            "analysis": label, "cutoff_mm": cutoff_mm,
            "n": int(values.size), "excluded_n": len(pairs) - len(selected),
            "excluded_ids": ";".join(excluded) if label != "all_measurable" else "",
            "median_gap_mm": round(float(np.median(values)), 2),
            "median_gap_ci": _ci(lo, hi, 2),
            "mean_gap_mm": round(float(np.mean(values)), 2),
            "max_gap_mm": round(float(np.max(values)), 2),
            "gap_present_pct": round(pp, 2), "gap_present_ci": _ci(plo, phi, 2),
        })
    return out


def _contrast_token(desc):
    """Map one German study/series description to a contrast token, if present."""
    desc = str(desc or "").lower()
    if "mit km" in desc or "p.v." in desc or "portalven" in desc:
        return "with"
    if "nativ" in desc or "ohne km" in desc:
        return "without"
    return None


def _autopet_contrast(study_desc, series_desc=""):
    """Contrast phase with the selected CT series taking precedence over study text.

    ``mit KM`` and portal-venous (``p.v.``/``portalven``) map to ``with``;
    ``nativ``/``ohne KM`` map to ``without``. Nine released records describe the
    overall study as ``nativ u. mit KM`` while naming the selected CT series ``p.v.``;
    these are transparently classified as contrast-enhanced from the series-level phase.
    """
    return _contrast_token(series_desc) or _contrast_token(study_desc)


def _study_date_key(s):
    """(month, day, year) from a CSV 'Study Date' (M/D/YYYY or MM-DD-YYYY), or None."""
    m = re.match(r"\s*(\d{1,2})[/-](\d{1,2})[/-](\d{4})", str(s or ""))
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def _meta_autopet(path):
    """Per-series TCIA manifest -> subject-keyed PATIENT-level records (sex/age/scanner/
    region) PLUS a PER-STUDY contrast map keyed by (subject, study-date).

    The returned _SubjectKeyedMeta normalizes a full study-folder lookup key to the bare
    PETCT_<hash> subject for the patient-level fields and to (subject, date) for contrast."""
    import csv
    by = _SubjectKeyedMeta()
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            subj = (r.get("Subject ID") or "").strip()
            if not subj:
                continue
            rec = by.setdefault(subj, dict(sex=None, age=None, scanner=None,
                                           contrast=None, region=None))
            rec["sex"] = rec["sex"] or _sex(r.get("sex"))
            rec["age"] = rec["age"] if rec["age"] is not None else _age(r.get("age"))
            rec["region"] = rec["region"] or (r.get("diagnosis") or None)
            if r.get("Modality") == "CT":
                rec["scanner"] = rec["scanner"] or _scanner(r.get("Manufacturer"))
                c = _autopet_contrast(r.get("Study Description"),
                                      r.get("Series Description"))
                dk = _study_date_key(r.get("Study Date"))
                if c is not None and dk is not None:
                    by._contrast_by_study[(subj, dk)] = c   # PER-STUDY (not subject-collapsed)
    return by


_META_LOADERS = {"amos": _meta_amos, "saros": _meta_saros, "autopet": _meta_autopet}


def load_meta(dataset):
    _, _, fname = LAYOUT[dataset]
    path = META_DIR / fname
    if not path.exists():
        print(f"  [metadata not found: {path} — correlates skipped]")
        return {}
    return _META_LOADERS[dataset](path)


# The per-case tables print every distance to two decimals, so one measurement rendered twice can
# differ by at most one printed ulp, and a delta printed from UNROUNDED operands can disagree with
# the difference of the two ROUNDED operands by up to three half-ulps. Both bounds are properties of
# the artifact's own format, not tuned thresholds: on the study's tables every compare row is off by
# exactly 0.00 or 0.01, and the paired full/surf values are equal to the last digit.
PRINTED_ULP_MM = 0.01
DELTA_ROUNDING_BOUND_MM = 3 * (PRINTED_ULP_MM / 2)


def parse_gap_table(path, z_idx, d_idx, delta_idx=None):
    """Parse a per-case results table into [(case, val_a, val_b)] plus a closed
    accounting of every case-bearing line.

    A line is a "case line" iff its first token contains '_' (case ids look like
    amos_0001 / case_000 / PETCT_xxx); header and separator rows are skipped and
    never counted. Each case line is classified into exactly one of:
      * empty     — flagged ERROR / empty / missing by the producer
      * unparsed  — looks like data but the value columns are non-finite/garbage,
                    NEGATIVE (these are distances), or self-contradictory
                    (counted, NEVER silently dropped — protects the denominator)
      * measurable — a usable (case, val_a, val_b) row
    z_idx/d_idx select the two value columns (fullres: 3,4 ; compare: 1,2).
    delta_idx, when given, is a column the producer wrote as (val_b - val_a); a row
    that contradicts its own delta is not a row whose two values can be trusted, so it is refused rather than half-believed.
    Returns (rows, counts) with counts = {measurable, empty, unparsed}.
    """
    rows, empty, unparsed = [], 0, 0
    for ln in Path(path).read_text().splitlines():
        p = ln.split()
        if not p or "_" not in p[0]:
            continue
        low = " ".join(p[1:]).lower()
        if "error" in low or "empty" in low or "missing" in low:
            empty += 1
            continue
        try:
            a, b = float(p[z_idx]), float(p[d_idx])
        except (IndexError, ValueError):
            unparsed += 1
            continue
        if not (math.isfinite(a) and math.isfinite(b)):
            unparsed += 1
            continue
        # A negative one cannot be a measurement, and admitting it lets a sign error pull a median
        # toward zero silently.
        if a < 0 or b < 0:
            unparsed += 1
            continue
        if delta_idx is not None:
            try:
                d = float(p[delta_idx])
            except (IndexError, ValueError):
                unparsed += 1
                continue
            if not math.isfinite(d) or abs((b - a) - d) > DELTA_ROUNDING_BOUND_MM:
                unparsed += 1
                continue
        rows.append((p[0], a, b))
    return rows, {"measurable": len(rows), "empty": empty, "unparsed": unparsed}


def require_surface_metric(counts, path, *, consumer):
    """Refuse a full-resolution table whose column-4 measurand is not the surface metric.

    For consumers that label their output as a surface gap (the console summary in this module
    reports a v1 table's metric under its own name instead). Raises ValueError naming the consumer,
    the path and the detected metric.
    """
    metric = counts.get("metric", "surface(v2)")
    if metric != "surface(v2)":
        raise ValueError(
            f"{consumer} requires the surface-to-surface measurand, but {path} parses as "
            f"'{metric}': in a v1 table column 4 is a CENTRE-TO-CENTRE distance and "
            f"load_fullres sets surf3d = c2c3d = col4. Publishing it under a surface-gap "
            f"label would relabel the measurand. Run measure_all.py on the masks, "
            f"or use stats.py, which reports the v1 metric under its own name.")
    return metric


def load_fullres(path):
    """Parse the full-res table into [(case, z_gap, surf3d, c2c3d, qc)] + counts."""
    lines = Path(path).read_text().splitlines()
    # a data line's first token must LOOK like a case id (amos_0001 / case_000 / PETCT_<hex>), so a
    # header row whose first token contains '_' (e.g. `case_id`) is not read as data.
    data_lines = [ln for ln in lines
                  if ln.split() and _looks_like_case_id(ln.split()[0])]

    def _is_v2(ln):
        p = ln.split()
        low = " ".join(p[1:]).lower()        # status tokens only (not the case-id p[0])
        if "error" in low or "empty" in low or "missing" in low:
            return None                      # error rows are format-uninformative
        try:
            float(p[3])                      # a real data row has numeric core cols;
            float(p[4])                      # a header (e.g. 'case_id ivc_z ...') whose
        except (IndexError, ValueError):     # first token has '_' must NOT decide the
            return None  # cohort metric -> treat as non-data
        if len(p) >= 8:
            try:
                float(p[6])
                return True                  # numeric c2c at col 6 -> v2
            except ValueError:
                return False
        return False
    # Each row is read by its own shape (`_is_v2`). The cohort label is a separate question: a
    # mixed file has no single answer, since v1's column 4 is a centre-to-centre measurement and
    # v2's a surface measurement, so the pooled `surf` array would mix two measurands.
    seen = {_is_v2(ln) for ln in data_lines} - {None}
    metric = ("mixed(v1+v2)" if seen == {True, False}
              else "c2c(v1)" if seen == {False} else "surface(v2)")

    from collections import Counter
    rows, empty, failed, unparsed, qc_small, qc_overlap = [], 0, 0, 0, 0, 0
    empty_ids, failed_ids, unparsed_ids = [], [], []
    failure_reasons = Counter()
    for ln in data_lines:
        p = ln.split()
        z = surf = c2c = float("nan")
        try:
            z = float(p[3])
            surf = float(p[4])
            # On a uniform file the per-row and per-file readings agree.
            c2c = float(p[6]) if _is_v2(ln) else surf
            is_data = True
        except (IndexError, ValueError):
            is_data = False
        if not is_data:
            status = " ".join(p[1:])         # status tokens only (not the case-id p[0])
            low = status.lower()
            if "error" in low or "empty" in low or "missing" in low:
                if "error:" in low:
                    reason = status.split(":", 1)[1].strip().replace(" ", "_").upper()
                else:
                    reason = status.strip().replace(" ", "_").upper()
                failure_reasons[reason or "UNSPECIFIED"] += 1
                if ("error:" not in low or "NO_MASK" in reason or "EMPTY" in reason
                        or "MISSING" in reason):
                    empty += 1
                    empty_ids.append(p[0])
                else:
                    failed += 1
                    failed_ids.append(p[0])
            else:
                unparsed += 1
                unparsed_ids.append(p[0])
            continue
        if not (math.isfinite(z) and math.isfinite(surf) and math.isfinite(c2c)):
            unparsed += 1
            unparsed_ids.append(p[0])
            continue
        # These columns are distances with two stated relations, c2c >= z and surf <= c2c. A
        # negative row, or one that breaks either relation beyond gap_config.INVARIANT_SLACK_MM
        # (which absorbs two-decimal rounding on touching cases), is counted as `unparsed`, so the
        # accounting stays closed.
        if (z < 0 or surf < 0 or c2c < 0
                or z - c2c > C.INVARIANT_SLACK_MM
                or surf - c2c > C.INVARIANT_SLACK_MM):
            unparsed += 1
            unparsed_ids.append(p[0])
            continue
        qc = p[-1]
        if "SMALL" in qc:
            qc_small += 1
        if "OVERLAP" in qc:               # 3-D label overlap (surf==0, seg error)
            qc_overlap += 1
        rows.append((p[0], z, surf, c2c, qc))
    return rows, {"measurable": len(rows), "empty": empty, "failed": failed,
                  "unparsed": unparsed,
                  "qc_small": qc_small, "qc_overlap": qc_overlap, "metric": metric,
                  "empty_ids": empty_ids, "failed_ids": failed_ids,
                  "unparsed_ids": unparsed_ids,
                  "failure_reasons": dict(sorted(failure_reasons.items()))}


def load_compare(path):
    """[(case, fast3d, full3d)] + counts. fast=col1, full=col2, delta=col3."""
    return parse_gap_table(path, z_idx=1, d_idx=2, delta_idx=3)


def assert_compare_matches_fullres(fr_rows, cmp_rows, fr_name, cmp_name):
    """The compare file's `full` column and the fullres file's `surf` column are the SAME measurement
    of the same case, written by the same run.

    Roster validation cannot catch that: both files list the same cases.

    Raises rather than counting, because a mismatch is not a bad ROW -- it means the
    two files are not from the same run, so every paired statistic downstream is
    describing something other than what it is labelled.
    """
    surf = {r[0]: r[2] for r in fr_rows}
    bad = [(c, full, surf[c]) for (c, _fast, full) in cmp_rows
           if c in surf and abs(full - surf[c]) > PRINTED_ULP_MM]
    if bad:
        shown = "; ".join(f"{c}: compare full={f:.2f} vs fullres surf={s:.2f}"
                          for c, f, s in bad[:5])
        raise RuntimeError(
            f"{cmp_name} and {fr_name} disagree on {len(bad)} of "
            f"{sum(1 for (c, _f, _v) in cmp_rows if c in surf)} shared cases, beyond the "
            f"{PRINTED_ULP_MM:g} mm printed resolution -- they are not from one run, so the "
            f"fast-vs-full agreement statistics would not be measuring resolution. {shown}")
    return len(bad)


def _publication_invariants_hold(z, surf, c2c, source):
    """The two relations `summary()` reports as `invariant c2c>=z` / `surf<=c2c` are evaluated HERE,
    before anything is written.

    Row-level refusal in `load_fullres` should make this unreachable; it is kept as a postcondition
    precisely because "should" is not a guarantee, and because the arrays reaching publication can
    be built by callers other than the loader."""
    z = np.asarray(z, float); surf = np.asarray(surf, float); c2c = np.asarray(c2c, float)
    viol_z = int((z - c2c > C.INVARIANT_SLACK_MM).sum())
    viol_s = int((surf - c2c > C.INVARIANT_SLACK_MM).sum())
    if viol_z or viol_s:
        raise RuntimeError(
            f"{source}: refusing to publish -- "
            f"{viol_z} case(s) violate c2c>=z and {viol_s} violate surf<=c2c beyond "
            f"INVARIANT_SLACK_MM={C.INVARIANT_SLACK_MM} mm. These are the relations the "
            f"cohort summary reports as invariants; a table that breaks them is not a "
            f"table whose median means what the report says it means.")
    return True


def load_collapsed(path):
    """Parse the collapsed-class results table (`<prefix>_results_collapsed.txt`) into paired rows +
    closed counts. Missing-mask failures are `empty`; algorithmic/probability/grid failures are
    `failed`. Mirrors load_fullres' case-id + status-token exclusion (autoPET ids are free-text;
    error/empty/missing scanned in the STATUS tokens only) so the exclusion accounting stays closed.
    Missing file -> ([], zeros)."""
    p = Path(path)
    if not p.exists():
        return [], {"measurable": 0, "empty": 0, "failed": 0, "unparsed": 0,
                    "n_connected": 0, "failure_reasons": {}, "mode": "labels"}
    from collections import Counter
    rows, empty, failed, unparsed, n_conn, mode = [], 0, 0, 0, 0, "labels"
    failure_reasons = Counter()
    for ln in p.read_text().splitlines():
        t = ln.split()
        if ln.startswith("# mode="):
            mode = ln.split("=", 1)[1].strip() or "labels"
            continue
        if not t or not _looks_like_case_id(t[0]):
            continue
        # The variant column's value MISSING_VEIN contains "missing", so status text is read only
        # from a row that does not parse as data (every failure row starts with non-numeric status
        # text such as `ERROR: ...`).
        try:
            float(t[1]); float(t[2])
            is_data = True
        except (IndexError, ValueError):
            is_data = False
        if not is_data:
            status = " ".join(t[1:])              # status tokens only (not free-text id)
            low = status.lower()
            if "error" in low or "empty" in low or "missing" in low:
                if "error:" in low:
                    reason = status.split(":", 1)[1].strip().replace(" ", "_").upper()
                else:
                    reason = status.strip().replace(" ", "_").upper()
                failure_reasons[reason or "UNSPECIFIED"] += 1
                # NO_MASK / missing-or-empty input is primary inference missingness. An
                # empty re-split, grid/probability failure, or exception is an algorithmic
                # unevaluable case and is accounted separately.
                if ("error:" not in low or reason in {
                        "NO_MASK", "EMPTY_MASK", "MISSING/EMPTY_IVC_OR_LCIV",
                        "EMPTY_INPUT_MASK"}):
                    empty += 1
                else:
                    failed += 1
            else:
                unparsed += 1
                failure_reasons["MALFORMED_RESULT_ROW"] += 1
            continue
        try:
            orig, coll = float(t[1]), float(t[2])
            o_pres, c_pres = int(t[4]), int(t[5])
            ncomp, conn = int(t[6]), int(t[7])
            variant, qc = t[8], t[9]
            if len(t) == 13:
                post_ncomp, post_conn, resolved = int(t[10]), int(t[11]), int(t[12])
            elif len(t) == 10:
                post_ncomp = post_conn = resolved = None
            else:
                raise ValueError(f"expected 10 or 13 columns, got {len(t)}")
        except (IndexError, ValueError):
            unparsed += 1
            failure_reasons["MALFORMED_RESULT_ROW"] += 1
            continue
        if not (math.isfinite(orig) and math.isfinite(coll)):
            unparsed += 1
            failure_reasons["NONFINITE_RESULT"] += 1
            continue
        # Negative gaps, a delta that contradicts its own two columns, and non-binary flags are
        # refused and counted, never silently dropped, so the denominator is protected.
        if orig < 0 or coll < 0:
            unparsed += 1
            failure_reasons["NEGATIVE_GAP"] += 1
            continue
        # `bool(int(...))` turns ANY nonzero integer into True, so a corrupted 7 would read as
        # "present"; these columns are indicators, so anything outside {0, 1} is refused.
        flags = [o_pres, c_pres, conn]
        if post_conn is not None:
            flags.append(post_conn)
        if resolved is not None:
            flags.append(resolved)
        if any(v not in (0, 1) for v in flags):
            unparsed += 1
            failure_reasons["NONBINARY_FLAG"] += 1
            continue
        # The delta column is the producer's OWN statement of the difference; a row that
        # contradicts it is refused. CONVENTION, measured from the study's tables rather than
        # assumed: delta = orig - coll. Across all 1,855 paired rows max|delta - (orig - coll)| =
        # 0.01 while max|delta - (coll - orig)| = 122.14, so the opposite sign (the one
        # `parse_gap_table` uses for the fullres/compare tables) would refuse essentially every
        # real row.
        try:
            delta = float(t[3])
        except (IndexError, ValueError):
            unparsed += 1
            failure_reasons["MALFORMED_RESULT_ROW"] += 1
            continue
        if (not math.isfinite(delta)
                or abs((orig - coll) - delta) > DELTA_ROUNDING_BOUND_MM):
            unparsed += 1
            failure_reasons["DELTA_CONTRADICTS_VALUES"] += 1
            continue
        rows.append((t[0], orig, coll, bool(o_pres), bool(c_pres), int(ncomp),
                     bool(conn), variant, qc,
                     (int(post_ncomp) if post_ncomp is not None else None),
                     (bool(post_conn) if post_conn is not None else None),
                     (bool(resolved) if resolved is not None else None)))
        n_conn += int(bool(conn))
    endpoint_n = sum(r[11] is not None for r in rows)
    return rows, {"measurable": len(rows), "empty": empty, "failed": failed,
                  "unparsed": unparsed, "n_connected": n_conn,
                  "resolved_endpoint_available": endpoint_n,
                  "failure_reasons": dict(sorted(failure_reasons.items())), "mode": mode}


def load_spacing(path):
    """Read the aux file (lines: case sz sy sx ...) into {case: sz_mm} for finite
    z-spacings. Coarse axial spacing inflates the apparent z_gap, so the secondary
    axial metric should be covaried by it. Missing file -> {} (graceful)."""
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    for ln in p.read_text().splitlines():
        toks = ln.split()
        if len(toks) >= 2 and "_" in toks[0]:
            try:
                sz = float(toks[1])
            except ValueError:
                continue
            if math.isfinite(sz):
                out[toks[0]] = sz
    return out


def load_spacing_xyz(path):
    """Read <prefix>_results_aux.txt into {case: (sz, sy, sx)}.

    `load_spacing` returns the z spacing alone, which is all the slice-normalized diagnostic
    needs. The normalizer sensitivity needs the IN-PLANE spacings too -- the whole point being
    that dividing by z and dividing by the in-plane geometric mean move the heterogeneity in
    OPPOSITE directions -- so this reads the triple rather than widening `load_spacing`'s
    contract under its existing callers. Missing file -> {}.
    """
    out = {}
    p = Path(path)
    if not p.exists():
        return out
    for ln in p.read_text().splitlines():
        tok = ln.split()
        if len(tok) >= 4 and "_" in tok[0]:
            try:
                out[tok[0]] = (float(tok[1]), float(tok[2]), float(tok[3]))
            except ValueError:
                continue
    return out


def load_aux(path):
    """Read <prefix>_results_aux.txt (case sz sy sx lcc n_comp body_csa) into
    {case: {sz, lcc, n_comp, csa}} (NaN where unparseable). Missing file -> {}."""
    out = {}
    p = Path(path)
    if not p.exists():
        return out

    def _f(tok):
        try:
            return float(tok)
        except ValueError:
            return float("nan")
    for ln in p.read_text().splitlines():
        t = ln.split()
        if len(t) >= 7 and "_" in t[0]:
            out[t[0]] = {"sz": _f(t[1]), "lcc": _f(t[4]),
                         "n_comp": _f(t[5]), "csa": _f(t[6])}
    return out


def _size_band_edges(aux):
    """Body-CSA tertile edges for `patient_size`, or `(None, None)` when fewer than
    `3 * MIN_GROUP_N` rows carry a finite body CSA, with that count. The one rule both the
    injection and its provenance line use."""
    csa = np.array([a["csa"] for a in aux.values()
                    if math.isfinite(a.get("csa", float("nan")))], float)
    if csa.size >= 3 * C.MIN_GROUP_N:
        lo, hi = np.percentile(csa, [100.0 / 3.0, 200.0 / 3.0])
        return lo, hi, int(csa.size)
    return None, None, int(csa.size)


def aux_correlate_availability(aux):
    """The provenance line saying whether `patient_size` was banded, and from how many rows.

    The correlate is DROPPED, not reported empty, when too few rows carry a finite body CSA, and
    `measure_ivc_lciv_gap.body_csa` reads that area off the CT image. When the measurement runs on
    masks alone, body CSA is NA and patient_size is not banded; this line records that.
    """
    if not aux:
        return "no aux rows: seg_quality and patient_size were not computed"
    lo, _hi, n_finite = _size_band_edges(aux)
    if lo is not None:
        return (f"patient_size (body-CSA tertiles) banded from {n_finite} of {len(aux)} rows with a "
                f"finite body CSA")
    return (f"patient_size (body-CSA tertiles) NOT COMPUTED: {n_finite} of {len(aux)} rows carry a "
            f"finite body CSA and banding needs {3 * C.MIN_GROUP_N}; the body CSA is read from the CT "
            f"image (measure_ivc_lciv_gap.body_csa), so a run without the CT leaves it NA")


def _inject_aux_correlates(meta, aux):
    """Add the two Phase-1 correlates, `seg_quality` (clean vs fragmented LCIV mask, a
    'segmentation confidence' proxy) and `patient_size` (`size_band`: tertiles of body CSA), to the
    metadata records. Size is only banded when enough cases carry a finite CSA (>= 3*MIN_GROUP_N).

    For the per-case AMOS/SAROS metadata the value is written in place."""
    lo, hi, _n_finite = _size_band_edges(aux)
    per_study = hasattr(meta, "set_study")     # autoPET _SubjectKeyedMeta vs plain dict

    def _put(cid, rec, field, value):
        if per_study:
            meta.set_study(cid, field, value)
        else:
            rec[field] = value

    for cid, a in aux.items():
        rec = meta.get(cid)
        if rec is None:
            continue
        nc, lcc = a.get("n_comp", float("nan")), a.get("lcc", float("nan"))
        # `nc >= 1`, not merely finite. A component count of ZERO cannot describe a mask that was
        # measured -- a real mask has at least one component -- and three separate paths produce it,
        # only one of which raises: an ABSENT LCIV mask file (case_aux_metrics returns its default),
        # an EMPTY mask (lciv_quality returns (nan, 0)), and any exception inside case_aux_metrics.
        # Requiring nc >= 1 leaves seg_quality unset for an unmeasured case, so it drops out of the
        # Kruskal-Wallis comparison instead of being labelled clean.
        if math.isfinite(nc) and nc >= 1:
            _put(cid, rec, "seg_quality",
                 "fragmented" if (nc > 1 or (math.isfinite(lcc) and lcc < 0.99)) else "clean")
        v = a.get("csa", float("nan"))
        if lo is not None and math.isfinite(v):
            _put(cid, rec, "size_band",
                 "small" if v < lo else ("large" if v > hi else "mid"))


# ---------------------------------------------------------------------------
# Geometry / classification helpers
# ---------------------------------------------------------------------------
def lateral_residual(z, d):
    """Lateral leg sqrt(max(0, d^2 - z^2)), clamped at 0 (never NaN).
    This is an UPPER BOUND on the true lateral separation (the nearest-distance
    pair's axial component can exceed the axial-extent gap z)."""
    z = np.asarray(z, float)
    d = np.asarray(d, float)
    return np.sqrt(np.maximum(0.0, d ** 2 - z ** 2))


def classify_gap(z, d, primary, present=None, lat_tol=None, ax_tol=None):
    """Scale-invariant axial/lateral/mixed/absent label for one case.

    'axial'   : resolvable gap with a negligible lateral leg (<= lat_tol mm)
    'lateral' : resolvable gap with sub-voxel axial extent (z <= ax_tol mm)
    'mixed'   : resolvable gap with both axial and lateral components
    'absent'  : no resolvable gap (`primary` <= present)
    The lateral leg is in mm and does not shrink with gap size, so the split is
    comparable across the gap-size distribution and across cohorts.

    TWO DIFFERENT MEASURANDS, AND THAT IS DELIBERATE. `primary` decides PRESENCE and is the study's
    primary metric, the 3-D surface-to-surface gap. `d` decides DIRECTION and is the
    centre-to-centre separation. A keyword default would let a caller decide presence on `d` by
    omission, so `primary` is required.

    WHY DIRECTION IS NOT ALSO TAKEN ON THE PRIMARY METRIC. `lateral_residual` is
    `sqrt(max(0, d^2 - z^2))`, which is a decomposition only where `d >= z`. Measured over the
    study's tables: `c2c < z` on **0 of 1,873** rows, but `surf < z` on **880 of 1,873
    (47.0 %)** -- the surface distance and the axial EXTENT gap use different voxel-boundary
    conventions. Substituting `surf` for `d` therefore clamps the lateral leg to zero on nearly half
    the cohort and labels those cases `axial` by arithmetic rather than geometry: **86.2 % of
    the `axial` labels it produces are the clamp.** So direction uses c2c, for which
    d^2 = z^2 + lat^2 holds exactly (as all_stats does), and presence uses the primary metric.
    """
    present = C.PRESENT_MM if present is None else present
    lat_tol = C.LATERAL_TOL_MM if lat_tol is None else lat_tol
    ax_tol = C.AXIAL_TOL_MM if ax_tol is None else ax_tol
    if primary <= present:
        return "absent"
    if z <= ax_tol:
        return "lateral"
    lat = math.sqrt(max(0.0, d * d - z * z))
    if lat <= lat_tol:
        return "axial"
    return "mixed"


def censored_fraction(z, boundary=None):
    """Percent of cases at/below the censoring boundary (z_gap clamped at 0)."""
    boundary = C.CENSOR_BOUNDARY_MM if boundary is None else boundary
    z = np.asarray(z, float)
    return 100.0 * float((z <= boundary).mean()) if z.size else float("nan")


def _ci(lo, hi, nd=1, sep="-"):
    # The one interval formatter is `statlib.format_ci`; the other Phase-1 producers call it too.
    # Every table and console site in this module calls `_ci`;
    # `test_every_interval_the_module_prints_goes_through_the_one_formatter` pins that.
    return S.format_ci(lo, hi, nd, sep=sep)


def _cluster_robust_console(r):
    """One console line of the cluster-robust table. Its ends are the table's own, already rounded to
    four decimals by `cluster_robust_rows`, so the console says what the CSV says: the token here
    would mean the CSV's two ends are equal."""
    iid = _ci(r["iid_lo"], r["iid_hi"], 4, sep=",") or "NA"
    clu = _ci(r["cluster_lo"], r["cluster_hi"], 4, sep=",") or "NA"
    return (f"    {r['metric']:<28s} iid [{iid}] -> cluster [{clu}]  "
            f"ICC={r['icc']} DEFF={r['design_effect']} width x{r['width_ratio']}")


def _distribution_annotation(pk, n, pp, plo, phi, med, mlo, mhi, p25, p75, mean, sd):
    """The per-cohort `distribution.png` annotation. Both intervals go through the one formatter;
    `sep` keeps the en dash for an interval, and a point prints as the token, not `m–m`. The IQR is
    a range of the data, not an interval estimate."""
    return (f"present {pk}/{n} ({pp:.0f}%, 95% CI {_ci(plo, phi, 0, sep='–') or 'NA'})\n"
            f"median {med:.1f} mm (95% CI {_ci(mlo, mhi, 1, sep='–') or 'NA'}; "
            f"IQR {p25:.1f}–{p75:.1f})\n"
            f"mean {mean:.1f} \u00b1 {sd:.1f} mm (SD)")


def _pfmt(x):
    """Returns '' for None, scientific notation below 1e-4, else 4 dp."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return ""
    x = float(x)
    if x == 0.0:
        return "0"
    if 0 < x < 1e-4:
        return f"{x:.2e}"
    return f"{x:.4f}"


def _looks_like_case_id(tok):
    """True iff `tok` looks like a per-case id (amos_0001 / case_000 / PETCT_<hex>),
    i.e. letters + '_' + a digit-or-hex char. Guards the parsers against a header row
    whose first token contains '_' (e.g. 'case_id ...') being mis-counted as an
    unparsed case and inflating n_total / triggering a spurious INVESTIGATE flag."""
    return bool(re.match(r"^[A-Za-z]+_[0-9A-Fa-f]", str(tok)))


def quantization_frac_percase(v, steps, tol=0.05, exclude_zero=False):
    """Fraction of values within `tol` of an integer multiple of THAT CASE's own step
    (per-case slice thickness), the correct heaping fingerprint for a mixed-resolution
    cohort. `steps` is aligned with `v`; entries with a non-finite step are skipped.

    THE EXACT-ZERO ATOM IS NOT A RESOLUTION FINGERPRINT. `v = 0` satisfies
    `|v - round(v/s)*s| <= tol` for EVERY step, so with `exclude_zero=False` the whole
    left-censored spike -- masks that touch, where there is no gap to quantize -- is scored as
    slice-multiple heaping. That spike is 15.0 / 25.4 / 23.6 % of AMOS / SAROS / autoPET, and it
    inflates the fraction by 10.5 / 14.6 / 14.8 percentage points (40.6 -> 30.1, 57.1 -> 42.5,
    52.1 -> 37.3). Both figures are reported side by side with the zero share.

    `exclude_zero` defaults to False so this stays a pure measurement of what it says on the
    tin; the CALLER chooses which quantity it is reporting and labels it.
    """
    v = np.asarray(v, float)
    steps = np.asarray(steps, float)
    ok = np.isfinite(v) & np.isfinite(steps) & (steps > 0)
    if exclude_zero:
        ok = ok & (v != 0.0)
    if not ok.any():
        return float("nan")
    vv, ss = v[ok], steps[ok]
    nearest = np.round(vv / ss) * ss
    return float((np.abs(vv - nearest) <= tol).mean())


def slice_normalized(surf, sz):
    """(surf/sz) in slice-thickness units over cases with a finite positive sz, plus
    the count used. This acquisition-scaled diagnostic tests whether a gap tracks the
    through-plane sampling interval; it does not make the measurement resolution-
    invariant or remove other acquisition/cohort differences."""
    surf = np.asarray(surf, float)
    sz = np.asarray(sz, float)
    ok = np.isfinite(surf) & np.isfinite(sz) & (sz > 0)
    return surf[ok] / sz[ok], int(ok.sum())


# ---------------------------------------------------------------------------
# all_stats.csv
# ---------------------------------------------------------------------------
def _stat_row(metric, source, v, counts, z=None, d=None, primary=None, censored=None,
              qc_flagged=None):
    """One distribution row. `v` is the metric being summarized; (z, d) are the
    axial/3D pair, supplied only for the secondary axial-z_gap row that also reports
    the axial/lateral decomposition; `primary` is the study's primary surface metric and is
    REQUIRED alongside them, because it -- not `d` -- decides which cases have a gap at all. Supplying (z, d) without it raises rather than silently falling back to `d`.

    `qc_flagged` IS NOT ONE QUANTITY, and the 32-col schema cannot carry two. `all_stats()` fills it
    per row: the axial z_gap row gets `counts['qc_small']`, the degenerate-mask count
    (SMALL_IVC/SMALL_LCIV below `QC_MIN_VOXELS`); the PRIMARY 3D surface row gets the
    implausible-gap outlier count `(surf > QC_MAX_GAP_MM).sum()`; the center-to-center and the two
    compare rows have NO QC rule and pass nothing.

    It is also NOT the whole QC picture. Both counts, and a drop-`qc_overlap` sensitivity, are
    printed by the console cohort-flow block instead."""
    if (z is not None or d is not None) and primary is None:
        raise ValueError(
            "_stat_row: the axial/lateral decomposition needs `primary` (the surface metric) "
            "to decide presence; deciding it from `d` makes 'absent' unreachable")
    v = np.asarray(v, float)
    n = len(v)
    n_empty = counts.get("empty", 0)
    n_unparsed = counts.get("unparsed", 0)
    n_total = n + n_empty + counts.get("failed", 0) + n_unparsed
    if n == 0:
        # No measurable cases (e.g. an all-empty compare file alongside a populated
        # fullres file): emit a placeholder row with the accounting columns filled
        # and the distribution columns blank, instead of crashing on desc['median']
        # (S.describe([]) returns only {'n': 0}).
        blank = dict.fromkeys(
            ["gap_present_pct", "gap_present_ci", "median_mm", "median_ci",
             "p25_mm", "p75_mm", "iqr_mm", "mean_mm", "sd_mm", "skew", "normality_p",
             "p10_mm", "p90_mm", "min_mm", "max_mm", "band_2p5_7p5_pct", "band_ci",
             "n_gt_30mm", "axial_pct", "lateral_pct", "mixed_pct",
             "median_axial_fraction", "median_lateral_residual_mm"], "")
        return dict(
            metric=metric, source=source, n_total=n_total, n_measurable=0,
            n_empty=n_empty, n_unparsed=n_unparsed,
            empty_rate_pct=round(100 * n_empty / n_total, 1) if n_total else "",
            censored_at_0_pct=(round(censored, 1) if censored is not None else ""),
            qc_flagged=("" if qc_flagged is None else qc_flagged), **blank)
    desc = S.describe(v, alpha=C.ALPHA, n_boot=C.N_BOOT, seed=C.SEED,
                      ci_method=C.CI_METHOD)
    present_k = int((v > C.PRESENT_MM).sum())
    pres_pct, pres_lo, pres_hi = S.proportion_pct_ci(present_k, n, C.ALPHA)
    band_k = int(((v >= C.BAND_LO_MM) & (v <= C.BAND_HI_MM)).sum())
    band_pct, band_lo, band_hi = S.proportion_pct_ci(band_k, n, C.ALPHA)
    r = dict(
        metric=metric, source=source, n_total=n_total, n_measurable=n,
        n_empty=n_empty, n_unparsed=n_unparsed,
        empty_rate_pct=round(100 * n_empty / n_total, 1) if n_total else "",
        censored_at_0_pct=(round(censored, 1) if censored is not None else ""),
        qc_flagged=("" if qc_flagged is None else qc_flagged),
        gap_present_pct=round(pres_pct, 1), gap_present_ci=_ci(pres_lo, pres_hi),
        median_mm=round(desc["median"], 2),
        median_ci=_ci(desc["median_ci_lo"], desc["median_ci_hi"], 2),
        p25_mm=round(desc["p25"], 2), p75_mm=round(desc["p75"], 2),
        iqr_mm=f"{desc['p25']:.2f}-{desc['p75']:.2f}",
        mean_mm=round(desc["mean"], 2), sd_mm=round(desc["sd"], 2),
        skew=round(desc["skew"], 2),
        normality_p=_pfmt(desc["shapiro_p"]),
        p10_mm=round(desc["p10"], 2), p90_mm=round(desc["p90"], 2),
        min_mm=round(desc["min"], 2), max_mm=round(desc["max"], 2),
        band_2p5_7p5_pct=round(band_pct, 1), band_ci=_ci(band_lo, band_hi),
        n_gt_30mm=int((v > C.LARGE_GAP_MM).sum()),
        axial_pct="", lateral_pct="", mixed_pct="",
        median_axial_fraction="", median_lateral_residual_mm="",
    )
    if z is not None and d is not None:
        z = np.asarray(z, float)
        d = np.asarray(d, float)
        primary = np.asarray(primary, float)
        # PRESENCE ON THE PRIMARY METRIC, DIRECTION AMONG PRESENT CASES ONLY. `gap_present_pct` on
        # this row summarises `v` (= z), so the absent count is not recoverable from it; the console
        # block prints it.
        present_mask = primary > C.PRESENT_MM
        npr = int(present_mask.sum())
        cats = [classify_gap(zz, dd, pp) for zz, dd, pp in zip(z, d, primary)]
        lat = lateral_residual(z[present_mask], d[present_mask])
        frac = z[present_mask] / d[present_mask] if npr else np.array([])
        r.update(
            axial_pct=round(100 * cats.count("axial") / npr, 1) if npr else "",
            lateral_pct=round(100 * cats.count("lateral") / npr, 1) if npr else "",
            mixed_pct=round(100 * cats.count("mixed") / npr, 1) if npr else "",
            median_axial_fraction=(round(float(np.median(frac)), 2)
                                   if frac.size else ""),
            median_lateral_residual_mm=(round(float(np.median(lat)), 2)
                                        if lat.size else ""),
        )
    return r


def all_stats(z, surf, c2c, fr_counts, fast, full, cmp_counts, fr_name, cmp_name,
              censored, cmp_metric="3D surface"):
    # PRIMARY magnitude = surface-to-surface; geometry (axial/lateral split, the d>=z invariant) is
    # computed on the center-to-center pair (z, c2c) for which d^2 = z^2 + lat^2 holds exactly.
    qc = int((surf > C.QC_MAX_GAP_MM).sum())
    qc_small = int(fr_counts.get("qc_small", 0))
    return [
        _stat_row("axial z_gap (secondary; left-censored at 0)", fr_name, z,
                  fr_counts, z=z, d=c2c, primary=surf, censored=censored,
                  qc_flagged=qc_small),
        _stat_row("3D surface-to-surface (PRIMARY)", fr_name, surf, fr_counts,
                  qc_flagged=qc),
        _stat_row("3D center-to-center (geometry/secondary)", fr_name, c2c,
                  fr_counts),
        _stat_row(f"{cmp_metric} (full-res, compare)", cmp_name, full, cmp_counts),
        _stat_row(f"{cmp_metric} (fast 3mm, compare)", cmp_name, fast, cmp_counts),
    ]


# ---------------------------------------------------------------------------
# Method-agreement (axial vs 3D; fast vs full-res)
# ---------------------------------------------------------------------------
def agreement_rows(dir_name, fast, full):
    """Lin's CCC, ICC(A,1), and Bland-Altman for the fast(3mm)-vs-full-res 3D gap.

    ONLY this comparison is a true method-agreement study: two estimates of the
    SAME quantity (the 3D gap) at different segmentation resolutions. We do NOT
    compute agreement statistics for axial-z_gap-vs-3D, because those are not two
    interchangeable measurements — z is the axial COMPONENT of d (d^2=z^2+lat^2,
    so d>=z by construction). For axial vs 3D the bias would just be -mean(lateral
    leg) and a depressed CCC/ICC would reflect geometry, not measurement error.
    That relationship is reported separately as a decomposition (median lateral
    leg + median axial fraction z/d) in summary().
    """
    out, binned_out = [], []
    if len(fast) >= 2 and len(full) >= 2:
        ba = S.bland_altman(fast, full, C.ALPHA)
        dnp = ba.get("diff_normal_p")
        bn = S.binned_empirical_loa(fast, full, alpha=C.ALPHA)
        sel = S.primary_loa(ba, C.ALPHA, bn)
        for band in bn["bins"]:
            binned_out.append(dict(
                dataset=dir_name, comparison="fast3mm vs fullres 3D (paired)",
                band_index=band["index"], band_of=bn["n_bins"],
                mean_gap_lo_mm=round(band["m_lo"], 2), mean_gap_hi_mm=round(band["m_hi"], 2),
                n=band["n"],
                loa_lo_mm=round(band["loa_lo"], 2), loa_hi_mm=round(band["loa_hi"], 2),
                width_mm=round(band["loa_hi"] - band["loa_lo"], 2)))
        out.append(dict(
            dataset=dir_name, comparison="fast3mm vs fullres 3D (paired)", n=ba["n"],
            lin_ccc=round(S.lin_ccc(fast, full), 3),
            icc_a1=round(S.icc_a1(fast, full), 3),
            bias_mm=round(ba["bias"], 2),
            bias_ci=_ci(ba.get("bias_ci_lo"), ba.get("bias_ci_hi"), 2),
            sd_diff_mm=round(ba["sd_diff"], 2),
            loa_lo_mm=round(ba["loa_lo"], 2), loa_hi_mm=round(ba["loa_hi"], 2),
            loa_lo_np_mm=round(ba["loa_lo_np"], 2), loa_hi_np_mm=round(ba["loa_hi_np"], 2),
            # keep the RAW numeric p (not rounded to 0.0000) for downstream logic; it is
            # written to CSV verbatim (pandas keeps scientific notation for tiny values).
            diff_normal_p=dnp,
            prop_bias_r=round(ba["prop_bias_r"], 3),
            # The CSV states which LoA pair is primary, and why.
            **{"loa_primary_kind": sel["kind"],
               "loa_primary_lo_mm": round(sel["lo"], 2),
               "loa_primary_hi_mm": round(sel["hi"], 2),
               "loa_primary_reason": sel["reason"]},
        ))
    return out, binned_out


def collapsed_class_stats(dir_name, ds, collapsed_rows, counts=None, mode="labels"):
    """Tests whether closure is consistent with an internal class-boundary artifact; persistence is
    not itself anatomical proof. NOT primary (the primary Phase-1 summary is
    gap_config.PRIMARY_INFERENCE). Returns
    (rows, summary): a one-element CSV-row list (COLLAPSED_HEADER) and a summary dict (no caller
    pools it across cohorts). ([], {}) when no paired cases."""
    n = len(collapsed_rows)
    if n == 0:
        return [], {}
    orig = np.array([r[1] for r in collapsed_rows], float)
    coll = np.array([r[2] for r in collapsed_rows], float)
    o_pres = np.array([r[3] for r in collapsed_rows], bool)
    c_pres = np.array([r[4] for r in collapsed_rows], bool)
    connected = np.array([r[6] for r in collapsed_rows], bool)
    variants = [r[7] for r in collapsed_rows]
    endpoint_available = np.array(
        [len(r) > 11 and r[11] is not None for r in collapsed_rows], bool)
    resolved_connected = np.array(
        [bool(r[11]) if len(r) > 11 and r[11] is not None else False
         for r in collapsed_rows], bool)

    # These intervals follow gap_config.CI_METHOD, so a percentile sensitivity run applies to them
    # too.
    mo_lo, mo_hi = S.median_ci_auto(orig, C.CI_METHOD, C.ALPHA, C.N_BOOT, C.SEED)
    mcoll_lo, mcoll_hi = S.median_ci_auto(coll, C.CI_METHOD, C.ALPHA, C.N_BOOT, C.SEED)
    d_med, d_lo, d_hi = S.paired_median_diff_ci(orig, coll, alpha=C.ALPHA,
                                                n_boot=C.N_BOOT, seed=C.SEED,
                                                method=C.CI_METHOD)
    # Most pairs are unchanged by the merge (63-72 % per cohort), so the paired median sits on that
    # atom of zeros; the atom straddles the middle of the sample, so every resample's median sits
    # there too, and the interval collapses to [0, 0], which `_ci` publishes as the token rather
    # than as "0.00-0.00" (no change, no uncertainty) beside a McNemar p of 1e-54. The change is
    # reported by DIRECTION instead: the atom's share, and the shares of pairs that changed, shrank
    # (d = orig - collapsed > 0) and grew (d < 0), each with a Wilson interval. Each pair counts
    # once, so no outlier dominates a share. A change here is the merge and the nearest-trunk
    # re-split together: 86 of AMOS's 145 shrinkages, and 14 of its 16 growths (all five of 36-61
    # mm), sit on a disconnected union, where the re-split can move the gap with no voxel filling
    # the void. Where the merge and re-split change a gap they almost
    # always shrink it (145 of 161, 97 of 99, 360 of 372).
    pair = S._clean_pair(orig, coll)
    d = (pair[0] - pair[1]) if pair is not None else np.array([], float)
    atom = float(np.mean(d == d_med)) if d.size else float("nan")
    n_d = int(d.size)
    pch, pch_lo, pch_hi = S.proportion_pct_ci(int((d != 0).sum()), n_d, C.ALPHA)
    psh, psh_lo, psh_hi = S.proportion_pct_ci(int((d > 0).sum()), n_d, C.ALPHA)
    pgr, pgr_lo, pgr_hi = S.proportion_pct_ci(int((d < 0).sum()), n_d, C.ALPHA)
    k_o, k_c = int(o_pres.sum()), int(c_pres.sum())
    po, po_lo, po_hi = S.proportion_pct_ci(k_o, n, C.ALPHA)
    pc, pc_lo, pc_hi = S.proportion_pct_ci(k_c, n, C.ALPHA)
    b = int((o_pres & ~c_pres).sum())      # present-before, absent-after = gap FIXED by the merge
    cc = int((~o_pres & c_pres).sum())     # absent-before, present-after = gap CREATED (rare)
    mcn = S.mcnemar_test(b, cc)
    k_conn = int(connected.sum())
    pconn, pconn_lo, pconn_hi = S.proportion_pct_ci(k_conn, n, C.ALPHA)
    # The schema-v2 endpoint requires both proximity and post-resplit 0D connectivity under the
    # configured voxel-neighbourhood convention. It is not proof of biological patency.
    eligible_resolve = o_pres & endpoint_available
    n_op = int(eligible_resolve.sum())
    n_resolved = int((eligible_resolve & resolved_connected).sum())
    if n_op > 0:
        pres_pct, pres_lo, pres_hi = S.proportion_pct_ci(n_resolved, n_op, C.ALPHA)
    else:
        pres_pct = pres_lo = pres_hi = float("nan")
    # `MULTI_COMPONENT` is a mask-topology/fragmentation flag. It is not proof of
    # an anatomical venous variant, so publication-facing fields use topology terms.
    topology_ok = np.array([v == "OK" for v in variants], bool)
    multicomponent = np.array([v == "MULTI_COMPONENT" for v in variants], bool)
    topology_ok_pct = 100.0 * int(topology_ok.sum()) / n

    def _resolved_within(mask):
        eligible = mask & eligible_resolve
        resolved = eligible & resolved_connected
        den, num = int(eligible.sum()), int(resolved.sum())
        if den:
            pct, lo, hi = S.proportion_pct_ci(num, den, C.ALPHA)
        else:
            pct = lo = hi = float("nan")
        return den, num, pct, lo, hi

    ok_den, ok_num, ok_pct, ok_lo, ok_hi = _resolved_within(topology_ok)
    mc_den, mc_num, mc_pct, mc_res_lo, mc_res_hi = _resolved_within(multicomponent)
    counts = counts or {}
    failure_reasons = counts.get("failure_reasons", {})
    failure_text = ";".join(f"{k}:{v}" for k, v in sorted(failure_reasons.items()))

    row = {
        "dataset": dir_name, "mode": mode, "n_paired": n,
        "n_empty": counts.get("empty", ""), "n_failed": counts.get("failed", ""),
        "n_unparsed": counts.get("unparsed", ""), "failure_reasons": failure_text,
        "median_orig_mm": round(float(np.median(orig)), 2), "median_orig_ci": _ci(mo_lo, mo_hi, 2),
        "median_collapsed_mm": round(float(np.median(coll)), 2),
        "median_collapsed_ci": _ci(mcoll_lo, mcoll_hi, 2),
        "median_paired_delta_mm": round(float(d_med), 2),
        "paired_delta_ci": _ci(d_lo, d_hi, 2),
        "paired_delta_atom_pct": (round(100.0 * atom, 1) if math.isfinite(atom) else ""),
        "pct_pairs_changed": (round(pch, 1) if math.isfinite(pch) else ""),
        "pct_pairs_changed_ci": _ci(pch_lo, pch_hi, 1),
        "pct_pairs_shrank": (round(psh, 1) if math.isfinite(psh) else ""),
        "pct_pairs_shrank_ci": _ci(psh_lo, psh_hi, 1),
        "pct_pairs_grew": (round(pgr, 1) if math.isfinite(pgr) else ""),
        "pct_pairs_grew_ci": _ci(pgr_lo, pgr_hi, 1),
        "present_orig_pct": round(po, 1), "present_orig_ci": _ci(po_lo, po_hi, 1),
        "present_collapsed_pct": round(pc, 1), "present_collapsed_ci": _ci(pc_lo, pc_hi, 1),
        "mcnemar_b_fixed": b, "mcnemar_c_created": cc, "mcnemar_p": _pfmt(mcn["p"]),
        "mcnemar_method": mcn["method"],
        "union_connected_pct": round(pconn, 1), "union_connected_ci": _ci(pconn_lo, pconn_hi, 1),
        "resolved_endpoint_available_n": int(endpoint_available.sum()),
        "resolved_endpoint_status": (
            "complete" if bool(endpoint_available.all())
            else ("partial" if bool(endpoint_available.any()) else "unavailable_10_column_schema")),
        "resolved_pct": (round(pres_pct, 1) if math.isfinite(pres_pct) else ""),
        "resolved_ci": _ci(pres_lo, pres_hi, 1),
        "topology_ok_pct": round(topology_ok_pct, 1),
        "topology_ok_present_n": ok_den, "topology_ok_resolved_n": ok_num,
        "topology_ok_resolved_pct": (round(ok_pct, 1) if math.isfinite(ok_pct) else ""),
        "topology_ok_resolved_ci": _ci(ok_lo, ok_hi, 1),
        "multicomponent_present_n": mc_den, "multicomponent_resolved_n": mc_num,
        "multicomponent_resolved_pct": (round(mc_pct, 1) if math.isfinite(mc_pct) else ""),
        "multicomponent_resolved_ci": _ci(mc_res_lo, mc_res_hi, 1),
    }
    summ = {"dataset": dir_name, "ds": ds, "n": n, "b": b, "c": cc,
            "mcnemar_p": mcn["p"], "delta_med": d_med,
            "delta_ci": ((float("nan"), float("nan")) if S.interval_is_degenerate(d_lo, d_hi)
                         else (d_lo, d_hi)),
            "connected_pct": pconn, "resolved_pct": pres_pct,
            "resolved_endpoint_available_n": int(endpoint_available.sum()),
            "topology_ok_resolved": (ok_num, ok_den, ok_pct),
            "multicomponent_resolved": (mc_num, mc_den, mc_pct),
            "present_orig_pct": po, "present_collapsed_pct": pc}

    print(f"  collapsed-class diagnostic [{mode}] (descriptive), n={n} paired:")
    print(f"    median gap  {np.median(orig):.2f} [{_ci(mo_lo, mo_hi, 2)}] mm  ->  "
          f"{np.median(coll):.2f} [{_ci(mcoll_lo, mcoll_hi, 2)}] mm  "
          f"(paired median delta {d_med:+.2f} [{_ci(d_lo, d_hi, 2)}] mm"
          + (f"; {100 * atom:.1f}% of pairs at the median" if d.size else "") + ")")
    if d.size:
        print(f"    pairs changed {pch:.1f}% [{_ci(pch_lo, pch_hi, 1)}]: gap shrank in {psh:.1f}% "
              f"[{_ci(psh_lo, psh_hi, 1)}], grew in {pgr:.1f}% [{_ci(pgr_lo, pgr_hi, 1)}]")
    print(f"    present     {po:.1f}% [{_ci(po_lo, po_hi, 1)}]  ->  "
          f"{pc:.1f}% [{_ci(pc_lo, pc_hi, 1)}]   McNemar b(fixed)={b} c(created)={cc} "
          f"p={_pfmt(mcn['p'])} ({mcn['method']})")
    print(f"    union connected (IVC-LCIV share a venous component): {pconn:.1f}% "
          f"[{_ci(pconn_lo, pconn_hi, 1)}]  (raw-union connectivity diagnostic; not the "
          "resolved-gap rate)")
    if n_op > 0:
        # n_resolved, NOT the McNemar b: b counts present->absent transitions under the proximity
        # rule alone, while the schema-v2 endpoint also requires post-re-split connectivity; count
        # and percentage must be the same quantity.
        print(f"    of the {n_op} pre-merge gaps carrying the connectivity "
              f"endpoint, {n_resolved} resolved (proximity AND post-re-split connectivity) "
              f"after the merge ({pres_pct:.1f}% [{_ci(pres_lo, pres_hi, 1)}])")
    print("    topology sensitivity (mask fragmentation flag; not proven anatomy): "
          f"OK {ok_num}/{ok_den} ({ok_pct:.1f}% [{_ci(ok_lo, ok_hi, 1)}]); "
          f"MULTI_COMPONENT {mc_num}/{mc_den} ({mc_pct:.1f}% "
          f"[{_ci(mc_res_lo, mc_res_hi, 1)}])")
    if counts.get("empty") or counts.get("failed") or counts.get("unparsed"):
        print(f"    input-empty={counts.get('empty', 0)}, "
              f"algorithmically unevaluable={counts.get('failed', 0)}, "
              f"unparsed={counts.get('unparsed', 0)}; reasons={failure_text or 'none'}")
    return [row], summ


# ---------------------------------------------------------------------------
# Console summary (axial vs 3D, with CIs, censoring, invariant, agreement)
# ---------------------------------------------------------------------------
def _desc_line(v):
    dsc = S.describe(v, alpha=C.ALPHA, n_boot=C.N_BOOT, seed=C.SEED,
                     ci_method=C.CI_METHOD)
    return (f"n={dsc['n']:3d} median={dsc['median']:5.1f} "
            f"(95%CI {_ci(dsc['median_ci_lo'], dsc['median_ci_hi'], 1)}) "
            f"IQR[{dsc['p25']:.1f}-{dsc['p75']:.1f}] "
            f"mean={dsc['mean']:5.1f} sd={dsc['sd']:4.1f} "
            f"skew={dsc['skew']:+.1f} p90={dsc['p90']:5.1f} max={dsc['max']:6.1f}")


def quantization_frac(v, step=5.0, tol=0.05):
    """Fraction of values that sit within tol of an integer multiple of `step`
    (a heaping fingerprint of the slice-thickness-quantized center-to-center
    metric). Reported as a diagnostic, not a correction."""
    v = np.asarray(v, float)
    if v.size == 0:
        return float("nan")
    nearest = np.round(v / step) * step
    return float((np.abs(v - nearest) <= tol).mean())


def summary(ds, dir_name, z, surf, c2c, counts, agreements, qc=None, sz=None,
            empty_strata=None, strat_name=None):
    """Console summary. MAGNITUDE statistics (median/present/band/distribution) use the
    surface-to-surface gap `surf`; GEOMETRY (axial/lateral split, the d>=z invariant) uses the
    center-to-center pair (z, c2c). `sz` is the per-case slice thickness (aligned with surf) for the
    resolution / slice-normalized analysis. The name is required, not cosmetic: whether an
    anatomical-coverage mechanism may even be proposed depends on WHICH variable concentrated, and
    the caller is the only one that knows."""
    n = len(z)
    n_empty, n_unparsed = counts.get("empty", 0), counts.get("unparsed", 0)
    n_failed = counts.get("failed", 0)
    n_total = n + n_empty + n_failed + n_unparsed
    metric = counts.get("metric", "surface(v2)")
    print("=" * 78)
    print(f"{dir_name}: confluence gap — 3D surface-to-surface (PRIMARY) vs axial z_gap")
    print("=" * 78)
    print(f"  PRIMARY summary: {C.PRIMARY_INFERENCE}.")
    print("  Everything else below (correlates, agreement, geometry, heterogeneity,")
    print("  band/present-rate CIs) is DESCRIPTIVE and labelled as such.")
    if metric != "surface(v2)":
        print("  *** WARNING: results file is in the v1 format (centre-to-centre metric, "
              "no c2c column). Run measure_all.py on the masks to get the "
              "surface-to-surface primary metric. Numbers below use c2c. ***")
    elig = C.COHORT_ELIGIBLE.get(ds)
    print("exclusion accounting (dataset eligible -> processed -> analyzed):")
    if elig:
        label, n_elig, note = elig
        print(f"  eligible ({label}): {n_elig}")
        print(f"    selection -> processed: {note}")
    print(f"  processed case lines  : {n_total}")
    empty_rate = f"({100 * n_empty / n_total:.1f}%)" if n_total else ""
    print(f"  - empty (no IVC/LCIV) : {n_empty} {empty_rate}")
    if empty_strata and n_empty:
        brk = ", ".join(f"{k}={v}" for k, v in sorted(empty_strata.items(),
                                                       key=lambda kv: -kv[1]))
        print(f"      empties by stratum: {brk}")
        top_k, top_v = max(empty_strata.items(), key=lambda kv: kv[1])
        if n_empty >= C.MIN_GROUP_N and top_v / max(1, n_empty) >= 0.8:
            # Empties concentrated in one level of an observed stratifier are consistent with MAR
            # given that stratifier; this does not identify MNAR, which means dependence on the
            # unobserved gap itself. Only `region` (SAROS) is a coverage variable, so a field-of-view
            # mechanism is offered only there (`STRATIFIER` is `scanner` for AMOS and autoPET). With
            # the mechanism unidentified, the unobserved gaps can move the median either way.
            region_like = strat_name in ("region", "body_region")
            print(f"      <-- NON-RANDOM MISSINGNESS (descriptive): {100*top_v/n_empty:.0f}% "
                  f"of empties fall in the '{top_k}' level of the OBSERVED stratifier "
                  f"'{strat_name}'.")
            print(f"          This is concentration in a RECORDED variable, i.e. consistent "
                  f"with MAR given '{strat_name}'. It does NOT identify MNAR, which would "
                  f"require dependence on the unmeasured gap itself, and no analysis here "
                  f"tests that.")
            if region_like:
                print(f"          A pelvic-FOV truncation is a PLAUSIBLE mechanism for a "
                      f"'{top_k}' concentration, and it is a hypothesis: nothing in this "
                      f"pipeline measures field of view or verifies it.")
            else:
                print(f"          '{strat_name}' is not a coverage variable, so no "
                      f"anatomical-coverage mechanism is proposed for this concentration.")
            print(f"          The '{top_k}' / pooled median is CONDITIONAL ON MEASURABILITY. "
                  f"Its direction relative to the full-cohort median is NOT identified; the "
                  f"only bound is the two-sided worst case printed in the cross-cohort "
                  f"summary.")
    print(f"  - failed (read/producer error) : {n_failed}"
          + ("  <-- INVESTIGATE (data integrity)" if n_failed else ""))
    if n_failed and counts.get("failure_reasons"):
        brk = ", ".join(f"{k}={v}" for k, v in counts["failure_reasons"].items())
        print(f"      failure reasons: {brk}")
    print(f"  - unparsed (bad row)  : {n_unparsed}"
          + ("  <-- INVESTIGATE" if n_unparsed else ""))
    print(f"  = measurable          : {n}")
    qc_small = int(counts.get("qc_small", 0))
    qc_overlap = int(counts.get("qc_overlap", 0))
    print(f"  degenerate masks (SMALL_IVC/SMALL_LCIV, <{C.QC_MIN_VOXELS} vox): "
          f"{qc_small} (retained, flagged)")
    print(f"  3D label-overlap masks (OVERLAP, surf=0, seg error): "
          f"{qc_overlap} (retained, flagged)"
          + ("  <-- depresses median/present; see sensitivity" if qc_overlap else ""))
    print(f"  3D surface gap (PRI): {_desc_line(surf)}")
    print(f"  3D center-to-center : {_desc_line(c2c)}")
    print(f"  axial z_gap (cens.) : {_desc_line(z)}")
    cens = censored_fraction(z)
    print(f"  axial z_gap left-censored at 0: {cens:.1f}% of measurable "
          f"(reported descriptively; not used for parametric inference)")
    # sensitivity: drop physiologically implausible QC outliers (>QC_MAX_GAP_MM)
    qc_big = int((surf > C.QC_MAX_GAP_MM).sum())
    if qc_big:
        keep = surf[surf <= C.QC_MAX_GAP_MM]
        print(f"  sensitivity (drop {qc_big} QC>{C.QC_MAX_GAP_MM:.0f}mm outlier(s)): "
              f"surf {_desc_line(keep)}")
    # sensitivity: drop 3-D label-overlap (OVERLAP) seg-error cases (surf forced 0)
    if qc_overlap and qc is not None and len(qc) == surf.size:
        keep = surf[np.array(["OVERLAP" not in f for f in qc])]
        print(f"  sensitivity (drop {qc_overlap} OVERLAP seg-error case(s)): "
              f"surf {_desc_line(keep)}")
    # zero-inflation: the primary surface gap is floored at 0 (touching/overlap), so
    # report the exact-zero fraction explicitly (it is NOT an uncensored continuous
    # magnitude; median + present-rate + zero-fraction is the honest summary).
    zero_frac = 100.0 * float((surf <= 0).mean()) if n else float("nan")
    print(f"  surface zero-fraction (touching/overlap, gap=0): {zero_frac:.1f}% "
          f"(metric is zero-inflated/left-censored at the resolution floor)")
    # present-rate with Wilson CI (touching masks read 0 on the surface gap)
    pk = int((surf > C.PRESENT_MM).sum())
    pp, plo, phi = S.proportion_pct_ci(pk, n, C.ALPHA)
    bk = int(((surf >= C.BAND_LO_MM) & (surf <= C.BAND_HI_MM)).sum())
    bp, blo, bhi = S.proportion_pct_ci(bk, n, C.ALPHA)
    print(f"  present(>{C.PRESENT_MM}mm surface): {pk}/{n} ({pp:.0f}%, 95%CI {_ci(plo, phi, 0)})"
          f"   ~5mm band({C.BAND_LO_MM}-{C.BAND_HI_MM}): {bp:.0f}% (95%CI {_ci(blo, bhi, 0)})")
    sweep = "  ".join(f">{t}mm:{100*float((surf > t).mean()):.0f}%"
                      for t in C.PRESENT_SWEEP_MM)
    print(f"  present-rate threshold sweep (measurable): {sweep}")
    # DUAL present-rate denominator: among all PROCESSED cases the empties are not
    # assessable (no masks), so present-among-processed bounds the rate when empty
    # rates differ across cohorts (autoPET 0% vs SAROS 15%).
    if n_total:
        pp_all = 100.0 * pk / n_total
        print(f"  present among ALL processed (empties not-assessable): "
              f"{pk}/{n_total} ({pp_all:.0f}%)  [vs {pp:.0f}% among measurable]")
    # Scale-invariant axial/lateral split on the CENTER-TO-CENTER pair (z, c2c), among the
    # cases the PRIMARY metric says have a gap at all. `pm` is `surf`, not `c2c`:
    # presence read off the centroid separation would make `absent` unreachable and put every
    # touching mask into the direction denominator.
    pm = surf > C.PRESENT_MM
    npr = int(pm.sum())
    cats = [classify_gap(zz, dd, ss) for zz, dd, ss in zip(z, c2c, surf)]
    lat = lateral_residual(z[pm], c2c[pm])
    frac = z[pm] / c2c[pm] if npr else np.array([])
    if npr:
        print(f"  gap geometry (direction on c2c, presence on the PRIMARY surface gap; "
              f"{npr}/{n} present, {cats.count('absent')} absent): "
              f"axial {100*cats.count('axial')/npr:.0f}%  "
              f"lateral {100*cats.count('lateral')/npr:.0f}%  "
              f"mixed {100*cats.count('mixed')/npr:.0f}%   "
              f"[percentages are among PRESENT cases]")
        print(f"  median lateral leg (upper bound, present only): {np.median(lat):.1f} mm   "
              f"median axial fraction z/c2c: "
              f"{np.median(frac) if frac.size else float('nan'):.2f}")
    else:
        print(f"  gap geometry: no case has a gap on the primary metric "
              f"(0/{n} present); direction split not defined")
    # For an axial-dominated gap surf = (#void slices)*sz, so it heaps on slice multiples.
    szc = None if sz is None else np.asarray(sz, float)
    if szc is not None and np.isfinite(szc).any():
        qf_surf = quantization_frac_percase(surf, szc, tol=0.05)
        qf_c2c = quantization_frac_percase(c2c, szc, tol=0.05)
        qf_surf_nz = quantization_frac_percase(surf, szc, tol=0.05, exclude_zero=True)
        okz = np.isfinite(surf) & np.isfinite(szc) & (szc > 0)
        zero_share = 100.0 * float((surf[okz] == 0.0).mean()) if okz.any() else float("nan")
        print(f"  quantization (PRIMARY surf, per-case slice step): {100*qf_surf:.0f}% of "
              f"surf gaps within 0.05mm of a slice-thickness multiple (c2c {100*qf_c2c:.0f}%); "
              f"slice-thickness heaping -> medians are resolution-quantized")
        # The figure above counts the exact-zero atom, which is a multiple of every step and is not
        # evidence of heaping at all. Reported BOTH ways, with the share that drives the difference,
        # so the fingerprint can be read without the artifact.
        print(f"  quantization EXCLUDING the exact-zero atom: {100*qf_surf_nz:.0f}% "
              f"(zero share {zero_share:.0f}% of measurable; the atom accounts for "
              f"{100*(qf_surf - qf_surf_nz):+.0f} points of the figure above). The "
              f"zero-excluded figure is the resolution fingerprint; the inclusive one is not.")
        # SLICE-NORMALIZED diagnostic (surf/slice). This probes through-plane sampling
        # dependence; it neither removes in-plane resolution nor isolates acquisition
        # from cohort, contrast, scanner, or model-path differences.
        su, n_su = slice_normalized(surf, szc)
        if su.size:
            lo, hi = S.median_ci_auto(su, C.CI_METHOD, C.ALPHA, C.N_BOOT, C.SEED)
            print(f"  slice-normalized gap (surf/slice): median {np.median(su):.2f} "
                  "slice-thickness units "
                  f"(95%CI {_ci(lo, hi, 2)}), n={n_su}  "
                  "<-- acquisition-scaled sensitivity diagnostic")
        # WITHIN-slice-thickness stratification — an acquisition-scaling diagnostic:
        # ~constant in mm across thicknesses => a real fixed buffer; ~constant in slices
        # (mm grows with thickness) => a discretization artifact.
        bands = {}
        for s, g in zip(szc, surf):
            if np.isfinite(s) and s > 0:
                bands.setdefault(round(float(s), 2), []).append(g)
        bands = {k: np.array(v) for k, v in bands.items() if len(v) >= C.MIN_GROUP_N}
        if len(bands) >= 2:
            print("  WITHIN-slice-thickness (artifact[~const in slices] vs "
                  "buffer[~const in mm]):")
            for s in sorted(bands):
                g = bands[s]
                print(f"    slice {s:>4.2f}mm: n={len(g):4d}  median {np.median(g):5.2f}mm "
                      f"= {np.median(g)/s:.2f} slice-thickness units")
    else:
        qf = quantization_frac(c2c, step=5.0, tol=0.05)
        print(f"  quantization: {100*qf:.0f}% of c2c gaps within 0.05mm of a 5mm multiple "
              f"(no per-case slice file; slice-thickness heaping, medians resolution-quantized)")
    # invariants: c2c>=z (center-based, must hold) and surf<=c2c (de-biasing)
    viol = int((z - c2c > C.INVARIANT_SLACK_MM).sum())
    viol2 = int((surf - c2c > C.INVARIANT_SLACK_MM).sum())
    print(f"  QC: surface gap > {C.QC_MAX_GAP_MM:.0f}mm: {qc_big}   "
          f"invariant c2c>=z: {'PASS' if viol == 0 else f'FAIL ({viol})'}   "
          f"surf<=c2c: {'PASS' if viol2 == 0 else f'FAIL ({viol2})'}")
    # normality caveat
    dsc = S.describe(surf, alpha=C.ALPHA, n_boot=C.N_BOOT, seed=C.SEED,
                     ci_method=C.CI_METHOD)
    if dsc["shapiro_p"] is not None and dsc["shapiro_p"] < 0.05:
        print(f"  note: surface gap non-normal (Shapiro p={dsc['shapiro_p']:.3g}, "
              f"skew={dsc['skew']:+.1f}) -> median/IQR are the primary descriptors")
    # method agreement (with non-parametric LoA + assumption diagnostics)
    if agreements:
        print("  method agreement (fast 3mm vs full-res):")
        for a in agreements:
            dnp = a.get("diff_normal_p")
            print(f"    n={a['n']} CCC={a['lin_ccc']:.3f} ICC(A,1)={a['icc_a1']:.3f} "
                  f"bias={a['bias_mm']:+.2f}mm")
            print(f"      LoA parametric[{a['loa_lo_mm']:+.1f},{a['loa_hi_mm']:+.1f}] "
                  f"non-param[{a['loa_lo_np_mm']:+.1f},{a['loa_hi_np_mm']:+.1f}]  "
                  f"diff-normal p={_pfmt(dnp)} prop-bias r={a['prop_bias_r']:+.2f}")
            print(f"      -> primary LoA = {a['loa_primary_kind']} "
                  f"[{a['loa_primary_lo_mm']:+.1f},{a['loa_primary_hi_mm']:+.1f}] "
                  f"({a['loa_primary_reason']})")
            if dnp is not None and dnp < 0.05:
                print("      NOTE: fast-vs-full changes the TotalSegmentator inference "
                      "path as well as working resolution. The directional bias is "
                      "consistent with a coarse-resolution floor but does not isolate it.")


# ---------------------------------------------------------------------------
# Correlates (effect size + Wilson present-rate + BH-FDR + transparency)
# ---------------------------------------------------------------------------
VARIABLES = {
    "sex": lambda r: r.get("sex"),
    "scanner": lambda r: r.get("scanner"),
    "age_band": lambda r: age_band(r.get("age")),
    "contrast": lambda r: r.get("contrast"),
    "region": lambda r: r.get("region"),
    # Phase-1 correlates from the aux file (populated when present):
    "patient_size": lambda r: r.get("size_band"),     # body-CSA tertiles (TECHNICAL:
                                                      # acquisition/FOV/resolution proxy,
                                                      # see gap_config.CONFOUNDED_VARIABLES)
    "seg_quality": lambda r: r.get("seg_quality"),    # clean/fragmented (technical)
}


def _eps_mag(eps):
    """Qualitative magnitude for epsilon-squared (Cohen/Tomczak mapping):
    <0.01 negligible, <0.06 small, <0.14 medium, else large. Forces the reader
    to weigh effect size, not just the n-driven p-value."""
    if eps is None or (isinstance(eps, float) and math.isnan(eps)):
        return "NA"
    if eps < 0.01:
        return "negligible"
    if eps < 0.06:
        return "small"
    if eps < 0.14:
        return "medium"
    return "large"


_EPS_CUTS = ((0.01, "negligible"), (0.06, "small"), (0.14, "medium"))


def _eps_label_ci(eps, lo, hi):
    """Magnitude label for eps^2, BRACKETED across cutpoints when the bootstrap CI
    [lo, hi] straddles a Cohen/Tomczak boundary (0.01/0.06/0.14). 'large' at 0.143
    whose CI dips below 0.14 becomes 'medium-large', not an asserted 'large'."""
    if eps is None or (isinstance(eps, float) and math.isnan(eps)):
        return "NA"
    if lo is None or hi is None or (isinstance(lo, float) and math.isnan(lo)):
        return _eps_mag(eps)
    lab_lo, lab_hi = _eps_mag(lo), _eps_mag(hi)
    return lab_lo if lab_lo == lab_hi else f"{lab_lo}-{lab_hi}"


def _level_stats(vals):
    v = np.asarray(vals, float)
    lo, hi = S.median_ci_auto(v, C.CI_METHOD, C.ALPHA, C.N_BOOT, C.SEED)
    pk = int((v > C.PRESENT_MM).sum())
    pp, plo, phi = S.proportion_pct_ci(pk, len(v), C.ALPHA)
    return dict(n=len(v), median=round(float(np.median(v)), 1),
                median_ci=_ci(lo, hi, 1),
                p25=round(float(np.percentile(v, 25)), 1),
                p75=round(float(np.percentile(v, 75)), 1),
                mean=round(float(v.mean()), 1), sd=round(S.sample_sd(v), 1),
                pct_present=round(pp, 1))


def correlates(dir_name, fullres_rows, meta, analysis_unit=None,
               inference_scope=None):
    """Returns (csv_rows, panels). Metric = full-res 3D gap (the PRIMARY outcome).

    Omnibus Kruskal-Wallis per variable, with an epsilon-squared effect size and
    a Benjamini-Hochberg q-value computed across the per-cohort family of tests.
    Dropped levels (n < MIN_GROUP_N) and the join rate are reported, not hidden.

    autoPET's committed rows are study-level repeats. A separate call with
    ``analysis_unit='subject'`` is used for the deterministic earliest-study-per-subject sensitivity
    artifact."""
    is_autopet = "autopet" in dir_name.lower()
    if analysis_unit is None:
        analysis_unit = "study" if is_autopet else "case"
    if inference_scope is None:
        if is_autopet and analysis_unit == "study":
            inference_scope = "descriptive_repeated_studies_not_cluster_adjusted"
        elif is_autopet and analysis_unit == "subject":
            inference_scope = "subject_level_sensitivity_earliest_study"
        else:
            inference_scope = "descriptive"
    gaps = {r[0]: r[2] for r in fullres_rows}      # surface-to-surface (PRIMARY)
    n_join = sum(1 for cid in gaps if cid in meta)
    print(f"  correlates (PRIMARY metric=3D surface gap; unit={analysis_unit}; "
          f"scope={inference_scope}): {len(gaps)} measurable, "
          f"{n_join} metadata-joined ({100*n_join/max(1,len(gaps)):.0f}%)")
    if is_autopet and analysis_unit == "study":
        print("    [autoPET repeated-study caveat: study-level inference is descriptive "
              "and is not patient-cluster adjusted. KW p-values, epsilon-squared, and "
              "its ordinary bootstrap CI all resample studies and are not cluster-robust. "
              "Use autopet_subject_sensitivity.csv for the independent-unit check.]")
    elif is_autopet and analysis_unit == "subject":
        print("    [autoPET subject sensitivity: one chronologically earliest study "
              "selected per subject before measurability filtering; independent analysis "
              "units under the subject design.]")
    overall = _level_stats(list(gaps.values()))
    out = [dict(dataset=dir_name, variable="overall", level="all",
                analysis_unit=analysis_unit, inference_scope=inference_scope,
                pvalue="", eps_squared="", qvalue="", **overall)]

    tested = []   # (var, levels_dict, pval, eps)
    for var, getter in VARIABLES.items():
        levels, dropped = {}, {}
        for cid, g in gaps.items():
            rec = meta.get(cid)
            lv = getter(rec) if rec else None
            if lv is not None:
                levels.setdefault(str(lv), []).append(g)
        for k in list(levels):
            if len(levels[k]) < C.MIN_GROUP_N:
                dropped[k] = len(levels.pop(k))
        if dropped:
            drop = ", ".join(f"{k}(n={v})" for k, v in dropped.items())
            print(f"    {var:9s} dropped low-n levels: {drop}")
        if len(levels) < 2:
            print(f"    {var:9s} skipped (<2 testable levels)")
            continue
        kw = S.kruskal_test(*levels.values())
        tested.append((var, levels, kw["p"], kw["eps_squared"]))

    # Benjamini-Hochberg across the per-cohort family of tests actually run.
    # The family size m is reported so a 2-3 test "q" is not over-read as strong
    # multiplicity control; in `all` mode main() also pools the family across
    # cohorts (see cross_cohort / pooled_fdr).
    # FDR is computed WITHIN each family separately: the biological correlates
    # (sex/age/contrast) and the CONFOUNDED technical stratifiers (scanner/region) are
    # distinct families, so a technical test neither dilutes nor inflates a biological
    # q (and vice versa). Each q is reported with its family label and size.
    qmap, fam = {}, {}
    for members, label in (
            ([t for t in tested if t[0] not in C.CONFOUNDED_VARIABLES], "bio"),
            ([t for t in tested if t[0] in C.CONFOUNDED_VARIABLES], "tech")):
        for (var, _, _, _), q in zip(members, S.bh_fdr([p for _, _, p, _ in members])):
            qmap[var] = q
            fam[var] = (label, len(members))
    panels = []
    cohort_tests = []   # (dir_name, var, p, eps, elo, ehi) for cross-cohort pooled FDR
    for (var, levels, pval, eps) in tested:
        q = qmap.get(var)
        fam_label, m = fam.get(var, ("?", 0))
        if var in C.CONFOUNDED_VARIABLES:
            conf = "  [CONFOUNDED -> technical stratifier, not a biological correlate]"
        else:
            conf = ""
        e0, elo, ehi = S.kruskal_eps2_ci(list(levels.values()), C.ALPHA,
                                         C.N_BOOT, C.SEED)
        cohort_tests.append((dir_name, var, pval, eps, elo, ehi))
        eps2_ci = _ci(elo, ehi, 3)
        elabel = _eps_label_ci(eps, elo, ehi)
        for lv in sorted(levels, key=lambda k: -np.median(levels[k])):
            out.append(dict(dataset=dir_name, variable=var, level=lv,
                            analysis_unit=analysis_unit,
                            inference_scope=inference_scope,
                            pvalue=_pfmt(pval),
                            eps_squared=(round(eps, 3) if eps is not None else ""),
                            eps2_ci=eps2_ci, eps2_label=elabel,
                            qvalue=_pfmt(q),
                            **_level_stats(levels[lv])))
        panels.append((var, levels))
        spread = "  ".join(f"{lv}:{np.median(v):.1f}"
                           for lv, v in sorted(levels.items()))
        pstr = _pfmt(pval) or "NA"
        qstr = _pfmt(q) or "NA"
        estr = (f"{eps:.3f} [{eps2_ci}] ({elabel})" if eps is not None else "NA")
        print(f"    {var:9s} p={pstr} q(BH-{fam_label},m={m})={qstr} eps^2={estr}{conf}")
        print(f"              median/level -> {spread}")
    print("    note: KW is a stochastic-dominance test, NOT a test of medians "
          "(distributions differ in shape/spread); per-level medians are "
          "DESCRIPTIVE (pairwise would need Dunn + correction); lead with eps^2, "
          "not p. CIs are per-estimate, not simultaneous-coverage.")
    return out, panels, cohort_tests


# ---------------------------------------------------------------------------
# Figures (lazy matplotlib; dataset-general)
# ---------------------------------------------------------------------------
def _mpl():
    """Matplotlib with sober, publication-grade defaults: sans-serif, thin axes,
    no top/right spines, no gridlines, restrained type sizes, 300-dpi export."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "axes.titleweight": "regular", "axes.titlepad": 6,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "legend.frameon": False,
        "lines.solid_capstyle": "round",
    })
    return plt


# Sober near-monochrome academic palette (colorblind-safe): one ink, one restrained
# accent, the rest neutral grays. INK = primary data; ACCENT = the single highlighted
# reference (median / bias / 5 mm); SECOND = the secondary series; BAR = histogram fill.
_INK, _ACCENT, _SECOND = "#2b3a4a", "#9e2a2b", "#5f6b76"
_BAR, _BAR_EDGE, _MUTE = "#dde1e6", "#aab0b7", "#8c949e"


def _kde_reflected(v, xs, binw, cap):
    """KDE with reflection at 0 to remove boundary leakage for a zero-bounded
    quantity, scaled to histogram counts. None if too few distinct values."""
    try:
        from scipy.stats import gaussian_kde
    except Exception:
        return None
    vv = v[(v >= 0) & (v <= cap)]
    if len(np.unique(vv)) < 3:
        return None
    refl = np.concatenate([vv, -vv])           # reflect across 0
    dens = gaussian_kde(refl)(xs) * 2.0        # fold the negative half back
    dens[xs < 0] = 0.0
    return dens * len(vv) * binw


def figures(dir_name, z, d, panels, outdir, agreements=None, cap=None, c2c=None):
    cap = C.LARGE_GAP_MM if cap is None else cap
    try:
        plt = _mpl()
    except Exception as e:
        print(f"  [figures skipped: matplotlib unavailable — {e}]")
        return
    outdir.mkdir(parents=True, exist_ok=True)
    INK, ACCENT, SECOND, BAR, BAR_EDGE, MUTE = (_INK, _ACCENT, _SECOND, _BAR,
                                                _BAR_EDGE, _MUTE)
    bins = np.arange(0, cap + 1.5, 1.5)
    n = len(d)
    med, p25, p75 = np.median(d), np.percentile(d, 25), np.percentile(d, 75)
    mlo, mhi = S.median_ci_auto(d, C.CI_METHOD, C.ALPHA, C.N_BOOT, C.SEED)
    pk = int((d > C.PRESENT_MM).sum())
    pp, plo, phi = S.proportion_pct_ci(pk, n, C.ALPHA)
    n_over = int((d > cap).sum())

    # 1. distribution of the PRIMARY 3D gap
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.hist(d[d <= cap], bins=bins, color=BAR, edgecolor="white", lw=0.5, zorder=2)
    xs = np.linspace(0, cap, 400)
    kc = _kde_reflected(d, xs, 1.5, cap)
    if kc is not None:
        ax.plot(xs, kc, color=INK, lw=1.3, zorder=3)
    ax.axvspan(p25, p75, color=INK, alpha=0.06, zorder=1)
    ax.axvline(med, color=ACCENT, lw=1.3, zorder=4)
    ax.text(0.97, 0.95,
            _distribution_annotation(pk, n, pp, plo, phi, med, mlo, mhi, p25, p75,
                                     d.mean(), S.sample_sd(d)),
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            color=INK, linespacing=1.5,
            bbox=dict(boxstyle="square,pad=0.5", fc="white", ec=BAR_EDGE, lw=0.5))
    if n_over:
        ax.annotate(f"+{n_over} case(s) > {cap:.0f} mm (retained in statistics)",
                    xy=(cap, ax.get_ylim()[1] * 0.10), ha="right",
                    fontsize=6.5, color=MUTE)
    ax.set_xlim(0, cap)
    ax.set_xlabel("IVC–LCIV gap, 3D nearest distance (mm)")
    ax.set_ylabel("cases")
    ax.set_title(f"{dir_name}: confluence gap distribution")
    fig.text(0.01, -0.02, f"Histogram truncated at {cap:.0f} mm for display; all "
             f"statistics over n = {n} measurable cases. KDE reflection-corrected at 0.",
             fontsize=6, color=MUTE)
    _FIGIO.savefig_atomic(fig, outdir / "distribution.png")
    plt.close(fig)

    # 2. axial vs 3D
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    a = axes[0]
    a.hist(d[d <= cap], bins=bins, color=SECOND, alpha=0.40, edgecolor="white", lw=0.4,
           label=f"3D distance (median {np.median(d):.1f})")
    a.hist(z[z <= cap], bins=bins, color=INK, alpha=0.55, edgecolor="white", lw=0.4,
           label=f"axial z-gap (median {np.median(z):.1f}, censored)")
    a.axvline(np.median(z), color=INK, lw=1.1, ls="--")
    a.axvline(np.median(d), color=SECOND, lw=1.1, ls="--")
    a.set_xlim(0, cap)
    a.set_xlabel("gap (mm)")
    a.set_ylabel("cases")
    a.set_title("Axial vs 3D gap")
    a.legend(loc="upper right")
    a = axes[1]
    # The axial FRACTION is z_gap / centre-to-centre distance, which is the denominator the tabular
    # median_axial_fraction uses and the only one for which a y = x "purely axial" reference means
    # anything. `d` is the fallback for callers that do not supply c2c.
    ref = d if c2c is None else np.asarray(c2c, float)
    a.scatter(ref, z, s=12, color=INK, alpha=0.50, edgecolor="white", lw=0.3)
    a.plot([0, cap], [0, cap], ls="--", color=MUTE, lw=1.0, label="y = x (purely axial)")
    a.set_xlim(0, cap)
    a.set_ylim(0, cap)
    a.set_aspect("equal", adjustable="box")
    a.set_xlabel("centre-to-centre distance (mm)" if c2c is not None
                 else "3D nearest distance (mm)")
    a.set_ylabel("axial z-gap (mm)")
    a.set_title("Axial fraction of the gap")
    a.legend(loc="upper left")
    fig.suptitle(f"{dir_name}: is the confluence gap axial or lateral?", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _FIGIO.savefig_atomic(fig, outdir / "axial_vs_3d.png")
    plt.close(fig)

    # 3. ECDF
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for v, c, ls, lab in [(z, INK, "-", "axial z-gap"), (d, SECOND, "-", "3D distance")]:
        xs = np.sort(v)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ys, where="post", color=c, lw=1.4, ls=ls, label=lab)
    ax.axvline(5, color=ACCENT, lw=1.0, ls=":", label="5 mm")
    ax.set_xlim(0, cap)
    ax.set_ylim(0, 1)
    ax.set_xlabel("gap (mm)")
    ax.set_ylabel("proportion of cases ≤ x")
    ax.set_title(f"{dir_name}: cumulative distribution")
    ax.legend(loc="lower right")
    _FIGIO.savefig_atomic(fig, outdir / "ecdf.png")
    plt.close(fig)

    # 4. correlate boxes (jitter seeded for reproducibility)
    panels = [(v, lv) for v, lv in panels if lv]
    if panels:
        cols = min(3, len(panels))
        rows = (len(panels) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(2.45 * cols, 2.3 * rows),
                                 squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for i, (var, levels) in enumerate(panels):
            ax = axes[i // cols][i % cols]
            ax.axis("on")
            labels = sorted(levels, key=lambda k: -np.median(levels[k]))
            ax.boxplot([levels[k] for k in labels], showfliers=False, widths=0.5,
                       medianprops=dict(color=ACCENT, lw=1.3),
                       boxprops=dict(color=INK, lw=0.8),
                       whiskerprops=dict(color=INK, lw=0.8),
                       capprops=dict(color=INK, lw=0.8))
            for j, k in enumerate(labels, 1):
                vals = levels[k]
                jit = np.random.default_rng(C.SEED + j).random(len(vals))
                xs = j + (jit - 0.5) * 0.26
                ax.scatter(xs, vals, s=5, alpha=0.30, color=INK, lw=0, zorder=3)
            ax.set_xticks(range(1, len(labels) + 1))
            def _short(k):
                k = str(k)
                return k if len(k) <= 14 else k[:13] + "…"
            ticklabs = [f"{_short(k)}\n(n={len(levels[k])})" for k in labels]
            longish = (len(labels) > 3) or any(len(str(k)) > 8 for k in labels)
            ax.set_xticklabels(ticklabs, fontsize=6.5,
                               rotation=30 if longish else 0,
                               ha="right" if longish else "center")
            ax.set_ylim(0, cap)
            ax.set_ylabel("3D gap (mm)")
            ax.set_title(var.replace("_", " "))
        fig.suptitle(f"{dir_name}: gap by correlate (3D, capped at {cap:.0f} mm)",
                     fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        _FIGIO.savefig_atomic(fig, outdir / "correlates.png")
        plt.close(fig)

    # (Bland-Altman is rendered by bland_altman_figure() from the raw paired data)
    print(f"  wrote figures -> {outdir}")


def bland_altman_figure(dir_name, fast, full, outdir):
    """Bland-Altman panel showing BOTH limit-of-agreement estimators: the pair the declared rule
    selects is emphasised, the other is drawn faintly so a reader can see the difference the choice
    makes, and the diagnostics that drive the rule are printed under the axes.

    Returns the values it drew, so a test can bind the figure to `statlib.bland_altman`
    instead of trusting the picture. Returns None when no figure was produced.
    """
    try:
        plt = _mpl()
    except Exception:
        return None
    if len(fast) < 2:
        return None
    outdir.mkdir(parents=True, exist_ok=True)
    fast = np.asarray(fast, float)
    full = np.asarray(full, float)
    mean = (fast + full) / 2
    diff = fast - full
    ba = S.bland_altman(fast, full, C.ALPHA)
    bn = S.binned_empirical_loa(fast, full, alpha=C.ALPHA)
    sel = S.primary_loa(ba, C.ALPHA, bn)
    primary_np = sel["kind"] == "nonparametric"

    par = [(ba["loa_lo"], f"−1.96 SD {ba['loa_lo']:+.2f}"),
           (ba["loa_hi"], f"+1.96 SD {ba['loa_hi']:+.2f}")]
    npm = [(ba["loa_lo_np"], f"2.5 pct {ba['loa_lo_np']:+.2f}"),
           (ba["loa_hi_np"], f"97.5 pct {ba['loa_hi_np']:+.2f}")]
    primary, secondary = (npm, par) if primary_np else (par, npm)
    tag = "distribution-free" if primary_np else "parametric"
    other = "parametric" if primary_np else "distribution-free"

    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    ax.scatter(mean, diff, s=13, color=_INK, alpha=0.5, edgecolor="white", lw=0.3)
    ax.axhline(0, color=_MUTE, lw=0.6, ls="-", zorder=1)
    ax.axhline(ba["bias"], color=_ACCENT, lw=1.3, ls="-",
               label=f"bias {ba['bias']:+.2f} mm")
    # `drawn` is appended to HERE, as each line is actually plotted, and is what the function
    # returns, so a test can detect a line that was not drawn.
    drawn = []
    for i, (y, lab) in enumerate(primary):
        ax.axhline(y, color=_SECOND, lw=1.2, ls="--",
                   label=(f"{tag} LoA: {lab}" if i == 0 else lab))
        drawn.append(float(y))
    for i, (y, lab) in enumerate(secondary):
        ax.axhline(y, color=_MUTE, lw=0.9, ls=":",
                   label=(f"{other} LoA: {lab}" if i == 0 else lab))
        drawn.append(float(y))

    # A constant-width interval is a horizontal line by construction and so cannot show what these
    # show. `bands_drawn` is appended to HERE, as each segment is plotted, for the same reason
    # `drawn` is: a seam recomputed afterwards cannot detect a figure that omitted a band.
    bands_drawn = []
    if sel["kind"] == "binned_conditional":
        for j, band in enumerate(bn["bins"]):
            xs = [band["m_lo"], band["m_hi"]]
            ax.plot(xs, [band["loa_lo"]] * 2, color=_ACCENT, lw=1.6, ls="-", zorder=4,
                    label="conditional LoA (per band)" if j == 0 else None)
            ax.plot(xs, [band["loa_hi"]] * 2, color=_ACCENT, lw=1.6, ls="-", zorder=4)
            bands_drawn.append((float(band["loa_lo"]), float(band["loa_hi"])))

    ax.set_xlabel("mean of fast (3 mm) and full-res 3D gap (mm)")
    ax.set_ylabel("difference, fast − full-res (mm)")
    # The legend sits above the plotting area and the title is lifted by its measured height, so
    # neither can cover a point.
    legend = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.5,
                       frameon=False, borderaxespad=0.2)
    legend_pt = legend.get_window_extent(fig.canvas.get_renderer()).height * 72.0 / fig.dpi
    ax.set_title(f"{dir_name}: Bland–Altman, fast (3 mm) vs full-res "
                 f"(CCC = {S.lin_ccc(fast, full):.3f})", pad=legend_pt + 4.0)

    p = ba.get("diff_normal_p")
    r = ba.get("prop_bias_r")
    line1 = (f"n = {ba['n']};   Shapiro-Wilk p on differences = "
             + ("n/a" if p is None else f"{p:.2g}")
             + ";   proportional bias r = "
             + ("n/a" if r is None or (isinstance(r, float) and math.isnan(r))
                else f"{r:+.3f}"))
    line2 = ("Emphasised: conditional LoA per band of mean gap. Both constant-width "
             "pairs are drawn faintly, because neither can be conditionally valid here."
             if sel["kind"] == "binned_conditional" else
             f"Emphasised: {tag} limits. Both pairs are drawn because the methods "
             f"default to the distribution-free pair when the differences are non-normal.")
    fig.text(0.012, 0.055, line1, fontsize=6.0, color=_MUTE, va="bottom")
    fig.text(0.012, 0.012, line2, fontsize=6.0, color=_MUTE, va="bottom")
    fig.subplots_adjust(bottom=0.22)

    _FIGIO.savefig_atomic(fig, outdir / "bland_altman_fast_vs_full.png")
    plt.close(fig)
    return {"n": ba["n"], "bias": ba["bias"],
            "loa_lo": ba["loa_lo"], "loa_hi": ba["loa_hi"],
            "loa_lo_np": ba["loa_lo_np"], "loa_hi_np": ba["loa_hi_np"],
            "primary_kind": sel["kind"], "primary_lo": sel["lo"], "primary_hi": sel["hi"],
            "drawn": sorted(round(y, 10) for y in drawn),
            "bands_drawn": bands_drawn}


# Per-cohort technical stratifier (acquisition/FOV proxy) for the primary
# distribution. SAROS 'region' = abdomen vs wholebody (resolution/FOV);
# AMOS/autoPET 'scanner' = manufacturer (site/protocol). These are NOT biological
# correlates — the pooled median over a heterogeneous cohort is a MIXTURE.
STRATIFIER = {"amos": "scanner", "saros": "region", "autopet": "scanner"}

SOURCE_RELEASE = {
    "amos": "AMOS22 released 600-case metadata; CT cases 0001-0500",
    "saros": "SAROS (TCIA) NIfTI-files-v2; processed-case roster",
    "autopet": "TCIA FDG-PET-CT-LESIONS / autoPET-I; DOI 10.7937/gkr0-xv29",
}


def _count_text(values):
    from collections import Counter
    c = Counter(str(v) for v in values if v not in (None, "", "NA"))
    return ";".join(f"{key}:{c[key]}" for key in sorted(c))


def cohort_table1_row(ds, fr_rows, fr_counts, meta, spacing_by_case):
    """Machine-readable cohort/processing summary for manuscript Table 1."""
    case_ids = ([r[0] for r in fr_rows] + list(fr_counts.get("empty_ids", []))
                + list(fr_counts.get("unparsed_ids", [])))
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError(f"{ds} Table 1 case ids are not unique")
    subjects = {participant_key(ds, cid, meta) for cid in case_ids}
    # Demographics are summarized once per participant. For repeated-study autoPET,
    # use one deterministic subject key rather than weighting patients by study count.
    # autoPET's metadata object accepts the bare participant key; other metadata
    # are case-keyed. Select one case per linked participant before counting sex/age.
    first_case = {}
    for cid in sorted(case_ids):
        first_case.setdefault(participant_key(ds, cid, meta), cid)
    demographic_keys = (sorted(subjects) if ds == "autopet"
                        else [first_case[s] for s in sorted(subjects)])
    demographic_records = [meta.get(key) for key in demographic_keys if meta.get(key)]
    ages = np.array([r["age"] for r in demographic_records
                     if isinstance(r.get("age"), (int, float))], float)
    sexes = [r.get("sex") for r in demographic_records if r.get("sex")]
    analysis_records = [(cid, meta.get(cid)) for cid in case_ids if meta.get(cid)]
    scanners = [r.get("scanner") for _, r in analysis_records if r.get("scanner")]
    regions = [r.get("region") for _, r in analysis_records if r.get("region")]
    contrasts = [r.get("contrast") for _, r in analysis_records if r.get("contrast")]
    spacings = np.array([spacing_by_case[cid] for cid in case_ids
                         if cid in spacing_by_case and np.isfinite(spacing_by_case[cid])], float)

    def _median_iqr(values, digits=1):
        if values.size == 0:
            return "NA"
        return (f"{np.median(values):.{digits}f} "
                f"[{np.percentile(values,25):.{digits}f}-{np.percentile(values,75):.{digits}f}]")

    female = sum(str(s).upper() == "F" for s in sexes)
    return {
        "cohort": LAYOUT[ds][0], "source_release": SOURCE_RELEASE[ds],
        "analysis_unit": "study" if ds == "autopet" else "case",
        "demographic_unit": "participant",
        "n_analysis_units": len(case_ids), "n_participants": len(subjects),
        "n_measurable": len(fr_rows), "n_empty_masks": int(fr_counts.get("empty", 0)),
        "n_unparsed": int(fr_counts.get("unparsed", 0)),
        "age_available_n": int(ages.size), "age_years_median_iqr": _median_iqr(ages),
        "sex_available_n": len(sexes), "female_n": female,
        "female_pct": (round(100 * female / len(sexes), 1) if sexes else "NA"),
        "manufacturer_counts_analysis_unit": _count_text(scanners) or "NA",
        "region_counts_analysis_unit": _count_text(regions) or "NA",
        "contrast_counts_analysis_unit": _count_text(contrasts) or "NA",
        "contrast_definition": ("series-first description tokens: mit KM/p.v./portalven=with; "
                                "nativ/ohne KM=without; selected series overrides study text"
                                if ds == "autopet" else "NA"),
        "slice_spacing_available_n": int(spacings.size),
        "slice_spacing_mm_median_iqr": _median_iqr(spacings, digits=2),
        "slice_spacing_mm_range": (f"{spacings.min():.2f}-{spacings.max():.2f}"
                                   if spacings.size else "NA"),
        # Derived from the pinned constant, so the version has one source.
        "mask_producer": f"TotalSegmentator {C.TOTALSEG_VERSION.split(' (')[0]} "
                         f"(see provenance.txt for the full attestation)",
    }


def stratified_primary(dir_name, ds, fr_rows, meta):
    """Report the primary surface gap stratified by the cohort's technical
    stratifier, so the pooled median is not mistaken for one population."""
    stratifier = STRATIFIER.get(ds)
    if not stratifier or not meta:
        return
    gaps = {r[0]: r[2] for r in fr_rows}                 # surface (primary)
    strata = {}
    for cid, g in gaps.items():
        rec = meta.get(cid)
        lv = rec.get(stratifier) if rec else None
        if lv is not None:
            strata.setdefault(str(lv), []).append(g)
    strata = {k: v for k, v in strata.items() if len(v) >= C.MIN_GROUP_N}
    if len(strata) < 2:
        return
    print(f"  STRATIFIED primary distribution by {stratifier} "
          f"(technical stratifier; the pooled median is a MIXTURE of these):")
    for k, v in sorted(strata.items(), key=lambda kv: -np.median(kv[1])):
        a = np.array(v, float)
        lo, hi = S.median_ci_auto(a, C.CI_METHOD, C.ALPHA, C.N_BOOT, C.SEED)
        print(f"    {k:14s} n={len(a):4d} median={np.median(a):5.1f} "
              f"(95%CI {_ci(lo, hi, 1)}) "
              f"IQR[{np.percentile(a,25):.1f}-{np.percentile(a,75):.1f}]")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_dataset(ds, do_correlates=True, do_figures=False, outdir=None):
    dir_name, prefix, _ = LAYOUT[ds]
    rdir = RESULTS / dir_name                 # inputs are always read from here
    # outputs go in-place by default; --outdir redirects them (e.g. a temp tree) so a
    # reproduction run can be compared without overwriting the outputs in Results/.
    wdir = rdir if outdir is None else (Path(outdir) / dir_name)
    wdir.mkdir(parents=True, exist_ok=True)
    fr_path = rdir / f"{prefix}_results_fullres.txt"
    cmp_path = rdir / f"{prefix}_results_compare.txt"
    # The two files are not equally load-bearing. `fr_path` carries the PRIMARY endpoint; `cmp_path`
    # carries only the SECONDARY fast-vs-full agreement, so a missing compare file disables only the
    # agreement statistics (reported as `fast-vs-full agreement: NOT RUN`).
    if not fr_path.exists():
        print(f"[skip {dir_name}] primary results not found (expected {fr_path.name})")
        return None

    validate_result_roster(ds, fr_path)
    fr_rows, fr_counts = load_fullres(fr_path)
    if cmp_path.exists():
        validate_result_roster(ds, cmp_path)
        cmp_rows, cmp_counts = load_compare(cmp_path)
    else:
        print(f"[{dir_name}] {cmp_path.name} absent -- the PRIMARY full-resolution "
              f"analysis proceeds; only the fast-vs-full agreement is unavailable")
        cmp_rows = []
        cmp_counts = {"measurable": 0, "empty": 0, "unparsed": 0}
    z = np.array([r[1] for r in fr_rows])
    surf = np.array([r[2] for r in fr_rows])
    c2c = np.array([r[3] for r in fr_rows])
    qc = [r[4] for r in fr_rows]
    fast = np.array([r[1] for r in cmp_rows])
    full = np.array([r[2] for r in cmp_rows])
    if surf.size == 0:
        print(f"[skip {dir_name}] no measurable cases")
        return None
    censored = censored_fraction(z)

    # measure_all.py writes the fullres + compare files TOGETHER in one run with the SAME metric, so
    # the compare metric == the fullres metric — infer it from the (column-versioned) fullres file,
    # NOT from filesystem mtimes.
    cmp_stale = fr_counts.get("metric") != "surface(v2)"
    cmp_metric = "c2c [v1 format]" if cmp_stale else "3D surface"

    # Both checks run BEFORE anything is written.
    _publication_invariants_hold(z, surf, c2c, fr_path.name)
    assert_compare_matches_fullres(fr_rows, cmp_rows, fr_path.name, cmp_path.name)

    rows = all_stats(z, surf, c2c, fr_counts, fast, full, cmp_counts,
                     fr_path.name, cmp_path.name, censored, cmp_metric=cmp_metric)
    out_csv = wdir / "all_stats.csv"
    pd.DataFrame(rows)[STATS_HEADER].to_csv(out_csv, index=False)
    aux_path = rdir / f"{prefix}_results_aux.txt"
    write_provenance(wdir / "provenance.txt", ds, aux=load_aux(aux_path))

    # The aux (spacing) file is optional.
    if aux_path.exists():
        validate_result_roster(ds, aux_path)
    spc = load_spacing(aux_path)
    sz_arr = np.array([spc.get(r[0], float("nan")) for r in fr_rows], float)
    spc_xyz = load_spacing_xyz(aux_path)
    nan3 = (float("nan"),) * 3
    sy_arr = np.array([spc_xyz.get(r[0], nan3)[1] for r in fr_rows], float)
    sx_arr = np.array([spc_xyz.get(r[0], nan3)[2] for r in fr_rows], float)
    if np.isfinite(sz_arr).any():
        szv = sz_arr[np.isfinite(sz_arr)]
        print(f"  axial slice spacing (mm): median {np.median(szv):.2f} "
              f"IQR[{np.percentile(szv, 25):.2f}-{np.percentile(szv, 75):.2f}] "
              f"range [{szv.min():.2f}-{szv.max():.2f}], n={szv.size} "
              f"(coarse z-spacing inflates the SECONDARY axial z_gap; primary metric "
              f"also quantizes to it — see slice-normalized analysis below)")

    # Load metadata EARLY (also reused by correlates) so the summary can surface non-random
    # missingness: which stratum the EMPTY (unmeasurable) cases fall in. Deliberately NOT called
    # MNAR -- see the print site for why. Metadata is loaded unconditionally: Table 1, the AMOS
    # adult sensitivity, the aux-correlate injection and the empty-stratum diagnostic need it.
    # `load_meta` returns {} when the metadata file is absent; `--no-correlates` skips only the
    # correlate analysis below.
    meta = load_meta(ds)
    if meta:
        _inject_aux_correlates(meta, load_aux(aux_path))
    table1 = cohort_table1_row(ds, fr_rows, fr_counts, meta, spc) if meta else None
    if ds == "amos" and meta:
        adult_row = amos_adult_sensitivity(fr_rows, fr_counts, meta)
        adult_path = wdir / "amos_adult_sensitivity.csv"
        pd.DataFrame([adult_row]).to_csv(adult_path, index=False)
        print(f"  wrote {adult_path} (adult-only scope sensitivity)")
    if ds == "saros":
        tail_path = wdir / "saros_extreme_tail_sensitivity.csv"
        pd.DataFrame(saros_extreme_tail_sensitivity(fr_rows)).to_csv(tail_path, index=False)
        print(f"  wrote {tail_path} (QC-valid >200 mm tail sensitivity)")
    empty_strata = None
    strat = STRATIFIER.get(ds)
    if meta and strat and fr_counts.get("empty_ids"):
        from collections import Counter
        es = Counter()
        for cid in fr_counts["empty_ids"]:
            rec = meta.get(cid)
            lv = rec.get(strat) if rec else None
            es[str(lv) if lv is not None else "unknown"] += 1
        empty_strata = dict(es)

    agreements, agreements_binned = agreement_rows(dir_name, fast, full)
    summary(ds, dir_name, z, surf, c2c, fr_counts, agreements, qc=qc, sz=sz_arr,
            empty_strata=empty_strata, strat_name=strat)
    # Agreement is paired on the compare file's both-pass cases. Fast-specific empties (full present,
    # fast empty) make the retained CCC a conditional, possibly optimistic estimate; dual-empties
    # (neither pass produced a mask, e.g. SAROS pelvic-FOV crops above the confluence) are a
    # coverage issue, not an agreement-ceiling one. Report the split.
    if agreements:
        n_pair = agreements[0]["n"]
        cmp_warn = ("  <-- compare metric is the v1 centre-to-centre distance (c2c)"
                    if cmp_stale else "")
        paired_ids = {r[0] for r in cmp_rows}
        fr_ids = {r[0] for r in fr_rows}
        n_fast_specific = len(fr_ids - paired_ids)        # full present, fast empty
        n_drop = cmp_counts.get("empty", 0)
        n_dual = max(0, n_drop - n_fast_specific)         # neither pass produced a mask
        print(f"  agreement N={n_pair} (paired both-pass) vs distribution N={surf.size}"
              f"  [dropped {n_drop}: {n_fast_specific} FAST-PASS-SPECIFIC (full present, "
              f"fast empty — availability-selected, so retained CCC is conditional and may "
              f"be optimistic) + {n_dual} DUAL-EMPTY (neither pass; FOV/coverage, "
              f"not an agreement-ceiling effect)]{cmp_warn}")
        pd.DataFrame(agreements).to_csv(wdir / f"{prefix}_agreement.csv", index=False)
        # The scalar columns in the file above can only hold an ENVELOPE; a conditional LoA is a
        # function of magnitude and this is where it lives.
        if agreements_binned:
            pd.DataFrame(agreements_binned).to_csv(
                wdir / f"{prefix}_agreement_binned.csv", index=False)
    elif cmp_counts.get("measurable", 0) == 0:
        print(f"  fast-vs-full agreement: NOT RUN; the fast (3mm) compare pass is empty "
              f"for all {cmp_counts.get('empty', 0)} cases (no masks); cross-resolution "
              f"reproducibility of the primary metric is UNTESTED for this cohort.")
    print(f"  wrote {out_csv} + provenance.txt"
          + (f" + {prefix}_agreement.csv" if agreements else "")
          + (f" + {prefix}_agreement_binned.csv" if agreements_binned else ""))

    panels, cohort_tests = [], []
    if do_correlates:
        # `meta` (with the Phase-1 aux correlates seg_quality + patient_size already
        # injected) was loaded above for the empty-stratum / MNAR diagnostic; reuse it.
        stratified_primary(dir_name, ds, fr_rows, meta)
        cor_rows, panels, cohort_tests = correlates(
            dir_name, fr_rows, meta,
            analysis_unit=("study" if ds == "autopet" else "case"))
        cor_csv = wdir / f"{prefix}_correlates.csv"
        cor_cols = ["dataset", "analysis_unit", "inference_scope",
                    "variable", "level", "n", "median", "median_ci",
                    "p25", "p75", "mean", "sd", "pct_present",
                    "pvalue", "qvalue", "eps_squared", "eps2_ci", "eps2_label"]
        pd.DataFrame(cor_rows).reindex(columns=cor_cols).to_csv(cor_csv, index=False)
        print(f"  wrote {cor_csv}")

        if ds == "autopet":
            # Independent-unit sensitivity: choose the earliest study using ALL case
            # ids first, then retain it only if measurable. This prevents an empty
            # earliest study from being silently replaced by a later usable scan.
            all_case_ids = ([r[0] for r in fr_rows] +
                            list(fr_counts.get("empty_ids", [])) +
                            list(fr_counts.get("unparsed_ids", [])))
            subject_rows = autopet_one_study_per_subject(fr_rows, all_case_ids)
            n_subjects = len({_autopet_subject_id(cid) for cid in all_case_ids})
            print(f"  autoPET subject-level sensitivity: selected earliest study for "
                  f"{n_subjects} subjects; {len(subject_rows)} selections measurable")
            sens_rows, _sens_panels, _sens_tests = correlates(
                dir_name, subject_rows, meta, analysis_unit="subject",
                inference_scope="subject_level_sensitivity_earliest_study")
            sens_csv = wdir / f"{prefix}_subject_sensitivity.csv"
            pd.DataFrame(sens_rows).reindex(columns=cor_cols).to_csv(sens_csv,
                                                                     index=False)
            print(f"  wrote {sens_csv}")

            # The subject sensitivity is a different estimand; the cluster-robust rows give the
            # primary estimand resampled by subject, with the design effect that sizes the
            # dependency.
            cr_rows = cluster_robust_rows(fr_rows)
            if cr_rows:
                cr_csv = wdir / f"{prefix}_cluster_robust.csv"
                pd.DataFrame(cr_rows).to_csv(cr_csv, index=False)
                print(f"  wrote {cr_csv}")
                for r in cr_rows:
                    print(_cluster_robust_console(r))

    # Its own <prefix>_collapsed_diag.csv, NOT appended to the contractual all_stats.csv.
    collapsed_summary = None
    collapsed_inputs = [
        ("labels", rdir / f"{prefix}_results_collapsed.txt", f"{prefix}_collapsed_diag.csv"),
        ("probs", rdir / f"{prefix}_results_collapsed_probs.txt",
         f"{prefix}_collapsed_probs_diag.csv"),
    ]
    for expected_mode, coll_path, csv_name in collapsed_inputs:
        if not coll_path.exists():
            continue
        validate_result_roster(ds, coll_path)
        coll_rows, coll_counts = load_collapsed(coll_path)
        observed_mode = coll_counts.get("mode", "labels")
        if observed_mode != expected_mode:
            raise RuntimeError(
                f"collapsed table mode mismatch: {coll_path.name} declares {observed_mode}, "
                f"expected {expected_mode}")
        cd_rows, summary_for_mode = collapsed_class_stats(
            dir_name, ds, coll_rows, counts=coll_counts, mode=observed_mode)
        if cd_rows:
            cd_csv = wdir / csv_name
            pd.DataFrame(cd_rows)[COLLAPSED_HEADER].to_csv(cd_csv, index=False)
            print(f"  wrote {cd_csv}")
        if expected_mode == "labels":
            collapsed_summary = summary_for_mode

    if do_figures:
        figures(dir_name, z, surf, panels, wdir / "figures", agreements, c2c=c2c)
        bland_altman_figure(dir_name, fast, full, wdir / "figures")
    print()
    return {"dir": dir_name, "ds": ds, "z": z, "surf": surf, "c2c": c2c, "sz": sz_arr,
            "sy": sy_arr, "sx": sx_arr, "n_empty": fr_counts.get("empty", 0),
            "counts": fr_counts, "cohort_tests": cohort_tests,
            "collapsed": collapsed_summary, "table1": table1}


def _mm(x):
    """Two decimals with one trailing zero trimmed: 2.670 -> '2.67', 5.000 -> '5.0'."""
    s = f"{x:.2f}"
    return s[:-1] if s.endswith("0") else s


def _spacing_bands(surf, sz, min_n):
    """{spacing_mm: median gap} over bands with at least `min_n` cases."""
    groups = {}
    for s, g in zip(np.asarray(sz, float), np.asarray(surf, float)):
        if np.isfinite(s) and s > 0 and np.isfinite(g):
            groups.setdefault(round(float(s), 2), []).append(g)
    return {k: float(np.median(v)) for k, v in groups.items() if len(v) >= min_n}


def normalizer_sensitivity(results):
    """-> [(label, eps^2, H, p)] for every spacing normalization, raw mm first.

    THE ATTENUATION IS A PROPERTY OF THE CHOSEN DIVISOR, and publishing one divisor makes a
    choice look like a finding. Measured on the study's tables (n = 1873): raw mm gives
    eps^2 0.0628; dividing by the z spacing gives 0.0073, an 88.5 % drop; dividing by the 3-D
    geometric-mean spacing gives 0.0599, a 5 % drop; and dividing by the IN-PLANE
    geometric-mean spacing gives 0.0979, a 56 % INCREASE. Only the z-spacing divisor attenuates the
    difference strongly; the 3-D divisor barely moves it and the in-plane divisor increases it.

    All of them are emitted, labelled, so a reader can see that the headline is a choice. No
    divisor is privileged here and none is recommended: which normalization is appropriate is
    a question about the estimand, not about this function.
    """
    divisors = (
        ("raw mm", lambda sz, sy, sx: np.ones_like(sz)),
        ("/ z spacing", lambda sz, sy, sx: sz),
        ("/ 3-D geometric-mean spacing", lambda sz, sy, sx: np.cbrt(sz * sy * sx)),
        ("/ in-plane geometric-mean spacing", lambda sz, sy, sx: np.sqrt(sy * sx)),
    )
    out = []
    for label, divisor in divisors:
        groups = []
        for r in results:
            surf = np.asarray(r.get("surf"), float)
            sz = np.asarray(r.get("sz"), float)
            sy = np.asarray(r.get("sy"), float) if r.get("sy") is not None else None
            sx = np.asarray(r.get("sx"), float) if r.get("sx") is not None else None
            if sy is None or sx is None or surf.size != sz.size:
                return []          # in-plane spacing unavailable: report nothing, not a subset
            if not (surf.size == sy.size == sx.size):
                return []
            d = divisor(sz, sy, sx)
            ok = np.isfinite(surf) & np.isfinite(d) & (d > 0)
            if ok.any():
                groups.append(surf[ok] / d[ok])
        if len(groups) < 2:
            return []
        kw = S.kruskal_test(*groups)
        if kw["p"] is None:
            return []
        out.append((label, kw["eps_squared"], kw["H"], kw["p"]))
    return out


def worst_case_median_bounds(values, n_unmeasured):
    """-> (lower, upper, tipping_k) for the median under UNRESTRICTED missingness.

    A Manski-style bound: it assumes nothing whatever about the cases that could not be
    measured. The lower bound places every one of them at 0 mm, the upper bound above every
    observed value, and the truth lies between. `tipping_k` is the smallest number of
    unmeasured cases that must sit above every observed value before the median RISES ABOVE
    THE OBSERVED MEDIAN at all: how much would have to be true before the point estimate moves
    upward.

    The smallest k that reaches the upper END is uninformative, because the upper end is reached
    only when every unmeasured case is extreme.

    Both halves are reported because the width alone misleads. The tipping count is what says
    whether a wide interval is a live threat."""
    v = np.sort(np.asarray([x for x in np.asarray(values, float) if np.isfinite(x)]))
    n_unmeasured = int(n_unmeasured)
    if v.size == 0:
        return None, None, None
    if n_unmeasured <= 0:
        m = float(np.median(v))
        return m, m, 0
    low = float(np.median(np.concatenate([np.zeros(n_unmeasured), v])))
    high = float(np.median(np.concatenate([v, np.full(n_unmeasured, np.inf)])))
    observed = float(np.median(v))
    tipping = n_unmeasured
    for k in range(n_unmeasured + 1):
        padded = np.concatenate([np.zeros(n_unmeasured - k), v, np.full(k, np.inf)])
        if float(np.median(padded)) > observed:
            tipping = k
            break
    return low, high, tipping


def worst_case_rows(results):
    """-> [(dir, n_measured, n_empty, observed median, low, high, tipping)] per cohort.

    Separate from the emitter so the WIRING is testable, not only the estimator.
    `worst_case_median_bounds` can be perfect while its caller passes a zero, and the
    published line would then read "empty=0  POINT-IDENTIFIED" for a cohort with unmeasured
    cases -- the most flattering error available here, and one no test of the estimator alone
    can see.
    """
    rows = []
    for r in results:
        surf = np.asarray(r.get("surf"), float)
        finite = surf[np.isfinite(surf)]
        n_empty = int(r.get("n_empty") or 0)
        low, high, tipping = worst_case_median_bounds(surf, n_empty)
        if low is None:
            continue
        rows.append((r["dir"], int(surf.size), n_empty, float(np.median(finite)),
                     low, high, tipping))
    return rows


def eligible_frame_worst_case_rows(results):
    """-> [(dir, ds, n_eligible, n_measured, n_unmeasured, observed, low, high, tipping)] per cohort.

    The same Manski bound as `worst_case_rows`, over the DATASET-eligible denominator recorded in
    `COHORT_ELIGIBLE` rather than the processed one. Publishing both frames is what lets a reader
    see which one an interval is on; the ordering autoPET < SAROS < AMOS happens to survive on both.

    An eligible denominator smaller than the measured count is a contradiction, not a zero, and raises
    rather than publishing the flattering bound that would follow from it.
    """
    rows = []
    for r in results:
        surf = np.asarray(r.get("surf"), float)
        finite = surf[np.isfinite(surf)]
        eligible = C.COHORT_ELIGIBLE.get(r.get("ds"))
        if eligible is None or finite.size == 0:
            continue
        n_eligible, n_measured = int(eligible[1]), int(finite.size)
        if n_eligible < n_measured:
            raise RuntimeError(
                f"{r.get('dir')}: COHORT_ELIGIBLE gives {n_eligible} eligible cases but {n_measured} were "
                f"measured; the eligible-frame bound would understate the missingness")
        n_unmeasured = n_eligible - n_measured
        low, high, tipping = worst_case_median_bounds(finite, n_unmeasured)
        rows.append((r["dir"], r.get("ds"), n_eligible, n_measured, n_unmeasured,
                     float(np.median(finite)), low, high, tipping))
    return rows


def _acquisition_scaling_note(results, mm_eps, slice_eps):
    """Derive the acquisition-scaling paragraph instead of asserting it.

    The matched reference is the MODAL spacing across the pooled cohorts (the one
    thickness at which the largest number of cases can actually be compared) and it is
    named in the sentence, so a change of reference is visible rather than silent.
    "Thin" is then defined against that same reference, not against a new constant.
    The direction is read off the two effect sizes, using the module's own
    `_eps_mag` bands rather than a threshold invented here: a drop that crosses a band
    is "strongly", a drop within one band is plain attenuation, and no drop at all is
    reported as such, with the acquisition/resolution reading withheld.

    The drop is real for THIS divisor and is not a property of the data: dividing by the 3-D
    geometric-mean spacing leaves eps^2 almost where it started (0.0628 -> 0.0599) and dividing by
    the in-plane geometric mean RAISES it (0.0628 -> 0.0979). A reader who saw only the z-normalized
    number would take a choice of divisor for a finding, so the sentence points at the
    sensitivity block rather than concluding over it."""
    per_cohort = []          # (dir, median slice-thickness units)
    bands = {}               # dir -> {spacing: median gap}
    for r in results:
        su, _n = slice_normalized(r["surf"], r.get("sz"))
        if su.size:
            per_cohort.append((r["dir"], float(np.median(su))))
        bands[r["dir"]] = _spacing_bands(r["surf"], r.get("sz"), C.MIN_GROUP_N)
    if not per_cohort:
        return None
    units_lo = min(u for _d, u in per_cohort)
    units_hi = max(u for _d, u in per_cohort)

    # matched reference = the spacing carrying the most cases across all cohorts
    counts = {}
    for r in results:
        for s in np.asarray(r.get("sz"), float):
            if np.isfinite(s) and s > 0:
                counts[round(float(s), 2)] = counts.get(round(float(s), 2), 0) + 1
    ref = max(counts, key=lambda k: (counts[k], -k)) if counts else None

    matched = sorted(((d, b[ref]) for d, b in bands.items() if ref in b),
                     key=lambda t: t[1])
    thin = [g / s for b in bands.values() for s, g in b.items()
            if ref is not None and s < ref]

    if slice_eps < mm_eps:
        strong = _eps_mag(slice_eps) != _eps_mag(mm_eps)
        head = (f"the cross-cohort mm difference attenuates "
                f"{'strongly ' if strong else ''}after slice-thickness normalization "
                f"(~{units_lo:.1f}-{units_hi:.1f} slice-thickness units per cohort), "
                f"which is CONSISTENT WITH an acquisition/resolution contribution but does not "
                f"establish one: the attenuation is a property of dividing by the Z spacing, "
                f"and the NORMALIZER SENSITIVITY block below shows it nearly vanishes under a "
                f"3-D geometric-mean divisor and REVERSES under an in-plane one.")
    else:
        head = (f"the cross-cohort mm difference does not attenuate after slice-thickness "
                f"normalization (~{units_lo:.1f}-{units_hi:.1f} slice-thickness units per "
                f"cohort; eps^2 {mm_eps:.3f} -> {slice_eps:.3f}), so these data do not "
                f"support reading the difference as acquisition/resolution scaling.")

    parts = []
    if len(matched) >= 2:
        parts.append(f"A residual cohort difference persists at matched {_mm(ref)}mm ("
                     + " / ".join(f"{d} {_mm(v)}" for d, v in matched) + " mm)")
    if thin:
        parts.append(f"thin-slice gaps remain {min(thin):.0f}-{max(thin):.0f} "
                     f"z-slice-thickness units")
    tail = (", and ".join(parts) + "; " if parts else "")
    return ("  -> " + head + " " + tail
            + "therefore neither pure discretization nor a biological mechanism is "
              "established by this observational comparison.")


def cross_cohort(results, outdir=None):
    """Between-cohort heterogeneity of the PRIMARY surface gap, reported as a
    DESCRIPTIVE index (a KW p-value here just confirms two different datasets are
    different — not an actionable hypothesis test), plus a POOLED Benjamini-
    Hochberg correction over every correlate test run across all cohorts."""
    results = [r for r in results if r is not None]
    if len(results) < 2:
        return
    print("=" * 78)
    print("BETWEEN-COHORT heterogeneity — 3D surface gap (DESCRIPTIVE)")
    print("=" * 78)
    lines = ["between-cohort heterogeneity (3D surface gap; descriptive)"]
    for r in results:
        d = r["surf"]
        lo, hi = S.median_ci_auto(d, C.CI_METHOD, C.ALPHA, C.N_BOOT, C.SEED)
        n_total = (len(d) + r["counts"].get("empty", 0)
                   + r["counts"].get("failed", 0)
                   + r["counts"].get("unparsed", 0))
        su, _ = slice_normalized(d, r.get("sz"))
        su_str = f"  = {np.median(su):.2f} slice-thickness units" if su.size else ""
        msg = (f"  {r['dir']:8s} n={len(d):4d}  median={np.median(d):5.2f}mm "
               f"(95%CI {_ci(lo, hi, 2)}){su_str}  empty_rate="
               f"{100*r['counts'].get('empty',0)/max(1,n_total):.1f}%")
        print(msg)
        lines.append(msg)
    kw = S.kruskal_test(*[r["surf"] for r in results])
    if kw["p"] is not None:
        msg = (f"  KW heterogeneity in mm (DESCRIPTIVE, not a hypothesis test): "
               f"H={kw['H']:.1f} p={_pfmt(kw['p'])} eps^2={kw['eps_squared']:.3f} "
               f"({_eps_mag(kw['eps_squared'])}, k={kw['k']})")
    else:
        msg = "  KW heterogeneity index: NA (scipy unavailable or <2 cohorts)"
    print(msg)
    lines.append(msg)
    # Acquisition-scaling diagnostic: the SAME KW on the SLICE-NORMALIZED gap
    # (surf/slice). If the cross-cohort difference is resolution, eps^2 collapses here.
    su_groups = [g for g in (slice_normalized(r["surf"], r.get("sz"))[0]
                             for r in results) if g.size]
    if len(su_groups) >= 2:
        kws = S.kruskal_test(*su_groups)
        if kws["p"] is not None and kw["p"] is not None:
            msg = (f"  KW on SLICE-NORMALIZED gap (surf/slice): H={kws['H']:.1f} "
                   f"p={_pfmt(kws['p'])} eps^2={kws['eps_squared']:.3f} "
                   f"({_eps_mag(kws['eps_squared'])})   "
                   f"[mm eps^2 {kw['eps_squared']:.3f} -> slice-units "
                   f"{kws['eps_squared']:.3f}]")
            print(msg)
            lines.append(msg)
            # See _acquisition_scaling_note.
            note = _acquisition_scaling_note(results, kw["eps_squared"],
                                             kws["eps_squared"])
            if note is not None:
                print(note)
                lines.append(note)
    sensitivity = normalizer_sensitivity(results)
    if sensitivity:
        head = ("  NORMALIZER SENSITIVITY (the attenuation above is a property of the divisor; "
                "no divisor is privileged here):")
        print(head)
        lines.append(head)
        raw = sensitivity[0][1]
        for label, eps, H, pv in sensitivity:
            delta = "" if label == "raw mm" else f"  [{(eps - raw) / raw * 100:+.1f}% vs raw mm]"
            row = (f"    {label:34s} eps^2={eps:.4f} H={H:.2f} p={_pfmt(pv)} "
                   f"({_eps_mag(eps)}){delta}")
            print(row)
            lines.append(row)

    bound_rows = worst_case_rows(results)
    if bound_rows:
        head = ("  WORST-CASE MEDIAN under UNRESTRICTED missingness (Manski bounds; assumes "
                "NOTHING about the unmeasured cases):")
        print(head)
        lines.append(head)
        for dn, n_meas, n_empty, obs, low, high, tipping in bound_rows:
            if n_empty == 0:
                row = (f"    {dn:8s} n={n_meas} empty=0  median={obs:.3f}mm  "
                       f"POINT-IDENTIFIED (no unmeasured case to bound)")
            else:
                # With empty cases the median is not point-identified here either, so the label must
                # say which quantity the number is, and the same file must not give it two labels.
                row = (f"    {dn:8s} n={n_meas} empty={n_empty}  median(measured)={obs:.3f}mm  "
                       f"worst case [{low:.3f}, {high:.3f}]mm  "
                       f"({tipping} of the {n_empty} unmeasured cases must be above every "
                       f"observed gap before the median rises at all)")
            print(row)
            lines.append(row)

    # The line above counts the cases that entered the pipeline and came back empty, and cannot
    # count a study that was excluded before processing (SAROS: 30 on confluence eligibility, 6
    # absent from the export). This one assumes nothing about ANY eligible study that was not
    # measured, whatever the reason, so it needs no per-study reason to be honest.
    elig_rows = eligible_frame_worst_case_rows(results)
    if elig_rows:
        head = ("  WORST-CASE MEDIAN over the ELIGIBLE frame (every eligible study that was not measured "
                "counted as unmeasured, whatever the reason):")
        print(head)
        lines.append(head)
        for dn, _ds, n_elig, n_meas, n_unmeas, obs, low, high, tipping in elig_rows:
            if n_unmeas == 0:
                row = (f"    {dn:8s} eligible={n_elig} measured={n_meas} unmeasured=0  "
                       f"median={obs:.3f}mm  POINT-IDENTIFIED (every eligible study was measured)")
            else:
                # The number is the median of the MEASURED studies; what the eligible frame
                # identifies is the interval printed beside it, which is why the two must not share
                # a label.
                row = (f"    {dn:8s} eligible={n_elig} measured={n_meas} unmeasured={n_unmeas}  "
                       f"median(measured)={obs:.3f}mm  worst case [{low:.3f}, {high:.3f}]mm  "
                       f"({tipping} of the {n_unmeas} unmeasured studies must be above every "
                       f"observed gap before the median rises at all)")
            print(row)
            lines.append(row)

    pooled = [t for r in results for t in r.get("cohort_tests", []) if t[2] is not None]
    for fam_label, in_fam in (("bio", lambda v: v not in C.CONFOUNDED_VARIABLES),
                              ("tech", lambda v: v in C.CONFOUNDED_VARIABLES)):
        members = [t for t in pooled if in_fam(t[1])]
        if not members:
            continue
        qs = S.bh_fdr([t[2] for t in members])
        print(f"  POOLED Benjamini-Hochberg [{fam_label}] across cohorts "
              f"(family m={len(members)}):")
        lines.append(f"pooled BH [{fam_label}] family m={len(members)}:")
        for (dn, var, p, eps, elo, ehi), q in sorted(zip(members, qs), key=lambda t: t[1]):
            row = (f"    {dn:8s} {var:9s} p={_pfmt(p)} q(pooled BH-{fam_label})={_pfmt(q)} "
                   f"eps^2={eps:.3f} ({_eps_label_ci(eps, elo, ehi)})")
            print(row)
            lines.append(row)
    lines.append("note: cohorts differ in scanner/contrast/population/RESOLUTION; the mm "
                 "heterogeneity index is descriptive and largely reflects slice thickness "
                 "(see the slice-normalized KW). Bio and technical correlate families are "
                 "FDR-corrected separately.")
    xc_dir = RESULTS if outdir is None else Path(outdir)
    xc_dir.mkdir(parents=True, exist_ok=True)
    (xc_dir / "_cross_cohort.txt").write_text("\n".join(lines) + "\n")
    table_rows = [r["table1"] for r in results if r.get("table1")]
    if table_rows:
        pd.DataFrame(table_rows).to_csv(xc_dir / "cohort_table1.csv", index=False)
        print(f"  wrote {xc_dir / 'cohort_table1.csv'}")
    print()


def main():
    ap = argparse.ArgumentParser(description="IVC-LCIV gap stats for all cohorts")
    ap.add_argument("dataset", choices=list(LAYOUT) + ["all"],
                    help="amos | saros | autopet | all")
    ap.add_argument("--figures", action="store_true", help="also render figures")
    ap.add_argument("--publication", action="store_true",
                    help="also run connectivity_statistics.py (the connectivity tables)")
    ap.add_argument("--no-correlates", action="store_true",
                    help="skip the metadata correlate analysis")
    ap.add_argument("--outdir", default=None,
                    help="write outputs under OUTDIR/<COHORT>/ instead of in-place "
                         "(inputs are read from Results/phase1_characterize/<COHORT>/); use "
                         "to reproduce the tables without overwriting the committed CSVs")
    args = ap.parse_args()
    datasets = list(LAYOUT) if args.dataset == "all" else [args.dataset]
    results = []
    for ds in datasets:
        results.append(run_dataset(ds, do_correlates=not args.no_correlates,
                                   do_figures=args.figures, outdir=args.outdir))
    if not any(r is not None for r in results):
        sys.exit("no datasets had results to analyze.")
    if args.dataset == "all":
        cross_cohort(results, outdir=args.outdir)
        if args.publication:
            import connectivity_statistics
            publication_args = [] if args.outdir is None else ["--outdir", args.outdir]
            connectivity_statistics.main(publication_args)


if __name__ == "__main__":
    main()
