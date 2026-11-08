#!/usr/bin/env python3
"""Connectivity statistics for the analyzed cohorts.

Computes the connectivity classes on the dominant components, the label-distance analysis, the audit
of the released annotations and the agreement between repeat examinations, and writes the tables in
Results/phase1_characterize/connectivity/. Distances describe labels on the native CT grid, not
anatomy. No clinical bias, equivalence or test-retest claim is made.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sqlite3
import time
from pathlib import Path

import stats as ST  # establishes sibling phase/common imports for direct CLI execution

# isort: split
import gap_config as CFG
import numpy as np
import pandas as pd
import scipy
import statlib as S

# Grid sampling and participant bootstraps use NumPy/SciPy, not regression models.
# Keep the segmentation environment usable without installing a regression stack;
# reproducing the regression tables requires the full dependency, which is loaded explicitly.
statsmodels = sm = smf = multipletests = variance_inflation_factor = None

OUT = ST.RESULTS / "connectivity"
THIN_MM = 1.5


def _load_regression_dependencies():
    global statsmodels, sm, smf, multipletests, variance_inflation_factor
    if statsmodels is not None:
        return
    try:
        import statsmodels as package
        import statsmodels.api as api
        import statsmodels.formula.api as formula_api
        from statsmodels.stats.multitest import multipletests as adjust
        from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "statsmodels and its dependencies are required to regenerate the regression "
            "analyses; --grid-resampling-only does not require them.", name=exc.name
        ) from exc
    statsmodels, sm, smf = package, api, formula_api
    multipletests, variance_inflation_factor = adjust, vif


def load_analysis(recovered_path):
    recovered = pd.read_csv(recovered_path)
    if recovered.duplicated(["dataset", "case"]).any():
        raise ValueError("duplicate recovered topology case")
    extra = {(r.dataset, r.case): r for r in recovered.itertuples()}
    used = set()
    data, flows, participants = [], [], []
    for ds, (dirname, prefix, _) in ST.LAYOUT.items():
        directory = ST.RESULTS / dirname
        path = directory / f"{prefix}_results_fullres.txt"
        ST.validate_result_roster(ds, path)
        full, counts = ST.load_fullres(path)
        ST.require_surface_metric(counts, path, consumer="connectivity statistics")
        if counts["failed"] or counts["unparsed"]:
            raise ValueError(f"unresolved primary input failures: {dirname}")
        paired, pc = ST.load_collapsed(directory / f"{prefix}_results_collapsed.txt")
        if pc["unparsed"]:
            raise ValueError(f"unparsed paired diagnostic: {dirname}")
        pairs = {r[0]: r for r in paired}
        if len(pairs) != len(paired):
            raise ValueError("duplicate paired diagnostic case")
        aux = ST.load_aux(directory / f"{prefix}_results_aux.txt")
        meta = ST.load_meta(ds)
        primary_ids = {r[0] for r in full}
        if len(primary_ids) != len(full) or not set(pairs) <= primary_ids:
            raise ValueError("duplicate primary case or foreign paired case")
        all_ids = sorted(primary_ids | set(counts["empty_ids"]))
        for cid in all_ids:
            participants.append(
                {
                    "cohort": dirname,
                    "case": cid,
                    "participant": ST.participant_key(ds, cid, meta),
                    "gap_measurable": int(cid in primary_ids),
                }
            )
        for cid, _, gap, _, qc in full:
            p = pairs.get(cid)
            if p is not None:
                if abs(p[1] - gap) > ST.DELTA_ROUNDING_BOUND_MM:
                    raise ValueError(f"paired gap disagrees: {cid}")
                connected, nc = int(p[6]), p[5]
                topology_source = "committed_paired_diagnostic"
            else:
                key = (dirname, cid)
                if key not in extra:
                    raise ValueError(f"missing recovered raw topology: {key}")
                x = extra[key]
                if x.union_connected not in (0, 1) or x.direct_connected not in (0, 1):
                    raise ValueError("nonbinary recovered connectivity")
                if x.direct_connected and gap > 0:
                    raise ValueError(
                        "positive box separation cannot have direct connectivity"
                    )
                connected, nc = int(x.union_connected), int(x.union_ncomp)
                topology_source = x.source
                used.add(key)
            m = meta.get(cid, {})
            a = aux.get(cid, {})
            data.append(
                {
                    "cohort": dirname,
                    "case": cid,
                    "participant": f"{dirname}:{ST.participant_key(ds, cid, meta)}",
                    "gap_mm": gap,
                    "slice_mm": a.get("sz", np.nan),
                    "contrast": m.get("contrast"),
                    "body_csa": a.get("csa", np.nan),
                    "softmax": np.nan,
                    "qc": qc,
                    "union_connected": connected,
                    "union_ncomp": nc,
                    "topology_source": topology_source,
                    "paired_diagnostic": int(p is not None),
                }
            )
        flows.append(
            {
                "cohort": dirname,
                "processed": len(all_ids),
                "measurable": len(full),
                "empty": counts["empty"],
                "paired_diagnostic": len(paired),
                "resplit_failures": pc["failed"],
                "resplit_failure_reasons": json.dumps(
                    pc["failure_reasons"], sort_keys=True
                ),
                "raw_topology": len(full),
            }
        )
    if used != set(extra):
        raise ValueError("unused recovered topology rows")
    return pd.DataFrame(data), pd.DataFrame(flows), pd.DataFrame(participants)


def median_interval(values, participants, n_boot=None, seed=None):
    """CI of the study-weighted median, resampling participants when they repeat.

    The publication path keeps the module's resample count and seed; a caller that records its
    own bootstrap parameters (the gap-content summaries) passes them explicitly.
    """
    n_boot = N_BOOT if n_boot is None else int(n_boot)
    seed = CFG.SEED if seed is None else int(seed)
    values = np.asarray(values, float)
    if len(values) < 2:
        return np.nan, np.nan, "not_estimable"
    if len(set(participants)) < len(values):
        lo, hi = S.cluster_bca_ci(
            values, participants, np.median, CFG.ALPHA, n_boot, seed
        )
        return lo, hi, "participant_cluster_BCa"
    lo, hi = S.median_ci_auto(values, CFG.CI_METHOD, CFG.ALPHA, n_boot, seed)
    return (
        lo,
        hi,
        "iid_BCa",
    )  # independent case proxies in AMOS, linked subjects in SAROS


def summaries(data):
    rows = []
    for cohort, frame in data.groupby("cohort", sort=True):
        selections = {
            "all": frame,
            "thin_le_1p5mm": frame[frame.slice_mm <= THIN_MM],
            "thin_with_contrast": frame[
                (frame.slice_mm <= THIN_MM) & (frame.contrast == "with")
            ],
        }
        for stratum, d in selections.items():
            n = len(d)
            if not n:
                rows.append(
                    {
                        "cohort": cohort,
                        "stratum": stratum,
                        "n": 0,
                        "status": "no_observations_with_available_metadata",
                    }
                )
                continue
            v = d.gap_mm.to_numpy()
            lo, hi, method = median_interval(v, d.participant.tolist())
            row = {
                "cohort": cohort,
                "stratum": stratum,
                "n": n,
                "status": "computed",
                "n_participants": d.participant.nunique(),
                "zero_n": int((v == 0).sum()),
                "zero_pct": 100 * np.mean(v == 0),
                "median_mm": np.median(v),
                "q25_mm": np.quantile(v, 0.25),
                "q75_mm": np.quantile(v, 0.75),
                "median_ci_lo": lo,
                "median_ci_hi": hi,
                "median_ci_method": method,
                "mean_mm": v.mean(),
                "sd_mm": v.std(ddof=1) if n > 1 else np.nan,
                "no_path_n": int((d.union_connected == 0).sum()),
            }
            for threshold, label in ((CFG.PRESENT_MM, "present"), (2.0, "gt2mm")):
                yes = (v > threshold).astype(float)
                wlo, whi = S.wilson_ci(int(yes.sum()), n, CFG.ALPHA)
                row.update(
                    {
                        f"{label}_n": int(yes.sum()),
                        f"{label}_pct": 100 * yes.mean(),
                        f"{label}_wilson_lo_pct": 100 * wlo,
                        f"{label}_wilson_hi_pct": 100 * whi,
                    }
                )
                # Wilson is descriptive iid. Keep the dependence-adjusted CI separately.
                if d.participant.nunique() < n:
                    cl, ch = S.cluster_bca_ci(
                        yes,
                        d.participant.tolist(),
                        np.mean,
                        CFG.ALPHA,
                        N_BOOT,
                        CFG.SEED,
                    )
                    row.update(
                        {
                            f"{label}_cluster_lo_pct": 100 * cl,
                            f"{label}_cluster_hi_pct": 100 * ch,
                        }
                    )
            large = d[d.gap_mm > 2]
            row["connected_gt2mm_n"] = int(large.union_connected.sum())
            row["connected_gt2mm_denominator"] = len(large)
            row["connected_gt2mm_pct"] = (
                100 * large.union_connected.mean() if len(large) else np.nan
            )
            for label, values, groups in (
                (
                    "no_path",
                    (d.union_connected == 0).to_numpy(float),
                    d.participant.tolist(),
                ),
                (
                    "connected_gt2mm",
                    large.union_connected.to_numpy(float),
                    large.participant.tolist(),
                ),
            ):
                if not len(values):
                    continue
                wlo, whi = S.wilson_ci(int(values.sum()), len(values), CFG.ALPHA)
                row.update(
                    {
                        f"{label}_wilson_lo_pct": 100 * wlo,
                        f"{label}_wilson_hi_pct": 100 * whi,
                    }
                )
                if len(set(groups)) < len(values):
                    cl, ch = S.cluster_bca_ci(
                        values, groups, np.mean, CFG.ALPHA, N_BOOT, CFG.SEED
                    )
                    row.update(
                        {
                            f"{label}_cluster_lo_pct": 100 * cl,
                            f"{label}_cluster_hi_pct": 100 * ch,
                        }
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def model_results(data):
    """Available-covariate models; no independence/equivalence claim follows a null p."""
    _load_regression_dependencies()
    coefficients, diagnostics, partitions = [], [], []
    base = data[np.isfinite(data.slice_mm) & (data.slice_mm > 0)].copy()
    base["present"] = (base.gap_mm > CFG.PRESENT_MM).astype(int)
    if "class26" not in base or not base.class26.isin(CLASSES).all():
        raise ValueError("primary connectivity classes are required for publication models")
    base["disconnected"] = (base.class26 == "disconnected").astype(int)
    base["spacing_centered_mm"] = base.slice_mm - 2.5
    base["log1p_gap"] = np.log1p(base.gap_mm)
    base["spacing_category"] = pd.cut(
        base.slice_mm, [0, 1.5, 3.0, np.inf],
        labels=["le1p5", "gt1p5_le3", "gt3"], right=True,
    )
    specs = [
        ("pooled", base, "slice_mm + C(cohort)"),
        ("pooled_spacing_interaction", base, "spacing_centered_mm * C(cohort)"),
        (
            "autoPET_contrast",
            base[(base.cohort == "autoPET") & base.contrast.isin(["with", "without"])],
            "slice_mm + C(contrast)",
        ),
        ("pooled_spacing_categorical", base, "C(spacing_category) + C(cohort)"),
        ("disconnection_spacing_categorical", base, "C(spacing_category) + C(cohort)"),
    ]
    for name, frame, rhs in specs:
        endpoints = (("disconnected",) if name == "disconnection_spacing_categorical"
                     else ("gap_mm", "present", "log_gap_given_present"))
        for endpoint in endpoints:
            d = frame.copy()
            if endpoint == "log_gap_given_present":
                d = d[d.present == 1].copy()
                d[endpoint] = np.log(d.gap_mm)
            formula = f"{endpoint} ~ {rhs}"
            model = (
                smf.glm(formula, data=d, family=sm.families.Binomial())
                if endpoint in ("present", "disconnected")
                else smf.ols(formula, data=d)
            )
            if np.linalg.matrix_rank(model.exog) != model.exog.shape[1]:
                raise ValueError(f"rank-deficient model: {name}/{endpoint}")
            kwargs = {
                "cov_type": "cluster",
                "cov_kwds": {"groups": d.participant, "use_correction": True},
                "use_t": True,
            }
            fit = model.fit(**kwargs)
            if endpoint in ("present", "disconnected") and not fit.converged:
                raise ValueError(f"logistic model did not converge: {name}")
            ci = fit.conf_int(alpha=CFG.ALPHA)
            for term in fit.params.index:
                coefficients.append(
                    {
                        "model": name,
                        "endpoint": endpoint,
                        "term": term,
                        "estimate": fit.params[term],
                        "ci_lo": ci.loc[term, 0],
                        "ci_hi": ci.loc[term, 1],
                        "p": fit.pvalues[term],
                        "n": len(d),
                        "n_participants": d.participant.nunique(),
                        "covariance": "participant_cluster_small_sample_t",
                        "scale": (
                            "log_odds"
                            if endpoint in ("present", "disconnected")
                            else "log_mm"
                            if endpoint.startswith("log_")
                            else "mm"
                        ),
                    }
                )
            for j, term in enumerate(model.exog_names):
                if term != "Intercept":
                    diagnostics.append(
                        {
                            "model": name,
                            "endpoint": endpoint,
                            "term": term,
                            "vif": variance_inflation_factor(model.exog, j),
                        }
                    )
    # OLS commonality, same complete-case frame and transformed outcome in all fits.
    # Shared R2 may be negative (suppression); it is not causal variance attribution.
    r_cohort = smf.ols("log1p_gap ~ C(cohort)", base).fit().rsquared
    r_slice = smf.ols("log1p_gap ~ slice_mm", base).fit().rsquared
    r_full = smf.ols("log1p_gap ~ slice_mm + C(cohort)", base).fit().rsquared
    partitions.append(
        {
            "outcome": "log1p_gap_mm",
            "n": len(base),
            "cohort_only_r2": r_cohort,
            "spacing_only_r2": r_slice,
            "full_r2": r_full,
            "unique_cohort_r2": r_full - r_slice,
            "unique_spacing_r2": r_full - r_cohort,
            "shared_r2": r_cohort + r_slice - r_full,
        }
    )
    coeff = pd.DataFrame(coefficients)
    coeff["q_bh"] = np.nan
    non_intercepts = coeff.term != "Intercept"
    coeff.loc[non_intercepts, "q_bh"] = multipletests(
        coeff.loc[non_intercepts, "p"], method="fdr_bh"
    )[1]
    coeff["q_by"] = np.nan
    coeff.loc[non_intercepts, "q_by"] = multipletests(
        coeff.loc[non_intercepts, "p"], method="fdr_by"
    )[1]
    coeff["multiplicity_family_n"] = int(non_intercepts.sum())
    return coeff, pd.DataFrame(diagnostics), pd.DataFrame(partitions)


def repeat_summary(data):
    detail, summaries_ = [], []
    for cohort, d in data.groupby("cohort", sort=True):
        groups = [g for _, g in d.groupby("participant", sort=True) if len(g) >= 2]
        for g in groups:
            vals = g.gap_mm.to_numpy()
            mean = vals.mean()
            detail.append(
                {
                    "cohort": cohort,
                    "participant": g.participant.iloc[0],
                    "n": len(g),
                    "mean_gap_mm": mean,
                    "sd_gap_mm": vals.std(ddof=1),
                    "range_mm": np.ptp(vals),
                    "cv_pct": 100 * vals.std(ddof=1) / mean if mean > 0 else np.nan,
                    "cv_status": "computed" if mean > 0 else "undefined_zero_mean",
                }
            )
        if not groups:
            summaries_.append(
                {
                    "cohort": cohort,
                    "repeated_participants": 0,
                    "status": "no_repeated_measurable_participants",
                }
            )
            continue
        rep = pd.concat(groups)
        icc = one_way_icc([g.gap_mm.to_numpy() for g in groups])
        records = [r for r in detail if r["cohort"] == cohort]
        cv = [r["cv_pct"] for r in records if np.isfinite(r["cv_pct"])]
        sw = np.sqrt(
            sum((r["n"] - 1) * r["sd_gap_mm"] ** 2 for r in records)
            / sum(r["n"] - 1 for r in records)
        )
        # Unequal numbers of longitudinal scans: ICC(1,1), not ICC(2,1).
        # Bootstrap repeated participants; preserve all their scans in each draw.
        rng = np.random.default_rng(CFG.SEED)
        blocks = [g.gap_mm.to_numpy() for g in groups]
        boot, boot_sd = [], []
        for _ in range(N_BOOT):
            indices = rng.integers(0, len(blocks), len(blocks))
            values = np.concatenate([blocks[i] for i in indices])
            labels = np.concatenate(
                [np.repeat(j, len(blocks[i])) for j, i in enumerate(indices)]
            )
            value = one_way_icc([values[labels == j] for j in range(len(indices))])
            boot.append(value)
            selected = [records[i] for i in indices]
            boot_sd.append(np.sqrt(
                sum((r["n"] - 1) * r["sd_gap_mm"] ** 2 for r in selected)
                / sum(r["n"] - 1 for r in selected)))
        if not np.isfinite(boot).all():
            raise ValueError(
                "undefined ICC bootstrap replicate; no silent replicate deletion"
            )
        lo, hi = np.quantile(boot, [CFG.ALPHA / 2, 1 - CFG.ALPHA / 2])
        sd_lo, sd_hi = np.quantile(boot_sd, [CFG.ALPHA / 2, 1 - CFG.ALPHA / 2])
        summaries_.append(
            {
                "cohort": cohort,
                "repeated_participants": len(groups),
                "repeated_scans": len(rep),
                "status": "longitudinal_descriptive",
                "icc_oneway_random_single": icc,
                "icc_percentile_lo": lo,
                "icc_percentile_hi": hi,
                "pooled_within_subject_sd_mm": sw,
                "within_subject_sd_percentile_lo": sd_lo,
                "within_subject_sd_percentile_hi": sd_hi,
                "bootstrap_resamples": N_BOOT,
                "cv_defined_n": len(cv),
                "cv_undefined_zero_mean_n": len(groups) - len(cv),
                "median_subject_cv_pct": np.median(cv) if cv else np.nan,
                "rms_subject_cv_pct": np.sqrt(np.mean(np.square(cv))) if cv else np.nan,
            }
        )
    return pd.DataFrame(detail), pd.DataFrame(summaries_)


def one_way_icc(groups):
    """Unbalanced one-way ANOVA single-measure ICC; retain negative estimates.

    The existing design-effect helper truncates ICC to [0,1] for its variance-
    inflation purpose. That truncation is inappropriate for this signed summary.
    """
    k, n = len(groups), sum(len(g) for g in groups)
    if k < 2 or n <= k:
        return np.nan
    grand = np.concatenate(groups).mean()
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
    msw = sum(np.sum((g - g.mean()) ** 2) for g in groups) / (n - k)
    n0 = (n - sum(len(g) ** 2 for g in groups) / n) / (k - 1)
    denominator = msb + (n0 - 1) * msw
    return (msb - msw) / denominator if denominator > 0 else np.nan


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=OUT)
    ap.add_argument("--measure-topology", action="store_true",
                    help="Recompute dominant-component topology from hash-verified masks")
    ap.add_argument("--refresh-reference-only", action="store_true",
                    help="Acquire and audit the official reference masks using the existing exporter")
    ap.add_argument("--reference-cache", type=Path, default=reference_cache_path(),
                    help="Existing official annotation masks and ZIP metadata")
    ap.add_argument("--grid-resampling-only", action="store_true",
                    help="Opt-in within-mask spacing/phase experiment; never alters primary masks")
    ap.add_argument("--model-grid-mode", choices=["prepare", "preflight", "run", "status", "analyze", "supersede"],
                    help="Full 1,899-case internal/native-grid inference audit; separate from sampling")
    ap.add_argument("--model-grid-supersede-reason", type=str,
                    help="supersede: the reason the analysis/launcher code differs from the frozen environment record")
    ap.add_argument("--model-grid-supersede-decision", type=str,
                    help="supersede: the decision the code change implements (free text stored in the supersession record)")
    ap.add_argument("--model-grid-root", type=Path,
                    help="Dedicated persistent experiment root; never the publication directory")
    ap.add_argument("--model-grid-participants", type=Path, default=OUT / "participant_mapping.csv")
    ap.add_argument("--model-grid-study-root", type=Path,
                    help="Study root that holds the source CT; required for the preflight")
    ap.add_argument("--model-grid-stage-root", type=Path,
                    help="Node-local directory for transient source CT and inference files")
    ap.add_argument("--model-grid-weights-root", type=Path,
                    help="Preinstalled full-resolution weights; defaults to TotalSegmentator configuration")
    ap.add_argument("--model-grid-index", type=int, help="Zero-based index in the preflight manifest")
    ap.add_argument("--model-grid-bootstrap", type=int, default=10000)
    ap.add_argument("--model-grid-seed", type=int, default=20261025)
    grid_source = ap.add_mutually_exclusive_group()
    grid_source.add_argument("--mask-root", type=Path, default=ST.REPO_ROOT / "Local masks",
                    help="Hash-bound cohort/mask_case/structure.nii.gz exports")
    grid_source.add_argument("--multilabel-root", type=Path,
                    help="Existing 2.17.0 baseline cohort/mask_case/{segmentation.nii.gz,report.json}")
    ap.add_argument("--case-table", type=Path, default=OUT / "analysis_cases.csv",
                    help="Canonical case identities, classes and source-mask hashes")
    ap.add_argument("--target-spacings", type=float, nargs="+", default=[2.5, 3.0, 5.0])
    ap.add_argument("--phase-fractions", type=float, nargs="+", default=[0., .25, .5, .75])
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel mask workers for the opt-in resampling experiment")
    ap.add_argument("--grid-preflight-only", action="store_true",
                    help="Verify all selected mask hashes/headers and report resources; no experiment")
    ap.add_argument("--grid-validate-only", action="store_true",
                    help="Verify a completed experiment against sources, code, controls and summaries")
    ap.add_argument("--grid-pilot-cases", type=int,
                    help="Resource pilot: largest-volume cases per cohort; never a publication run")
    ap.add_argument("--stage-mask-root", type=Path,
                    help="Optional node-local byte-identical staging of selected masks")
    ap.add_argument("--worker-memory-gib", type=float,
                    help="Per-worker memory planning budget from the resource pilot; default is conservative full-grid estimate")
    ap.add_argument(
        "--recovered", type=Path, default=OUT / "raw_connectivity_recovered.csv"
    )
    args = ap.parse_args(argv)
    if args.model_grid_mode:
        if args.grid_resampling_only or args.refresh_reference_only or args.measure_topology:
            ap.error("the model-grid audit must run separately from other experiments/regeneration")
        if args.model_grid_mode == "preflight" and (args.multilabel_root is None or args.model_grid_stage_root is None
                                                     or args.model_grid_study_root is None):
            ap.error("model-grid preflight requires --multilabel-root, --model-grid-stage-root and --model-grid-study-root")
        if args.model_grid_mode == "supersede" and not (args.model_grid_supersede_reason and args.model_grid_supersede_decision):
            ap.error("model-grid supersede requires --model-grid-supersede-reason and --model-grid-supersede-decision")
        import model_grid_audit
        model_grid_audit.dispatch(args)
        return
    if args.model_grid_root or args.model_grid_index is not None or args.model_grid_stage_root or args.model_grid_weights_root:
        ap.error("model-grid options require --model-grid-mode")
    if args.grid_resampling_only:
        if args.refresh_reference_only or args.measure_topology:
            ap.error("the resampling experiment must be run separately from primary regeneration")
        grid_resampling_experiment(args.case_table, args.mask_root, args.outdir,
                                   args.target_spacings, args.phase_fractions, args.workers,
                                   preflight_only=args.grid_preflight_only,
                                   validate_only=args.grid_validate_only,
                                   pilot_cases=args.grid_pilot_cases,
                                   stage_root=args.stage_mask_root,
                                   worker_memory_gib=args.worker_memory_gib,
                                   multilabel_root=args.multilabel_root)
        return
    if (args.grid_preflight_only or args.grid_validate_only or args.grid_pilot_cases
            or args.stage_mask_root or args.worker_memory_gib is not None or args.multilabel_root):
        ap.error("grid options require --grid-resampling-only")
    if args.refresh_reference_only:
        refresh_public_reference(args.outdir, args.reference_cache)
        return
    _load_regression_dependencies()
    data, flow, participants = load_analysis(args.recovered)
    data = publication_topology(data, args.outdir, args.measure_topology)
    data = add_examination_dates(data)
    flow = add_missing_label_bounds(data, flow)
    summary = summaries(data)
    coefficients, vif, partition = model_results(data)
    repeat_detail, repeats = repeat_summary(data)
    repeat_detail, repeats = add_followup_intervals(data, repeat_detail, repeats)
    topology = topology_summaries(data)
    repeated = data[data.cohort == "autoPET"]
    binary, binary_detail = longitudinal_agreement(
        repeated, 2, {"direct": 0, "right_iliac_mediated": 0, "disconnected": 1})
    nominal, nominal_detail = longitudinal_agreement(
        repeated, 3, {name: i for i, name in enumerate(CLASSES)})
    binary.insert(0, "endpoint", "binary")
    nominal.insert(0, "endpoint", "three_class_nominal")
    agreements = pd.concat([binary, nominal], ignore_index=True)
    repeat_detail = repeat_detail.merge(
        binary_detail.rename(columns={"all_visits_agree": "binary_all_visits_agree",
                                      "pair_agreement": "binary_pair_agreement"})[
            ["participant", "binary_all_visits_agree", "binary_pair_agreement"]],
        on="participant", how="left", validate="one_to_one")
    repeat_detail = repeat_detail.merge(
        nominal_detail.rename(columns={"all_visits_agree": "nominal_all_visits_agree",
                                       "pair_agreement": "nominal_pair_agreement"})[
            ["participant", "nominal_all_visits_agree", "nominal_pair_agreement"]],
        on="participant", how="left", validate="one_to_one")
    reference_path = args.outdir / "reference_cases.csv"
    if not reference_path.exists():
        raise ValueError("Run --refresh-reference-only to audit the released reference labels first")
    reference = pd.read_csv(reference_path)
    topology = pd.concat([topology, reference_summaries(reference)], ignore_index=True)
    gee_verified = verify_population_average_models(data, coefficients)
    availability = (
        data.groupby("cohort")
        .agg(
            n=("case", "size"),
            slice_spacing_available=("slice_mm", "count"),
            contrast_available=("contrast", "count"),
            body_csa_available=("body_csa", "count"),
            softmax_available=("softmax", "count"),
        )
        .reset_index()
    )
    outputs = {
        "analysis_cases.csv": data,
        "cohort_flow.csv": flow,
        "participant_mapping.csv": participants,
        "summary.csv": summary,
        "covariate_availability.csv": availability,
        "model_coefficients.csv": coefficients,
        "model_vif.csv": vif,
        "variance_partition.csv": partition,
        "repeat_participants.csv": repeat_detail,
        "repeat_summary.csv": repeats,
        "topology_summary.csv": topology,
        "longitudinal_agreement.csv": agreements,
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(args.outdir / name, index=False, float_format="%.10g")
    inputs = ST._analysis_input_paths() + [
        args.recovered,
        ST.REPO_ROOT / "Local masks/MANIFEST.csv",
        ST.REPO_ROOT / "Code/phase1_characterize/measure_collapsed_gap.py",
        ST.REPO_ROOT / "Code/phase0_data/export_relevant_masks.py",
        reference_path,
        args.outdir / "reference_source_files.csv",
        ST.RESULTS / "cohort_table1.csv",
    ]
    # Bind the released split and ZIP index as well as the extracted mask inventory.
    reference_cache = args.reference_cache
    inputs += [reference_cache / name for name in
               ("meta.csv", "central_directory.bin", "selected_entries.json", "zenodo_record.json")]

    def entry(path):
        path = Path(path)
        try:
            display = str(path.relative_to(ST.REPO_ROOT))
        except ValueError:
            display = str(path)
        return {
            "path": display,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    payload = {
        "seed": CFG.SEED,
        "n_boot": N_BOOT,
        "topology_and_agreement_bootstraps": N_BOOT,
        "regression_multiplicity_family_n": int((coefficients.term != "Intercept").sum()),
        "regression_multiplicity_methods": ["Benjamini-Hochberg", "Benjamini-Yekutieli"],
        "intervals": "participant percentile for pooled/repeated topology; Wilson for independent single-cohort strata; participant percentile for agreement/SD/ICC; BCa for medians",
        "population_average_model_point_estimates_match_independence_GEE": gee_verified,
        "public_reference": {
            "record": EM.PUBLIC_REFERENCE_API,
            "archive": EM.PUBLIC_REFERENCE_URL,
            "version": "2.0.1",
            "whole_archive_checksum_verified": False,
            "entry_integrity": "ZIP CRC32, byte length and extracted-file SHA-256",
            "released_cases": len(reference),
            "both_primary_labels": int((reference.status == "measurable").sum()),
            "separate_from_predicted_cohorts": True,
            "prevalence_ci_method": "Wilson score, independent released cases assumed; patient linkage unavailable",
            "boundary_contact_rule": "any left iliac voxel in an outermost array slice on any axis",
        },
        "thin_slice_cutoff_mm": THIN_MM,
        "status": "exploratory_no_reference_standard_claims",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "inputs": [entry(p) for p in inputs],
        "outputs": [
            entry(args.outdir / n)
            for n in sorted(outputs)
        ],
    }
    (args.outdir / "supplement_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(f"Wrote {len(data)} measurable cases -> {args.outdir}")


# Endpoints shared by the analyses below.
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import importlib.util

# Load Code/phase0_data/export_relevant_masks.py explicitly by path, regardless of PYTHONPATH or
# the working directory.
_exporter_path = Path(__file__).resolve().parent.parent / "phase0_data/export_relevant_masks.py"
_exporter_spec = importlib.util.spec_from_file_location("ivcs_phase0_exporter", _exporter_path)
EM = importlib.util.module_from_spec(_exporter_spec)
_exporter_spec.loader.exec_module(EM)
import measure_collapsed_gap as MC

N_BOOT = 10000
SEED = CFG.SEED
CLASSES = MC.TOPOLOGY_CLASSES


def nearest_neighbor_plane_indices(n_planes, native_mm, target_mm, phase_fraction):
    """Sample centers inside the original closed-voxel extent; phase is anchored at z=0.

    Native voxel slabs are half open [-s/2, (n-.5)s), making the nearest-neighbor
    tie rule explicit (ties select the higher index). No intensity interpolation,
    padding, or inferred fine-grid prediction is introduced.
    """
    if n_planes < 1 or not np.isfinite([native_mm, target_mm, phase_fraction]).all():
        raise ValueError("invalid resampling grid")
    if native_mm <= 0 or target_mm <= 0 or not 0 <= phase_fraction < 1:
        raise ValueError("spacing must be positive and phase must lie in [0,1)")
    shift = phase_fraction * target_mm
    first = int(np.ceil((-.5*native_mm-shift)/target_mm))
    upper = np.nextafter((n_planes-.5)*native_mm, -np.inf)
    last = int(np.floor((upper-shift)/target_mm))
    centers = shift + np.arange(first, last+1)*target_mm
    indices = np.floor(centers/native_mm+.5).astype(int)
    if ((indices < 0) | (indices >= n_planes)).any():
        raise AssertionError("resampling escaped the original voxel extent")
    return indices


def _grid_mask_paths(row, mask_root):
    """Resolve flat exports or existing full-resolution separate-mask folders.

    A matching byte hash is mandatory. A multilabel re-export is not silently
    substituted, since its identity would differ from the canonical audit.
    """
    folder = Path(mask_root) / row["cohort"] / row["mask_case"]
    paths = []
    for name in MC.REFERENCE_VEIN_FILES:
        candidates = [folder / f"{name}.nii.gz",
                      folder / "segmentations_fullres" / f"{name}.nii.gz"]
        present = [p for p in candidates if p.is_file()]
        matching = [p for p in present if MC.topology_digest(p) == row[f"{name}_sha256"]]
        if not matching:
            if present:
                raise ValueError(f"mask hash mismatch: {row['cohort']}/{row['case']}/{name}")
            raise FileNotFoundError(f"Missing separate full-resolution mask; tried {candidates}")
        paths.append(matching[0])
    return paths


def _grid_frame(case_table, spacings, phases, pilot_cases=None):
    if not spacings or not phases or len(set(spacings)) != len(spacings) or len(set(phases)) != len(phases):
        raise ValueError("grid settings must be nonempty and unique")
    for spacing in spacings:
        for phase in phases:
            nearest_neighbor_plane_indices(2, 1.5, spacing, phase)
    if pilot_cases is not None and pilot_cases < 1:
        raise ValueError("pilot cases per cohort must be positive")
    data = pd.read_csv(case_table)
    required = {"cohort", "case", "participant", "mask_case", "slice_mm", "class26", "class6",
                "ivc_voxels", "lciv_voxels", "rciv_voxels"} | {
                    f"{name}_sha256" for name in MC.REFERENCE_VEIN_FILES}
    if not required <= set(data) or data[list(required)].isna().any().any():
        raise ValueError("resampling requires the complete canonical case schema")
    if not np.isfinite(data.slice_mm).all() or (data.slice_mm <= 0).any():
        raise ValueError("missing or invalid spacing must not shrink the resampling frame")
    if data.duplicated(["cohort", "case"]).any() or data.duplicated(["cohort", "mask_case"]).any():
        raise ValueError("duplicate case or source mask in resampling frame")
    if not data.class26.isin(CLASSES).all() or not data.class6.isin(CLASSES).all():
        raise ValueError("unknown canonical topology class")
    # Identifiers are path components, never executable paths or traversal tokens.
    for col in ("cohort", "case", "mask_case"):
        if any(not isinstance(x, str) or x in (".", "..") or "/" in x or "\\" in x
               for x in data[col]):
            raise ValueError("unsafe case path component")
    for name in MC.REFERENCE_VEIN_FILES:
        if not data[f"{name}_sha256"].str.fullmatch(r"[0-9a-f]{64}").all():
            raise ValueError("invalid source SHA-256")
    selected = data[data.slice_mm <= THIN_MM].sort_values(["cohort", "case"])
    if selected.empty:
        raise ValueError("no cases in the ≤1.5-mm resampling frame")
    return selected


def _check_grid_image(img, row, path):
    axes = img.affine[:3, :3]
    lengths = np.linalg.norm(axes, axis=0)
    if (len(img.shape) != 3 or not np.isfinite(img.affine).all()
            or (lengths <= 0).any()
            or not np.allclose((axes / lengths).T @ (axes / lengths), np.eye(3), atol=1e-5)
            or not np.allclose(lengths, img.header.get_zooms()[:3], atol=1e-4)
            or not np.isclose(lengths[2], row["slice_mm"], atol=1e-4, rtol=1e-5)):
        raise ValueError(f"invalid or nonorthogonal physical grid: {path}")


def _grid_multilabel_headers(selected, source_root, pilot_cases):
    """Read each report's own class map and bind the immutable multilabel sources."""
    import nibabel as nib
    rows = []
    for i, row in enumerate(selected.to_dict("records"), 1):
        folder = Path(source_root)/row["cohort"]/row["mask_case"]
        EM.baseline_class_index(folder, MC.REFERENCE_VEIN_FILES)
        seg = folder/"segmentation.nii.gz"
        img = nib.load(seg)
        _check_grid_image(img, row, seg)
        if not np.issubdtype(img.get_data_dtype(), np.integer):
            raise ValueError(f"multilabel volume must have integer storage: {seg}")
        row["full_grid_voxels"] = int(np.prod(img.shape, dtype=np.int64))
        row["multilabel_sha256"] = MC.topology_digest(seg)
        row["report_sha256"] = MC.topology_digest(folder/"report.json")
        rows.append(row)
        if i % 25 == 0 or i == len(selected):
            print("Multilabel source inventory:", i, "of", len(selected), flush=True)
    frame = pd.DataFrame(rows)
    if pilot_cases is not None:
        frame = (frame.sort_values(["cohort", "full_grid_voxels", "case"], ascending=[True, False, True])
                 .groupby("cohort", sort=True).head(pilot_cases))
    return frame.sort_values(["cohort", "case"])


def _grid_export_multilabel_case(task):
    """Use the original exporter, then require canonical bytes and voxel counts."""
    import resource
    row, source_root, stage_root = task
    started = time.monotonic()
    folder = Path(source_root)/row["cohort"]/row["mask_case"]
    # Never mix a changing upstream volume/report with a frozen experiment identity.
    for name, col in (("segmentation.nii.gz", "multilabel_sha256"), ("report.json", "report_sha256")):
        if MC.topology_digest(folder/name) != row[col]:
            raise ValueError("multilabel source changed after inventory")
    dest = Path(stage_root)/row["cohort"]/row["mask_case"]
    complete = all((dest/f"{name}.nii.gz").is_file() and
                   MC.topology_digest(dest/f"{name}.nii.gz") == row[f"{name}_sha256"]
                   for name in MC.REFERENCE_VEIN_FILES)
    if not complete:
        outputs, error = EM.export_case((row["cohort"], row["mask_case"], str(source_root),
            str(stage_root), True, "", None), structures=MC.REFERENCE_VEIN_FILES)
        if error:
            raise ValueError(f"multilabel extraction failed: {error}")
        by_name = {x["structure"]: x for x in outputs}
        for name, short in zip(MC.REFERENCE_VEIN_FILES, ("ivc", "lciv", "rciv")):
            result = by_name[name]
            if result["voxels"] != int(row[f"{short}_voxels"]):
                raise ValueError(f"multilabel extraction voxel mismatch: {row['cohort']}/{row['case']}/{name}")
            if result["sha256"] != row[f"{name}_sha256"]:
                raise ValueError(f"canonical export hash mismatch: {row['cohort']}/{row['case']}/{name}; "
                    "packing, geometry or voxel identity must be checked; equal counts alone are insufficient")
    for name, col in (("segmentation.nii.gz", "multilabel_sha256"), ("report.json", "report_sha256")):
        if MC.topology_digest(folder/name) != row[col]:
            raise ValueError("multilabel source changed during extraction")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("Multilabel extraction verified:", row["cohort"], row["case"],
          "cached="+str(complete), "seconds="+str(round(time.monotonic()-started, 1)),
          "process_peak_bytes="+str(int(peak if platform.system() == "Darwin" else peak*1024)), flush=True)
    return row["cohort"], row["case"]


def _grid_prepare_multilabel(selected, source_root, stage_root, workers, pilot_cases):
    import hpc_resources as HR
    from multiprocessing import get_context
    if stage_root is None or Path(stage_root).resolve() == Path(source_root).resolve():
        raise ValueError("multilabel input requires a distinct temporary --stage-mask-root")
    # Prevent writing exports anywhere in the original source hierarchy.
    if Path(stage_root).resolve().is_relative_to(Path(source_root).resolve()):
        raise ValueError("temporary exports must be outside the original multilabel tree")
    selected = _grid_multilabel_headers(selected, source_root, pilot_cases)
    # Extraction is separately bounded; the topology pilot budget cannot under-size it.
    estimate = max(1024**3, int(selected.full_grid_voxels.max())*16 + 512*1024**2)
    memory = HR.available_memory_bytes()
    cap = max(0, (memory-4*1024**3)//estimate) if memory else 2
    resolved = min(workers, HR.visible_cpu_count(), cap, len(selected))
    if resolved < 1:
        raise ValueError("allocation cannot fit a multilabel extraction worker plus parent reserve")
    print("Temporary vein extraction: workers=", resolved, "estimated bytes per worker=", estimate, flush=True)
    HR.limit_blas_threads(1)
    tasks = [(row, source_root, stage_root) for row in selected.to_dict("records")]
    with ProcessPoolExecutor(max_workers=resolved, mp_context=get_context("spawn"),
                             initializer=HR.limit_blas_threads, initargs=(1,)) as pool:
        finished = list(pool.map(_grid_export_multilabel_case, tasks))
    if set(finished) != set(zip(selected.cohort, selected.case)):
        raise AssertionError("incomplete multilabel extraction")
    return _grid_preflight(selected, stage_root)

def _grid_preflight(selected, mask_root, stage_root=None, pilot_cases=None):
    """Verify every source header and hash before choosing a worst-volume pilot."""
    import nibabel as nib
    rows, paths_by_key = [], {}
    for i, row in enumerate(selected.to_dict("records"), 1):
        paths = _grid_mask_paths(row, mask_root)
        first = None
        for path in paths:
            img = nib.load(path)
            _check_grid_image(img, row, path)
            if first is not None and (img.shape != first.shape or not np.allclose(img.affine, first.affine, atol=1e-5)):
                raise ValueError(f"grid mismatch: {path}")
            first = img
        row["full_grid_voxels"] = int(np.prod(first.shape, dtype=np.int64))
        paths_by_key[(row["cohort"], row["case"])] = paths
        rows.append(row)
        if i % 25 == 0 or i == len(selected):
            print(f"Grid source preflight: {i}/{len(selected)} cases verified", flush=True)
    frame = pd.DataFrame(rows)
    if pilot_cases is not None:
        frame = (frame.sort_values(["cohort", "full_grid_voxels", "case"], ascending=[True, False, True])
                 .groupby("cohort", sort=True).head(pilot_cases))
    frame = frame.sort_values(["cohort", "case"])
    if stage_root is not None:
        if Path(stage_root).resolve() == Path(mask_root).resolve():
            raise ValueError("staging must have a distinct root")
        for row in frame.to_dict("records"):
            dest = Path(stage_root) / row["cohort"] / row["mask_case"]
            dest.mkdir(parents=True, exist_ok=True)
            for source, name in zip(paths_by_key[(row["cohort"], row["case"])], MC.REFERENCE_VEIN_FILES):
                target = dest / f"{name}.nii.gz"
                if not target.is_file() or MC.topology_digest(target) != row[f"{name}_sha256"]:
                    pending = target.with_name(target.name + ".partial")
                    shutil.copyfile(source, pending)
                    if MC.topology_digest(pending) != row[f"{name}_sha256"]:
                        raise ValueError("staged mask hash mismatch")
                    pending.replace(target)
        mask_root = stage_root
    return frame, Path(mask_root)


def _classify_fixed_grid(masks, a, b):
    """Reachability for sampled original endpoint sets, without selecting new trunks."""
    if not a.any() or not b.any():
        return {"class26": "empty_primary_label", "class6": "empty_primary_label"}
    result = {}
    for conn in (26, 6):
        structure = scipy.ndimage.generate_binary_structure(3, 3 if conn == 26 else 1)
        pair_lab, _ = scipy.ndimage.label(masks[0] | masks[1], structure)
        direct = MC.share_component(pair_lab, a, b)
        del pair_lab
        lab, _ = scipy.ndimage.label(masks[0] | masks[1] | masks[2], structure)
        reach = MC.share_component(lab, a, b)
        result[f"class{conn}"] = CLASSES[0 if direct else 1 if reach else 2]
    if result["class26"] == "disconnected" and result["class6"] != "disconnected":
        raise AssertionError("fixed-endpoint adjacency nesting failed")
    return result


def _grid_resampling_case(task):
    import nibabel as nib
    import resource
    started = time.monotonic()
    row, mask_root, spacings, phases = task
    # Workers always read a flat staging view. Direct callers may use either layout.
    paths = _grid_mask_paths(row, mask_root)
    arrays, boxes = [], []
    first = None
    for path, short in zip(paths, ("ivc", "lciv", "rciv")):
        img = nib.load(path)
        arr = np.asanyarray(img.dataobj)
        if (arr.ndim != 3 or not np.isin(arr, [0, 1]).all()
                or int(np.count_nonzero(arr)) != int(row[f"{short}_voxels"])):
            raise ValueError("manifest voxel count mismatch or nonbinary mask")
        if not np.isclose(img.header.get_zooms()[2], row["slice_mm"], atol=1e-4):
            raise ValueError("resampling through-plane axis disagrees with case spacing")
        if first is not None and (img.shape != first.shape or not np.allclose(img.affine, first.affine, atol=1e-5)):
            raise ValueError("grid mismatch")
        first = img
        # Full z extent preserves the physical origin and subvoxel phase.
        if arr.any():
            xy = [np.flatnonzero(arr.any(axis=tuple(j for j in range(3) if j != i))) for i in (0, 1)]
            box = tuple(slice(int(x[0]), int(x[-1])+1) for x in xy)
            crop = arr[box+(slice(None),)].astype(bool)
        else:
            box, crop = None, None
        boxes.append(box)
        arrays.append(crop)
        n_planes = arr.shape[2]
        del arr
    nonempty = [b for b in boxes if b is not None]
    lo = np.min([[s.start for s in b] for b in nonempty], axis=0)
    hi = np.max([[s.stop for s in b] for b in nonempty], axis=0)
    masks = []
    for arr, box in zip(arrays, boxes):
        mask = np.zeros((*tuple(hi-lo), n_planes), bool)
        if box is not None:
            mask[tuple(slice(s.start-a, s.stop-a) for s, a in zip(box, lo))+(slice(None),)] = arr
        masks.append(mask)
    del arrays
    baseline = MC.classify_topology(masks)
    a = MC.largest_component26(masks[0])[0]
    b = MC.largest_component26(masks[1])[0]
    identity = nearest_neighbor_plane_indices(n_planes, row["slice_mm"], row["slice_mm"], 0.)
    if not np.array_equal(identity, np.arange(n_planes)):
        raise AssertionError("identity sampling altered native planes")
    fixed_identity = _classify_fixed_grid(masks, a, b)
    for conn in (26, 6):
        if baseline[f"class{conn}"] != row[f"class{conn}"] or fixed_identity[f"class{conn}"] != row[f"class{conn}"]:
            raise ValueError("baseline resampling case does not match canonical topology")
    records = []
    for spacing in spacings:
        for phase in phases:
            planes = nearest_neighbor_plane_indices(n_planes, row["slice_mm"], spacing, phase)
            sampled = [m[:, :, planes] for m in masks]
            aa, bb = a[:, :, planes], b[:, :, planes]
            lost = not len(planes) or not sampled[0].any() or not sampled[1].any()
            fixed_lost = not aa.any() or not bb.any()
            measured = {} if lost else MC.classify_topology(sampled)
            fixed = _classify_fixed_grid(sampled, aa, bb)
            new_a = MC.largest_component26(sampled[0])[0] if not lost else aa
            new_b = MC.largest_component26(sampled[1])[0] if not lost else bb
            records.append({"cohort": row["cohort"], "case": row["case"],
                "participant": row["participant"], "mask_case": row["mask_case"],
                "native_spacing_mm": row["slice_mm"], "target_spacing_mm": spacing,
                "phase_fraction": phase, "sampled_planes": len(planes),
                "baseline_class26": row["class26"], "baseline_class6": row["class6"],
                "resampled_class26": measured.get("class26", "empty_primary_label"),
                "resampled_class6": measured.get("class6", "empty_primary_label"),
                "fixed_class26": fixed["class26"], "fixed_class6": fixed["class6"],
                "primary_label_lost": int(lost), "fixed_endpoint_lost": int(fixed_lost),
                "endpoint_sets_changed": int(not np.array_equal(aa, new_a) or not np.array_equal(bb, new_b)),
                "ivc_endpoint_sampled_voxels": int(aa.sum()), "lciv_endpoint_sampled_voxels": int(bb.sum()),
                "identity_control_passed": 1})
    # Operational data are separate from scientific outcomes and deterministic tables.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"records": records, "resource": {"cohort": row["cohort"], "case": row["case"],
        "elapsed_seconds": time.monotonic()-started,
        "worker_process_peak_rss_bytes": int(peak if platform.system() == "Darwin" else peak*1024)}}


def _grid_summaries(cases, phases):
    summaries_, paired = [], []
    for spacing, frame in cases.groupby("target_spacing_mm", sort=True):
        for cohort, group in [("Pooled", frame), *frame.groupby("cohort", sort=True)]:
            for endpoint, prefix, missing in [("reidentified", "resampled", "primary_label_lost"),
                                               ("fixed_native", "fixed", "fixed_endpoint_lost")]:
                for conn in (26, 6):
                    baseline = (group[f"baseline_class{conn}"] == "disconnected").astype(int)
                    value = group[f"{prefix}_class{conn}"]
                    low = (value == "disconnected").astype(int)
                    high = ((low == 1) | (group[missing] == 1)).astype(int)
                    common = {"cohort": cohort, "target_spacing_mm": spacing, "endpoint_rule": endpoint,
                              "connectivity": conn, "distinct_cases": len(group[["cohort", "case"]].drop_duplicates()),
                              "phases_per_case": len(phases)}
                    for name, event in [("disconnected_lower_bound", low), ("disconnected_upper_bound", high),
                        ("primary_label_lost", group[missing] == 1),
                        ("configuration_changed", value != group[f"baseline_class{conn}"])]:
                        summaries_.append({**common, "outcome": name,
                            **proportion_intervals(group, event, independent_wilson=False)})
                    # Resample the signed within-case contrast directly: paired, never independent CIs.
                    for name, values in [("disconnection_change_lower_bound", low-baseline),
                                         ("disconnection_change_upper_bound", high-baseline)]:
                        stats = proportion_intervals(group, values, independent_wilson=False)
                        stats["mean_change"] = stats.pop("proportion")
                        stats["signed_sum"] = stats.pop("events")
                        paired.append({**common, "outcome": name, **stats})
                    for start_class in CLASSES:
                        for end_class in (*CLASSES, "empty_primary_label"):
                            part = group[group[f"baseline_class{conn}"] == start_class]
                            # Conditional transitions use the fixed native baseline stratum.
                            summaries_.append({**common, "outcome": f"transition:{start_class}->{end_class}",
                                "distinct_cases": len(part[["cohort", "case"]].drop_duplicates()),
                                **proportion_intervals(part, part[f"{prefix}_class{conn}"] == end_class,
                                                       independent_wilson=False)})
    return pd.DataFrame(summaries_), pd.DataFrame(paired)


def _grid_identity(case_table, selected, spacings, phases, pilot_cases):
    import hpc_resources as HR
    code = [Path(__file__), Path(MC.__file__), Path(S.__file__), Path(CFG.__file__),
            Path(ST.__file__), Path(EM.__file__), Path(HR.__file__)]
    source_columns = ["cohort", "case", "mask_case"]+[f"{n}_sha256" for n in MC.REFERENCE_VEIN_FILES]
    source_columns += [c for c in ("multilabel_sha256", "report_sha256") if c in selected]
    return {"schema": 3, "case_table_sha256": MC.topology_digest(case_table),
        "case_n": len(selected), "target_spacings_mm": list(spacings), "phase_fractions": list(phases),
        "seed": SEED, "n_boot": N_BOOT, "pilot_cases_per_cohort": pilot_cases,
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "pandas": pd.__version__, "nibabel": MC.nib.__version__,
        "code": {p.name: MC.topology_digest(p) for p in code},
        "source_case_keys": selected[source_columns].to_dict("records")}


def _grid_output_frames(payloads, phases):
    records = [r for p in payloads for r in p["records"]]
    cases = pd.DataFrame(records).sort_values(["cohort", "case", "target_spacing_mm", "phase_fraction"])
    controls = cases.groupby(["cohort", "case", "participant"], sort=True).agg(
        identity_control_passed=("identity_control_passed", "min"),
        baseline_class26=("baseline_class26", "first"), baseline_class6=("baseline_class6", "first")).reset_index()
    summary, paired = _grid_summaries(cases, phases)
    resources = pd.DataFrame([p["resource"] for p in payloads]).sort_values(["cohort", "case"])
    frames = {"grid_resampling_cases.csv": cases, "grid_resampling_summary.csv": summary,
            "grid_resampling_paired.csv": paired, "grid_resampling_controls.csv": controls,
            "grid_resampling_resources.csv": resources}
    # JSON checkpoint serialization sorts keys; fix column order for identical
    # first-run/resumed bytes, regardless of which cases came from the cache.
    return {name: frame.reindex(sorted(frame.columns), axis=1) for name, frame in frames.items()}


def grid_resampling_experiment(case_table, mask_root, outdir, spacings, phases, workers=1, *,
                               preflight_only=False, validate_only=False, pilot_cases=None, stage_root=None,
                               worker_memory_gib=None, multilabel_root=None):
    """CPU sampling sensitivity with source verification and transactional resumption.

    Equal weight is assigned to each scheduled phase and examination; participants
    (with every phase/visit) are bootstrapped within cohort. This is a discrete
    sensitivity design, not a model of CT acquisition or latent anatomy.
    """
    import fcntl
    import hpc_resources as HR
    from multiprocessing import get_context
    if workers < 1 or (preflight_only and validate_only):
        raise ValueError("positive workers and a single grid execution mode required")
    if worker_memory_gib is not None and (not np.isfinite(worker_memory_gib) or worker_memory_gib < .5):
        raise ValueError("worker memory budget must be finite and at least 0.5 GiB")
    selected = _grid_frame(case_table, spacings, phases, pilot_cases)
    if multilabel_root is not None:
        selected, run_root = _grid_prepare_multilabel(selected, multilabel_root, stage_root, workers, pilot_cases)
    else:
        selected, run_root = _grid_preflight(selected, mask_root, stage_root, pilot_cases)
    identity = _grid_identity(case_table, selected, spacings, phases, pilot_cases)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    # Conservative per-worker memory estimate for topology labelling (64 bytes per full-grid voxel).
    estimate = max(512*1024**2, int(selected.full_grid_voxels.max())*64)
    if worker_memory_gib is not None:
        estimate = int(worker_memory_gib*1024**3)
    memory = HR.available_memory_bytes()
    memory_cap = max(0, (memory-4*1024**3)//estimate) if memory else 8
    resolved = min(workers, HR.visible_cpu_count(), memory_cap, len(selected))
    print(f"Grid preflight: {len(selected)} cases; {len(selected)*len(spacings)*len(phases)} rows; "
          f"workers requested={workers}, resolved={resolved}; estimated worker bytes={estimate}; "
          f"allocation bytes={memory}; pilot={pilot_cases is not None}", flush=True)
    if resolved < 1:
        raise ValueError("allocation cannot fit one conservatively estimated worker plus 4 GiB reserve")
    if preflight_only:
        return
    outdir = Path(outdir)
    if outdir.resolve() == Path(case_table).resolve().parent:
        raise ValueError("experiment output must be separate from primary publication results")
    outdir.mkdir(parents=True, exist_ok=True)
    checkpoint = outdir / "grid_resampling_checkpoint.sqlite3"
    if validate_only and not checkpoint.is_file():
        raise ValueError("completed experiment checkpoint missing")
    # Separate lock inode: on macOS flock on the database conflicts with SQLite's
    # own POSIX record locks. This lock is also held through final publication.
    with checkpoint.with_suffix(".lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another job is already using this experiment directory") from exc
        with sqlite3.connect(checkpoint) as db:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS cases (cohort TEXT, id TEXT, payload TEXT NOT NULL, sha256 TEXT NOT NULL, PRIMARY KEY(cohort,id))")
            prior = db.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()
            if prior is not None and prior[0] != fingerprint:
                raise ValueError("checkpoint settings/code/environment/source identity changed; use a new output directory")
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('fingerprint', ?)", (fingerprint,))
            db.commit()
            payloads = {}
            expected = {(r.cohort, r.case) for r in selected.itertuples()}
            for cohort, case, payload, digest in db.execute("SELECT cohort,id,payload,sha256 FROM cases"):
                if (cohort, case) not in expected or hashlib.sha256(payload.encode()).hexdigest() != digest:
                    raise ValueError("foreign or corrupt checkpoint case")
                payloads[(cohort, case)] = json.loads(payload)
            pending = [r for r in selected.to_dict("records") if (r["cohort"], r["case"]) not in payloads]
            if validate_only and pending:
                raise ValueError("incomplete experiment cannot be validated for publication")
            if not validate_only:
                HR.limit_blas_threads(1)
                print(f"Grid resume: {len(payloads)} complete; {len(pending)} pending", flush=True)
                tasks = [(r, run_root, spacings, phases) for r in pending]
                # Spawn avoids inheriting initialized BLAS state and the checkpoint lock.
                with ProcessPoolExecutor(max_workers=resolved, mp_context=get_context("spawn"),
                                         initializer=HR.limit_blas_threads, initargs=(1,)) as pool:
                    for row, payload in zip(pending, pool.map(_grid_resampling_case, tasks)):
                        key = (row["cohort"], row["case"])
                        raw = json.dumps(payload, sort_keys=True)
                        db.execute("INSERT INTO cases VALUES (?,?,?,?)", (*key, raw, hashlib.sha256(raw.encode()).hexdigest()))
                        db.commit()
                        payloads[key] = payload
                        print("Grid progress:", len(payloads), "of", len(selected), "case", key,
                              "worker seconds:", round(payload['resource']['elapsed_seconds'], 1), flush=True)
            ordered = [payloads[key] for key in sorted(expected)]
            frames = _grid_output_frames(ordered, phases)
            cases = frames["grid_resampling_cases.csv"]
            expected_rows = len(selected)*len(spacings)*len(phases)
            cell_columns = ["cohort", "case", "target_spacing_mm", "phase_fraction"]
            expected_cells = {(c, k, s, p) for c, k in expected for s in spacings for p in phases}
            actual_cells = set(cases[cell_columns].itertuples(index=False, name=None))
            if len(cases) != expected_rows or actual_cells != expected_cells:
                raise AssertionError("incomplete case-by-spacing-by-phase experiment")
            canonical = selected.set_index(["cohort", "case"])
            for row in cases.itertuples():
                source = canonical.loc[(row.cohort, row.case)]
                if row.participant != source.participant or any(
                        getattr(row, f"baseline_class{conn}") != source[f"class{conn}"] for conn in (26, 6)):
                    raise AssertionError("experiment row identity disagrees with canonical case")
            if not frames["grid_resampling_controls.csv"].identity_control_passed.eq(1).all():
                raise AssertionError("identity control failed")
            manifest_path = outdir / "grid_resampling_manifest.json"
            if validate_only:
                manifest = json.loads(manifest_path.read_text())
                if manifest["fingerprint"] != fingerprint or manifest["case_phase_grid_rows"] != expected_rows:
                    raise ValueError("result manifest identity mismatch")
                names = [item["path"] for item in manifest["outputs"]]
                if len(names) != len(frames) or set(names) != set(frames):
                    raise ValueError("experiment manifest has an incomplete or foreign output inventory")
                for item in manifest["outputs"]:
                    path = outdir / item["path"]
                    if MC.topology_digest(path) != item["sha256"]:
                        raise ValueError("experiment output hash mismatch")
                    # Deterministic reconstruction of every summary from case checkpoints.
                    actual = pd.read_csv(path)
                    pd.testing.assert_frame_equal(actual, frames[item["path"]].reset_index(drop=True),
                                                  check_dtype=False, rtol=1e-8, atol=1e-9)
                print("Grid experiment validation MATCH (source hashes, identity, completeness, all summaries)", flush=True)
                return
            # A completion manifest is published last; interrupted partial CSVs cannot be mistaken for a completed run.
            manifest_path.unlink(missing_ok=True)
            for name, frame in frames.items():
                pending_path = outdir / (name+".partial")
                frame.to_csv(pending_path, index=False, float_format="%.10g")
                pending_path.replace(outdir/name)
            provenance = {**identity, "fingerprint": fingerprint,
                "status": "completed_resource_pilot_not_publication" if pilot_cases else "executed_grid_sampling_sensitivity_not_inference",
                "publication_eligible_frame": pilot_cases is None,
                "source_hashes_verified": True, "case_phase_grid_rows": len(cases),
                "identity_controls_passed": len(selected), "resolved_workers": resolved,
                "source_format": "multilabel_with_canonical_binary_verification" if multilabel_root else "separate_binary_exports",
                "estimated_peak_bytes_per_worker": estimate,
                "memory_estimate_source": "user pilot-based budget" if worker_memory_gib is not None else "64 bytes per full-image voxel heuristic",
                "endpoint_rule": "reidentified dominant components plus sampled fixed native endpoint sets",
                "in_plane_sampling": "unchanged", "through_plane_axis": "native NIfTI index axis 2",
                "uncertainty": "pointwise paired cohort-stratified participant percentile intervals; equal phase/examination weights",
                "phase_interpretation": "discrete perturbations, not independent observations or uniform physical-origin probabilities",
                "inference_grid_recovered": False, "source_ct_or_clinical_accuracy_assessed": False,
                "resource_rss": "worker process high-water mark, potentially includes earlier cases in that process",
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "outputs": [{"path": name, "sha256": MC.topology_digest(outdir/name)} for name in frames]}
            pending_manifest = manifest_path.with_suffix(".json.partial")
            pending_manifest.write_text(json.dumps(provenance, indent=2)+"\n")
            pending_manifest.replace(manifest_path)
    print(f"Grid experiment completed: {len(selected)} cases; {len(cases)} case/spacing/phase rows", flush=True)


def reference_cache_path():
    moved = ST.REPO_ROOT / "Totalsegmentor_traning_data"
    return moved if moved.is_dir() else Path.home() / ".cache/ivcs/totalsegmentator-v201"


def add_examination_dates(data, ct_metadata=None):
    """Match autoPET folder dates to released CT metadata before chronological selection."""
    d = data.copy()
    d["study_date"] = ""
    d["earliest_examination"] = 1
    pet = d[d.cohort == "autoPET"]
    if pet.empty:
        return d
    meta = (pd.read_csv(ST.META_DIR / "autoPET.csv")
            if ct_metadata is None else ct_metadata.copy())
    meta = meta[meta.Modality == "CT"]
    released = set(zip(meta["Subject ID"], pd.to_datetime(meta["Study Date"]).dt.date))
    for r in pet.itertuples():
        date = ST._SubjectKeyedMeta._study_date(r.case)
        if date is None:
            raise ValueError(f"invalid autoPET examination date: {r.case}")
        month, day, year = date
        stamp = pd.Timestamp(year=year, month=month, day=day)
        if (ST._autopet_subject_id(r.case), stamp.date()) not in released:
            raise ValueError("autoPET folder date does not match released CT study metadata")
        d.loc[r.Index, "study_date"] = stamp.date().isoformat()
    first = (d.loc[pet.index].sort_values(["participant", "study_date", "case"])
             .drop_duplicates("participant").index)
    d.loc[pet.index, "earliest_examination"] = 0
    d.loc[first, "earliest_examination"] = 1
    return d


def add_missing_label_bounds(data, flow):
    """Extreme classifications concern only empty-label examinations in the processed frame."""
    f = flow.copy()
    for r in f.itertuples():
        cases = data[data.cohort == r.cohort]
        if len(cases) != r.measurable or r.processed != r.measurable + r.empty:
            raise ValueError("missing-label bounds do not reconcile with the processed frame")
        events = int((cases.class26 == "disconnected").sum())
        values = {"no_path_lower_n": events, "no_path_upper_n": events + r.empty,
                  "no_path_bound_n": r.processed,
                  "no_path_lower_prop": events / r.processed,
                  "no_path_upper_prop": (events + r.empty) / r.processed}
        for name, value in values.items():
            f.loc[r.Index, name] = value
    return f


def add_followup_intervals(data, detail, summaries_):
    """Describe consecutive released-date intervals, distinct from all-pairs agreement."""
    detail, summaries_ = detail.copy(), summaries_.copy()
    all_intervals = {}
    for participant, g in data[data.cohort == "autoPET"].groupby("participant", sort=True):
        if len(g) < 2:
            continue
        dates = pd.to_datetime(g.study_date).sort_values()
        intervals = np.diff(dates.to_numpy()).astype("timedelta64[D]").astype(int)
        if (intervals < 0).any():
            raise ValueError("negative consecutive examination interval")
        all_intervals[participant] = intervals
        selected = detail.participant == participant
        detail.loc[selected, "consecutive_intervals_days"] = json.dumps(intervals.tolist())
        detail.loc[selected, "consecutive_interval_n"] = len(intervals)
        detail.loc[selected, "consecutive_interval_median_days"] = float(np.median(intervals))
    for r in summaries_.itertuples():
        intervals = np.concatenate(list(all_intervals.values())) if r.cohort == "autoPET" and all_intervals else np.array([])
        summaries_.loc[r.Index, "consecutive_interval_n"] = len(intervals)
        quantiles = np.quantile(intervals, [0, .25, .5, .75, 1]) if len(intervals) else [np.nan] * 5
        for name, value in zip(("min", "q25", "median", "q75", "max"), quantiles):
            summaries_.loc[r.Index, f"consecutive_interval_{name}_days"] = value
    return detail, summaries_


def reference_summaries(reference):
    """Case-level Wilson intervals explicitly assume independent released cases."""
    measurable = reference[reference.status == "measurable"]
    frames = {"reference_all": measurable,
              **{f"reference_{split}": frame for split, frame in measurable.groupby("split")}}
    frames.update({f"{name}_no_lciv_image_boundary_contact": frame[frame.lciv_touches_image_boundary == 0]
                   for name, frame in list(frames.items())})
    rows = []
    for name, frame in frames.items():
        for connectivity in (26, 6):
            for category in CLASSES:
                events = int((frame[f"class{connectivity}"] == category).sum())
                lo, hi = S.wilson_ci(events, len(frame), CFG.ALPHA) if len(frame) else (np.nan, np.nan)
                rows.append({"frame": "reference", "stratum": name,
                             "connectivity": connectivity, "class": category,
                             "n": len(frame), "events": events,
                             "proportion": events / len(frame) if len(frame) else np.nan,
                             "ci_lo": lo, "ci_hi": hi, "participants": np.nan,
                             "ci_method": "Wilson_independent_cases_assumed"})
    return pd.DataFrame(rows)


def refresh_public_reference(outdir=OUT, cache=None, workers=4):
    """Audit all released reference-mask triplets; preserve them as a separate frame."""
    cache = Path(cache or reference_cache_path())
    outdir = Path(outdir)
    sources, meta, record, directory_hash = EM.fetch_public_reference_masks(cache)
    sources.to_csv(outdir / "reference_source_files.csv", index=False)
    cases = sorted({name.split("/")[0] for name in sources.name})
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(MC.audit_reference_case, [(cache, c) for c in cases]), 1):
            rows.append(result)
            if i % 50 == 0 or i == len(cases):
                print(f"Public-reference topology: {i}/{len(cases)}", flush=True)
    data = pd.DataFrame(rows)
    meta = meta.rename(columns={"image_id": "case"})
    if meta.case.duplicated().any() or set(meta.case) != set(data.case):
        raise ValueError("reference metadata must match the archive one-to-one")
    data = data.merge(meta, on="case", validate="one_to_one")
    data.to_csv(outdir / "reference_cases.csv", index=False)
    provenance = {"record": EM.PUBLIC_REFERENCE_API,
                  "archive": EM.PUBLIC_REFERENCE_URL,
                  "archive_checksum_verified": False,
                  "partial_entry_integrity": "ZIP CRC32 and SHA-256 of each extracted mask",
                  "central_directory_sha256": directory_hash,
                  "released_case_count": len(data),
                  "mask_count": len(sources),
                  "dataset_version": record["metadata"].get("version", "2.0.1")}
    # The full statistics run (main) rewrites this manifest with every input and output.
    (outdir / "supplement_manifest.json").write_text(json.dumps({"reference_source": provenance}, indent=2) + "\n")
    print(data.groupby(["status", "split"], dropna=False).size().to_string(), flush=True)

def proportion_intervals(frame, selector, n_boot=N_BOOT, independent_wilson=True, seed=SEED):
    """Wilson for independent single-cohort cases; otherwise participant bootstrap.

    Cohort composition in participants is held fixed. Each sampled participant
    contributes all their scans, preserving within-participant dependence. AMOS
    has case proxies because cross-case patient linkage is unavailable.
    """
    d = frame.assign(value=np.asarray(selector, float))
    if independent_wilson and len(d) and d.cohort.nunique() == 1 and d.participant.nunique() == len(d):
        lo, hi = S.wilson_ci(int(d.value.sum()), len(d), CFG.ALPHA)
        return {"n": len(d), "events": int(d.value.sum()),
                "proportion": d.value.mean(), "ci_lo": lo, "ci_hi": hi,
                "participants": len(d), "ci_method": "Wilson_independent_case_proxies"}
    if d.participant.nunique() < 2:
        return {"n": len(d), "events": int(d.value.sum()),
                "proportion": d.value.mean(), "ci_lo": np.nan, "ci_hi": np.nan,
                "participants": d.participant.nunique(), "ci_method": "not_estimable"}
    blocks = [
        g.groupby("participant")
        .agg(events=("value", "sum"), n=("value", "size"))
        .to_numpy()
        for _, g in d.groupby("cohort", sort=True)
    ]
    rng = np.random.default_rng(seed)
    events = np.zeros(n_boot)
    sizes = np.zeros(n_boot)
    for b in blocks:
        # Limit temporary allocation for the largest stratum.
        for start in range(0, n_boot, 250):
            stop = min(start + 250, n_boot)
            idx = rng.integers(0, len(b), (stop - start, len(b)))
            sampled = b[idx].sum(axis=1)
            events[start:stop] += sampled[:, 0]
            sizes[start:stop] += sampled[:, 1]
    lo, hi = np.quantile(events / sizes, [0.025, 0.975])
    return {
        "n": len(d),
        "events": int(d.value.sum()),
        "proportion": d.value.mean(),
        "ci_lo": lo,
        "ci_hi": hi,
        "participants": d.participant.nunique(),
        "ci_method": "cohort_stratified_participant_percentile",
    }


def model_grid_summaries(cases, n_boot=10000, seed=20261025):
    """Internal minus native paired processed-frame bounds, with all visits retained.

    This analysis writes only its own model-grid tables and does not modify the primary
    connectivity tables. Native and internal unknown outcomes are bounded jointly (ilo-nhi,
    ihi-nlo).
    """
    if n_boot < 2 or not isinstance(seed, int):
        raise ValueError("at least two bootstrap draws and an integer seed are required")
    states = (*CLASSES, "nonmeasurable")
    if cases.empty or cases.duplicated(["cohort", "case"]).any():
        raise ValueError("paired model-grid frame is empty or duplicated")
    for conn in [26, 6]:
        for stage in ["native", "internal"]:
            if not cases[f"{stage}_class{conn}"].isin(states).all():
                raise ValueError("foreign or missing model-grid topology class")
    strata = {"Pooled": cases, **{c: d for c, d in cases.groupby("cohort", sort=True)},
              "spacing_le_1p5mm": cases[cases.native_spacing_mm <= 1.5],
              "spacing_gt1p5_to3mm": cases[(cases.native_spacing_mm > 1.5) & (cases.native_spacing_mm <= 3)],
              "spacing_gt3mm": cases[cases.native_spacing_mm > 3]}
    summary, paired, transitions = [], [], []
    for stratum, frame in strata.items():
        if frame.empty:
            continue
        for conn in [26, 6]:
            bounds = {}
            for stage in ["native", "internal"]:
                value = frame[f"{stage}_class{conn}"]
                lo = value.eq("disconnected").astype(int)
                hi = value.isin(["disconnected", "nonmeasurable"]).astype(int)
                bounds[stage] = (lo, hi)
                for outcome, selector in [(c, value.eq(c)) for c in states] + [("disconnected_upper_bound", hi)]:
                    summary.append({"stratum": stratum, "connectivity": conn, "stage": stage,
                                    "outcome": outcome, **proportion_intervals(frame, selector, n_boot, False, seed)})
            for name, selector in [("lower_bound", bounds["internal"][0] - bounds["native"][1]),
                                   ("upper_bound", bounds["internal"][1] - bounds["native"][0])]:
                stats = proportion_intervals(frame, selector, n_boot, False, seed)
                stats["mean_change"] = stats.pop("proportion")
                stats["signed_sum"] = stats.pop("events")
                paired.append({"stratum": stratum, "connectivity": conn, "outcome": name, **stats})
            for native in states:
                part = frame[frame[f"native_class{conn}"] == native]
                for internal in states:
                    events = int(part[f"internal_class{conn}"].eq(internal).sum())
                    transitions.append({"stratum": stratum, "connectivity": conn, "native": native,
                                        "internal": internal, "events": events, "processed_n": len(frame),
                                        "native_class_n": len(part), "conditional_proportion": events / len(part) if len(part) else np.nan})
    return {"model_grid_prevalence.csv": pd.DataFrame(summary),
            "model_grid_paired.csv": pd.DataFrame(paired),
            "model_grid_transitions.csv": pd.DataFrame(transitions)}


def longitudinal_agreement(frame, categories, mapping):
    """Exchangeable visit-pair agreement with equal participant weights."""
    joints, detail = [], []
    for participant, g in frame.groupby("participant", sort=True):
        if len(g) < 2:
            continue
        values = [mapping[x] for x in g.class26]
        pairs = list(combinations(values, 2))
        joint = np.zeros((categories, categories))
        for x, y in pairs:
            joint[x, y] += 0.5 / len(pairs)
            joint[y, x] += 0.5 / len(pairs)
        joints.append(joint)
        counts = np.bincount(values, minlength=categories)
        # Independent count-formula oracle, with no enumeration of visit pairs.
        expected = (counts * (counts - 1)).sum() / (len(values) * (len(values) - 1))
        if not np.isclose(np.trace(joint), expected):
            raise AssertionError("pair enumeration disagrees with count formula")
        detail.append(
            {
                "participant": participant,
                "scans": len(values),
                "pairs": len(pairs),
                "all_visits_agree": int(len(set(values)) == 1),
                "pair_agreement": float(expected),
            }
        )
    joints = np.asarray(joints)

    def evaluate(joint):
        po = np.trace(joint, axis1=-2, axis2=-1)
        marginal = joint.sum(axis=-1)
        pe = (marginal**2).sum(axis=-1)
        kappa = (po - pe) / (1 - pe)
        ac1_chance = (marginal * (1 - marginal)).sum(axis=-1) / (categories - 1)
        ac1 = (po - ac1_chance) / (1 - ac1_chance)
        metrics = {"agreement": po, "nominal_kappa": kappa, "nominal_ac1": ac1}
        if categories == 2:
            for c, name in enumerate(("connected", "disconnected")):
                metrics[f"specific_agreement_{name}"] = joint[..., c, c] / marginal[..., c]
        return metrics

    j = joints.mean(axis=0)
    point = evaluate(j)
    rng = np.random.default_rng(CFG.SEED)
    draws = rng.integers(0, len(joints), (N_BOOT, len(joints)))
    estimates = evaluate(joints[draws].mean(axis=1))
    rows = []
    for metric, value in point.items():
        boot = estimates[metric]
        if not np.isfinite(boot).all():
            raise ValueError("undefined bootstrap replicate; no silent deletion")
        lo, hi = np.quantile(boot, [0.025, 0.975])
        rows.append(
            {
                "metric": metric,
                "estimate": float(value),
                "ci_lo": lo,
                "ci_hi": hi,
                "participants": len(joints),
                "scans": sum(x["scans"] for x in detail),
                "pairs": sum(x["pairs"] for x in detail),
                "all_visits_agree_participants": sum(
                    x["all_visits_agree"] for x in detail
                ),
                "marginals": json.dumps(j.sum(axis=1).tolist()),
                "joint_distribution": json.dumps(j.tolist()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(detail)


def publication_topology(data, outdir, remeasure=False, workers=4):
    """Join current dominant-component measurements to the canonical analysis frame."""
    mask_root = ST.REPO_ROOT / "Local masks"
    manifest = pd.read_csv(mask_root / "MANIFEST.csv").set_index(
        ["dataset", "case", "structure"], verify_integrity=True)
    crosswalk = pd.read_csv(ST.PHASE0 / "autopet_ts217_crosswalk.csv").set_index(
        "phase1_case", verify_integrity=True)
    if crosswalk.baseline_case.duplicated().any():
        raise ValueError("non-unique baseline mask crosswalk")
    cache_path = Path(outdir) / "analysis_cases.csv"
    if not remeasure:
        cached = pd.read_csv(cache_path)
        required = {"class26", "class6", "main_path26", "main_path6", "any_path26", "mask_case"}
        if not required <= set(cached):
            raise ValueError("No dominant-component cache; run --measure-topology")
        columns = [c for c in cached if c not in data or c in ("cohort", "case")]
        measured = cached[columns]
    else:
        tasks = []
        for r in data.itertuples():
            mask_case = crosswalk.loc[r.case, "baseline_case"] if r.cohort == "autoPET" else r.case
            expected = [(name, manifest.loc[(r.cohort, mask_case, name), "sha256"],
                         int(manifest.loc[(r.cohort, mask_case, name), "voxels"]))
                        for name in MC.REFERENCE_VEIN_FILES]
            tasks.append((r.cohort, r.case, mask_case, str(mask_root), expected))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            measured = pd.DataFrame(pool.map(MC.read_topology_case, tasks))
    result = data.merge(measured, on=["cohort", "case"], validate="one_to_one")
    if len(result) != len(data) or not np.array_equal(result.any_path26, result.union_connected):
        raise ValueError("dominant topology does not reconcile with the original path endpoint")
    if (result.main_path6 > result.main_path26).any():
        raise ValueError("6-connectivity path exists without a 26-connectivity path")
    for r in result.itertuples():
        wanted_case = crosswalk.loc[r.case, "baseline_case"] if r.cohort == "autoPET" else r.case
        if r.mask_case != wanted_case:
            raise ValueError("mask case identity disagrees with the input-path crosswalk")
        for name in MC.REFERENCE_VEIN_FILES:
            wanted = manifest.loc[(r.cohort, r.mask_case, name), "sha256"]
            path = mask_root / r.cohort / r.mask_case / f"{name}.nii.gz"
            if getattr(r, name + "_sha256") != wanted or MC.topology_digest(path) != wanted:
                raise ValueError(f"cached topology has changed mask bytes: {path}")
    return result


def topology_summaries(data):
    pet = data[data.cohort == "autoPET"]
    strata = {"all": data, **{c: d for c, d in data.groupby("cohort")},
              "thin_le_1p5mm": data[data.slice_mm <= 1.5],
              "thin_contrast_autoPET": pet[(pet.slice_mm <= 1.5) & (pet.contrast == "with")],
              "autoPET_with_description": pet[pet.contrast == "with"],
              "autoPET_without_description": pet[pet.contrast == "without"],
              "no_lciv_image_boundary_contact": data[data.lciv_touches_image_boundary == 0]}
    if "earliest_examination" in data:
        strata.update({"earliest_autoPET_per_participant": data[data.earliest_examination == 1],
                       "autoPET_earliest_per_participant": pet[pet.earliest_examination == 1]})
    # Same sensitivity definitions within each source cohort, rather than only pooled.
    for cohort, frame in data.groupby("cohort", sort=True):
        strata[f"{cohort}__thin_le_1p5mm"] = frame[frame.slice_mm <= THIN_MM]
        strata[f"{cohort}__spacing_gt1p5mm"] = frame[frame.slice_mm > THIN_MM]
        strata[f"{cohort}__no_lciv_image_boundary_contact"] = frame[frame.lciv_touches_image_boundary == 0]
        if "earliest_examination" in data:
            strata[f"{cohort}__earliest_examination"] = frame[frame.earliest_examination == 1]
    rows = []
    for name, frame in strata.items():
        for connectivity in (26, 6):
            for category in CLASSES:
                rows.append({"frame": "prediction", "ci_method": "cohort_stratified_participant_percentile",
                             "stratum": name, "connectivity": connectivity, "class": category,
                             **proportion_intervals(frame, frame[f"class{connectivity}"] == category)})
    for name, frame in [("all", data), *data.groupby("cohort", sort=True)]:
        # Closed-box contact is equivalent to some 26-adjacent cross-label pair.
        any_class = np.where(frame.gap_mm == 0, "direct",
                             np.where(frame.any_path26 == 1, "right_iliac_mediated", "disconnected"))
        for category in CLASSES:
            rows.append({"frame": "prediction", "ci_method": "cohort_stratified_participant_percentile",
                         "stratum": f"{name}__any_endpoint26", "connectivity": 26, "class": category,
                         **proportion_intervals(frame, any_class == category)})
    return pd.DataFrame(rows)


def verify_population_average_models(data, coefficients):
    """Independent-working GEE verifies the estimand without requiring random effects."""
    _load_regression_dependencies()
    base = data.copy()
    base["present"] = (base.gap_mm > CFG.PRESENT_MM).astype(int)
    positive = base[base.present == 1].copy()
    positive["log_gap_given_present"] = np.log(positive.gap_mm)
    for outcome, frame, family in (("present", base, sm.families.Binomial()),
                                    ("log_gap_given_present", positive, sm.families.Gaussian())):
        fit = smf.gee(f"{outcome} ~ slice_mm + C(cohort)", groups="participant", data=frame,
                      family=family, cov_struct=sm.cov_struct.Independence()).fit()
        original = coefficients[(coefficients.model == "pooled") & (coefficients.endpoint == outcome)].set_index("term")
        if not np.allclose(fit.params, original.loc[fit.params.index, "estimate"], atol=1e-9):
            raise ValueError("population-average coefficients disagree with GEE")
    return True


if __name__ == "__main__":
    main()
