#!/usr/bin/env python3
"""The summary tables: numbers computed from the released per-examination outputs only.

    python Code/phase1_characterize/summary/summary_tables.py

writes one CSV per analysis, RESULTS_SUMMARY.md and run_record.json to
Results/phase1_characterize/summary/. No image is read and no inference is rerun.
Each analysis first reproduces figures already reported from the same inputs and raises if
it cannot, so a changed input or a misread column fails instead of yielding a plausible number.

The analyses: the measurable-frame paired change between grids with a clustered McNemar test;
participant-block percentile intervals for the reported medians; corridor volume as a fraction of
the venous labels; the bottleneck at a fixed 1.5 mm; non-vascular labels in the corridor; SAROS
disconnection by region; autoPET disconnection by contrast description; age by sex; agreement
between repeat examinations; the bound that submodel merging places on the corridors, from the
order in which TotalSegmentator writes its submodels; the released annotations' connectivity under
six-connectivity, with their separation in slice units; the paired change where projection can act
(native spacing above 1.5 mm) and under six-connectivity; inference-grid disconnection by cohort;
the cohort-and-spacing model fitted to each grid and adjacency rule; sensitivity to participants
under 18 and to the autoPET diagnosis recorded for each study; the overlap of the SAROS source
collections with autoPET; the comparators (the IVC to the right iliac vein and the aorta to each
common iliac artery, beside the left iliac endpoint, on both grids and under both adjacency rules),
read from the table and run record of confluence_comparators.py; whether the connected examination of
a repeat pair that disagrees needed an edge or corner contact; and which labels the nonmeasurable
examinations lacked.

Every input is parsed from the bytes that were hashed, the two metadata files that stats.load_meta
reads itself are hashed on either side of that read, and every output is hashed as the in-memory
payload published, written to a `.part` and renamed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
import pathlib as _pl
import platform
from collections import Counter
from itertools import combinations

# --- run-from-anywhere: put sibling phase dirs on sys.path (tests use Code/conftest.py) ---
import sys as _sys

for _p in sorted(_pl.Path(__file__).resolve().parents[2].iterdir()):
    if _p.is_dir() and _p.name[0] not in "._" and str(_p) not in _sys.path:
        _sys.path.append(str(_p))  # append, never shadow stdlib

import connectivity_statistics as FS
import gap_config as CFG
import gap_content_analysis as GC
import numpy as np
import pandas as pd
import statlib as S
import stats as ST
import tableio as TIO
from scipy import stats as sps

ROOT = _pl.Path(__file__).resolve().parents[3]
RESULTS = ROOT / "Results" / "phase1_characterize"
OUT = RESULTS / "summary"
INPUTS = {
    "analysis_cases": RESULTS / "connectivity" / "analysis_cases.csv",
    "participant_mapping": RESULTS / "connectivity" / "participant_mapping.csv",
    "longitudinal_agreement": RESULTS / "connectivity" / "longitudinal_agreement.csv",
    "reference_cases": RESULTS / "connectivity" / "reference_cases.csv",
    "paired_connectivity": RESULTS / "model_grid" / "paired_connectivity.csv",
    "reproduction_fidelity": RESULTS / "model_grid" / "reproduction_fidelity.csv",
    "gap_native": RESULTS / "gap_content_native" / "gap_content_cases.csv",
    "gap_inference": RESULTS / "gap_content_internal" / "gap_content_cases.csv",
    "cohort_table1": RESULTS / "cohort_table1.csv",
    "saros_metadata": ROOT / "METADATA" / "SAROS.csv",
    "amos_metadata": ROOT / "METADATA" / "AMOS.csv",
    "autopet_metadata": ROOT / "METADATA" / "autoPET.csv",
    "autopet_crosswalk": ROOT / "Code" / "phase0_data" / "autopet_ts217_crosswalk.csv",
    "model_coefficients": RESULTS / "connectivity" / "model_coefficients.csv",
    "comparators": RESULTS / "comparators" / "comparators_cases.csv",
    "comparators_record": RESULTS / "comparators" / "run_record.json",
    "attenuation_summary": RESULTS / "comparators" / "attenuation_summary.csv",
    "mask_topology": RESULTS / "summary" / "mask_topology_cases.csv",
    "reference_topology": RESULTS / "summary" / "reference_topology_cases.csv",
    "mask_topology_record": RESULTS / "summary" / "mask_topology_run_record.json",
}
METADATA_INPUT = {"amos": "amos_metadata", "autopet": "autopet_metadata"}
GRIDS = ("native", "inference")
# The resample count and seeds of the producers whose figures these analyses extend, taken from
# them rather than restated: the topology intervals (Tables 1 and 2) and the model-grid intervals.
N_BOOT = FS.N_BOOT
SEED_TOPOLOGY = CFG.SEED
SEED_GRID = inspect.signature(FS.model_grid_summaries).parameters["seed"].default
# Every class of TotalSegmentator 2.17.0 task 293, the cardiac submodel that holds all vessels
# (map_to_binary.class_map_parts["class_map_part_cardiac"] at tag v2.17.0).
VASCULAR = frozenset({
    "heart", "aorta", "pulmonary_vein", "brachiocephalic_trunk", "subclavian_artery_right",
    "subclavian_artery_left", "common_carotid_artery_right", "common_carotid_artery_left",
    "brachiocephalic_vein_left", "brachiocephalic_vein_right", "atrial_appendage_left",
    "superior_vena_cava", "inferior_vena_cava", "portal_vein_and_splenic_vein",
    "iliac_artery_left", "iliac_artery_right", "iliac_vena_left", "iliac_vena_right"})
# The total task's five submodels in the order TotalSegmentator 2.17.0 runs them, each writing its
# nonzero labels over those already written (map_tasks_config.py lines 38-41; nnunet.py lines
# 409-411, 605 and 634-635; class lists from map_to_binary.class_map_5_parts, line 874). A label
# written before the cardiac submodel cannot remain where that submodel predicted a vein.
SUBMODELS = (
    ("organs", frozenset({
        "spleen", "kidney_right", "kidney_left", "gallbladder", "liver", "stomach", "pancreas",
        "adrenal_gland_right", "adrenal_gland_left", "lung_upper_lobe_left", "lung_lower_lobe_left",
        "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right", "esophagus", "trachea",
        "thyroid_gland", "small_bowel", "duodenum", "colon", "urinary_bladder", "prostate",
        "kidney_cyst_left", "kidney_cyst_right"})),
    ("vertebrae", frozenset({"sacrum", *(f"vertebrae_{v}" for v in (
        "S1", "L5", "L4", "L3", "L2", "L1", "T12", "T11", "T10", "T9", "T8", "T7", "T6", "T5", "T4", "T3",
        "T2", "T1", "C7", "C6", "C5", "C4", "C3", "C2", "C1"))})),
    ("cardiac", VASCULAR),
    ("muscles", frozenset({
        "humerus_left", "humerus_right", "scapula_left", "scapula_right", "clavicula_left", "clavicula_right",
        "femur_left", "femur_right", "hip_left", "hip_right", "spinal_cord", "gluteus_maximus_left",
        "gluteus_maximus_right", "gluteus_medius_left", "gluteus_medius_right", "gluteus_minimus_left",
        "gluteus_minimus_right", "autochthon_left", "autochthon_right", "iliopsoas_left", "iliopsoas_right",
        "brain", "skull"})),
    ("ribs", frozenset({*(f"rib_{side}_{i}" for side in ("left", "right") for i in range(1, 13)),
                        "sternum", "costal_cartilages"})),
)
_ORDER = tuple(name for name, _ in SUBMODELS)
WRITTEN_AFTER_CARDIAC = frozenset(_ORDER[_ORDER.index("cardiac") + 1:])
# Abdominal organs whose presence in a corridor places it well above the confluence (a
# descriptive count, not a reported analysis).
VISCERA = frozenset({
    "liver", "spleen", "stomach", "pancreas", "gallbladder", "duodenum", "small_bowel", "colon",
    "esophagus", "kidney_left", "kidney_right", "adrenal_gland_left", "adrenal_gland_right"})
# The arterial labels have their own corridor voxel counts in the gap-content tables.
ARTERIAL_COLUMNS = {"aorta": "corridor_aorta_voxels", "iliac_artery_left": "corridor_iliac_artery_left_voxels",
                    "iliac_artery_right": "corridor_iliac_artery_right_voxels"}
# Released study descriptions (study_type) naming the head, neck, chest or heart, and those naming the
# abdomen, pelvis, legs or whole body; a descriptive rule for the released cases that lack a label, with
# the descriptions it cannot place kept as their own count.
UPPER_BODY_STUDY = r"head|neck|thorax|chest|heart|coron|aort|orbita|gesicht|artrium"
LOWER_BODY_STUDY = r"abd|pelvi|leg|polytrauma|whole|body|lumb|becken|bauch|urolog|kidney|niere|colon|liver|leber"
# The comparator table's three pairs, each tested within its own two labels.
COMPARATORS = ("ivc_rciv", "aorta_iliac_artery_left", "aorta_iliac_artery_right")
_DIGESTS = {}   # input name -> SHA-256 of the bytes this run parsed


def _check(condition, message):
    if not condition:
        raise ValueError(f"control failed: {message}")


def _record(name, digest):
    if _DIGESTS.setdefault(name, digest) != digest:
        raise ValueError(f"{name} changed during the run")


def _read(name):
    """Parse an input from the very bytes that are hashed for the run record."""
    data = INPUTS[name].read_bytes()
    _record(name, hashlib.sha256(data).hexdigest())
    return pd.read_csv(io.BytesIO(data), keep_default_na=False)


def _read_json(name):
    """Parse a JSON input from the very bytes that are hashed for the run record."""
    data = INPUTS[name].read_bytes()
    _record(name, hashlib.sha256(data).hexdigest())
    return json.loads(data)


def _load_meta(dataset):
    """stats.load_meta reads its file itself, so the file is hashed on either side of that read."""
    name = METADATA_INPUT[dataset]
    _check(INPUTS[name] == ST.META_DIR / ST.LAYOUT[dataset][2], f"{dataset} metadata path")
    before = hashlib.sha256(INPUTS[name].read_bytes()).hexdigest()
    meta = ST.load_meta(dataset)
    _check(hashlib.sha256(INPUTS[name].read_bytes()).hexdigest() == before, f"{name} unchanged while read")
    _record(name, before)
    return meta


def _publish(path, data):
    """Write bytes to a `.part`, make them durable and rename them into place; return their hash."""
    part = path.with_name(path.name + ".part")
    with open(part, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(part, path)
    TIO.fsync_directory(path.parent)
    return hashlib.sha256(data).hexdigest()


def _numeric(frame, columns):
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column].replace("", np.nan), errors="raise")
    return out


def _proportion(frame, selector, seed, n_boot, wilson=True):
    """Percentages with the project's intervals (Wilson for an independent single-cohort stratum,
    otherwise cohort-stratified participant-block percentile)."""
    s = FS.proportion_intervals(frame, np.asarray(selector, bool), n_boot, wilson, seed)
    return {"n": int(s["n"]), "events": int(s["events"]), "estimate": 100 * s["proportion"],
            "ci_lo": 100 * s["ci_lo"], "ci_hi": 100 * s["ci_hi"],
            "participants": int(s["participants"]), "ci_method": s["ci_method"]}


def _gap(grid):
    expected = {"native": {"disconnected": 892, "right_iliac_mediated": 597, "direct": 384},
                "inference": {"disconnected": 1095, "right_iliac_mediated": 527, "direct": 251}}[grid]
    frame = _read(f"gap_{grid}")
    frame = _numeric(frame, ["spacing_mm_0", "spacing_mm_1", "spacing_mm_2", "ivc_voxels", "lciv_voxels",
                             "rciv_voxels", "corridor_voxels", "corridor_volume_mm3",
                             "corridor_other_label_voxels", "bottleneck_radius_mm", "arterial_in_corridor",
                             *ARTERIAL_COLUMNS.values(),
                             "corridor_centroid_z_minus_rciv_cranial_mm"])
    _check(len(frame) == 1873 and frame.class26.value_counts().to_dict() == expected,
           f"{grid} gap-content classes {expected}")
    return frame


def weighted_median(sorted_values, weights):
    """np.median of the sample that repeats sorted_values[i] weights[i] times (weights >= 0)."""
    cumulative = np.cumsum(weights)
    total = int(round(cumulative[-1])) if len(cumulative) else 0
    if total == 0:
        return float("nan")
    if total % 2:
        return float(sorted_values[np.searchsorted(cumulative, (total + 1) // 2)])
    low = sorted_values[np.searchsorted(cumulative, total // 2)]
    high = sorted_values[np.searchsorted(cumulative, total // 2 + 1)]
    return float((low + high) / 2)


def median_interval(frame, column, seed, n_boot=N_BOOT):
    """Point median and participant-block percentile 95% interval.

    Participants are resampled with replacement within each cohort, cohort sizes held fixed,
    each bringing all of their examinations: the resampling unit and stratification of the
    prevalence intervals, with percentile endpoints in place of BCa.
    """
    frame = frame.reset_index(drop=True)
    values = frame[column].to_numpy(float)
    _check(len(values) > 1 and np.isfinite(values).all(), f"{column}: finite values required")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    strata = []
    for _, group in frame.groupby("cohort", sort=True):
        codes, uniques = pd.factorize(group["participant"], sort=True)
        strata.append((group.index.to_numpy(), codes, len(uniques)))
    rng = np.random.default_rng(seed)
    weights = np.zeros(len(values))
    medians = np.empty(n_boot)
    for b in range(n_boot):
        for rows, codes, count in strata:
            weights[rows] = np.bincount(rng.integers(0, count, count), minlength=count)[codes]
        medians[b] = weighted_median(sorted_values, weights[order])
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(np.median(values)), float(low), float(high)


def durkalski_test(participants, before, after):
    """Clustered McNemar test of Durkalski et al. (Stat Med 2003;22:2417-2428).

    X^2 = (sum_k d_k)^2 / sum_k d_k^2 with d_k = (b_k - c_k) / n_k for participant k, where b_k
    counts pairs positive only after, c_k pairs positive only before and n_k all pairs; referred
    to chi-square on 1 df. The form is that of CRAN clust.bin.pair (R/durkalski.R), whose tests
    reproduce the paper's published statistics; with one pair per participant it is McNemar's
    (b - c)^2 / (b + c) without continuity correction.
    """
    before = np.asarray(before, bool)
    after = np.asarray(after, bool)
    pairs = pd.DataFrame({"participant": np.asarray(participants), "b": after & ~before,
                          "c": before & ~after})
    clusters = pairs.groupby("participant", sort=True).agg(b=("b", "sum"), c=("c", "sum"),
                                                           n=("b", "size"))
    d = (clusters.b - clusters.c) / clusters.n
    denominator = float((d ** 2).sum())
    statistic = float(d.sum() ** 2 / denominator) if denominator else float("nan")
    return {"statistic": statistic, "p_value": float(sps.chi2.sf(statistic, 1)),
            "clusters": int(len(clusters)), "b": int(clusters.b.sum()), "c": int(clusters.c.sum())}


def participant_contingency(frame, before, after):
    """Per-participant 2 x 2 counts (ak, bk, ck, dk) in clust.bin.pair's layout, t1 = after."""
    t1 = np.asarray(after, bool)
    t2 = np.asarray(before, bool)
    cells = pd.DataFrame({"participant": frame["participant"].to_numpy(), "ak": t1 & t2,
                          "bk": t1 & ~t2, "ck": ~t1 & t2, "dk": ~t1 & ~t2})
    return cells.groupby("participant", sort=True)[["ak", "bk", "ck", "dk"]].sum().reset_index()


def paired_projection(n_boot=N_BOOT):
    """Inference-grid minus archived native-grid disconnection in the same examinations."""
    frame = _numeric(_read("paired_connectivity"), ["native_spacing_mm"])
    _check(len(frame) == 1899, "1,899 processed examinations")
    measurable = frame[frame.archive_class26 != "nonmeasurable"].reset_index(drop=True)
    _check(len(measurable) == 1873 and measurable.internal_class26.ne("nonmeasurable").all(),
           "1,873 examinations measurable on both grids")
    native = measurable.archive_class26.eq("disconnected").to_numpy()
    inference = measurable.internal_class26.eq("disconnected").to_numpy()
    closed = int((inference & ~native).sum())
    opened = int((native & ~inference).sum())
    _check((int(native.sum()), int(inference.sum()), closed, opened) == (892, 1095, 205, 2),
           "Figure 5c: 892 and 1,095 disconnected, 205 closed, 2 opened")
    native_hi = frame.archive_class26.isin(["disconnected", "nonmeasurable"]).to_numpy()
    inference_hi = frame.internal_class26.isin(["disconnected", "nonmeasurable"]).to_numpy()
    native_lo = frame.archive_class26.eq("disconnected").to_numpy()
    inference_lo = frame.internal_class26.eq("disconnected").to_numpy()
    published_bounds = (round(100 * (inference_lo.sum() - native_hi.sum()) / 1899, 1),
                        round(100 * (inference_hi.sum() - native_lo.sum()) / 1899, 1))
    _check(published_bounds == (9.3, 12.1), "processed-frame paired bounds +9.3 to +12.1")
    rows = []

    def change(label, part, before, after):
        s = FS.proportion_intervals(part, after.astype(int) - before.astype(int), n_boot, False, SEED_GRID)
        rows.append({"row": f"paired_change:{label}", "n": int(s["n"]),
                     "net_examinations": int(after.sum() - before.sum()),
                     "estimate": 100 * s["proportion"], "ci_lo": 100 * s["ci_lo"], "ci_hi": 100 * s["ci_hi"],
                     "participants": int(s["participants"]), "ci_method": s["ci_method"]})

    change("measurable", measurable, native, inference)
    processed = frame.reset_index(drop=True)
    change("processed_empty_counted_disconnected", processed, native_hi, inference_hi)
    for grid, selector in (("native", native_hi), ("inference", inference_hi)):
        rows.append({"row": f"prevalence:{grid}:processed_empty_counted_disconnected",
                     **_proportion(processed, selector, SEED_GRID, n_boot, wilson=False)})
    spacing = measurable.native_spacing_mm
    for label, keep in (("spacing_le_1p5mm", spacing <= 1.5), ("spacing_gt1p5_to3mm", (spacing > 1.5) & (spacing <= 3)),
                        ("spacing_gt3mm", spacing > 3)):
        keep = keep.to_numpy()
        change(label, measurable[keep], native[keep], inference[keep])
    for cohort, part in measurable.groupby("cohort", sort=True):
        index = part.index.to_numpy()
        rows.append({"row": f"discordant:{cohort}", "n": len(part),
                     "closed_by_projection": int((inference[index] & ~native[index]).sum()),
                     "opened_by_projection": int((native[index] & ~inference[index]).sum())})
    rows.append({"row": "discordant:pooled", "n": len(measurable), "closed_by_projection": closed,
                 "opened_by_projection": opened})
    test = durkalski_test(measurable.participant, native, inference)
    _check((test["b"], test["c"]) == (205, 2), "Durkalski discordant totals")
    rows.append({"row": "clustered_mcnemar:durkalski_2003", "n": len(measurable),
                 "statistic": test["statistic"], "p_value": test["p_value"], "participants": test["clusters"]})
    exact = S.mcnemar_test(closed, opened, exact=True)
    rows.append({"row": "mcnemar:exact_unclustered", "n": len(measurable), "statistic": exact["chi2"],
                 "p_value": exact["p"]})
    contingency = participant_contingency(measurable, native, inference)
    return pd.DataFrame(rows), contingency


def median_intervals(n_boot=N_BOOT):
    """Table 1 separation medians and Table S3 bottleneck medians with percentile intervals."""
    _check(inspect.signature(GC.summaries).parameters["seed"].default == SEED_GRID,
           "the gap-content and model-grid producers share one seed")
    rows = []
    cases = _numeric(_read("analysis_cases"), ["gap_mm"])
    published = {"AMOS": 5.24, "SAROS": 3.50, "autoPET": 2.63}
    for cohort, group in cases.groupby("cohort", sort=True):
        median, low, high = median_interval(group, "gap_mm", SEED_TOPOLOGY, n_boot)
        _check(abs(median - published[cohort]) < 0.0051, f"Table 1 median separation {cohort}")
        rows.append({"table": "Table 1", "quantity": "median separation, mm", "stratum": cohort,
                     "n": len(group), "participants": group.participant.nunique(), "median": median,
                     "ci_lo": low, "ci_hi": high, "ci_method": "participant_block_percentile"})
    published = {"native": 1.61, "inference": 1.50}
    for grid in GRIDS:
        gap = _gap(grid)
        connected = gap[gap.class26 != "disconnected"]
        _check(connected.bottleneck_radius_mm.notna().all(), f"{grid} bottleneck defined for connected")
        median, low, high = median_interval(connected, "bottleneck_radius_mm", SEED_GRID, n_boot)
        _check(abs(median - published[grid]) < 0.0051, f"Table S3 median bottleneck {grid}")
        rows.append({"table": "Table S3", "quantity": "median bottleneck radius, mm", "stratum": grid,
                     "n": len(connected), "participants": connected.participant.nunique(), "median": median,
                     "ci_lo": low, "ci_hi": high, "ci_method": "cohort_stratified_participant_block_percentile"})
    return pd.DataFrame(rows)


def corridor_fraction():
    """Corridor volume as a percentage of the three venous labels, disconnected examinations."""
    rows = []
    for grid in GRIDS:
        gap = _gap(grid)
        dis = gap[gap.class26 == "disconnected"]
        voxel = dis.spacing_mm_0 * dis.spacing_mm_1 * dis.spacing_mm_2
        _check(dis.corridor_volume_mm3.notna().all(), f"{grid} corridor volume present")
        _check(np.allclose(dis.corridor_volume_mm3, dis.corridor_voxels * voxel, rtol=1e-6, atol=1e-6),
               f"{grid} corridor volume equals voxels times voxel volume")
        venous = (dis.ivc_voxels + dis.lciv_voxels + dis.rciv_voxels) * voxel
        _check((venous > 0).all(), f"{grid} venous labels present")
        percent = (100 * dis.corridor_volume_mm3 / venous).to_numpy()
        q25, q50, q75 = np.percentile(percent, [25, 50, 75])
        rows.append({"grid": grid, "subset": "disconnected", "n": len(dis),
                     "participants": dis.participant.nunique(), "median_percent": q50,
                     "q25_percent": q25, "q75_percent": q75, "max_percent": float(percent.max()),
                     "median_corridor_voxels": float(dis.corridor_voxels.median()),
                     "median_corridor_mm3": float(dis.corridor_volume_mm3.median()),
                     "median_venous_mm3": float(venous.median()),
                     "zero_voxel_corridors": int((dis.corridor_voxels == 0).sum())})
    return pd.DataFrame(rows)


def bottleneck_fixed(n_boot=N_BOOT):
    """Connected examinations whose bottleneck radius is at most 1.5 mm, on both grids."""
    rows = []
    published = {"native": (299, 981), "inference": (721, 778)}
    for grid in GRIDS:
        gap = _gap(grid)
        connected = gap[gap.class26 != "disconnected"].reset_index(drop=True)
        minimum = connected[["spacing_mm_0", "spacing_mm_1", "spacing_mm_2"]].min(axis=1)
        old = connected.bottleneck_radius_mm <= minimum + 1e-9      # the summaries' own slack
        _check((int(old.sum()), len(connected)) == published[grid], f"Table S3 bottleneck row {grid}")
        rows.append({"grid": grid, "outcome": "bottleneck_radius<=min_spacing (published row)",
                     **_proportion(connected, old, SEED_GRID, n_boot)})
        rows.append({"grid": grid, "outcome": "bottleneck_radius<=1.5mm",
                     **_proportion(connected, connected.bottleneck_radius_mm <= 1.5 + 1e-9, SEED_GRID, n_boot)})
    return pd.DataFrame(rows)


def corridor_labels(n_boot=N_BOOT):
    """Labels outside the venous and arterial categories found in disconnected corridors, and the
    arterial labels held by the corridors flagged as arterial."""
    rows, frequency, location, arteries = [], [], [], []
    for grid in GRIDS:
        gap = _gap(grid)
        dis = gap[gap.class26 == "disconnected"].reset_index(drop=True)
        parsed = dis.corridor_other_labels.map(lambda text: json.loads(text) if text else {})
        _check(np.allclose(parsed.map(lambda labels: sum(labels.values())).to_numpy(float),
                           dis.corridor_other_label_voxels.fillna(0).to_numpy(float)),
               f"{grid} other-label voxel totals")
        nonvascular = parsed.map(lambda labels: any(v > 0 and k not in VASCULAR for k, v in labels.items()))
        vascular = parsed.map(lambda labels: any(v > 0 and k in VASCULAR for k, v in labels.items()))
        viscera = parsed.map(lambda labels: any(v > 0 and k in VISCERA for k, v in labels.items()))
        rows.append({"grid": grid, "outcome": "corridor_contains_nonvascular_label",
                     **_proportion(dis, nonvascular, SEED_GRID, n_boot)})
        rows.append({"grid": grid, "outcome": "corridor_contains_other_vascular_label",
                     **_proportion(dis, vascular, SEED_GRID, n_boot)})
        counts = Counter(k for labels in parsed for k, v in labels.items() if v > 0)
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            frequency.append({"grid": grid, "label": label, "vascular": label in VASCULAR,
                              "examinations": count, "disconnected_n": len(dis)})
        height = dis.corridor_centroid_z_minus_rciv_cranial_mm
        quantiles = np.nanpercentile(height, [10, 25, 50, 75, 90]) if height.notna().any() else [np.nan] * 5
        location.append({"grid": grid, "disconnected_n": len(dis),
                         "corridor_contains_abdominal_organ_label": int(viscera.sum()),
                         "centroid_minus_rciv_cranial_mm_missing": int(height.isna().sum()),
                         **{f"centroid_minus_rciv_cranial_mm_p{p}": float(q) for p, q in zip((10, 25, 50, 75, 90), quantiles)}})
        held = pd.DataFrame({label: dis[column].fillna(0) > 0 for label, column in ARTERIAL_COLUMNS.items()})
        _check((held.any(axis=1) == (dis.arterial_in_corridor == 1)).all(), f"{grid} arterial corridors match their flag")
        for label in ARTERIAL_COLUMNS:
            arteries.append({"grid": grid, "label": label, "examinations": int(held[label].sum()),
                             "arterial_corridors": int(held.any(axis=1).sum()), "disconnected_n": len(dis)})
    return pd.DataFrame(rows), pd.DataFrame(frequency), pd.DataFrame(location), pd.DataFrame(arteries)


def _submodel(label):
    """The submodel that writes a label; a label outside the total task is refused."""
    owners = [name for name, labels in SUBMODELS if label in labels]
    if len(owners) != 1:
        raise ValueError(f"{label!r} is not a class of exactly one TotalSegmentator 2.17.0 total-task submodel")
    return owners[0]


def corridor_exposure(labels):
    """For one corridor's {label: voxels}: whether it holds a non-vascular label, and whether it holds a
    label written after the cardiac submodel. A label with no voxels is absent."""
    owners = {_submodel(label) for label, voxels in labels.items() if voxels > 0}
    return bool(owners - {"cardiac"}), bool(owners & WRITTEN_AFTER_CARDIAC)


def merge_bound():
    """Disconnected corridors in which merging could have replaced a venous label. A voxel that is
    background in the final mask was background in every submodel, and a venous voxel can be
    overwritten only by a submodel written after the cardiac one, whose label then remains."""
    rows = []
    for grid in GRIDS:
        gap = _gap(grid)
        dis = gap[gap.class26 == "disconnected"].reset_index(drop=True)
        parsed = dis.corridor_other_labels.map(lambda text: json.loads(text) if text else {})
        exposure = parsed.map(corridor_exposure)
        nonvascular, later = exposure.map(lambda pair: pair[0]), exposure.map(lambda pair: pair[1])
        _check(int(nonvascular.sum()) == {"native": 94, "inference": 90}[grid],
               f"{grid} corridors holding a non-vascular label (Table S3)")
        labels = sorted({label for corridor in parsed for label, voxels in corridor.items()
                         if voxels > 0 and _submodel(label) in WRITTEN_AFTER_CARDIAC})
        rows.append({"grid": grid, "disconnected_n": len(dis), "nonvascular_n": int(nonvascular.sum()),
                     "later_submodel_n": int(later.sum()), "later_submodel_percent": 100 * later.mean(),
                     "later_submodel_labels": ";".join(labels)})
    return pd.DataFrame(rows)


def reference_annotations(n_boot=N_BOOT):
    """The released annotations under both adjacency rules and their separation in slice units (the
    Table S3 definitions), after reproducing the published counts, and the released cases that lack a
    label, by the label absent and by their released study description. Released cases are treated as
    independent, as in the published Wilson interval."""
    frame = _numeric(_read("reference_cases"), ["gap_mm", "slice_mm", "lciv_touches_image_boundary"])
    cases = frame[frame.status == "measurable"].reset_index(drop=True)
    cases = cases.assign(cohort="TotalSegmentator dataset v2.0.1", participant=cases.case)
    dis = cases[cases.class26 == "disconnected"].reset_index(drop=True)
    _check((len(cases), len(dis)) == (600, 326), "326/600 released annotations disconnected")
    _check(cases.split.value_counts().to_dict() == {"train": 521, "test": 56, "val": 23}
           and dis.split.value_counts().to_dict() == {"train": 284, "test": 31, "val": 11}, "per-split counts")
    kept = cases[cases.lciv_touches_image_boundary == 0].reset_index(drop=True)
    _check((len(kept), int(kept.class26.eq("disconnected").sum())) == (511, 272), "272/511 after boundary exclusion")
    _check(bool(dis.gap_mm.notna().all() and dis.slice_mm.gt(0).all()), "every disconnected annotation has a separation")
    connected = cases[cases.class26 != "disconnected"].reset_index(drop=True)
    units = dis.gap_mm / dis.slice_mm
    rows = [{"outcome": outcome, **_proportion(part, selector, SEED_TOPOLOGY, n_boot)}
            for outcome, part, selector in (
                ("disconnected_26", cases, cases.class26 == "disconnected"),
                ("disconnected_6", cases, cases.class6 == "disconnected"),
                ("connected_26_disconnected_6", connected, connected.class6 == "disconnected"),
                ("disconnected_26_boundary_excluded", kept, kept.class26 == "disconnected"),
                ("disconnected_6_boundary_excluded", kept, kept.class6 == "disconnected"),
                ("separation_le_1_slice_unit", dis, units <= 1 + 1e-9),
                ("separation_gt_2_slice_units", dis, units > 2 + 1e-9))]
    _check(_ci(rows[0]["ci_lo"], rows[0]["ci_hi"]) == "50.3, 58.3", "published Wilson interval 50.3, 58.3")
    q25, median, q75 = np.percentile(dis.gap_mm, [25, 50, 75])
    rows.append({"outcome": "separation_mm_median", "n": len(dis), "estimate": float(median),
                 "q25": float(q25), "q75": float(q75)})
    absent = frame[frame.status != "measurable"]
    ivc, lciv = absent.ivc_present.astype(int).eq(1), absent.lciv_present.astype(int).eq(1)
    _check(len(frame) == 1228 and len(absent) == 628 and not (ivc & lciv).any(),
           "the other 628 released cases each lack the IVC or the left iliac label")
    study = absent.study_type.astype(str)
    lower = study.str.contains(LOWER_BODY_STUDY, case=False)
    upper = study.str.contains(UPPER_BODY_STUDY, case=False) & ~lower
    rows += [{"outcome": f"nonmeasurable:{name}", "n": len(absent), "events": int(selector.sum())}
             for name, selector in (("left_iliac_label_only_absent", ivc & ~lciv), ("ivc_label_only_absent", ~ivc & lciv),
                                    ("both_labels_absent", ~ivc & ~lciv), ("head_neck_chest_or_heart_study", upper),
                                    ("abdomen_pelvis_legs_or_whole_body_study", lower),
                                    ("study_description_unplaced", ~upper & ~lower))]
    return pd.DataFrame(rows)


def saros_region(n_boot=N_BOOT):
    """SAROS disconnection by anatomic region (abdomen, whole body)."""
    meta = _read("saros_metadata")[["id", "anatomic_region"]]
    mapping = _read("participant_mapping")
    processed = mapping[mapping.cohort == "SAROS"].merge(meta, left_on="case", right_on="id", how="left",
                                                         validate="one_to_one")
    _check(processed.anatomic_region.value_counts().to_dict() == {"abdomen": 270, "wholebody": 122},
           "Table 1 SAROS region counts 270 and 122")
    cases = _read("analysis_cases")
    saros = cases[cases.cohort == "SAROS"].merge(meta, left_on="case", right_on="id", how="left",
                                                 validate="one_to_one").reset_index(drop=True)
    _check(len(saros) == 366 and saros.anatomic_region.isin(["abdomen", "wholebody"]).all(),
           "366 measurable SAROS examinations with a region")
    rows = [{"region": "all", "processed": len(processed),
             **_proportion(saros, saros.class26 == "disconnected", SEED_TOPOLOGY, n_boot)}]
    _check(rows[0]["events"] == 177, "SAROS 177/366 disconnected")
    for region, part in saros.groupby("anatomic_region", sort=True):
        rows.append({"region": region, "processed": int((processed.anatomic_region == region).sum()),
                     **_proportion(part.reset_index(drop=True), part.class26 == "disconnected", SEED_TOPOLOGY, n_boot)})
    return pd.DataFrame(rows)


def contrast_metadata(n_boot=N_BOOT):
    """autoPET disconnection by the metadata contrast description, all native spacings."""
    cases = _numeric(_read("analysis_cases"), ["slice_mm"])
    autopet = cases[cases.cohort == "autoPET"].reset_index(drop=True)
    _check(autopet.contrast.value_counts().to_dict() == {"with": 953, "without": 61}, "contrast 953 with, 61 without")
    thin = autopet[(autopet.slice_mm <= 1.5) & (autopet.contrast == "with")]
    _check((len(thin), int(thin.class26.eq("disconnected").sum())) == (192, 87), "Table 2 row 87/192")
    rows = [{"contrast_description": "all",
             **_proportion(autopet, autopet.class26 == "disconnected", SEED_TOPOLOGY, n_boot)}]
    _check(rows[0]["events"] == 381, "autoPET 381/1,014 disconnected")
    for label in ("with", "without"):
        part = autopet[autopet.contrast == label].reset_index(drop=True)
        rows.append({"contrast_description": label,
                     **_proportion(part, part.class26 == "disconnected", SEED_TOPOLOGY, n_boot)})
    return pd.DataFrame(rows)


def age_by_sex():
    """Median (IQR) age by sex with Table 1's participant rule (stats.cohort_table1_row)."""
    mapping = _read("participant_mapping")
    table1 = _read("cohort_table1").set_index("cohort")
    rows = []

    def text(values):
        return f"{np.median(values):.1f} [{np.percentile(values, 25):.1f}-{np.percentile(values, 75):.1f}]"

    for cohort, dataset in (("AMOS", "amos"), ("autoPET", "autopet")):
        meta = _load_meta(dataset)
        first_case = {}
        for case in sorted(mapping.loc[mapping.cohort == cohort, "case"]):
            first_case.setdefault(ST.participant_key(dataset, case, meta), case)
        keys = sorted(first_case) if dataset == "autopet" else [first_case[k] for k in sorted(first_case)]
        records = [meta.get(key) for key in keys if meta.get(key)]
        aged = [r for r in records if isinstance(r.get("age"), (int, float))]
        ages = np.array([r["age"] for r in aged], float)
        females = sum(str(r.get("sex")).upper() == "F" for r in records if r.get("sex"))
        _check(len(first_case) == int(table1.loc[cohort, "n_participants"]), f"{cohort} participants")
        _check((len(ages), text(ages)) == (int(table1.loc[cohort, "age_available_n"]),
                                           table1.loc[cohort, "age_years_median_iqr"]), f"{cohort} Table 1 age")
        _check(females == int(table1.loc[cohort, "female_n"]), f"{cohort} Table 1 female count")
        for sex, label in (("F", "female"), ("M", "male"), (None, "all")):
            values = np.array([r["age"] for r in aged if sex is None or str(r.get("sex", "")).upper() == sex], float)
            rows.append({"cohort": cohort, "sex": label, "n": len(values), "median": float(np.median(values)),
                         "q25": float(np.percentile(values, 25)), "q75": float(np.percentile(values, 75)),
                         "mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())})
    return pd.DataFrame(rows)


def repeat_examinations():
    """The agreement between repeat examinations already estimated (Table S4), read, not recomputed."""
    frame = _numeric(_read("longitudinal_agreement"), ["estimate", "ci_lo", "ci_hi", "participants", "scans", "pairs"])
    keep = frame[frame.metric.isin(["agreement", "nominal_kappa", "nominal_ac1"])].copy()
    binary = keep[(keep.endpoint == "binary") & (keep.metric == "agreement")].iloc[0]
    kappa = keep[(keep.endpoint == "binary") & (keep.metric == "nominal_kappa")].iloc[0]
    _check((round(100 * binary.estimate, 1), round(kappa.estimate, 2)) == (79.1, 0.56), "Table S4 79.1% and 0.56")
    _check((int(binary.participants), int(binary.scans), int(binary.pairs)) == (81, 195, 161), "Table S4 81/195/161")
    return keep[["endpoint", "metric", "estimate", "ci_lo", "ci_hi", "participants", "scans", "pairs"]]


def discordant_contacts(groups):
    """Over every unordered pair within each group of (class26, class6) records: the number of pairs, of
    pairs whose binary 26-connectivity differs, and of those whose connected member is face-connected."""
    pairs = discordant = face = 0
    for records in groups:
        for first, second in combinations(records, 2):
            pairs += 1
            connected = [r for r in (first, second) if r[0] != "disconnected"]
            if len(connected) == 1:
                discordant += 1
                face += connected[0][1] != "disconnected"
    return pairs, discordant, face


def repeat_contacts():
    """In repeat autoPET pairs that disagree on binary connectivity, whether the connected examination's
    path needed an edge or corner contact (connected under 26- but not 6-connectivity), on each grid.

    The pairs are the longitudinal agreement's (connectivity_statistics.longitudinal_agreement): every
    unordered pair of a participant's measurable autoPET examinations, with the direct and
    right-iliac-mediated classes counted as connected.
    """
    frame = _joined_grids()
    autopet = frame[frame.cohort == "autoPET"]
    repeat = autopet[autopet.participant.map(autopet.participant.value_counts()) >= 2]
    agreement = _numeric(_read("longitudinal_agreement"), ["participants", "scans", "pairs"])
    binary = agreement[(agreement.endpoint == "binary") & (agreement.metric == "agreement")].iloc[0]
    _check((repeat.participant.nunique(), len(repeat)) == (int(binary.participants), int(binary.scans)),
           "the 81 participants and 195 examinations of Table S4")
    rows = []
    for grid, prefix in (("native", ""), ("inference", "internal_")):
        c26, c6 = f"{prefix}class26", f"{prefix}class6"
        groups = [list(zip(g[c26], g[c6])) for _, g in repeat.groupby("participant", sort=True)]
        pairs, discordant, face = discordant_contacts(groups)
        connected = frame[c26].ne("disconnected")
        rows.append({"grid": grid, "pairs": pairs, "discordant_pairs": discordant,
                     "discordant_connected_face_connected": face,
                     "discordant_connected_edge_or_corner_only": discordant - face,
                     "connected_examinations": int(connected.sum()),
                     "connected_edge_or_corner_only": int((connected & frame[c6].eq("disconnected")).sum())})
    out = pd.DataFrame(rows).set_index("grid")
    _check(int(out.loc["native", "pairs"]) == int(binary.pairs) == 161, "the 161 unordered pairs of Table S4")
    _check(int(out.loc["native", "discordant_pairs"]) == 34, "34 of the 161 pairs disagree on binary connectivity")
    _check(out.loc["native", ["connected_examinations", "connected_edge_or_corner_only"]].tolist() == [981, 218],
           "Table S3: 218 of 981 native connections need an edge or corner contact")
    _check(out.loc["inference", ["connected_examinations", "connected_edge_or_corner_only"]].tolist() == [778, 397],
           "Table S3: 397 of 778 inference-grid connections need an edge or corner contact")
    return out.reset_index()


def nonmeasurable_labels():
    """Which labels were empty in the nonmeasurable examinations, from the archived export's voxel counts."""
    names = {"ivc": "inferior_vena_cava", "left": "iliac_vena_left", "right": "iliac_vena_right"}
    frame = _numeric(_read("reproduction_fidelity"), [f"archive_voxels_{name}" for name in names.values()])
    _check(len(frame) == 1899, "1,899 processed examinations")
    empty = {key: frame[f"archive_voxels_{name}"].eq(0) for key, name in names.items()}
    nonmeasurable = frame.archive_class26.eq("nonmeasurable")
    _check(nonmeasurable.eq(empty["ivc"] | empty["left"]).all(),
           "nonmeasurable exactly where the IVC or left iliac label is empty")
    rows = []
    for cohort, part in [("pooled", frame)] + list(frame.groupby("cohort", sort=True)):
        at = part.index
        rows.append({"cohort": cohort, "processed": len(part), "nonmeasurable": int(nonmeasurable[at].sum()),
                     "ivc_empty_only": int((empty["ivc"] & ~empty["left"])[at].sum()),
                     "left_iliac_empty_only": int((empty["left"] & ~empty["ivc"])[at].sum()),
                     "both_empty": int((empty["ivc"] & empty["left"])[at].sum()),
                     "left_iliac_and_right_iliac_empty": int((empty["left"] & empty["right"])[at].sum())})
    out = pd.DataFrame(rows).set_index("cohort")
    _check(int(out.loc["pooled", "nonmeasurable"]) == 26 == int(out.loc["SAROS", "nonmeasurable"]),
           "the 26 nonmeasurable examinations, all SAROS")
    return out.reset_index()


def _topology_tables():
    """The per-examination Betti numbers and gap levels, and the released annotations' Betti numbers, as
    the run recorded them (mask_topology.py)."""
    record = _read_json("mask_topology_record")
    cases = _numeric(_read("mask_topology"), ["b0", "b1", "b2", "gap_level"])
    references = _numeric(_read("reference_topology"), ["b0", "b1", "b2"])
    _check(record["outputs"] == {"mask_topology_cases.csv": _DIGESTS["mask_topology"],
                                 "reference_topology_cases.csv": _DIGESTS["reference_topology"]},
           "the topology tables are the ones their run recorded")
    return cases, references


def betti_rates(n_boot=N_BOOT):
    """Betti numbers of the venous union, (b0, b1, b2) = components, tunnels and cavities: the share of masks
    with the topology of the anatomical confluence, one tube dividing in two (1, 0, 0), and of masks with more
    than one component, a tunnel or a cavity, by cohort on the native grid and among connected examinations;
    and the same in the released annotations (Wilson intervals, as for their audit)."""
    cases, references = _topology_tables()
    frame = _joined_grids().merge(cases[["cohort", "case", "b0", "b1", "b2"]], on=["cohort", "case"],
                                  how="left", validate="one_to_one")
    _check(frame.b0.notna().all() and frame.b0.eq(pd.to_numeric(frame.ncomp26)).all(),
           "b0 equals the component count the analysis recorded")
    _check(int(frame.b0.gt(1).sum()) == 977 and frame.loc[frame.class26.eq("disconnected"), "b0"].ge(2).all(),
           "977 venous unions with more than one component, every disconnected one among them")

    def outcomes(d):
        return {"anatomical_topology": d.b0.eq(1) & d.b1.eq(0) & d.b2.eq(0), "b0_gt1": d.b0.gt(1),
                "b1_gt0": d.b1.gt(0), "b2_gt0": d.b2.gt(0)}
    rows = []
    strata = [("all", frame), ("connected", frame[frame.class26.ne("disconnected")])]
    for stratum, part in strata:
        part = part.reset_index(drop=True)
        for outcome in outcomes(part):
            for cohort, sub in [("pooled", part)] + list(part.groupby("cohort", sort=True)):
                sub = sub.reset_index(drop=True)
                rows.append({"source": "predictions", "stratum": stratum, "outcome": outcome, "cohort": cohort,
                             **_proportion(sub, outcomes(sub)[outcome], SEED_TOPOLOGY, n_boot)})
    _check(len(references) == 600, "the 600 released annotations the venous audit measured")
    for outcome, selector in outcomes(references).items():
        events, n = int(selector.sum()), len(references)
        lo, hi = S.wilson_ci(events, n, CFG.ALPHA)
        rows.append({"source": "released_annotations", "stratum": "all", "outcome": outcome, "cohort": "all",
                     "n": n, "events": events, "estimate": 100 * events / n, "ci_lo": 100 * lo, "ci_hi": 100 * hi,
                     "participants": n, "ci_method": "Wilson_independent_case_proxies"})
    return pd.DataFrame(rows)


def gap_levels():
    """The vertebral level of each disconnected examination's corridor (mask_topology.py: the corridor
    centroid on a ruler from the examination's own vertebra labels), by cohort, among rulers that pass
    measure_bifurcation's checks; by vertebral bin [k - 0.5, k + 0.5) and by body or disc label."""
    cases, _ = _topology_tables()
    disconnected = cases[cases.class26.eq("disconnected")]
    _check(len(disconnected) == 892, "the 892 disconnected examinations")
    valid = disconnected[disconnected.ruler_qc.eq("OK") & disconnected.gap_level.notna()]
    rows = []
    for cohort, part in [("pooled", valid)] + list(valid.groupby("cohort", sort=True)):
        total = len(disconnected) if cohort == "pooled" else int(disconnected.cohort.eq(cohort).sum())
        for scale, column in (("bin", "gap_level_bin"), ("label", "gap_level_label")):
            for level, n in part[column].value_counts().sort_index().items():
                rows.append({"cohort": cohort, "scale": scale, "level": level, "n": int(n), "valid": len(part),
                             "disconnected": total})
        rows.append({"cohort": cohort, "scale": "median_level", "level": f"{part.gap_level.median():.2f}",
                     "n": len(part), "valid": len(part), "disconnected": total})
    return pd.DataFrame(rows)


def attenuation_quartiles():
    """Disconnection by quartile of whole-label venous attenuation, from the comparator run's aggregates.
    The values describe the labels the network drew; they select, classify and exclude nothing. Counts by
    cohort and quartile with Wilson intervals, and the lowest quartile against the other three as a
    cohort-stratified Mantel-Haenszel odds ratio."""
    from statsmodels.stats.contingency_tables import StratifiedTable
    run = _read_json("comparators_record")
    summary = _read("attenuation_summary")
    _check(run["outputs"]["attenuation_summary.csv"] == _DIGESTS["attenuation_summary"],
           "the attenuation summary is the one its run recorded")
    cuts = _numeric(summary[summary.kind.eq("quartile_cut")], ["median"])["median"].tolist()
    _check(cuts == [55.0, 88.0, 108.0], "the quartile cuts 55, 88 and 108 HU")
    counts = _numeric(summary[summary.kind.eq("count_by_quartile")], ["n", "quartile"])
    totals = counts.groupby("cohort").n.sum().to_dict()
    disconnected = counts[counts.native_class26.eq("disconnected")].groupby("cohort").n.sum().to_dict()
    _check(totals == {"AMOS": 493, "SAROS": 366, "autoPET": 1014} and
           disconnected == {"AMOS": 334, "SAROS": 177, "autoPET": 381},
           "the measurable and disconnected counts of each cohort")
    rows, tables = [], []
    for cohort, part in [("pooled", counts)] + list(counts.groupby("cohort", sort=True)):
        for quartile, cell in part.groupby("quartile", sort=True):
            events, n = int(cell[cell.native_class26.eq("disconnected")].n.sum()), int(cell.n.sum())
            lo, hi = S.wilson_ci(events, n, CFG.ALPHA)
            rows.append({"row": "rate", "cohort": cohort, "quartile": int(quartile), "n": n, "events": events,
                         "estimate": 100 * events / n, "ci_lo": 100 * lo, "ci_hi": 100 * hi, "ci_method": "Wilson"})
        if cohort != "pooled":
            low = part[part.quartile.eq(1)]
            high = part[part.quartile.ne(1)]
            d_low, d_high = int(low[low.native_class26.eq("disconnected")].n.sum()), int(high[high.native_class26.eq("disconnected")].n.sum())
            tables.append([[d_low, int(low.n.sum()) - d_low], [d_high, int(high.n.sum()) - d_high]])
    table = StratifiedTable([np.asarray(t, float) for t in tables])
    lo, hi = table.oddsratio_pooled_confint()
    rows.append({"row": "mantel_haenszel_lowest_vs_others", "cohort": "stratified", "quartile": 1,
                 "estimate": float(table.oddsratio_pooled), "ci_lo": float(lo), "ci_hi": float(hi),
                 "p_value": float(table.test_null_odds().pvalue), "ci_method": "Mantel-Haenszel"})
    return pd.DataFrame(rows)


def projection_extended(n_boot=N_BOOT):
    """The paired change where projection can act and under six-connectivity.

    At native spacing of 1.5 mm or less the projection is a refinement and cannot change a class,
    so the change is also reported over the examinations above 1.5 mm. The frame, its order and the
    seed are paired_projection's, so the pooled 26-connectivity row reproduces its interval.
    """
    frame = _numeric(_read("paired_connectivity"), ["native_spacing_mm"])
    measurable = frame[frame.archive_class26 != "nonmeasurable"].reset_index(drop=True)
    _check(len(measurable) == 1873 and measurable.archive_class6.ne("nonmeasurable").all()
           and measurable.internal_class6.ne("nonmeasurable").all(), "1,873 measurable under both rules")
    thick = (measurable.native_spacing_mm > 1.5).to_numpy()
    rows = []
    for rule in ("26", "6"):
        native = measurable[f"archive_class{rule}"].eq("disconnected").to_numpy()
        inference = measurable[f"internal_class{rule}"].eq("disconnected").to_numpy()
        for stratum, keep in (("all", np.ones(len(measurable), bool)), ("spacing_gt1p5mm", thick),
                              ("spacing_le1p5mm", ~thick)):
            part = measurable[keep].reset_index(drop=True)
            before, after = native[keep], inference[keep]
            s = FS.proportion_intervals(part, after.astype(int) - before.astype(int), n_boot, False, SEED_GRID)
            test = durkalski_test(part.participant, before, after)
            rows.append({"adjacency": rule, "stratum": stratum, "n": len(part), "participants": int(s["participants"]),
                         "native_disconnected": int(before.sum()), "inference_disconnected": int(after.sum()),
                         "closed_by_projection": int((after & ~before).sum()),
                         "opened_by_projection": int((before & ~after).sum()),
                         "net_examinations": int(after.sum() - before.sum()), "estimate": 100 * s["proportion"],
                         "ci_lo": 100 * s["ci_lo"], "ci_hi": 100 * s["ci_hi"], "ci_method": s["ci_method"],
                         "statistic": test["statistic"], "p_value": test["p_value"]})
    out = pd.DataFrame(rows)
    cell = lambda rule, stratum: out[(out.adjacency == rule) & (out.stratum == stratum)].iloc[0]  # noqa: E731
    counts = ["native_disconnected", "inference_disconnected", "closed_by_projection", "opened_by_projection"]
    _check(cell("26", "all")[counts].tolist() == [892, 1095, 205, 2], "26: Figure 5c")
    if n_boot == N_BOOT:   # the published interval holds at the published resample count
        _check((round(cell("26", "all").ci_lo, 1), round(cell("26", "all").ci_hi, 1)) == (9.4, 12.3),
               "26: the published interval 9.4 to 12.3")
    _check(cell("26", "spacing_gt1p5mm")[["n", "net_examinations"]].tolist() == [1597, 203],
           "26 above 1.5 mm: 142 of 1,145 plus 61 of 452")
    _check(cell("26", "spacing_le1p5mm")[["n", "net_examinations", "closed_by_projection"]].tolist() == [276, 0, 0],
           "26 at or below 1.5 mm: no change")
    _check(cell("6", "all")[counts].tolist() == [1110, 1492, 390, 8], "6: the transitions table")
    return out


def _joined_grids():
    """The measurable frame in Table 2's order and participant keys, with each examination's
    inference-grid classes from the model-grid audit."""
    cases = _numeric(_read("analysis_cases"), ["slice_mm"])
    paired = _numeric(_read("paired_connectivity"), ["native_spacing_mm"])[
        ["cohort", "case", "archive_class26", "archive_class6", "internal_class26", "internal_class6",
         "native_spacing_mm"]]
    frame = cases.merge(paired, on=["cohort", "case"], how="left", validate="one_to_one")
    _check(len(frame) == 1873 and frame.internal_class26.notna().all()
           and frame.internal_class26.ne("nonmeasurable").all() and frame.internal_class6.ne("nonmeasurable").all(),
           "1,873 measurable examinations on both grids")
    _check(frame.class26.eq(frame.archive_class26).all() and frame.class6.eq(frame.archive_class6).all(),
           "the archived native classes are the analysed classes")
    _check(np.allclose(frame.slice_mm, frame.native_spacing_mm, rtol=0, atol=5e-4),   # slice_mm is kept to 0.001 mm
           "one native slice spacing in both tables")
    return frame


def inference_grid_rates(n_boot=N_BOOT):
    """Disconnection on each grid by cohort under both adjacency rules (Table 2 rows)."""
    frame = _joined_grids()
    published = {("26", "AMOS"): (334, 63.5, 71.7), ("26", "SAROS"): (177, 43.3, 53.5),
                 ("26", "autoPET"): (381, 34.4, 40.8), ("26", "pooled"): (892, 45.4, 49.9),
                 ("6", "AMOS"): (368, 70.6, 78.3), ("6", "SAROS"): (205, 50.9, 61.0),
                 ("6", "autoPET"): (537, 49.7, 56.3), ("6", "pooled"): (1110, 57.0, 61.5)}
    rows = []
    for grid, prefix in (("native", ""), ("inference", "internal_")):
        for rule in ("26", "6"):
            column = f"{prefix}class{rule}"
            parts = [("pooled", frame)] + [(c, p.reset_index(drop=True)) for c, p in frame.groupby("cohort", sort=True)]
            for cohort, part in parts:
                r = {"grid": grid, "adjacency": rule, "cohort": cohort,
                     **_proportion(part, part[column] == "disconnected", SEED_TOPOLOGY, n_boot)}
                if grid == "native":
                    expected = published[(rule, cohort)]
                    observed = (r["events"], round(r["ci_lo"], 1), round(r["ci_hi"], 1))
                    _check(observed[0] == expected[0] and (n_boot != N_BOOT or observed == expected),
                           f"Table 2 native {rule}-connectivity {cohort}")
                rows.append(r)
    out = pd.DataFrame(rows)
    pooled = out[(out.grid == "inference") & (out.cohort == "pooled")].set_index("adjacency").events.to_dict()
    _check(pooled == {"26": 1095, "6": 1492}, "inference grid: 1,095 and 1,492 of 1,873")
    return out


def inference_grid_model():
    """The cohort-and-spacing model of disconnection (connectivity_statistics.model_results,
    disconnection_spacing_categorical) refit on each grid and adjacency rule."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    frame = _joined_grids()
    base = frame[np.isfinite(frame.slice_mm) & (frame.slice_mm > 0)].copy()
    base["spacing_category"] = pd.cut(base.slice_mm, [0, 1.5, 3.0, np.inf], labels=["le1p5", "gt1p5_le3", "gt3"],
                                      right=True)
    published = _numeric(_read("model_coefficients"), ["estimate"])
    published = published[published.model == "disconnection_spacing_categorical"].set_index("term").estimate
    rows = []
    for grid, column, rule in (("native", "class26", "26"), ("inference", "internal_class26", "26"),
                               ("native", "class6", "6"), ("inference", "internal_class6", "6")):
        d = base.assign(disconnected=base[column].eq("disconnected").astype(int))
        fit = smf.glm("disconnected ~ C(spacing_category) + C(cohort)", data=d, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": d.participant, "use_correction": True}, use_t=True)
        interval = fit.conf_int()
        if (grid, rule) == ("native", "26"):
            _check(np.allclose(fit.params[published.index], published, atol=1e-6), "the published native coefficients")
        for term in fit.params.index:
            rows.append({"grid": grid, "adjacency": rule, "term": term, "estimate": float(fit.params[term]),
                         "ci_lo": float(interval.loc[term, 0]), "ci_hi": float(interval.loc[term, 1]),
                         "p": float(fit.pvalues[term]), "odds_ratio": float(np.exp(fit.params[term])),
                         "or_lo": float(np.exp(interval.loc[term, 0])), "or_hi": float(np.exp(interval.loc[term, 1])),
                         "n": int(fit.nobs), "n_participants": int(d.participant.nunique())})
    return pd.DataFrame(rows)


def _autopet_studies():
    """One record per autoPET study: diagnosis, age and sex as released for that study.

    stats._meta_autopet keeps one record per subject, so a subject whose studies carry different
    diagnoses gets the first one for all of them; here each study keeps its own.
    """
    meta = _read("autopet_metadata")
    crosswalk = _read("autopet_crosswalk")
    ct = meta[meta.Modality == "CT"].copy()
    parts = ct["File Location"].str.split("/")
    ct["subject"], ct["study_dir"] = parts.str[2], parts.str[3]
    _check(ct.groupby(["subject", "study_dir"])[["diagnosis", "age", "sex"]].nunique().le(1).all().all()
           and not ct.duplicated(["subject", "study_dir"]).any(), "one CT series and one record per study")
    studies = crosswalk.merge(ct[["subject", "study_dir", "diagnosis", "age", "sex"]], on=["subject", "study_dir"],
                              how="left", validate="one_to_one")
    _check(len(studies) == 1014 and studies.diagnosis.notna().all(), "every analysed autoPET study has its record")
    studies["age_years"] = studies.age.str.rstrip("Y").astype(int)
    return studies


def population_sensitivity(n_boot=N_BOOT):
    """Disconnection without the participants under 18 and in autoPET by the diagnosis of each study."""
    studies = _autopet_studies()
    cases = _read("analysis_cases")
    autopet = cases[cases.cohort == "autoPET"].merge(studies, left_on="case", right_on="phase1_case", how="left",
                                                     validate="one_to_one").reset_index(drop=True)
    _check(len(autopet) == 1014 and autopet.diagnosis.notna().all(), "1,014 measurable autoPET examinations")
    meta = _load_meta("autopet")
    collapsed = Counter(meta[subject]["region"] for subject in autopet.subject)
    _check(collapsed == {"NEGATIVE": 508, "MELANOMA": 191, "LUNG_CANCER": 169, "LYMPHOMA": 146},
           "Table 1's diagnosis counts come from the subject-level record")
    rows = [{"row": f"table1_subject_record:{label}", "n": int(n)} for label, n in sorted(collapsed.items())]
    for label, part in autopet.groupby("diagnosis", sort=True):
        rows.append({"row": f"diagnosis_per_study:{label}",
                     **_proportion(part.reset_index(drop=True), part.class26 == "disconnected", SEED_TOPOLOGY, n_boot)})
    minor = autopet.age_years < 18
    rows.append({"row": "under_18:autoPET", "n": int(minor.sum()), "participants": int(autopet[minor].subject.nunique()),
                 "events": int(autopet[minor].class26.eq("disconnected").sum())})
    adult = autopet[~minor].reset_index(drop=True)
    rows.append({"row": "without_under_18:autoPET",
                 **_proportion(adult, adult.class26 == "disconnected", SEED_TOPOLOGY, n_boot)})
    keep = ~cases.case.isin(autopet.loc[minor, "case"])
    pooled = cases[keep].reset_index(drop=True)
    rows.append({"row": "without_under_18:pooled",
                 **_proportion(pooled, pooled.class26 == "disconnected", SEED_TOPOLOGY, n_boot)})
    out = pd.DataFrame(rows)
    per_study = out[out.row.str.startswith("diagnosis_per_study")].set_index("row").n.to_dict()
    _check(sum(per_study.values()) == 1014, "per-study diagnoses cover all 1,014")
    return out


def cross_dataset_overlap():
    """Whether any processed SAROS examination comes from autoPET's TCIA collection or study."""
    saros = _read("saros_metadata")
    mapping = _read("participant_mapping")
    processed = mapping[mapping.cohort == "SAROS"].merge(saros, left_on="case", right_on="id", how="left",
                                                         validate="one_to_one")
    _check(len(processed) == 392 and processed.tcia_collection.notna().all(), "392 processed SAROS cases")
    autopet = _read("autopet_metadata")
    rows = [{"item": f"saros_collection:{name}", "cases": int(n)}
            for name, n in processed.tcia_collection.value_counts().sort_index().items()]
    rows += [{"item": "saros_collections", "cases": int(processed.tcia_collection.nunique())},
             {"item": "saros_cases_in_autopet_collection",
              "cases": int(processed.tcia_collection.isin(set(autopet.Collection)).sum())},
             {"item": "study_uid_shared_with_autopet",
              "cases": int(processed.tcia_study_instance_uid.isin(set(autopet["Study UID"])).sum())},
             {"item": "series_uid_shared_with_autopet",
              "cases": int(processed.tcia_series_instance_uid.isin(set(autopet["Series UID"])).sum())}]
    return pd.DataFrame(rows)


def comparator_reachability(n_boot=N_BOOT):
    """Reachability of the IVC to the right iliac vein and of the aorta to each common iliac artery,
    beside the left iliac endpoint, on the 1,873 measurable examinations by grid, adjacency rule and
    cohort; and the right iliac result within each left iliac class.

    The comparator table comes from confluence_comparators.py, which recomputes the left iliac
    classes of every examination on both grids and refuses any that differ from the recorded ones.
    Each comparator pair is tested within its own two labels, so the left iliac vein is reported
    both ways: disconnected (no path even through the right iliac label, the primary
    endpoint) and not directly connected (no path within the IVC and left iliac labels alone).
    """
    run = _read_json("comparators_record")
    comparators = _read("comparators")
    _check(run["outputs"]["comparators_cases.csv"] == _DIGESTS["comparators"],
           "the comparator table is the one its run recorded")
    _check(run["failures"] == {} and run["examinations"] == len(comparators) == 1899,
           "the comparator run processed all 1,899 examinations without a failure")
    frame = _joined_grids().merge(comparators.drop(columns=["participant", "mask_case"]), on=["cohort", "case"],
                                  how="left", validate="one_to_one")
    _check(frame[[f"{grid}_{pair}_{rule}" for grid in GRIDS for pair in COMPARATORS for rule in ("26", "6")]]
           .isin(["connected", "disconnected", "nonmeasurable"]).all().all(),
           "every measurable examination has all twelve comparator results")
    rows, cross = [], []
    for grid, prefix in (("native", ""), ("inference", "internal_")):
        for rule in ("26", "6"):
            venous = frame[f"{prefix}class{rule}"]
            events = {"ivc_lciv": venous.eq("disconnected"), "ivc_lciv_not_direct": venous.ne("direct")}
            measurable = dict.fromkeys(events, venous.ne("nonmeasurable"))
            for pair in COMPARATORS:
                result = frame[f"{grid}_{pair}_{rule}"]
                events[pair], measurable[pair] = result.eq("disconnected"), result.ne("nonmeasurable")
            for pair, event in events.items():
                kept = frame.assign(comparator_event=event)[measurable[pair]]
                for cohort, part in [("pooled", kept)] + list(kept.groupby("cohort", sort=True)):
                    part = part.reset_index(drop=True)
                    rows.append({"grid": grid, "adjacency": rule, "pair": pair, "cohort": cohort,
                                 **_proportion(part, part.comparator_event, SEED_TOPOLOGY, n_boot)})
            right = frame[f"{grid}_ivc_rciv_{rule}"]
            for (label, reach), n in pd.crosstab(venous, right).stack().items():
                cross.append({"grid": grid, "adjacency": rule, "left_class": label, "ivc_rciv": reach, "n": int(n)})
    out = pd.DataFrame(rows)
    pooled = out[out.cohort == "pooled"].set_index(["grid", "adjacency", "pair"])
    _check({key: int(pooled.loc[(*key, "ivc_lciv"), "events"]) for key in
            [("native", "26"), ("native", "6"), ("inference", "26"), ("inference", "6")]}
           == {("native", "26"): 892, ("native", "6"): 1110, ("inference", "26"): 1095, ("inference", "6"): 1492}
           and pooled.xs("ivc_lciv", level="pair").n.eq(1873).all(),
           "the published left iliac disconnection on each grid and adjacency rule")
    _check([int(pooled.loc[(grid, "26", "ivc_lciv_not_direct"), "events"]) for grid in GRIDS] == [1873 - 384, 1873 - 251],
           "the published direct connections, 384 natively and 251 on the inference grid")
    return out, pd.DataFrame(cross)


METHOD_LABELS = {"Wilson_independent_case_proxies": "Wilson",
                 "cohort_stratified_participant_percentile": "participant-block percentile",
                 "participant_block_percentile": "participant-block percentile",
                 "cohort_stratified_participant_block_percentile": "participant-block percentile",
                 "participant_cluster_BCa": "participant-cluster BCa", "iid_BCa": "BCa"}


def _f(value, digits=1, signed=False):
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _ci(low, high, digits=1):
    """Intervals print through statlib.format_ci, the project's one interval formatter."""
    return S.format_ci(low, high, nd=digits, sep=", ") or "not estimable"


def _m(method):
    return METHOD_LABELS.get(method, method)


def _exact(value):
    """One decimal, or two when one would round a value that ends in 5 (an exact quartile)."""
    return f"{value:.1f}" if abs(10 * value - round(10 * value)) < 1e-9 else f"{value:.2f}"


def summary_markdown(t):
    """The results in reportable form, each with the table and row it comes from."""
    paired, medians, fraction, bottleneck = t["t1_paired_projection.csv"], t["t1_median_intervals.csv"], \
        t["t1_corridor_fraction.csv"], t["t1_bottleneck_fixed.csv"]
    labels, location = t["t1_corridor_labels.csv"], t["t1_corridor_location.csv"]
    saros, contrast, ages, repeat = t["t1_saros_region.csv"], t["t1_contrast_metadata.csv"], \
        t["t1_age_by_sex.csv"], t["t1_repeat_examinations.csv"]
    row = lambda frame, column, value: frame[frame[column] == value].iloc[0]  # noqa: E731
    lines = ["# Summary tables: results", "",
             "Produced by `Code/phase1_characterize/summary/summary_tables.py` from the case-level "
             "outputs in `Results/phase1_characterize/`; inputs, hashes, versions and seeds are in "
             "`run_record.json`. Percentages are rounded plainly to one decimal and intervals print "
             "through `statlib.format_ci`.", ""]
    m = row(paired, "row", "paired_change:measurable")
    test = row(paired, "row", "clustered_mcnemar:durkalski_2003")
    exact = row(paired, "row", "mcnemar:exact_unclustered")
    pooled = row(paired, "row", "discordant:pooled")
    conservative = row(paired, "row", "paired_change:processed_empty_counted_disconnected")
    nat = row(paired, "row", "prevalence:native:processed_empty_counted_disconnected")
    inf = row(paired, "row", "prevalence:inference:processed_empty_counted_disconnected")
    lines += ["## 1. Paired change, inference grid minus native grid (`t1_paired_projection.csv`)", "",
              f"- Measurable frame: {int(m.net_examinations)}/{int(m.n):,} = {_f(m.estimate, signed=True)} percentage points "
              f"(95% CI: {_ci(m.ci_lo, m.ci_hi)}; {int(m.participants):,} participants, {_m(m.ci_method)}).",
              f"- Discordant examinations: {int(pooled.closed_by_projection)} disconnected only on the inference grid, "
              f"{int(pooled.opened_by_projection)} disconnected only on the native grid.",
              f"- Clustered McNemar (Durkalski 2003, participants as clusters): X² = {test.statistic:.1f}, "
              f"P = {test.p_value:.1e}; exact unclustered McNemar P = {exact.p_value:.1e}. Both P < .001.",
              f"- Conservative sensitivity (26 empty-label examinations counted as disconnected, 1,899 processed): "
              f"native {_f(nat.estimate)}% (95% CI: {_ci(nat.ci_lo, nat.ci_hi)}), inference {_f(inf.estimate)}% "
              f"(95% CI: {_ci(inf.ci_lo, inf.ci_hi)}), paired change {_f(conservative.estimate, signed=True)} points "
              f"(95% CI: {_ci(conservative.ci_lo, conservative.ci_hi)}).", ""]
    for _, r in paired[paired.row.str.startswith("paired_change:spacing")].iterrows():
        lines.append(f"- {r.row.split(':')[1]}: {int(r.net_examinations)}/{int(r.n):,} = {_f(r.estimate, signed=True)} points "
                     f"(95% CI: {_ci(r.ci_lo, r.ci_hi)})")
    lines += ["", "## 2. Medians with participant-block percentile intervals (`t1_median_intervals.csv`)", ""]
    for _, r in medians.iterrows():
        lines.append(f"- {r.table}, {r.quantity}, {r.stratum}: {r['median']:.2f} (95% CI: {_ci(r.ci_lo, r.ci_hi, 2)})")
    lines += ["", "## 3. Corridor volume as a percentage of the venous labels, disconnected examinations "
              "(`t1_corridor_fraction.csv`)", ""]
    for _, r in fraction.iterrows():
        lines.append(f"- {r.grid} grid (n = {int(r.n):,}): median {r.median_percent:.2f}% (IQR, {r.q25_percent:.2f}-"
                     f"{r.q75_percent:.2f}); maximum {r.max_percent:.2f}%; median corridor {r.median_corridor_voxels:.0f} voxels "
                     f"({r.median_corridor_mm3:.1f} mm³) against median venous labels {r.median_venous_mm3:,.0f} mm³")
    lines += ["", "## 4. Bottleneck at a fixed 1.5 mm, connected examinations (`t1_bottleneck_fixed.csv`)", ""]
    for _, r in bottleneck.iterrows():
        lines.append(f"- {r.grid} grid, {r.outcome}: {int(r.events)}/{int(r.n):,} ({_f(r.estimate)}%; 95% CI: "
                     f"{_ci(r.ci_lo, r.ci_hi)})")
    lines += ["", "## 5. Labels in disconnected corridors (`t1_corridor_labels.csv`, `t1_corridor_label_frequency.csv`)", ""]
    for _, r in labels.iterrows():
        lines.append(f"- {r.grid} grid, {r.outcome}: {int(r.events)}/{int(r.n):,} ({_f(r.estimate)}%; 95% CI: "
                     f"{_ci(r.ci_lo, r.ci_hi)})")
    arteries = t["t1_corridor_arterial_labels.csv"]
    for grid in GRIDS:
        held = arteries[arteries.grid == grid]
        lines.append(f"- {grid} grid, arterial corridors ({int(held.arterial_corridors.iloc[0])} of "
                     f"{int(held.disconnected_n.iloc[0]):,}) by label (`t1_corridor_arterial_labels.csv`): "
                     + "; ".join(f"{x.label} {int(x.examinations)}" for _, x in held.iterrows()))
    frequency = t["t1_corridor_label_frequency.csv"]
    for grid in GRIDS:
        top = frequency[frequency.grid == grid].head(8)
        lines.append(f"- {grid} grid, most frequent: " + "; ".join(f"{x.label} {int(x.examinations)}" for _, x in top.iterrows()))
    for _, r in location.iterrows():
        lines.append(f"- Corridor location, {r.grid} grid: {int(r.corridor_contains_abdominal_organ_label)} of "
                     f"{int(r.disconnected_n):,} corridors contain an abdominal organ label; corridor centroid minus "
                     f"right iliac cranial end, median {r.centroid_minus_rciv_cranial_mm_p50:.1f} mm "
                     f"(10th-90th percentile, {r.centroid_minus_rciv_cranial_mm_p10:.1f} to {r.centroid_minus_rciv_cranial_mm_p90:.1f})")
    lines += ["", "## 6. SAROS disconnection by region (`t1_saros_region.csv`)", ""]
    for _, r in saros.iterrows():
        lines.append(f"- {r.region} ({int(r.processed)} processed): {int(r.events)}/{int(r.n)} ({_f(r.estimate)}%; 95% CI: "
                     f"{_ci(r.ci_lo, r.ci_hi)}; {_m(r.ci_method)})")
    lines += ["", "## 7. autoPET disconnection by contrast description, all spacings (`t1_contrast_metadata.csv`)", ""]
    for _, r in contrast.iterrows():
        lines.append(f"- {r.contrast_description}: {int(r.events)}/{int(r.n):,} ({_f(r.estimate)}%; 95% CI: "
                     f"{_ci(r.ci_lo, r.ci_hi)}; {_m(r.ci_method)})")
    lines += ["", "## 8. Age by sex, Table 1 participant rule (`t1_age_by_sex.csv`)", ""]
    for _, r in ages.iterrows():
        lines.append(f"- {r.cohort}, {r.sex} (n = {int(r.n)}): median {_exact(r['median'])} years (IQR, {_exact(r.q25)}-{_exact(r.q75)}); "
                     f"mean {r['mean']:.1f}; range {r['min']:.0f}-{r['max']:.0f}")
    lines += ["", "## 9. Agreement between repeat autoPET examinations (`t1_repeat_examinations.csv`, Table S4)", ""]
    for _, r in repeat.iterrows():
        scale = 100 if r.metric == "agreement" else 1
        digits = 1 if r.metric == "agreement" else 2
        lines.append(f"- {r.endpoint}, {r.metric}: {scale * r.estimate:.{digits}f} (95% CI: "
                     f"{_ci(scale * r.ci_lo, scale * r.ci_hi, digits)}); {int(r.participants)} participants, "
                     f"{int(r.scans)} examinations, {int(r.pairs)} pairs")
    lines += ["", "## 10. Corridors that submodel merging could have altered (`t1_merge_bound.csv`)", ""]
    for _, r in t["t1_merge_bound.csv"].iterrows():
        lines.append(f"- {r.grid} grid: {int(r.later_submodel_n)} of {int(r.disconnected_n):,} disconnected corridors "
                     f"({_f(r.later_submodel_percent)}%) hold a label written after the cardiac submodel "
                     f"({r.later_submodel_labels or 'none'}); {int(r.nonvascular_n)} hold any non-vascular label")
    lines += ["", "## 11. Released annotations: adjacency and separation (`t1_reference_annotations.csv`)", ""]
    for _, r in t["t1_reference_annotations.csv"].iterrows():
        if r.outcome == "separation_mm_median":
            lines.append(f"- {r.outcome}: {r.estimate:.2f} mm (IQR, {r.q25:.2f}-{r.q75:.2f}); n = {int(r.n)}")
        elif r.outcome.startswith("nonmeasurable:"):
            lines.append(f"- {r.outcome}: {int(r.events)} of {int(r.n)}")
        else:
            lines.append(f"- {r.outcome}: {int(r.events)}/{int(r.n)} ({_f(r.estimate)}%; 95% CI: "
                         f"{_ci(r.ci_lo, r.ci_hi)}; {_m(r.ci_method)})")
    lines += ["", "## 12. Paired change where projection can act, and under six-connectivity "
              "(`t1_projection_extended.csv`)", ""]
    for _, r in t["t1_projection_extended.csv"].iterrows():
        lines.append(f"- {r.adjacency}-connectivity, {r.stratum}: {int(r.net_examinations)}/{int(r.n):,} = "
                     f"{_f(r.estimate, signed=True)} points (95% CI: {_ci(r.ci_lo, r.ci_hi)}); closed "
                     f"{int(r.closed_by_projection)}, opened {int(r.opened_by_projection)}; clustered McNemar P = "
                     f"{r.p_value:.1e}")
    lines += ["", "## 13. Disconnection by grid, adjacency and cohort (`t1_inference_grid_rates.csv`)", ""]
    for _, r in t["t1_inference_grid_rates.csv"].iterrows():
        lines.append(f"- {r.grid} grid, {r.adjacency}-connectivity, {r.cohort}: {int(r.events)}/{int(r.n):,} "
                     f"({_f(r.estimate)}%; 95% CI: {_ci(r.ci_lo, r.ci_hi)}; {_m(r.ci_method)})")
    lines += ["", "## 14. Cohort-and-spacing model of disconnection by grid and adjacency "
              "(`t1_inference_grid_model.csv`)", ""]
    for _, r in t["t1_inference_grid_model.csv"].iterrows():
        if r.term != "Intercept":
            lines.append(f"- {r.grid} grid, {r.adjacency}-connectivity, {r.term}: odds ratio {r.odds_ratio:.3f} "
                         f"(95% CI: {_ci(r.or_lo, r.or_hi, 3)}); P = {r.p:.1e}")
    lines += ["", "## 15. Participants under 18 and autoPET diagnosis per study (`t1_population_sensitivity.csv`)", ""]
    for _, r in t["t1_population_sensitivity.csv"].iterrows():
        if pd.notna(r.get("estimate")):
            lines.append(f"- {r.row}: {int(r.events)}/{int(r.n):,} ({_f(r.estimate)}%; 95% CI: "
                         f"{_ci(r.ci_lo, r.ci_hi)}; {_m(r.ci_method)})")
        else:
            lines.append(f"- {r.row}: {int(r.n)}" + (f" examinations, {int(r.participants)} participants, "
                                                     f"{int(r.events)} disconnected" if pd.notna(r.get("participants"))
                                                     else ""))
    lines += ["", "## 16. SAROS source collections and autoPET (`t1_cross_dataset_overlap.csv`)", ""]
    for _, r in t["t1_cross_dataset_overlap.csv"].iterrows():
        lines.append(f"- {r['item']}: {int(r.cases)}")
    lines += ["", "## 17. Comparators beside the left iliac endpoint, disconnected examinations "
              "(`t1_comparators.csv`)", ""]
    for _, r in t["t1_comparators.csv"].iterrows():
        lines.append(f"- {r.grid} grid, {r.adjacency}-connectivity, {r.pair}, {r.cohort}: {int(r.events)}/{int(r.n):,} "
                     f"({_f(r.estimate)}%; 95% CI: {_ci(r.ci_lo, r.ci_hi)}; {_m(r.ci_method)})")
    lines += ["", "## 18. IVC to right iliac vein within each left iliac class (`t1_comparator_crosstab.csv`)", ""]
    for _, r in t["t1_comparator_crosstab.csv"].iterrows():
        lines.append(f"- {r.grid} grid, {r.adjacency}-connectivity, {r.left_class}: right iliac {r.ivc_rciv} in {int(r.n)}")
    lines += ["", "## 19. Repeat pairs that disagree, and edge or corner contacts (`t1_repeat_contacts.csv`)", ""]
    for _, r in t["t1_repeat_contacts.csv"].iterrows():
        lines.append(f"- {r.grid} grid: {int(r.discordant_pairs)} of {int(r.pairs)} pairs disagree; the connected "
                     f"examination had a face-connected path in {int(r.discordant_connected_face_connected)} and "
                     f"needed an edge or corner contact in {int(r.discordant_connected_edge_or_corner_only)} (all "
                     f"connections: {int(r.connected_edge_or_corner_only)} of {int(r.connected_examinations)})")
    lines += ["", "## 20. Empty labels in the nonmeasurable examinations (`t1_nonmeasurable_labels.csv`)", ""]
    for _, r in t["t1_nonmeasurable_labels.csv"].iterrows():
        lines.append(f"- {r.cohort}: {int(r.nonmeasurable)} of {int(r.processed)} nonmeasurable; left iliac label "
                     f"empty only {int(r.left_iliac_empty_only)}, IVC only {int(r.ivc_empty_only)}, both "
                     f"{int(r.both_empty)}; left and right iliac labels both empty {int(r.left_iliac_and_right_iliac_empty)}")
    lines += ["", "## 21. Betti numbers of the venous union, (1, 0, 0) the anatomical topology (`t1_betti.csv`)", ""]
    for _, r in t["t1_betti.csv"].iterrows():
        lines.append(f"- {r.source}, {r.stratum}, {r.outcome}, {r.cohort}: {int(r.events)}/{int(r.n):,} "
                     f"({_f(r.estimate)}%; 95% CI: {_ci(r.ci_lo, r.ci_hi)}; {_m(r.ci_method)})")
    lines += ["", "## 22. Vertebral level of the corridor, disconnected examinations (`t1_gap_level.csv`)", ""]
    for _, r in t["t1_gap_level.csv"].iterrows():
        lines.append(f"- {r.cohort}, {r.scale} {r.level}: {int(r.n)} of {int(r.valid)} with a valid ruler "
                     f"({int(r.disconnected)} disconnected)")
    lines += ["", "## 23. Disconnection by quartile of whole-label venous attenuation (`t1_attenuation_quartiles.csv`)", ""]
    for _, r in t["t1_attenuation_quartiles.csv"].iterrows():
        if r.row == "rate":
            lines.append(f"- {r.cohort}, quartile {int(r.quartile)}: {int(r.events)}/{int(r.n)} ({_f(r.estimate)}%; "
                         f"95% CI: {_ci(r.ci_lo, r.ci_hi)}; Wilson)")
        else:
            lines.append(f"- lowest quartile against the other three, cohort-stratified Mantel-Haenszel odds ratio "
                         f"{_f(r.estimate, 2)} (95% CI: {_ci(r.ci_lo, r.ci_hi, 2)}; P = {r.p_value:.2g})")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--out", type=_pl.Path, default=OUT)
    args = parser.parse_args(argv)
    paired, contingency = paired_projection(args.n_boot)
    labels, frequency, location, arteries = corridor_labels(args.n_boot)
    comparators, crosstab = comparator_reachability(args.n_boot)
    tables = {"t1_paired_projection.csv": paired, "t1_paired_participant_contingency.csv": contingency,
              "t1_median_intervals.csv": median_intervals(args.n_boot), "t1_corridor_fraction.csv": corridor_fraction(),
              "t1_bottleneck_fixed.csv": bottleneck_fixed(args.n_boot), "t1_corridor_labels.csv": labels,
              "t1_corridor_label_frequency.csv": frequency, "t1_corridor_location.csv": location,
              "t1_corridor_arterial_labels.csv": arteries,
              "t1_saros_region.csv": saros_region(args.n_boot), "t1_contrast_metadata.csv": contrast_metadata(args.n_boot),
              "t1_age_by_sex.csv": age_by_sex(), "t1_repeat_examinations.csv": repeat_examinations(),
              "t1_merge_bound.csv": merge_bound(), "t1_reference_annotations.csv": reference_annotations(args.n_boot),
              "t1_projection_extended.csv": projection_extended(args.n_boot),
              "t1_inference_grid_rates.csv": inference_grid_rates(args.n_boot),
              "t1_inference_grid_model.csv": inference_grid_model(),
              "t1_population_sensitivity.csv": population_sensitivity(args.n_boot),
              "t1_cross_dataset_overlap.csv": cross_dataset_overlap(),
              "t1_comparators.csv": comparators, "t1_comparator_crosstab.csv": crosstab,
              "t1_repeat_contacts.csv": repeat_contacts(), "t1_nonmeasurable_labels.csv": nonmeasurable_labels(),
              "t1_betti.csv": betti_rates(args.n_boot), "t1_gap_level.csv": gap_levels(),
              "t1_attenuation_quartiles.csv": attenuation_quartiles()}
    _check(set(_DIGESTS) == set(INPUTS), "every declared input was read and hashed")
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = {name: _publish(args.out / name, frame.to_csv(index=False, float_format="%.10g").encode("utf-8"))
               for name, frame in tables.items()}   # %.10g keeps P values like 1e-43
    summary = summary_markdown(tables)
    outputs["RESULTS_SUMMARY.md"] = _publish(args.out / "RESULTS_SUMMARY.md", summary.encode("utf-8"))
    producer = _pl.Path(__file__).resolve()
    record = {"producer": producer.relative_to(ROOT).as_posix(),
              "producer_sha256": hashlib.sha256(producer.read_bytes()).hexdigest(), "n_boot": args.n_boot,
              "seeds": {"topology": SEED_TOPOLOGY, "grid": SEED_GRID},
              "python": platform.python_version(),
              "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy",
                                                                               "statsmodels")},
              "inputs": {name: {"path": INPUTS[name].relative_to(ROOT).as_posix(), "sha256": _DIGESTS[name]}
                         for name in INPUTS},
              "outputs": outputs}
    _publish(args.out / "run_record.json", (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(summary)


if __name__ == "__main__":
    main()
