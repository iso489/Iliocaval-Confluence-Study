import hashlib
import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

_PATH = pathlib.Path(__file__).with_name("summary_tables.py")
_SPEC = importlib.util.spec_from_file_location("summary_tables", _PATH)
RA = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(RA)


def _sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def test_weighted_median_equals_numpy_median_of_the_expanded_sample():
    rng = np.random.default_rng(20261103)
    for _ in range(500):
        n = int(rng.integers(1, 12))
        values = np.sort(rng.normal(size=n).round(2))
        weights = rng.integers(0, 4, n)
        if weights.sum() == 0:
            weights[int(rng.integers(0, n))] = 1
        assert RA.weighted_median(values, weights) == pytest.approx(np.median(np.repeat(values, weights)))


def test_median_interval_point_estimate_and_resampling_bounds():
    # One participant brings three examinations; every resampled median stays inside the data.
    frame = pd.DataFrame({"cohort": ["A"] * 5 + ["B"] * 4,
                          "participant": ["a1", "a1", "a1", "a2", "a3", "b1", "b2", "b3", "b4"],
                          "x": [1.0, 2.0, 3.0, 10.0, 11.0, 4.0, 5.0, 6.0, 7.0]})
    median, low, high = RA.median_interval(frame, "x", seed=1, n_boot=500)
    assert median == np.median(frame.x)
    assert frame.x.min() <= low <= median <= high <= frame.x.max()


def test_durkalski_reduces_to_uncorrected_mcnemar_with_one_pair_per_participant():
    before = np.array([1, 0, 0, 1, 0, 1, 0, 0], bool)
    after = np.array([1, 1, 1, 0, 1, 1, 0, 1], bool)
    b, c = int((after & ~before).sum()), int((before & ~after).sum())
    out = RA.durkalski_test([f"p{i}" for i in range(8)], before, after)
    assert out["statistic"] == pytest.approx((b - c) ** 2 / (b + c))
    assert (out["b"], out["c"]) == (b, c)


def test_durkalski_weights_each_participant_by_its_number_of_pairs():
    participants = ["p1", "p1", "p2", "p3", "p3", "p3", "p3"]
    before = np.array([0, 0, 1, 0, 0, 0, 0], bool)
    after = np.array([1, 1, 0, 1, 0, 0, 0], bool)
    d = np.array([1.0, -1.0, 0.25])
    assert RA.durkalski_test(participants, before, after)["statistic"] == pytest.approx(d.sum() ** 2 / (d ** 2).sum())


def test_durkalski_holds_its_size_under_a_clustered_null():
    # Discordance direction shares a participant effect but is symmetric overall (H0 true).
    rng = np.random.default_rng(7)
    trials, rejections = 400, 0
    for _ in range(trials):
        sizes = rng.integers(1, 4, 150)
        participants = np.repeat(np.arange(150), sizes)
        tilt = rng.normal(0, 1, 150)[participants]
        discordant = rng.random(len(participants)) < 0.3
        upward = rng.random(len(participants)) < 1 / (1 + np.exp(-tilt))
        rejections += RA.durkalski_test(participants, discordant & ~upward, discordant & upward)["p_value"] < 0.05
    assert 0.02 <= rejections / trials <= 0.09


@pytest.mark.needs_metadata
def test_every_analysis_reproduces_its_published_controls_and_pins_its_counts():
    # Each analysis raises on any departure from the published figures it is bound to.
    paired, contingency = RA.paired_projection(n_boot=50)
    measurable = paired[paired.row == "paired_change:measurable"].iloc[0]
    assert (measurable.n, measurable.net_examinations) == (1873, 203)
    assert contingency[["bk", "ck"]].sum().tolist() == [205, 2] and len(contingency) == 1759
    assert paired[paired.row == "clustered_mcnemar:durkalski_2003"].iloc[0].p_value < 1e-3
    assert RA.corridor_fraction().n.tolist() == [892, 1095]
    bottleneck = RA.bottleneck_fixed(n_boot=50)
    assert bottleneck[bottleneck.outcome == "bottleneck_radius<=1.5mm"].n.tolist() == [981, 778]
    labels, _, location, arteries = RA.corridor_labels(n_boot=50)
    assert labels.n.tolist() == [892, 892, 1095, 1095] and location.disconnected_n.tolist() == [892, 1095]
    assert arteries.arterial_corridors.tolist() == [77] * 3 + [73] * 3   # the published arterial counts
    assert arteries[arteries.label == "iliac_artery_right"].examinations.tolist() == [59, 57]
    assert RA.saros_region(n_boot=50).set_index("region").n.to_dict() == {"all": 366, "abdomen": 244, "wholebody": 122}
    assert RA.contrast_metadata(n_boot=50).set_index("contrast_description").n.to_dict() == \
        {"all": 1014, "with": 953, "without": 61}
    assert RA.age_by_sex().set_index(["cohort", "sex"]).n.to_dict() == {
        ("AMOS", "female"): 184, ("AMOS", "male"): 309, ("AMOS", "all"): 493,
        ("autoPET", "female"): 396, ("autoPET", "male"): 500, ("autoPET", "all"): 900}
    assert len(RA.median_intervals(n_boot=50)) == 5
    assert len(RA.repeat_examinations()) == 6
    bound = RA.merge_bound()
    assert bound.nonvascular_n.tolist() == [94, 90] and bound.later_submodel_n.tolist() == [2, 3]
    assert bound.later_submodel_labels.tolist() == ["iliopsoas_left"] * 2
    reference = RA.reference_annotations(n_boot=50).set_index("outcome")
    assert reference.loc[["disconnected_26", "disconnected_6", "connected_26_disconnected_6",
                          "separation_le_1_slice_unit"], ["events", "n"]].to_numpy().tolist() == \
        [[326, 600], [361, 600], [35, 274], [26, 326]]
    assert reference.loc["separation_mm_median", "n"] == 326
    absent = [f"nonmeasurable:{name}" for name in (
        "left_iliac_label_only_absent", "ivc_label_only_absent", "both_labels_absent", "head_neck_chest_or_heart_study",
        "abdomen_pelvis_legs_or_whole_body_study", "study_description_unplaced")]
    assert reference.loc[absent, ["events", "n"]].to_numpy().tolist() == \
        [[377, 628], [36, 628], [215, 628], [393, 628], [161, 628], [74, 628]]
    extended = RA.projection_extended(n_boot=50).set_index(["adjacency", "stratum"])
    assert extended.loc[("26", "spacing_gt1p5mm"), ["n", "net_examinations"]].tolist() == [1597, 203]
    assert extended.loc[("6", "all"), ["closed_by_projection", "opened_by_projection"]].tolist() == [390, 8]
    assert extended.loc[("6", "spacing_le1p5mm"), "net_examinations"] == 0
    rates = RA.inference_grid_rates(n_boot=50).set_index(["grid", "adjacency", "cohort"]).events
    assert (rates[("inference", "26", "pooled")], rates[("inference", "6", "pooled")]) == (1095, 1492)
    model = RA.inference_grid_model()
    assert len(model) == 20 and set(zip(model.grid, model.adjacency)) == {
        ("native", "26"), ("inference", "26"), ("native", "6"), ("inference", "6")}
    population = RA.population_sensitivity(n_boot=50).set_index("row")
    assert population.loc[[f"diagnosis_per_study:{d}" for d in ("LUNG_CANCER", "LYMPHOMA", "MELANOMA", "NEGATIVE")],
                          "n"].tolist() == [168, 145, 188, 513]
    assert (population.loc["under_18:autoPET", "n"], population.loc["without_under_18:pooled", "n"]) == (4, 1869)
    overlap = RA.cross_dataset_overlap().set_index("item").cases
    assert overlap["saros_collections"] == 20 and overlap[[
        "saros_cases_in_autopet_collection", "study_uid_shared_with_autopet",
        "series_uid_shared_with_autopet"]].tolist() == [0, 0, 0]
    comparators, crosstab = RA.comparator_reachability(n_boot=50)
    pooled = comparators[comparators.cohort == "pooled"].set_index(["grid", "adjacency", "pair"])[["events", "n"]]
    # Over the 1,873 measurable examinations; the comparator table's own measurable rows would give 866/1,873.
    assert pooled.loc[[("native", "26", pair) for pair in RA.COMPARATORS]].to_numpy().tolist() == \
        [[865, 1871], [130, 1873], [114, 1873]]
    assert pooled.loc[[("inference", "26", pair) for pair in RA.COMPARATORS]].to_numpy().tolist() == \
        [[1107, 1871], [139, 1873], [119, 1873]]
    assert pooled.loc[[("native", "6", "ivc_rciv"), ("inference", "6", "ivc_rciv")]].to_numpy().tolist() == \
        [[1044, 1871], [1508, 1871]]
    cohorts = comparators[(comparators.grid == "native") & (comparators.adjacency == "26")
                          & (comparators.pair == "ivc_rciv")].set_index("cohort")
    assert cohorts.loc[["AMOS", "SAROS", "autoPET"], ["events", "n"]].to_numpy().tolist() == \
        [[309, 493], [177, 364], [379, 1014]]
    cross = crosstab.set_index(["grid", "adjacency", "left_class", "ivc_rciv"]).n
    assert (cross[("native", "26", "disconnected", "disconnected")], cross[("inference", "26", "disconnected", "disconnected")]) \
        == (799, 1040)
    assert cross.groupby(level=["grid", "adjacency"]).sum().eq(1873).all()   # each measurable examination once
    contacts = RA.repeat_contacts().set_index("grid")[
        ["pairs", "discordant_pairs", "discordant_connected_face_connected", "discordant_connected_edge_or_corner_only"]]
    assert contacts.to_numpy().tolist() == [[161, 34, 24, 10], [161, 36, 11, 25]]   # native, inference
    empty = RA.nonmeasurable_labels().set_index("cohort")[
        ["nonmeasurable", "left_iliac_empty_only", "ivc_empty_only", "both_empty", "left_iliac_and_right_iliac_empty"]]
    assert empty.loc["SAROS"].tolist() == [26, 26, 0, 0, 24] and empty.loc["pooled"].tolist() == [26, 26, 0, 0, 24]
    betti = RA.betti_rates(n_boot=50).set_index(["source", "stratum", "outcome", "cohort"])
    assert betti.loc[("predictions", "all", "anatomical_topology", "pooled"), ["events", "n"]].tolist() == [156, 1873]
    assert betti.loc[("predictions", "connected", "anatomical_topology", "pooled"), ["events", "n"]].tolist() == [156, 981]
    assert betti.loc[[("predictions", "all", o, "pooled") for o in ("b0_gt1", "b1_gt0", "b2_gt0")], "events"].tolist() \
        == [977, 989, 346]
    assert betti.loc[("released_annotations", "all", "anatomical_topology", "all"), ["events", "n"]].tolist() == [96, 600]
    levels = RA.gap_levels()
    bins = levels[(levels.cohort == "pooled") & (levels.scale == "bin")].set_index("level").n
    assert (bins["L4"], bins["L5"], int(levels[levels.cohort == "pooled"].valid.iloc[0])) == (454, 335, 872)
    attenuation = RA.attenuation_quartiles()
    rates = attenuation[attenuation.row == "rate"].set_index(["cohort", "quartile"])
    assert rates.loc[[("pooled", q) for q in (1, 2, 3, 4)], ["events", "n"]].to_numpy().tolist() == \
        [[327, 457], [192, 465], [186, 476], [187, 475]]
    mh = attenuation[attenuation.row != "rate"].iloc[0]
    assert (round(mh.estimate, 2), round(mh.ci_lo, 2), round(mh.ci_hi, 2)) == (2.65, 2.05, 3.42) and mh.p_value < 1e-10


def test_discordant_contacts_counts_each_unordered_pair_once_and_reads_the_connected_member():
    groups = [[("direct", "direct"), ("disconnected", "disconnected"), ("right_iliac_mediated", "disconnected")],
              [("disconnected", "disconnected")], []]
    # pairs: direct/disconnected disagree, face-connected; direct/mediated agree; disconnected/mediated
    # disagree with an edge or corner path; the singleton and the empty group give none
    assert RA.discordant_contacts(groups) == (3, 2, 1)
    assert RA.discordant_contacts([[("disconnected", "disconnected"), ("direct", "disconnected")]]) == (1, 1, 0)


def test_the_study_description_rule_places_only_descriptions_that_name_a_region():
    import re

    def place(text):
        if re.search(RA.LOWER_BODY_STUDY, text, re.I):
            return "lower"
        return "upper" if re.search(RA.UPPER_BODY_STUDY, text, re.I) else "unplaced"
    assert [place(t) for t in ("ct heart-thorakale aorta", "ct gesichtshead", "ct angiography neck",
                               "ct thorax-abdomen-pelvis", "ct angiography neck-thx-abd-pelvis-leg", "ct polytrauma",
                               "ct spine", "ct hip right", "ct  intervention ", "")] == \
        ["upper", "upper", "upper", "lower", "lower", "lower", "unplaced", "unplaced", "unplaced", "unplaced"]


def test_submodels_partition_the_117_total_task_classes_in_write_order():
    assert [name for name, _ in RA.SUBMODELS] == ["organs", "vertebrae", "cardiac", "muscles", "ribs"]
    sizes = [len(labels) for _, labels in RA.SUBMODELS]
    assert sizes == [24, 26, 18, 23, 26]
    assert len(set().union(*(labels for _, labels in RA.SUBMODELS))) == sum(sizes) == 117   # disjoint
    assert dict(RA.SUBMODELS)["cardiac"] == RA.VASCULAR
    assert RA.WRITTEN_AFTER_CARDIAC == {"muscles", "ribs"}


def test_corridor_exposure_counts_only_labels_written_after_the_cardiac_submodel():
    assert RA.corridor_exposure({}) == (False, False)
    assert RA.corridor_exposure({"aorta": 4, "iliac_vena_right": 2}) == (False, False)
    assert RA.corridor_exposure({"vertebrae_L4": 9, "liver": 1}) == (True, False)   # written before the veins
    assert RA.corridor_exposure({"vertebrae_L4": 9, "iliopsoas_left": 1}) == (True, True)
    assert RA.corridor_exposure({"rib_right_12": 3}) == (True, True)
    assert RA.corridor_exposure({"iliopsoas_left": 0, "vertebrae_L5": 2}) == (True, False)   # no voxels: absent
    with pytest.raises(ValueError, match="not a class of exactly one"):
        RA.corridor_exposure({"lung_vessels": 1})


@pytest.mark.needs_metadata
def test_main_writes_every_table_and_the_summary(tmp_path):
    RA.main(["--n-boot", "50", "--out", str(tmp_path)])
    names = {path.name for path in tmp_path.iterdir()}
    assert names == set(json.loads((RA.OUT / "run_record.json").read_text(encoding="utf-8"))["outputs"]) | {"run_record.json"}
    summary = (tmp_path / "RESULTS_SUMMARY.md").read_text(encoding="utf-8")
    assert "- native grid: 2 of 892 disconnected corridors (0.2%) hold a label written after the cardiac submodel " \
           "(iliopsoas_left); 94 hold any non-vascular label" in summary
    assert "- separation_mm_median: 6.36 mm (IQR, 4.24-11.95); n = 326" in summary
    assert "- disconnected_6: 361/600 (60.2%; 95% CI: 56.2, 64.0; Wilson)" in summary
    assert "- 6-connectivity, all: 382/1,873 = +20.4 points (95% CI:" in summary
    assert "- under_18:autoPET: 4 examinations, 4 participants, 3 disconnected" in summary
    assert "- native grid, 26-connectivity, ivc_rciv, pooled: 865/1,871 (46.2%; 95% CI:" in summary
    assert "- native grid, 26-connectivity, disconnected: right iliac disconnected in 799" in summary
    assert "- native grid: 34 of 161 pairs disagree; the connected examination had a face-connected path in 24 " \
           "and needed an edge or corner contact in 10 (all connections: 218 of 981)" in summary
    assert "- SAROS: 26 of 392 nonmeasurable; left iliac label empty only 26, IVC only 0, both 0; " \
           "left and right iliac labels both empty 24" in summary
    assert "- predictions, all, anatomical_topology, pooled: 156/1,873 (8.3%; 95% CI:" in summary
    assert "- pooled, bin L4: 454 of 872 with a valid ruler (892 disconnected)" in summary
    assert "cohort-stratified Mantel-Haenszel odds ratio 2.65 (95% CI: 2.05, 3.42; P = " in summary


@pytest.mark.needs_metadata
def test_committed_outputs_were_produced_by_this_code_from_these_inputs():
    record = json.loads((RA.OUT / "run_record.json").read_text(encoding="utf-8"))
    assert record["producer_sha256"] == _sha(_PATH)
    assert record["n_boot"] == RA.N_BOOT
    for name, entry in record["inputs"].items():
        assert entry["sha256"] == _sha(RA.ROOT / entry["path"]), name
    for name, digest in record["outputs"].items():
        assert digest == _sha(RA.OUT / name), name
