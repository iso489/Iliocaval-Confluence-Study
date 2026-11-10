#!/usr/bin/env python3
"""95% intervals for the class-score resampling experiment, by the paper's own method.

    python class_score_intervals.py [FOLDER] [--no-controls]

Reads class_score_cases.csv in FOLDER (default: Results/phase1_characterize/class_score_resampling) and writes class_score_intervals.csv
beside it. Prevalences and paired changes use connectivity_statistics.proportion_intervals: 10,000
cohort-stratified participant-block resamples, percentile intervals, the seed of the paired-change
analysis in the manuscript. A paired change is after minus before in the same examinations.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for folder in sorted(ROOT.iterdir()):
    if folder.is_dir() and folder.name[0] not in "._" and str(folder) not in sys.path:
        sys.path.append(str(folder))
import connectivity_statistics as FS  # noqa: E402

N_BOOT, SEED = 10000, 20261025
ARMS = ("inference", "nearest", "linear", "slab")
CHANGES = (("inference", "nearest"), ("linear", "nearest"), ("slab", "nearest"),
           ("linear", "inference"), ("slab", "inference"))
STRATA = ("spacing_le_1p5mm", "spacing_gt1p5_to3mm", "spacing_gt3mm")


def main(argv):
    controls = "--no-controls" not in argv
    named = [a for a in argv if not a.startswith("--")]
    folder = pathlib.Path(named[0]) if named else ROOT.parent / "Results" / "phase1_characterize" / "class_score_resampling"
    cases = pd.read_csv(folder / "class_score_cases.csv", dtype={"participant": str, "case": str})
    frame = cases[cases.nearest_class26 != "nonmeasurable"].reset_index(drop=True)
    if controls and (len(cases), len(frame)) != (1899, 1873):
        raise SystemExit(f"expected 1,899 examinations, 1,873 measurable; found {len(cases)}, {len(frame)}")
    rows = []
    for group in ("all", *STRATA):
        keep = np.ones(len(frame), bool) if group == "all" else frame.stratum.eq(group).to_numpy()
        part = frame[keep]
        for rule in ("26", "6"):
            disconnected = {arm: frame[f"{arm}_class{rule}"].eq("disconnected").to_numpy()[keep] for arm in ARMS}
            base = {"group": group, "adjacency": rule}
            for arm in ARMS:
                s = FS.proportion_intervals(part, disconnected[arm], N_BOOT, False, SEED)
                rows.append({**base, "kind": "prevalence", "quantity": arm, "n": int(s["n"]), "count": int(s["events"]),
                             "estimate": 100 * s["proportion"], "ci_lo": 100 * s["ci_lo"], "ci_hi": 100 * s["ci_hi"],
                             "participants": int(s["participants"]), "ci_method": s["ci_method"]})
            for after, before in CHANGES:
                change = disconnected[after].astype(int) - disconnected[before].astype(int)
                s = FS.proportion_intervals(part, change, N_BOOT, False, SEED)
                rows.append({**base, "kind": "paired_change", "quantity": f"{after}_minus_{before}", "n": int(s["n"]),
                             "count": int(disconnected[after].sum() - disconnected[before].sum()),
                             "estimate": 100 * s["proportion"], "ci_lo": 100 * s["ci_lo"], "ci_hi": 100 * s["ci_hi"],
                             "participants": int(s["participants"]), "ci_method": s["ci_method"]})
    table = pd.DataFrame(rows)
    if controls:
        line = table.query("group == 'all' and adjacency == '26' and quantity == 'inference_minus_nearest'").iloc[0]
        if (round(line.estimate, 1), line["count"]) != (10.8, 203) or not 9.0 < line.ci_lo < line.ci_hi < 12.8:
            raise SystemExit(f"control failed: the manuscript's paired change is not reproduced: {line.to_dict()}")
    table.to_csv(folder / "class_score_intervals.csv", index=False)
    pd.set_option("display.width", 200, "display.max_rows", 200)
    shown = table[["group", "adjacency", "kind", "quantity", "n", "count", "estimate", "ci_lo", "ci_hi"]].round(1)
    print(shown.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
