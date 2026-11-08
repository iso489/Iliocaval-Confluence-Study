#!/usr/bin/env python3
"""Compute raw venous-union topology for measurable cases that have no row in the paired
(collapsed-class) diagnostic.

Run before connectivity_statistics.py. Reads only the measurable cases absent from the paired
diagnostic. All masks must match the local export's SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
import stats as ST
from measure_collapsed_gap import FILES, load_masks, union_connectivity

ROOT = ST.REPO_ROOT
DEFAULT_OUT = ST.RESULTS / "connectivity" / "raw_connectivity_recovered.csv"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recover(mask_root, out):
    mask_root, out = Path(mask_root), Path(out)
    manifest = pd.read_csv(mask_root / "MANIFEST.csv")
    keys = ["dataset", "case", "structure"]
    if manifest.duplicated(keys).any():
        raise ValueError("duplicate local mask manifest key")
    manifest = manifest.set_index(keys)
    recovered = []
    for ds, (dirname, prefix, _) in ST.LAYOUT.items():
        directory = ST.RESULTS / dirname
        full_path = directory / f"{prefix}_results_fullres.txt"
        ST.validate_result_roster(ds, full_path)
        full, counts = ST.load_fullres(full_path)
        ST.require_surface_metric(
            counts, full_path, consumer="raw connectivity recovery"
        )
        paired, pair_counts = ST.load_collapsed(
            directory / f"{prefix}_results_collapsed.txt"
        )
        if pair_counts["unparsed"]:
            raise ValueError("malformed paired topology input")
        pair_ids = {r[0] for r in paired}
        for case, *_ in full:
            if case in pair_ids:
                continue
            seg = mask_root / dirname / case
            record = {
                "dataset": dirname,
                "case": case,
                "source": "raw_masks_sha256_verified",
            }
            for cls, filename in FILES.items():
                path = seg / filename
                digest = sha256(
                    path
                )  # a missing mask is an explicit failure, never a zero
                expected = manifest.loc[
                    (dirname, case, filename.removesuffix(".nii.gz"))
                ]
                if digest != expected.sha256:
                    raise ValueError(f"mask digest mismatch: {path}")
                record[f"{cls}_sha256"] = digest
            masks, ref, _ = load_masks(seg)
            if ref is None or not masks["ivc"].any() or not masks["lciv"].any():
                raise ValueError(
                    f"measurable case has empty primary mask: {dirname}/{case}"
                )
            ncomp, connected = union_connectivity(masks)
            _, direct = union_connectivity({k: masks[k] for k in ("ivc", "lciv")})
            record.update(
                union_ncomp=ncomp,
                union_connected=int(connected),
                direct_connected=int(direct),
                rciv_voxels=int(masks["rciv"].sum()),
            )
            recovered.append(record)
    if not recovered:
        raise ValueError("no raw topology requires recovery")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recovered).to_csv(out, index=False)
    print(f"Recovered {len(recovered)} cases -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mask-root", type=Path, default=ROOT / "Local masks")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    recover(args.mask_root, args.out)
