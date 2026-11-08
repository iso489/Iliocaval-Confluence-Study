#!/usr/bin/env python3
"""Deterministically normalize the released AMOS metadata identifier typo.

The immutable upstream snapshot has two distinct rows with ``amos_id=500`` and
no row 600. The CT/Optima record is the real case 0500; the MRI/Ingenia record is
case 0600. This script applies that one asserted correction and nothing else.
"""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RAW = HERE / "raw" / "AMOS_upstream.csv"
DEFAULT_OUT = HERE / "AMOS.csv"


def normalized_text(raw_path=DEFAULT_RAW):
    with Path(raw_path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or "amos_id" not in fieldnames:
        raise ValueError("upstream AMOS metadata lacks amos_id")
    ids = [row["amos_id"] for row in rows]
    if len(rows) != 600 or ids.count("500") != 2 or "600" in ids:
        raise ValueError("unexpected upstream AMOS identity contract; refusing normalization")
    candidates = [row for row in rows
                  if row["amos_id"] == "500"
                  and row.get("Manufacturer") == "Philips"
                  and row.get("Manufacturer's Model Name") == "Ingenia"
                  and row.get("Site") == "center"]
    if len(candidates) != 1:
        raise ValueError("cannot uniquely identify the duplicated MRI case 0600 row")
    candidates[0]["amos_id"] = "600"
    normalized_ids = [row["amos_id"] for row in rows]
    if len(set(normalized_ids)) != 600 or set(normalized_ids) != {str(i) for i in range(1, 601)}:
        raise ValueError("normalized AMOS ids are not exactly 1..600")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true",
                    help="verify OUT is exactly the deterministic normalization")
    args = ap.parse_args(argv)
    expected = normalized_text(args.raw)
    if args.check:
        if not args.out.is_file() or args.out.read_text(encoding="utf-8") != expected:
            raise SystemExit(f"normalized AMOS metadata differs from the deterministic normalization: {args.out}")
        print(f"OK: {args.out}")
        return 0
    args.out.write_text(expected, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
