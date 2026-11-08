#!/usr/bin/env python3
"""    python Code/phase1_characterize/summary/reference_arterial.py --cache VENOUS_CACHE \
        --arterial-cache FOLDER [--fetch] [--workers 2]

The venous audit (connectivity_statistics --refresh-reference-only) read the IVC and iliac vein masks of
the 1,228 released v2.0.1 cases (Zenodo record 10047292). This script reads the aorta and both common
iliac artery masks of the same cases and applies the comparators' direct-path rule
(confluence_comparators.reach, applied to the binary masks, which may share voxels): the largest
26-connected component of each label is an endpoint, and a path is sought within the union of the two
labels under 26- and 6-connectivity; a pair with an empty label is nonmeasurable.

The archive's central directory and released metadata are read from the venous cache, and each must
have the SHA-256 the venous audit recorded (supplement_manifest.json). --fetch downloads only the
three arterial masks of each case, by HTTP range requests (export_relevant_masks.get_range: ranged,
rate-limited, length-checked), never a CT; each file must match its ZIP entry's length and CRC-32
before it is written, and files already present and valid are kept. Every run reads each mask once,
hashes those bytes and parses the same bytes.

Outputs, in Results/phase1_characterize/summary/: reference_arterial_cases.csv (one row per
case), t1_reference_arterial.csv (disconnection by side, adjacency rule and split, in every case with
both labels and in the 600 cases the venous audit measured; Wilson intervals, as for the venous audit)
and reference_arterial_run_record.json.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pathlib as _pl
import platform
import struct
import sys as _sys
import types
import zlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

for _p in sorted(_pl.Path(__file__).resolve().parents[2].iterdir()):
    if _p.is_dir() and _p.name[0] not in "._" and str(_p) not in _sys.path:
        _sys.path.append(str(_p))  # append, never shadow stdlib

import export_relevant_masks as EM
import gap_config as CFG
import gap_content_analysis as GC
import numpy as np
import pandas as pd
import statlib as S

HERE = _pl.Path(__file__).resolve().parent
PRODUCER = _pl.Path(__file__).resolve()
PRODUCER_SHA256 = hashlib.sha256(PRODUCER.read_bytes()).hexdigest()   # stamped when the run starts
# The comparator rule runs from the very bytes whose digest the run record carries.
_CC_PATH = HERE / "confluence_comparators.py"
_CC_SOURCE = _CC_PATH.read_bytes()
CC_SHA256 = hashlib.sha256(_CC_SOURCE).hexdigest()
CC = types.ModuleType("confluence_comparators")
CC.__file__ = str(_CC_PATH)
exec(compile(_CC_SOURCE, str(_CC_PATH), "exec"), CC.__dict__)

ROOT = HERE.parents[2]
OUT = ROOT / "Results" / "phase1_characterize" / "summary"
VENOUS = ROOT / "Results" / "phase1_characterize" / "connectivity"
ARTERIAL = ("aorta", "iliac_artery_left", "iliac_artery_right")
SIDES = {"left": "iliac_artery_left", "right": "iliac_artery_right"}
RULES = ("26", "6")


def parse_directory(directory):
    """ZIP central-directory records, read as export_relevant_masks.archive_index reads them."""
    entries, offset = [], 0
    while offset < len(directory):
        f = struct.unpack("<4s6H3L5H2L", directory[offset:offset + 46])
        if f[0] != b"PK\x01\x02":
            raise ValueError("invalid central directory entry")
        nl, el, cl = f[10:13]
        name = directory[offset + 46:offset + 46 + nl].decode("utf-8")
        extra = directory[offset + 46 + nl:offset + 46 + nl + el]
        comp, uncomp, local = f[8], f[9], f[16]
        p = 0
        while p + 4 <= len(extra):
            tag, length = struct.unpack("<HH", extra[p:p + 4])
            body, p = extra[p + 4:p + 4 + length], p + 4 + length
            if tag == 1:
                q = 0
                if uncomp == 0xFFFFFFFF:
                    uncomp, q = struct.unpack_from("<Q", body, q)[0], q + 8
                if comp == 0xFFFFFFFF:
                    comp, q = struct.unpack_from("<Q", body, q)[0], q + 8
                if local == 0xFFFFFFFF:
                    local = struct.unpack_from("<Q", body, q)[0]
        entries.append({"name": name, "method": f[4], "crc32": f[7], "compressed_size": comp,
                        "uncompressed_size": uncomp, "offset": local})
        offset += 46 + nl + el + cl
    return entries


def arterial_entries(entries):
    """{case: [entry of each arterial mask, in archive order]}, each entry with its directory index."""
    wanted = {f"{name}.nii.gz" for name in ARTERIAL}
    cases = {}
    for i, entry in enumerate(entries):
        case, _, rest = entry["name"].partition("/")
        if rest.startswith("segmentations/") and rest.split("/", 1)[1] in wanted:
            cases.setdefault(case, []).append({**entry, "index": i})
    if any(len(group) != len(ARTERIAL) for group in cases.values()):
        raise ValueError("a released case lacks one of the three arterial masks")
    return cases


def _valid(raw, entry):
    return len(raw) == entry["uncompressed_size"] and zlib.crc32(raw) & 0xFFFFFFFF == entry["crc32"]


def _write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_bytes(raw)
    os.replace(part, path)


def fetch_case(task):
    """Download a case's arterial masks that are absent or invalid; consecutive archive entries share
    one range request. Returns the number of bytes received."""
    cache, group = task
    slack = 64      # bytes requested past a local header's name for its extra field
    missing = [e for e in group if not ((cache / e["name"]).is_file() and _valid((cache / e["name"]).read_bytes(), e))]
    runs, received = [], 0
    for entry in sorted(missing, key=lambda e: e["index"]):
        if runs and entry["index"] == runs[-1][-1]["index"] + 1:
            runs[-1].append(entry)
        else:
            runs.append([entry])
    for run in runs:
        start = run[0]["offset"]
        end = run[-1]["offset"] + 30 + len(run[-1]["name"].encode()) + slack + run[-1]["compressed_size"]
        blob = EM.get_range(start, end - 1)
        received += len(blob)
        for entry in run:
            i = entry["offset"] - start
            header = struct.unpack("<4s5H3L2H", blob[i:i + 30])
            if header[0] != b"PK\x03\x04" or header[3] != entry["method"]:
                raise ValueError(f"invalid local entry: {entry['name']}")
            j = i + 30 + header[-2] + header[-1]
            if j + entry["compressed_size"] > len(blob):      # a local extra field longer than the slack
                more = EM.get_range(start + len(blob), start + j + entry["compressed_size"] - 1)
                received += len(more)
                blob += more
            packed = blob[j:j + entry["compressed_size"]]
            raw = zlib.decompress(packed, -15) if entry["method"] == 8 else packed
            if not _valid(raw, entry):
                raise ValueError(f"ZIP entry failed length/CRC32 verification: {entry['name']}")
            _write(cache / entry["name"], raw)
    return received


def grid_offset(shape, first, second):
    """Largest distance, in mm, between the positions two affines give the corner voxels of one grid."""
    corners = np.array([[i, j, k, 1.0] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1) for k in (0, shape[2] - 1)])
    return float(np.abs(corners @ np.asarray(first).T - corners @ np.asarray(second).T)[:, :3].max())


def reach_masks(first, second):
    """confluence_comparators.reach on two binary masks: whether their largest 26-connected components
    share a component of the union. The released masks are separate files that may share voxels, which
    a label volume could not hold; on masks that do not overlap this is the comparators' rule exactly."""
    if not first.any() or not second.any():
        return {rule: "nonmeasurable" for rule in CC.ADJACENCY}
    head, _, _ = CC.MCG.largest_component26(first)
    tail, _, _ = CC.MCG.largest_component26(second)
    out = {}
    for rule, structure in CC.ADJACENCY.items():
        labels, _ = CC.ndi.label(first | second, structure)
        out[rule] = "connected" if CC.MCG.share_component(labels, head.astype(bool), tail.astype(bool)) else "disconnected"
    return out


def audit_case(task):
    """Presence and direct-path classes of the aorta with each common iliac artery, for one case.

    The released masks of a case share one voxel array, but their stored affines can differ by header
    rounding (an aorta rotation term in the fifth significant digit); the masks are taken as one grid
    when their shapes agree and every corner voxel lies within a quarter of a voxel under both affines.
    Voxels shared by two arterial masks are counted, not refused."""
    cache, case = task
    masks, shape, affine, row = {}, None, None, {"case": case, "grid_offset_mm": 0.0}
    for name in ARTERIAL:
        data, digest = GC.read_hashed(cache / case / "segmentations" / f"{name}.nii.gz")
        image = GC.nifti_from_bytes(data, f"{name}.nii.gz")
        if shape is not None:
            offset = grid_offset(shape, image.affine, affine) if image.shape == shape else np.inf
            if offset > 0.25 * np.linalg.norm(np.asarray(affine)[:3, :3], axis=0).min():   # voxel sizes
                raise ValueError(f"{case}: arterial mask grids differ")
            row["grid_offset_mm"] = max(row["grid_offset_mm"], offset)
        shape, affine = image.shape, image.affine
        arr = np.asanyarray(image.dataobj)
        if arr.dtype.kind not in "bui" or arr.min() < 0 or arr.max() > 1:
            raise ValueError(f"{case}: {name} is not a binary integer mask")
        masks[name] = arr.astype(bool)
        row[f"{name}_present"] = int(masks[name].any())
        row[f"{name}_sha256"] = digest
    row["slice_mm"] = float(np.linalg.norm(np.asarray(affine)[:3, 2]))   # reachability needs no orthogonal grid
    row["shared_voxels"] = int(sum((masks[a] & masks[b]).sum() for a, b in
                                   (("aorta", "iliac_artery_left"), ("aorta", "iliac_artery_right"),
                                    ("iliac_artery_left", "iliac_artery_right"))))
    union = masks["aorta"] | masks["iliac_artery_left"] | masks["iliac_artery_right"]
    if union.any():
        support = [np.flatnonzero(union.any(axis=tuple(j for j in range(3) if j != i))) for i in range(3)]
        box = tuple(slice(int(s[0]), int(s[-1]) + 1) for s in support)
        masks = {name: mask[box] for name, mask in masks.items()}
    for side, name in SIDES.items():
        result = reach_masks(masks["aorta"], masks[name])
        for rule in RULES:
            row[f"{side}_{rule}"] = result[rule]
    return row


def rates(cases):
    """Disconnection with Wilson intervals by frame, side, adjacency rule and released split."""
    rows = []
    frames = {"arterial_measurable": cases, "venous_measurable": cases[cases.venous_status == "measurable"]}
    for frame_name, frame in frames.items():
        for side in SIDES:
            for rule in RULES:
                column = f"{side}_{rule}"
                kept = frame[frame[column] != "nonmeasurable"]
                for split, part in [("all", kept)] + list(kept.groupby("split", sort=True)):
                    events, n = int(part[column].eq("disconnected").sum()), len(part)
                    lo, hi = S.wilson_ci(events, n, CFG.ALPHA) if n else (np.nan, np.nan)
                    rows.append({"frame": frame_name, "side": side, "adjacency": rule, "split": split,
                                 "events": events, "n": n, "estimate": 100 * events / n if n else np.nan,
                                 "ci_lo": 100 * lo, "ci_hi": 100 * hi, "ci_method": "Wilson"})
    return pd.DataFrame(rows)


def _publish(path, data):
    part = path.with_name(path.name + ".part")
    with open(part, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(part, path)
    return hashlib.sha256(data).hexdigest()


def _recorded(manifest, name):
    hits = [e["sha256"] for e in manifest["inputs"] if e["path"].endswith(f"Totalsegmentor_traning_data/{name}")]
    if len(hits) != 1:
        raise ValueError(f"the venous audit recorded {len(hits)} digests for {name}")
    return hits[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=_pl.Path, required=True, help="the venous audit's annotation cache")
    parser.add_argument("--arterial-cache", type=_pl.Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", type=_pl.Path, default=OUT)
    args = parser.parse_args(argv)
    if args.arterial_cache.resolve() == args.cache.resolve():
        raise ValueError("the arterial masks go into their own folder")
    manifest_bytes = (VENOUS / "supplement_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    directory = (args.cache / "central_directory.bin").read_bytes()
    meta_bytes = (args.cache / "meta.csv").read_bytes()
    venous_bytes = (VENOUS / "reference_cases.csv").read_bytes()
    digests = {"central_directory.bin": hashlib.sha256(directory).hexdigest(),
               "meta.csv": hashlib.sha256(meta_bytes).hexdigest()}
    for name, digest in digests.items():
        if digest != _recorded(manifest, name):
            raise ValueError(f"{name} is not the file the venous audit recorded")
    entries = parse_directory(directory)
    selected = json.loads((args.cache / "selected_entries.json").read_bytes())
    by_name = {e["name"]: e for e in entries}
    if any(by_name.get(s["name"]) != s for s in selected) or len(selected) != 3 * 1228:
        raise ValueError("the directory does not reproduce the venous audit's 3,684 mask records")
    groups = arterial_entries(entries)
    if len(groups) != 1228:
        raise ValueError(f"{len(groups)} released cases hold arterial masks (1,228 expected)")
    received = 0
    if args.fetch:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, n in enumerate(pool.map(fetch_case, [(args.arterial_cache, groups[c]) for c in sorted(groups)]), 1):
                received += n
                if i % 100 == 0 or i == len(groups):
                    print(f"fetched {i}/{len(groups)} cases; {received:,} bytes received", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(audit_case, [(args.arterial_cache, c) for c in sorted(groups)]), 1):
            rows.append(row)
            if i % 200 == 0 or i == len(groups):
                print(f"audited {i}/{len(groups)} cases", flush=True)
    cases = pd.DataFrame(rows)
    meta = pd.read_csv(io.BytesIO(meta_bytes), sep=";", encoding="utf-8-sig").rename(columns={"image_id": "case"})
    venous = pd.read_csv(io.BytesIO(venous_bytes), keep_default_na=False)[["case", "status", "split"]]
    if set(meta.case) != set(cases.case) or set(venous.case) != set(cases.case):
        raise ValueError("the arterial cases, released metadata and venous audit must match one to one")
    if int(venous.status.eq("measurable").sum()) != 600:
        raise ValueError("the venous audit measured 600 cases")
    cases = cases.merge(meta[["case", "split"]], on="case", validate="one_to_one").merge(
        venous.rename(columns={"status": "venous_status", "split": "venous_split"}), on="case", validate="one_to_one")
    if not cases.split.eq(cases.venous_split).all():
        raise ValueError("the released split differs between the metadata and the venous audit")
    order = ["case", "split", "venous_status", *(f"{n}_present" for n in ARTERIAL),
             *(f"{s}_{r}" for s in SIDES for r in RULES), "slice_mm", "grid_offset_mm", "shared_voxels",
             *(f"{n}_sha256" for n in ARTERIAL)]
    cases = cases[order].sort_values("case").reset_index(drop=True)
    table = rates(cases)
    args.out.mkdir(parents=True, exist_ok=True)
    outputs = {"reference_arterial_cases.csv": _publish(args.out / "reference_arterial_cases.csv",
                                                        cases.to_csv(index=False).encode("utf-8")),
               "t1_reference_arterial.csv": _publish(args.out / "t1_reference_arterial.csv",
                                                     table.to_csv(index=False, float_format="%.10g").encode("utf-8"))}
    record = {"producer": PRODUCER.relative_to(ROOT).as_posix(), "producer_sha256": PRODUCER_SHA256,
              "comparator_rule": "confluence_comparators.reach", "comparator_sha256": CC_SHA256,
              "archive": EM.PUBLIC_REFERENCE_URL, "python": platform.python_version(),
              "inputs": {"central_directory.bin": digests["central_directory.bin"], "meta.csv": digests["meta.csv"],
                         "Results/phase1_characterize/connectivity/reference_cases.csv":
                             hashlib.sha256(venous_bytes).hexdigest(),
                         "Results/phase1_characterize/connectivity/supplement_manifest.json":
                             hashlib.sha256(manifest_bytes).hexdigest()},
              "arterial_masks": int(3 * len(cases)), "fetched_bytes_this_run": received,
              "max_grid_offset_mm": float(cases.grid_offset_mm.max()),
              "cases_with_shared_arterial_voxels": int(cases.shared_voxels.gt(0).sum()),
              "outputs": outputs}
    _publish(args.out / "reference_arterial_run_record.json",
             (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
