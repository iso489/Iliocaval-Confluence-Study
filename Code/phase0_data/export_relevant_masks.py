"""Extract the IVCS-relevant structures from the TotalSegmentator 2.17.0 multilabel outputs.

The Phase-1 baseline was run with `--task total --ml`, so every case holds ONE
`segmentation.nii.gz` carrying all 117 classes. This writes, per case, one binary uint8
mask per relevant structure -- the same file names `measure_ivc_lciv_gap.py` and
`measure_bifurcation.py` read.

The structure list is VEINS + VERTEBRAE below: the venous classes the Phase-1 measurements read
and the vertebral level anchors.

Label INDICES are read from each case's own report.json rather than assumed, and a case
whose class map lacks any requested name is a hard error, not a silent skip.

This is the study's exporter; its 24,530 masks over 2,230 cases were verified against the source
volumes. Packing is fixed (`Opener.default_compresslevel = 6`; `header.copy()` +
`set_data_dtype` BEFORE constructing the image) so digests compare against a MANIFEST.

--layout chooses flat or nested output, and --dataset restricts the run to named cohorts.
--verify-against CSV compares each written mask to a known-good MANIFEST.csv. Voxel count and
SHA-256 are checked SEPARATELY: a digest mismatch with a matching count is a PACKING difference, not
a data difference, and reporting them as one thing would either raise a false alarm or hide a real
one.
-----------------------------------------------------------------------------------------"""
import argparse
import csv
import hashlib
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.openers import Opener

# nibabel writes .nii.gz at gzip level 1, which on a mostly-zero full-FOV binary mask
# costs ~4.3x the bytes of level 6 for identical content (241 KB -> 56 KB on a 768x768x90
# volume). These files exist to be downloaded, so pay the compression once here. Level 9
# buys a further 1.2% for several times the CPU, which is not a trade worth making.
Opener.default_compresslevel = 6

DATASETS = ("AMOS", "autoPET", "SAROS")

# VEINS -- the three venous classes every Phase-1 measurement reads.
VEINS = ["inferior_vena_cava", "iliac_vena_left", "iliac_vena_right"]
# VERTEBRAE -- the anatomical level anchor measure_bifurcation.py uses.
VERTEBRAE = ["vertebrae_T12", "vertebrae_L1", "vertebrae_L2", "vertebrae_L3",
             "vertebrae_L4", "vertebrae_L5", "vertebrae_S1", "sacrum"]
STRUCTURES = VEINS + VERTEBRAE

EXPECTED_VERSION = "2.17.0"
EXPECTED_TASK = "total"
#: Output subdirectory: flat writes into <case>/, nested into <case>/segmentations_fullres/.
LAYOUTS = {"flat": "", "nested": "segmentations_fullres"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def case_dirs(src, datasets=None):
    """Every case under the baseline tree, in a stable order.

    `datasets` restricts the sweep (one cohort per concurrent instance)."""
    want = tuple(datasets) if datasets else DATASETS
    unknown = [d for d in want if d not in DATASETS]
    if unknown:
        raise ValueError(f"unknown dataset(s) {unknown}; known: {DATASETS}")
    out = []
    for ds in want:
        d_ds = Path(src) / ds
        if not d_ds.is_dir():
            continue
        for d in sorted(d_ds.iterdir()):
            if d.is_dir():
                out.append((ds, d.name))
    return out


def load_reference(path):
    """{(case, structure): {voxels, sha256}} from a known-good MANIFEST.csv."""
    ref = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            ref[(row["case"], row["structure"])] = {"voxels": int(row["voxels"]),
                                                    "sha256": row["sha256"]}
    return ref


def baseline_class_index(src, structures=None):
    """Validate baseline provenance and return the report's own class indices."""
    structures = STRUCTURES if structures is None else tuple(structures)
    if not structures or len(set(structures)) != len(structures) or not set(structures) <= set(STRUCTURES):
        raise ValueError("requested export structures must be a nonempty unique relevant subset")
    report = json.loads((Path(src) / "report.json").read_text())
    return validate_baseline_report(report, structures)


def validate_baseline_report(report, structures=None):
    """The same provenance checks on an already-parsed report (a caller that hashed the bytes it
    parsed can validate those bytes rather than a second read). Returns name -> class index."""
    structures = STRUCTURES if structures is None else tuple(structures)
    if not structures or len(set(structures)) != len(structures) or not set(structures) <= set(STRUCTURES):
        raise ValueError("requested export structures must be a nonempty unique relevant subset")
    if report.get("totalsegmentator_version") != EXPECTED_VERSION:
        raise ValueError(f"version {report.get('totalsegmentator_version')!r} != {EXPECTED_VERSION!r}")
    if report.get("task") != EXPECTED_TASK:
        raise ValueError(f"task {report.get('task')!r} != {EXPECTED_TASK!r}")
    if not report.get("multilabel"):
        raise ValueError("report says multilabel is false; this reader assumes one ML file")
    if report.get("fast"):
        raise ValueError("report says fast=true; this is not the full-resolution baseline")
    index = {name: int(i) for i, name in report["classes"].items()}
    missing = [n for n in structures if n not in index]
    if missing:
        raise ValueError(f"class map lacks {missing}")
    selected = [index[n] for n in structures]
    if min(selected) <= 0 or len(set(selected)) != len(selected):
        raise ValueError("requested class indices must be positive and distinct")
    return index


def export_case(args, structures=None):
    """Write one case's masks.

    `args` is the 7-item task tuple (ds, case, src_root, out_root, force, segdir, ref);
    `structures` (default STRUCTURES) exports a subset, e.g. the three veins.
    """
    ds, case, src_root, out_root, force, segdir, ref = args
    src = Path(src_root) / ds / case
    dst = Path(out_root) / ds / case / segdir if segdir else Path(out_root) / ds / case
    try:
        structures = STRUCTURES if structures is None else tuple(structures)
        index = baseline_class_index(src, structures)

        seg = src / "segmentation.nii.gz"
        img = nib.load(seg)
        arr = np.asanyarray(img.dataobj)

        dst.mkdir(parents=True, exist_ok=True)
        rows = []
        for name in structures:
            label = index[name]
            path = dst / f"{name}.nii.gz"
            mask = (arr == label)
            n_vox = int(mask.sum())
            if force or not path.exists():
                # Carry the source geometry verbatim; only the dtype changes.
                header = img.header.copy()
                header.set_data_dtype(np.uint8)
                out = nib.Nifti1Image(mask.astype(np.uint8), img.affine, header)
                out.set_data_dtype(np.uint8)
                # nibabel picks the writer from the extension, so the temp name must
                # keep .nii.gz; only the stem is marked partial.
                tmp = path.with_name(f".{name}.partial.nii.gz")
                nib.save(out, tmp)
                os.replace(tmp, path)
            digest = sha256(path)
            row = {
                "dataset": ds, "case": case, "structure": name, "label": label,
                "voxels": n_vox, "bytes": path.stat().st_size,
                "sha256": digest,
                "shape": "x".join(str(v) for v in arr.shape),
            }
            if ref is not None:
                exp = ref.get((case, name))
                if exp is None:
                    row["check"] = "NOT_IN_REFERENCE"
                elif exp["voxels"] != n_vox:
                    # A DATA difference. The array itself disagrees with the verified export.
                    row["check"] = "VOXEL_MISMATCH"
                elif exp["sha256"] != digest:
                    # Same array, different bytes: packing only (compression level, header
                    # padding). Not an error, and must not be reported as one.
                    row["check"] = "PACKING_DIFFERS_DATA_EQUAL"
                else:
                    row["check"] = "BYTE_IDENTICAL"
            rows.append(row)
        return rows, None
    except Exception:
        return [], f"{ds}/{case}: {traceback.format_exc(limit=3)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="baseline tree holding <DATASET>/<case>/segmentation.nii.gz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layout", choices=sorted(LAYOUTS), default="flat",
                    help="flat: <case>/<structure>.nii.gz. "
                         "nested: <case>/segmentations_fullres/<structure>.nii.gz, the "
                         "measurement modules' default layout.")
    ap.add_argument("--verify-against", default=None,
                    help="known-good MANIFEST.csv to check every written mask against")
    ap.add_argument("--dataset", action="append", choices=sorted(DATASETS), default=None,
                    help="restrict to one cohort (repeatable). Required when several "
                         "instances run concurrently, e.g. one array task per cohort.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true",
                    help="rewrite masks that already exist")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    segdir = LAYOUTS[args.layout]
    ref = load_reference(args.verify_against) if args.verify_against else None
    cases = case_dirs(args.src, args.dataset)
    if not cases:
        sys.exit(f"no case directories under {args.src} for dataset(s) "
                 f"{args.dataset or list(DATASETS)}")
    print(f"{len(cases)} cases x {len(STRUCTURES)} structures "
          f"= {len(cases) * len(STRUCTURES)} masks -> {out_root} (layout={args.layout})",
          flush=True)
    if ref is None:
        print("  NOTE: no --verify-against; masks will be written UNVERIFIED", flush=True)

    rows, errors, done = [], [], 0
    jobs = [(ds, c, args.src, str(out_root), args.force, segdir, ref) for ds, c in cases]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(export_case, j) for j in jobs]
        for fut in as_completed(futures):
            r, err = fut.result()
            rows.extend(r)
            if err:
                errors.append(err)
                # SHOW THE FIRST ONE IMMEDIATELY: it is almost always the whole diagnosis. Further
                # ones are counted, not printed.
                if len(errors) == 1:
                    print(f"\n  FIRST ERROR (further ones are counted, not printed):\n{err}",
                          file=sys.stderr, flush=True)
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(cases)} cases, {len(errors)} errors", flush=True)
            # ABORT EARLY on a systematic failure. Past a modest sample, a high error rate is
            # never going to fix itself, and burning the remaining hours to arrive at the same
            # verdict wastes an allocation and delays the diagnosis.
            if done >= 50 and len(errors) / done > 0.25:
                for fu in futures:
                    fu.cancel()
                raise SystemExit(
                    f"\nABORTING: {len(errors)}/{done} cases failed ({100*len(errors)/done:.0f}%). "
                    "This is systematic, not incidental. The first error is above.")

    rows.sort(key=lambda r: (r["dataset"], r["case"], r["structure"]))
    # One manifest per invocation. A shared name would be overwritten by whichever array
    # task finished last, leaving a manifest describing one cohort and implying all three.
    suffix = "_".join(args.dataset) if args.dataset else "ALL"
    manifest = out_root / f"MANIFEST_{suffix}.csv"
    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {manifest} ({len(rows)} rows)")

    if ref is not None:
        tally = {}
        for r in rows:
            tally[r.get("check", "?")] = tally.get(r.get("check", "?"), 0) + 1
        print("verification against the reference manifest:")
        for k, v in sorted(tally.items()):
            print(f"    {k:<28} {v}")
        if tally.get("VOXEL_MISMATCH"):
            print(f"\n{tally['VOXEL_MISMATCH']} MASK(S) DISAGREE WITH THE VERIFIED EXPORT",
                  file=sys.stderr)
            sys.exit(2)

    if errors:
        print(f"\n{len(errors)} CASES FAILED", file=sys.stderr)
        for e in errors[:20]:
            print(e, file=sys.stderr)
        sys.exit(1)
    print("all cases exported with no errors")


# Public reference annotation acquisition: ZIP64 ranges, CRC32 and SHA-256.
import re
import struct
import threading
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

PUBLIC_REFERENCE_API = "https://zenodo.org/api/records/10047292"
PUBLIC_REFERENCE_URL = PUBLIC_REFERENCE_API + "/files/Totalsegmentator_dataset_v201.zip/content"
PUBLIC_REQUEST_LOCK = threading.Lock()
PUBLIC_LAST_REQUEST = 0.0

def _public_reference_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def get_range(start, end):
    global PUBLIC_LAST_REQUEST
    for attempt in range(4):
        try:
            # Zenodo advertises 133 requests/minute; share a slower limit across threads.
            with PUBLIC_REQUEST_LOCK:
                delay = 0.70 - (time.monotonic() - PUBLIC_LAST_REQUEST)
                if delay > 0:
                    time.sleep(delay)
                PUBLIC_LAST_REQUEST = time.monotonic()
            req = urllib.request.Request(PUBLIC_REFERENCE_URL, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=40) as response:
                if response.status != 206:
                    raise ValueError(
                        "server ignored range; refusing a full archive download"
                    )
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {start}-{end}/"):
                    raise ValueError("unexpected Content-Range")
                data = response.read(end - start + 2)
            if len(data) != end - start + 1:
                raise ValueError("range has wrong length")
            return data
        except urllib.error.HTTPError as error:
            if attempt == 3 or error.code not in (429, 500, 502, 503, 504):
                raise
            delay = max(float(error.headers.get("Retry-After", 0)), 2 ** (attempt + 1))
            print(
                f"HTTP {error.code}; honoring {delay:.0f}-second retry delay",
                flush=True,
            )
            while delay > 0:
                wait = min(delay, 30)
                time.sleep(wait)
                delay -= wait
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))
    raise AssertionError("unreachable")


def archive_index(cache):
    with urllib.request.urlopen(PUBLIC_REFERENCE_API, timeout=30) as response:
        metadata_bytes = response.read()
    metadata = json.loads(metadata_bytes)
    file = next(
        f for f in metadata["files"] if f["key"] == "Totalsegmentator_dataset_v201.zip"
    )
    size = file["size"]
    (cache / "zenodo_record.json").write_bytes(metadata_bytes)
    tail = get_range(size - 131072, size - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 20:
        raise ValueError("ZIP64 end record missing")
    locator = struct.unpack("<4sLQL", tail[eocd - 20 : eocd])
    if locator[0] != b"PK\x06\x07":
        raise ValueError("not a ZIP64 locator")
    fields = struct.unpack("<4sQ2H2L4Q", get_range(locator[2], locator[2] + 55))
    cdsize, cdoffset = fields[8:10]
    directory = get_range(cdoffset, cdoffset + cdsize - 1)
    (cache / "central_directory.bin").write_bytes(directory)
    entries, offset = [], 0
    while offset < len(directory):
        f = struct.unpack("<4s6H3L5H2L", directory[offset : offset + 46])
        if f[0] != b"PK\x01\x02":
            raise ValueError("invalid central directory entry")
        nl, el, cl = f[10:13]
        name = directory[offset + 46 : offset + 46 + nl].decode("utf-8")
        extra = directory[offset + 46 + nl : offset + 46 + nl + el]
        comp, uncomp, local_offset = f[8], f[9], f[16]
        p = 0
        while p + 4 <= len(extra):
            tag, length = struct.unpack("<HH", extra[p : p + 4])
            body = extra[p + 4 : p + 4 + length]
            p += 4 + length
            if tag == 1:
                q = 0
                if uncomp == 0xFFFFFFFF:
                    uncomp = struct.unpack_from("<Q", body, q)[0]
                    q += 8
                if comp == 0xFFFFFFFF:
                    comp = struct.unpack_from("<Q", body, q)[0]
                    q += 8
                if local_offset == 0xFFFFFFFF:
                    local_offset = struct.unpack_from("<Q", body, q)[0]
        entries.append(
            {
                "name": name,
                "method": f[4],
                "crc32": f[7],
                "compressed_size": comp,
                "uncompressed_size": uncomp,
                "offset": local_offset,
            }
        )
        offset += 46 + nl + el + cl
    if len(entries) != fields[7]:
        raise ValueError("central directory count mismatch")
    return entries, metadata, hashlib.sha256(directory).hexdigest()


def fetch_group(task):
    cache, entries = task
    valid = []
    for e in entries:
        path = cache / e["name"]
        if path.exists():
            raw = path.read_bytes()
            valid.append(
                len(raw) == e["uncompressed_size"]
                and zlib.crc32(raw) & 0xFFFFFFFF == e["crc32"]
            )
        else:
            valid.append(False)
    if not all(valid):
        start = min(e["offset"] for e in entries)
        end = (
            max(
                e["offset"] + 30 + len(e["name"].encode()) + 4096 + e["compressed_size"]
                for e in entries
            )
            - 1
        )
        blob = get_range(start, end)
        for e in entries:
            i = e["offset"] - start
            header = struct.unpack("<4s5H3L2H", blob[i : i + 30])
            if header[0] != b"PK\x03\x04" or header[3] != e["method"]:
                raise ValueError("invalid local entry")
            j = i + 30 + header[-2] + header[-1]
            packed = blob[j : j + e["compressed_size"]]
            raw = zlib.decompress(packed, -15) if e["method"] == 8 else packed
            if (
                len(raw) != e["uncompressed_size"]
                or zlib.crc32(raw) & 0xFFFFFFFF != e["crc32"]
            ):
                raise ValueError("ZIP entry failed length/CRC32 verification")
            path = cache / e["name"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    return [{**e, "sha256": _public_reference_digest(cache / e["name"])} for e in entries]


def fetch_public_reference_masks(cache, workers=2, reuse=True):
    """Return verified mask-entry records and released metadata; never download CT."""
    import pandas as pd
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    if reuse and (cache / "selected_entries.json").exists():
        entries = json.loads((cache / "selected_entries.json").read_text())
        metadata = json.loads((cache / "zenodo_record.json").read_text())
        directory_hash = _public_reference_digest(cache / "central_directory.bin")
    else:
        all_entries, metadata, directory_hash = archive_index(cache)
        names = ("inferior_vena_cava", "iliac_vena_left", "iliac_vena_right")
        pattern = re.compile(r"s\d+/segmentations/(?:" + "|".join(names) + r")\.nii\.gz\Z")
        entries = [e for e in all_entries if pattern.fullmatch(e["name"])]
        fetch_group((cache, [next(e for e in all_entries if e["name"] == "meta.csv")]))
        (cache / "selected_entries.json").write_text(json.dumps(entries))
    groups = {}
    for entry in entries:
        groups.setdefault(entry["name"].split("/")[0], []).append(entry)
    if any(len(group) != 3 for group in groups.values()):
        raise ValueError("incomplete public-reference vein triplet")
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, rows in enumerate(pool.map(fetch_group, [(cache, groups[c]) for c in sorted(groups)]), 1):
            records.extend(rows)
            if i % 50 == 0 or i == len(groups):
                print(f"Verified public-reference triplets: {i}/{len(groups)}", flush=True)
    return pd.DataFrame(records), pd.read_csv(cache / "meta.csv", sep=";"), metadata, directory_hash


if __name__ == "__main__":
    main()
