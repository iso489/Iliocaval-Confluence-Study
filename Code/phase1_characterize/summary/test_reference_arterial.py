"""Tests for the released-annotation arterial control (reference_arterial.py), on synthetic data."""
import importlib.util
import io
import pathlib
import struct
import zipfile

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

_PATH = pathlib.Path(__file__).with_name("reference_arterial.py")
_SPEC = importlib.util.spec_from_file_location("reference_arterial", _PATH)
RA = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(RA)
GRID = np.diag([1.5, 1.5, 1.5, 1.0])


def _archive(files, padded=(), method=zipfile.ZIP_DEFLATED):
    """An in-memory ZIP; names in `padded` carry a 100-byte local extra field."""
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", method) as archive:
        for name, data in files:
            info = zipfile.ZipInfo(name)
            info.compress_type = method
            if name in padded:
                info.extra = struct.pack("<HH", 0x9999, 100) + bytes(100)
            archive.writestr(info, data)
    data = stream.getvalue()
    end = data.rfind(b"PK\x05\x06")
    size, offset = struct.unpack("<LL", data[end + 12:end + 20])
    return data, data[offset:offset + size]


def test_the_directory_parser_reads_what_zipfile_wrote():
    files = [(f"s0001/segmentations/{n}.nii.gz", bytes(range(i, i + 50)) * 9) for i, n in enumerate("abc")]
    data, directory = _archive(files)
    parsed = RA.parse_directory(directory)
    infos = zipfile.ZipFile(io.BytesIO(data)).infolist()
    assert [(e["name"], e["crc32"], e["compressed_size"], e["uncompressed_size"], e["offset"], e["method"])
            for e in parsed] == [(i.filename, i.CRC, i.compress_size, i.file_size, i.header_offset, 8) for i in infos]
    with pytest.raises(ValueError, match="invalid central directory"):
        RA.parse_directory(b"XXXX" + directory[4:])


def test_fetch_downloads_only_the_arterial_masks_in_contiguous_runs_and_verifies_them(tmp_path, monkeypatch):
    masks = {name: bytes([i]) * (300 + 40 * i) + name.encode() for i, name in enumerate(RA.ARTERIAL)}
    files = [("s0001/segmentations/aorta.nii.gz", masks["aorta"]),
             ("s0001/segmentations/heart.nii.gz", b"not arterial" * 40),
             ("s0001/segmentations/iliac_artery_left.nii.gz", masks["iliac_artery_left"]),
             ("s0001/segmentations/iliac_artery_right.nii.gz", masks["iliac_artery_right"]),
             ("s0001/ct.nii.gz", b"CT" * 5000)]
    data, directory = _archive(files, padded={"s0001/segmentations/iliac_artery_right.nii.gz"})
    groups = RA.arterial_entries(RA.parse_directory(directory))
    assert list(groups) == ["s0001"] and [e["name"].split("/")[-1] for e in groups["s0001"]] == \
        ["aorta.nii.gz", "iliac_artery_left.nii.gz", "iliac_artery_right.nii.gz"]
    requests = []

    def serve(start, end):
        requests.append((start, end))
        return data[start:end + 1]
    monkeypatch.setattr(RA.EM, "get_range", serve)
    received = RA.fetch_case((tmp_path, groups["s0001"]))
    # the aorta alone; the two adjacent iliac masks in one request, extended once for the long extra field
    assert len(requests) == 3 and received == sum(e - s + 1 for s, e in requests)
    for name, raw in masks.items():
        assert (tmp_path / "s0001" / "segmentations" / f"{name}.nii.gz").read_bytes() == raw
    assert not (tmp_path / "s0001" / "segmentations" / "heart.nii.gz").exists()
    assert not (tmp_path / "s0001" / "ct.nii.gz").exists()
    requests.clear()
    assert RA.fetch_case((tmp_path, groups["s0001"])) == 0 and requests == []    # valid files are kept
    (tmp_path / "s0001" / "segmentations" / "aorta.nii.gz").write_bytes(b"truncated")
    corrupt = bytearray(data)
    first = groups["s0001"][0]
    corrupt[first["offset"] + 30 + len(first["name"]) + 5] ^= 0xFF
    monkeypatch.setattr(RA.EM, "get_range", lambda s, e: bytes(corrupt[s:e + 1]))
    with pytest.raises(Exception):
        RA.fetch_case((tmp_path, groups["s0001"]))
    assert (tmp_path / "s0001" / "segmentations" / "aorta.nii.gz").read_bytes() == b"truncated"   # nothing written


def test_a_stored_entry_with_a_flipped_byte_fails_its_crc(tmp_path, monkeypatch):
    # no decompression to trip over: only the length and CRC-32 check can refuse it
    files = [(f"s0002/segmentations/{name}.nii.gz", name.encode() * 30) for name in RA.ARTERIAL]
    data, directory = _archive(files, method=zipfile.ZIP_STORED)
    group = RA.arterial_entries(RA.parse_directory(directory))["s0002"]
    corrupt = bytearray(data)
    corrupt[group[1]["offset"] + 30 + len(group[1]["name"]) + 3] ^= 0x01
    monkeypatch.setattr(RA.EM, "get_range", lambda s, e: bytes(corrupt[s:e + 1]))
    with pytest.raises(ValueError, match="CRC32"):
        RA.fetch_case((tmp_path, group))
    assert not (tmp_path / "s0002" / "segmentations" / "iliac_artery_left.nii.gz").exists()


def _case(tmp_path, case, aorta, left, right, affine=GRID):
    folder = tmp_path / case / "segmentations"
    folder.mkdir(parents=True)
    for name, array, grid in (("aorta", aorta, GRID), ("iliac_artery_left", left, affine),
                              ("iliac_artery_right", right, affine)):
        nib.save(nib.Nifti1Image(array.astype(np.uint8), grid), folder / f"{name}.nii.gz")


def test_the_audit_reads_face_corner_and_empty_junctions(tmp_path):
    aorta, left, right = (np.zeros((8, 8, 8), bool) for _ in range(3))
    aorta[3, 3, 4:7] = True
    left[3, 3, 1:4] = True           # (3,3,3) shares a face with the aorta's (3,3,4)
    right[4, 4, 1:4] = True          # (4,4,3) and the aorta's (3,3,4) share only a corner
    _case(tmp_path, "s0001", aorta, left, right)
    row = RA.audit_case((tmp_path, "s0001"))
    assert (row["left_26"], row["left_6"], row["right_26"], row["right_6"]) == \
        ("connected", "connected", "connected", "disconnected")
    assert row["slice_mm"] == 1.5 and all(row[f"{n}_present"] == 1 for n in RA.ARTERIAL)
    _case(tmp_path, "s0002", aorta, left, np.zeros((8, 8, 8), bool))
    empty = RA.audit_case((tmp_path, "s0002"))
    assert (empty["right_26"], empty["right_6"], empty["iliac_artery_right_present"]) == \
        ("nonmeasurable", "nonmeasurable", 0)
    gap = np.zeros((8, 8, 8), bool)
    gap[3, 3, 0:2] = True            # two planes below the aorta: no contact under either rule
    stray = aorta.copy()
    stray[3, 3, 2] = True            # a one-voxel aorta fragment touching the iliac label, apart from the trunk
    _case(tmp_path, "s0005", stray, gap, right)
    gapped = RA.audit_case((tmp_path, "s0005"))
    assert (gapped["left_26"], gapped["left_6"], gapped["shared_voxels"]) == ("disconnected", "disconnected", 0)
    _case(tmp_path, "s0003", aorta, aorta, right)     # released masks may share voxels: counted, not refused
    shared = RA.audit_case((tmp_path, "s0003"))
    assert (shared["left_26"], shared["shared_voxels"]) == ("connected", 3)
    _case(tmp_path, "s0004", aorta, left, right, affine=np.diag([1.5, 1.5, 3.0, 1.0]))
    with pytest.raises(ValueError, match="grids differ"):
        RA.audit_case((tmp_path, "s0004"))
    rounded = GRID.copy()
    rounded[0, 1] = 1e-4             # header rounding of a rotation term: far below a quarter voxel
    _case(tmp_path, "s0006", aorta, left, right, affine=rounded)
    kept = RA.audit_case((tmp_path, "s0006"))
    assert kept["left_6"] == "connected" and 0 < kept["grid_offset_mm"] < 0.01
    shifted = GRID.copy()
    shifted[0, 3] = 0.5              # a third of a voxel along x: a different grid
    _case(tmp_path, "s0007", aorta, left, right, affine=shifted)
    with pytest.raises(ValueError, match="grids differ"):
        RA.audit_case((tmp_path, "s0007"))


def test_rates_use_the_measurable_cases_of_each_frame_with_wilson_intervals():
    cases = pd.DataFrame({
        "case": [f"s{i}" for i in range(6)], "split": ["train"] * 4 + ["test"] * 2,
        "venous_status": ["measurable", "measurable", "empty_primary_label", "measurable", "measurable",
                          "empty_primary_label"],
        "left_26": ["connected", "disconnected", "disconnected", "nonmeasurable", "connected", "connected"],
        "left_6": ["connected", "disconnected", "disconnected", "nonmeasurable", "disconnected", "connected"],
        "right_26": ["connected"] * 6, "right_6": ["connected"] * 6})
    table = RA.rates(cases).set_index(["frame", "side", "adjacency", "split"])
    assert table.loc[("arterial_measurable", "left", "26", "all"), ["events", "n"]].tolist() == [2, 5]
    assert table.loc[("venous_measurable", "left", "26", "all"), ["events", "n"]].tolist() == [1, 3]
    assert table.loc[("venous_measurable", "left", "6", "test"), ["events", "n"]].tolist() == [1, 1]
    lo, hi = RA.S.wilson_ci(2, 5, RA.CFG.ALPHA)
    row = table.loc[("arterial_measurable", "left", "26", "all")]
    assert (row.estimate, row.ci_lo, row.ci_hi) == pytest.approx((40.0, 100 * lo, 100 * hi))


def _central_record(name, comp, uncomp, offset, extra=b""):
    """One central-directory record as APPNOTE 4.3.12 lays it out (CRC 0x1234ABCD, deflated)."""
    raw = name.encode()
    return struct.pack("<4s6H3L5H2L", b"PK\x01\x02", 45, 45, 0, 8, 0, 0, 0x1234ABCD, comp, uncomp,
                       len(raw), len(extra), 0, 0, 0, 0, offset) + raw + extra


def test_the_directory_parser_reads_zip64_sizes_and_offsets():
    # a ZIP64 field holds, in order, the size, compressed size and offset whose 32-bit fields read 0xFFFFFFFF
    full = 0xFFFFFFFF
    unrelated = struct.pack("<HH", 0x5455, 5) + bytes(5)
    every = struct.pack("<HHQQQ", 1, 24, 5_000_000_000, 4_900_000_000, 6_000_000_000)
    offset_only = struct.pack("<HHQ", 1, 8, 7_000_000_000)
    directory = (_central_record("s0001/segmentations/aorta.nii.gz", full, full, full, unrelated + every)
                 + _central_record("s0001/segmentations/iliac_artery_left.nii.gz", 321, 654, full, offset_only))
    first, second = RA.parse_directory(directory)
    assert (first["uncompressed_size"], first["compressed_size"], first["offset"]) == \
        (5_000_000_000, 4_900_000_000, 6_000_000_000)
    assert (second["compressed_size"], second["uncompressed_size"], second["offset"]) == (321, 654, 7_000_000_000)
    assert (first["crc32"], first["method"], second["name"]) == \
        (0x1234ABCD, 8, "s0001/segmentations/iliac_artery_left.nii.gz")


def test_a_released_case_missing_an_arterial_mask_is_refused():
    files = [(f"s0003/segmentations/{name}.nii.gz", b"x" * 40) for name in RA.ARTERIAL[:2]]
    _, directory = _archive(files)
    with pytest.raises(ValueError, match="lacks one of the three arterial masks"):
        RA.arterial_entries(RA.parse_directory(directory))


def test_a_local_header_that_contradicts_its_directory_entry_is_refused(tmp_path, monkeypatch):
    files = [(f"s0004/segmentations/{name}.nii.gz", name.encode() * 30) for name in RA.ARTERIAL]
    data, directory = _archive(files, method=zipfile.ZIP_STORED)
    group = RA.arterial_entries(RA.parse_directory(directory))["s0004"]
    # the signature, then the method (stored in the directory, deflated in the local header)
    for position, value in ((0, b"PK\x03\x05"), (8, struct.pack("<H", 8))):
        corrupt = bytearray(data)
        at = group[0]["offset"] + position
        corrupt[at:at + len(value)] = value
        monkeypatch.setattr(RA.EM, "get_range", lambda s, e, blob=bytes(corrupt): blob[s:e + 1])
        with pytest.raises(ValueError, match="invalid local entry"):
            RA.fetch_case((tmp_path, group))
        assert not (tmp_path / "s0004").exists()


def test_the_audit_refuses_a_mask_that_is_not_binary(tmp_path):
    one = np.zeros((6, 6, 6), np.uint8)
    one[2, 2, 2] = 1
    _case(tmp_path, "s0008", one * 2, one, one)
    with pytest.raises(ValueError, match="s0008: aorta is not a binary integer mask"):
        RA.audit_case((tmp_path, "s0008"))
    _case(tmp_path, "s0009", one, one, one)
    nib.save(nib.Nifti1Image(one.astype(np.float32), GRID),
             tmp_path / "s0009" / "segmentations" / "iliac_artery_left.nii.gz")
    with pytest.raises(ValueError, match="s0009: iliac_artery_left is not a binary integer mask"):
        RA.audit_case((tmp_path, "s0009"))
    _case(tmp_path, "s0010", one, one, one)
    assert RA.audit_case((tmp_path, "s0010"))["aorta_present"] == 1
