#!/usr/bin/env python3
"""Tests for pathsafe.case_fold_duplicates."""


import pathsafe as PS


def test_case_fold_duplicates_reports_the_offending_spellings():
    """The expectation is derived from the property, not from the implementation: the function
    returns every SPELLING involved in a collision, sorted, and nothing else.
    """
    assert PS.case_fold_duplicates(["amos_0102", "AMOS_0102"]) == ["AMOS_0102", "amos_0102"]
    assert PS.case_fold_duplicates(["Probe_Case", "probe_case", "PROBE_CASE"]) == [
        "PROBE_CASE", "Probe_Case", "probe_case"]
    # a literal repeat is still a collision, and is reported once
    assert PS.case_fold_duplicates(["a", "a"]) == ["a"]
    # distinct ids are not flagged, including ones that merely share a prefix
    assert PS.case_fold_duplicates(["amos_0102", "amos_0103", "amos_01020"]) == []
    assert PS.case_fold_duplicates([]) == []
    # only the colliding group is reported, not the whole roster
    assert PS.case_fold_duplicates(["keep", "DUP", "dup", "other"]) == ["DUP", "dup"]


def test_case_fold_duplicates_agrees_with_this_filesystem(tmp_path):
    """The rule is only worth having if the filesystem really does fold, so that is MEASURED
    rather than assumed, and the test states what it found either way.

    On a case-insensitive volume the two spellings are one directory, which is the defect.
    On a case-sensitive one they are two, which is also a defect (the same study becomes two
    cases), so the guard fires in both worlds and this test asserts the guard, not the
    filesystem.
    """
    (tmp_path / "probe_case").mkdir()
    folds = (tmp_path / "PROBE_CASE").exists()
    assert PS.case_fold_duplicates(["probe_case", "PROBE_CASE"]) == [
        "PROBE_CASE", "probe_case"], (
        f"the guard must fire whether or not this filesystem folds (it folds: {folds})")
