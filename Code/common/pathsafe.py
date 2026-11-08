#!/usr/bin/env python3
"""pathsafe.py: case-fold duplicate detection for identifiers that become filesystem path components.

A case id is data, not a path fragment; on a case-insensitive filesystem two spellings of one id name
the same directory.
"""
from __future__ import annotations


def case_fold_duplicates(cases):
    """Ids that collide once case is disregarded -> the offending spellings, sorted.

    A raw string comparison is not enough: `probe_case`, `PROBE_CASE` and `Probe_Case` are distinct
    strings, and the default macOS filesystem (APFS) is case-insensitive, so they can name one
    directory.

    Folded unconditionally rather than only on a case-insensitive filesystem.

    IT LIVES HERE, not in one driver. This module is the single place allowed to decide whether a
    string may become a path component, so the rule cannot drift between readers."""
    seen = {}
    for case in cases:
        seen.setdefault(case.casefold(), []).append(case)
    return sorted(spelling for group in seen.values() if len(group) > 1
                  for spelling in dict.fromkeys(group))


