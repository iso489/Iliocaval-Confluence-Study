"""Pytest bootstrap.

Each code folder (common/, phase0_data/, phase1_characterize/) is appended to sys.path, so the flat
imports the modules use (`import gap_config`, `import stats`) resolve from any folder. The folders are
appended, not prepended, so a module here never shadows a standard-library or installed package.
Runnable scripts carry the same bootstrap at their top.

Tests marked `needs_metadata` read the three source metadata tables (METADATA/README.md); they are
skipped, with that reason, when the tables are absent.
"""
import pathlib
import sys

import pytest

_CODE = pathlib.Path(__file__).resolve().parent
for _p in sorted(_CODE.iterdir()):
    if _p.is_dir() and _p.name[0] not in "._":
        _entry = str(_p)
        if _entry not in sys.path:
            sys.path.append(_entry)

_METADATA = [_CODE.parent / "METADATA" / name for name in ("AMOS.csv", "SAROS.csv", "autoPET.csv")]


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_metadata: reads the source metadata tables in METADATA/")


def pytest_collection_modifyitems(config, items):
    if all(path.is_file() for path in _METADATA):
        return
    skip = pytest.mark.skip(reason="needs METADATA/AMOS.csv, SAROS.csv and autoPET.csv (METADATA/README.md)")
    for item in items:
        if "needs_metadata" in item.keywords:
            item.add_marker(skip)


def pytest_terminal_summary(terminalreporter):
    """List every skipped test with its reason (pytest -q prints only a count)."""
    skipped = terminalreporter.stats.get("skipped", [])
    if not skipped:
        return
    terminalreporter.write_sep("=", f"{len(skipped)} skipped, each with its reason")
    for report in skipped:
        if isinstance(report.longrepr, tuple):
            reason = report.longrepr[2].removeprefix("Skipped: ")
        else:
            reason = f"no reason recorded ({report.longrepr!r})"
        label = "SKIPPED IN TEARDOWN" if report.when == "teardown" else "NOT RUN"
        terminalreporter.write_line(f"{label} {report.nodeid}: {' / '.join(reason.splitlines())}")
