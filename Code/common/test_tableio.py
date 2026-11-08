"""Without atomic publication a kill mid-run would leave a truncated table indistinguishable from a
smaller complete cohort; `stats.validate_result_roster` catches this only for cohorts with an exact
roster, and a consumer that reads a table without calling it would accept the partial file.

A `measure_bifurcation --limit N` quick test writes `<prefix>_results_bifurcation.PARTIAL_limitN.txt`
(`tableio.results_path`), so it cannot replace the canonical table.

The sentinel is a `#` line, which every Phase-1 parser skips (they require an underscore in, or a
case-id shape for, the first token)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import tableio

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "Results" / "phase1_characterize"


def test_rows_are_published_atomically_and_the_target_never_holds_a_partial(tmp_path):
    target = tmp_path / "amos_results_fullres.txt"
    target.write_text("amos_0001 220 140 25.00 20.59 +25.00 25.62 OK\n")
    original = target.read_text()
    with tableio.AtomicTable(target) as table:
        table.write("amos_0002 240 20 10.00 9.57 +10.00 13.56 OK")
        # mid-run the canonical path must still hold the OLD complete artifact, not a
        # half-written new one -- that is the whole point of publishing atomically
        assert target.read_text() == original
        assert (tmp_path / "amos_results_fullres.txt.part").exists()
    assert "amos_0002" in target.read_text()
    assert not (tmp_path / "amos_results_fullres.txt.part").exists()


def test_an_interrupted_run_leaves_the_previous_artifact_intact(tmp_path):
    target = tmp_path / "t.txt"
    target.write_text("case_000 complete previous run\n")
    with pytest.raises(KeyboardInterrupt):
        with tableio.AtomicTable(target) as table:
            table.write("case_001 partial")
            raise KeyboardInterrupt
    assert target.read_text() == "case_000 complete previous run\n"


def test_a_failed_enter_does_not_leak_the_in_flight_key(tmp_path):
    """`__exit__` never runs for a failed `__enter__`, so the path must be registered in
    `_IN_FLIGHT` only after the file is open; otherwise a retry in the same process would report a
    concurrency conflict that does not exist and mask the real error."""
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    target = blocked / "t.txt"
    (blocked / "t.txt.part").mkdir()          # make open() fail: the .part path is a directory
    for _attempt in range(2):
        with pytest.raises(Exception) as exc:
            with tableio.AtomicTable(target):
                pass
        assert "already being published" not in str(exc.value), (
            "the first failed __enter__ leaked its key, so the retry reports a phantom "
            "concurrency conflict instead of the real error")


def test_the_sentinel_is_invisible_to_the_existing_parsers(tmp_path):
    """Checked against the real parsers rather than assumed.

    Every Phase-1 reader skips a line whose first token is not case-id shaped, so the
    sentinel must not change any count. If this breaks, adding the marker would silently
    move a published denominator.
    """
    import sys
    sys.path.insert(0, str(ROOT / "Code" / "phase1_characterize"))
    import stats as ST
    plain = tmp_path / "plain.txt"
    plain.write_text("case_000 210 185 15.00 12.53 +15.00 15.04 OK\n"
                     "case_001 215 175 5.00 4.00 +5.00 5.20 OK\n")
    marked = tmp_path / "marked.txt"
    with tableio.AtomicTable(marked) as table:
        table.write("case_000 210 185 15.00 12.53 +15.00 15.04 OK")
        table.write("case_001 215 175 5.00 4.00 +5.00 5.20 OK")
    assert ST.load_fullres(plain)[1] == ST.load_fullres(marked)[1]
    assert ST.load_fullres(plain)[0] == ST.load_fullres(marked)[0]


def test_publishing_twice_to_one_path_is_refused(tmp_path):
    target = tmp_path / "t.txt"
    with tableio.AtomicTable(target) as first:
        first.write("case_000 1 2 3")
        with pytest.raises(RuntimeError, match="already"):
            with tableio.AtomicTable(target):
                pass
    assert os.path.exists(target)


# ===========================================================================
# The drivers must publish through the helper.

import ast

DRIVERS = {
    "measure_all.py": ("compare", "fullres", "aux"),
    "measure_bifurcation.py": ("bifurcation",),
}


def _driver_tree(name):
    path = ROOT / "Code" / "phase1_characterize" / name
    if not path.is_file():
        pytest.skip(f"{name} absent")
    return ast.parse(path.read_text(encoding="utf-8")), path


@pytest.mark.parametrize("name", sorted(DRIVERS))
def test_each_driver_publishes_through_the_atomic_writer(name):
    tree, path = _driver_tree(name)
    used = {n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    used |= {n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "AtomicTable" in used, (
        f"{name} does not publish through AtomicTable, so an interrupted run "
        f"leaves a truncated table indistinguishable from a smaller cohort")


@pytest.mark.parametrize("name", sorted(DRIVERS))
def test_no_driver_opens_a_results_table_for_writing_directly(name):
    """Matched on the ARGUMENT, not on the call name, so a driver that keeps `open()` for
    a roster or a log is untouched; only a write to a `*_results_*` path is refused.
    """
    tree, path = _driver_tree(name)
    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "open"):
            continue
        mode = node.args[1].value if (len(node.args) > 1
                                      and isinstance(node.args[1], ast.Constant)) else "r"
        if "w" not in str(mode) and "a" not in str(mode):
            continue
        target = ast.unparse(node.args[0]) if node.args else ""
        if "_results_" in target or "path" in target.lower():
            offenders.append(f"line {node.lineno}: open({target}, {mode!r})")
    assert not offenders, (
        f"{name} writes a results table with a bare open(): {offenders}")


def test_a_limited_bifurcation_run_cannot_land_on_the_canonical_filename():
    assert tableio.results_path("amos", "bifurcation", 0) == "amos_results_bifurcation.txt"
    partial = tableio.results_path("amos", "bifurcation", 5)
    assert partial != "amos_results_bifurcation.txt"
    assert "PARTIAL" in partial and "5" in partial
    # and it must not collide with any committed artifact name
    committed = {p.name for p in RESULTS.rglob("*_results_*.txt")}
    assert partial not in committed


def test_the_bifurcation_driver_routes_its_output_through_results_path():
    """Otherwise the guard above is a function nobody calls."""
    tree, _path = _driver_tree("measure_bifurcation.py")
    main = next((n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "main"), None)
    assert main is not None
    calls = [n for n in ast.walk(main)
             if isinstance(n, ast.Call)
             and ((isinstance(n.func, ast.Name) and n.func.id == "results_path")
                  or (isinstance(n.func, ast.Attribute) and n.func.attr == "results_path"))]
    assert calls, "measure_bifurcation.main() does not derive its output name via results_path"
    # STRUCTURAL, not substring. `results_path(prefix, suffix, limit=0)` takes the limit third, so
    # require that the argument IS `args.limit` and nothing built from it.
    def _is_args_limit(node):
        return (isinstance(node, ast.Attribute) and node.attr == "limit"
                and isinstance(node.value, ast.Name) and node.value.id == "args")

    passes_limit = any(
        (len(c.args) >= 3 and _is_args_limit(c.args[2]))
        or any(kw.arg == "limit" and _is_args_limit(kw.value) for kw in c.keywords)
        for c in calls)
    assert passes_limit, (
        "results_path's `limit` argument is not exactly `args.limit`. Anything derived from "
        "it -- `0 * args.limit`, a renamed suffix, a constant -- would let a --limit run "
        "overwrite the canonical results table.")


# `os.replace` is atomic on POSIX, but the rename it performs is not DURABLE until the containing
# directory is fsynced. These are spy tests rather than content tests on purpose.

import stat as _stat


def _fsync_spy(monkeypatch):
    """Record an ordered trace of `(kind, resolved path)` for every fsync and every replace.

    Recording only FILE-vs-DIR cannot tell the correct directory from any other (fsyncing the
    grandparent, or the parent before the rename, would pass), so descriptors are mapped back to
    paths through `os.open`, and `os.replace` is recorded in the same trace, so both the target and
    the ordering are observable."""
    trace = []
    fds = {}
    real_open, real_fsync, real_replace = os.open, os.fsync, os.replace

    def spy_open(path, *a, **k):
        fd = real_open(path, *a, **k)
        fds[fd] = os.fspath(path)
        return fd

    def spy_fsync(fd):
        try:
            kind = "DIR" if _stat.S_ISDIR(os.fstat(fd).st_mode) else "FILE"
        except OSError:                        # pragma: no cover - defensive only
            kind = "?"
        trace.append((kind, fds.get(fd)))
        return real_fsync(fd)

    def spy_replace(src, dst, *a, **k):
        trace.append(("REPLACE", os.fspath(dst)))
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(os, "open", spy_open)
    monkeypatch.setattr(os, "fsync", spy_fsync)
    monkeypatch.setattr(os, "replace", spy_replace)
    return trace


def _assert_durable_publication(trace, target):
    target = Path(target)
    kinds = [k for k, _ in trace]
    assert "FILE" in kinds, f"the payload itself was never fsynced; trace={trace}"
    assert "REPLACE" in kinds, f"nothing was published by os.replace; trace={trace}"

    dir_syncs = [(i, path) for i, (k, path) in enumerate(trace) if k == "DIR"]
    assert dir_syncs, f"the directory the rename lands in was never fsynced; trace={trace}"

    on_target = [i for i, path in dir_syncs
                 if path is not None and Path(path).resolve() == target.parent.resolve()]
    assert on_target, (
        f"a directory was fsynced but not {target.parent}, so the published rename is still "
        f"not durable; trace={trace}")

    replace_at = kinds.index("REPLACE")
    assert max(on_target) > replace_at, (
        "the directory was fsynced BEFORE the rename, when the entry being made durable does "
        f"not exist yet; trace={trace}")


def test_tableio_makes_the_published_rename_durable(tmp_path, monkeypatch):
    trace = _fsync_spy(monkeypatch)
    path = tmp_path / "amos_results_fullres.txt"
    with tableio.AtomicTable(path) as table:
        table.write("case_0001 1.0\n")
    _assert_durable_publication(trace, path)
    assert path.exists()


def test_savefig_atomic_makes_both_the_bytes_and_the_rename_durable(tmp_path,
                                                                       monkeypatch):
    """`figio.savefig_atomic` must fsync both the bytes and the directory entry."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import figio

    trace = _fsync_spy(monkeypatch)
    target = tmp_path / "figure1_phase1.png"
    fig = plt.figure()
    try:
        figio.savefig_atomic(fig, target)
    finally:
        plt.close(fig)

    _assert_durable_publication(trace, target)
