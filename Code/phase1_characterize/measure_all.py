#!/usr/bin/env python3
"""Measure the IVC-LCIV gap over every case in a dataset and write three result
files:

  <prefix>_results_compare.txt -> case fast_3d full_3d delta fast_vox full_vox qc
  <prefix>_results_fullres.txt -> case ivc_z lciv_z z_gap surf_3d raw_z_gap c2c_3d qc
  <prefix>_results_aux.txt     -> case sz sy sx lcc_frac n_comp body_csa

  surf_3d    PRIMARY: 3D nearest SURFACE-to-surface distance (de-biased for the
             half-voxel center offset; fast_3d/full_3d are this metric too).
  c2c_3d     center-to-center 3D distance (kept so the geometry invariant
             d^2 = z^2 + lat^2 holds for the center-based pair (z_gap, c2c_3d)).
  raw_z_gap  signed axial gap (negative == IVC/iliac label overlap); z_gap is
             this value clamped at 0. Emitting the sign makes the left-censoring
             recoverable.
  qc         OK, or a '+'-joined set of SMALL_IVC, SMALL_LCIV, OVERLAP, REVERSED_Z,
             OBLIQUE and LARGE_GAP_IMPLAUSIBLE.
  Gap values are written at 0.01 mm so downstream stats are not re-quantized by
  output rounding.

A fast pass that reads the full-res pass's own masks (by --seg-subdir, a link or a byte copy) is no
second inference, and a fast-pass mask that cannot be read costs its compare row only, never the
case's primary measurement. A partial failure rate is data, and is published. On a clean exit the
three tables are published one after another, not as one transaction.

Reuses measure_one().

Usage:
  python3 measure_all.py --data-dir amos-data --prefix amos --jobs 8
"""
import argparse
import collections
import filecmp
import math
import sys
from pathlib import Path
from multiprocessing import Pool

# --- run-from-anywhere: put sibling phase dirs on sys.path (tests use Code/conftest.py) ---
import sys as _sys, pathlib as _pl
for _p in sorted(_pl.Path(__file__).resolve().parent.parent.iterdir()):
    if _p.is_dir() and _p.name[0] not in "._":
        if str(_p) not in _sys.path:
            _sys.path.append(str(_p))  # append, never shadow stdlib
from measure_ivc_lciv_gap import measure_one, case_aux_metrics
from tableio import AtomicTable  # atomic + completion-marked
import hpc_resources as HR  # SLURM/cgroup-aware workers + thread pinning
import pathsafe as PS  # the one module allowed to judge a path component

DATA = None   # set in main()/initializer; see _init_worker
# Subdirectory of a case directory holding the per-structure masks. An EMPTY string means "directly
# in the case directory" and `Path / ""` is a no-op, so the join sites need no branch -- only the
# globs do, which is what `_seg_glob` is for.
SEGDIR = "segmentations_fullres"


def _seg_glob(name, segdir=None):
    """Glob pattern matching `<case>/<segdir>/<name>` under a cohort data directory."""
    sd = SEGDIR if segdir is None else segdir
    return f"*/{sd}/{name}" if sd else f"*/{name}"
PHASE0 = Path(__file__).resolve().parent.parent / "phase0_data"
DEFAULT_ROSTERS = {
    "amos": PHASE0 / "amos_ct_cases.txt",
    "saros": PHASE0 / "saros_processed_cases.txt",
    "autopet": PHASE0 / "autopet_cases.txt",
}


def _init_worker(data_dir, segdir=None):
    """Under spawn the module is re-imported with DATA=None, so relying on the inherited global
    silently breaks every case off-Linux."""
    # Pin BLAS/OpenMP to one thread PER WORKER. It has to happen here rather than at import: a
    # forked worker already has numpy loaded, so only a runtime limiter reaches the library that is
    # actually running.
    HR.limit_blas_threads(1)
    global DATA, SEGDIR
    DATA = Path(data_dir)
    if segdir is not None:
        SEGDIR = segdir


def load_expected_roster(prefix, roster_path=None):
    """Load an explicit/default exhaustive roster, or return None for open cohorts."""
    path = Path(roster_path) if roster_path else DEFAULT_ROSTERS.get(str(prefix).lower())
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"cohort roster not found: {path}")
    ids = [ln.split()[0] for ln in path.read_text().splitlines() if ln.split()]
    # Case-FOLDED, not a raw set. `amos_0102` and `AMOS_0102` are distinct strings, and on a
    # case-insensitive filesystem both spellings address ONE case directory: the roster would
    # claim n+1 cases where n exist, and this roster is the EXPECTED side of the
    # closed-accounting check, so the miscount lands directly in the cohort denominator rather
    # than being caught downstream.
    collisions = PS.case_fold_duplicates(ids)
    if collisions:
        raise ValueError(
            f"cohort roster case ids collide once case is disregarded, and this filesystem "
            f"may treat them as one case: {collisions} in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate case ids in cohort roster: {path}")
    return ids


def case_roster(data_dir, expected=None, segdir=None):
    """Closed-accounting roster: (roster, produced, missing).

    `produced` = cases that wrote a full-res IVC mask. `roster` =
    every ATTEMPTED case directory (one holding a CT or any segmentations dir).
    `missing` = roster - produced: cases where TotalSegmentator produced no IVC mask
    at all; emitting them as NO_MASK keeps total-failure cases in the denominator and
    closes the accounting.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"cohort data directory does not exist: {data_dir}")
    sd = SEGDIR if segdir is None else segdir
    produced = sorted({
        (p.parent.parent.name if sd else p.parent.name)
        for p in data_dir.glob(_seg_glob("inferior_vena_cava.nii.gz", sd))
    })

    def _is_case_dir(p):
        if not (p.is_dir() and not p.name.startswith(".")):
            return False
        if not sd:
            # Flat layout: the masks ARE the case directory's contents, so a case is a
            # directory that holds the IVC mask. There is no separate "attempted but produced
            # nothing" state to detect here, because the export writes every case that ran.
            return (p / "inferior_vena_cava.nii.gz").exists()
        return ((p / "ct.nii.gz").exists()
                or (p / "segmentations").is_dir()
                or (p / sd).is_dir())

    discovered = (sorted(p.name for p in data_dir.iterdir() if _is_case_dir(p))
                  if data_dir.exists() else [])
    roster = list(expected) if expected is not None else discovered
    expected_set = set(roster)
    produced = [c for c in produced if c in expected_set]
    pset = set(produced)
    missing = [c for c in roster if c not in pset]
    return roster, produced, missing


def _aux_component_token(n):
    """The aux table's `n_comp` column, or NA when the count was not measured.

    A component count of ZERO cannot describe a mask that was measured -- a real mask
    has at least one component -- yet three separate paths produce it, only one of
    which raises: `case_aux_metrics` returns its default when the LCIV mask file is
    ABSENT, `lciv_quality` returns (nan, 0) for an EMPTY mask, and the caller below
    catches any exception. Writing `0` for all three would put a failure into the same
    domain as a measurement (stats.py would label it seg_quality="clean" and test it), so
    the column writes NA, the convention its two neighbouring columns use for "not
    measured".
    """
    if isinstance(n, bool) or not isinstance(n, (int, float)):
        return "NA"
    if isinstance(n, float) and not math.isfinite(n):
        return "NA"
    return str(int(n)) if n >= 1 else "NA"


def work(case):
    try:
        return _work(case)
    except Exception as e:                       # one corrupt NIfTI must not kill
        print(f"  [ERROR {case}: {type(e).__name__}: {e}]", file=sys.stderr,
              flush=True)                         # the whole Pool.imap iterator
        return (f"{case} empty mask", f"{case} ERROR: {type(e).__name__}",
                f"{case} NA NA NA NA NA NA")   # aux schema is 7 tokens (case sz sy sx lcc_frac n_comp body_csa)


def _same_masks(cd):
    """True when the fast pass reads the full-res pass's own masks: the same path (`--seg-subdir
    segmentations`), a link, or a byte copy."""
    for name in ("inferior_vena_cava.nii.gz", "iliac_vena_left.nii.gz"):
        fast, full = cd / "segmentations" / name, cd / SEGDIR / name
        if fast.is_file() and full.is_file() and filecmp.cmp(fast, full, shallow=False):
            return True
    return False


def _work(case):
    cd = DATA / case
    full = measure_one(cd, SEGDIR)
    if _same_masks(cd):
        fast = {"error": "the fast pass is the full-res pass"}
    else:
        try:
            fast = measure_one(cd, "segmentations")
        except (RuntimeError, OSError, ValueError) as exc:   # what a mask read raises
            print(f"  [FAST-PASS ERROR {case}: {type(exc).__name__}: {exc}]", file=sys.stderr,
                  flush=True)
            fast = {"error": type(exc).__name__}
    # compare line (empty if EITHER pass lacks a usable IVC/LCIV mask)
    if "error" in fast or "error" in full:
        cmp = f"{case} empty mask"
    else:
        delta = full["min_3d_gap_mm"] - fast["min_3d_gap_mm"]
        cmp = (f"{case} "
               f"{fast['min_3d_gap_mm']:.2f} "
               f"{full['min_3d_gap_mm']:.2f} "
               f"{delta:+.2f} "
               f"{fast['ivc_voxels']} "
               f"{full['ivc_voxels']} "
               f"{full['qc_flag']}")
    # full-res line: surf_3d (primary) then raw_z_gap + c2c_3d + qc appended
    if "error" in full:
        fr = f"{case} ERROR: {full['error']}"
    else:
        fr = (f"{case} "
              f"{full['ivc_z_ext']:.0f} "
              f"{full['lciv_z_ext']:.0f} "
              f"{full['z_gap_mm']:.2f} "
              f"{full['min_3d_gap_mm']:.2f} "
              f"{full['raw_z_gap_mm']:+.2f} "
              f"{full['c2c_3d_gap_mm']:.2f} "
              f"{full['qc_flag']}")
    if "error" in full:
        spc = f"{case} NA NA NA NA NA NA"
    else:
        try:
            aux = case_aux_metrics(DATA / case, SEGDIR)
        except Exception as exc:
            print(f"  [AUX ERROR {case}: {type(exc).__name__}: {exc}]", file=sys.stderr,
                  flush=True)
            aux = {"lciv_lcc_frac": float("nan"), "lciv_n_components": 0,
                   "body_csa_mm2": float("nan")}
        def _f(x):
            return f"{x:.2f}" if isinstance(x, float) and math.isfinite(x) else "NA"
        spc = (f"{case} {full['sz_mm']:.3f} {full['sy_mm']:.3f} {full['sx_mm']:.3f} "
               f"{_f(aux['lciv_lcc_frac'])} {_aux_component_token(aux['lciv_n_components'])} "
               f"{_f(aux['body_csa_mm2'])}")
    return cmp, fr, spc


def main():
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument(
        "--jobs", type=int, default=None,
        help="worker processes. Default: resolved from the SLURM/cgroup CPU and MEMORY allocation, capped by the measured ~5.57 GiB per-worker peak. An explicit value is obeyed exactly.")
    ap.add_argument("--seg-subdir", default="segmentations_fullres",
                    help="subdirectory of each case dir holding the full-res per-structure "
                         "masks (default: segmentations_fullres). Pass '' for the "
                         "relevant-mask export, which writes them directly into <case>/. The "
                         "fast pass is read from <case>/segmentations/ either way, and a tree "
                         "without it is refused.")
    ap.add_argument("--roster", type=Path, default=None,
                    help="exhaustive case-id roster; defaults to the committed roster when "
                         "--prefix is amos, saros or autopet")
    args = ap.parse_args()
    global SEGDIR
    SEGDIR = args.seg_subdir
    DATA = Path(args.data_dir)
    if not DATA.is_dir():
        ap.error(f"cohort data directory does not exist: {DATA}")

    expected = load_expected_roster(args.prefix, args.roster)
    roster, cases, missing = case_roster(DATA, expected=expected)
    if expected is not None:
        discovered_masks = {(p.parent.parent.name if SEGDIR else p.parent.name) for p in
                            DATA.glob(_seg_glob("inferior_vena_cava.nii.gz"))}
        extras = sorted(discovered_masks - set(expected))
        if extras:
            print(f"ignoring {len(extras)} out-of-roster case dir(s): {extras[:10]}",
                  file=sys.stderr, flush=True)
    print(f"roster {len(roster)} attempted case dirs; produced full-res IVC mask "
          f"for {len(cases)}; {len(missing)} with NO mask -> emitted as NO_MASK "
          f"(closed exclusion accounting)", flush=True)
    print(f"measuring {len(cases)} cases from {DATA}", flush=True)

    cmp_path = f"{args.prefix}_results_compare.txt"
    fr_path = f"{args.prefix}_results_fullres.txt"
    spc_path = f"{args.prefix}_results_aux.txt"
    # Resolve BEFORE the Pool: `Pool(None)` falls back to `os.cpu_count()`, which is the node, not
    # the allocation, and ignores memory entirely -- the exact oversubscription this resolves.
    # `--jobs N` is obeyed exactly; the default is the min of the visible CPU allocation and what
    # the memory budget affords at the measured per-worker peak.
    jobs = HR.resolve_worker_count(requested=args.jobs)
    print(f"[resources] {HR.describe()} -> {jobs} worker(s)", flush=True)
    with Pool(jobs, initializer=_init_worker, initargs=(str(DATA), SEGDIR)) as pool, \
            AtomicTable(cmp_path) as fc, AtomicTable(fr_path) as ff, \
            AtomicTable(spc_path) as fs:
        measured = pairs = 0
        reasons = collections.Counter()
        for i, (cmp, fr, spc) in enumerate(pool.imap(work, cases), 1):
            fc.write(cmp)
            ff.write(fr)
            fs.write(spc)
            # A measurement row carries numbers where an error row carries `ERROR:` (full-res)
            # or `empty mask` (compare). The aux row is all NA exactly when the full-res row is
            # an error, so these two counts decide the shape of all three tables.
            if fr.split()[1:2] != ["ERROR:"]:
                measured += 1
            else:
                reason = fr.split(None, 2)[2] if len(fr.split()) > 2 else "unstated"
                reasons["missing files" if reason.startswith("missing files in ") else reason] += 1
            if cmp.split()[1:] != ["empty", "mask"]:
                pairs += 1
            if i % 25 == 0:
                fc.flush()
                ff.flush()
                fs.flush()
                print(f"  {i}/{len(cases)}", flush=True)
        # total-failure cases: emit explicit NO_MASK rows so the denominator is
        # closed over the whole attempted roster, not just cases that produced a mask
        for c in missing:
            fc.write(f"{c} empty mask")
            ff.write(f"{c} ERROR: NO_MASK")
            fs.write(f"{c} NA NA NA NA NA NA")
        if missing:
            reasons["NO_MASK"] += len(missing)
        # POSTCONDITION. Both refusals are raised INSIDE the AtomicTable block, so no `.part` is
        # published and any previous tables stand. Only an all-failure table is refused: the
        # committed cohorts mix measurements with errors, and a partial failure rate is data. The
        # rows are discarded with the `.part`, so the refusal names the reasons they carried.
        n = len(cases) + len(missing)
        if not measured:
            found = (f"a roster of {len(roster)}" if expected is not None else
                     f"{len(roster)} case directories found under {DATA}")
            why = ", ".join(f"{k} x{v}" for k, v in reasons.most_common(5)) or "no case at all"
            raise SystemExit(
                f"{fr_path}: nothing was measured -- 0 of {n} rows hold a measurement ({found}: "
                f"{len(missing)} without a full-res IVC mask under --seg-subdir={SEGDIR!r}; of the "
                f"rest, {len(cases)} read without a measurement; reasons {why}). None of the three "
                f"tables was published; any previous ones stand. Check --seg-subdir against the "
                f"export layout (flat 2.17.0 vs nested), and that each case holds readable IVC and "
                f"LCIV masks.")
        if not pairs:
            raise SystemExit(
                f"{cmp_path}: no fast-vs-full-res pair -- the full-res pass measured {measured} of "
                f"{n} cases, but none of them has a usable mask on the fast pass "
                f"(<case>/segmentations/) as well, so every compare row is `empty mask`; a fast "
                f"pass that reads the full-res masks themselves, by --seg-subdir, a link or a copy, "
                f"is not one. None of the three tables was published; any previous ones stand. This "
                f"driver reads both passes side by side, and a tree holding the full pass only is "
                f"incomplete: join the fast pass in per case (<case>/segmentations/).")
    print(f"done -> {cmp_path} , {fr_path} , {spc_path}", flush=True)


if __name__ == "__main__":
    main()
