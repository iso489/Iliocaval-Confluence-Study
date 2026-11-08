"""How much CPU and memory this process may actually use, and how to stop BLAS stealing it.

Parallel entry points size their worker pools here and pin BLAS/OpenMP threads per worker.

Errors here are asymmetric: under-using a node costs wall-clock time, while over-committing memory
kills the job, so every estimate may be wrong low and must never be wrong high.

Cores are not the binding constraint on this workload; memory is, so `resolve_worker_count` takes
the smaller of the two limits.

WHAT THIS MODULE DELIBERATELY DOES NOT DO. It does not change any number the study publishes.
Worker count is result-neutral: every consumer collects results in submission order (`Pool.imap`
or `ProcessPoolExecutor.map`), and every seed is a fixed constant (`gap_config.SEED`, or a seed
given on the command line), never a worker id.
"""
from __future__ import annotations

import os
from pathlib import Path

try:                                              # optional, and treated as optional
    from threadpoolctl import threadpool_limits as _threadpool_limits
except Exception:                                 # pragma: no cover - absence is the tested path
    _threadpool_limits = None

#: Every thread-count variable a scientific Python stack reads. ONE of these left unset is
#: enough to oversubscribe: N worker processes each spawning N OpenMP threads is N^2 threads on
#: N cores, and the resulting contention is slower than running serially.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",            # OpenMP: scipy.ndimage, scikit-image, most compiled kernels
    "OPENBLAS_NUM_THREADS",       # OpenBLAS
    "MKL_NUM_THREADS",            # Intel MKL
    "VECLIB_MAXIMUM_THREADS",     # Apple Accelerate (macOS)
    "NUMEXPR_NUM_THREADS",        # numexpr, pulled in by pandas
    "BLIS_NUM_THREADS",           # BLIS (AMD AOCL)
)

#: 5.57 GiB at 512x512x400. It is the WORST case in the cohorts, not the median, because the
#: resolver must size for the case that OOMs the node rather than the case that does not.
DEFAULT_WORKER_PEAK_BYTES = int(5.57 * 1024**3)

#: Held back from the worker budget for the parent process, the Python runtime and page cache.
PARENT_RESERVE_BYTES = 4 * 1024**3

#: Used only when the memory budget cannot be determined at all. Small on purpose: an unknown
#: budget must not authorise a whole node. It is a fallback, not an estimate.
UNKNOWN_MEMORY_WORKER_CAP = 8


def _positive_int(value):
    """Parse a positive int, or return None. Never raises on operator/scheduler input."""
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def visible_cpu_count() -> int:
    """The number of CPUs this process may actually use.

    Order matters. ``os.sched_getaffinity`` is preferred over ``os.cpu_count`` because under cgroups
    the latter reports the machine, not the allocation."""
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        parsed = _positive_int(os.environ.get(var))
        if parsed is not None:
            return parsed
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:                        # not Linux
        return max(1, os.cpu_count() or 1)


def _read_int_file(path):
    try:
        return _positive_int(Path(path).read_text(encoding="utf-8"))
    except OSError:
        return None


def _cgroup_memory_limit():
    """The cgroup memory cap, or None. This is the limit that actually kills the process.

    cgroup v2 writes the literal ``max`` when uncapped and v1 writes a near-2^63 sentinel;
    both parse to "no limit" rather than to a number, so an uncapped cgroup must return None
    and not a meaningless 8 EiB.
    """
    v2 = _read_int_file("/sys/fs/cgroup/memory.max")
    if v2 is not None and v2 < 2**62:
        return v2
    v1 = _read_int_file("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if v1 is not None and v1 < 2**62:
        return v1
    return None


def _proc_meminfo_available():
    """``MemAvailable`` in bytes, or None. Reported in kB by the kernel."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                kb = _positive_int(line.split()[1])
                return kb * 1024 if kb is not None else None
    except (OSError, IndexError):
        return None
    return None


def available_memory_bytes():
    """The smallest binding memory limit in bytes, or **None when it cannot be determined**.

    None means UNKNOWN and must never be read as "plenty". A probe that returns a large number
    when it has no idea is worse than one that returns nothing, because the caller then cannot
    distinguish a real large-memory node from a failed lookup, and the failure mode is
    oversubscription. `resolve_worker_count` treats None as a reason to be small.

    SLURM reports its memory variables in MEGABYTES."""
    limits = []
    cgroup = _cgroup_memory_limit()
    if cgroup is not None:
        limits.append(cgroup)

    per_node = _positive_int(os.environ.get("SLURM_MEM_PER_NODE"))
    if per_node is not None:
        limits.append(per_node * 1024**2)
    else:
        per_cpu = _positive_int(os.environ.get("SLURM_MEM_PER_CPU"))
        if per_cpu is not None:
            limits.append(per_cpu * 1024**2 * visible_cpu_count())

    if not limits:
        meminfo = _proc_meminfo_available()
        if meminfo is not None:
            limits.append(meminfo)

    return min(limits) if limits else None


def resolve_worker_count(per_worker_bytes: int = DEFAULT_WORKER_PEAK_BYTES,
                         requested=None, cpu_cap=None) -> int:
    """How many `Pool` workers fit, by CORES **and** by MEMORY, never fewer than one.

    `requested` is the operator speaking through ``--jobs`` and is obeyed exactly, including
    above the safe estimate: `per_worker_bytes` is a worst-case heuristic and the operator may
    know the real footprint of the cohort being run. It is validated, not silently corrected --
    ``--jobs 0`` is a mistake worth reporting, and `Pool(0)` raises anyway.
    """
    if not isinstance(per_worker_bytes, int) or per_worker_bytes <= 0:
        raise ValueError(f"per_worker_bytes must be a positive int, got {per_worker_bytes!r}")
    if requested is not None:
        parsed = _positive_int(requested)
        if parsed is None:
            raise ValueError(f"--jobs must be a positive integer, got {requested!r}")
        return parsed

    cpus = visible_cpu_count() if cpu_cap is None else max(1, int(cpu_cap))
    budget = available_memory_bytes()
    if budget is None:
        # UNKNOWN, not unlimited.
        return max(1, min(cpus, UNKNOWN_MEMORY_WORKER_CAP))
    usable = max(0, budget - PARENT_RESERVE_BYTES)
    return max(1, min(cpus, usable // per_worker_bytes))


def limit_blas_threads(n: int = 1):
    """Pin every BLAS/OpenMP backend to `n` threads. Call in the `Pool` initializer.

    TWO MECHANISMS, because one is not enough. The environment variables are the floor, but
    they are only read when the backend LOADS -- in a forked worker numpy is already imported,
    so setting them there alone does nothing to the running library. `threadpoolctl` re-limits
    an already-loaded backend, which is what actually works on the Pool path. It is optional:
    a missing optional dependency must not take a cluster job down, so its absence degrades to
    the variables rather than raising.
    """
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError(f"thread count must be a positive int, got {n!r}")
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(n)
    if _threadpool_limits is not None:
        try:
            _threadpool_limits(limits=n)
        except Exception:                         # pragma: no cover - backend-specific
            pass                                  # the variables above still stand
    return n


def describe() -> dict:
    """A one-line resource summary for run logs, so a slow run is diagnosable after the fact."""
    budget = available_memory_bytes()
    return {
        "visible_cpus": visible_cpu_count(),
        "memory_budget_bytes": budget,
        "memory_budget_gib": None if budget is None else round(budget / 1024**3, 1),
        "per_worker_peak_gib": round(DEFAULT_WORKER_PEAK_BYTES / 1024**3, 2),
        "resolved_workers": resolve_worker_count(),
        "threadpoolctl": _threadpool_limits is not None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
