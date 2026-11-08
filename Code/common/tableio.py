"""tableio: atomic, completion-marked publication for the Phase-1 result tables.

Two mechanisms, deliberately kept separate:

1. **Atomic publication.** Rows go to `<path>.part` in the same directory and are published
   with `os.replace`, which is atomic on POSIX. The canonical path therefore only ever holds
   the previous complete artifact or the new complete one -- never a half-written table.

2. **A completion sentinel.** A trailing `# complete N=<rows>` line, counting DATA rows
   only. Every Phase-1 parser skips a line whose first token is not case-id shaped, so the
   marker is invisible to the readers.

The count is declared TWICE, at both ends, and the pair is what makes a failure legible:

  * `# rows N=` immediately BEFORE THE FIRST DATA ROW, which truncation preserves;
  * `# complete N=` as the LAST line, which truncation destroys.

Head present and tail gone is TRUNCATION. Both present and disagreeing, or either disagreeing
with the rows actually counted, is an incomplete table. Neither present means completeness is
unverifiable.

This is deliberately NOT folded into `figio`: that module's contract is a matplotlib figure
plus per-suffix metadata stripping, and it can neither stream rows nor carry a sentinel.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Set

SENTINEL_PREFIX = "# complete N="     # LAST line; destroyed by truncation
HEADER_PREFIX = "# rows N="  # precedes the data; survives truncation
COMMENT = "#"

# Every in-flight table registers here so a second writer cannot target a path a first one is
# already publishing.
_IN_FLIGHT: Set[str] = set()


def fsync_directory(path) -> None:
    """Persist a directory ENTRY, which is what makes an ``os.replace`` survive a crash.

    ``os.replace`` is atomic on POSIX -- a reader sees the old file or the new one, never a mixture
    -- but atomicity is a statement about concurrent READERS, not about power loss. The rename is
    not on stable storage until the containing directory is fsynced; this makes a publication
    survive a crash, not only stay untorn."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _is_data_row(line):
    """A data row: non-blank, not a `#` comment, and not the bare column header.

    The header is recognised the way every Phase-1 parser already recognises it -- its
    first token is not case-id shaped (`case`, `case_id`) whereas a real row's is
    (`amos_0001`, `case_000`, `PETCT_<hex>_<date>-...`). Counting it would put the
    sentinel exactly one above every consumer's denominator.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith(COMMENT):
        return False
    first = stripped.split()[0]
    if first in _HEADER_FIRST_TOKENS:
        return False
    return True


# The literal first tokens the Phase-1 column headers use.
_HEADER_FIRST_TOKENS = frozenset({"case", "case_id"})


class AtomicTable:
    """Context manager that streams rows to `<path>.part` and publishes on clean exit.

    On an exception the partial file is removed and the canonical path is left untouched,
    so an interrupted run cannot be mistaken for a completed smaller cohort.
    """

    def __init__(self, path, flush_every=25):
        self.path = Path(path)
        self.part = self.path.with_name(self.path.name + ".part")
        self.flush_every = flush_every
        self.rows = 0
        self._header_lines = 0
        self._fh = None
        self._key = None

    def __enter__(self):
        key = str(self.path.resolve())
        if key in _IN_FLIGHT:
            raise RuntimeError(
                f"{self.path} is already being published by another AtomicTable in this "
                f"process; two writers on one canonical name means whoever finishes last "
                f"silently wins")
        # Claim the key only once the file is actually open.
        self.part.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.part.open("w", encoding="utf-8")
        _IN_FLIGHT.add(key)
        self._key = key
        return self

    def comment(self, text):
        """A `#` header line (e.g. `mode=labels`). Not counted as a data row."""
        self._fh.write(f"{COMMENT} {text}\n")

    def header(self, line):
        """A bare column-name header, written verbatim and NOT counted as a data row.

        `measure_bifurcation` writes `" ".join(COLUMNS)` as its first line. It is not a
        comment (it carries no `#`, and consumers skip it because its first token is not
        case-id shaped), but counting it would make the declared total exactly one more
        than every consumer's denominator -- a sentinel that disagrees with the readers is
        worse than no sentinel, because it makes a complete table look truncated.
        """
        self._header_lines += 1
        self._fh.write(str(line).rstrip("\n") + "\n")

    def write(self, row):
        self._fh.write(str(row).rstrip("\n") + "\n")
        self.rows += 1
        if self.flush_every and self.rows % self.flush_every == 0:
            self._fh.flush()

    def flush(self):
        self._fh.flush()

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                # The sentinel is written INSIDE the .part, before the rename, so a table
                # can never be published without one.
                self._fh.write(f"{SENTINEL_PREFIX}{self.rows}\n")
                self._fh.flush()
                os.fsync(self._fh.fileno())
                self._fh.close()
                self._fh = None
                # The row total is only known once the stream has ended, so this is a second pass
                # rather than a reserved slot -- a padded placeholder patched by seek() would couple
                # correctness to a field width, and these tables are ~10^3 lines, so a copy costs
                # nothing worth saving. It is published by a single os.replace, so the canonical
                # path holds either the previous complete artifact or the new one. PLACEMENT IS
                # LOAD-BEARING: the count sits immediately before the first data row, in the head,
                # so truncation -- which removes the tail -- cannot reach it.
                final = self.path.with_name(self.path.name + ".final")
                lines = self.part.read_text(encoding="utf-8").splitlines(keepends=True)
                cut = next((i for i, ln in enumerate(lines) if _is_data_row(ln)), len(lines))
                with final.open("w", encoding="utf-8") as dst:
                    dst.writelines(lines[:cut])
                    dst.write(f"{HEADER_PREFIX}{self.rows}\n")
                    dst.writelines(lines[cut:])
                    dst.flush()
                    os.fsync(dst.fileno())
                os.replace(final, self.path)
                # The bytes were durable before the rename; the NAME is not until the directory
                # entry is. Without this the guarantee is "never a torn table", not "the publication
                # survives a crash".
                fsync_directory(self.path.parent)
                self.part.unlink(missing_ok=True)
            else:
                self._fh.close()
                self._fh = None
                self.part.unlink(missing_ok=True)
                self.path.with_name(self.path.name + ".final").unlink(missing_ok=True)
        finally:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            if self._key is not None:
                _IN_FLIGHT.discard(self._key)
        return False


def results_path(prefix, suffix, limit=0):
    """The output filename for a driver run.

    A `--limit` run is a quick test over the first N cases and must never be able to land
    on the canonical name. The limit is carried in the FILENAME rather than only in a marker
    inside the file, so the artifact cannot be mistaken for the real one by anything that
    looks at paths -- which includes the roster validator and a human running `ls`.
    """
    if limit:
        return f"{prefix}_results_{suffix}.PARTIAL_limit{int(limit)}.txt"
    return f"{prefix}_results_{suffix}.txt"
