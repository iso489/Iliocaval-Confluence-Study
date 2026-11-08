"""figio: deterministic, atomic figure publication for the Phase-1 figures (stats.py).

Two properties every committed figure artifact needs and plain ``Figure.savefig``
does not provide:

* **Byte determinism.** Matplotlib embeds a wall-clock ``/CreationDate`` in PDF
  (and a writer tag in PNG/SVG), so an unchanged figure would get different
  bytes on every render. Wall-clock metadata is dropped here, so identical
  content gives identical bytes in the pinned environment.
* **Atomic publication.** ``savefig`` writes in place, so an interrupted run can
  leave a torn, well-formed-looking artifact at the target path. Here the render goes
  to a temporary file in the SAME directory and is published with ``os.replace``
  (atomic on POSIX), so the target path only ever holds the old artifact or the
  complete new one.
"""
import os
import tempfile
from pathlib import Path

from tableio import fsync_directory

# wall-clock / environment-dependent metadata dropped per format; a None value
# tells matplotlib to omit the field entirely
_STRIP = {
    ".pdf": {"CreationDate": None, "ModDate": None},
    ".png": {"Software": None},
    ".svg": {"Date": None},
}


def savefig_atomic(fig, path, **savefig_kw):
    """``fig.savefig(path)`` with wall-clock metadata dropped and an atomic
    temp-then-``os.replace`` publication. Extra keywords (``dpi``,
    ``bbox_inches``, ...) pass through; a caller-supplied ``metadata`` dict is
    merged OVER the determinism defaults, so callers can add fields but only
    override the stripping deliberately."""
    path = Path(path)
    metadata = dict(_STRIP.get(path.suffix.lower(), {}))
    metadata.update(savefig_kw.pop("metadata", None) or {})
    fd, tmp = tempfile.mkstemp(dir=str(path.parent),
                               prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        fig.savefig(tmp, format=path.suffix.lstrip(".").lower(),
                    metadata=metadata, **savefig_kw)
        # `savefig` has already closed the file, so the payload is re-opened read-only purely to
        # flush it.
        fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
        fsync_directory(path.parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
