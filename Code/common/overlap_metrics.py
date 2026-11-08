#!/usr/bin/env python3
"""Topology of binary masks: connected components and Betti numbers.

n_components counts components at 6-, 18- or 26-connectivity (default
gap_config.OVERLAP_CONNECTIVITY). betti_numbers returns (b0, b1, b2): components, handles and
enclosed cavities, with b2 counted under the Jordan-compatible dual background adjacency and b1
from the Euler characteristic; it serves 6- and 26-connectivity only. The cores need numpy, scipy
and scikit-image and are unit-tested in test_overlap_metrics.py.
"""
from __future__ import annotations

import sys

import numpy as np

# --- run-from-anywhere: put sibling phase dirs on sys.path (tests use Code/conftest.py) ---
import sys as _sys, pathlib as _pl
for _p in sorted(_pl.Path(__file__).resolve().parent.parent.iterdir()):
    if _p.is_dir() and _p.name[0] not in "._":
        if str(_p) not in _sys.path:
            _sys.path.append(str(_p))  # append, never shadow stdlib
import gap_config as C


def _structure(conn, ndim=3):
    """A binary structuring element for the requested 3-D connectivity (6 / 18 / 26).

    Unknown values raise rather than fall back to a default connectivity."""
    from scipy.ndimage import generate_binary_structure
    rank = {6: 1, 18: 2, 26: 3}.get(conn) if isinstance(conn, int) else None
    if rank is None:
        raise ValueError(
            f"connectivity must be one of 6, 18, 26 (got {conn!r})")
    return generate_binary_structure(ndim, rank)


def n_components(mask, conn=None) -> int:
    """Number of connected components in a boolean mask (3-D `conn`-connectivity)."""
    conn = C.OVERLAP_CONNECTIVITY if conn is None else conn
    from scipy.ndimage import label
    mask = np.asarray(mask, bool)
    if not mask.any():
        return 0
    return int(label(mask, _structure(conn, mask.ndim))[1])


def _enclosed_cavities(mask, conn=None) -> int:
    """Background components fully enclosed by the object (Betti-2).

    Uses the DUAL connectivity: a 26-connected object must be paired with a 6-connected background,
    otherwise the Jordan property fails and a diagonal background thread can "escape" through a
    solid wall. The Jordan-compatible pairs in Z^3 are (6,26), (26,6) and (18,6); the map below is
    exactly those."""
    from scipy.ndimage import label
    conn = C.OVERLAP_CONNECTIVITY if conn is None else conn
    _structure(conn, 3)                       # reject an unsupported connectivity here, not later
    dual = {6: 26, 18: 6, 26: 6}[conn]
    mask = np.asarray(mask, bool)
    lab, n = label(~mask, _structure(dual, mask.ndim))
    if n == 0:
        return 0
    touching = set()
    for axis in range(mask.ndim):
        for end in (0, -1):
            index = [slice(None)] * mask.ndim
            index[axis] = end
            touching.update(np.unique(lab[tuple(index)]).tolist())
    touching.discard(0)
    return int(n - len(touching))


def betti_numbers(mask, conn=None) -> tuple:
    """``(b0, b1, b2)``: connected components, handles/tunnels, and enclosed cavities.

    Betti-0 alone cannot see a false shortcut. A Y-confluence with a spurious iliac-to-iliac
    bypass has the same component count as the true anatomy, so Betti-0 and face connectivity
    cannot see it, while the bypass is a handle that b1 counts (Clough et al. warn that Betti-0
    pressure makes a model "join together using a small connecting region").

    b1 comes from the Euler characteristic, chi = b0 - b1 + b2, which is exact for a binary
    3-D image and needs no persistent-homology dependency.

    ONLY 6- AND 26-CONNECTIVITY ARE SERVED, and `conn=18` raises. A Betti triple is only meaningful
    when b0, b2 and chi all come from ONE Jordan-compatible adjacency pair. Mixing adjacencies at
    18 is not merely approximate: over the 256 two-by-two-by-two configurations it gives a
    **negative b1 in 84 of them**, which no space has.

    18 is refused because no consistent computation is available: b1 at 18-connectivity needs an
    18-connectivity Euler number, and scikit-image raises `NotImplementedError: For 3D images, Euler
    number is implemented for connectivities 1 and 3 only`. `n_components(mask, 18)` is valid
    (scipy labels at 18); only the Betti path refuses 18, and the default
    `OVERLAP_CONNECTIVITY = 26` is served.
    """
    from skimage.measure import euler_number
    conn = C.OVERLAP_CONNECTIVITY if conn is None else conn
    euler_conn = {6: 1, 26: 3}.get(conn if isinstance(conn, int) else None)
    if euler_conn is None:
        _structure(conn, 3)                   # unsupported value -> the module's standard message
        raise ValueError(
            f"betti_numbers is defined for connectivity 6 and 26, not {conn!r}: "
            "b0, b2 and the Euler characteristic must come from one Jordan-compatible adjacency "
            "pair, and scikit-image implements no 3-D Euler number at 18-connectivity. "
            "n_components() does accept 18.")
    mask = np.asarray(mask, bool)
    if not mask.any():
        return (0, 0, 0)
    b0 = n_components(mask, conn)
    b2 = _enclosed_cavities(mask, conn)
    chi = int(euler_number(mask, connectivity=euler_conn))
    return (int(b0), int(b0 + b2 - chi), int(b2))


