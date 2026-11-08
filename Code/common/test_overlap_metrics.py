"""Synthetic masks with KNOWN geometry validate every core."""

import numpy as np
import pytest

import overlap_metrics as M
import gap_config as C


# --- topology beyond Betti-0 (loops and cavities) --------------------------
def _ball(shape=(26, 26, 26), c=13, r=8):
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return ((z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2) <= r * r


def _solid_torus(shape=(26, 26, 26), c=13, big=7.0, small=3.0):
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return ((np.sqrt((y - c) ** 2 + (x - c) ** 2) - big) ** 2 + (z - c) ** 2) <= small * small


def _hollow_shell(shape=(26, 26, 26), c=13, outer=9, inner=6):
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    d2 = (z - c) ** 2 + (y - c) ** 2 + (x - c) ** 2
    return (d2 <= outer * outer) & (d2 >= inner * inner)


def test_betti_numbers_of_solids_with_known_topology():
    """b0 components, b1 handles/tunnels, b2 enclosed cavities, via chi = b0 - b1 + b2.

    A solid ball is contractible (chi = 1), a solid torus has one handle (chi = 0), and a
    hollow shell encloses one cavity (chi = 2). These three cases separate the three Betti
    numbers from one another, so they pin the identity rather than a single number.
    """
    assert M.betti_numbers(_ball()) == (1, 0, 0)
    assert M.betti_numbers(_solid_torus()) == (1, 1, 0)
    assert M.betti_numbers(_hollow_shell()) == (1, 0, 1)


# =============================================================================================
# Input guards and the connectivity contract
# =============================================================================================


def _all_2x2x2_configurations():
    """All 256 two-by-two-by-two masks, each padded into a 4x4x4 box so no face touches the
    array edge (which would change the background-component bookkeeping)."""
    for bits in range(256):
        arr = np.array([(bits >> i) & 1 for i in range(8)], bool).reshape(2, 2, 2)
        padded = np.zeros((4, 4, 4), bool)
        padded[1:3, 1:3, 1:3] = arr
        yield bits, padded


def test_every_connectivity_either_is_refused_or_never_returns_a_negative_betti_number():
    """A Betti number counts something, so it cannot be negative.

    For EACH adjacency the module recognises it must refuse or be non-negative over all 256
    two-by-two-by-two configurations; the test asks 18 as well and accepts a refusal. Mixing
    18-connected components, a 26-connected background (the compatible dual is 6) and a
    6-connectivity Euler characteristic gives a NEGATIVE b1 for **84** of the 256.
    """
    refused, served = [], []
    for conn in (6, 18, 26):
        try:
            M.betti_numbers(np.ones((4, 4, 4), bool), conn)
        except ValueError:
            refused.append(conn)
            continue
        served.append(conn)
        for bits, mask in _all_2x2x2_configurations():
            b0, b1, b2 = M.betti_numbers(mask, conn)
            assert b0 >= 0 and b1 >= 0 and b2 >= 0, (
                f"conn={conn} configuration {bits:08b} gives Betti {(b0, b1, b2)}, "
                "which no space has; a connectivity that cannot be served must RAISE")
    assert served, "the module must serve at least one connectivity"
    assert C.OVERLAP_CONNECTIVITY in served, (
        f"the default OVERLAP_CONNECTIVITY = {C.OVERLAP_CONNECTIVITY} must be a served adjacency")
    assert refused == [18], (
        f"expected exactly 18 to be refused (no 3-D Euler number exists for it), got {refused}")


def _shell_with_a_corner_pinhole():
    """A cubic shell whose only opening is DIAGONAL, the textbook Jordan case.

    A round shell will not do this, and neither will a one-voxel-thick wall: removing any single
    voxel from a 1-thick wall already opens a straight, face-connected hole. So the wall here is
    exactly two layers, and one voxel is removed from EACH layer, positioned corner-to-corner:

        solid  [2:9]^3, cavity [4:7]^3   ->  a two-voxel-thick wall
        remove (2, 6, 6)  from the outer layer   -> its -z neighbour (1, 6, 6) is outside
        remove (3, 5, 5)  from the inner layer   -> its +z neighbour (4, 5, 5) is cavity

    The two holes differ by (1, 1, 1), so background can pass from cavity to outside under
    26-adjacency and cannot under 6. That is the entire discrimination this fixture exists for,
    and the test asserts it holds before relying on it.
    """
    shell = np.zeros((11, 11, 11), bool)
    shell[2:9, 2:9, 2:9] = True
    shell[4:7, 4:7, 4:7] = False
    shell[2, 6, 6] = False
    shell[3, 5, 5] = False
    return shell


def test_enclosed_cavities_uses_a_jordan_compatible_dual():
    """b2 must pair the object's adjacency with the compatible background one.

    The Jordan-compatible pairs in Z^3 are (6,26), (26,6) and (18,6). A plain hollow shell gives
    one cavity under EVERY pair, compatible or not, so it cannot tell a compatible map from an
    incompatible one; a shell with a corner pinhole discriminates, and it is checked below to
    actually do so before it is relied on.
    """
    shell = _hollow_shell()
    assert M._enclosed_cavities(shell, 26) == 1
    assert M._enclosed_cavities(shell, 6) == 1
    assert M._enclosed_cavities(shell, 18) == 1      # 18 -> 6-connected background

    # The discriminating fixture. Its premise is asserted, not assumed: if a future skimage or
    # scipy changed this, the test would silently stop pinning anything.
    pin = _shell_with_a_corner_pinhole()
    from scipy.ndimage import label as _label
    esc26 = _label(~pin, M._structure(26, 3))[1]
    esc6 = _label(~pin, M._structure(6, 3))[1]
    assert esc6 > esc26, (
        "fixture must let a 26-connected background escape the pinhole while a 6-connected one "
        f"cannot ({esc6} vs {esc26} background components)")
    # conn=26 -> dual 6 -> the cavity is still enclosed; a 26-dual would let it escape
    assert M._enclosed_cavities(pin, 26) == 1
    assert M._enclosed_cavities(pin, 18) == 1        # 18 -> dual 6, same answer
    # conn=6 -> dual 26 -> the background escapes and there is no enclosed cavity
    assert M._enclosed_cavities(pin, 6) == 0

    for bad in (0, 7, "26"):
        with pytest.raises(ValueError, match="must be one of 6, 18, 26"):
            M._enclosed_cavities(shell, bad)


def test_betti_numbers_at_conn_6_are_pinned_to_known_topology():
    """The 6-connectivity Euler convention, which nothing else in the suite touches.

    `test_betti_numbers_of_solids_with_known_topology` calls `betti_numbers` at the default (26)
    only, and the exhaustive sweep above checks non-negativity -- which catches a too-FINE Euler
    number for a coarse adjacency but is blind to the opposite.

    Analytic truth, so this is an oracle and not a regression lock: a solid ball is one component
    with no handle and no cavity; a solid torus has one handle; a hollow shell encloses one
    cavity. All three hold under either adjacency, which is what makes them usable here.
    """
    assert M.betti_numbers(_ball(), 6) == (1, 0, 0)
    assert M.betti_numbers(_solid_torus(), 6) == (1, 1, 0)
    assert M.betti_numbers(_hollow_shell(), 6) == (1, 0, 1)
    # two solids that touch only at a corner are TWO components under 6-connectivity and one
    # under 26, with no handle either way -- a case where the two adjacencies must disagree
    corner = np.zeros((9, 9, 9), bool)
    corner[1:4, 1:4, 1:4] = True
    corner[4:7, 4:7, 4:7] = True
    assert M.betti_numbers(corner, 6) == (2, 0, 0)
    assert M.betti_numbers(corner, 26) == (1, 0, 0)
