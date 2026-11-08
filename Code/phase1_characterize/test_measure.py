"""Masks are written as tiny NIfTI volumes so measure_one's real I/O path is exercised."""
import numpy as np
import pytest
import SimpleITK as sitk

import gap_config as C
from measure_ivc_lciv_gap import surface_gap, measure_one

# (sz, sy, sx) array-axis spacing used by surface_gap / measure_one
SP = (2.0, 1.0, 1.0)


def _write(path, arr, spacing_xyz=(1.0, 1.0, 2.0), direction=None):
    img = sitk.GetImageFromArray(arr.astype(np.uint8))   # array is (z,y,x)
    img.SetSpacing(spacing_xyz)                           # sitk wants (sx,sy,sz)
    if direction is not None:                             # row-major 3x3 (sitk order)
        img.SetDirection([float(d) for d in direction])
    sitk.WriteImage(img, str(path))


def _case(tmp_path, ivc, lciv, ril=None, spacing=(1.0, 1.0, 2.0), direction=None):
    seg = tmp_path / "seg"
    seg.mkdir(parents=True, exist_ok=True)
    _write(seg / "inferior_vena_cava.nii.gz", ivc, spacing, direction)
    _write(seg / "iliac_vena_left.nii.gz", lciv, spacing, direction)
    if ril is not None:
        _write(seg / "iliac_vena_right.nii.gz", ril, spacing, direction)
    return tmp_path


# Direction matrices (sitk row-major). Identity = standard positive-z (cranial = +z).
_IDENTITY_DIR = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
# Reversed z: D[2,2] = -1 -> the ARRAY INDEX k runs caudal-to-cranial. Physical z is
# still Superior (cranial+); only which index is cranial flips (NOT the z meaning).
_REVZ_DIR = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0)


# --- surface_gap geometry -------------------------------------------------
def test_surface_gap_touching_is_zero():
    A = np.zeros((5, 5, 5), bool); B = np.zeros((5, 5, 5), bool)
    A[1, 2, 2] = True; B[2, 2, 2] = True          # face-adjacent along z
    surf, c2c = surface_gap(A, B, SP)
    assert c2c == pytest.approx(2.0)               # one slice apart by centers
    assert surf == pytest.approx(0.0, abs=1e-9)    # walls touch


def test_surface_gap_one_slice_void():
    A = np.zeros((6, 5, 5), bool); B = np.zeros((6, 5, 5), bool)
    A[1, 2, 2] = True; B[3, 2, 2] = True
    surf, c2c = surface_gap(A, B, SP)
    assert c2c == pytest.approx(4.0)
    assert surf == pytest.approx(2.0)              # c2c - one slice thickness


def test_surface_gap_in_plane_adjacent_is_zero():
    A = np.zeros((5, 5, 5), bool); B = np.zeros((5, 5, 5), bool)
    A[2, 2, 1] = True; B[2, 2, 2] = True           # adjacent along x (sx=1)
    surf, c2c = surface_gap(A, B, SP)
    assert c2c == pytest.approx(1.0)
    assert surf == pytest.approx(0.0, abs=1e-9)


def test_surface_never_exceeds_c2c():
    A = np.zeros((8, 8, 8), bool); B = np.zeros((8, 8, 8), bool)
    A[1, 1, 1] = True; B[6, 6, 6] = True
    surf, c2c = surface_gap(A, B, SP)
    assert surf <= c2c + 1e-9


def test_surface_gap_diagonal_anisotropic_exact():
    # offset (1,3,0) voxels at spacing (sz,sy,sx)=(5,1,1): true box gap is
    # ||(max(1-1,0)*5, max(3-1,0)*1, 0)|| = 2.0; the dilated-IVC box distance is exact for
    # oblique/anisotropic pairs
    A = np.zeros((8, 10, 8), bool); B = np.zeros((8, 10, 8), bool)
    A[1, 2, 2] = True; B[2, 5, 2] = True
    surf, c2c = surface_gap(A, B, (5.0, 1.0, 1.0))
    assert surf == pytest.approx(2.0, abs=1e-9)
    assert c2c == pytest.approx(np.hypot(5.0, 3.0))


# --- measure_one ----------------------------------------------------------
def test_positive_gap_and_invariants(tmp_path):
    ivc = np.zeros((10, 8, 8), bool); ivc[6:9, 3:5, 3:5] = True
    lciv = np.zeros((10, 8, 8), bool); lciv[1:3, 3:5, 3:5] = True
    r = measure_one(_case(tmp_path, ivc, lciv), "seg")
    assert "error" not in r
    assert r["z_gap_mm"] > 0
    assert r["min_3d_gap_mm"] <= r["c2c_3d_gap_mm"] + 1e-9      # surf <= c2c
    assert r["c2c_3d_gap_mm"] >= r["z_gap_mm"] - C.INVARIANT_SLACK_MM


def test_measure_one_rejects_same_shape_different_physical_grids(tmp_path):
    ivc = np.zeros((10, 8, 8), bool); ivc[6:9, 3:5, 3:5] = True
    lciv = np.zeros((10, 8, 8), bool); lciv[1:3, 3:5, 3:5] = True
    seg = tmp_path / "seg"; seg.mkdir()
    _write(seg / "inferior_vena_cava.nii.gz", ivc, spacing_xyz=(1.0, 1.0, 2.0))
    _write(seg / "iliac_vena_left.nii.gz", lciv, spacing_xyz=(1.0, 1.0, 3.0))
    result = measure_one(tmp_path, "seg")
    assert result.get("error") == "IVC/LCIV grid mismatch"


def test_primary_euclidean_gap_is_rotation_invariant_on_tilted_grid(tmp_path):
    ivc = np.zeros((10, 8, 8), bool); ivc[6:9, 3:5, 3:5] = True
    lciv = np.zeros((10, 8, 8), bool); lciv[1:3, 3:5, 3:5] = True
    axial = measure_one(_case(tmp_path / "axial", ivc, lciv, direction=_IDENTITY_DIR), "seg")
    tilted = measure_one(_case(tmp_path / "tilted", ivc, lciv, direction=_TILT_DIR), "seg")
    assert "error" not in tilted
    # A global orthonormal rotation changes coordinates, not Euclidean center/box distances.
    assert tilted["min_3d_gap_mm"] == pytest.approx(axial["min_3d_gap_mm"], abs=1e-9)
    assert tilted["c2c_3d_gap_mm"] == pytest.approx(axial["c2c_3d_gap_mm"], abs=1e-9)


def test_tilted_grid_is_measured_and_flagged_oblique_not_failed_closed(tmp_path):
    """A tilted orthonormal grid is measured and flagged OBLIQUE, with the same gap as the
    axis-aligned grid; the endpoint does not fail closed on it."""
    ivc = np.zeros((10, 8, 8), bool); ivc[6:9, 3:5, 3:5] = True
    lciv = np.zeros((10, 8, 8), bool); lciv[1:3, 3:5, 3:5] = True

    axial = measure_one(_case(tmp_path / "axial", ivc, lciv, direction=_IDENTITY_DIR), "seg")
    tilted = measure_one(_case(tmp_path / "tilted", ivc, lciv, direction=_TILT_DIR), "seg")

    # (1) measured, not refused
    assert "error" not in tilted, "a tilted orthonormal grid must be MEASURED, not failed closed"
    assert "OBLIQUE" in tilted["qc_flag"]
    # (2) informational only: obliquity does not move the number
    assert tilted["min_3d_gap_mm"] == pytest.approx(axial["min_3d_gap_mm"], abs=1e-9)
    # (3) the flag discriminates -- an axis-aligned acquisition is not marked oblique
    assert "OBLIQUE" not in axial["qc_flag"]


def test_signed_axis_permutation_is_measured_and_not_flagged_oblique(tmp_path):
    """A signed axis permutation is orthonormal AND axis-aligned, so it is measured and clean.

    This is the boundary of the `OBLIQUE` predicate, and the reason it is defined as "orthonormal
    but not a signed permutation" rather than "not the identity"."""
    ivc = np.zeros((10, 8, 8), bool); ivc[6:9, 3:5, 3:5] = True
    lciv = np.zeros((10, 8, 8), bool); lciv[1:3, 3:5, 3:5] = True
    perm = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0)          # y <-> z, det = -1
    r = measure_one(_case(tmp_path / "perm", ivc, lciv, direction=perm), "seg")
    assert "error" not in r
    assert "OBLIQUE" not in r["qc_flag"]


def test_overlap_is_censored_but_recoverable(tmp_path):
    # IVC and LCIV overlap in z -> z_gap clamps to 0, raw stays negative
    ivc = np.zeros((10, 8, 8), bool); ivc[2:8, 1:3, 1:3] = True
    lciv = np.zeros((10, 8, 8), bool); lciv[3:9, 5:7, 5:7] = True
    r = measure_one(_case(tmp_path, ivc, lciv), "seg")
    assert r["z_gap_mm"] == pytest.approx(0.0)
    assert r["raw_z_gap_mm"] < 0                  # sign preserved (recoverable)


def test_degenerate_mask_flagged(tmp_path):
    ivc = np.zeros((10, 8, 8), bool); ivc[6:9, 3:5, 3:5] = True
    lciv = np.zeros((10, 8, 8), bool); lciv[1, 3, 3] = True   # 1 voxel << QC_MIN
    r = measure_one(_case(tmp_path, ivc, lciv), "seg")
    assert "SMALL_LCIV" in r["qc_flag"]


def test_empty_mask_errors(tmp_path):
    ivc = np.zeros((10, 8, 8), bool); ivc[6:9, 3:5, 3:5] = True
    lciv = np.zeros((10, 8, 8), bool)            # empty
    r = measure_one(_case(tmp_path, ivc, lciv), "seg")
    assert "error" in r


def test_mask_overlap_is_flagged(tmp_path):
    # IVC and LCIV share voxels (segmentation error) -> OVERLAP qc, gap 0
    ivc = np.zeros((10, 8, 8), bool); ivc[4:8, 3:6, 3:6] = True
    lciv = np.zeros((10, 8, 8), bool); lciv[5:9, 4:7, 4:7] = True   # overlaps IVC
    r = measure_one(_case(tmp_path, ivc, lciv), "seg")
    assert "OVERLAP" in r["qc_flag"]
    assert r["min_3d_gap_mm"] == pytest.approx(0.0)


def test_large_gap_is_flagged_implausible(tmp_path):
    # A surface gap above QC_MAX_GAP_MM is physiologically impossible
    # and must surface in the PRODUCER qc_flag, not silently read "OK". A large z-spacing makes
    # a tiny volume yield a > 200 mm surface gap; masks are >= QC_MIN_VOXELS so no SMALL flag.
    big = (1.0, 1.0, 40.0)                                          # sz = 40 mm
    ivc = np.zeros((13, 10, 10), bool); ivc[10:13, 2:7, 2:7] = True   # cranial (75 vox)
    lciv = np.zeros((13, 10, 10), bool); lciv[0:3, 2:7, 2:7] = True   # caudal, same (y,x) (75 vox)
    r = measure_one(_case(tmp_path, ivc, lciv, spacing=big), "seg")
    assert r["min_3d_gap_mm"] > C.QC_MAX_GAP_MM                     # the surface gap is implausible
    assert "LARGE_GAP_IMPLAUSIBLE" in r["qc_flag"]


def test_normal_gap_is_not_flagged_implausible(tmp_path):
    # negative control: a plausible confluence gap must NOT trip the large-gap flag (and, with
    # >= QC_MIN_VOXELS masks and no overlap/reversed-z, the token stays "OK").
    ivc = np.zeros((12, 10, 10), bool); ivc[8:11, 2:7, 2:7] = True    # 75 vox
    lciv = np.zeros((12, 10, 10), bool); lciv[1:4, 2:7, 2:7] = True   # 75 vox, ~10 mm gap
    r = measure_one(_case(tmp_path, ivc, lciv), "seg")               # default sz = 2 mm
    assert r["min_3d_gap_mm"] < C.QC_MAX_GAP_MM
    assert "LARGE_GAP_IMPLAUSIBLE" not in r["qc_flag"]
    assert r["qc_flag"] == "OK"


# --- cranial = MAX PHYSICAL Z, always (SimpleITK LPS) ----------------------- In SimpleITK's LPS
# world frame the third physical axis is Superior (cranial+), so the most-cranial voxel is simply
# the one with the LARGEST physical z, independent of the direction matrix. A negative D[2,2] only
# reverses which ARRAY INDEX k is cranial, NOT the physical-z meaning. Reversed-z storage is present
# in SAROS, so this is part of the exercised cohort contract.)
def _ivc_above_lciv():
    # IVC at high k, LCIV at low k. Under the IDENTITY direction (physical z = +2k)
    # this puts the IVC physically CRANIAL (higher z); 6 void slices -> (11-4)*2 = 14 mm.
    ivc = np.zeros((20, 8, 8), bool); ivc[11:14, 3:5, 3:5] = True
    lciv = np.zeros((20, 8, 8), bool); lciv[2:5, 3:5, 3:5] = True
    return ivc, lciv


def test_reversed_z_cranial_from_physical_z(tmp_path):
    # Under REVERSED z (physical z = -2k) the IVC is physically cranial only when it
    # sits at LOW k. The gap is read off PHYSICAL z, not the array index: IVC at k=2,3,4
    # (physical z -4,-6,-8) is cranial of LCIV at k=11,12,13 (physical z -22,-24,-26).
    # caudal-most IVC z=-8 (k=4), cranial-most LCIV z=-22 (k=11) -> (11-4)*2 = 14 mm > 0.
    ivc = np.zeros((20, 8, 8), bool); ivc[2:5, 3:5, 3:5] = True
    lciv = np.zeros((20, 8, 8), bool); lciv[11:14, 3:5, 3:5] = True
    r = measure_one(_case(tmp_path, ivc, lciv, direction=_REVZ_DIR), "seg")
    assert "error" not in r
    assert r["z_gap_mm"] == pytest.approx(14.0)        # positive, from physical z
    assert r["raw_z_gap_mm"] == pytest.approx(14.0)
    assert "REVERSED_Z" in r["qc_flag"]                # informational QC flag still fires


def test_gap_is_physical_relationship_invariant(tmp_path):
    # Mirroring the array in k AND flipping the direction is a physical no-op for the
    # ANATOMY relationship (same patient stored head-first vs feet-first): the gap, a
    # physical separation, is invariant. Identity (IVC high-k) vs reversed-z with the
    # array mirrored in k both place the IVC 14 mm cranial of the LCIV.
    iv, lv = _ivc_above_lciv()
    r_id = measure_one(_case(tmp_path / "id", iv, lv, direction=_IDENTITY_DIR), "seg")
    iv_m, lv_m = iv[::-1].copy(), lv[::-1].copy()      # mirror in k
    r_rz = measure_one(_case(tmp_path / "rz", iv_m, lv_m, direction=_REVZ_DIR), "seg")
    assert r_id["z_gap_mm"] == pytest.approx(14.0)
    assert r_rz["z_gap_mm"] == pytest.approx(r_id["z_gap_mm"])
    # PRIMARY surface metric is symmetric -> identical across orientations too.
    assert r_id["min_3d_gap_mm"] == pytest.approx(r_rz["min_3d_gap_mm"])


def test_reversed_z_qc_flag_fires_only_when_reversed(tmp_path):
    # a REVERSED_Z QC flag must surface scans with D[2,2] < 0, and must NOT
    # fire for a standard identity / positive-z orientation.
    iv, lv = _ivc_above_lciv()
    r_id = measure_one(_case(tmp_path / "id", iv, lv, direction=_IDENTITY_DIR), "seg")
    r_rz = measure_one(_case(tmp_path / "rz", iv, lv, direction=_REVZ_DIR), "seg")
    assert "REVERSED_Z" not in r_id["qc_flag"]
    assert "REVERSED_Z" in r_rz["qc_flag"]


# A tilted direction (rotation about x by 30 deg) under which physical z depends on
# the row index j, so the most-cranial PRESENT voxel of a mask need NOT lie on the
# highest array-index slice. This is the case where a raw-index choice, ztop = ks.max(),
# picks the WRONG (non-cranial) slice and the physical-cranial criterion is
# required. Row-major sitk direction; det = +1 (proper rotation).
_TH = np.deg2rad(30.0)
_TILT_DIR = (1.0, 0.0, 0.0,
             0.0, float(np.cos(_TH)), float(-np.sin(_TH)),
             0.0, float(np.sin(_TH)), float(np.cos(_TH)))


def test_aux_picks_cranial_lciv_slice_by_physical_direction(tmp_path):
    # aux samples body CSA at the LCIV "most-cranial" slice. Under a tilted
    # gantry physical z = sin(th)*j*sy + cos(th)*k*sz, so a low-k voxel at a large
    # row j can be MORE cranial than the highest-k voxel. A raw-index ztop = ks.max()
    # would sample the wrong end; the physical-cranial criterion samples the right one.
    from measure_ivc_lciv_gap import case_aux_metrics
    seg = tmp_path / "seg"; seg.mkdir()
    # LCIV: a cranial fragment at low k but high row j, and a caudal fragment at
    # high k but low row j. With th=30deg, sy=sz=1: phys z(k=4,j=24)=sin30*24+cos30*4
    # = 12+3.46 = 15.46  >  phys z(k=12,j=2)=sin30*2+cos30*12 = 1+10.39 = 11.39.
    # So the most-cranial slice is k=4, NOT ks.max()=12.
    lciv = np.zeros((16, 30, 30), bool)
    lciv[4, 24:26, 14:16] = True       # cranial-most (high j, low k)
    lciv[12, 2:4, 14:16] = True        # caudal (low j, high k)
    _write(seg / "iliac_vena_left.nii.gz", lciv, spacing_xyz=(1.0, 1.0, 1.0),
           direction=_TILT_DIR)
    # CT: distinguishable body sizes at the two ends. Float HU written directly
    # (the uint8 _write helper would corrupt -1000 HU).
    ct = np.full((16, 30, 30), -1000.0, np.float32)
    ct[4, 10:16, 10:16] = 50.0         # k=4 (cranial): 6x6 = 36 vox body
    ct[12, 4:24, 4:24] = 50.0          # k=12 (caudal): 20x20 = 400 vox body
    cimg = sitk.GetImageFromArray(ct)
    cimg.SetSpacing((1.0, 1.0, 1.0))
    cimg.SetDirection([float(d) for d in _TILT_DIR])
    sitk.WriteImage(cimg, str(tmp_path / "ct.nii.gz"))
    r = case_aux_metrics(tmp_path, "seg")
    # most-cranial slice is k=4 (small body) -> 36 * sx*sy = 36 mm^2, NOT 400.
    assert r["body_csa_mm2"] == pytest.approx(36.0)


# --- slice spacing is reported for the z-spacing covariate ------------
def test_measure_one_reports_slice_spacing(tmp_path):
    ivc = np.zeros((10, 8, 8), bool); ivc[6:9, 3:5, 3:5] = True
    lciv = np.zeros((10, 8, 8), bool); lciv[1:3, 3:5, 3:5] = True
    r = measure_one(_case(tmp_path, ivc, lciv, spacing=(1.0, 1.0, 2.0)), "seg")
    assert r["sz_mm"] == pytest.approx(2.0)        # sitk spacing[2] = z
    assert r["sy_mm"] == pytest.approx(1.0)
    assert r["sx_mm"] == pytest.approx(1.0)


# --- Phase-1 correlates: segmentation quality + patient size ---------------
def test_lciv_quality_single_component():
    from measure_ivc_lciv_gap import lciv_quality
    m = np.zeros((10, 8, 8), bool); m[2:8, 3:5, 3:5] = True   # one solid vessel
    lcc, n = lciv_quality(m)
    assert lcc == pytest.approx(1.0) and n == 1


def test_lciv_quality_two_components_is_fragmented():
    from measure_ivc_lciv_gap import lciv_quality
    m = np.zeros((10, 8, 8), bool)
    m[1:3, 1:3, 1:3] = True            # fragment A
    m[6:9, 5:7, 5:7] = True            # fragment B (disconnected)
    lcc, n = lciv_quality(m)
    assert n == 2 and lcc < 1.0


def test_case_aux_metrics_rejects_ct_mask_grid_mismatch(tmp_path):
    from measure_ivc_lciv_gap import case_aux_metrics
    seg = tmp_path / "seg"; seg.mkdir()
    lciv = np.zeros((10, 8, 8), bool); lciv[2:8, 3:5, 3:5] = True
    _write(seg / "iliac_vena_left.nii.gz", lciv, spacing_xyz=(1.0, 1.0, 2.0))
    ct = sitk.GetImageFromArray(np.zeros((10, 8, 8), np.float32))
    ct.SetSpacing((1.0, 1.0, 3.0))
    sitk.WriteImage(ct, str(tmp_path / "ct.nii.gz"))
    with pytest.raises(ValueError, match="CT grid does not match"):
        case_aux_metrics(tmp_path, "seg")


def test_body_csa_thresholds_body_voxels():
    from measure_ivc_lciv_gap import body_csa
    sl = np.full((20, 20), -1000.0)    # air background
    sl[5:15, 4:16] = 50.0              # 10 x 12 soft-tissue body block (HU 50)
    csa = body_csa(sl, sx=2.0, sy=1.0)         # 120 voxels * (2*1) mm^2
    assert csa == pytest.approx(120 * 2.0 * 1.0)


def test_case_aux_metrics_masks_only_no_ct(tmp_path):
    from measure_ivc_lciv_gap import case_aux_metrics
    seg = tmp_path / "seg"; seg.mkdir()
    lciv = np.zeros((10, 8, 8), bool); lciv[2:8, 3:5, 3:5] = True
    _write(seg / "iliac_vena_left.nii.gz", lciv)
    r = case_aux_metrics(tmp_path, "seg")
    assert r["lciv_lcc_frac"] == pytest.approx(1.0)
    assert r["lciv_n_components"] == 1
    import math as _m
    assert _m.isnan(r["body_csa_mm2"])         # no CT present -> size unavailable


# --- closed exclusion accounting: roster vs produced -----------------------
def test_case_roster_accounts_for_missing_masks(tmp_path):
    import measure_all as MA
    # case A produced a full-res IVC mask
    (tmp_path / "caseA" / "segmentations_fullres").mkdir(parents=True)
    (tmp_path / "caseA" / "segmentations_fullres" / "inferior_vena_cava.nii.gz").write_bytes(b"x")
    # case B has a CT but segmentation produced NO mask (total failure)
    (tmp_path / "caseB").mkdir()
    (tmp_path / "caseB" / "ct.nii.gz").write_bytes(b"x")
    # a hidden / non-case dir must be ignored
    (tmp_path / ".cache").mkdir()
    roster, produced, missing = MA.case_roster(tmp_path)
    assert produced == ["caseA"]
    assert set(roster) == {"caseA", "caseB"}       # both attempted; .cache excluded
    assert missing == ["caseB"]                    # total-failure case is accounted


def test_case_roster_enforces_expected_identity_and_ignores_extra_mask(tmp_path):
    import measure_all as MA
    for case in ("case_expected", "case_extra"):
        seg = tmp_path / case / "segmentations_fullres"
        seg.mkdir(parents=True)
        (seg / "inferior_vena_cava.nii.gz").write_bytes(b"mask")

    roster, produced, missing = MA.case_roster(
        tmp_path, expected=["case_expected", "case_missing"])

    assert roster == ["case_expected", "case_missing"]
    assert produced == ["case_expected"]
    assert missing == ["case_missing"]


def test_case_roster_rejects_nonexistent_data_directory(tmp_path):
    import measure_all as MA
    with pytest.raises(FileNotFoundError, match="data directory does not exist"):
        MA.case_roster(tmp_path / "missing", expected=["case_expected"])


# ==========================================================================================
# The axial void-slice gap: k void axial slices give a surface gap of k*sz, and 0 when the masks
# touch.
# ==========================================================================================


def _measured_void_slice_formula(sz):
    """Drive the real `surface_gap` over k void axial slices and return the observed gaps."""
    observed = []
    for k in range(6):
        ivc = np.zeros((20, 8, 8), bool)
        lciv = np.zeros((20, 8, 8), bool)
        ivc[5, 3:6, 3:6] = True
        lciv[5 + k + 1, 3:6, 3:6] = True
        void = list(range(6, 6 + k))
        assert len(void) == k
        if void:                       # the band must actually be empty, not assumed empty
            assert not ivc[void].any() and not lciv[void].any()
        surf, _c2c = surface_gap(ivc, lciv, (sz, 1.0, 1.0))
        observed.append(surf)
    return observed


@pytest.mark.parametrize("sz", [1.0, 2.5, 3.0, 5.0])
def test_the_axial_void_slice_gap_is_k_times_sz(sz):
    """Measured by driving `surface_gap` over k void slices: the gap is k*sz, and 0 at k = 0."""
    observed = _measured_void_slice_formula(sz)
    assert observed == pytest.approx([k * sz for k in range(6)], abs=1e-9), observed
    assert observed[0] == pytest.approx(0.0, abs=1e-9), (
        "a zero-void-slice gap must read 0")


# The default layout nests the masks under `<case>/segmentations_fullres/`; the relevant-mask export
# writes them directly into `<case>/`. These tests pin BOTH layouts
# and, more importantly, that neither is discovered under the other's setting -- a discovery that
# silently succeeded across layouts would measure the wrong set.

def _flat_case(root, name):
    d = root / name
    d.mkdir(parents=True)
    (d / "inferior_vena_cava.nii.gz").write_bytes(b"mask")
    return d


def _nested_case(root, name):
    d = root / name / "segmentations_fullres"
    d.mkdir(parents=True)
    (d / "inferior_vena_cava.nii.gz").write_bytes(b"mask")
    return d


def test_seg_glob_builds_both_layouts():
    import measure_all as MA
    assert MA._seg_glob("inferior_vena_cava.nii.gz", "segmentations_fullres") == \
        "*/segmentations_fullres/inferior_vena_cava.nii.gz"
    # empty segdir means the mask sits directly in the case dir; the pattern must lose the
    # level, not keep an empty one (`*//name` would match nothing).
    assert MA._seg_glob("inferior_vena_cava.nii.gz", "") == "*/inferior_vena_cava.nii.gz"


def test_case_roster_discovers_the_flat_2_17_layout(tmp_path):
    import measure_all as MA
    for n in ("amos_0001", "amos_0002"):
        _flat_case(tmp_path, n)
    (tmp_path / ".cache").mkdir()
    # A NON-HIDDEN directory that is not a case. This is the discriminating fixture: the
    # hidden-name check above it already excludes `.cache`, so an `_is_case_dir` that returned
    # True unconditionally would still pass on `.cache` alone. In the flat layout a stray sibling
    # (logs, tars, a partial export) would otherwise enter the roster and be emitted as NO_MASK,
    # inflating the denominator with something that was never a case.
    (tmp_path / "archives").mkdir()
    (tmp_path / "archives" / "notes.txt").write_bytes(b"x")
    roster, produced, missing = MA.case_roster(tmp_path, segdir="")
    assert produced == ["amos_0001", "amos_0002"]
    assert set(roster) == {"amos_0001", "amos_0002"}   # .cache AND archives excluded
    assert missing == []


def test_the_default_layout_is_nested(tmp_path):
    import measure_all as MA
    _nested_case(tmp_path, "caseA")
    roster, produced, missing = MA.case_roster(tmp_path)          # no segdir -> default
    assert produced == ["caseA"] and missing == []


def test_neither_layout_is_discovered_under_the_other_setting(tmp_path):
    """The discriminating test."""
    import measure_all as MA
    flat, nested = tmp_path / "flat", tmp_path / "nested"
    _flat_case(flat, "c1")
    _nested_case(nested, "c1")
    # flat tree read with the nested default: nothing produced
    assert MA.case_roster(flat)[1] == []
    # nested tree read as flat: nothing produced
    assert MA.case_roster(nested, segdir="")[1] == []


_PROBE_TABLES = ("probe_results_compare.txt", "probe_results_fullres.txt",
                 "probe_results_aux.txt")
_IVC, _LCIV = "inferior_vena_cava.nii.gz", "iliac_vena_left.nii.gz"


def _write_pass(case_dir, sub, names):
    """A cranial IVC column over a caudal LCIV column with a void between them, written to
    `<case_dir>/<sub>/` (an empty `sub` is the flat 2.17.0 export)."""
    ivc = np.zeros((60, 24, 24), np.uint8)
    lciv = np.zeros((60, 24, 24), np.uint8)
    ivc[35:60, 10:14, 10:14] = 1
    lciv[0:25, 10:14, 10:14] = 1
    seg = case_dir / sub
    seg.mkdir(parents=True, exist_ok=True)
    for name in names:
        _write(seg / name, {_IVC: ivc, _LCIV: lciv}[name], spacing_xyz=(1.0, 1.0, 1.0))


def _write_other_fast_pass(case_dir):
    """A genuine second inference: the same anatomy, segmented one slice shorter at each end, so
    its bytes and its gap differ from the full-res pass's."""
    ivc = np.zeros((60, 24, 24), np.uint8)
    lciv = np.zeros((60, 24, 24), np.uint8)
    ivc[36:59, 10:14, 10:14] = 1
    lciv[1:24, 10:14, 10:14] = 1
    seg = case_dir / "segmentations"
    seg.mkdir(parents=True, exist_ok=True)
    _write(seg / _IVC, ivc, spacing_xyz=(1.0, 1.0, 1.0))
    _write(seg / _LCIV, lciv, spacing_xyz=(1.0, 1.0, 1.0))


def _run_measure_all(monkeypatch, tmp_path, data, roster_ids):
    import sys

    import measure_all as MA
    argv = ["measure_all.py", "--data-dir", str(data), "--prefix", "probe", "--jobs", "1"]
    if roster_ids is not None:
        roster = tmp_path / "roster.txt"
        roster.write_text("".join(f"{c}\n" for c in roster_ids), encoding="utf-8")
        argv += ["--roster", str(roster)]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(MA, "SEGDIR", "segmentations_fullres")   # restored after the test
    monkeypatch.setattr(MA, "DATA", None)
    monkeypatch.setattr(sys, "argv", argv)
    return MA.main()


def _probe_rows(tmp_path, name):
    text = (tmp_path / name).read_text(encoding="utf-8")
    return [ln.split() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]


@pytest.mark.parametrize("layout, reason", [
    ("flat_export_read_as_nested", "NO_MASK x2"),
    ("nothing_discovered", "0 case directories found under"),
    ("every_mask_empty", "empty mask x2"),
    ("every_mask_unreadable", "RuntimeError x2"),
])
def test_measure_all_publishes_nothing_when_nothing_was_measured(tmp_path, monkeypatch, layout,
                                                                 reason):
    """Guards against a `main()` with no postcondition, which would publish three tables of error
    rows (or three empty ones) and exit 0. Four ways to measure nothing, one per error spelling a
    full-res row can carry: the flat 2.17.0 export read with the nested default and a roster
    (`NO_MASK`), an open cohort in which nothing is discovered (zero rows), masks with no
    foreground (`ERROR: empty mask`, what SAROS's 26 error rows say), and masks that cannot be
    read (`ERROR: RuntimeError`, through `work`'s handler). The message must name the reason."""
    data = tmp_path / "data"
    roster_ids = ["case_a", "case_b"]
    if layout == "flat_export_read_as_nested":
        for c in roster_ids:
            _write_pass(data / c, "", (_IVC, _LCIV))
    elif layout == "nothing_discovered":
        (data / "logs").mkdir(parents=True)           # a directory that is not a case
        roster_ids = None
    else:
        for c in roster_ids:
            seg = data / c / "segmentations_fullres"
            seg.mkdir(parents=True)
            for name in (_IVC, _LCIV):
                if layout == "every_mask_empty":
                    _write(seg / name, np.zeros((60, 24, 24), np.uint8), spacing_xyz=(1.0, 1.0, 1.0))
                else:
                    (seg / name).write_bytes(b"not a NIfTI")
    with pytest.raises(SystemExit) as exc:
        _run_measure_all(monkeypatch, tmp_path, data, roster_ids)
    assert exc.value.code not in (None, 0), "a run that measured nothing exited successfully"
    assert "nothing was measured" in str(exc.value)
    assert reason in str(exc.value), str(exc.value)
    left = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("probe_results"))
    assert left == [], f"a run that measured nothing left tables behind: {left}"


def test_measure_all_publishes_a_partial_failure_and_a_refusal_keeps_the_previous_tables(
        tmp_path, monkeypatch):
    """Both directions, because a postcondition that refuses every failure would be wrong: the
    study's cohorts mix measurements with errors (SAROS: 26 of 392 full-res rows are errors,
    30 of 392 compare rows `empty mask`). Guards against an over-strict rule (the first run must
    publish), and against a refusal raised after publication (the second run must leave every
    previous table byte-identical, which only a refusal inside the `AtomicTable` block does)."""
    data = tmp_path / "data"
    for c in ("case_a", "case_b"):                             # discovered, not measurable
        _write_pass(data / c, "segmentations_fullres", (_IVC,))
    _write_pass(data / "case_c", "segmentations_fullres", (_IVC, _LCIV))   # measurable, LAST
    _write_other_fast_pass(data / "case_c")                     # a genuine second inference
    ids = ["case_a", "case_b", "case_c"]
    _run_measure_all(monkeypatch, tmp_path, data, ids)
    fullres = _probe_rows(tmp_path, "probe_results_fullres.txt")
    compare = _probe_rows(tmp_path, "probe_results_compare.txt")
    assert [r[0] for r in fullres] == ids
    assert [r[1] == "ERROR:" for r in fullres] == [True, True, False]
    assert [r[1:] == ["empty", "mask"] for r in compare] == [True, True, False]
    before = {t: (tmp_path / t).read_bytes() for t in _PROBE_TABLES}

    for sub in ("segmentations", "segmentations_fullres"):
        (data / "case_c" / sub / _LCIV).unlink()
    with pytest.raises(SystemExit) as exc:
        _run_measure_all(monkeypatch, tmp_path, data, ids)
    assert exc.value.code not in (None, 0), "a run that measured nothing exited successfully"
    assert "nothing was measured" in str(exc.value)
    # The rows are discarded with the `.part`, so the refusal is the only place the operator can
    # learn WHY: it must name the reasons the rows carried, not only their number.
    assert "missing files" in str(exc.value), str(exc.value)
    for t in _PROBE_TABLES:
        assert (tmp_path / t).read_bytes() == before[t], (
            f"{t}: the refusal replaced the previous complete table")


def test_measure_all_refuses_a_compare_arm_without_a_single_fast_vs_full_pair(
        tmp_path, monkeypatch):
    """Guards against a missing postcondition on the compare table. Positive control: once one case
    has its fast pass, the run publishes."""
    data = tmp_path / "data"
    for c in ("case_a", "case_c"):
        _write_pass(data / c, "segmentations_fullres", (_IVC, _LCIV))   # full pass only
    _write_pass(data / "case_b", "segmentations_fullres", (_IVC,))      # a full-res failure
    ids = ["case_a", "case_b", "case_c", "case_d"]                      # case_d: NO_MASK
    with pytest.raises(SystemExit) as exc:
        _run_measure_all(monkeypatch, tmp_path, data, ids)
    assert exc.value.code not in (None, 0), "a compare table of `empty mask` exited successfully"
    assert "no fast-vs-full-res pair" in str(exc.value)
    assert not any((tmp_path / t).exists() for t in _PROBE_TABLES)

    _write_other_fast_pass(data / "case_c")                             # the LAST measurable case
    _run_measure_all(monkeypatch, tmp_path, data, ids)
    compare = _probe_rows(tmp_path, "probe_results_compare.txt")
    fullres = _probe_rows(tmp_path, "probe_results_fullres.txt")
    assert [r[0] for r in compare] == ids
    assert [r[1:] == ["empty", "mask"] for r in compare] == [True, True, False, True]
    assert [r[1] == "ERROR:" for r in fullres] == [False, True, False, True]


def test_measure_all_does_not_pair_a_pass_with_itself(tmp_path, monkeypatch):
    """Breaks caught: a pair predicate that only rejects `empty mask` (the first run must refuse: every
    case is self-paired, so there is no pair), and a self-pair check by path alone (the second run
    must write `empty mask` for the byte copy while the genuine pair is published)."""
    import shutil
    data = tmp_path / "data"
    for c in ("case_a", "case_b"):
        _write_pass(data / c, "segmentations_fullres", (_IVC, _LCIV))
        (data / c / "segmentations").symlink_to(data / c / "segmentations_fullres",
                                                target_is_directory=True)
    with pytest.raises(SystemExit) as exc:
        _run_measure_all(monkeypatch, tmp_path, data, ["case_a", "case_b"])
    assert exc.value.code not in (None, 0), "a pass paired with itself was published as agreement"
    assert "no fast-vs-full-res pair" in str(exc.value)
    assert not any((tmp_path / t).exists() for t in _PROBE_TABLES)

    for c in ("case_a", "case_b"):
        (data / c / "segmentations").unlink()
    _write_other_fast_pass(data / "case_a")
    shutil.copytree(data / "case_b" / "segmentations_fullres", data / "case_b" / "segmentations")
    _run_measure_all(monkeypatch, tmp_path, data, ["case_a", "case_b"])
    compare = _probe_rows(tmp_path, "probe_results_compare.txt")
    assert [r[0] for r in compare] == ["case_a", "case_b"]
    assert compare[0][1] != compare[0][2], f"the genuine pair should differ: {compare[0]}"
    assert compare[1][1:] == ["empty", "mask"], f"a byte copy was paired: {compare[1]}"


def test_a_corrupt_fast_pass_blanks_only_its_compare_row(tmp_path, monkeypatch):
    """Guards against reading both passes inside one `try`, which would turn a measurable full-res
    case into `ERROR: RuntimeError` and, across a cohort, send the operator to check `--seg-subdir`
    when the full-res pass is fine."""
    data = tmp_path / "data"
    for c in ("case_a", "case_b"):
        _write_pass(data / c, "segmentations_fullres", (_IVC, _LCIV))
    _write_other_fast_pass(data / "case_a")
    _write_other_fast_pass(data / "case_b")
    (data / "case_b" / "segmentations" / _IVC).write_bytes(b"not a NIfTI")
    _run_measure_all(monkeypatch, tmp_path, data, ["case_a", "case_b"])
    fullres = _probe_rows(tmp_path, "probe_results_fullres.txt")
    compare = _probe_rows(tmp_path, "probe_results_compare.txt")
    assert [r[1] == "ERROR:" for r in fullres] == [False, False], (
        f"a corrupt fast-pass mask erased a measurable full-res case: {fullres}")
    assert [r[1:] == ["empty", "mask"] for r in compare] == [False, True]
