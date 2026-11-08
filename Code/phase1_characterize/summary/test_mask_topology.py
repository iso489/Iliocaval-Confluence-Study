"""Tests for the Betti numbers and gap levels (mask_topology.py), on synthetic masks."""
import hashlib
import importlib.util
import pathlib

import nibabel as nib
import numpy as np
import pytest

_PATH = pathlib.Path(__file__).with_name("mask_topology.py")
_SPEC = importlib.util.spec_from_file_location("mask_topology", _PATH)
MT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(MT)
# LPS-style storage (x and y flipped), 2-mm slices, origin away from zero: the frame must not matter
AFFINE = np.array([[-1.0, 0, 0, 40.0], [0, -1.0, 0, 30.0], [0, 0, 2.0, -100.0], [0, 0, 0, 1.0]])


def _save(folder, name, array, affine=AFFINE):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.nii.gz"
    nib.save(nib.Nifti1Image(array.astype(np.uint8), affine), path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_betti_numbers_of_the_shapes_the_confluence_can_take():
    y = np.zeros((9, 9, 12), bool)
    y[4, 4, 6:12] = True                                  # trunk
    y[2:5, 4, 5] = y[4:7, 4, 5] = True                    # two limbs from its foot
    y[2, 4, 0:5] = y[6, 4, 0:5] = True
    assert MT.betti(y) == (1, 0, 0)
    gap = y.copy()
    gap[4, 4, 6] = False                                  # the trunk lifted off the limbs
    assert MT.betti(gap) == (2, 0, 0)
    ring = y.copy()
    ring[2:7, 4, 0] = True                                # a bar joining the two limbs' ends: a tunnel
    assert MT.betti(ring) == (1, 1, 0)
    hollow = np.zeros((7, 7, 7), bool)
    hollow[1:6, 1:6, 1:6] = True
    hollow[3, 3, 3] = False                               # an enclosed cavity
    assert MT.betti(hollow) == (1, 0, 1)
    assert MT.betti(np.zeros((3, 3, 3), bool)) == (0, 0, 0)


def _vertebrae(folder, slabs):
    digests = {}
    for name, level in MT.MB.VERT_LEVELS:
        mask = np.zeros((10, 10, 60), bool)
        if name in slabs:
            low, high = slabs[name]
            mask[3:7, 3:7, low:high] = True
        digests[name] = _save(folder, name, mask)
    sacrum = np.zeros((10, 10, 60), bool)
    sacrum[3:7, 3:7, 0:4] = True
    digests["sacrum"] = _save(folder, "sacrum", sacrum)
    return digests


def test_vertebra_anchors_match_measure_bifurcation_and_fall_back_to_the_sacrum(tmp_path):
    slabs = {"vertebrae_L3": (44, 52), "vertebrae_L4": (32, 40), "vertebrae_L5": (20, 28), "vertebrae_S1": (8, 16)}
    digests = _vertebrae(tmp_path / "a", slabs)
    ours = MT.vertebra_anchors(tmp_path / "a", digests)
    theirs = MT.MB.read_vertebra_anchors(tmp_path / "a")          # SimpleITK, physical LPS z
    assert ours == pytest.approx(theirs) and [lvl for lvl, _ in ours] == [3.0, 4.0, 5.0, 6.0]
    assert dict(ours)[4.0] == pytest.approx(-100.0 + 2.0 * (32 + 39) / 2)
    no_s1 = {k: v for k, v in slabs.items() if k != "vertebrae_S1"}
    digests = _vertebrae(tmp_path / "b", no_s1)
    anchors = MT.vertebra_anchors(tmp_path / "b", digests)
    assert anchors[-1] == pytest.approx((5.5, -100.0 + 2.0 * 3)) == MT.MB.read_vertebra_anchors(tmp_path / "b")[-1]
    with pytest.raises(ValueError, match="not the file"):
        MT.vertebra_anchors(tmp_path / "b", {**digests, "vertebrae_L4": "0" * 64})


def _venous(folder, ivc_low, left_high):
    ivc, left, right = (np.zeros((10, 10, 60), bool) for _ in range(3))
    ivc[4:6, 4:6, ivc_low:55] = True
    left[2:4, 4:6, 10:left_high] = True
    right[6:8, 4:6, 10:ivc_low] = True                   # the right iliac reaches up to the IVC's foot
    return {name: _save(folder, name, mask) for name, mask in
            (("inferior_vena_cava", ivc), ("iliac_vena_left", left), ("iliac_vena_right", right))}


def test_a_disconnected_examination_gets_its_betti_numbers_and_the_level_of_its_corridor(tmp_path):
    case = tmp_path / "AMOS" / "x1"
    slabs = {"vertebrae_L3": (44, 52), "vertebrae_L4": (32, 40), "vertebrae_L5": (20, 28), "vertebrae_S1": (8, 16)}
    vertebrae = _vertebrae(case, slabs)
    veins = _venous(case, ivc_low=36, left_high=26)
    row = {"cohort": "AMOS", "case": "c1", "mask_case": "x1", "class26": "disconnected", "ncomp26": 2,
           **{f"{n}_sha256": d for n, d in veins.items()}}
    caudal = -100.0 + 2.0 * 36                            # the dominant IVC's lowest voxel centre
    gap = {"ivc_caudal_z_mm": caudal, "corridor_centroid_z_minus_ivc_caudal_mm": -2.0 * 3.5}
    out = MT.measure_case((tmp_path, row, vertebrae, gap))
    assert (out["b0"], out["b1"], out["b2"]) == (2, 0, 0)
    # the centroid sits at slice 32.5, between the L4 anchor (35.5) and the L5 anchor (23.5): level 4.25, in
    # the L4 bin [3.5, 4.5) and, by measure_bifurcation's labels, at the start of the L4/L5 disc [4.25, 4.75)
    assert out["gap_z_mm"] == pytest.approx(-100.0 + 2.0 * 32.5) and out["gap_level"] == pytest.approx(4.25)
    assert (out["gap_level_bin"], out["gap_level_label"], out["ruler_qc"], out["gap_in_range"]) == ("L4", "L4/L5", "OK", 1)
    with pytest.raises(ValueError, match="ncomp26"):
        MT.measure_case((tmp_path, {**row, "ncomp26": 3}, vertebrae, gap))
    with pytest.raises(ValueError, match="not bound"):
        MT.measure_case((tmp_path, row, vertebrae, {**gap, "ivc_caudal_z_mm": caudal + 2.0}))
    with pytest.raises(ValueError, match="not the file"):
        MT.measure_case((tmp_path, {**row, "iliac_vena_left_sha256": "0" * 64}, vertebrae, gap))
    connected = MT.measure_case((tmp_path, {**row, "class26": "right_iliac_mediated"}, vertebrae, None))
    assert connected["gap_level_bin"] == "" and np.isnan(connected["gap_level"])


def test_released_annotations_get_the_same_numbers_and_a_mismatch_is_refused(tmp_path):
    folder = tmp_path / "s0001" / "segmentations"
    veins = _venous(folder, ivc_low=36, left_high=26)
    assert MT.measure_reference((tmp_path, "s0001", veins, 2)) == {"case": "s0001", "b0": 2, "b1": 0, "b2": 0}
    with pytest.raises(ValueError, match="ncomp26"):
        MT.measure_reference((tmp_path, "s0001", veins, 1))
    _save(folder, "iliac_vena_right", np.zeros((10, 10, 60), bool), affine=np.diag([1.0, 1.0, 3.0, 1.0]))
    other = {**veins, "iliac_vena_right": hashlib.sha256((folder / "iliac_vena_right.nii.gz").read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="not on the grid"):
        MT.measure_reference((tmp_path, "s0001", other, 2))


def test_read_mask_refuses_a_mask_that_is_not_binary(tmp_path):
    mask = np.zeros((4, 4, 4), np.uint8)
    mask[1, 1, 1] = 1
    image, arr = MT.read_mask(tmp_path / "one.nii.gz", _save(tmp_path, "one", mask))
    assert arr.dtype == bool and int(arr.sum()) == 1
    with pytest.raises(ValueError, match="not a binary integer mask"):
        MT.read_mask(tmp_path / "two.nii.gz", _save(tmp_path, "two", mask * 2))
    for name, array in (("float", mask.astype(np.float32)), ("negative", -mask.astype(np.int16))):
        path = tmp_path / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(array, AFFINE), path)
        with pytest.raises(ValueError, match="not a binary integer mask"):
            MT.read_mask(path, hashlib.sha256(path.read_bytes()).hexdigest())
