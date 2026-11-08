"""Scientific boundary checks for the gap-content and spacing-aware separation analysis.

Oracles are independent of the implementation: the published `surface_gap` for the box
separation, hand-built geometries for orientation, corridor content, position and bottleneck,
and `hashlib` for the read-once digests.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import gap_content_analysis as GC
import measure_collapsed_gap as MC
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from measure_ivc_lciv_gap import box_to_box_mm, surface_gap, surface_nearest_pair

INDEX = {"aorta": 52, "inferior_vena_cava": 63, "iliac_artery_left": 65, "iliac_artery_right": 66,
         "iliac_vena_left": 67, "iliac_vena_right": 68, "vertebrae_L5": 32}
IVC, LCIV, RCIV = INDEX["inferior_vena_cava"], INDEX["iliac_vena_left"], INDEX["iliac_vena_right"]
ARTERY_R, ARTERY_L, AORTA = INDEX["iliac_artery_right"], INDEX["iliac_artery_left"], INDEX["aorta"]


def _volume(shape, blocks):
    data = np.zeros(shape, np.uint8)
    for region, label in blocks:
        data[region] = label
    return data


def _row(volume, affine, cohort="AMOS", case="amos_0001", participant=None, mask_case=None):
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    masks = [volume == IVC, volume == LCIV, volume == RCIV]
    topology = MC.classify_topology(masks)
    gap = surface_gap(masks[0], masks[1], tuple(spacing))[0]
    return {"cohort": cohort, "case": case, "mask_case": mask_case or case,
            "participant": participant or f"{cohort}:{case}",
            "class26": topology["class26"], "class6": topology["class6"],
            "gap_mm": round(float(gap), 2), "slice_mm": float(spacing[2]),
            "ivc_voxels": int(masks[0].sum()), "lciv_voxels": int(masks[1].sum()),
            "rciv_voxels": int(masks[2].sum())}


def _disconnected_volume(filler=ARTERY_R, shape=(9, 9, 12)):
    """IVC block on planes 0-1, left iliac block on planes 5-7, planes 2-4 carry `filler`."""
    blocks = [((slice(3, 6), slice(3, 6), slice(0, 2)), IVC),
              ((slice(3, 6), slice(3, 6), slice(5, 8)), LCIV),
              ((slice(0, 2), slice(0, 2), slice(0, 12)), RCIV)]
    if filler:
        blocks.append(((slice(3, 6), slice(3, 6), slice(2, 5)), filler))
    return _volume(shape, blocks)


# --- geometry primitives ------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(5))
def test_box_separation_and_center_distance_match_the_published_metric(seed):
    rng = np.random.default_rng(seed)
    shape = (14, 12, 10)
    a = rng.random(shape) < 0.04
    b = rng.random(shape) < 0.04
    a[7:, :, :] = False
    b[:7, :, :] = False
    a[2, 2, 2] = True
    b[12, 9, 8] = True
    spacing = tuple(rng.uniform(0.5, 4.0, 3))
    result = GC.closest_pairs(a, b, spacing)
    published_box, published_center = surface_gap(a, b, spacing)
    assert result["d_box"] == pytest.approx(published_box, abs=1e-9)
    assert result["d_center"] == pytest.approx(published_center, abs=1e-9)
    assert result["minimizers"].shape[1:] == (2, 3) and result["n_minimizers"] == len(result["minimizers"])
    for pair in result["minimizers"]:
        assert box_to_box_mm(pair[0], pair[1], spacing) == pytest.approx(published_box, abs=1e-9)
        assert a[tuple(pair[0])] and b[tuple(pair[1])]
    achieving = surface_nearest_pair(a, b, spacing)
    assert any(np.array_equal(pair[0], achieving[0]) and np.array_equal(pair[1], achieving[1])
               for pair in result["minimizers"])
    gaps = result["axis_gap_voxels"]
    assert gaps.shape == (result["n_minimizers"], 3) and (gaps >= 0).all()
    assert np.allclose(np.sqrt(((gaps * np.asarray(spacing)) ** 2).sum(axis=1)), published_box)


def test_orientation_classes_follow_the_axis_gaps_of_the_minimizing_pairs():
    spacing = (0.8, 0.8, 2.5)
    shape = (9, 9, 9)
    through = _volume(shape, [((slice(2, 5), slice(2, 5), slice(0, 2)), 1),
                              ((slice(2, 5), slice(2, 5), slice(4, 6)), 2)])
    result = GC.closest_pairs(through == 1, through == 2, spacing)
    summary = GC.orientation_summary(result["axis_gap_voxels"])
    assert summary["orientation"] == "through_plane" and summary["missing_planes"] == 2
    assert result["d_box"] == pytest.approx(2 * 2.5)
    in_plane = _volume(shape, [((slice(0, 2), slice(2, 5), slice(2, 5)), 1),
                               ((slice(4, 6), slice(2, 5), slice(2, 5)), 2)])
    result = GC.closest_pairs(in_plane == 1, in_plane == 2, spacing)
    summary = GC.orientation_summary(result["axis_gap_voxels"])
    assert summary["orientation"] == "in_plane" and summary["missing_planes"] == 0
    assert result["d_box"] == pytest.approx(2 * 0.8)
    oblique = _volume(shape, [((slice(0, 2), slice(2, 5), slice(0, 2)), 1),
                              ((slice(4, 6), slice(2, 5), slice(4, 6)), 2)])
    result = GC.closest_pairs(oblique == 1, oblique == 2, spacing)
    summary = GC.orientation_summary(result["axis_gap_voxels"])
    assert summary["orientation"] == "oblique" and summary["missing_planes"] == 2
    assert result["d_box"] == pytest.approx(np.hypot(2 * 0.8, 2 * 2.5))
    corner = np.zeros(shape, bool)
    corner[0, 0, 0] = True
    partner = np.zeros(shape, bool)
    partner[1, 1, 1] = True
    result = GC.closest_pairs(corner, partner, spacing)
    assert result["d_box"] == 0 and GC.orientation_summary(result["axis_gap_voxels"])["orientation"] == "contact"


def test_disagreeing_minimizers_read_as_mixed_with_their_counts_kept():
    a = np.zeros((11, 11, 11), bool)
    a[5, 5, 5] = True
    b = np.zeros((11, 11, 11), bool)
    b[5, 5, 8] = True     # two missing planes along axis 2
    b[5, 8, 5] = True     # two missing columns in plane
    result = GC.closest_pairs(a, b, (1., 1., 1.))
    assert result["n_minimizers"] == 2 and result["d_box"] == pytest.approx(2.0)
    summary = GC.orientation_summary(result["axis_gap_voxels"])
    assert summary["orientation"] == "mixed"
    assert summary["counts"] == {"through_plane": 1, "in_plane": 1}
    assert summary["missing_planes"] == 0, "the smallest through-plane gap among minimizers"


def test_the_corridor_is_the_lens_between_the_components_and_excludes_them():
    a = np.zeros((7, 7, 9), bool)
    a[3, 3, 1] = True
    b = np.zeros((7, 7, 9), bool)
    b[3, 3, 6] = True
    spacing = (1., 1., 1.)
    corridor, d_center = GC.corridor(a, b, spacing, slack_mm=0.)
    assert d_center == pytest.approx(5.)
    assert set(map(tuple, np.argwhere(corridor))) == {(3, 3, k) for k in range(2, 6)}
    fat, _ = GC.corridor(a, b, spacing)        # default slack: one voxel diagonal
    assert fat[3, 3, 2:6].all() and fat[3, 4, 3] and not fat[0, 0, 3]
    assert not fat[a].any() and not fat[b].any()
    touching = np.zeros((7, 7, 9), bool)
    touching[3, 3, 2] = True
    empty, d = GC.corridor(a, touching, spacing)
    assert d == pytest.approx(1.) and not empty.any()


def test_segment_corridor_follows_the_shortest_connections_and_is_reversal_invariant():
    a = np.zeros((7, 7, 9), bool)
    a[3, 3, 1] = True
    b = np.zeros((7, 7, 9), bool)
    b[3, 3, 6] = True
    pairs = GC.closest_pairs(a, b, (1., 1., 1.))["minimizers"]
    corridor = GC.segment_corridor(pairs, a, b)
    assert set(map(tuple, np.argwhere(corridor))) == {(3, 3, k) for k in range(2, 6)}
    oblique_a = np.zeros((9, 9, 9), bool)
    oblique_a[1, 1, 1] = True
    oblique_b = np.zeros((9, 9, 9), bool)
    oblique_b[4, 4, 7] = True
    forward = GC.segment_corridor(GC.closest_pairs(oblique_a, oblique_b, (1., 1., 1.))["minimizers"], oblique_a, oblique_b)
    backward = GC.segment_corridor(GC.closest_pairs(oblique_b, oblique_a, (1., 1., 1.))["minimizers"], oblique_b, oblique_a)
    assert np.array_equal(forward, backward)
    voxels = np.argwhere(forward)
    assert (voxels.min(axis=0) >= [1, 1, 1]).all() and (voxels.max(axis=0) <= [4, 4, 7]).all()
    _labels, count = GC.ndi.label(forward | oblique_a | oblique_b, np.ones((3, 3, 3), bool))
    assert count == 1, "the crossed voxels form one 26-connected digital segment with the endpoints"
    assert not forward[oblique_a].any() and not forward[oblique_b].any()
    volume = _disconnected_volume(filler=ARTERY_R)
    a3, b3 = GC.dominant_components(volume == IVC, volume == LCIV)
    footprint = GC.segment_corridor(GC.closest_pairs(a3, b3, (0.8, 0.8, 2.5))["minimizers"], a3, b3)
    assert footprint.sum() == 27 and (volume[footprint] == ARTERY_R).all()


def _open_box_crossing(start, stop, shape):
    """Brute force: voxels whose open boxes the segment enters with positive length."""
    start, stop = np.asarray(start, float), np.asarray(stop, float)
    lo_box = np.minimum(start, stop).astype(int)
    hi_box = np.maximum(start, stop).astype(int)
    hits = set()
    for voxel in np.ndindex(*(hi_box - lo_box + 1)):
        c = lo_box + np.asarray(voxel)
        t_enter, t_exit = 0.0, 1.0
        for k in range(3):
            d = stop[k] - start[k]
            if d == 0:
                if not (c[k] - 0.5 < start[k] < c[k] + 0.5):
                    t_enter, t_exit = 1.0, 0.0
                continue
            t1, t2 = (c[k] - 0.5 - start[k]) / d, (c[k] + 0.5 - start[k]) / d
            t_enter, t_exit = max(t_enter, min(t1, t2)), min(t_exit, max(t1, t2))
        if t_exit - t_enter > 1e-9 and all(0 <= c[k] < shape[k] for k in range(3)):
            hits.add(tuple(int(x) for x in c))
    return hits


@pytest.mark.parametrize("seed", range(6))
def test_segment_corridor_is_exactly_the_open_box_crossing_set(seed):
    rng = np.random.default_rng(seed)
    shape = (14, 13, 15)
    a = np.zeros(shape, bool)
    b = np.zeros(shape, bool)
    p = rng.integers(0, 5, 3)
    q = rng.integers(8, 13, 3)
    a[tuple(p)] = True
    b[tuple(q)] = True
    pairs = GC.closest_pairs(a, b, (1., 1., 1.))["minimizers"]
    corridor = GC.segment_corridor(pairs, a, b)
    expected = _open_box_crossing(p, q, shape) - {tuple(p), tuple(q)}
    assert set(map(tuple, np.argwhere(corridor))) == expected
    for d in range(1, 7):
        a = np.zeros((8, 8, 8), bool)
        b = np.zeros((8, 8, 8), bool)
        a[0, 0, 0] = True
        b[d, d, d] = True
        voxels = set(map(tuple, np.argwhere(GC.segment_corridor(GC.closest_pairs(a, b, (1., 1., 1.))["minimizers"], a, b))))
        assert voxels == {(k, k, k) for k in range(1, d)}, "a diagonal segment crosses only the diagonal voxels"
    blocks = _disconnected_volume(filler=0)
    a3, b3 = GC.dominant_components(blocks == IVC, blocks == LCIV)
    pairs = GC.closest_pairs(a3, b3, (0.8, 0.8, 2.5))["minimizers"]
    union = set()
    for pair in pairs:
        union |= _open_box_crossing(pair[0], pair[1], blocks.shape)
    union -= set(map(tuple, np.argwhere(a3 | b3)))
    assert set(map(tuple, np.argwhere(GC.segment_corridor(pairs, a3, b3)))) == union


def test_zero_separation_pairs_have_no_corridor_and_a_finite_bottleneck():
    volume = _volume((9, 9, 9), [((slice(2, 5), slice(2, 5), slice(0, 3)), IVC),
                                 ((slice(5, 8), slice(5, 8), slice(3, 6)), LCIV)])   # corner contact only
    affine = np.diag([1., 1., 1., 1.])
    row = _row(volume, affine)
    assert row["class26"] == "direct" and row["class6"] == "disconnected" and row["gap_mm"] == 0.
    result = GC.measure_case(volume, INDEX, affine, row)
    assert result["corridor_voxels"] == 0 and result["lens_voxels"] == 0
    assert result["dominant_orientation"] == "contact" and result["corridor_dominant_category"] == ""
    assert result["bottleneck_radius_mm"] == pytest.approx(1.), "the call site measures 26-connectivity"
    assert np.isnan(result["corridor_inplane_offset_mm"])


def test_the_lens_is_never_clipped_by_the_crop_and_uses_the_dominant_components():
    shape = (40, 40, 45)
    far = _volume(shape, [((slice(18, 21), slice(18, 21), slice(0, 2)), IVC),
                          ((slice(18, 21), slice(18, 21), slice(40, 42)), LCIV)])
    affine = np.diag([0.8, 0.8, 1.0, 1.])
    result = GC.measure_case(far, INDEX, affine, _row(far, affine))
    a, b = GC.dominant_components(far == IVC, far == LCIV)
    full_lens, _ = GC.corridor(a, b, (0.8, 0.8, 1.0))
    assert result["lens_voxels"] == int(full_lens.sum()) > result["corridor_voxels"]
    fragments = _disconnected_volume(filler=0)
    fragments[3:6, 3:6, 3] = IVC
    affine = np.diag([1., 1., 1., 1.])
    result = GC.measure_case(fragments, INDEX, affine, _row(fragments, affine))
    assert result["lens_ivc_fragment_voxels"] == 9 and result["corridor_ivc_fragment_voxels"] == 9


def test_measure_case_binds_the_through_plane_spacing_to_the_table():
    volume = _disconnected_volume()
    affine = np.diag([0.8, 0.8, 2.5, 1.])
    row = _row(volume, affine)
    with pytest.raises(ValueError, match="slice spacing"):
        GC.measure_case(volume, INDEX, affine, {**row, "slice_mm": 5.0})
    rotated = np.diag([2.5, 0.8, 0.8, 1.])
    with pytest.raises(ValueError, match="slice spacing"):
        GC.measure_case(volume, INDEX, rotated, {**_row(volume, rotated), "slice_mm": 2.5})


def test_measure_case_accepts_the_table_rounding_of_the_slice_spacing_and_nothing_beyond_it():
    """The canonical table stores slice_mm rounded to three decimals (all 1,873 rows carry at most
    three; SAROS/case_137 reads 1.991 for a header spacing of 1.9907 mm), so the binding must
    tolerate half a unit in the third decimal and refuse the next unit."""
    volume = _disconnected_volume()
    affine = np.diag([0.8, 0.8, 1.9907, 1.])
    row = _row(volume, affine)
    assert isinstance(GC.measure_case(volume, INDEX, affine, {**row, "slice_mm": 1.991}), dict)
    assert isinstance(GC.measure_case(volume, INDEX, affine, {**row, "slice_mm": 1.9905}), dict)
    for beyond in (1.9913, 1.992, 1.99):
        with pytest.raises(ValueError, match="slice spacing"):
            GC.measure_case(volume, INDEX, affine, {**row, "slice_mm": beyond})


def test_corridor_content_categories_dominance_and_purity():
    spacing = (1., 1., 1.)
    volume = _disconnected_volume(filler=ARTERY_R)
    a, b = GC.dominant_components(volume == IVC, volume == LCIV)
    corridor, _ = GC.corridor(a, b, spacing, slack_mm=0.)
    content = GC.corridor_content(corridor, volume, INDEX, a, b)
    assert content["n_voxels"] == 27 and content["counts_by_name"] == {"iliac_artery_right": 27}
    assert content["dominant_category"] == "iliac_artery_right" and content["dominant_fraction"] == 1.
    assert content["pure"] and content["arterial_voxels"] == 27
    background = _disconnected_volume(filler=0)
    corridor, _ = GC.corridor(a, b, spacing, slack_mm=0.)
    content = GC.corridor_content(corridor, background, INDEX, a, b)
    assert content["dominant_category"] == "background" and content["counts_by_name"] == {}
    partial = _disconnected_volume(filler=ARTERY_R)
    partial[3:6, 3:6, 4] = 0
    content = GC.corridor_content(corridor, partial, INDEX, a, b)
    assert content["dominant_category"] == "iliac_artery_right"
    assert content["dominant_fraction"] == pytest.approx(18 / 27) and not content["pure"]
    assert content["category_counts"]["background"] == 9
    fragments = _disconnected_volume(filler=0)
    fragments[3:6, 3:6, 2] = RCIV
    fragments[3:6, 3:6, 3] = IVC          # an IVC island inside the gap, separated from the trunk
    fragments[3:6, 3:6, 4] = INDEX["vertebrae_L5"]
    a2, b2 = GC.dominant_components(fragments == IVC, fragments == LCIV)
    assert a2.sum() == 18, "the dominant IVC component is the 2-plane trunk, not the island"
    corridor2, _ = GC.corridor(a2, b2, spacing, slack_mm=0.)
    content = GC.corridor_content(corridor2, fragments, INDEX, a2, b2)
    assert content["category_counts"] == {"ivc_fragment": 9, "right_iliac_vein": 9, "other_label": 9}
    assert json.loads(content["other_labels"]) == {"vertebrae_L5": 9}
    assert not content["pure"] and content["dominant_fraction"] == pytest.approx(1 / 3)


def test_position_descriptors_are_physical_and_invariant_to_a_flipped_axis():
    shape = (9, 9, 9)
    cranial = _volume(shape, [((slice(2, 5), slice(2, 5), slice(6, 9)), IVC),
                              ((slice(2, 5), slice(2, 5), slice(0, 2)), LCIV),
                              ((slice(6, 8), slice(6, 8), slice(4, 9)), RCIV)])
    affine = np.diag([1., 1., 2.5, 1.])
    row = _row(cranial, affine)
    result = GC.measure_case(cranial, INDEX, affine, row)
    assert result["ivc_caudal_z_mm"] == pytest.approx(15.) and result["lciv_cranial_z_mm"] == pytest.approx(2.5)
    assert result["si_overlap_mm"] == pytest.approx(-12.5)
    assert result["corridor_centroid_z_minus_ivc_caudal_mm"] == pytest.approx(8.75 - 15.)
    assert result["corridor_centroid_z_minus_rciv_cranial_mm"] == pytest.approx(8.75 - 20.)
    assert result["corridor_inplane_offset_mm"] == pytest.approx(0.)
    flipped = cranial[:, :, ::-1].copy()
    flipped_affine = np.diag([1., 1., -2.5, 1.])
    flipped_affine[2, 3] = 20.
    flipped_row = _row(flipped, flipped_affine)
    mirrored = GC.measure_case(flipped, INDEX, flipped_affine, flipped_row)
    for key in ["ivc_caudal_z_mm", "lciv_cranial_z_mm", "si_overlap_mm", "whole_d_box_mm",
                "corridor_centroid_z_minus_ivc_caudal_mm", "corridor_centroid_z_minus_rciv_cranial_mm",
                "corridor_volume_mm3", "whole_missing_planes"]:
        assert mirrored[key] == pytest.approx(result[key]), key
    assert mirrored["whole_orientation"] == result["whole_orientation"] == "through_plane"
    assert mirrored["displacement_mm_z"] == pytest.approx(result["displacement_mm_z"])
    assert mirrored["displacement_index_2"] == -result["displacement_index_2"] != 0
    shifted = _volume(shape, [((slice(2, 5), slice(2, 5), slice(6, 9)), IVC),
                              ((slice(5, 8), slice(2, 5), slice(0, 2)), LCIV)])
    assert GC.measure_case(shifted, INDEX, affine, _row(shifted, affine))["corridor_inplane_offset_mm"] > 0


def test_bottleneck_radius_is_the_inscribed_radius_of_the_narrowest_connection():
    spacing = (1., 1., 1.)
    shape = (9, 9, 12)
    a = np.zeros(shape, bool)
    a[2:7, 2:7, 0:3] = True
    b = np.zeros(shape, bool)
    b[2:7, 2:7, 8:11] = True
    thin = a | b
    thin[4, 4, 3:8] = True
    assert GC.bottleneck_radius(a, b, thin, spacing) == pytest.approx(1.)
    wide = a | b
    wide[3:6, 3:6, 3:8] = True
    assert GC.bottleneck_radius(a, b, wide, spacing) == pytest.approx(2.)
    assert GC.bottleneck_radius(a, b, wide, (0.5, 0.5, 0.5)) == pytest.approx(1.)
    assert np.isnan(GC.bottleneck_radius(a, b, a | b, spacing))
    corner_a = np.zeros((4, 4, 4), bool)
    corner_a[0, 0, 0] = True
    corner_b = np.zeros((4, 4, 4), bool)
    corner_b[1, 1, 1] = True
    assert GC.bottleneck_radius(corner_a, corner_b, corner_a | corner_b, spacing) == pytest.approx(1.)
    assert np.isnan(GC.bottleneck_radius(corner_a, corner_b, corner_a | corner_b, spacing, connectivity=6))


# --- per-case measurement ---------------------------------------------------------------

def test_measure_case_binds_to_the_canonical_row_before_measuring():
    volume = _disconnected_volume()
    affine = np.diag([0.8, 0.8, 2.5, 1.])
    row = _row(volume, affine)
    GC.measure_case(volume, INDEX, affine, row)
    for field, value, message in [("ivc_voxels", row["ivc_voxels"] + 1, "voxel count"),
                                  ("class26", "direct", "class"),
                                  ("gap_mm", row["gap_mm"] + 0.5, "published separation")]:
        with pytest.raises(ValueError, match=message):
            GC.measure_case(volume, INDEX, affine, {**row, field: value})
    with pytest.raises(ValueError, match="orthogonal"):
        sheared = affine.copy()
        sheared[0, 1] = 0.3
        GC.measure_case(volume, INDEX, sheared, row)


def test_measure_case_describes_a_disconnected_examination():
    volume = _disconnected_volume(filler=ARTERY_R)
    affine = np.diag([0.8, 0.8, 2.5, 1.])
    row = _row(volume, affine)
    assert row["class26"] == "disconnected" and row["gap_mm"] == pytest.approx(7.5)
    result = GC.measure_case(volume, INDEX, affine, row)
    assert result["label_scope"] == "total"
    assert result["whole_d_box_mm"] == pytest.approx(7.5) and result["dominant_d_box_mm"] == pytest.approx(7.5)
    assert result["whole_orientation"] == result["dominant_orientation"] == "through_plane"
    assert result["whole_missing_planes"] == 3 and result["whole_separation_in_slice_units"] == pytest.approx(3.)
    assert result["displacement_through_plane_fraction"] == pytest.approx(1.)
    assert result["displacement_mm_z"] == pytest.approx(4 * 2.5)
    assert result["corridor_dominant_category"] == "iliac_artery_right" and result["corridor_pure"] == 1
    assert result["corridor_iliac_artery_right_voxels"] == 27 and result["corridor_volume_mm3"] == pytest.approx(27 * 0.8 * 0.8 * 2.5)
    assert result["arterial_in_corridor"] == 1 and result["nearest_arterial_label"] == "iliac_artery_right"
    assert result["nearest_arterial_distance_mm"] == 0.
    assert np.isnan(result["bottleneck_radius_mm"])
    assert result["spacing_mm_2"] == 2.5 and result["cohort"] == "AMOS" and result["gap_mm"] == row["gap_mm"]


def test_measure_case_describes_a_mediated_examination_with_its_bottleneck():
    shape = (9, 9, 12)
    volume = _volume(shape, [((slice(3, 6), slice(3, 6), slice(0, 2)), IVC),
                             ((slice(3, 6), slice(3, 6), slice(5, 8)), LCIV),
                             ((slice(4, 5), slice(4, 5), slice(2, 5)), RCIV)])
    affine = np.diag([1., 1., 1., 1.])
    row = _row(volume, affine)
    assert row["class26"] == "right_iliac_mediated"
    result = GC.measure_case(volume, INDEX, affine, row)
    # A one-voxel bridge leaves most of the straight connections in background; the right
    # iliac label is present in the corridor, not dominant.
    assert result["corridor_right_iliac_vein_voxels"] == 3 and result["corridor_voxels"] == 27
    assert result["corridor_dominant_category"] == "background"
    assert result["lens_right_iliac_vein_voxels"] == 3 and result["lens_voxels"] >= 27
    assert result["bottleneck_radius_mm"] == pytest.approx(1.)
    assert result["bottleneck_radius_min_spacing_units"] == pytest.approx(1.)
    assert result["arterial_in_corridor"] == 0 and result["nearest_arterial_label"] == ""
    assert np.isnan(result["nearest_arterial_distance_mm"])


def test_measure_case_with_venous_only_labels_cannot_see_arteries():
    volume = _disconnected_volume(filler=ARTERY_R)
    venous = np.zeros(volume.shape, np.uint8)
    for code, label in enumerate([IVC, LCIV, RCIV], 1):
        venous[volume == label] = code
    index = {name: code for code, name in enumerate(MC.REFERENCE_VEIN_FILES, 1)}
    affine = np.diag([0.8, 0.8, 2.5, 1.])
    row = _row(volume, affine)
    result = GC.measure_case(venous, index, affine, row, label_scope="venous_only")
    assert result["label_scope"] == "venous_only"
    assert result["corridor_dominant_category"] == "background"
    assert np.isnan(result["arterial_in_corridor"]) and np.isnan(result["nearest_arterial_distance_mm"])
    assert result["whole_d_box_mm"] == pytest.approx(7.5)


# --- sources ----------------------------------------------------------------------------

def _write_native_case(root, cohort, case, volume, affine, report_extra=None):
    folder = root / cohort / case
    folder.mkdir(parents=True)
    nib.save(nib.Nifti1Image(volume, affine), folder / "segmentation.nii.gz")
    report = {"totalsegmentator_version": "2.17.0", "task": "total", "multilabel": True, "fast": False,
              "classes": {str(v): k for k, v in INDEX.items()}, "input": "/site/Data/x.nii.gz"}
    report.update(report_extra or {})
    (folder / "report.json").write_text(json.dumps(report))
    return folder


def test_native_source_reads_each_file_once_and_validates_the_report(tmp_path):
    volume = _disconnected_volume()
    affine = np.diag([0.8, 0.8, 2.5, 1.])
    folder = _write_native_case(tmp_path, "AMOS", "amos_0001", volume, affine)
    loaded = GC.load_native(folder)
    assert np.array_equal(np.asanyarray(loaded["image"].dataobj), volume)
    assert loaded["index"] == INDEX and loaded["label_scope"] == "total"
    for name in ["segmentation.nii.gz", "report.json"]:
        assert loaded["source_hashes"][name] == hashlib.sha256((folder / name).read_bytes()).hexdigest()
    _write_native_case(tmp_path, "AMOS", "fast", volume, affine, {"fast": True})
    with pytest.raises(ValueError, match="fast"):
        GC.load_native(tmp_path / "AMOS" / "fast")


def test_venous_source_is_hash_bound_to_the_canonical_row(tmp_path):
    volume = _disconnected_volume()
    affine = np.diag([0.8, 0.8, 2.5, 1.])
    row = _row(volume, affine)
    folder = tmp_path / row["cohort"] / row["mask_case"]
    folder.mkdir(parents=True)
    for name, label in zip(MC.REFERENCE_VEIN_FILES, [IVC, LCIV, RCIV]):
        path = folder / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image((volume == label).astype(np.uint8), affine), path)
        row[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = GC.load_venous(tmp_path, row)
    assert loaded["label_scope"] == "venous_only"
    assert loaded["index"] == {name: code for code, name in enumerate(MC.REFERENCE_VEIN_FILES, 1)}
    data = np.asanyarray(loaded["image"].dataobj)
    assert (data == 1).sum() == row["ivc_voxels"] and (data == 2).sum() == row["lciv_voxels"]
    with pytest.raises(ValueError, match="hash"):
        GC.load_venous(tmp_path, {**row, "inferior_vena_cava_sha256": "0" * 64})


def test_internal_source_requires_a_passed_case_with_matching_output_hashes(tmp_path):
    volume = _disconnected_volume()
    affine = np.diag([1.5, 1.5, 1.5, 1.])
    folder = tmp_path / "AMOS" / "amos_0001"
    folder.mkdir(parents=True)
    nib.save(nib.Nifti1Image(volume, affine), folder / "internal_multilabel.nii.gz")
    sidecar = {"total_class_map": {str(v): k for k, v in INDEX.items()},
               "cardiac_submodel": {"task_id": 293, "classes": {}}}
    (folder / "internal_labels.json").write_text(json.dumps(sidecar))
    qc = {"status": "pass", "outputs": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                                        for name in ["internal_multilabel.nii.gz", "internal_labels.json"]},
          **_internal_record(volume, affine)}
    (folder / "qc.json").write_text(json.dumps(qc))
    loaded = GC.load_internal(folder)
    assert loaded["index"] == INDEX and loaded["label_scope"] == "total"
    assert loaded["binding"] == {"spacing_mm": [1.5, 1.5, 1.5], "voxels": qc["internal_topology"]["voxels"],
                                 "class26": "disconnected", "class6": "disconnected"}
    stripped = {k: v for k, v in qc.items() if k not in ("internal_geometry", "internal_topology")}
    (folder / "qc.json").write_text(json.dumps(stripped))
    with pytest.raises(ValueError, match="record lacks"):
        GC.load_internal(folder)
    (folder / "qc.json").write_text(json.dumps(qc))
    assert np.array_equal(np.asanyarray(loaded["image"].dataobj), volume)
    (folder / "qc.json").write_text(json.dumps({**qc, "status": "mismatch"}))
    assert np.array_equal(np.asanyarray(GC.load_internal(folder)["image"].dataobj), volume)
    (folder / "internal_labels.json").write_text(json.dumps({**sidecar, "extra": 1}))
    with pytest.raises(ValueError, match="hash"):
        GC.load_internal(folder)
    (folder / "internal_labels.json").write_text(json.dumps(sidecar))
    for status in ["failed", "started", "invalid"]:
        (folder / "qc.json").write_text(json.dumps({**qc, "status": status}))
        with pytest.raises(ValueError, match="complete"):
            GC.load_internal(folder)


def _internal_record(volume, affine):
    """What run_case records about the internal capture: its grid and its own venous topology."""
    masks = [volume == IVC, volume == LCIV, volume == RCIV]
    topology = MC.classify_topology(masks)
    spacing = [float(x) for x in np.linalg.norm(np.asarray(affine, float)[:3, :3], axis=0)]
    return {"internal_geometry": {"spacing_mm": spacing, "shape": list(volume.shape)},
            "internal_topology": {"class26": topology["class26"], "class6": topology["class6"],
                                  "voxels": [int(m.sum()) for m in masks]}}


def _internal_case(tmp_path, cohort="AMOS", case="amos_0001"):
    """An internal capture on the 1.5 mm grid whose canonical row is a DIFFERENT, native-grid fact."""
    volume = _disconnected_volume(filler=ARTERY_R)
    affine = np.diag([1.5, 1.5, 1.5, 1.])
    folder = tmp_path / "grid" / cohort / case
    folder.mkdir(parents=True)
    nib.save(nib.Nifti1Image(volume, affine), folder / "internal_multilabel.nii.gz")
    sidecar = {"total_class_map": {str(v): k for k, v in INDEX.items()}, "cardiac_submodel": {"task_id": 293, "classes": {}}}
    (folder / "internal_labels.json").write_text(json.dumps(sidecar))
    qc = {"status": "mismatch", **_internal_record(volume, affine),
          "outputs": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                      for name in ["internal_multilabel.nii.gz", "internal_labels.json"]}}
    (folder / "qc.json").write_text(json.dumps(qc))
    # The canonical row describes the NATIVE grid: 5 mm slices, three times the voxels, a wider
    # published separation and, here, a connected native class against a disconnected internal one.
    internal = _row(volume, affine, cohort=cohort, case=case)
    native = {**internal, "slice_mm": 5.0, "gap_mm": 12.5, "class26": "direct", "class6": "direct",
              "ivc_voxels": 3 * internal["ivc_voxels"], "lciv_voxels": 3 * internal["lciv_voxels"],
              "rciv_voxels": 3 * internal["rciv_voxels"]}
    return volume, affine, folder, native, qc


def test_internal_source_binds_to_its_own_model_grid_record_not_to_the_native_table(tmp_path):
    volume, affine, folder, native, qc = _internal_case(tmp_path)
    loaded = GC.load_internal(folder)
    with pytest.raises(ValueError, match="slice spacing disagrees with the canonical table"):
        GC.measure_case(volume, INDEX, affine, native)                       # the native binding refuses the internal grid
    result = GC.measure_case(volume, INDEX, affine, native, loaded["label_scope"], binding=loaded["binding"])
    assert result["grid"] == "internal"
    assert result["class26"] == "disconnected" and result["class6"] == "disconnected", "the internal classes select the subsets"
    assert result["native_class26"] == "direct" and result["native_class6"] == "direct"
    assert result["slice_mm"] == 5.0 and result["gap_mm"] == 12.5, "native identity values carried for the strata"
    assert result["spacing_mm_2"] == pytest.approx(1.5) and result["ivc_voxels"] == qc["internal_topology"]["voxels"][0]
    assert result["whole_d_box_mm"] == pytest.approx(4.5), "three empty 1.5 mm planes on the internal grid"
    assert np.isnan(result["bottleneck_radius_mm"]), "the radius follows the internal class, not the native one"


def test_internal_binding_refuses_a_record_that_disagrees_with_the_volume(tmp_path):
    volume, affine, folder, native, _ = _internal_case(tmp_path)
    binding = GC.load_internal(folder)["binding"]
    for field, value, message in [("spacing_mm", [1.5, 1.5, 1.0], "spacing"),
                                  ("voxels", [binding["voxels"][0] + 1, *binding["voxels"][1:]], "voxel count"),
                                  ("class26", "direct", "class26"), ("class6", "direct", "class6")]:
        with pytest.raises(ValueError, match=f"model-grid record.*{message}|{message}.*model-grid record"):
            GC.measure_case(volume, INDEX, affine, native, "total", binding={**binding, field: value})


def test_internal_worker_measures_a_real_shaped_case_end_to_end(tmp_path):
    _, _, folder, native, qc = _internal_case(tmp_path)
    payload = GC._worker((native, "internal", str(tmp_path / "grid"), "fingerprint"))
    row = payload["row"]
    assert row["grid"] == "internal" and row["class26"] == "disconnected" and row["native_class26"] == "direct"
    assert json.loads(row["source_hashes"])["internal_multilabel.nii.gz"] == qc["outputs"]["internal_multilabel.nii.gz"]


def test_internal_connected_case_gets_a_bottleneck_radius_even_when_the_native_class_is_disconnected(tmp_path):
    # Internally the left iliac vein reaches the IVC (one connected block); natively the table calls the
    # case disconnected, which is the direction the paired audit found for native spacing above 1.5 mm.
    volume = _volume((9, 9, 12), [((slice(3, 6), slice(3, 6), slice(0, 2)), IVC),
                                  ((slice(3, 6), slice(3, 6), slice(2, 8)), LCIV),
                                  ((slice(0, 2), slice(0, 2), slice(0, 12)), RCIV)])
    affine = np.diag([1.5, 1.5, 1.5, 1.])
    internal = _row(volume, affine)
    assert internal["class26"] != "disconnected"
    native = {**internal, "slice_mm": 5.0, "gap_mm": 10.0, "class26": "disconnected", "class6": "disconnected",
              "ivc_voxels": 3 * internal["ivc_voxels"], "lciv_voxels": 3 * internal["lciv_voxels"],
              "rciv_voxels": 3 * internal["rciv_voxels"]}
    record = _internal_record(volume, affine)
    binding = {"spacing_mm": record["internal_geometry"]["spacing_mm"], **record["internal_topology"]}
    result = GC.measure_case(volume, INDEX, affine, native, "total", binding=binding)
    assert result["class26"] == internal["class26"] and result["native_class26"] == "disconnected"
    assert np.isfinite(result["bottleneck_radius_mm"]) and result["bottleneck_radius_mm"] > 0


def test_native_rows_carry_no_internal_grid_columns():
    volume = _disconnected_volume(filler=ARTERY_R)
    affine = np.diag([0.8, 0.8, 2.5, 1.])
    row = _row(volume, affine)
    result = GC.measure_case(volume, INDEX, affine, row)
    assert not {"grid", "native_class26", "native_class6"} & set(result)
    assert result["class26"] == row["class26"] and result["slice_mm"] == row["slice_mm"]


def test_the_internal_report_says_which_grid_each_column_describes():
    cases = pd.DataFrame({"cohort": ["AMOS"], "case": ["a"], "participant": ["p"], "class26": ["disconnected"],
                          "class6": ["disconnected"], "slice_mm": [5.0], "label_scope": ["total"], "grid": ["internal"],
                          "whole_d_box_mm": [4.5], "whole_orientation": ["through_plane"], "whole_missing_planes": [3],
                          "whole_separation_in_slice_units": [3.], "corridor_dominant_category": ["background"],
                          "corridor_dominant_fraction": [1.], "corridor_pure": [1], "arterial_in_corridor": [0],
                          "corridor_right_iliac_vein_voxels": [0], "corridor_ivc_fragment_voxels": [0],
                          "corridor_lciv_fragment_voxels": [0], "lens_arterial_voxels": [0], "bottleneck_radius_mm": [np.nan],
                          "spacing_mm_0": [1.5], "spacing_mm_1": [1.5], "spacing_mm_2": [1.5]})
    summary = GC.summaries(cases, n_boot=50, seed=1)
    report = GC.render_report(summary, cases, {"source": "internal", "cases": 1})
    assert "internal" in report and "native slice spacing" in report and "model-grid record" in report
    native_report = GC.render_report(summary, cases.assign(grid="native"), {"source": "native", "cases": 1})
    assert "model-grid record" not in native_report


# --- the run ----------------------------------------------------------------------------

def _native_frame(tmp_path):
    affine = np.diag([0.8, 0.8, 2.5, 1.])
    rows = []
    specs = [("AMOS", "amos_0001", _disconnected_volume(filler=ARTERY_R)),
             ("AMOS", "amos_0002", _disconnected_volume(filler=0)),
             ("autoPET", "PETCT_a", _volume((9, 9, 12), [((slice(3, 6), slice(3, 6), slice(0, 2)), IVC),
                                                          ((slice(3, 6), slice(3, 6), slice(5, 8)), LCIV),
                                                          ((slice(4, 5), slice(4, 5), slice(2, 5)), RCIV)]))]
    for cohort, case, volume in specs:
        _write_native_case(tmp_path / "multilabel", cohort, case, volume, affine)
        rows.append(_row(volume, affine, cohort=cohort, case=case))
    table = tmp_path / "analysis_cases.csv"
    pd.DataFrame(rows).to_csv(table, index=False)
    return table


def test_run_measures_every_case_resumes_and_validates(tmp_path, monkeypatch):
    table = _native_frame(tmp_path)
    out = tmp_path / "out"
    GC.run(table, "native", tmp_path / "multilabel", out, workers=1)
    cases = pd.read_csv(out / "gap_content_cases.csv")
    assert len(cases) == 3 and set(cases.case) == {"amos_0001", "amos_0002", "PETCT_a"}
    assert cases.set_index("case").loc["amos_0001", "corridor_dominant_category"] == "iliac_artery_right"
    assert cases.set_index("case").loc["PETCT_a", "corridor_right_iliac_vein_voxels"] == 3
    assert np.isfinite(cases.set_index("case").loc["PETCT_a", "bottleneck_radius_mm"])
    summary = pd.read_csv(out / "gap_content_summary.csv")
    orientation = summary[(summary.stratum == "all") & (summary.subset == "disconnected")
                          & (summary.outcome.str.startswith("whole_orientation="))]
    assert orientation.n.eq(2).all() and orientation.proportion.sum() == pytest.approx(1.)
    manifest = json.loads((out / "gap_content_manifest.json").read_text())
    assert manifest["source"] == "native" and manifest["cases"] == 3
    for item in manifest["outputs"]:
        assert hashlib.sha256((out / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    report = (out / "gap_content_report.md").read_text()
    assert "iliac_artery_right" in report and "not causes" in report
    assert list(cases.columns[:9]) == list(GC.IDENTITY_FIRST)
    # Validation must pass on the first run's outputs, before any resume rewrites them.
    GC.run(table, "native", tmp_path / "multilabel", out, workers=1, validate_only=True)
    # A second run finds every case checkpointed and measures nothing.
    monkeypatch.setattr(GC, "measure_case", lambda *a, **k: (_ for _ in ()).throw(AssertionError("remeasured")))
    GC.run(table, "native", tmp_path / "multilabel", out, workers=1)
    GC.run(table, "native", tmp_path / "multilabel", out, workers=1, validate_only=True)
    with (out / "gap_content_cases.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash"):
        GC.run(table, "native", tmp_path / "multilabel", out, workers=1, validate_only=True)


def test_checkpoint_files_keep_dotted_case_identifiers(tmp_path):
    root = tmp_path / "cases"
    first = GC._case_file(root, {"cohort": "autoPET", "case": "PETCT_1.a"})
    second = GC._case_file(root, {"cohort": "autoPET", "case": "PETCT_1.b"})
    assert first.name == "PETCT_1.a.json" and second.name == "PETCT_1.b.json" and first != second


def test_run_keeps_two_cases_that_differ_only_after_a_dot(tmp_path):
    table = _native_frame(tmp_path)
    frame = pd.read_csv(table)
    renames = {"amos_0001": "amos_1.a", "amos_0002": "amos_1.b"}
    for old_name, new_name in renames.items():
        (tmp_path / "multilabel" / "AMOS" / old_name).rename(tmp_path / "multilabel" / "AMOS" / new_name)
    frame["case"] = frame.case.replace(renames)
    frame["mask_case"] = frame.mask_case.replace(renames)
    frame.to_csv(table, index=False)
    out = tmp_path / "out"
    GC.run(table, "native", tmp_path / "multilabel", out, workers=1)
    GC.run(table, "native", tmp_path / "multilabel", out, workers=1, validate_only=True)
    manifest = json.loads((out / "gap_content_manifest.json").read_text())
    assert {"Code/common/statlib.py", "Code/common/gap_config.py"} <= set(manifest["code"])


def test_median_intervals_follow_the_requested_bootstrap_parameters():
    cases = pd.DataFrame({
        "cohort": ["autoPET"] * 8, "case": list("abcdefgh"),
        "participant": ["p1", "p1", "p2", "p3", "p4", "p5", "p6", "p7"],
        "class26": ["disconnected"] * 8, "class6": ["disconnected"] * 8, "slice_mm": [2.5] * 8,
        "whole_d_box_mm": [2.5, 5., 7.5, 10., 2.5, 5., 12.5, 15.],
        "whole_orientation": ["through_plane"] * 8, "whole_missing_planes": [1, 2, 3, 4, 1, 2, 5, 6],
        "whole_separation_in_slice_units": [1., 2., 3., 4., 1., 2., 5., 6.],
        "corridor_dominant_category": ["background"] * 8, "corridor_dominant_fraction": [1.] * 8,
        "corridor_pure": [1] * 8, "arterial_in_corridor": [0] * 8,
        "corridor_right_iliac_vein_voxels": [0] * 8, "corridor_ivc_fragment_voxels": [0] * 8,
        "corridor_lciv_fragment_voxels": [0] * 8, "lens_arterial_voxels": [0] * 8,
        "bottleneck_radius_mm": [np.nan] * 8, "spacing_mm_0": [0.8] * 8, "spacing_mm_1": [0.8] * 8,
        "spacing_mm_2": [2.5] * 8})
    one = GC.summaries(cases, n_boot=150, seed=1)
    two = GC.summaries(cases, n_boot=150, seed=2)
    pick = lambda frame: frame[(frame.stratum == "all") & (frame.outcome == "median:whole_separation_in_slice_units")].iloc[0]
    assert pick(one).estimate == pick(two).estimate
    assert (pick(one).ci_lo, pick(one).ci_hi) != (pick(two).ci_lo, pick(two).ci_hi), "the seed must reach the median bootstrap"


def test_run_refuses_a_case_whose_source_disagrees_with_the_table(tmp_path):
    table = _native_frame(tmp_path)
    frame = pd.read_csv(table)
    frame.loc[frame.case == "amos_0002", "ivc_voxels"] += 1
    frame.to_csv(table, index=False)
    with pytest.raises(ValueError, match="voxel count"):
        GC.run(table, "native", tmp_path / "multilabel", tmp_path / "out", workers=1)
    assert not (tmp_path / "out" / "gap_content_cases.csv").exists()


def test_summaries_use_participant_blocks_when_participants_repeat():
    cases = pd.DataFrame({
        "cohort": ["autoPET"] * 6, "case": list("abcdef"), "participant": ["p1", "p1", "p2", "p3", "p4", "p5"],
        "class26": ["disconnected"] * 4 + ["direct", "right_iliac_mediated"],
        "class6": ["disconnected"] * 5 + ["right_iliac_mediated"],
        "slice_mm": [2.5] * 6, "whole_d_box_mm": [2.5, 5., 0., 7.5, 0., 2.5],
        "whole_orientation": ["through_plane", "oblique", "contact", "through_plane", "contact", "through_plane"],
        "whole_missing_planes": [1, 1, 0, 3, 0, 1],
        "whole_separation_in_slice_units": [1., 2., 0., 3., 0., 1.],
        "corridor_dominant_category": ["background", "iliac_artery_right", "", "right_iliac_vein", "", "right_iliac_vein"],
        "corridor_dominant_fraction": [1., 0.6, np.nan, 1., np.nan, 1.],
        "corridor_pure": [1, 0, 0, 1, 0, 1], "arterial_in_corridor": [0, 1, 0, 0, 0, 0],
        "corridor_right_iliac_vein_voxels": [0, 0, 0, 9, 0, 3], "corridor_ivc_fragment_voxels": [0] * 6,
        "corridor_lciv_fragment_voxels": [0, 2, 0, 0, 0, 0], "lens_arterial_voxels": [0, 5, 0, 0, 0, 0],
        "bottleneck_radius_mm": [np.nan] * 4 + [0.8, 2.5], "spacing_mm_0": [0.8] * 6,
        "spacing_mm_1": [0.8] * 6, "spacing_mm_2": [2.5] * 6})
    summary = GC.summaries(cases, n_boot=200, seed=3)
    row = summary[(summary.stratum == "all") & (summary.subset == "disconnected")
                  & (summary.outcome == "whole_orientation=through_plane")].iloc[0]
    assert row.n == 4 and row.events == 2 and row.ci_method == "cohort_stratified_participant_percentile"
    purity = summary[(summary.stratum == "all") & (summary.subset == "disconnected")
                     & (summary.outcome == "corridor_dominant_category=iliac_artery_right;dominance>=0.5")].iloc[0]
    assert purity.events == 1
    strict = summary[(summary.stratum == "all") & (summary.subset == "disconnected")
                     & (summary.outcome == "corridor_dominant_category=iliac_artery_right;dominance>=1.0")].iloc[0]
    assert strict.events == 0
    thin = summary[(summary.stratum == "all") & (summary.subset == "connected")
                   & (summary.outcome == "bottleneck_radius<=min_spacing")].iloc[0]
    assert thin.n == 2 and thin.events == 1
    median = summary[(summary.stratum == "all") & (summary.subset == "positive_separation")
                     & (summary.outcome == "median:whole_separation_in_slice_units")].iloc[0]
    assert median.n == 4 and median.estimate == pytest.approx(1.5)


def test_read_hashed_returns_the_digest_of_the_bytes_it_parses(tmp_path):
    affine = np.diag([1., 1., 2., 1.])
    volume = _disconnected_volume()
    path = tmp_path / "seg.nii.gz"
    nib.save(nib.Nifti1Image(volume, affine), path)
    image, digest = GC.read_hashed_nifti(path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert np.array_equal(np.asanyarray(image.dataobj), volume) and np.allclose(image.affine, affine)
    plain = tmp_path / "seg.nii"
    nib.save(nib.Nifti1Image(volume, affine), plain)
    image, digest = GC.read_hashed_nifti(plain)
    assert digest == hashlib.sha256(plain.read_bytes()).hexdigest()
    assert np.array_equal(np.asanyarray(image.dataobj), volume)


def test_gap_content_command_starts_without_the_pytest_path_bootstrap(tmp_path):
    result = subprocess.run([sys.executable, str(Path(GC.__file__)), "--help"], cwd=tmp_path,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout and "--source" in result.stdout


