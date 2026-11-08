import hashlib
import importlib.util
import json
import pathlib

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

_PATH = pathlib.Path(__file__).with_name("confluence_comparators.py")
_SPEC = importlib.util.spec_from_file_location("confluence_comparators", _PATH)
CC = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(CC)

NAMES = ("inferior_vena_cava", "iliac_vena_left", "iliac_vena_right", "aorta", "iliac_artery_left", "iliac_artery_right")
INDEX = {name: label for label, name in enumerate(NAMES, start=1)}


def _volume():
    """IVC above the left iliac vein by a shared face; the right iliac vein meets the IVC at one corner;
    the aorta meets the right iliac artery by a face and is one plane short of the left iliac artery."""
    v = np.zeros((8, 8, 10), np.int16)
    v[1:3, 1:3, 5:9] = 1          # IVC
    v[1:3, 1:3, 1:5] = 2          # left iliac vein, face contact at z 4/5
    v[3, 3, 4] = 3                # right iliac vein, corner contact with (2, 2, 5)
    v[5:7, 0:2, 5:9] = 4          # aorta
    v[5:7, 0:2, 1:4] = 5          # left iliac artery, a one-plane gap at z 4
    v[5:7, 2:4, 5:9] = 6          # right iliac artery, face contact
    return v


def test_translate_maps_the_recorded_root_and_refuses_anything_else(tmp_path):
    assert CC.translate("/old/root/Data/a.nii.gz", "/old/root/", tmp_path) == tmp_path / "Data" / "a.nii.gz"
    with pytest.raises(ValueError, match="not under the recorded study root"):
        CC.translate("/elsewhere/a.nii.gz", "/old/root", tmp_path)


def test_reach_separates_face_corner_and_gap_contacts_and_absent_labels():
    v = _volume()
    assert CC.reach(v, INDEX, "inferior_vena_cava", "iliac_vena_left") == {"26": "connected", "6": "connected"}
    assert CC.reach(v, INDEX, "inferior_vena_cava", "iliac_vena_right") == {"26": "connected", "6": "disconnected"}
    assert CC.reach(v, INDEX, "aorta", "iliac_artery_left") == {"26": "disconnected", "6": "disconnected"}
    assert CC.reach(v, INDEX, "aorta", "iliac_artery_right") == {"26": "connected", "6": "connected"}
    empty = v.copy()
    empty[empty == 3] = 0
    assert CC.reach(empty, INDEX, "inferior_vena_cava", "iliac_vena_right") == {"26": "nonmeasurable", "6": "nonmeasurable"}


def test_venous_classes_and_attenuation_follow_the_labels():
    v = _volume()
    assert CC.venous_classes(v, INDEX) == {"26": "direct", "6": "direct"}
    no_ivc = np.where(v == 1, 0, v)
    assert CC.venous_classes(no_ivc, INDEX) == {"26": "nonmeasurable", "6": "nonmeasurable"}
    ct = np.where(v == 1, 180, np.where(v == 2, 120, np.where(v == 3, 60, -50))).astype(np.int16)
    hu = CC.attenuation(v, INDEX, ct)
    assert (hu["median_hu_ivc"], hu["median_hu_lciv"], hu["median_hu_rciv"]) == (180.0, 120.0, 60.0)
    # the union holds one voxel at 60, sixteen at 120 and sixteen at 180, so its 17th value is 120
    assert hu["median_hu_venous"] == 120.0 and hu["voxels_ivc"] == 16 and hu["voxels_rciv"] == 1
    assert np.isnan(CC.attenuation(no_ivc, INDEX, ct)["median_hu_ivc"])


def test_summarize_reports_aggregates_only():
    hu = pd.DataFrame({"cohort": ["A"] * 4 + ["B"] * 4,
                       "native_class26": ["disconnected", "direct"] * 3 + ["nonmeasurable", "direct"],
                       "median_hu_venous": [50, 150, 60, 160, 70, 170, 999, 180.0],
                       "median_hu_ivc": [55, 155, 65, 165, 75, 175, 999, 185.0],
                       "median_hu_lciv": [45, 145, 55, 155, 65, 165, 999, 175.0],
                       "median_hu_rciv": [np.nan] * 8})
    out = CC.summarize(hu)
    assert set(out.kind) == {"quartiles", "count_by_quartile", "quartile_cut"}
    assert out[out.kind == "count_by_quartile"].n.sum() == 7          # the nonmeasurable row is left out
    pooled = out[(out.kind == "quartiles") & (out.cohort == "pooled") & (out.native_class26 == "all")
                 & (out.measure == "median_hu_venous")].iloc[0]
    assert (pooled.n, pooled["median"]) == (7, 150.0)
    assert not (out.measure == "median_hu_rciv").any()                  # an all-missing measure yields no row


def _case(tmp_path, volume, expected=None, internal=None):
    """One synthetic examination on disk, with the loaders patched to return the synthetic volume."""
    study = tmp_path / "study"
    folder = study / "Segmentations" / "A" / "case1"
    folder.mkdir(parents=True)
    (folder / "segmentation.nii.gz").write_bytes(b"labels")
    (folder / "report.json").write_bytes(b"{}")
    ct = np.where(volume == 1, 180, np.where(volume == 2, 120, -50)).astype(np.int16)
    nib.save(nib.Nifti1Image(ct, np.eye(4)), str(study / "ct.nii.gz"))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    classes = CC.venous_classes(volume, INDEX)
    row = {"cohort": "A", "case": "case1", "mask_case": "case1", "participant": "A:case1",
           "frozen_native": "/rec/Segmentations/A/case1/segmentation.nii.gz",
           "frozen_report": "/rec/Segmentations/A/case1/report.json", "ct_path": "/rec/ct.nii.gz",
           "expected_class26": (expected or classes)["26"], "expected_class6": (expected or classes)["6"],
           "source_hashes": {"frozen_native": digest(folder / "segmentation.nii.gz"),
                             "frozen_report": digest(folder / "report.json"), "ct_path": digest(study / "ct.nii.gz")}}
    root = tmp_path / "model_grid"
    (root / "A" / "case1").mkdir(parents=True)
    (root / "cases.json").write_text(json.dumps([row]))
    return study, root, row, (internal or classes)


@pytest.fixture
def patched(monkeypatch):
    def install(volume, internal_classes):
        image = nib.Nifti1Image(volume, np.eye(4))
        monkeypatch.setattr(CC.GC, "load_native", lambda folder: {"image": image, "index": INDEX})
        monkeypatch.setattr(CC.GC, "load_internal", lambda folder: {
            "image": image, "index": INDEX,
            "binding": {"class26": internal_classes["26"], "class6": internal_classes["6"]}})
    return install


def test_main_writes_reachability_attenuation_and_a_manifest(tmp_path, patched):
    v = _volume()
    study, root, row, internal = _case(tmp_path, v)
    patched(v, internal)
    args = ["--model-grid-root", str(root), "--recorded-root", "/rec", "--study-root", str(study)]
    assert CC.main(args + ["--out", str(tmp_path / "unused"), "--validate-only"]) == 0
    assert not (tmp_path / "unused").exists()
    assert CC.main(args + ["--out", str(tmp_path / "out")]) == 0
    reach = pd.read_csv(tmp_path / "out" / "comparators_cases.csv")
    assert reach.loc[0, "native_ivc_rciv_26"] == "connected" and reach.loc[0, "inference_ivc_rciv_6"] == "disconnected"
    assert reach.loc[0, "native_aorta_iliac_artery_left_26"] == "disconnected"
    assert not any(column.startswith("median_hu") for column in reach.columns)   # no image values in this file
    hu = pd.read_csv(tmp_path / "out" / "attenuation_cases.csv")
    assert (hu.loc[0, "median_hu_ivc"], hu.loc[0, "native_class26"]) == (180.0, "direct")
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["failures"] == {} and set(manifest["outputs"]) == {
        "comparators_cases.csv", "attenuation_cases.csv", "attenuation_summary.csv"}


def test_a_changed_input_or_a_class_disagreement_is_a_recorded_failure(tmp_path, patched, capsys):
    v = _volume()
    study, root, row, internal = _case(tmp_path, v, internal={"26": "disconnected", "6": "disconnected"})
    patched(v, internal)
    args = ["--model-grid-root", str(root), "--recorded-root", "/rec", "--study-root", str(study), "--out",
            str(tmp_path / "out"), "--validate-only"]
    assert CC.main(args) == 1
    assert "inference grid: recomputed classes" in capsys.readouterr().out
    (study / "ct.nii.gz").write_bytes(b"changed")
    assert CC.main(args) == 1
    assert "ct_path does not match its recorded SHA-256" in capsys.readouterr().out


def test_a_ct_on_another_grid_is_refused(tmp_path, patched, capsys):
    v = _volume()
    study, root, row, internal = _case(tmp_path, v)
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.int16), np.eye(4)), str(study / "ct.nii.gz"))
    cases = json.loads((root / "cases.json").read_text())
    cases[0]["source_hashes"]["ct_path"] = hashlib.sha256((study / "ct.nii.gz").read_bytes()).hexdigest()
    (root / "cases.json").write_text(json.dumps(cases))
    patched(v, internal)
    assert CC.main(["--model-grid-root", str(root), "--recorded-root", "/rec", "--study-root", str(study),
                    "--out", str(tmp_path / "out"), "--validate-only", "--workers", "1"]) == 1
    assert "do not share a grid" in capsys.readouterr().out
