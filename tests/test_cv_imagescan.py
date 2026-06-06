"""CV image-quality scan — brightness/blur/entropy/duplicates (needs Pillow)."""
import shutil

import numpy as np
import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from src.services.cv import imagescan  # noqa: E402


def _img(path, arr):
    Image.fromarray(arr.astype(np.uint8)).save(path)


def test_scan_missing_dir():
    assert "error" in imagescan.scan_images("/no/such/dir")


def test_scan_detects_dups_and_dark(tmp_path):
    rng = np.random.RandomState(0)
    for i in range(5):
        _img(tmp_path / f"{i}.png", rng.randint(0, 255, (48, 48, 3)))
    _img(tmp_path / "dark.png", np.full((48, 48, 3), 8))     # very dark
    shutil.copy(tmp_path / "0.png", tmp_path / "dup0.png")    # exact duplicate

    r = imagescan.scan_images(str(tmp_path), sample=100)
    assert r["scanned"] == 7
    assert r["total_images"] == 7
    # the duplicate is found exactly
    assert r["exact_duplicates"]["count"] >= 2
    grp = r["exact_duplicates"]["groups"][0]["files"]
    assert "0.png" in grp and "dup0.png" in grp
    # dark image flagged in the brightness band
    assert r["brightness"]["dark_or_bright"]["count"] >= 1
    # histograms present
    assert r["brightness"]["hist"] and r["blur"]["hist"]


def test_scan_sampling_caps_count(tmp_path):
    rng = np.random.RandomState(1)
    for i in range(20):
        _img(tmp_path / f"{i}.png", rng.randint(0, 255, (32, 32, 3)))
    r = imagescan.scan_images(str(tmp_path), sample=5)
    assert r["total_images"] == 20
    assert r["scanned"] <= 5


def test_sample_annotations(tmp_path):
    img = tmp_path / "images"; lbl = tmp_path / "labels"
    img.mkdir(); lbl.mkdir()
    rng = np.random.RandomState(0)
    for i in range(6):
        _img(img / f"{i}.png", rng.randint(0, 255, (80, 100, 3)))
        (lbl / f"{i}.txt").write_text("0 0.5 0.5 0.4 0.4\n1 0.2 0.2 0.1 0.1\n")
    samples = imagescan.sample_annotations(str(img), str(lbl), n=4)
    assert 1 <= len(samples) <= 4
    s = samples[0]
    assert s["data_uri"].startswith("data:image/jpeg;base64,")
    assert s["width"] > 0 and s["height"] > 0
    assert len(s["boxes"]) == 2
    b = s["boxes"][0]
    assert all(k in b for k in ("cls", "x", "y", "w", "h"))


def test_sample_annotations_missing_dir():
    assert imagescan.sample_annotations("/no/img", "/no/lbl") == []


def test_laplacian_and_entropy_pure():
    flat = np.full((20, 20), 100.0)
    noisy = np.random.RandomState(0).rand(20, 20) * 255
    # a flat image has ~zero Laplacian variance; noise has high
    assert imagescan._laplacian_var(flat) < imagescan._laplacian_var(noisy)
    # flat image has ~zero entropy; noise high
    assert imagescan._entropy(flat) < imagescan._entropy(noisy)
