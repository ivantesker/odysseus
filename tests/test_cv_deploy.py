"""CV edge deploy: config generators (synthetic I/O) + calibration set."""
import numpy as np
import pytest

from src.services.cv import deploy


_FAKE_IO = {
    "inputs": [{"name": "images", "shape": [1, 3, 256, 256], "dtype": "tensor(float)"}],
    "outputs": [{"name": "output0", "shape": [1, 84, 1344], "dtype": "tensor(float)"}],
    "size_mb": 12.3,
}


def test_input_hw_nchw():
    assert deploy._input_hw(_FAKE_IO) == (256, 256, 3)


def test_deepstream_config(monkeypatch):
    monkeypatch.setattr(deploy, "read_onnx_io", lambda p: _FAKE_IO)
    r = deploy.deepstream_config("m.onnx", class_names=["car", "bus", "truck"], network_mode=2)
    cfg = r["deepstream_config"]
    assert "infer-dims=3;256;256" in cfg
    assert "output-blob-names=output0" in cfg
    assert "num-detected-classes=3" in cfg
    assert "network-mode=2" in cfg
    assert r["labels_txt"] == "car\nbus\ntruck"
    assert r["input_hw"] == [256, 256]


def test_triton_config(monkeypatch):
    monkeypatch.setattr(deploy, "read_onnx_io", lambda p: _FAKE_IO)
    r = deploy.triton_config("m.onnx", model_name="det", max_batch_size=4)
    cfg = r["config_pbtxt"]
    assert 'name: "det"' in cfg
    assert "max_batch_size: 4" in cfg
    assert 'name: "images"' in cfg and 'name: "output0"' in cfg
    assert "[ 3, 256, 256 ]" in cfg            # batch dim dropped


def test_read_onnx_io_missing():
    assert "error" in deploy.read_onnx_io("/no/such/model.onnx")


def test_select_calibration_set(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image
    rng = np.random.RandomState(0)
    for i in range(30):
        Image.fromarray(rng.randint(0, 255, (32, 32, 3), dtype=np.uint8)).save(tmp_path / f"{i}.png")
    out = tmp_path / "calib.txt"
    r = deploy.select_calibration_set(str(tmp_path), n=10, out_file=str(out))
    assert r["count"] == 10
    assert r["total_images"] == 30
    assert out.exists() and len(out.read_text().splitlines()) == 10


def test_select_calibration_missing():
    assert "error" in deploy.select_calibration_set("/no/such/dir")
