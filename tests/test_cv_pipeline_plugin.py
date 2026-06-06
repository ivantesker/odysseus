"""CV pipeline plugin tools — dataset_tools real; eval/convert degrade cleanly."""
import importlib


cv = importlib.import_module("plugins.cv_pipeline")


def test_dataset_tools_lint(tmp_path):
    lbl = tmp_path / "labels"
    lbl.mkdir()
    (lbl / "a.txt").write_text("0 .5 .5 .2 .2\n")
    (lbl / "b.txt").write_text("9 .5 .5 .2 .2\n")  # class >= num_classes
    res = cv.dataset_tools({"action": "lint", "labels_dir": str(lbl), "num_classes": 3})
    assert res["action"] == "lint"
    assert res["issue_count"] == 1
    assert res["ok"] is False


def test_dataset_tools_stats(tmp_path):
    img = tmp_path / "images"; lbl = tmp_path / "labels"
    img.mkdir(); lbl.mkdir()
    (img / "a.jpg").write_bytes(b"x")
    (lbl / "a.txt").write_text("0 .5 .5 .2 .2\n")
    res = cv.dataset_tools({"action": "stats", "images_dir": str(img), "labels_dir": str(lbl)})
    assert res["images"] == 1 and res["boxes"] == 1


def test_dataset_tools_split(tmp_path):
    img = tmp_path / "images"; lbl = tmp_path / "labels"; out = tmp_path / "out"
    img.mkdir(); lbl.mkdir()
    for i in range(5):
        (img / f"i{i}.jpg").write_bytes(b"x")
        (lbl / f"i{i}.txt").write_text("0 .5 .5 .2 .2\n")
    res = cv.dataset_tools({"action": "split", "images_dir": str(img),
                            "labels_dir": str(lbl), "out_dir": str(out), "val_frac": 0.4})
    assert res["train"] + res["val"] == 5
    assert (out / "val" / "images").exists()


def test_dataset_tools_bad_action():
    assert cv.dataset_tools({"action": "frobnicate"})["exit_code"] == 1
    assert cv.dataset_tools({"action": "lint"})["exit_code"] == 1  # missing labels_dir


def test_eval_detector_missing_args():
    assert cv.eval_detector({})["exit_code"] == 1


def test_eval_detector_without_ultralytics_is_graceful():
    # ultralytics is not a dependency of the ollama-only app; the tool must
    # return a clear install message, not raise.
    res = cv.eval_detector({"model": "x.pt", "data": "d.yaml"})
    assert res["exit_code"] == 1
    assert "ultralytics" in res["error"].lower() or "eval failed" in res["error"].lower()


def test_convert_model_missing_source():
    res = cv.convert_model({"source": "/no/such/model.pt"})
    assert res["exit_code"] == 1
    assert "not found" in res["error"].lower()


def test_parity_input_never_zeros(tmp_path):
    import numpy as np
    from plugins.cv_pipeline import _parity_input
    x = _parity_input(64, None)
    assert x.shape == (1, 3, 64, 64)
    assert x.min() >= 0.0 and x.max() <= 1.0
    assert x.std() > 0.01  # realistic, not degenerate zeros

    pytest = __import__("pytest")
    pytest.importorskip("PIL")
    from PIL import Image
    p = tmp_path / "s.png"
    Image.fromarray((np.random.RandomState(0).rand(50, 70, 3) * 255).astype(np.uint8)).save(p)
    xi = _parity_input(64, str(p))
    assert xi.shape == (1, 3, 64, 64) and xi.max() <= 1.0


def test_convert_model_existing_source_without_stack(tmp_path):
    pt = tmp_path / "m.pt"
    pt.write_bytes(b"not a real model")
    res = cv.convert_model({"source": str(pt), "formats": ["onnx"]})
    # No ultralytics/torch in the app env → ONNX export step fails cleanly.
    assert res["exit_code"] == 1
    assert res["steps"].get("onnx", "").startswith("failed")
