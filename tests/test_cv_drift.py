"""CV prediction-drift (PSI) monitoring."""
from src.services.cv import drift


def _preds(d, name, n, conf, boxes=1, cls=0):
    pd = d / name
    pd.mkdir()
    for i in range(n):
        lines = "\n".join(f"{cls} 0.5 0.5 0.1 0.1 {conf}" for _ in range(boxes))
        (pd / f"{i}.txt").write_text(lines)
    return pd


def test_pred_distribution(tmp_path):
    pd = _preds(tmp_path, "p", 10, 0.8, boxes=2)
    dist = drift.pred_distribution(str(pd))
    assert dist["n_images"] == 10
    assert dist["detection_rate"] == 1.0
    assert dist["avg_boxes"] == 2.0
    assert len(dist["confs"]) == 20


def test_drift_stable_when_identical(tmp_path):
    a = _preds(tmp_path, "a", 30, 0.8, boxes=2)
    b = _preds(tmp_path, "b", 30, 0.8, boxes=2)
    d = drift.drift(str(a), str(b))
    assert d["overall_verdict"] == "stable"
    assert d["max_psi"] < drift.PSI_MODERATE
    assert d["alert"] is False


def test_drift_large_when_different(tmp_path):
    # baseline: high conf, 1 box; new: low conf, many boxes
    a = _preds(tmp_path, "a", 40, 0.9, boxes=1, cls=0)
    b = _preds(tmp_path, "b", 40, 0.3, boxes=4, cls=1)
    d = drift.drift(str(a), str(b))
    assert d["overall_verdict"] == "large"
    assert d["alert"] is True
    assert d["signals"]["confidence"]["psi"] > drift.PSI_MODERATE


def test_drift_missing_dir(tmp_path):
    a = _preds(tmp_path, "a", 5, 0.8)
    assert "error" in drift.drift(str(a), "/no/such")
