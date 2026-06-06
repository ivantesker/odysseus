"""CV detection metrics — IoU / AP / mAP / latency."""
import pytest

from src.services.cv import metrics as mt


def test_iou_xyxy():
    assert mt.iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert mt.iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # half-overlap on x
    assert mt.iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3, abs=1e-6)


def test_xywh_to_xyxy():
    import numpy as np
    out = mt.xywh_to_xyxy([0.5, 0.5, 0.2, 0.2])
    assert np.allclose(out, [0.4, 0.4, 0.6, 0.6])


def test_average_precision_perfect_and_empty():
    # one TP then one FP, 1 GT: AP at full recall = 1.0
    assert mt.average_precision([True, False], [0.9, 0.8], n_gt=1) == pytest.approx(1.0)
    assert mt.average_precision([], [], n_gt=0) == 0.0
    assert mt.average_precision([], [], n_gt=5) == 0.0


def test_match_predictions_greedy():
    preds = [((0, 0, 10, 10), 0.9), ((0, 0, 10, 10), 0.5)]  # both overlap same GT
    gts = [(0, 0, 10, 10)]
    matched, scores = mt.match_predictions(preds, gts, iou_thr=0.5)
    # highest-score pred takes the GT; second becomes a false positive
    assert matched == [True, False]
    assert scores == [0.9, 0.5]


def test_map_at_iou():
    res = mt.map_at_iou(
        {0: [((0, 0, 10, 10), 0.9)], 1: [((0, 0, 10, 10), 0.8)]},
        {0: [(0, 0, 9, 9)], 1: [(50, 50, 60, 60)]},  # class1 pred misses GT
    )
    assert res["per_class_ap"][0] == pytest.approx(1.0)
    assert res["per_class_ap"][1] == 0.0
    assert res["map"] == pytest.approx(0.5)


def test_pr_curve_best_f1():
    curve = mt.pr_curve([True, False, True], [0.9, 0.8, 0.4], n_gt=2)
    assert 0.0 <= curve["ap"] <= 1.0
    assert "conf" in curve["best"] and "f1" in curve["best"]
    assert len(curve["recall"]) == len(curve["precision"]) == len(curve["conf"])
    perfect = mt.pr_curve([True, True], [0.9, 0.8], n_gt=2)
    assert perfect["ap"] == 1.0
    assert perfect["best"]["f1"] == 1.0


def test_pr_curve_empty():
    c = mt.pr_curve([], [], 0)
    assert c["ap"] == 0.0 and c["recall"] == []


def test_latency_summary():
    s = mt.latency_summary([10, 20, 30, 40])
    assert s["count"] == 4
    assert s["mean_ms"] == pytest.approx(25.0)
    assert s["fps"] == pytest.approx(40.0)
    assert mt.latency_summary([])["count"] == 0
