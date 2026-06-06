"""Detection metrics — IoU, average precision, mAP — and a latency summary.

Pure numpy. Boxes are xywh-normalized (YOLO) unless noted. Used by the
eval_detector tool to score predictions against ground truth without depending
on the training framework's own metric code.
"""

from __future__ import annotations

import numpy as np


def xywh_to_xyxy(box) -> np.ndarray:
    b = np.asarray(box, dtype=float)
    cx, cy, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def iou_xyxy(a, b) -> float:
    """IoU of two xyxy boxes."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def average_precision(matched: list[bool], scores: list[float], n_gt: int) -> float:
    """All-point AP from per-prediction (is_true_positive, score) and #GT boxes.

    matched[i] is True if prediction i (at confidence scores[i]) is a true
    positive. Implements the COCO-style all-point interpolation of the
    precision/recall curve.
    """
    if n_gt == 0:
        return 0.0
    if not matched:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=float))
    tp = np.asarray(matched, dtype=float)[order]
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    # All-point interpolation: precision envelope, integrate over recall.
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def pr_curve(matched: list[bool], scores: list[float], n_gt: int, points: int = 50) -> dict:
    """Precision/recall/F1 vs confidence threshold + the best-F1 operating point.

    Feeds the per-class threshold tuning: the returned ``best`` is the confidence
    that maximizes F1, which is what you put in the runtime/NMS config per class.
    """
    if not matched or n_gt == 0:
        return {"recall": [], "precision": [], "conf": [], "f1": [],
                "ap": 0.0, "best": {"conf": 0.25, "f1": 0.0, "precision": 0.0, "recall": 0.0}}
    order = np.argsort(-np.asarray(scores, dtype=float))
    tp = np.asarray(matched, dtype=float)[order]
    conf = np.asarray(scores, dtype=float)[order]
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(1.0 - tp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    bi = int(np.argmax(f1))
    ap = average_precision(matched, scores, n_gt)
    # Downsample the curve for compact SVG.
    n = len(recall)
    idx = np.linspace(0, n - 1, min(points, n)).astype(int)
    return {
        "recall": [round(float(recall[i]), 4) for i in idx],
        "precision": [round(float(precision[i]), 4) for i in idx],
        "conf": [round(float(conf[i]), 4) for i in idx],
        "f1": [round(float(f1[i]), 4) for i in idx],
        "ap": round(float(ap), 4),
        "best": {"conf": round(float(conf[bi]), 4), "f1": round(float(f1[bi]), 4),
                 "precision": round(float(precision[bi]), 4), "recall": round(float(recall[bi]), 4)},
    }


def match_predictions(preds, gts, iou_thr: float = 0.5):
    """Greedy match predictions to GT (single class), highest score first.

    preds: list of (xyxy, score); gts: list of xyxy. Returns (matched, scores)
    aligned to score-desc order — feed straight into average_precision.
    """
    order = sorted(range(len(preds)), key=lambda i: -preds[i][1])
    used = set()
    matched, scores = [], []
    for i in order:
        box, score = preds[i]
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if j in used:
                continue
            v = iou_xyxy(box, g)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_thr:
            used.add(best_j)
            matched.append(True)
        else:
            matched.append(False)
        scores.append(score)
    return matched, scores


def map_at_iou(per_class_preds: dict, per_class_gt: dict, iou_thr: float = 0.5) -> dict:
    """mAP@iou across classes.

    per_class_preds: {cls: [(xyxy, score), ...]}
    per_class_gt:    {cls: [xyxy, ...]}
    Returns {"map": float, "per_class_ap": {cls: ap}}.
    """
    classes = set(per_class_preds) | set(per_class_gt)
    aps = {}
    for c in classes:
        preds = per_class_preds.get(c, [])
        gts = per_class_gt.get(c, [])
        matched, scores = match_predictions(preds, gts, iou_thr)
        aps[c] = average_precision(matched, scores, len(gts))
    mean_ap = float(np.mean(list(aps.values()))) if aps else 0.0
    return {"map": mean_ap, "iou_thr": iou_thr, "per_class_ap": aps}


def latency_summary(times_ms: list[float]) -> dict:
    """Summarize per-inference latencies (ms) → mean/p50/p90/p99/fps."""
    if not times_ms:
        return {"count": 0}
    arr = np.asarray(times_ms, dtype=float)
    mean = float(arr.mean())
    return {
        "count": int(arr.size),
        "mean_ms": round(mean, 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p90_ms": round(float(np.percentile(arr, 90)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "fps": round(1000.0 / mean, 2) if mean > 0 else None,
    }
