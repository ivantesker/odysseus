"""Prediction-distribution drift (M1): compare a model's outputs on new
(unlabeled) data against an eval baseline — confidence distribution, boxes per
image, detection rate, per-class frequency — via PSI. Catches the "trained on
day, now seeing night" shift without needing new labels. Pure numpy + the
prediction .txt parser.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .dataset import _iter_label_files
from .review import _parse_pred_line

# PSI bands (industry convention): <0.1 stable, 0.1–0.25 moderate, >0.25 large.
PSI_MODERATE = 0.1
PSI_LARGE = 0.25


def pred_distribution(preds_dir: str) -> dict:
    """Summarize a prediction dir: conf samples, boxes/image, class freq, det-rate."""
    root = Path(preds_dir)
    if not root.exists():
        return {"error": f"preds dir not found: {preds_dir}"}
    confs, bpi = [], []
    per_class: Counter = Counter()
    n_images = n_with_det = 0
    for lf in _iter_label_files(root):
        n_images += 1
        cnt = 0
        for raw in lf.read_text(encoding="utf-8", errors="replace").splitlines():
            p = _parse_pred_line(raw.strip()) if raw.strip() else None
            if not p:
                continue
            cls, _cx, _cy, _w, _h, conf = p
            confs.append(conf)
            per_class[cls] += 1
            cnt += 1
        bpi.append(cnt)
        if cnt:
            n_with_det += 1
    return {
        "n_images": n_images,
        "confs": confs,
        "boxes_per_image": bpi,
        "per_class": dict(per_class),
        "detection_rate": round(n_with_det / n_images, 4) if n_images else 0.0,
        "avg_boxes": round(float(np.mean(bpi)), 3) if bpi else 0.0,
    }


def _psi(base: np.ndarray, new: np.ndarray, bins) -> tuple[float, list, list]:
    b, _ = np.histogram(base, bins=bins)
    n, _ = np.histogram(new, bins=bins)
    bp = b / max(1, b.sum())
    npr = n / max(1, n.sum())
    eps = 1e-6
    bp = np.clip(bp, eps, None)
    npr = np.clip(npr, eps, None)
    psi = float(np.sum((npr - bp) * np.log(npr / bp)))
    return psi, bp.tolist(), npr.tolist()


def _class_psi(base: dict, new: dict) -> float:
    classes = set(base) | set(new)
    bt = sum(base.values()) or 1
    nt = sum(new.values()) or 1
    eps = 1e-6
    psi = 0.0
    for c in classes:
        bp = max(base.get(c, 0) / bt, eps)
        npr = max(new.get(c, 0) / nt, eps)
        psi += (npr - bp) * np.log(npr / bp)
    return float(psi)


def _verdict(psi: float) -> str:
    return "large" if psi > PSI_LARGE else "moderate" if psi > PSI_MODERATE else "stable"


def drift(baseline_preds: str, new_preds: str) -> dict:
    """PSI drift between a baseline and new prediction set across signals."""
    base = pred_distribution(baseline_preds)
    new = pred_distribution(new_preds)
    for d in (base, new):
        if d.get("error"):
            return d

    conf_psi, conf_b, conf_n = _psi(np.asarray(base["confs"] or [0]),
                                    np.asarray(new["confs"] or [0]), bins=np.linspace(0, 1, 11))
    bpi_max = max(base["boxes_per_image"] + new["boxes_per_image"] + [1])
    bpi_psi, bpi_b, bpi_n = _psi(np.asarray(base["boxes_per_image"] or [0]),
                                 np.asarray(new["boxes_per_image"] or [0]),
                                 bins=np.linspace(0, bpi_max + 1, min(12, bpi_max + 2)))
    cls_psi = _class_psi(base["per_class"], new["per_class"])
    det_delta = round(new["detection_rate"] - base["detection_rate"], 4)

    signals = {
        "confidence": {"psi": round(conf_psi, 4), "verdict": _verdict(conf_psi),
                       "baseline": conf_b, "new": conf_n},
        "boxes_per_image": {"psi": round(bpi_psi, 4), "verdict": _verdict(bpi_psi),
                            "baseline": bpi_b, "new": bpi_n},
        "class_frequency": {"psi": round(cls_psi, 4), "verdict": _verdict(cls_psi)},
    }
    worst = max(conf_psi, bpi_psi, cls_psi)
    return {
        "signals": signals,
        "overall_verdict": _verdict(worst),
        "max_psi": round(worst, 4),
        "detection_rate": {"baseline": base["detection_rate"], "new": new["detection_rate"], "delta": det_delta},
        "n_images": {"baseline": base["n_images"], "new": new["n_images"]},
        "alert": worst > PSI_MODERATE or abs(det_delta) > 0.1,
    }
