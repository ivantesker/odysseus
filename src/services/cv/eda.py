"""Exploratory data analysis for a YOLO object-detection dataset.

Computed from the label .txt files alone (normalized coords) in a SINGLE pass:
distributions + geometric health warnings + actionable recommendations. Label
reads are parallelized (I/O-bound) so a multi-thousand-file set scans in seconds.
numpy only — no image decode, no torch.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np

from .dataset import IMAGE_EXTS, _iter_label_files, parse_label_line

_AREA_BUCKETS = [("tiny <0.1%", 0.001), ("small <1%", 0.01), ("medium <10%", 0.10), ("large ≥10%", 1.01)]
_ASPECT_BUCKETS = [("tall <0.5", 0.5), ("portrait <1", 1.0), ("landscape <2", 2.0), ("wide ≥2", 1e9)]

# Geometric-warning thresholds (normalized).
TINY_AREA = 0.0008      # < ~0.08% of the frame — very small object
EDGE_EPS = 0.005        # box side within this of the frame border → clipped
DUP_IOU = 0.9           # same-class boxes overlapping more than this → duplicate
EXTREME_ASPECT = 10.0   # w/h above this (or below 1/this) → implausible


def _iou_xywh(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = a[3] * a[4] + b[3] * b[4] - inter
    return inter / ua if ua > 0 else 0.0


def _cls_name(c: int, class_names):
    if class_names and 0 <= c < len(class_names):
        return class_names[c]
    return str(c)


def _bucket(value: float, buckets) -> str:
    for name, hi in buckets:
        if value < hi:
            return name
    return buckets[-1][0]


def _read(lf: Path) -> tuple[str, str]:
    return lf.stem, lf.read_text(encoding="utf-8", errors="replace")


def compute_eda(labels_dir, images_dir=None, class_names=None, num_classes=None, grid: int = 12) -> dict:
    root = Path(labels_dir)
    if not root.exists():
        return {"error": f"labels dir not found: {labels_dir}"}
    if num_classes is None and class_names:
        num_classes = len(class_names)

    label_files = list(_iter_label_files(root))
    # Parallel read — file I/O dominates on a multi-thousand-file set.
    with ThreadPoolExecutor(max_workers=min(32, (len(label_files) or 1))) as pool:
        contents = list(pool.map(_read, label_files)) if label_files else []

    per_class: Counter = Counter()
    boxes_per_image: list[int] = []
    areas: list[float] = []
    aspects: list[float] = []
    area_buckets: Counter = Counter()
    aspect_buckets: Counter = Counter()
    heat = np.zeros((grid, grid), dtype=int)
    cooccur: Counter = Counter()
    n_empty = n_boxes = n_malformed = 0
    warn: dict[str, list] = {"tiny_box": [], "edge_clipped": [], "duplicate_overlap": [],
                             "implausible_aspect": [], "orphan_class": []}

    for stem, text in contents:
        if not text.strip():
            n_empty += 1
            boxes_per_image.append(0)
            continue
        classes_here = set()
        boxes_here = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            parsed = parse_label_line(line)
            if parsed is None:
                n_malformed += 1
                continue
            cls, cx, cy, w, h = parsed
            if w <= 0 or h <= 0 or not (0 <= cx <= 1 and 0 <= cy <= 1):
                n_malformed += 1
                continue
            per_class[cls] += 1
            classes_here.add(cls)
            n_boxes += 1
            boxes_here.append(parsed)
            area = w * h
            ar = w / h if h > 0 else 0.0
            areas.append(area)
            aspects.append(ar)
            area_buckets[_bucket(area, _AREA_BUCKETS)] += 1
            aspect_buckets[_bucket(ar, _ASPECT_BUCKETS)] += 1
            heat[min(grid - 1, int(cy * grid)), min(grid - 1, int(cx * grid))] += 1
            # ── geometric warnings ──
            if area < TINY_AREA:
                warn["tiny_box"].append({"file": stem, "cls": cls})
            if (cx - w / 2 <= EDGE_EPS or cx + w / 2 >= 1 - EDGE_EPS
                    or cy - h / 2 <= EDGE_EPS or cy + h / 2 >= 1 - EDGE_EPS):
                warn["edge_clipped"].append({"file": stem, "cls": cls})
            if ar > EXTREME_ASPECT or (ar and ar < 1 / EXTREME_ASPECT):
                warn["implausible_aspect"].append({"file": stem, "cls": cls, "aspect": round(ar, 2)})
            if num_classes is not None and cls >= num_classes:
                warn["orphan_class"].append({"file": stem, "cls": cls})
        boxes_per_image.append(len(boxes_here))
        for a, b in combinations(sorted(classes_here), 2):
            cooccur[(a, b)] += 1
        # duplicate-overlap: same class, IoU above threshold, within one image
        for i in range(len(boxes_here)):
            for j in range(i + 1, len(boxes_here)):
                if boxes_here[i][0] == boxes_here[j][0] and _iou_xywh(boxes_here[i], boxes_here[j]) > DUP_IOU:
                    warn["duplicate_overlap"].append({"file": stem, "cls": boxes_here[i][0]})
                    break

    # Orphan check (images vs labels).
    n_images = images_without_label = labels_without_image = 0
    if images_dir and Path(images_dir).exists():
        img_stems = {p.stem for p in Path(images_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTS}
        lbl_stems = {s for s, _ in contents}
        n_images = len(img_stems)
        images_without_label = len(img_stems - lbl_stems)
        labels_without_image = len(lbl_stems - img_stems)

    bpi = np.asarray(boxes_per_image) if boxes_per_image else np.array([0])
    counts = list(per_class.values())
    ratio = round(max(counts) / min(counts), 1) if counts and min(counts) else None

    warn_counts = {k: len(v) for k, v in warn.items()}
    summary = {
        "images": n_images,
        "label_files": len(label_files),
        "empty_label_files": n_empty,
        "boxes": n_boxes,
        "classes": len(per_class),
        "images_without_label": images_without_label,
        "labels_without_image": labels_without_image,
        "malformed_or_oob": n_malformed,
        "avg_boxes_per_image": round(float(bpi.mean()), 2),
        "max_boxes_per_image": int(bpi.max()),
        "class_balance_ratio": ratio,
    }
    per_class_named = {_cls_name(c, class_names): n for c, n in sorted(per_class.items(), key=lambda kv: -kv[1])}
    out = {
        "summary": summary,
        "per_class": per_class_named,
        "boxes_per_image_hist": _histogram(bpi, max_bins=12),
        "area_buckets": {k: area_buckets.get(k, 0) for k, _ in _AREA_BUCKETS},
        "aspect_buckets": {k: aspect_buckets.get(k, 0) for k, _ in _ASPECT_BUCKETS},
        "center_heatmap": heat.tolist(),
        "grid": grid,
        "cooccurrence": [
            {"a": _cls_name(a, class_names), "b": _cls_name(b, class_names), "count": n}
            for (a, b), n in cooccur.most_common(12)
        ],
        "area_stats": _stats(areas),
        "aspect_stats": _stats(aspects),
        "warnings": {k: {"count": warn_counts[k], "examples": warn[k][:8]} for k in warn},
        "warning_counts": warn_counts,
    }
    out["recommendations"] = recommendations(out)
    out["health_score"] = _health_score(out)
    return out


def recommendations(eda: dict) -> list[str]:
    """Plain-English next steps derived from the computed signals."""
    recs = []
    s = eda.get("summary", {})
    wc = eda.get("warning_counts", {})
    area = eda.get("area_buckets", {})
    per_class = eda.get("per_class", {})
    ratio = s.get("class_balance_ratio")
    if isinstance(ratio, (int, float)) and ratio >= 3 and per_class:
        rarest = min(per_class, key=per_class.get)
        recs.append(f"Class imbalance {ratio}× — collect/augment more '{rarest}' samples (rarest class).")
    boxes = s.get("boxes", 0) or 1
    tiny_frac = (area.get("tiny <0.1%", 0) + area.get("small <1%", 0)) / boxes
    if tiny_frac > 0.4:
        recs.append(f"{tiny_frac:.0%} of boxes are small/tiny — enable tiling or raise input resolution (e.g. 256→512) so the detector keeps small objects.")
    if wc.get("edge_clipped", 0) > 0.05 * boxes:
        recs.append("Many edge-clipped boxes — verify annotations near frame borders aren't truncated by the labeling tool.")
    if wc.get("duplicate_overlap", 0):
        recs.append(f"{wc['duplicate_overlap']} duplicate-overlapping boxes (IoU>0.9, same class) — likely double-annotations to dedupe.")
    if wc.get("orphan_class", 0):
        recs.append(f"{wc['orphan_class']} boxes use a class id outside num_classes — fix the class map or data.yaml.")
    if s.get("images_without_label", 0):
        recs.append(f"{s['images_without_label']} images have no label — confirm they're intentional negatives, not missed annotations.")
    if s.get("malformed_or_oob", 0):
        recs.append(f"{s['malformed_or_oob']} malformed/out-of-bounds label lines — clean before training.")
    if not recs:
        recs.append("No major issues detected — dataset looks training-ready.")
    return recs


def _health_score(eda: dict) -> dict:
    """0–100 rollup: start at 100, subtract for each issue category present."""
    s = eda.get("summary", {})
    wc = eda.get("warning_counts", {})
    boxes = max(1, s.get("boxes", 0))
    score = 100
    ratio = s.get("class_balance_ratio") or 1
    if ratio >= 5:
        score -= 15
    elif ratio >= 3:
        score -= 7
    score -= min(20, int(100 * s.get("malformed_or_oob", 0) / boxes) * 4)
    score -= min(15, int(100 * wc.get("duplicate_overlap", 0) / boxes) * 3)
    score -= 10 if wc.get("orphan_class", 0) else 0
    score -= min(10, int(100 * wc.get("edge_clipped", 0) / boxes))
    score = max(0, min(100, score))
    grade = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D"
    return {"score": score, "grade": grade}


def _histogram(values: np.ndarray, max_bins: int = 12) -> dict:
    if values.size == 0:
        return {}
    vmax = int(values.max())
    if vmax <= max_bins:
        return {str(v): int(np.sum(values == v)) for v in range(0, vmax + 1)}
    edges = np.linspace(0, vmax + 1, max_bins + 1).astype(int)
    out = {}
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        out[f"{lo}-{hi - 1}"] = int(np.sum((values >= lo) & (values < hi)))
    return out


def _stats(values) -> dict:
    if not values:
        return {"count": 0}
    a = np.asarray(values, dtype=float)
    return {
        "count": int(a.size),
        "min": round(float(a.min()), 5),
        "p50": round(float(np.percentile(a, 50)), 5),
        "mean": round(float(a.mean()), 5),
        "p95": round(float(np.percentile(a, 95)), 5),
        "max": round(float(a.max()), 5),
    }
