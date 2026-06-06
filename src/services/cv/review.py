"""Model-vs-ground-truth review: the single matcher behind suspect-label triage,
the confusion matrix, and two-model (PT vs quantized) accuracy diff.

Pure: it consumes YOLO-format label dirs — ground truth (``<cls> cx cy w h``) and
predictions (``<cls> cx cy w h [conf]``) — so the heavy inference happens
elsewhere (eval_detector / the user's own predict run) and this stays numpy-only
and unit-testable. Builds on metrics.iou_xyxy + map_at_iou.
"""

from __future__ import annotations

from pathlib import Path

from .dataset import _iter_label_files
from .metrics import iou_xyxy, map_at_iou, xywh_to_xyxy


def _parse_pred_line(line: str):
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(p) for p in parts[1:5])
        conf = float(parts[5]) if len(parts) >= 6 else 1.0
    except ValueError:
        return None
    return cls, cx, cy, w, h, conf


def load_labels(d, with_conf: bool = False) -> dict:
    """Return {stem: [(cls, xyxy[, conf]), ...]} from a YOLO label dir."""
    out: dict[str, list] = {}
    root = Path(d)
    if not root.exists():
        return out
    for lf in _iter_label_files(root):
        boxes = []
        for raw in lf.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line:
                continue
            p = _parse_pred_line(line)
            if p is None:
                continue
            cls, cx, cy, w, h, conf = p
            xyxy = tuple(xywh_to_xyxy([cx, cy, w, h]).tolist())
            boxes.append((cls, xyxy, conf) if with_conf else (cls, xyxy))
        out[lf.stem] = boxes
    return out


def _match_image(gt, preds, iou_thr):
    """Cross-class greedy match (highest conf first). Returns
    (pairs[(gt_cls,pred_cls,iou,conf)], unmatched_preds, unmatched_gts)."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i][2])
    used = set()
    pairs, fp = [], []
    for i in order:
        pcls, pbox, conf = preds[i]
        best_iou, best_j = 0.0, -1
        for j, (gcls, gbox) in enumerate(gt):
            if j in used:
                continue
            v = iou_xyxy(pbox, gbox)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_thr:
            used.add(best_j)
            pairs.append((gt[best_j][0], pcls, best_iou, conf))
        else:
            fp.append((pcls, conf))
    fn = [gt[j][0] for j in range(len(gt)) if j not in used]
    return pairs, fp, fn


def suspect_labels(labels_dir, preds_dir, iou_thr: float = 0.5,
                   conf_missing: float = 0.9, limit: int = 200) -> dict:
    """Rank likely annotation errors using model predictions vs ground truth.

    Emits: possible_missing (a high-conf pred with no GT → a missed annotation),
    possible_spurious (a GT box no prediction matches → likely an extra/wrong box),
    label_mismatch (matched box, predicted class ≠ labeled class). Sorted worst-first.
    """
    gt = load_labels(labels_dir)
    preds = load_labels(preds_dir, with_conf=True)
    stems = set(gt) & set(preds)
    suspects = []
    counts = {"possible_missing": 0, "possible_spurious": 0, "label_mismatch": 0}
    for stem in stems:
        pairs, fp, fn = _match_image(gt[stem], preds[stem], iou_thr)
        for gcls, pcls, iou, conf in pairs:
            if gcls != pcls:
                counts["label_mismatch"] += 1
                suspects.append({"file": stem, "reason": "label_mismatch",
                                 "gt_cls": gcls, "pred_cls": pcls, "conf": round(conf, 3),
                                 "iou": round(iou, 3), "score": round(conf, 3)})
        for pcls, conf in fp:
            if conf >= conf_missing:
                counts["possible_missing"] += 1
                suspects.append({"file": stem, "reason": "possible_missing",
                                 "pred_cls": pcls, "conf": round(conf, 3), "score": round(conf, 3)})
        for gcls in fn:
            counts["possible_spurious"] += 1
            suspects.append({"file": stem, "reason": "possible_spurious",
                             "gt_cls": gcls, "score": 0.5})
    suspects.sort(key=lambda s: -s["score"])
    return {"counts": counts, "total": len(suspects), "suspects": suspects[:limit]}


def confusion_matrix(labels_dir, preds_dir, iou_thr: float = 0.5,
                     class_names: list[str] | None = None) -> dict:
    """(n+1)×(n+1) confusion matrix incl a background row/col for FN/FP."""
    gt = load_labels(labels_dir)
    preds = load_labels(preds_dir, with_conf=True)
    stems = set(gt) & set(preds)
    classes = set()
    for boxes in list(gt.values()) + [[(c, b) for c, b, _ in v] for v in preds.values()]:
        for cls, *_ in boxes:
            classes.add(cls)
    classes = sorted(classes)
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    bg = n  # background index
    mat = [[0] * (n + 1) for _ in range(n + 1)]
    for stem in stems:
        pairs, fp, fn = _match_image(gt[stem], preds[stem], iou_thr)
        for gcls, pcls, _iou, _conf in pairs:
            mat[idx[gcls]][idx[pcls]] += 1
        for pcls, _conf in fp:
            mat[bg][idx[pcls]] += 1          # predicted but no GT → background row
        for gcls in fn:
            mat[idx[gcls]][bg] += 1          # GT but missed → background col
    labels = [(class_names[c] if class_names and c < len(class_names) else str(c)) for c in classes] + ["background"]
    return {"labels": labels, "matrix": mat, "classes": classes}


def _by_class(boxes_with_conf):
    out: dict[int, list] = {}
    for cls, box, conf in boxes_with_conf:
        out.setdefault(cls, []).append((box, conf))
    return out


def _gt_by_class(boxes):
    out: dict[int, list] = {}
    for cls, box in boxes:
        out.setdefault(cls, []).append(box)
    return out


def _map_for(gt: dict, preds: dict, iou_thr: float) -> dict:
    per_class_preds: dict[int, list] = {}
    per_class_gt: dict[int, list] = {}
    for stem in set(gt) | set(preds):
        for c, items in _by_class(preds.get(stem, [])).items():
            per_class_preds.setdefault(c, []).extend(items)
        for c, items in _gt_by_class(gt.get(stem, [])).items():
            per_class_gt.setdefault(c, []).extend(items)
    return map_at_iou(per_class_preds, per_class_gt, iou_thr)


def compare_models(labels_dir, preds_a_dir, preds_b_dir, iou_thr: float = 0.5,
                   class_names: list[str] | None = None) -> dict:
    """Compare two prediction sets (e.g. PT vs INT8-RKNN) against GT.

    Returns overall + per-class mAP for A and B, the per-class delta, and the
    images where B lost a true positive A had (regression list).
    """
    gt = load_labels(labels_dir)
    pa = load_labels(preds_a_dir, with_conf=True)
    pb = load_labels(preds_b_dir, with_conf=True)
    map_a = _map_for(gt, pa, iou_thr)
    map_b = _map_for(gt, pb, iou_thr)

    def _nm(c):
        return class_names[c] if class_names and c < len(class_names) else str(c)

    per_class_delta = {}
    for c in set(map_a["per_class_ap"]) | set(map_b["per_class_ap"]):
        a = map_a["per_class_ap"].get(c, 0.0)
        b = map_b["per_class_ap"].get(c, 0.0)
        per_class_delta[_nm(c)] = round(b - a, 4)

    # Per-image regressions: B has more FN than A.
    regressions = []
    for stem in set(gt):
        _, _, fn_a = _match_image(gt[stem], pa.get(stem, []), iou_thr)
        _, _, fn_b = _match_image(gt[stem], pb.get(stem, []), iou_thr)
        if len(fn_b) > len(fn_a):
            regressions.append({"file": stem, "extra_missed": len(fn_b) - len(fn_a)})
    regressions.sort(key=lambda r: -r["extra_missed"])

    return {
        "map_a": round(map_a["map"], 4),
        "map_b": round(map_b["map"], 4),
        "map_delta": round(map_b["map"] - map_a["map"], 4),
        "per_class_delta": per_class_delta,
        "regressions": regressions[:100],
        "n_regressions": len(regressions),
    }
