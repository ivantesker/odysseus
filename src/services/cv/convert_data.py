"""Dataset conversion + reconciliation: COCO/VOC/Label-Studio → YOLO, auto
class-map across sources, and stratified (leak-free) train/val split.

Pure stdlib (json + xml) + the existing dataset helpers. Targets the user's real
pain: per-dataset hand-written convert_annotation_file scripts and hardcoded
class maps in merge_datasets.py.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


def _write_yolo(out_dir: Path, stem: str, lines: list[str]):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _norm_box(x, y, w, h, iw, ih):
    """Abs top-left xywh → normalized cx,cy,w,h, clamped to [0,1]."""
    cx, cy = (x + w / 2) / iw, (y + h / 2) / ih
    nw, nh = w / iw, h / ih
    clamp = lambda v: max(0.0, min(1.0, v))  # noqa: E731
    return clamp(cx), clamp(cy), clamp(nw), clamp(nh)


def coco_to_yolo(coco_json: str, out_labels_dir: str) -> dict:
    """Convert a COCO instances JSON to YOLO labels (one .txt per image)."""
    p = Path(coco_json)
    if not p.exists():
        return {"error": f"file not found: {coco_json}"}
    data = json.loads(p.read_text(encoding="utf-8"))
    cats = sorted(data.get("categories", []), key=lambda c: c["id"])
    cat_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    class_names = [c["name"] for c in cats]
    images = {im["id"]: im for im in data.get("images", [])}
    by_img = defaultdict(list)
    for ann in data.get("annotations", []):
        by_img[ann["image_id"]].append(ann)
    out = Path(out_labels_dir)
    n_img = n_box = skipped = 0
    for img_id, im in images.items():
        iw, ih = im.get("width"), im.get("height")
        if not iw or not ih:
            skipped += 1
            continue
        lines = []
        for ann in by_img.get(img_id, []):
            x, y, w, h = ann["bbox"]
            cx, cy, nw, nh = _norm_box(x, y, w, h, iw, ih)
            if nw <= 0 or nh <= 0:
                continue
            lines.append(f"{cat_to_idx[ann['category_id']]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            n_box += 1
        _write_yolo(out, Path(im["file_name"]).stem, lines)
        n_img += 1
    return {"format": "coco", "images": n_img, "boxes": n_box,
            "skipped_images": skipped, "class_names": class_names, "out_dir": str(out)}


def voc_to_yolo(voc_xml_dir: str, out_labels_dir: str, class_names: list[str] | None = None) -> dict:
    """Convert a dir of Pascal VOC XML annotations to YOLO labels."""
    root = Path(voc_xml_dir)
    if not root.exists():
        return {"error": f"dir not found: {voc_xml_dir}"}
    names = list(class_names) if class_names else []
    name_to_idx = {n: i for i, n in enumerate(names)}
    out = Path(out_labels_dir)
    n_img = n_box = skipped = 0
    for xml in sorted(root.rglob("*.xml")):
        try:
            tree = ET.parse(xml)
        except ET.ParseError:
            skipped += 1
            continue
        r = tree.getroot()
        size = r.find("size")
        if size is None:
            skipped += 1
            continue
        iw, ih = int(size.findtext("width", 0)), int(size.findtext("height", 0))
        if not iw or not ih:
            continue
        lines = []
        for obj in r.findall("object"):
            name = obj.findtext("name", "").strip()
            if name not in name_to_idx:
                name_to_idx[name] = len(names)
                names.append(name)
            bb = obj.find("bndbox")
            if bb is None:
                continue
            xmin, ymin = float(bb.findtext("xmin", 0)), float(bb.findtext("ymin", 0))
            xmax, ymax = float(bb.findtext("xmax", 0)), float(bb.findtext("ymax", 0))
            cx, cy, nw, nh = _norm_box(xmin, ymin, xmax - xmin, ymax - ymin, iw, ih)
            if nw <= 0 or nh <= 0:
                continue
            lines.append(f"{name_to_idx[name]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            n_box += 1
        _write_yolo(out, xml.stem, lines)
        n_img += 1
    return {"format": "voc", "images": n_img, "boxes": n_box, "skipped_files": skipped,
            "class_names": names, "out_dir": str(out)}


def labelstudio_to_yolo(ls_json: str, out_labels_dir: str) -> dict:
    """Convert a Label Studio JSON export (rectanglelabels) to YOLO labels."""
    p = Path(ls_json)
    if not p.exists():
        return {"error": f"file not found: {ls_json}"}
    tasks = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(tasks, dict):
        tasks = [tasks]
    names: list[str] = []
    name_to_idx: dict[str, int] = {}
    out = Path(out_labels_dir)
    n_img = n_box = 0
    for task in tasks:
        # image path can live in several spots depending on export config
        data = task.get("data", {})
        img = data.get("image") or data.get("img") or next(iter(data.values()), "")
        stem = Path(str(img)).stem or f"task_{task.get('id', n_img)}"
        anns = task.get("annotations") or task.get("completions") or []
        results = anns[0].get("result", []) if anns else task.get("result", [])
        lines = []
        for res in results:
            if res.get("type") != "rectanglelabels":
                continue
            v = res.get("value", {})
            labels = v.get("rectanglelabels") or v.get("labels") or []
            if not labels:
                continue
            name = labels[0]
            if name not in name_to_idx:
                name_to_idx[name] = len(names); names.append(name)
            # LS stores x,y,width,height as percentages of the image
            x, y, w, h = v.get("x", 0), v.get("y", 0), v.get("width", 0), v.get("height", 0)
            cx, cy = (x + w / 2) / 100.0, (y + h / 2) / 100.0
            nw, nh = w / 100.0, h / 100.0
            if nw <= 0 or nh <= 0:
                continue
            lines.append(f"{name_to_idx[name]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            n_box += 1
        _write_yolo(out, stem, lines)
        n_img += 1
    return {"format": "labelstudio", "images": n_img, "boxes": n_box,
            "class_names": names, "out_dir": str(out)}


def convert_annotations(src: str, fmt: str, out_labels_dir: str, class_names=None) -> dict:
    fmt = (fmt or "").lower()
    if fmt == "coco":
        return coco_to_yolo(src, out_labels_dir)
    if fmt in ("voc", "pascal", "pascal_voc"):
        return voc_to_yolo(src, out_labels_dir, class_names)
    if fmt in ("labelstudio", "label_studio", "ls"):
        return labelstudio_to_yolo(src, out_labels_dir)
    return {"error": f"unknown format {fmt!r}; use coco|voc|labelstudio", "exit_code": 1}


def auto_class_map(sources: list[dict]) -> dict:
    """Build a unified class map across N sources.

    sources: [{name, class_names:[...]}]. Returns a unified name list (first-seen
    order, case-insensitive dedup) + per-source {old_id: new_id} remap.
    """
    unified: list[str] = []
    lower_to_idx: dict[str, int] = {}
    for s in sources:
        for nm in s.get("class_names", []):
            key = nm.strip().lower()
            if key and key not in lower_to_idx:
                lower_to_idx[key] = len(unified)
                unified.append(nm)
    remaps = []
    for s in sources:
        m = {}
        for old_id, nm in enumerate(s.get("class_names", [])):
            new = lower_to_idx.get(nm.strip().lower())
            if new is not None:
                m[old_id] = new
        remaps.append({"name": s.get("name", ""), "map": m})
    return {"unified_names": unified, "num_classes": len(unified), "remaps": remaps}


def stratified_split(labels_dir: str, val_frac: float = 0.2, seed: int = 0,
                     source_regex: str | None = None, class_names=None) -> dict:
    """Train/val split that keeps each source's frames together (no leakage) and
    reports the per-class balance achieved in each split."""
    import hashlib
    import re
    root = Path(labels_dir)
    if not root.exists():
        return {"error": f"labels dir not found: {labels_dir}"}
    from .dataset import _iter_label_files, parse_label_line

    pat = re.compile(source_regex) if source_regex else None
    groups: dict[str, list[str]] = defaultdict(list)
    group_classes: dict[str, Counter] = defaultdict(Counter)
    stem_classes: dict[str, Counter] = {}
    for lf in _iter_label_files(root):
        stem = lf.stem
        m = pat.search(stem) if pat else None
        key = (m.group(1) if (m and m.groups()) else m.group(0)) if m else stem
        cc = Counter()
        for raw in lf.read_text(encoding="utf-8", errors="replace").splitlines():
            pr = parse_label_line(raw.strip()) if raw.strip() else None
            if pr:
                cc[pr[0]] += 1
        groups[key].append(stem)
        group_classes[key] += cc
        stem_classes[stem] = cc

    total = sum(len(v) for v in groups.values())
    if total == 0:
        return {"error": "no labels found"}
    target_val = val_frac * total
    # Deterministic order by hash(seed+key); greedily fill val to the target.
    ordered = sorted(groups, key=lambda k: hashlib.md5(f"{seed}:{k}".encode()).hexdigest())
    val_groups, val_count = set(), 0
    for k in ordered:
        if val_count < target_val:
            val_groups.add(k); val_count += len(groups[k])
    train_stems, val_stems = [], []
    train_cls, val_cls = Counter(), Counter()
    for k, stems in groups.items():
        tgt_stems, tgt_cls = (val_stems, val_cls) if k in val_groups else (train_stems, train_cls)
        tgt_stems.extend(stems)
        tgt_cls += group_classes[k]

    def _named(cnt):
        if class_names:
            return {(class_names[c] if c < len(class_names) else str(c)): n for c, n in sorted(cnt.items())}
        return {str(c): n for c, n in sorted(cnt.items())}

    return {
        "n_train": len(train_stems), "n_val": len(val_stems),
        "n_sources": len(groups), "leak_free": bool(source_regex),
        "train_per_class": _named(train_cls), "val_per_class": _named(val_cls),
        "train": train_stems[:5000], "val": val_stems[:5000],
    }
