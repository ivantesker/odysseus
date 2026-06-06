"""YOLO-format dataset tools: lint, stats, class remap/merge, train/val split.

Pure filesystem + label math — no torch/ultralytics. A YOLO dataset here means
image files with a sibling `.txt` label per image, each line:
    <class_id> <cx> <cy> <w> <h>
with normalized [0,1] coordinates. Functions take/return plain data so they are
fully unit-testable on a temp directory.
"""

from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class LabelIssue:
    file: str
    line: int
    problem: str
    raw: str = ""


def parse_label_line(line: str) -> tuple[int, float, float, float, float] | None:
    """Parse one YOLO label line → (cls, cx, cy, w, h), or None if unparseable."""
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(p) for p in parts[1:])
    except ValueError:
        return None
    return cls, cx, cy, w, h


def lint_label_text(text: str, file: str = "", num_classes: int | None = None) -> list[LabelIssue]:
    """Return the issues in one label file's text (empty file is not an issue)."""
    issues: list[LabelIssue] = []
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parsed = parse_label_line(line)
        if parsed is None:
            issues.append(LabelIssue(file, i, "malformed (need 5 numeric fields)", raw))
            continue
        cls, cx, cy, w, h = parsed
        if cls < 0:
            issues.append(LabelIssue(file, i, f"negative class id {cls}", raw))
        if num_classes is not None and cls >= num_classes:
            issues.append(LabelIssue(file, i, f"class id {cls} >= num_classes {num_classes}", raw))
        for name, v in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
            if not (0.0 <= v <= 1.0):
                issues.append(LabelIssue(file, i, f"{name}={v} out of [0,1]", raw))
        if w <= 0 or h <= 0:
            issues.append(LabelIssue(file, i, f"non-positive box w={w} h={h}", raw))
    return issues


def _iter_label_files(labels_dir: Path):
    for p in sorted(labels_dir.rglob("*.txt")):
        if p.is_file():
            yield p


def lint_dataset(labels_dir: str | Path, num_classes: int | None = None) -> dict:
    """Lint every .txt under labels_dir. Returns a JSON-able report."""
    root = Path(labels_dir)
    if not root.exists():
        return {"error": f"labels dir not found: {labels_dir}"}
    issues: list[dict] = []
    n_files = n_empty = 0
    for lf in _iter_label_files(root):
        n_files += 1
        text = lf.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            n_empty += 1
        for iss in lint_label_text(text, str(lf), num_classes):
            issues.append(iss.__dict__)
    return {
        "label_files": n_files,
        "empty_files": n_empty,
        "issue_count": len(issues),
        "issues": issues[:500],  # cap for prompt safety
        "ok": len(issues) == 0,
    }


@dataclass
class DatasetStats:
    images: int = 0
    labels: int = 0
    images_without_label: int = 0
    labels_without_image: int = 0
    boxes: int = 0
    per_class: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return self.__dict__


def dataset_stats(images_dir: str | Path, labels_dir: str | Path,
                  class_names: list[str] | None = None) -> dict:
    """Count images/labels/boxes and per-class box counts; flag orphans."""
    img_root, lbl_root = Path(images_dir), Path(labels_dir)
    if not img_root.exists():
        return {"error": f"images dir not found: {images_dir}"}
    if not lbl_root.exists():
        return {"error": f"labels dir not found: {labels_dir}"}

    img_stems = {p.stem for p in img_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS}
    lbl_stems = {p.stem for p in _iter_label_files(lbl_root)}
    per_class: Counter = Counter()
    boxes = 0
    for lf in _iter_label_files(lbl_root):
        for raw in lf.read_text(encoding="utf-8", errors="replace").splitlines():
            parsed = parse_label_line(raw.strip()) if raw.strip() else None
            if parsed:
                per_class[parsed[0]] += 1
                boxes += 1

    def _cls_key(c: int):
        if class_names and 0 <= c < len(class_names):
            return class_names[c]
        return str(c)

    stats = DatasetStats(
        images=len(img_stems),
        labels=len(lbl_stems),
        images_without_label=len(img_stems - lbl_stems),
        labels_without_image=len(lbl_stems - img_stems),
        boxes=boxes,
        per_class={_cls_key(c): n for c, n in sorted(per_class.items())},
    )
    out = stats.as_dict()
    # Class-balance ratio (max/min) — a quick imbalance signal.
    counts = list(per_class.values())
    out["class_balance_ratio"] = round(max(counts) / min(counts), 2) if counts and min(counts) else None
    return out


def remap_label_text(text: str, mapping: dict[int, int], drop_unmapped: bool = False) -> str:
    """Apply a class-id remap to one label file's text.

    mapping: {old_id: new_id}. Unmapped classes are kept as-is unless
    drop_unmapped, in which case their lines are removed (used when merging
    datasets with different class sets).
    """
    out_lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parsed = parse_label_line(line)
        if parsed is None:
            out_lines.append(raw)
            continue
        cls, cx, cy, w, h = parsed
        if cls in mapping:
            cls = mapping[cls]
        elif drop_unmapped:
            continue
        out_lines.append(f"{cls} {cx:g} {cy:g} {w:g} {h:g}")
    return "\n".join(out_lines) + ("\n" if out_lines else "")


def plan_split(stems: list[str], val_frac: float = 0.2, seed: int = 0) -> dict:
    """Deterministic train/val split of image stems (no file I/O).

    Deterministic without RNG state: order by a salted hash of the stem so the
    same input always yields the same split (reproducible across runs/machines).
    """
    import hashlib

    val_frac = max(0.0, min(1.0, val_frac))

    def _h(stem: str) -> int:
        return int(hashlib.sha1(f"{seed}:{stem}".encode()).hexdigest(), 16)

    ordered = sorted(stems, key=_h)
    n_val = int(round(len(ordered) * val_frac))
    val = sorted(ordered[:n_val])
    train = sorted(ordered[n_val:])
    return {"train": train, "val": val, "n_train": len(train), "n_val": len(val)}


def apply_split(images_dir: str | Path, labels_dir: str | Path, out_dir: str | Path,
                val_frac: float = 0.2, seed: int = 0, copy: bool = True) -> dict:
    """Materialize a train/val split into out_dir/{train,val}/{images,labels}."""
    img_root, lbl_root, out_root = Path(images_dir), Path(labels_dir), Path(out_dir)
    imgs = {p.stem: p for p in img_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS}
    split = plan_split(list(imgs), val_frac=val_frac, seed=seed)
    moved = {"train": 0, "val": 0}
    op = shutil.copy2 if copy else shutil.move
    for subset in ("train", "val"):
        (out_root / subset / "images").mkdir(parents=True, exist_ok=True)
        (out_root / subset / "labels").mkdir(parents=True, exist_ok=True)
        for stem in split[subset]:
            src_img = imgs[stem]
            op(str(src_img), str(out_root / subset / "images" / src_img.name))
            src_lbl = lbl_root / f"{stem}.txt"
            if src_lbl.exists():
                op(str(src_lbl), str(out_root / subset / "labels" / f"{stem}.txt"))
            moved[subset] += 1
    return {"out_dir": str(out_root), **moved, "val_frac": val_frac, "seed": seed}
