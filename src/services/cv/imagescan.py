"""Image-level dataset quality scan: brightness, blur, entropy, exact/near dups.

Decodes images (lazy Pillow — already a dep via qrcode[pil]) so it is gated and
sampled by default. Pure numpy past the decode: Laplacian-variance blur, Shannon
entropy, average-hash near-duplicates. Surfaces the night/glare/motion-blur and
duplicate frames that quietly poison training — common in ADAS footage.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .dataset import IMAGE_EXTS

# 4-neighbour Laplacian; variance of the response is a classic sharpness proxy.
_LAP = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def _gray_small(path: Path, max_side: int = 256):
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("L")
        w, h = im.size
        scale = max_side / max(w, h) if max(w, h) > max_side else 1.0
        if scale < 1.0:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        return np.asarray(im, dtype=np.float32), (w, h)


def _laplacian_var(g: np.ndarray) -> float:
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    # valid 3x3 convolution, pure numpy.
    s = np.lib.stride_tricks.sliding_window_view(g, (3, 3))
    resp = np.einsum("ijkl,kl->ij", s, _LAP)
    return float(resp.var())


def _entropy(g: np.ndarray) -> float:
    hist, _ = np.histogram(g, bins=256, range=(0, 255))
    p = hist / max(1, hist.sum())
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _ahash(path: Path) -> int:
    # 8x8 average hash → stable 64-bit fingerprint for near-duplicate detection.
    from PIL import Image
    with Image.open(path) as im:
        small = np.asarray(im.convert("L").resize((8, 8)), dtype=np.float32)
    bits = (small > small.mean()).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def scan_images(images_dir, *, sample: int = 400, max_side: int = 256,
                near_dup_dist: int = 5) -> dict:
    """Scan up to `sample` images. Returns per-metric histograms, flagged bands,
    and exact/near-duplicate groups. A missing Pillow yields a clear error."""
    root = Path(images_dir)
    if not root.exists():
        return {"error": f"images dir not found: {images_dir}"}
    try:
        import PIL  # noqa: F401
    except Exception:
        return {"error": "Pillow not installed — `pip install pillow`"}

    paths = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
    total = len(paths)
    if sample and total > sample:
        step = total / sample
        paths = [paths[int(i * step)] for i in range(sample)]

    brightness, blur, entropy = [], [], []
    sha_groups: dict[str, list] = {}
    ahashes: list[tuple[str, int]] = []
    errors = 0
    for p in paths:
        try:
            g, _wh = _gray_small(p, max_side)
            brightness.append(float(g.mean()))
            blur.append(_laplacian_var(g))
            entropy.append(_entropy(g))
            sha = hashlib.sha1(p.read_bytes()).hexdigest()
            sha_groups.setdefault(sha, []).append(p.name)
            ahashes.append((p.name, _ahash(p)))
        except Exception:
            errors += 1

    def _band(vals, lo_pct, hi_pct):
        if not vals:
            return {"count": 0}
        a = np.asarray(vals)
        lo, hi = np.percentile(a, lo_pct), np.percentile(a, hi_pct)
        flagged = int(np.sum((a < lo) | (a > hi)))
        return {"count": flagged, "lo": round(float(lo), 2), "hi": round(float(hi), 2)}

    # near-dups by aHash Hamming distance (O(n^2) over the sample — fine for a few hundred)
    near = []
    for i in range(len(ahashes)):
        for j in range(i + 1, len(ahashes)):
            d = _hamming(ahashes[i][1], ahashes[j][1])
            if d <= near_dup_dist:
                near.append({"a": ahashes[i][0], "b": ahashes[j][0], "dist": d})
    exact = [{"hash": h[:10], "files": v} for h, v in sha_groups.items() if len(v) > 1]

    return {
        "scanned": len(brightness),
        "total_images": total,
        "errors": errors,
        "brightness": {"hist": _hist(brightness, 16, 0, 255), "dark_or_bright": _band(brightness, 3, 97)},
        "blur": {"hist": _hist(blur, 16), "blurry": _band_low(blur, 5)},
        "entropy": {"hist": _hist(entropy, 16)},
        "exact_duplicates": {"groups": exact[:50], "count": sum(len(g["files"]) for g in exact)},
        "near_duplicates": {"pairs": near[:100], "count": len(near)},
    }


def sample_annotations(images_dir, labels_dir, *, n: int = 9, max_side: int = 900) -> list:
    """Return up to n labelled sample images for the report.

    Each entry: {name, data_uri (downscaled JPEG/PNG base64), width, height,
    boxes:[{cls, x, y, w, h}]} with box coords in DISPLAYED pixels — the report
    overlays them as SVG <rect>s so annotation quality is visible at a glance.
    """
    import base64
    import io

    from .dataset import parse_label_line

    root = Path(images_dir)
    lroot = Path(labels_dir)
    if not root.exists() or not lroot.exists():
        return []
    try:
        from PIL import Image
    except Exception:
        return []

    from .dataset import parse_label_line as _pl

    # Index label files by stem (recursive) so train/val both work.
    lbl_by_stem = {p.stem: p for p in lroot.rglob("*.txt") if p.is_file()}
    imgs = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMAGE_EXTS and p.stem in lbl_by_stem]
    if not imgs:
        return []

    # Pick a REPRESENTATIVE spread (Roboflow-style preview): cover as many
    # classes as possible, then fill with a range of crowdedness. Peek each
    # candidate's class set + box count (cheap line parse).
    meta = []
    for p in imgs:
        classes, cnt = set(), 0
        for raw in lbl_by_stem[p.stem].read_text(encoding="utf-8", errors="replace").splitlines():
            pr = _pl(raw.strip()) if raw.strip() else None
            if pr:
                classes.add(pr[0]); cnt += 1
        meta.append((p, classes, cnt))
    picked, seen_classes, chosen = [], set(), set()
    for p, classes, _cnt in sorted(meta, key=lambda m: -m[2]):  # class coverage first
        if classes - seen_classes:
            picked.append(p); seen_classes |= classes; chosen.add(p)
            if len(picked) >= n:
                break
    if len(picked) < n:  # fill with an even spread across crowdedness
        rest = [m[0] for m in sorted(meta, key=lambda m: -m[2]) if m[0] not in chosen]
        step = max(1, len(rest) // max(1, n - len(picked)))
        picked += rest[::step][: n - len(picked)]

    out = []
    for p in picked:
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                w0, h0 = im.size
                scale = max_side / max(w0, h0) if max(w0, h0) > max_side else 1.0
                dw, dh = max(1, int(w0 * scale)), max(1, int(h0 * scale))
                disp = im.resize((dw, dh))
                buf = io.BytesIO()
                disp.save(buf, format="JPEG", quality=70)
                uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
            boxes = []
            for raw in lbl_by_stem[p.stem].read_text(encoding="utf-8", errors="replace").splitlines():
                pr = parse_label_line(raw.strip()) if raw.strip() else None
                if not pr:
                    continue
                cls, cx, cy, bw, bh = pr
                boxes.append({"cls": cls,
                              "x": round((cx - bw / 2) * dw, 1), "y": round((cy - bh / 2) * dh, 1),
                              "w": round(bw * dw, 1), "h": round(bh * dh, 1)})
            out.append({"name": p.name, "data_uri": uri, "width": dw, "height": dh, "boxes": boxes})
        except Exception:
            continue
    return out


def _band_low(vals, pct):
    if not vals:
        return {"count": 0}
    a = np.asarray(vals)
    thr = np.percentile(a, pct)
    return {"count": int(np.sum(a < thr)), "threshold": round(float(thr), 2)}


def _hist(vals, bins, lo=None, hi=None) -> dict:
    if not vals:
        return {}
    a = np.asarray(vals, dtype=float)
    lo = a.min() if lo is None else lo
    hi = a.max() if hi is None else hi
    if hi <= lo:
        hi = lo + 1
    counts, edges = np.histogram(a, bins=bins, range=(lo, hi))
    return {f"{edges[i]:.0f}": int(counts[i]) for i in range(len(counts))}
