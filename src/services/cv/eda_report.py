"""Render an EDA dict (src.services.cv.eda) to a static HTML report.

Inline CSS + inline SVG only (no JS, no chart libs) so it renders anywhere and
survives a strict CSP. Dark palette tuned to the Odysseus look. The page is
self-contained; the route just wraps it in an HTML response.
"""

from __future__ import annotations

import html

# Odysseus dark palette (matches static/style.css defaults).
_BG = "#282c34"
_CARD = "#1d2128"
_BORDER = "#355a66"
_FG = "#9cdef2"
_MUTED = "#8a93a6"
_ACCENT = "#e06c75"   # --red, the default accent
# Categorical class palette — index a class to keep its hue identical across
# every chart (so "Front" reads as the same colour in class-dist + co-occurrence).
_BARS = ["#61afef", "#50fa7b", "#d19a66", "#c678dd", "#56b6c2",
         "#e5c07b", "#e06c75", "#f0ad4e", "#9cdef2", "#4caf50"]


def _esc(s) -> str:
    return html.escape(str(s))


def _bar_chart(data: dict, *, width=560, bar_h=22, gap=8, color=None) -> str:
    """Horizontal bar chart from {label: value}. Returns inline SVG."""
    if not data:
        return '<p class="muted">no data</p>'
    items = list(data.items())
    vmax = max(v for _, v in items) or 1
    label_w = 150
    plot_w = width - label_w - 60
    h = len(items) * (bar_h + gap) + gap
    rows = []
    for i, (label, val) in enumerate(items):
        y = gap + i * (bar_h + gap)
        bw = max(1, int(plot_w * val / vmax))
        c = color or _BARS[i % len(_BARS)]
        rows.append(
            f'<text x="{label_w - 8}" y="{y + bar_h * 0.7}" text-anchor="end" class="lbl">{_esc(label)}</text>'
            f'<rect x="{label_w}" y="{y}" width="{bw}" height="{bar_h}" rx="3" fill="{c}"/>'
            f'<text x="{label_w + bw + 6}" y="{y + bar_h * 0.7}" class="val">{_esc(val)}</text>'
        )
    return f'<svg viewBox="0 0 {width} {h}" width="100%" class="chart" role="img">{"".join(rows)}</svg>'


def _heatmap(grid_data: list, *, size=300) -> str:
    """Square heatmap from a 2D int grid (box-center density). Inline SVG."""
    if not grid_data:
        return '<p class="muted">no data</p>'
    g = len(grid_data)
    vmax = max((max(row) for row in grid_data if row), default=0) or 1
    cell = size / g
    cells = []
    for y, row in enumerate(grid_data):
        for x, v in enumerate(row):
            if v <= 0:
                continue
            op = 0.12 + 0.88 * (v / vmax)
            cells.append(
                f'<rect x="{x * cell:.1f}" y="{y * cell:.1f}" width="{cell:.1f}" height="{cell:.1f}" '
                f'fill="{_ACCENT}" fill-opacity="{op:.2f}"><title>{v} boxes</title></rect>'
            )
    border = (
        f'<rect x="0" y="0" width="{size}" height="{size}" fill="none" stroke="{_BORDER}" stroke-width="1"/>'
        f'<line x1="{size/2}" y1="0" x2="{size/2}" y2="{size}" stroke="{_BORDER}" stroke-width="0.5" stroke-dasharray="3"/>'
        f'<line x1="0" y1="{size/2}" x2="{size}" y2="{size/2}" stroke="{_BORDER}" stroke-width="0.5" stroke-dasharray="3"/>'
    )
    return f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" class="chart" role="img">{"".join(cells)}{border}</svg>'


def _stat_cards(summary: dict) -> str:
    order = [
        ("images", "images"), ("boxes", "boxes"), ("classes", "classes"),
        ("avg_boxes_per_image", "avg boxes/img"), ("empty_label_files", "empty labels"),
        ("images_without_label", "imgs w/o label"), ("malformed_or_oob", "bad boxes"),
        ("class_balance_ratio", "imbalance ×"),
    ]
    cards = []
    for key, label in order:
        if key not in summary or summary[key] is None:
            continue
        val = summary[key]
        warn = key in ("empty_label_files", "images_without_label", "malformed_or_oob") and val
        bad = key == "class_balance_ratio" and isinstance(val, (int, float)) and val >= 5
        cls = "card warn" if (warn or bad) else "card"
        cards.append(f'<div class="{cls}"><div class="num">{_esc(val)}</div><div class="cap">{_esc(label)}</div></div>')
    return f'<div class="cards">{"".join(cards)}</div>'


def _section(title: str, body: str, note: str = "") -> str:
    n = f'<span class="note">{_esc(note)}</span>' if note else ""
    return f'<section><h2>{_esc(title)}{n}</h2>{body}</section>'


def _grade_color(grade: str) -> str:
    return {"A": "#50fa7b", "B": "#9cdef2", "C": "#f0ad4e", "D": "#e06c75"}.get(grade, _MUTED)


def _health_badge(hs: dict) -> str:
    if not hs:
        return ""
    c = _grade_color(hs.get("grade", ""))
    return (f'<div class="hscore" style="border-color:{c};color:{c}">'
            f'<span class="g">{_esc(hs.get("grade", "?"))}</span>'
            f'<span class="s">{_esc(hs.get("score", "?"))}/100</span>'
            f'<span class="cap">health</span></div>')


def _recommendations(recs: list) -> str:
    if not recs:
        return ""
    items = "".join(f"<li>{_esc(r)}</li>" for r in recs)
    return _section("Recommendations", f'<ul class="recs">{items}</ul>')


def _warnings_section(warnings: dict) -> str:
    if not warnings:
        return ""
    labels = {
        "orphan_class": ("ERROR", "class id outside num_classes"),
        "duplicate_overlap": ("WARN", "same-class boxes overlapping IoU>0.9 (double-annotation)"),
        "edge_clipped": ("WARN", "box touches the frame border (possibly truncated)"),
        "tiny_box": ("WARN", "very small box (<0.08% of frame)"),
        "implausible_aspect": ("WARN", "extreme width/height ratio"),
    }
    rows = []
    for key, (sev, desc) in labels.items():
        info = warnings.get(key, {})
        cnt = info.get("count", 0)
        if not cnt:
            continue
        ex = ", ".join(_esc(e.get("file", "")) for e in info.get("examples", [])[:4])
        sev_c = _ACCENT if sev == "ERROR" else "#f0ad4e"
        rows.append(
            f'<tr><td><span class="sev" style="color:{sev_c}">{sev}</span></td>'
            f'<td class="k">{_esc(key)}</td><td class="n">{cnt}</td>'
            f'<td class="d">{_esc(desc)}<div class="ex">{ex}</div></td></tr>'
        )
    if not rows:
        return _section("Label warnings", '<p class="ok">✓ no geometric label warnings</p>')
    return _section("Label warnings",
                    f'<table class="warn"><thead><tr><th></th><th>type</th><th>count</th><th>meaning · examples</th></tr></thead>'
                    f'<tbody>{"".join(rows)}</tbody></table>')


def _diverging_bars(delta: dict, *, width=560, bar_h=22, gap=8) -> str:
    """Signed two-colour bars from {label: value} (e.g. per-class AP delta)."""
    if not delta:
        return '<p class="muted">no data</p>'
    items = sorted(delta.items(), key=lambda kv: kv[1])
    amax = max((abs(v) for _, v in items), default=1) or 1
    label_w, mid = 150, 320
    half = width - mid - 40
    h = len(items) * (bar_h + gap) + gap
    rows = [f'<line x1="{mid}" y1="0" x2="{mid}" y2="{h}" stroke="{_BORDER}"/>']
    for i, (label, val) in enumerate(items):
        y = gap + i * (bar_h + gap)
        bw = int(half * abs(val) / amax)
        if val >= 0:
            x, c = mid, "#50fa7b"
        else:
            x, c = mid - bw, _ACCENT
        rows.append(
            f'<text x="{label_w}" y="{y + bar_h*0.7}" text-anchor="end" class="lbl">{_esc(label)}</text>'
            f'<rect x="{x}" y="{y}" width="{max(1,bw)}" height="{bar_h}" rx="3" fill="{c}"/>'
            f'<text x="{(mid+bw+6) if val>=0 else (mid-bw-6)}" y="{y+bar_h*0.7}" '
            f'text-anchor="{"start" if val>=0 else "end"}" class="val">{val:+.3f}</text>'
        )
    return f'<svg viewBox="0 0 {width} {h}" width="100%" class="chart" role="img">{"".join(rows)}</svg>'


def _confusion_svg(cm: dict, *, cell=46) -> str:
    """Confusion matrix as an SVG heatmap (rows=GT, cols=pred, last=background)."""
    labels = cm.get("labels", [])
    mat = cm.get("matrix", [])
    if not labels or not mat:
        return '<p class="muted">no data</p>'
    n = len(labels)
    pad_l, pad_t = 90, 70
    W, H = pad_l + n * cell + 20, pad_t + n * cell + 20
    vmax = max((max(r) for r in mat), default=1) or 1
    out = []
    for r in range(n):
        for c in range(n):
            v = mat[r][c]
            x, y = pad_l + c * cell, pad_t + r * cell
            op = 0.08 + 0.9 * (v / vmax) if v else 0.0
            fill = "#50fa7b" if r == c and r < n - 1 else _ACCENT
            out.append(f'<rect x="{x}" y="{y}" width="{cell-1}" height="{cell-1}" fill="{fill}" fill-opacity="{op:.2f}"/>')
            if v:
                tc = "#0e0f13" if op > 0.55 else _FG
                out.append(f'<text x="{x+cell/2}" y="{y+cell/2+4}" text-anchor="middle" font-size="12" fill="{tc}">{v}</text>')
    for i, lab in enumerate(labels):
        out.append(f'<text x="{pad_l+i*cell+cell/2}" y="{pad_t-8}" text-anchor="middle" class="lbl" transform="rotate(-35 {pad_l+i*cell+cell/2} {pad_t-8})">{_esc(lab)}</text>')
        out.append(f'<text x="{pad_l-8}" y="{pad_t+i*cell+cell/2+4}" text-anchor="end" class="lbl">{_esc(lab)}</text>')
    out.append(f'<text x="{pad_l+n*cell/2}" y="16" text-anchor="middle" class="val">predicted →</text>')
    out.append(f'<text x="14" y="{pad_t+n*cell/2}" text-anchor="middle" class="val" transform="rotate(-90 14 {pad_t+n*cell/2})">ground truth →</text>')
    return f'<svg viewBox="0 0 {W} {H}" width="100%" class="chart" role="img">{"".join(out)}</svg>'


def _suspect_table(sl: dict) -> str:
    counts = sl.get("counts", {})
    head = " · ".join(f"{k.replace('possible_','').replace('_',' ')}: {v}" for k, v in counts.items())
    rows = []
    for s in sl.get("suspects", [])[:60]:
        reason = s.get("reason", "")
        rc = {"possible_missing": _ACCENT, "label_mismatch": "#f0ad4e", "possible_spurious": "#9cdef2"}.get(reason, _MUTED)
        detail = []
        if "gt_cls" in s:
            detail.append(f'gt={s["gt_cls"]}')
        if "pred_cls" in s:
            detail.append(f'pred={s["pred_cls"]}')
        if "conf" in s:
            detail.append(f'conf={s["conf"]}')
        if "iou" in s:
            detail.append(f'iou={s["iou"]}')
        rows.append(f'<tr><td class="k">{_esc(s.get("file",""))}</td>'
                    f'<td><span class="sev" style="color:{rc}">{_esc(reason)}</span></td>'
                    f'<td class="d">{_esc(" ".join(detail))}</td></tr>')
    body = (f'<p class="muted">{_esc(head)} · {sl.get("total",0)} total (worst {len(rows)} shown)</p>'
            f'<table class="warn"><thead><tr><th>image</th><th>reason</th><th>detail</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')
    return body


def _class_color(cls: int) -> str:
    return _BARS[cls % len(_BARS)]


def _class_legend(per_class: dict, class_names=None) -> str:
    """Roboflow-style colour legend: a chip per class with its colour + count.

    Colours match the boxes drawn in the gallery (indexed by class id), so a
    class reads as the same hue in the legend, the bars, and the overlays.
    """
    if not per_class:
        return ""
    # Map display-name → class index to keep colours consistent with overlays.
    name_to_idx = {}
    if class_names:
        name_to_idx = {n: i for i, n in enumerate(class_names)}
    chips = []
    for name, cnt in per_class.items():
        idx = name_to_idx.get(name)
        if idx is None:
            try:
                idx = int(name)
            except (TypeError, ValueError):
                idx = abs(hash(name)) % len(_BARS)
        c = _class_color(idx)
        chips.append(f'<span class="lg"><span class="sw" style="background:{c}"></span>{_esc(name)}'
                     f'<span class="cnt">{_esc(cnt)}</span></span>')
    return f'<div class="legend">{"".join(chips)}</div>'


def _annotation_samples_section(samples: list, class_names=None, per_class=None) -> str:
    if not samples:
        return ""
    legend = _class_legend(per_class or {}, class_names)
    tiles, boxes_overlays = [], []
    for i, s in enumerate(samples):
        w, h = s.get("width", 1), s.get("height", 1)
        sw = max(2.0, round(max(w, h) / 150, 1))
        chip_h = max(11, int(max(w, h) / 26))
        overlay = []
        for b in s.get("boxes", []):
            c = _class_color(b["cls"])
            name = class_names[b["cls"]] if class_names and b["cls"] < len(class_names) else str(b["cls"])
            cw = max(18, len(name) * chip_h * 0.55)
            ty = b["y"] if b["y"] > chip_h else b["y"] + b["h"]
            overlay.append(
                f'<rect x="{b["x"]}" y="{b["y"]}" width="{b["w"]}" height="{b["h"]}" '
                f'fill="{c}" fill-opacity="0.10" stroke="{c}" stroke-width="{sw}"/>'
                f'<rect x="{b["x"]}" y="{ty-chip_h}" width="{cw}" height="{chip_h}" fill="{c}"/>'
                f'<text x="{b["x"]+3}" y="{ty-chip_h*0.25}" font-size="{int(chip_h*0.72)}" '
                f'font-weight="700" fill="#0e0f13">{_esc(name)}</text>'
            )
        ov = "".join(overlay)
        cap = f'{_esc(s.get("name",""))} · {len(s.get("boxes",[]))} boxes'
        svg = (f'<svg viewBox="0 0 {w} {h}" role="img" preserveAspectRatio="xMidYMid meet">'
               f'<image href="{s["data_uri"]}" x="0" y="0" width="{w}" height="{h}"/>{ov}</svg>')
        # Small tile is an anchor to the full-screen lightbox; pure CSS :target.
        tiles.append(f'<a class="atile" href="#z{i}">{svg}<div class="acap">{_esc(cap)}</div></a>')
        boxes_overlays.append(
            f'<div class="lb" id="z{i}"><a class="lbbg" href="#_"></a>'
            f'<div class="lbox">{svg}<div class="lbcap">{_esc(cap)} — click outside to close</div></div></div>'
        )
    body = legend + f'<div class="agrid">{"".join(tiles)}</div>' + "".join(boxes_overlays)
    return _section("Annotation gallery", body,
                    "click any image to zoom — eyeball label quality (like Roboflow / Ultralytics HUB)")


def _image_quality_section(scan: dict) -> str:
    if not scan or scan.get("error"):
        return ""
    cards = (
        f'<div class="cards">'
        f'<div class="card"><div class="num">{scan.get("scanned",0)}</div><div class="cap">images scanned</div></div>'
        f'<div class="card {"warn" if scan.get("brightness",{}).get("dark_or_bright",{}).get("count") else ""}">'
        f'<div class="num">{scan.get("brightness",{}).get("dark_or_bright",{}).get("count",0)}</div><div class="cap">too dark/bright</div></div>'
        f'<div class="card {"warn" if scan.get("blur",{}).get("blurry",{}).get("count") else ""}">'
        f'<div class="num">{scan.get("blur",{}).get("blurry",{}).get("count",0)}</div><div class="cap">blurry</div></div>'
        f'<div class="card {"warn" if scan.get("exact_duplicates",{}).get("count") else ""}">'
        f'<div class="num">{scan.get("exact_duplicates",{}).get("count",0)}</div><div class="cap">exact dups</div></div>'
        f'<div class="card {"warn" if scan.get("near_duplicates",{}).get("count") else ""}">'
        f'<div class="num">{scan.get("near_duplicates",{}).get("count",0)}</div><div class="cap">near dups</div></div>'
        f'</div>'
    )
    bright = _bar_chart(scan.get("brightness", {}).get("hist", {}), color="#e5c07b")
    blur = _bar_chart(scan.get("blur", {}).get("hist", {}), color="#56b6c2")
    return _section("Image quality", cards
                    + '<h2 style="font-size:12px">brightness distribution</h2>' + bright
                    + '<h2 style="font-size:12px">sharpness (Laplacian variance — low = blurry)</h2>' + blur,
                    "night/glare/blur frames + duplicates poison training")


def render_review_html(review: dict, title: str = "Model review") -> str:
    """Render the GT-vs-predictions review: suspects + confusion + model diff."""
    parts = []
    if review.get("suspect_labels"):
        parts.append(_section("Suspect labels (review these first)", _suspect_table(review["suspect_labels"]),
                              "ranked likely annotation errors from model vs ground truth"))
    if review.get("confusion_matrix"):
        parts.append(_section("Confusion matrix", _confusion_svg(review["confusion_matrix"]),
                              "diagonal = correct; off-diagonal = confused class pairs"))
    if review.get("compare"):
        cmp = review["compare"]
        cards = (f'<div class="cards">'
                 f'<div class="card"><div class="num">{cmp.get("map_a")}</div><div class="cap">mAP model A</div></div>'
                 f'<div class="card"><div class="num">{cmp.get("map_b")}</div><div class="cap">mAP model B</div></div>'
                 f'<div class="card {"warn" if (cmp.get("map_delta") or 0)<0 else ""}"><div class="num">{cmp.get("map_delta"):+}</div><div class="cap">mAP Δ (B−A)</div></div>'
                 f'<div class="card {"warn" if cmp.get("n_regressions") else ""}"><div class="num">{cmp.get("n_regressions")}</div><div class="cap">regressed images</div></div>'
                 f'</div>')
        parts.append(_section("Model comparison (A vs B)", cards + _diverging_bars(cmp.get("per_class_delta", {})),
                              "per-class AP change — red = the converted/quantized model got worse"))
    if not parts:
        parts.append(_section("Review", '<p class="muted">no review data</p>'))
    return _page(title, "".join(parts))


def render_eda_html(eda: dict, title: str = "Dataset EDA") -> str:
    if eda.get("error"):
        body = f'<section><p class="err">{_esc(eda["error"])}</p></section>'
        return _page(title, body)

    summary = eda.get("summary", {})
    health = []
    if summary.get("malformed_or_oob"):
        health.append(f'{summary["malformed_or_oob"]} malformed/out-of-bounds boxes')
    if summary.get("empty_label_files"):
        health.append(f'{summary["empty_label_files"]} empty label files')
    if summary.get("images_without_label"):
        health.append(f'{summary["images_without_label"]} images without a label')
    if summary.get("labels_without_image"):
        health.append(f'{summary["labels_without_image"]} labels without an image')
    ratio = summary.get("class_balance_ratio")
    if isinstance(ratio, (int, float)) and ratio >= 5:
        health.append(f'class imbalance {ratio}× (most vs least frequent)')
    health_html = (
        '<ul class="health">' + "".join(f"<li>⚠ {_esc(h)}</li>" for h in health) + "</ul>"
        if health else '<p class="ok">✓ no obvious dataset health issues</p>'
    )

    cooc = eda.get("cooccurrence", [])
    cooc_data = {f'{c["a"]} + {c["b"]}': c["count"] for c in cooc}

    parts = [
        _health_badge(eda.get("health_score", {})),
        _stat_cards(summary),
        _recommendations(eda.get("recommendations", [])),
    ]
    # Annotation gallery is the hero — show the labels first, Roboflow-style.
    if eda.get("samples"):
        parts.append(_annotation_samples_section(eda["samples"], eda.get("class_names"), eda.get("per_class")))
    parts += [
        _section("Health check", health_html),
        _warnings_section(eda.get("warnings", {})),
        _section("Class distribution", _bar_chart(eda.get("per_class", {})),
                 "box count per class — watch for imbalance"),
        _section("Boxes per image", _bar_chart(eda.get("boxes_per_image_hist", {}), color="#4aa3e0"),
                 "how crowded each image is"),
        _section("Object size (area % of image)", _bar_chart(eda.get("area_buckets", {}), color="#5ad08a"),
                 "small objects are the hard ones for edge detectors"),
        _section("Aspect ratio (w/h)", _bar_chart(eda.get("aspect_buckets", {}), color="#e0b84a")),
        _section("Where objects appear (center heatmap)", _heatmap(eda.get("center_heatmap", [])),
                 "bbox-center density across the frame"),
    ]
    if cooc_data:
        parts.append(_section("Class co-occurrence", _bar_chart(cooc_data, color="#a06ae0"),
                              "classes that appear together in the same image"))
    if eda.get("image_scan"):
        parts.append(_image_quality_section(eda["image_scan"]))
    return _page(title, "".join(parts))


def _page(title: str, body: str) -> str:
    return f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)} — Odysseus · ollama-only</title>
<style>
:root {{ color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:{_BG}; color:{_FG}; font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }}
.wrap {{ max-width: 900px; margin: 0 auto; padding: 28px 20px 80px; }}
header h1 {{ font-size: 22px; margin: 0 0 2px; }}
header .sub {{ color:{_MUTED}; font-size:12px; margin-bottom:22px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:10px; margin-bottom:24px; }}
.card {{ background:{_CARD}; border:1px solid {_BORDER}; border-radius:10px; padding:12px 14px; }}
.card.warn {{ border-color:{_ACCENT}; }}
.card .num {{ font-size:22px; font-weight:700; }}
.card .cap {{ color:{_MUTED}; font-size:11px; text-transform:uppercase; letter-spacing:.04em; margin-top:2px; }}
section {{ background:{_CARD}; border:1px solid {_BORDER}; border-radius:12px; padding:16px 18px; margin-bottom:16px; }}
h2 {{ font-size:14px; margin:0 0 12px; display:flex; align-items:baseline; gap:10px; }}
h2 .note {{ color:{_MUTED}; font-size:11px; font-weight:400; }}
.chart .lbl {{ fill:{_FG}; font-size:12px; }}
.chart .val {{ fill:{_MUTED}; font-size:11px; }}
.muted {{ color:{_MUTED}; }} .err {{ color:{_ACCENT}; }}
.ok {{ color:#5ad08a; }}
ul.health {{ margin:0; padding-left:18px; }} ul.health li {{ color:#e0b84a; margin:3px 0; }}
ul.recs {{ margin:0; padding-left:18px; }} ul.recs li {{ margin:5px 0; }}
.hscore {{ display:inline-flex; align-items:baseline; gap:8px; border:1.5px solid; border-radius:10px; padding:8px 14px; margin-bottom:18px; }}
.hscore .g {{ font-size:26px; font-weight:800; }} .hscore .s {{ font-size:14px; }} .hscore .cap {{ color:{_MUTED}; font-size:11px; text-transform:uppercase; letter-spacing:.05em; }}
table.warn {{ width:100%; border-collapse:collapse; font-size:12px; }}
table.warn th {{ text-align:left; color:{_MUTED}; font-weight:500; border-bottom:1px solid {_BORDER}; padding:4px 8px; }}
table.warn td {{ padding:6px 8px; border-bottom:1px solid {_BORDER}33; vertical-align:top; }}
table.warn td.k {{ font-family:ui-monospace,monospace; color:{_FG}; }} table.warn td.n {{ font-weight:700; }}
table.warn td.d {{ color:{_MUTED}; }} table.warn .ex {{ color:{_MUTED}; font-size:11px; opacity:.7; margin-top:2px; font-family:ui-monospace,monospace; }}
.sev {{ font-weight:700; font-size:11px; letter-spacing:.04em; }}
.legend {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:14px; }}
.legend .lg {{ display:inline-flex; align-items:center; gap:6px; background:{_BG}; border:1px solid {_BORDER}; border-radius:20px; padding:3px 10px 3px 6px; font-size:12px; }}
.legend .sw {{ width:12px; height:12px; border-radius:3px; display:inline-block; }}
.legend .cnt {{ color:{_MUTED}; font-size:11px; margin-left:2px; }}
.agrid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); gap:12px; }}
.atile {{ display:block; border:1px solid {_BORDER}; border-radius:10px; overflow:hidden; background:#000; text-decoration:none; color:inherit; cursor:zoom-in; transition:transform .12s, box-shadow .12s; }}
.atile:hover {{ transform:translateY(-2px); box-shadow:0 6px 18px #0008; border-color:{_ACCENT}; }}
.atile svg {{ display:block; width:100%; height:auto; }}
.acap {{ font-size:11px; color:{_MUTED}; padding:6px 8px; font-family:ui-monospace,monospace; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
/* Pure-CSS lightbox: clicking a tile targets its overlay. */
.lb {{ display:none; position:fixed; inset:0; z-index:1000; background:#000d; align-items:center; justify-content:center; padding:2vh; }}
.lb:target {{ display:flex; }}
.lb .lbbg {{ position:absolute; inset:0; cursor:zoom-out; }}
.lb .lbox {{ position:relative; display:flex; flex-direction:column; align-items:center; }}
.lb .lbox svg {{ height:86vh; width:auto; max-width:94vw; border:1px solid {_BORDER}; border-radius:8px; background:#000; }}
.lb .lbcap {{ color:{_MUTED}; font-size:12px; text-align:center; margin-top:8px; font-family:ui-monospace,monospace; }}
footer {{ color:{_MUTED}; font-size:11px; margin-top:24px; text-align:center; }}
</style></head>
<body><div class="wrap">
<header><h1>{_esc(title)}</h1><div class="sub">YOLO dataset EDA · generated by Odysseus CV pipeline</div></header>
{body}
<footer>Odysseus · ollama-only — CV pipeline</footer>
</div></body></html>"""
