"""CV dataset EDA — compute + HTML render."""
from src.services.cv import eda, eda_report


def _make_dataset(tmp_path, rows_per_img):
    lbl = tmp_path / "labels"
    lbl.mkdir()
    for i, rows in enumerate(rows_per_img):
        lines = [f"{c} {cx} {cy} {w} {h}" for (c, cx, cy, w, h) in rows]
        (lbl / f"{i}.txt").write_text("\n".join(lines))
    return lbl


def test_compute_eda_basic(tmp_path):
    lbl = _make_dataset(tmp_path, [
        [(0, 0.5, 0.5, 0.2, 0.2), (0, 0.1, 0.1, 0.05, 0.05)],
        [(1, 0.5, 0.5, 0.3, 0.15)],
        [],  # empty
    ])
    r = eda.compute_eda(str(lbl), class_names=["Front", "Back"])
    s = r["summary"]
    assert s["label_files"] == 3
    assert s["empty_label_files"] == 1
    assert s["boxes"] == 3
    assert s["classes"] == 2
    assert r["per_class"] == {"Front": 2, "Back": 1}
    # boxes-per-image hist: one img with 2, one with 1, one with 0
    assert r["boxes_per_image_hist"]["0"] == 1
    assert r["boxes_per_image_hist"]["2"] == 1
    # heatmap counts == boxes
    assert sum(sum(row) for row in r["center_heatmap"]) == 3


def test_compute_eda_counts_bad_boxes(tmp_path):
    lbl = tmp_path / "labels"
    lbl.mkdir()
    (lbl / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n0 1.5 0.5 0.2 0.2\nbadline\n")
    r = eda.compute_eda(str(lbl))
    assert r["summary"]["malformed_or_oob"] == 2  # oob cx + malformed
    assert r["summary"]["boxes"] == 1


def test_compute_eda_cooccurrence(tmp_path):
    lbl = _make_dataset(tmp_path, [
        [(0, 0.5, 0.5, 0.2, 0.2), (1, 0.3, 0.3, 0.2, 0.2)],  # 0+1 together
        [(0, 0.5, 0.5, 0.2, 0.2), (1, 0.3, 0.3, 0.2, 0.2)],
    ])
    r = eda.compute_eda(str(lbl), class_names=["A", "B"])
    assert r["cooccurrence"][0] == {"a": "A", "b": "B", "count": 2}


def test_compute_eda_missing_dir():
    assert "error" in eda.compute_eda("/no/such/dir")


def test_render_eda_html_has_charts(tmp_path):
    lbl = _make_dataset(tmp_path, [[(0, 0.5, 0.5, 0.2, 0.2)], [(1, 0.2, 0.2, 0.1, 0.1)]])
    r = eda.compute_eda(str(lbl), class_names=["Front", "Back"])
    html = eda_report.render_eda_html(r, "Test EDA")
    assert html.startswith("<!doctype html")
    assert html.rstrip().endswith("</html>")
    assert "<svg" in html
    assert "Test EDA" in html
    assert "Class distribution" in html
    # title is HTML-escaped (no raw injection)
    bad = eda_report.render_eda_html(r, "<script>x</script>")
    assert "<script>x" not in bad


def test_render_eda_html_error():
    html = eda_report.render_eda_html({"error": "boom"}, "x")
    assert "boom" in html and "<!doctype html" in html


def test_eda_warnings_recommendations_and_health(tmp_path):
    lbl = tmp_path / "labels"
    lbl.mkdir()
    # tiny + duplicate-overlap + edge-clipped + orphan class
    (lbl / "a.txt").write_text(
        "0 0.5 0.5 0.02 0.02\n0 0.5 0.5 0.02 0.02\n"   # tiny + dup-overlap
        "0 0.01 0.5 0.02 0.4\n"                          # edge-clipped
        "9 0.5 0.5 0.1 0.1\n"                            # orphan class (num_classes=3)
    )
    (lbl / "b.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    r = eda.compute_eda(str(lbl), class_names=["A", "B", "C"], num_classes=3)
    wc = r["warning_counts"]
    assert wc["tiny_box"] >= 1
    assert wc["duplicate_overlap"] >= 1
    assert wc["edge_clipped"] >= 1
    assert wc["orphan_class"] == 1
    assert r["health_score"]["grade"] in ("A", "B", "C", "D")
    assert any("class id outside" in x for x in r["recommendations"])
    html = eda_report.render_eda_html(r, "warn demo")
    assert "Label warnings" in html and "Recommendations" in html and "hscore" in html


def test_render_review_html():
    review = {
        "suspect_labels": {"counts": {"possible_missing": 1, "possible_spurious": 0, "label_mismatch": 1},
                           "total": 2, "suspects": [
                               {"file": "img1", "reason": "label_mismatch", "gt_cls": 0, "pred_cls": 1, "conf": 0.9, "iou": 0.7, "score": 0.9}]},
        "confusion_matrix": {"labels": ["car", "truck", "background"],
                             "matrix": [[3, 1, 0], [0, 2, 1], [1, 0, 0]], "classes": [0, 1]},
        "compare": {"map_a": 0.5, "map_b": 0.4, "map_delta": -0.1,
                    "per_class_delta": {"car": -0.2, "truck": 0.1}, "regressions": [], "n_regressions": 3},
    }
    html = eda_report.render_review_html(review, "review demo")
    assert html.startswith("<!doctype html")
    assert "Confusion matrix" in html and "Suspect labels" in html and "Model comparison" in html
    assert "<svg" in html
