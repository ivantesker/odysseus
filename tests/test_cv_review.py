"""CV review — suspect labels, confusion matrix, two-model diff (pure)."""
from src.services.cv import review


def _write(d, name, lines):
    (d / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def _setup(tmp_path):
    gt = tmp_path / "gt"; pa = tmp_path / "pa"; pb = tmp_path / "pb"
    for x in (gt, pa, pb):
        x.mkdir()
    return gt, pa, pb


def test_load_labels_with_and_without_conf(tmp_path):
    d = tmp_path / "p"; d.mkdir()
    _write(d, "1", ["0 0.5 0.5 0.2 0.2 0.9"])
    gt = review.load_labels(str(d))
    pr = review.load_labels(str(d), with_conf=True)
    assert len(gt["1"][0]) == 2          # (cls, xyxy)
    assert len(pr["1"][0]) == 3 and pr["1"][0][2] == 0.9


def test_suspect_labels_finds_all_reasons(tmp_path):
    gt, pa, _ = _setup(tmp_path)
    # img1: GT car; pred labels it truck (mismatch) + a high-conf extra pred (missing)
    _write(gt, "1", ["0 0.5 0.5 0.2 0.2"])
    _write(pa, "1", ["1 0.5 0.5 0.2 0.2 0.9", "0 0.9 0.9 0.05 0.05 0.97"])
    # img2: GT box with NO matching pred -> possible_spurious
    _write(gt, "2", ["0 0.2 0.2 0.1 0.1"])
    _write(pa, "2", [])
    sl = review.suspect_labels(str(gt), str(pa))
    assert sl["counts"]["label_mismatch"] == 1
    assert sl["counts"]["possible_missing"] == 1
    assert sl["counts"]["possible_spurious"] == 1
    assert sl["total"] == 3


def test_confusion_matrix_shape_and_background(tmp_path):
    gt, pa, _ = _setup(tmp_path)
    _write(gt, "1", ["0 0.5 0.5 0.2 0.2", "1 0.2 0.2 0.1 0.1"])
    _write(pa, "1", ["0 0.5 0.5 0.2 0.2 0.9"])  # truck missed -> truck->background
    cm = review.confusion_matrix(str(gt), str(pa), class_names=["car", "truck"])
    assert cm["labels"] == ["car", "truck", "background"]
    assert len(cm["matrix"]) == 3 and len(cm["matrix"][0]) == 3
    assert cm["matrix"][0][0] == 1           # car -> car
    assert cm["matrix"][1][2] == 1           # truck -> background (missed)


def test_failure_cases_ranks_errors(tmp_path):
    gt, pa, _ = _setup(tmp_path)
    for i in range(4):
        _write(gt, str(i), ["0 0.5 0.5 0.2 0.2"])
    _write(pa, "0", ["0 0.5 0.5 0.2 0.2 0.9"])                              # clean
    _write(pa, "1", ["0 0.5 0.5 0.2 0.2 0.9", "0 0.9 0.9 0.05 0.05 0.8"])  # FP
    _write(pa, "2", [])                                                     # FN
    _write(pa, "3", ["1 0.5 0.5 0.2 0.2 0.9"])                              # mismatch
    fc = review.failure_cases(str(gt), str(pa))
    assert fc["totals"] == {"fp": 1, "fn": 1, "mismatch": 1}
    assert fc["n_failing"] == 3                # img0 clean, excluded
    assert fc["cases"][0]["file"] == "3"       # mismatch weighted highest
    kinds = {b["kind"] for c in fc["cases"] for b in c["boxes"]}
    assert {"fp", "fn", "mismatch"} <= kinds


def test_pr_analysis_recommends_threshold(tmp_path):
    gt, pa, _ = _setup(tmp_path)
    for i in range(5):
        _write(gt, str(i), ["0 0.5 0.5 0.2 0.2"])
        _write(pa, str(i), ["0 0.5 0.5 0.2 0.2 0.9"])
    pr = review.pr_analysis(str(gt), str(pa), class_names=["car"])
    assert pr["per_class"]["car"]["ap"] == 1.0
    assert "car" in pr["recommended_conf"]
    assert 0.0 <= pr["map"] <= 1.0


def test_compare_models_delta_and_regressions(tmp_path):
    gt, pa, pb = _setup(tmp_path)
    for i in range(4):
        _write(gt, str(i), ["0 0.5 0.5 0.2 0.2"])
        _write(pa, str(i), ["0 0.5 0.5 0.2 0.2 0.9"])  # A finds all
        _write(pb, str(i), [] if i < 2 else ["0 0.5 0.5 0.2 0.2 0.9"])  # B misses 2
    cmp = review.compare_models(str(gt), str(pa), str(pb), class_names=["car"])
    assert cmp["map_a"] >= cmp["map_b"]      # B is worse
    assert cmp["map_delta"] <= 0
    assert cmp["n_regressions"] == 2
