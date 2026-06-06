"""CV dataset tools — lint / stats / remap / split (pure, temp dirs)."""
from pathlib import Path

from src.services.cv import dataset as ds


def test_parse_label_line():
    assert ds.parse_label_line("0 0.5 0.5 0.2 0.2") == (0, 0.5, 0.5, 0.2, 0.2)
    assert ds.parse_label_line("bad") is None
    assert ds.parse_label_line("0 0.5 0.5 0.2") is None


def test_lint_label_text_flags_problems():
    text = (
        "0 0.5 0.5 0.2 0.2\n"     # ok
        "5 0.5 0.5 0.2 0.2\n"     # class >= num_classes
        "0 1.5 0.5 0.2 0.2\n"     # cx out of [0,1]
        "0 0.5 0.5 0 0.2\n"       # non-positive w
        "garbage line\n"          # malformed
    )
    issues = ds.lint_label_text(text, "f.txt", num_classes=3)
    problems = " ".join(i.problem for i in issues)
    assert "class id 5" in problems
    assert "out of [0,1]" in problems
    assert "non-positive" in problems
    assert "malformed" in problems


def test_lint_dataset(tmp_path):
    lbl = tmp_path / "labels"
    lbl.mkdir()
    (lbl / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (lbl / "b.txt").write_text("9 0.5 0.5 0.2 0.2\n")  # bad class
    (lbl / "empty.txt").write_text("")
    rep = ds.lint_dataset(str(lbl), num_classes=3)
    assert rep["label_files"] == 3
    assert rep["empty_files"] == 1
    assert rep["issue_count"] == 1
    assert rep["ok"] is False


def test_dataset_stats(tmp_path):
    img = tmp_path / "images"
    lbl = tmp_path / "labels"
    img.mkdir(); lbl.mkdir()
    for name in ("a", "b", "c"):
        (img / f"{name}.jpg").write_bytes(b"x")
    (lbl / "a.txt").write_text("0 .5 .5 .2 .2\n1 .5 .5 .2 .2\n")
    (lbl / "b.txt").write_text("0 .5 .5 .2 .2\n")
    # c.jpg has no label -> images_without_label
    st = ds.dataset_stats(str(img), str(lbl), class_names=["Front", "Back", "Side"])
    assert st["images"] == 3
    assert st["boxes"] == 3
    assert st["images_without_label"] == 1
    assert st["per_class"] == {"Front": 2, "Back": 1}
    assert st["class_balance_ratio"] == 2.0


def test_remap_label_text():
    text = "0 .5 .5 .2 .2\n2 .5 .5 .2 .2\n"
    # 0->1, drop class 2
    out = ds.remap_label_text(text, {0: 1, 2: -1}, drop_unmapped=False)
    assert out.startswith("1 ")
    assert "\n2 " not in "\n" + out  # 2 was remapped to -1, not kept as 2
    dropped = ds.remap_label_text("3 .5 .5 .2 .2\n", {0: 1}, drop_unmapped=True)
    assert dropped == ""  # class 3 unmapped + drop_unmapped -> removed


def test_plan_split_deterministic():
    stems = [f"img{i}" for i in range(10)]
    a = ds.plan_split(stems, val_frac=0.3, seed=42)
    b = ds.plan_split(stems, val_frac=0.3, seed=42)
    assert a == b  # deterministic
    assert a["n_val"] == 3 and a["n_train"] == 7
    assert not (set(a["train"]) & set(a["val"]))  # disjoint


def test_apply_split_materializes(tmp_path):
    img = tmp_path / "images"; lbl = tmp_path / "labels"; out = tmp_path / "out"
    img.mkdir(); lbl.mkdir()
    for i in range(10):
        (img / f"i{i}.jpg").write_bytes(b"x")
        (lbl / f"i{i}.txt").write_text("0 .5 .5 .2 .2\n")
    res = ds.apply_split(str(img), str(lbl), str(out), val_frac=0.2, seed=0, copy=True)
    assert res["train"] + res["val"] == 10
    assert (out / "train" / "images").exists()
    assert len(list((out / "val" / "images").glob("*.jpg"))) == res["val"]
    # copy=True leaves originals in place
    assert len(list(img.glob("*.jpg"))) == 10
