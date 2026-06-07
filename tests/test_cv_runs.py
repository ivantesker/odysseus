"""CV training-run registry: ingest / register / list / compare."""
from src.services.cv import runs


def _make_run(d, name, epochs, final_map, lr=0.01):
    rd = d / name
    rd.mkdir()
    lines = ["epoch,train/box_loss,metrics/mAP50-95(B)"]
    for e in range(epochs):
        m = round(final_map * (e + 1) / epochs, 4)
        lines.append(f"{e},{1.0 - e * 0.1},{m}")
    (rd / "results.csv").write_text("\n".join(lines))
    (rd / "args.yaml").write_text(f"model: yolov10n\nepochs: {epochs}\nlr0: {lr}\n")
    return rd


def test_ingest_run(tmp_path):
    rd = _make_run(tmp_path, "run_a", 5, 0.5)
    rec = runs.ingest_run(str(rd))
    assert rec["epochs"] == 5
    assert rec["primary"] == "metrics/map50-95(b)"
    assert rec["best"] == 0.5
    assert rec["config"]["model"] == "yolov10n"
    assert len(rec["curves"]["metrics/map50-95(b)"]) == 5


def test_ingest_missing(tmp_path):
    assert "error" in runs.ingest_run(str(tmp_path / "nope"))
    (tmp_path / "empty").mkdir()
    assert "error" in runs.ingest_run(str(tmp_path / "empty"))  # no results.csv


def test_register_and_list(tmp_path):
    rd = _make_run(tmp_path, "reg_run", 3, 0.7)
    rec = runs.register_run(str(rd), owner="runner")
    assert rec["id"].startswith("run-")
    listed = runs.list_runs(owner="runner")
    assert any(r["id"] == rec["id"] for r in listed)
    # owner scoping
    assert all("run-" in r["id"] for r in listed)


def test_compare_runs_diff_and_primary(tmp_path):
    a = _make_run(tmp_path, "a", 5, 0.6, lr=0.01)
    b = _make_run(tmp_path, "b", 8, 0.8, lr=0.001)
    cmp = runs.compare_runs([str(a), str(b)])
    assert len(cmp["runs"]) == 2
    assert cmp["primary"] == "metrics/map50-95(b)"
    # lr0 + epochs differ → in config diff; model same → not in diff
    assert "lr0" in cmp["config_diff"] and "epochs" in cmp["config_diff"]
    assert "model" not in cmp["config_diff"]
