"""CV eval-CSV aggregator."""
from src.services.cv import evalcsv


def _run(d, name, text):
    rd = d / name
    rd.mkdir()
    (rd / "results.csv").write_text(text)


def test_aggregate_sorts_by_primary_and_strips_prefix(tmp_path):
    _run(tmp_path, "run_a", "epoch,metrics/map50,metrics/map50-95\n0,0.5,0.3\n1,0.7,0.45\n")
    _run(tmp_path, "run_b", "epoch,metrics/map50,metrics/map50-95\n0,0.4,0.25\n1,0.6,0.40\n")
    agg = evalcsv.aggregate_csvs(str(tmp_path))
    # prefix 'metrics/' stripped cleanly (not charset-stripped → no 'ap50')
    assert "map50-95" in agg["metrics"] and "map50" in agg["metrics"]
    assert agg["primary"] == "map50-95"
    # best run first
    assert agg["runs"][0]["name"] == "run_a"
    assert agg["runs"][0]["metrics"]["map50-95"] == 0.45


def test_aggregate_uses_last_row(tmp_path):
    _run(tmp_path, "r", "epoch,map\n0,0.1\n1,0.2\n2,0.9\n")
    agg = evalcsv.aggregate_csvs(str(tmp_path))
    assert agg["runs"][0]["metrics"]["map"] == 0.9  # final row


def test_aggregate_missing_and_empty(tmp_path):
    assert "error" in evalcsv.aggregate_csvs("/no/such")
    (tmp_path / "x").mkdir()
    assert "error" in evalcsv.aggregate_csvs(str(tmp_path))  # no CSVs


def test_aggregate_skips_deps_and_depth(tmp_path):
    # a CSV inside a venv/site-packages dir is ignored
    dep = tmp_path / "venv" / "site-packages"
    dep.mkdir(parents=True)
    (dep / "junk.csv").write_text("a,b\n1,2\n")
    # a too-deep CSV is ignored at max_depth=1
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "deep.csv").write_text("map,x\n0.5,1\n")
    _run(tmp_path, "run_ok", "epoch,map50\n0,0.5\n1,0.7\n")  # base/run_ok/results.csv = depth 2
    agg = evalcsv.aggregate_csvs(str(tmp_path), max_depth=2)
    names = {r["name"] for r in agg["runs"]}
    assert "run_ok" in names
    assert "junk" not in names and "deep" not in names  # dep-dir + depth-4 excluded
