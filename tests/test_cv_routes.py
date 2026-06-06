"""CV panel API routes — thin admin wrappers over the CV services."""
import sys
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _stub_multipart(monkeypatch):
    try:
        import python_multipart  # noqa: F401
        return
    except ImportError:
        pass
    stub = types.ModuleType("python_multipart")
    stub.__version__ = "0.0.20"
    monkeypatch.setitem(sys.modules, "python_multipart", stub)


@pytest.fixture
def client(monkeypatch):
    _stub_multipart(monkeypatch)
    import routes.cv_routes as cv_routes
    # Bypass the admin gate for the test.
    monkeypatch.setattr(cv_routes, "require_admin", lambda request: None)
    app = FastAPI()
    app.include_router(cv_routes.setup_cv_routes())
    return TestClient(app)


def test_dataset_lint_route(client, tmp_path):
    lbl = tmp_path / "labels"
    lbl.mkdir()
    (lbl / "a.txt").write_text("0 .5 .5 .2 .2\n")
    (lbl / "b.txt").write_text("9 .5 .5 .2 .2\n")  # class out of range
    r = client.post("/api/cv/dataset/lint", json={"labels_dir": str(lbl), "num_classes": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["issue_count"] == 1 and body["ok"] is False


def test_dataset_lint_requires_labels_dir(client):
    r = client.post("/api/cv/dataset/lint", json={})
    assert r.status_code == 400


def test_dataset_stats_route(client, tmp_path):
    img = tmp_path / "images"; lbl = tmp_path / "labels"
    img.mkdir(); lbl.mkdir()
    (img / "a.jpg").write_bytes(b"x")
    (lbl / "a.txt").write_text("0 .5 .5 .2 .2\n0 .4 .4 .1 .1\n")
    r = client.post("/api/cv/dataset/stats", json={
        "images_dir": str(img), "labels_dir": str(lbl), "class_names": ["Front"]})
    assert r.status_code == 200
    assert r.json()["per_class"] == {"Front": 2}


def test_dataset_split_route(client, tmp_path):
    img = tmp_path / "images"; lbl = tmp_path / "labels"; out = tmp_path / "out"
    img.mkdir(); lbl.mkdir()
    for i in range(6):
        (img / f"i{i}.jpg").write_bytes(b"x")
        (lbl / f"i{i}.txt").write_text("0 .5 .5 .2 .2\n")
    r = client.post("/api/cv/dataset/split", json={
        "images_dir": str(img), "labels_dir": str(lbl), "out_dir": str(out), "val_frac": 0.5})
    assert r.status_code == 200
    body = r.json()
    assert body["train"] + body["val"] == 6


def test_eda_route_generates_report_and_serves_it(client, tmp_path):
    lbl = tmp_path / "labels"
    lbl.mkdir()
    (lbl / "a.txt").write_text("0 .5 .5 .2 .2\n1 .3 .3 .1 .1\n")
    (lbl / "b.txt").write_text("0 .4 .4 .2 .2\n")
    # generate
    r = client.post("/api/cv/eda", json={"labels_dir": str(lbl),
                                          "class_names": ["Front", "Back"], "title": "Demo"})
    assert r.status_code == 200
    body = r.json()
    assert body["report_id"].startswith("cv-")
    assert body["report_url"] == f"/api/cv/report/{body['report_id']}"
    assert body["summary"]["boxes"] == 3
    # serve the generated report HTML
    page = client.get(body["report_url"])
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "Demo" in page.text and "<svg" in page.text


def test_eda_route_requires_labels_dir(client):
    assert client.post("/api/cv/eda", json={}).status_code == 400


def test_report_route_404_for_unknown(client):
    assert client.get("/api/cv/report/cv-000000000000").status_code == 404


def test_review_route_generates_report(client, tmp_path):
    gt = tmp_path / "gt"; pa = tmp_path / "pa"
    gt.mkdir(); pa.mkdir()
    for i in range(4):
        (gt / f"{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n")
        (pa / f"{i}.txt").write_text("0 0.5 0.5 0.2 0.2 0.9\n1 0.2 0.2 0.1 0.1 0.8\n")
    r = client.post("/api/cv/review", json={
        "labels_dir": str(gt), "preds_dir": str(pa), "class_names": ["car", "truck"]})
    assert r.status_code == 200
    body = r.json()
    assert body["report_id"].startswith("cv-")
    page = client.get(body["report_url"])
    assert page.status_code == 200
    assert "Confusion matrix" in page.text


def test_review_route_requires_dirs(client):
    assert client.post("/api/cv/review", json={"labels_dir": "x"}).status_code == 400


def test_inspect_route(client, tmp_path):
    (tmp_path / "images").mkdir(); (tmp_path / "labels").mkdir()
    (tmp_path / "data.yaml").write_text("nc: 2\nnames: ['a', 'b']\n")
    r = client.post("/api/cv/inspect", json={"root": str(tmp_path)})
    assert r.status_code == 200
    assert r.json()["class_names"] == ["a", "b"]


def test_inspect_route_requires_root(client):
    assert client.post("/api/cv/inspect", json={}).status_code == 400


def test_reports_list_route(client, tmp_path):
    lbl = tmp_path / "labels"; lbl.mkdir()
    (lbl / "a.txt").write_text("0 .5 .5 .2 .2\n")
    gen = client.post("/api/cv/eda", json={"labels_dir": str(lbl), "title": "ListMe"})
    rid = gen.json()["report_id"]
    lst = client.get("/api/cv/reports")
    assert lst.status_code == 200
    assert any(r["id"] == rid for r in lst.json()["reports"])


def test_eval_route_graceful_without_stack(client):
    # No ultralytics in the app env → 200 with an error payload, not a 500.
    r = client.post("/api/cv/eval", json={"model": "x.pt", "data": "d.yaml"})
    assert r.status_code == 200
    assert r.json()["exit_code"] == 1


def test_convert_route_missing_source(client):
    r = client.post("/api/cv/convert", json={"source": "/no/such.pt"})
    assert r.status_code == 200
    assert r.json()["exit_code"] == 1
