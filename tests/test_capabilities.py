"""Capabilities endpoint + headless toggle contract."""
import importlib

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _client():
    mod = importlib.import_module("routes.capabilities_routes")
    app = FastAPI()
    app.include_router(mod.setup_capabilities_routes(None))
    return TestClient(app)


def test_capabilities_shape():
    resp = _client().get("/api/v1/capabilities")
    assert resp.status_code == 200
    data = resp.json()
    for key in ("version", "auth_enabled", "web_ui", "features", "subsystems"):
        assert key in data
    subs = data["subsystems"]
    for key in ("models", "rag", "web_search", "email", "integrations"):
        assert key in subs
    assert isinstance(data["features"], dict)


def test_auth_enabled_reflects_env(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "false")
    assert _client().get("/api/v1/capabilities").json()["auth_enabled"] is False
    monkeypatch.setenv("AUTH_ENABLED", "true")
    assert _client().get("/api/v1/capabilities").json()["auth_enabled"] is True


def test_web_ui_flag_reflects_env(monkeypatch):
    monkeypatch.setenv("SERVE_WEB_UI", "false")
    assert _client().get("/api/v1/capabilities").json()["web_ui"] is False
    monkeypatch.setenv("SERVE_WEB_UI", "true")
    assert _client().get("/api/v1/capabilities").json()["web_ui"] is True
