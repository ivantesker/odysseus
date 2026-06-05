"""S4 pattern: service is callable without the web layer; the auth dependency
decouples handlers from request.state."""
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from src.dependencies import RequiredUser
from src.services.capabilities_service import build_capabilities


def test_capabilities_service_runs_without_web_layer():
    data = build_capabilities()
    assert data["version"]
    assert set(data["subsystems"]) == {"models", "rag", "web_search", "email", "integrations"}


def test_required_user_dependency_injects_without_request_state(monkeypatch):
    # The handler never mentions Request/request.state — the dependency supplies
    # the user. Stub require_user so the test doesn't need real auth middleware.
    import src.dependencies as deps
    monkeypatch.setattr(deps, "require_user", lambda request: "alice")

    app = FastAPI()
    router = APIRouter()

    @router.get("/whoami")
    def whoami(user: str = RequiredUser):
        return {"user": user}

    app.include_router(router)
    assert TestClient(app).get("/whoami").json() == {"user": "alice"}
