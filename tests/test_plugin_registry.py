"""Plugin registry: decorators register, load_plugins discovers, routes mount."""
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from src import plugin_registry as reg


def setup_function():
    reg._reset_for_tests()


def test_register_tool_and_action():
    @reg.register_tool("greet", schema={"type": "object"})
    def greet(args):
        return "hi"

    @reg.register_action("ping")
    def ping(ctx):
        return "pong"

    assert "greet" in reg.registered_tools()
    assert reg.registered_tools()["greet"]["schema"] == {"type": "object"}
    assert reg.registered_actions()["ping"]({}) == "pong"


def test_register_route_factory_and_instance():
    @reg.register_route
    def factory():
        r = APIRouter()

        @r.get("/api/v1/plugins/_t")
        def _t():
            return {"ok": True}

        return r

    direct = APIRouter()
    reg.register_route(direct)

    assert len(reg.registered_routers()) == 2


def test_load_plugins_discovers_example_and_route_works():
    n = reg.load_plugins("plugins")
    assert n >= 1
    app = FastAPI()
    for router in reg.registered_routers():
        app.include_router(router)
    resp = TestClient(app).get("/api/v1/plugins/ping")
    assert resp.status_code == 200
    assert resp.json()["plugin"] == "example_ping"


def test_load_plugins_missing_package_is_no_error():
    assert reg.load_plugins("nonexistent_pkg_xyz") == 0
