"""Plugin registry: decorators register, load_plugins discovers, routes mount."""
import asyncio

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from src import plugin_registry as reg


def setup_function():
    reg._reset_for_tests()


def test_register_tool_captures_metadata():
    @reg.register_tool("t1", description="does t1", admin=True,
                       schema={"type": "object", "properties": {"x": {}}},
                       keywords=["foo"])
    def t1(args, ctx=None):
        return {"output": "ok", "exit_code": 0}

    assert reg.plugin_tool_names() == {"t1"}
    assert reg.plugin_admin_tools() == {"t1"}
    assert reg.plugin_tool_descriptions()["t1"] == "does t1"
    assert reg.plugin_tool_keywords()["t1"] == ["foo"]
    schema = reg.plugin_openai_schemas()[0]
    assert schema["function"]["name"] == "t1"
    assert "x" in schema["function"]["parameters"]["properties"]
    assert "t1" in reg.plugin_fenced_sections()


def test_register_tool_description_defaults_to_docstring():
    @reg.register_tool("t2")
    def t2(args, ctx=None):
        """first line doc.

        more.
        """
        return {}
    assert reg.plugin_tool_descriptions()["t2"] == "first line doc."


def _run(name, content, **ctx):
    return asyncio.run(reg.run_plugin_tool(name, content, **ctx))


def test_run_plugin_tool_json_args_and_ctx():
    seen = {}

    @reg.register_tool("echo")
    def echo(args, ctx=None):
        seen["args"] = args
        seen["ctx"] = ctx
        return {"output": args.get("msg", ""), "exit_code": 0}

    out = _run("echo", '{"msg": "hi"}', owner="alice")
    assert out == {"output": "hi", "exit_code": 0}
    assert seen["args"] == {"msg": "hi"}
    assert seen["ctx"]["owner"] == "alice"


def test_run_plugin_tool_async_and_non_dict_return():
    @reg.register_tool("a")
    async def a(args, ctx=None):
        return "plain string"

    out = _run("a", "{}")
    assert out == {"output": "plain string", "exit_code": 0}


def test_run_plugin_tool_arity_one_and_bad_json():
    @reg.register_tool("one")
    def one(args):  # no ctx param
        return {"got": args}

    # invalid JSON is passed as {"_raw": ...} rather than crashing
    out = _run("one", "not json")
    assert out["got"] == {"_raw": "not json"}


def test_run_plugin_tool_error_is_captured():
    @reg.register_tool("boom")
    def boom(args, ctx=None):
        raise RuntimeError("kaboom")

    out = _run("boom", "{}")
    assert out["exit_code"] == 1
    assert "kaboom" in out["error"]


def test_run_plugin_tool_unknown():
    out = _run("nope", "{}")
    assert out["exit_code"] == 1


def test_execute_tool_block_dispatches_to_plugin_and_gates_admin(monkeypatch):
    # Full dispatch path: a registered tool is callable via execute_tool_block,
    # and an admin-only one is blocked for a non-admin owner.
    from src.agent_tools import ToolBlock
    from src import tool_execution
    # Force non-admin so the gate is deterministic (single-user envs treat any
    # owner as admin otherwise).
    monkeypatch.setattr(tool_execution, "_owner_is_admin", lambda owner: False)

    @reg.register_tool("plug_ok")
    def plug_ok(args, ctx=None):
        return {"output": "ran:" + args.get("v", ""), "exit_code": 0}

    @reg.register_tool("plug_admin", admin=True)
    def plug_admin(args, ctx=None):
        return {"output": "secret", "exit_code": 0}

    _desc, result = asyncio.run(tool_execution.execute_tool_block(ToolBlock("plug_ok", '{"v": "x"}'), owner="bob"))
    assert result == {"output": "ran:x", "exit_code": 0}

    _desc, blocked = asyncio.run(tool_execution.execute_tool_block(ToolBlock("plug_admin", "{}"), owner="bob"))
    assert blocked["exit_code"] == 1
    assert "admin" in blocked["error"].lower()


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
