"""Lightweight plugin registries — add a feature by dropping in a file.

Three decorators let a new module self-register without editing any central
wiring (app.py, the 4k-line tool module, the action dispatcher):

    from src.plugin_registry import register_tool, register_action, register_route

    @register_tool("greet", schema={...})
    def greet(args): ...

    @register_action("send_ping")
    def send_ping(ctx): ...

    @register_route          # decorate a function returning an APIRouter
    def my_routes():
        r = APIRouter(); ...; return r

`load_plugins("plugins")` imports every submodule of the given package so the
decorators run, then the app pulls what was registered:

    load_plugins()
    for router in registered_routers():
        app.include_router(router)

This is intentionally minimal and additive — existing tools/actions/routes keep
working untouched; new ones can opt into the registry. Re-registering the same
name overwrites (last definition wins) so a plugin can shadow a default.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Any, Dict, List
from collections.abc import Callable

logger = logging.getLogger(__name__)

_TOOLS: dict[str, dict[str, Any]] = {}
_ACTIONS: dict[str, Callable] = {}
_ROUTERS: list[Any] = []


def register_tool(
    name: str,
    *,
    description: str = "",
    schema: dict[str, Any] | None = None,
    fenced_help: str | None = None,
    admin: bool = False,
    keywords: list[str] | None = None,
) -> Callable:
    """Register an agent tool so it behaves like a built-in.

    name        fence tag / function name the model calls (```name\\n{json}```)
    description one-liner shown to the model + used for RAG selection
    schema      JSON-schema of the args object (native tool-calling models)
    fenced_help full prompt section for fenced-block models; defaults to a
                generic JSON-args block built from `description`
    admin       restrict to admin/single-user owners (shell-y tools should)
    keywords    extra words that should surface this tool in selection

    The wired functions (`wire_plugin_tools`) make the tool visible to the
    parser, the prompt, tool selection, and dispatch.
    """
    def deco(fn: Callable) -> Callable:
        _TOOLS[name] = {
            "fn": fn,
            "description": description or (fn.__doc__ or "").strip().split("\n", 1)[0],
            "schema": schema or {"type": "object", "properties": {}},
            "fenced_help": fenced_help,
            "admin": bool(admin),
            "keywords": list(keywords or []),
        }
        return fn
    return deco


def register_action(name: str) -> Callable:
    """Register a named action (e.g. a scheduled-task or webhook handler)."""
    def deco(fn: Callable) -> Callable:
        _ACTIONS[name] = fn
        return fn
    return deco


def register_route(target):
    """Register an APIRouter, or a zero-arg factory that returns one.

    Usable as `@register_route` on a factory function, or called directly
    `register_route(router)` with an already-built APIRouter. Note APIRouter is
    itself callable (an ASGI app), so dispatch on type, not callability.
    """
    from fastapi import APIRouter

    if isinstance(target, APIRouter):
        _ROUTERS.append(target)
        return target
    router = target()  # a zero-arg factory returning an APIRouter
    _ROUTERS.append(router)
    return target


def registered_tools() -> dict[str, dict[str, Any]]:
    return dict(_TOOLS)


def registered_actions() -> dict[str, Callable]:
    return dict(_ACTIONS)


def registered_routers() -> list[Any]:
    return list(_ROUTERS)


# ── Agent-tool bridge ─────────────────────────────────────────────────────────
# Accessors the wiring uses to make registered tools first-class agent tools.

def plugin_tool_names() -> set[str]:
    return set(_TOOLS.keys())


def plugin_admin_tools() -> set[str]:
    return {n for n, t in _TOOLS.items() if t.get("admin")}


def plugin_tool_descriptions() -> dict[str, str]:
    return {n: (t.get("description") or n) for n, t in _TOOLS.items()}


def plugin_tool_keywords() -> dict[str, list[str]]:
    return {n: list(t.get("keywords") or []) for n, t in _TOOLS.items()}


def plugin_openai_schemas() -> list[dict]:
    """OpenAI-style function schemas for native tool-calling models."""
    out = []
    for name, t in _TOOLS.items():
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": t.get("description") or name,
                "parameters": t.get("schema") or {"type": "object", "properties": {}},
            },
        })
    return out


def plugin_fenced_sections() -> dict[str, str]:
    """Per-tool prompt sections for fenced-block (e.g. Ollama) models."""
    out = {}
    for name, t in _TOOLS.items():
        help_text = t.get("fenced_help")
        if not help_text:
            help_text = (
                f"{t.get('description') or name}\n"
                f"```{name}\n{{ ...JSON args... }}\n```"
            )
        out[name] = help_text
    return out


async def run_plugin_tool(name: str, content, **ctx) -> dict:
    """Execute a registered tool. Parses JSON args from `content`, calls the fn
    (sync or async; with an optional ctx kwarg), and normalizes the result to a
    dict so the agent's result formatter can render it."""
    import inspect
    import json

    entry = _TOOLS.get(name)
    if not entry:
        return {"error": f"Unknown plugin tool: {name}", "exit_code": 1}
    fn = entry["fn"]

    args: Any
    if isinstance(content, (dict, list)):
        args = content
    else:
        text = (content or "").strip()
        if not text:
            args = {}
        else:
            try:
                args = json.loads(text)
            except (ValueError, TypeError):
                args = {"_raw": text}

    try:
        params = inspect.signature(fn).parameters
        pass_ctx = len(params) >= 2 or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        result = fn(args, ctx) if pass_ctx else fn(args)
        if inspect.isawaitable(result):
            result = await result
    except Exception as e:
        logger.exception("Plugin tool %s failed", name)
        return {"error": f"{name} failed: {e}", "exit_code": 1}

    if isinstance(result, dict):
        return result
    return {"output": "" if result is None else str(result), "exit_code": 0}


def wire_plugin_tools() -> int:
    """Make all registered plugin tools behave like built-ins.

    Mutates the shared tool structures (parser tags + regex, tool-index
    descriptions + always-available set, fenced prompt sections, native
    schemas) so the model is told about each tool, selection surfaces it, the
    fenced/XML/native parsers recognize it, and dispatch can route it. Returns
    the number of tools wired. Idempotent.
    """
    names = plugin_tool_names()
    if not names:
        return 0
    descriptions = plugin_tool_descriptions()

    # 1) Parser: register fence tags and rebuild the block regex.
    try:
        from src import tool_parsing
        tool_parsing.register_extra_tool_tags(names)
    except Exception as e:
        logger.warning("plugin wiring: parser tags failed: %s", e)

    # 2) Tool index: descriptions (RAG) + always-available so they're offered.
    try:
        from src import tool_index
        tool_index.BUILTIN_TOOL_DESCRIPTIONS.update(descriptions)
        tool_index.ALWAYS_AVAILABLE = frozenset(set(tool_index.ALWAYS_AVAILABLE) | names)
    except Exception as e:
        logger.warning("plugin wiring: tool index failed: %s", e)

    # 3) Fenced prompt sections (Ollama / text models learn the call format).
    try:
        from src import agent_loop
        agent_loop.TOOL_SECTIONS.update(plugin_fenced_sections())
    except Exception as e:
        logger.warning("plugin wiring: prompt sections failed: %s", e)

    # 4) Native function schemas (OpenAI-style tool-calling models).
    try:
        from src import tool_schemas
        existing = {s.get("function", {}).get("name") for s in tool_schemas.FUNCTION_TOOL_SCHEMAS}
        for s in plugin_openai_schemas():
            if s["function"]["name"] not in existing:
                tool_schemas.FUNCTION_TOOL_SCHEMAS.append(s)
    except Exception as e:
        logger.warning("plugin wiring: schemas failed: %s", e)

    logger.info("Wired %d plugin tool(s): %s", len(names), ", ".join(sorted(names)))
    return len(names)


def load_plugins(package: str = "plugins") -> int:
    """Import every submodule of `package` so its decorators run.

    Returns the number of modules imported. A missing package is not an error
    (a deployment may ship no plugins); an individual plugin that raises on
    import is logged and skipped so one bad file can't take down startup.
    """
    try:
        pkg = importlib.import_module(package)
    except ModuleNotFoundError:
        return 0
    count = 0
    loaded = []
    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{package}.{mod.name}")
            count += 1
            loaded.append(mod.name)
            logger.debug("Loaded plugin %s.%s", package, mod.name)
        except Exception as e:
            logger.warning("Skipping plugin %s.%s: %s", package, mod.name, e)
    if loaded:
        logger.info(
            "Plugins loaded from %s: %s (tools=%d, actions=%d, routes=%d)",
            package, ", ".join(loaded), len(_TOOLS), len(_ACTIONS), len(_ROUTERS),
        )
    return count


def _reset_for_tests() -> None:
    _TOOLS.clear()
    _ACTIONS.clear()
    _ROUTERS.clear()
