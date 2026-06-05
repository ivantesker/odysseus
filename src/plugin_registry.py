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
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)

_TOOLS: Dict[str, Dict[str, Any]] = {}
_ACTIONS: Dict[str, Callable] = {}
_ROUTERS: List[Any] = []


def register_tool(name: str, *, schema: Dict[str, Any] | None = None) -> Callable:
    """Register an agent tool. `schema` is an optional JSON-schema for args."""
    def deco(fn: Callable) -> Callable:
        _TOOLS[name] = {"fn": fn, "schema": schema or {}}
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


def registered_tools() -> Dict[str, Dict[str, Any]]:
    return dict(_TOOLS)


def registered_actions() -> Dict[str, Callable]:
    return dict(_ACTIONS)


def registered_routers() -> List[Any]:
    return list(_ROUTERS)


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
