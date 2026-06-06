"""Example plugin: adds GET /api/v1/plugins/ping with no central wiring.

Demonstrates the "feature = one new file" pattern. Delete or copy as a template.
"""

from fastapi import APIRouter

from src.plugin_registry import register_route


@register_route
def ping_router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/v1/plugins/ping")
    def ping():
        return {"ok": True, "plugin": "example_ping"}

    return router
