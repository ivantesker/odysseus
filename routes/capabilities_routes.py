"""Capabilities endpoint — what this instance offers, so any UI can adapt.

`GET /api/v1/capabilities` lets a client (bundled web UI, mobile companion, TUI,
third-party) discover at runtime which features are enabled and which subsystems
are configured, instead of hard-coding assumptions. Pairs with `/api/health`
(liveness) and `/api/ready` (readiness): this one answers "what can I show?".

The route is intentionally thin — all logic lives in
`src.services.capabilities_service` (pure, no FastAPI), so it is unit-testable
without the web layer.
"""

from typing import Any, Dict

from fastapi import APIRouter

from src.services.capabilities_service import build_capabilities

router = APIRouter()


def setup_capabilities_routes(model_discovery=None):
    @router.get("/api/v1/capabilities")
    def capabilities() -> dict[str, Any]:
        return build_capabilities()

    return router
