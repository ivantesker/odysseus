"""Capabilities service — builds the instance-capabilities report.

Pure logic, no FastAPI. The route in routes/capabilities_routes.py is a thin
wrapper that just returns build_capabilities(...).
"""

import os
from typing import Any, Dict


def auth_enabled() -> bool:
    return os.getenv("AUTH_ENABLED", "true").strip().lower() != "false"


def web_ui_enabled() -> bool:
    return os.getenv("SERVE_WEB_UI", "true").strip().lower() not in ("0", "false", "no", "off")


def _features() -> Dict[str, Any]:
    try:
        from src.settings import load_features
        return load_features() or {}
    except Exception:
        return {}


def _count_table(model_cls) -> int:
    try:
        from core.database import SessionLocal
        db = SessionLocal()
        try:
            return db.query(model_cls).count()
        finally:
            db.close()
    except Exception:
        return 0


def _model_subsystem() -> Dict[str, Any]:
    """Configured model endpoints + cached model count (no live probe)."""
    info: Dict[str, Any] = {"endpoints": 0, "models": 0}
    try:
        from core.database import SessionLocal, ModelEndpoint
        db = SessionLocal()
        try:
            rows = db.query(ModelEndpoint).filter(ModelEndpoint.is_enabled == True).all()
            info["endpoints"] = len(rows)
            total = 0
            for ep in rows:
                cached = getattr(ep, "cached_models", None)
                if isinstance(cached, list):
                    total += len(cached)
            info["models"] = total
        finally:
            db.close()
    except Exception:
        pass
    return info


def _rag_available() -> bool:
    """RAG works when both the vector-store client and local embeddings import."""
    try:
        import chromadb  # noqa: F401
        import fastembed  # noqa: F401
        return True
    except Exception:
        return False


def build_capabilities() -> Dict[str, Any]:
    """Assemble the capabilities report from config + cheap DB counts."""
    from core.constants import APP_VERSION

    try:
        from core.database import EmailAccount
        email_accounts = _count_table(EmailAccount)
    except Exception:
        email_accounts = 0
    try:
        from core.database import Integration
        integrations = _count_table(Integration)
    except Exception:
        integrations = 0

    searxng = os.getenv("SEARXNG_INSTANCE", "").strip()

    return {
        "version": APP_VERSION,
        "auth_enabled": auth_enabled(),
        "web_ui": web_ui_enabled(),
        "features": _features(),
        "subsystems": {
            "models": _model_subsystem(),
            "rag": {"available": _rag_available()},
            "web_search": {"configured": bool(searxng), "url": searxng or None},
            "email": {"accounts": email_accounts},
            "integrations": {"configured": integrations},
        },
    }
