"""Route observability helpers (Q4): make swallowed/unhandled route errors
visible. A silent ``except Exception: return False`` hides real bugs — these
helpers log with a stack trace while preserving behavior.
"""

from __future__ import annotations

import functools
import logging

logger = logging.getLogger("odysseus.routes")


def log_route_errors(fn):
    """Decorator: log any non-HTTP exception from a route handler (with stack
    trace) and re-raise. HTTPException passes straight through — those are the
    handler's intended responses, not bugs.

    Signature-preserving (functools.wraps) so FastAPI's parameter introspection
    still works. Supports async and sync handlers.
    """
    import inspect

    from fastapi import HTTPException

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def _aw(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except HTTPException:
                raise
            except Exception:
                logger.exception("Unhandled error in route %s", getattr(fn, "__name__", "?"))
                raise
        return _aw

    @functools.wraps(fn)
    def _w(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except HTTPException:
            raise
        except Exception:
            logger.exception("Unhandled error in route %s", getattr(fn, "__name__", "?"))
            raise
    return _w


def logged_rollback(db, log: logging.Logger, context: str, exc: BaseException | None = None):
    """Roll back a DB session and LOG why — replaces silent
    ``except Exception: db.rollback(); return False`` swallows so the failure is
    visible in logs instead of vanishing."""
    try:
        db.rollback()
    except Exception:
        log.exception("rollback failed in %s", context)
    log.warning("%s failed, rolled back", context, exc_info=exc if exc else True)
