"""Data-access layer.

Thin, owner-scoped repositories over the SQLAlchemy models. They centralize the
``owner == user OR owner IS NULL`` scoping (today duplicated across ~40 route
files) behind one tested seam, so owner-isolation correctness lives in one place
and the query patterns are mockable in tests.

Each repository wraps a caller-provided ``Session`` (use ``get_db_session()`` for
the unit-of-work boundary):

    from core.database import get_db_session
    from src.repositories import SessionRepository

    with get_db_session() as db:
        s = SessionRepository(db).get("abc123", owner="alice")
"""

from .base import BaseRepository
from .document_repository import DocumentRepository
from .session_repository import SessionRepository
from .task_repository import ScheduledTaskRepository

__all__ = [
    "BaseRepository",
    "SessionRepository",
    "DocumentRepository",
    "ScheduledTaskRepository",
]
