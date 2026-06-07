"""Base repository: owner-scoped CRUD over a single SQLAlchemy model."""

from __future__ import annotations

from typing import Any

from src.auth_helpers import owner_filter


class BaseRepository:
    """Owner-scoped data access for ``model`` over a provided ``Session``.

    Subclasses set ``model`` (and optionally ``id_attr``). All reads/deletes are
    scoped via auth_helpers.owner_filter — owner == user OR null-owner shared
    rows, and a no-op in single-user mode (empty user). The repository does NOT
    commit; the caller's ``get_db_session()`` context owns the transaction.
    """

    model: type
    id_attr: str = "id"

    def __init__(self, db):
        self.db = db

    def _scoped(self, owner: str | None, *, include_shared: bool = True):
        q = self.db.query(self.model)
        if owner:
            q = owner_filter(q, self.model, owner, include_shared=include_shared)
        return q

    def get(self, id_value: str, owner: str | None = None, *, include_shared: bool = True):
        """Return the row by id within the owner scope, or None."""
        return self._scoped(owner, include_shared=include_shared).filter(
            getattr(self.model, self.id_attr) == id_value).first()

    def list(self, owner: str | None = None, *, include_shared: bool = True,
             order_by=None, limit: int | None = None, offset: int = 0, **filters) -> list:
        """List owner-scoped rows, optionally filtered/ordered/paginated."""
        q = self._scoped(owner, include_shared=include_shared)
        for field, value in filters.items():
            q = q.filter(getattr(self.model, field) == value)
        if order_by is not None:
            q = q.order_by(order_by)
        if offset:
            q = q.offset(offset)
        if limit is not None:
            q = q.limit(limit)
        return q.all()

    def count(self, owner: str | None = None, *, include_shared: bool = True, **filters) -> int:
        q = self._scoped(owner, include_shared=include_shared)
        for field, value in filters.items():
            q = q.filter(getattr(self.model, field) == value)
        return q.count()

    def exists(self, id_value: str, owner: str | None = None, *, include_shared: bool = True) -> bool:
        return self.get(id_value, owner, include_shared=include_shared) is not None

    def delete(self, id_value: str, owner: str | None = None, *, include_shared: bool = True) -> bool:
        """Delete the owner-scoped row by id. Returns True if a row was deleted.

        Owner-scoped so a user can't delete another user's row by guessing its id.
        Caller commits via the get_db_session context.
        """
        obj = self.get(id_value, owner, include_shared=include_shared)
        if obj is None:
            return False
        self.db.delete(obj)
        self.db.flush()
        return True

    def add(self, obj: Any):
        self.db.add(obj)
        self.db.flush()
        return obj
