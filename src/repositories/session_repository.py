"""Owner-scoped access to chat Sessions."""

from __future__ import annotations

from core.database import Session

from .base import BaseRepository


class SessionRepository(BaseRepository):
    model = Session

    def list_active(self, owner: str | None = None, **kw) -> list:
        """Non-archived sessions for the owner."""
        return self.list(owner, archived=False, **kw)

    def list_archived(self, owner: str | None = None, **kw) -> list:
        return self.list(owner, archived=True, **kw)
