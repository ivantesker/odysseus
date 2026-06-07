"""Owner-scoped access to Documents."""

from __future__ import annotations

from core.database import Document

from .base import BaseRepository


class DocumentRepository(BaseRepository):
    model = Document
