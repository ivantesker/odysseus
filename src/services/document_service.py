"""Document domain operations, owner-scoped, built on the repository layer.

The A2 seam: domain logic that today lives inline in routes/document_routes.py
(1687 lines) moves here so it's reusable from agents/scheduler and testable
without the HTTP layer. Reads/writes go through DocumentRepository so owner
isolation is enforced in one place. Routes adopt these incrementally.
"""

from __future__ import annotations

from core.database import get_db_session
from src.repositories import DocumentRepository


def _to_dict(doc) -> dict:
    return {
        "id": doc.id,
        "title": getattr(doc, "title", None),
        "owner": getattr(doc, "owner", None),
        "content": getattr(doc, "current_content", None),
    }


def get_document(doc_id: str, owner: str | None) -> dict | None:
    """Return the owner-scoped document as a plain dict, or None."""
    with get_db_session() as db:
        doc = DocumentRepository(db).get(doc_id, owner)
        return _to_dict(doc) if doc else None


def list_documents(owner: str | None, *, limit: int | None = None, offset: int = 0) -> list[dict]:
    with get_db_session() as db:
        return [_to_dict(d) for d in DocumentRepository(db).list(owner, limit=limit, offset=offset)]


def count_documents(owner: str | None) -> int:
    with get_db_session() as db:
        return DocumentRepository(db).count(owner)


def delete_document(doc_id: str, owner: str | None) -> bool:
    """Delete one owner-scoped document. True if a row was deleted.

    Owner-scoped so a prompt-injected or cross-user request can't delete another
    owner's document by guessing its id.
    """
    with get_db_session() as db:
        return DocumentRepository(db).delete(doc_id, owner)
