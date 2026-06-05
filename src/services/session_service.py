"""Session domain service — archive/unarchive/list logic without FastAPI.

Pure functions over the DB + the session manager. They raise domain errors
(SessionNotFoundError) rather than HTTPException; the route maps those to HTTP.
Ownership checks stay in the route (they read the request); everything else —
the DB mutation and the in-memory session-manager sync — lives here so it can be
unit-tested without spinning up the web layer.
"""

import logging
from datetime import UTC, datetime

from core.database import Session as DbSession
from core.database import SessionLocal
from core.exceptions import SessionNotFoundError

logger = logging.getLogger(__name__)


def _naive_utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def archive_session(session_manager, sid: str) -> dict:
    """Mark a session archived in the DB and sync the in-memory copy."""
    db = SessionLocal()
    try:
        db_session = db.query(DbSession).filter(DbSession.id == sid).first()
        if not db_session:
            raise SessionNotFoundError(sid)
        db_session.archived = True
        db_session.updated_at = _naive_utc_now()
        db.commit()
        if sid in session_manager.sessions:
            session_manager.sessions[sid].archived = True
        logger.info("Archived session %s", sid)
        return {"status": "archived"}
    except SessionNotFoundError:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def unarchive_session(session_manager, sid: str) -> dict:
    """Clear the archived flag and make the session active again."""
    db = SessionLocal()
    try:
        db_session = db.query(DbSession).filter(DbSession.id == sid).first()
        if not db_session:
            raise SessionNotFoundError(sid)
        db_session.archived = False
        db_session.updated_at = _naive_utc_now()
        db.commit()
        # Reload into the manager so it reappears in the active list.
        try:
            if sid in session_manager.sessions:
                session_manager.sessions[sid].archived = False
            else:
                session_manager._load_session_from_db(sid)
        except Exception:
            pass  # non-fatal — it will load on next access
        return {"status": "unarchived"}
    except SessionNotFoundError:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


import html as _html
import re as _re
from datetime import datetime as _datetime


def sanitize_export_filename(name: str) -> str:
    """Return a conservative filename safe for Content-Disposition."""
    name = name if isinstance(name, str) else ""
    name = _re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name[:128]


def flatten_content(content) -> str:
    """Flatten a message's content to plain text for text-based exports.

    Handles the three stored shapes: a plain string, a multimodal block list,
    or None (a tool-only assistant turn). Returns "" for anything without text.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("text")
        )
    return ""


def render_session_export(session, fmt: str = "md", filename: str = "") -> tuple[str, str, str]:
    """Render a session's history to (content, media_type, out_name).

    Pure: takes the session object, returns bytes-as-str + the response
    metadata. The route wraps the result in a FastAPI Response.
    """
    safe_name = _re.sub(r"[^\w\-_]", "_", session.name or "")
    timestamp = _datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = sanitize_export_filename(filename)

    if fmt == "json":
        import json as _json
        data = {
            "name": session.name,
            "model": session.model,
            "exported": _datetime.now().isoformat(),
            "messages": [{"role": m.role, "content": m.content} for m in session.history],
        }
        out = filename or f"conversation_{safe_name}_{timestamp}.json"
        return _json.dumps(data, indent=2, ensure_ascii=False), "application/json", out

    if fmt == "txt":
        lines = []
        for m in session.history:
            lines.append(f"[{m.role.upper()}]")
            lines.append(flatten_content(m.content))
            lines.append("")
        out = filename or f"conversation_{safe_name}_{timestamp}.txt"
        return "\n".join(lines), "text/plain", out

    if fmt == "html":
        safe_title = _html.escape(session.name or "")
        parts = [
            "<!DOCTYPE html><html><head>",
            f"<meta charset='utf-8'><title>{safe_title}</title>",
            "<style>body{font-family:monospace;max-width:800px;margin:2rem auto;padding:0 1rem;background:#111;color:#ddd}",
            ".msg{margin:1rem 0;padding:0.8rem;border-radius:6px;border:1px solid #333}",
            ".user{background:#1a1a2e}.ai{background:#1a2e1a}",
            ".role{font-weight:bold;margin-bottom:0.4rem;opacity:0.7;text-transform:uppercase;font-size:0.85em}",
            "pre{background:#000;padding:0.5rem;border-radius:4px;overflow-x:auto}</style></head><body>",
            f"<h1>{safe_title}</h1>",
        ]
        for m in session.history:
            cls = "user" if m.role == "user" else "ai"
            content = flatten_content(m.content).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            content = content.replace("\n", "<br>")
            parts.append(f'<div class="msg {cls}"><div class="role">{m.role}</div>{content}</div>')
        parts.append("</body></html>")
        out = filename or f"conversation_{safe_name}_{timestamp}.html"
        return "\n".join(parts), "text/html", out

    # Default: markdown
    md = [
        f"# Conversation: {session.name}",
        f"*Exported on: {_datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        f"*Model: {session.model}*",
        "\n---\n",
    ]
    for message in session.history:
        md.append(f"### {message.role.upper()}")
        md.append(f"{flatten_content(message.content)}\n")
        md.append("---\n")
    if len(md) > 3:
        md.pop()
    out = filename or f"conversation_{safe_name}_{timestamp}.md"
    return "\n".join(md), "text/markdown", out


def purge_session(session_manager, sid: str) -> bool:
    """Delete a session everywhere: manager + its DB rows. Returns found-ness.

    Used by both single and bulk delete; ownership is checked by the caller.
    """
    from core.database import ChatMessage as _CM

    found = bool(session_manager.delete_session(sid))
    db = SessionLocal()
    try:
        db.query(_CM).filter(_CM.session_id == sid).delete()
        db.query(DbSession).filter(DbSession.id == sid).delete()
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()
    return found


def session_is_important(sid: str) -> bool:
    """True if the session is starred/important (delete is then blocked)."""
    db = SessionLocal()
    try:
        row = db.query(DbSession).filter(DbSession.id == sid).first()
        return bool(row and row.is_important)
    finally:
        db.close()


def delete_all_sessions(session_manager) -> int:
    """Delete every session + all chat messages. Returns the count removed."""
    from core.database import ChatMessage as _CM

    db = SessionLocal()
    try:
        count = db.query(DbSession).count()
        db.query(_CM).delete()
        db.query(DbSession).delete()
        db.commit()
        session_manager.sessions.clear()
        logger.info("Deleted all %d sessions", count)
        return count
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def inject_messages(session_manager, sid: str, messages: list[dict]) -> int:
    """Append messages to a session's history. Returns how many were added."""
    from core.models import ChatMessage

    try:
        sess = session_manager.get_session(sid)
    except KeyError:
        raise SessionNotFoundError(sid) from None
    for m in messages:
        sess.add_message(ChatMessage(m["role"], m["content"], metadata=m.get("metadata")))
    session_manager.save_sessions()
    return len(messages)


_ARCHIVED_SORT_KEYS = ("recent", "oldest", "most-messages", "alpha")


def list_archived_sessions(
    owner: str,
    search: str = "",
    offset: int = 0,
    limit: int = 20,
    sort: str = "recent",
    model: str = "",
) -> dict:
    """Owner-scoped, paginated archived-session listing for the browser."""
    db = SessionLocal()
    try:
        q = db.query(DbSession).filter(DbSession.archived == True)
        q = q.filter(DbSession.owner == owner)
        if search:
            safe = search.replace("%", r"\%").replace("_", r"\_")
            q = q.filter(DbSession.name.ilike(f"%{safe}%", escape="\\"))
        if model:
            safe_model = model.replace("%", r"\%").replace("_", r"\_")
            q = q.filter(DbSession.model.ilike(f"%{safe_model}%", escape="\\"))
        total = q.count()
        sort_map = {
            "recent": DbSession.updated_at.desc(),
            "oldest": DbSession.updated_at.asc(),
            "most-messages": DbSession.message_count.desc().nulls_last(),
            "alpha": DbSession.name.asc(),
        }
        order = sort_map.get(sort, DbSession.updated_at.desc())
        rows = q.order_by(order).offset(offset).limit(limit).all()
        sessions = [
            {
                "id": s.id,
                "name": s.name,
                "model": s.model,
                "message_count": s.message_count or 0,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "updated_at": s.updated_at.isoformat() if s.updated_at else None,
                "is_important": s.is_important,
            }
            for s in rows
        ]
        return {"sessions": sessions, "total": total}
    finally:
        db.close()
