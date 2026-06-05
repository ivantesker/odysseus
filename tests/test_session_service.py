"""Session service — archive/unarchive/list logic tested without the web layer."""
import tempfile
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

import core.database as cdb
from core.database import Session as DbSession
from core.exceptions import SessionNotFoundError
from src.services import session_service

_TMPDB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_ENGINE = create_engine(
    f"sqlite:///{_TMPDB.name}",
    connect_args={"check_same_thread": False},
    poolclass=NullPool,
)
cdb.Base.metadata.create_all(_ENGINE)
_TS = sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False)


class _FakeManager:
    def __init__(self):
        self.sessions = {}
        self.loaded = []

    def _load_session_from_db(self, sid):
        self.loaded.append(sid)


@pytest.fixture(autouse=True)
def _patch_db(monkeypatch):
    monkeypatch.setattr(session_service, "SessionLocal", _TS)
    db = _TS()
    try:
        db.query(DbSession).delete()
        db.commit()
    finally:
        db.close()


def _seed(owner="alice", *, archived=False, model="qwen2.5", name="chat"):
    sid = str(uuid.uuid4())
    db = _TS()
    try:
        db.add(DbSession(id=sid, owner=owner, name=name, endpoint_url="http://localhost",
                         model=model, archived=archived))
        db.commit()
    finally:
        db.close()
    return sid


def _archived(sid):
    db = _TS()
    try:
        return db.query(DbSession).filter(DbSession.id == sid).first().archived
    finally:
        db.close()


def test_archive_sets_flag_and_syncs_memory():
    sid = _seed(archived=False)
    mgr = _FakeManager()
    mgr.sessions[sid] = type("S", (), {"archived": False})()
    assert session_service.archive_session(mgr, sid) == {"status": "archived"}
    assert _archived(sid) is True
    assert mgr.sessions[sid].archived is True


def test_archive_missing_raises_domain_error():
    with pytest.raises(SessionNotFoundError):
        session_service.archive_session(_FakeManager(), "nope")


def test_unarchive_clears_flag_and_loads_into_manager():
    sid = _seed(archived=True)
    mgr = _FakeManager()  # not in memory -> should trigger a load
    assert session_service.unarchive_session(mgr, sid) == {"status": "unarchived"}
    assert _archived(sid) is False
    assert mgr.loaded == [sid]


def test_unarchive_missing_raises_domain_error():
    with pytest.raises(SessionNotFoundError):
        session_service.unarchive_session(_FakeManager(), "nope")


def test_list_archived_is_owner_scoped_and_paginated():
    _seed("alice", archived=True, model="qwen2.5", name="a1")
    _seed("alice", archived=True, model="llama3.1", name="a2")
    _seed("bob", archived=True, model="qwen2.5", name="b1")
    _seed("alice", archived=False, model="qwen2.5", name="active")
    res = session_service.list_archived_sessions("alice")
    assert res["total"] == 2
    assert {s["name"] for s in res["sessions"]} == {"a1", "a2"}


def test_list_archived_model_filter_is_contains():
    _seed("alice", archived=True, model="openai/gpt-4")
    _seed("alice", archived=True, model="gpt-4o")
    _seed("alice", archived=True, model="claude-3")
    res = session_service.list_archived_sessions("alice", model="gpt-4")
    assert {s["model"] for s in res["sessions"]} == {"openai/gpt-4", "gpt-4o"}


class _DelManager:
    def __init__(self, found=True):
        self.sessions = {}
        self._found = found
        self.deleted = []

    def delete_session(self, sid):
        self.deleted.append(sid)
        return self._found


def test_purge_session_removes_db_row_and_calls_manager():
    sid = _seed()
    mgr = _DelManager(found=True)
    assert session_service.purge_session(mgr, sid) is True
    assert mgr.deleted == [sid]
    db = _TS()
    try:
        assert db.query(DbSession).filter(DbSession.id == sid).first() is None
    finally:
        db.close()


def test_session_is_important():
    sid = _seed()
    assert session_service.session_is_important(sid) is False
    db = _TS()
    try:
        db.query(DbSession).filter(DbSession.id == sid).first().is_important = True
        db.commit()
    finally:
        db.close()
    assert session_service.session_is_important(sid) is True


def test_delete_all_sessions_counts_and_clears():
    _seed("alice")
    _seed("bob")
    mgr = _DelManager()
    mgr.sessions = {"x": object()}
    assert session_service.delete_all_sessions(mgr) == 2
    assert mgr.sessions == {}
    db = _TS()
    try:
        assert db.query(DbSession).count() == 0
    finally:
        db.close()


def test_inject_messages_appends_and_counts():
    captured = []

    class _S:
        history = []
        def add_message(self, m):
            captured.append(m)

    class _Mgr:
        def __init__(self):
            self.saved = False
        def get_session(self, sid):
            return _S()
        def save_sessions(self):
            self.saved = True

    mgr = _Mgr()
    n = session_service.inject_messages(mgr, "s1", [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo", "metadata": {"x": 1}},
    ])
    assert n == 2 and mgr.saved and len(captured) == 2


def test_inject_messages_missing_session():
    class _Mgr:
        def get_session(self, sid):
            raise KeyError(sid)
    with pytest.raises(SessionNotFoundError):
        session_service.inject_messages(_Mgr(), "nope", [])


def _export_session():
    from types import SimpleNamespace as N
    return N(
        name="My Chat",
        model="qwen2.5",
        history=[
            N(role="user", content="hi"),
            N(role="assistant", content=[{"type": "text", "text": "yo"}]),
        ],
    )


@pytest.mark.parametrize("fmt,media", [
    ("md", "text/markdown"),
    ("txt", "text/plain"),
    ("json", "application/json"),
    ("html", "text/html"),
])
def test_render_export_formats(fmt, media):
    content, media_type, out = session_service.render_session_export(_export_session(), fmt, "")
    assert media_type == media
    assert out.endswith(f".{fmt}")
    assert "yo" in content  # multimodal content flattened, not dropped


def test_render_export_sanitizes_filename():
    # Path separators are stripped (no traversal); dots stay for extensions.
    _, _, out = session_service.render_session_export(_export_session(), "md", "../../etc/passwd")
    assert "/" not in out
    assert "\\" not in out


def test_flatten_content_shapes():
    assert session_service.flatten_content("plain") == "plain"
    assert session_service.flatten_content([{"type": "text", "text": "a"}, {"text": "b"}]) == "a\nb"
    assert session_service.flatten_content(None) == ""
