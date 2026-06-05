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
