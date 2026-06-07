"""Owner-scoped repository layer (src/repositories)."""
import tempfile
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

import core.database as cdb
from core.database import Document, ScheduledTask
from core.database import Session as DbSession
from src.repositories import (
    DocumentRepository,
    ScheduledTaskRepository,
    SessionRepository,
)

_TMPDB = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_ENGINE = create_engine(f"sqlite:///{_TMPDB.name}",
                        connect_args={"check_same_thread": False}, poolclass=NullPool)
cdb.Base.metadata.create_all(_ENGINE)
_TS = sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False)


def _mk_session(db, owner, name="chat"):
    s = DbSession(id=str(uuid.uuid4()), owner=owner, name=name,
                  endpoint_url="http://localhost", model="m", archived=False)
    db.add(s); db.commit()
    return s.id


@pytest.fixture
def db():
    s = _TS()
    s.query(DbSession).delete(); s.query(Document).delete(); s.query(ScheduledTask).delete()
    s.commit()
    yield s
    s.close()


def test_get_is_owner_scoped(db):
    sid = _mk_session(db, "alice")
    repo = SessionRepository(db)
    assert repo.get(sid, owner="alice").id == sid
    assert repo.get(sid, owner="bob") is None          # not bob's
    assert repo.get(sid, owner=None).id == sid          # single-user: no scoping


def test_shared_null_owner_visible(db):
    sid = _mk_session(db, None)  # null owner = shared
    repo = SessionRepository(db)
    assert repo.get(sid, owner="alice").id == sid                      # shared → visible
    assert repo.get(sid, owner="alice", include_shared=False) is None  # strict → hidden


def test_list_and_count_scoped(db):
    _mk_session(db, "alice"); _mk_session(db, "alice"); _mk_session(db, "bob")
    repo = SessionRepository(db)
    assert repo.count(owner="alice") == 2
    assert repo.count(owner="bob") == 1
    assert len(repo.list(owner="alice")) == 2


def test_list_active_vs_archived(db):
    _mk_session(db, "alice", name="live")
    arch = DbSession(id=str(uuid.uuid4()), owner="alice", name="old",
                     endpoint_url="x", model="m", archived=True)
    db.add(arch); db.commit()
    repo = SessionRepository(db)
    assert len(repo.list_active("alice")) == 1
    assert len(repo.list_archived("alice")) == 1


def test_delete_is_owner_scoped(db):
    sid = _mk_session(db, "alice")
    repo = SessionRepository(db)
    assert repo.delete(sid, owner="bob") is False   # can't delete another's row
    assert repo.get(sid, owner="alice") is not None
    assert repo.delete(sid, owner="alice") is True
    assert repo.get(sid, owner="alice") is None


def test_document_and_task_repositories(db):
    db.add(Document(id="d1", owner="alice")); db.add(ScheduledTask(id="t1", owner="alice"))
    db.commit()
    assert DocumentRepository(db).get("d1", owner="alice").id == "d1"
    assert DocumentRepository(db).get("d1", owner="bob") is None
    assert ScheduledTaskRepository(db).count(owner="alice") == 1


def test_document_service_owner_scoped(db, monkeypatch):
    """A2 service slice: document_service delegates to the repository + scopes."""
    import contextlib

    from src.services import document_service

    @contextlib.contextmanager
    def _fake_session():
        s = _TS()
        try:
            yield s
            s.commit()
        finally:
            s.close()

    monkeypatch.setattr(document_service, "get_db_session", _fake_session)
    db.add(Document(id="ds1", owner="alice", title="A")); db.commit()

    assert document_service.get_document("ds1", "alice")["id"] == "ds1"
    assert document_service.get_document("ds1", "bob") is None
    assert document_service.count_documents("alice") == 1
    # owner-scoped delete: bob can't, alice can
    assert document_service.delete_document("ds1", "bob") is False
    assert document_service.delete_document("ds1", "alice") is True
    assert document_service.get_document("ds1", "alice") is None
