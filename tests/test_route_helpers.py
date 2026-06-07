"""Route observability helpers (Q4)."""
import asyncio
import inspect
import logging

import pytest
from fastapi import HTTPException

from src.route_helpers import log_route_errors, logged_rollback


def test_decorator_passes_http_exception_through():
    @log_route_errors
    def handler():
        raise HTTPException(404, "nope")
    with pytest.raises(HTTPException) as ei:
        handler()
    assert ei.value.status_code == 404


def test_decorator_logs_and_reraises_other(caplog):
    @log_route_errors
    def handler():
        raise ValueError("boom")
    with caplog.at_level(logging.ERROR, logger="odysseus.routes"):
        with pytest.raises(ValueError):
            handler()
    assert any("Unhandled error" in r.message for r in caplog.records)


def test_decorator_async():
    @log_route_errors
    async def handler(x):
        return x * 2
    assert asyncio.run(handler(3)) == 6
    assert inspect.iscoroutinefunction(handler)


def test_decorator_preserves_signature():
    @log_route_errors
    def handler(a, b, c=1):
        return a
    # FastAPI introspects the signature → must be preserved
    params = list(inspect.signature(handler).parameters)
    assert params == ["a", "b", "c"]
    assert handler.__name__ == "handler"


def test_logged_rollback_logs_and_rolls_back(caplog):
    class FakeDB:
        rolled = False

        def rollback(self):
            self.rolled = True

    db = FakeDB()
    log = logging.getLogger("test.rollback")
    with caplog.at_level(logging.WARNING, logger="test.rollback"):
        logged_rollback(db, log, "do thing")
    assert db.rolled is True
    assert any("do thing" in r.message for r in caplog.records)
