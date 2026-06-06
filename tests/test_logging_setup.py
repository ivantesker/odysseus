"""Logging config + per-request correlation id."""
import logging

from src.logging_setup import _RequestIdFilter, configure_logging, request_id_var


def test_configure_logging_installs_handlers_and_level(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.delenv("LOG_FILE", raising=False)
    configure_logging()
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert root.handlers, "expected at least a console handler"


def test_request_id_filter_stamps_current_id():
    rec = logging.LogRecord("n", logging.INFO, __file__, 1, "msg", None, None)
    token = request_id_var.set("deadbeef")
    try:
        assert _RequestIdFilter().filter(rec) is True
        assert rec.request_id == "deadbeef"
    finally:
        request_id_var.reset(token)


def test_json_format(monkeypatch, capsys):
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.delenv("LOG_FILE", raising=False)
    configure_logging()
    request_id_var.set("11112222")
    logging.getLogger("svc").info("structured")
    err = capsys.readouterr().err
    assert '"request_id": "11112222"' in err
    assert '"msg": "structured"' in err
    # restore plain format for other tests
    monkeypatch.setenv("LOG_FORMAT", "plain")
    configure_logging()


def test_file_handler_writes(tmp_path, monkeypatch):
    log_file = tmp_path / "app.log"
    monkeypatch.setenv("LOG_FILE", str(log_file))
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    configure_logging()
    logging.getLogger("svc").info("to-file")
    for h in logging.getLogger().handlers:
        h.flush()
    assert log_file.exists()
    assert "to-file" in log_file.read_text(encoding="utf-8")
    monkeypatch.delenv("LOG_FILE", raising=False)
    configure_logging()
