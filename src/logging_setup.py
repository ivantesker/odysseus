"""Central logging configuration so the whole service is traceable.

Goals: from the logs alone you can follow what the service did and debug an
error — every HTTP request is logged with a short request-id, its method, path,
status, duration and user; that same request-id is stamped on every log line
emitted while handling the request, so a failure and the work leading to it can
be correlated.

Environment knobs:
  LOG_LEVEL   DEBUG|INFO|WARNING|ERROR        (default INFO)
  LOG_FILE    path to a rotating log file      (default: none — console only)
  LOG_FORMAT  plain|json                       (default plain)
  LOG_MAX_BYTES / LOG_BACKUP_COUNT             (file rotation; sensible defaults)
"""

import json
import logging
import os
import sys
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler

# Per-request correlation id, set by the request-logging middleware and read by
# the filter below so it appears on every record emitted during the request.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _build_formatter() -> logging.Formatter:
    if os.getenv("LOG_FORMAT", "plain").strip().lower() == "json":
        return _JsonFormatter()
    return logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def configure_logging() -> None:
    """Install console (+ optional rotating file) handlers on the root logger.

    Safe to call once at startup; it replaces any handlers a prior
    basicConfig() installed so formatting/level are consistent everywhere.
    """
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    formatter = _build_formatter()
    id_filter = _RequestIdFilter()

    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    console.addFilter(id_filter)
    root.addHandler(console)

    log_file = os.getenv("LOG_FILE", "").strip()
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            fh = RotatingFileHandler(
                log_file,
                maxBytes=int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024))),
                backupCount=int(os.getenv("LOG_BACKUP_COUNT", "5")),
                encoding="utf-8",
            )
            fh.setFormatter(formatter)
            fh.addFilter(id_filter)
            root.addHandler(fh)
        except Exception as e:  # never let logging setup crash startup
            root.warning("Could not open LOG_FILE %r: %s", log_file, e)

    # Tame third-party noise so the app's own lines stay readable. Override per
    # library via LOG_LEVEL_<NAME> if you need to debug one of them.
    for noisy, lvl in (("httpx", "WARNING"), ("httpcore", "WARNING"), ("urllib3", "WARNING")):
        logging.getLogger(noisy).setLevel(os.getenv(f"LOG_LEVEL_{noisy.upper()}", lvl))

    root.info("Logging configured: level=%s file=%s format=%s", level, log_file or "-",
              os.getenv("LOG_FORMAT", "plain"))
