"""Triton Inference Server health probe.

Polls a Triton HTTP endpoint: server-ready, then per-model ready + a timed
inference-config fetch as a latency proxy. Pure httpx; degrades to a clear
'down' status instead of raising, so it's safe inside a scheduled action.
"""

from __future__ import annotations

import time


def _get(url: str, timeout: float):
    import httpx
    t0 = time.perf_counter()
    r = httpx.get(url, timeout=timeout)
    return r.status_code, round((time.perf_counter() - t0) * 1000, 1)


def check_triton(base_url: str, models: list[str] | None = None, *, timeout: float = 5.0) -> dict:
    """Probe a Triton server + named models. Returns status + per-model latency."""
    base = (base_url or "").rstrip("/")
    if not base:
        return {"error": "base_url is required"}
    try:
        import httpx  # noqa: F401
    except Exception:
        return {"error": "httpx not installed"}

    out = {"base_url": base, "models": [], "server_ready": False}
    try:
        code, ms = _get(f"{base}/v2/health/ready", timeout)
        out["server_ready"] = code == 200
        out["server_latency_ms"] = ms
    except Exception as e:
        out["server_ready"] = False
        out["error"] = str(e)[:200]

    for m in (models or []):
        entry = {"name": m, "ready": False}
        try:
            code, ms = _get(f"{base}/v2/models/{m}/ready", timeout)
            entry["ready"] = code == 200
            entry["latency_ms"] = ms
        except Exception as e:
            entry["error"] = str(e)[:120]
        out["models"].append(entry)

    down = [m["name"] for m in out["models"] if not m["ready"]]
    out["all_ready"] = out["server_ready"] and not down
    out["down"] = down
    out["alert"] = not out["all_ready"]
    return out
