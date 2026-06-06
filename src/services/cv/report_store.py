"""Persist + look up generated CV reports (the rendered EDA HTML).

Mirrors how Deep Research stores its reports (data/deep_research/rp-*.json,
owner-scoped, traversal-safe id). A CV report id is ``cv-<12 hex>``. We store
the rendered HTML plus a small JSON sidecar (owner + meta) so the GET endpoint
can owner-check before serving.
"""

from __future__ import annotations

import json
import os
import re
import secrets

_ID_RE = re.compile(r"^cv-[0-9a-f]{12}$")


def _reports_dir() -> str:
    from core.constants import DATA_DIR
    d = os.path.join(DATA_DIR, "cv_reports")
    os.makedirs(d, exist_ok=True)
    return d


def new_report_id() -> str:
    return "cv-" + secrets.token_hex(6)


def _paths(report_id: str):
    if not _ID_RE.match(report_id or ""):
        return None, None
    base = _reports_dir()
    return os.path.join(base, f"{report_id}.html"), os.path.join(base, f"{report_id}.json")


def save_report(html: str, *, owner: str | None, meta: dict | None = None) -> str:
    """Write the rendered HTML + sidecar, return the report id."""
    report_id = new_report_id()
    html_path, json_path = _paths(report_id)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    sidecar = {"id": report_id, "owner": owner or "", **(meta or {})}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, ensure_ascii=False)
    return report_id


def list_reports(*, owner: str | None, limit: int = 50) -> list:
    """List stored reports the owner may see, newest first."""
    base = _reports_dir()
    out = []
    for jp in os.listdir(base):
        if not jp.endswith(".json"):
            continue
        path = os.path.join(base, jp)
        try:
            with open(path, encoding="utf-8") as f:
                sc = json.load(f)
        except Exception:
            continue
        rep_owner = sc.get("owner") or ""
        if rep_owner and owner and rep_owner != owner:
            continue
        if rep_owner and not owner:
            continue
        rid = sc.get("id") or jp[:-5]
        out.append({
            "id": rid,
            "title": sc.get("title") or rid,
            "url": f"/api/cv/report/{rid}",
            "mtime": os.path.getmtime(path),
        })
    out.sort(key=lambda r: -r["mtime"])
    return out[:limit]


def load_report(report_id: str, *, owner: str | None) -> str | None:
    """Return the report HTML if it exists and the owner matches, else None.

    An unowned report (owner == "") is readable by anyone (single-user / no-auth
    deployments); an owned one only by its owner.
    """
    html_path, json_path = _paths(report_id)
    if not html_path or not os.path.exists(html_path):
        return None
    try:
        with open(json_path, encoding="utf-8") as f:
            sidecar = json.load(f)
    except Exception:
        sidecar = {}
    rep_owner = sidecar.get("owner") or ""
    if rep_owner and owner and rep_owner != owner:
        return None
    if rep_owner and not owner:
        return None
    with open(html_path, encoding="utf-8") as f:
        return f.read()
