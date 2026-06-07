"""Training-run registry (local W&B-lite): ingest an Ultralytics run directory,
extract its config + per-epoch metric curves + final metrics, persist a record,
and compare runs head-to-head. Ends the scattered eval_results_* / pick-an-epoch
chaos. Pure stdlib (csv) + optional yaml for the config.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

_PRIMARY = ["metrics/map50-95(b)", "metrics/map50-95", "map50-95",
            "metrics/map50(b)", "metrics/map50", "map50", "metrics/map", "map"]


def _runs_dir() -> str:
    from core.constants import DATA_DIR
    d = os.path.join(DATA_DIR, "cv_runs")
    os.makedirs(d, exist_ok=True)
    return d


def _read_results_csv(path: Path) -> dict:
    """Parse an Ultralytics results.csv → {curves:{col:[float]}, final:{col:float}, epochs}."""
    rows = list(csv.DictReader(path.read_text(encoding="utf-8", errors="replace").splitlines()))
    if not rows:
        return {"curves": {}, "final": {}, "epochs": 0}
    curves: dict[str, list] = {}
    for col in rows[0].keys():
        if not col:
            continue
        key = col.strip().lower()
        vals = []
        for r in rows:
            try:
                vals.append(float(r.get(col)))
            except (TypeError, ValueError):
                vals.append(None)
        if any(v is not None for v in vals):
            curves[key] = vals
    final = {k: next((v for v in reversed(vals) if v is not None), None) for k, vals in curves.items()}
    return {"curves": curves, "final": final, "epochs": len(rows)}


def _read_config(run_dir: Path) -> dict:
    for name in ("args.yaml", "opt.yaml", "hyp.yaml"):
        p = run_dir / name
        if p.exists():
            try:
                import yaml
                return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
    return {}


def ingest_run(run_dir: str, name: str | None = None) -> dict:
    """Read an Ultralytics run dir (results.csv + args.yaml) into a record."""
    rd = Path(run_dir)
    if not rd.exists():
        return {"error": f"run dir not found: {run_dir}"}
    csv_path = rd / "results.csv"
    if not csv_path.exists():
        hits = list(rd.rglob("results.csv"))
        if not hits:
            return {"error": f"no results.csv under {run_dir}"}
        csv_path = hits[0]
        rd = csv_path.parent
    parsed = _read_results_csv(csv_path)
    config = _read_config(rd)
    primary = next((p for p in _PRIMARY if p in parsed["final"]), None)
    return {
        "name": name or rd.name,
        "run_dir": str(rd),
        "config": config,
        "final": parsed["final"],
        "curves": parsed["curves"],
        "epochs": parsed["epochs"],
        "primary": primary,
        "best": round(parsed["final"].get(primary), 5) if primary and parsed["final"].get(primary) is not None else None,
    }


def register_run(run_dir: str, *, name: str | None = None, owner: str | None = None) -> dict:
    """Ingest + persist a run record; returns the record with an id."""
    import secrets
    rec = ingest_run(run_dir, name)
    if rec.get("error"):
        return rec
    rid = "run-" + secrets.token_hex(5)
    rec["id"] = rid
    rec["owner"] = owner or ""
    # Don't persist full curves twice — keep the record lean but keep final+config.
    lean = {k: rec[k] for k in ("id", "name", "run_dir", "config", "final", "epochs", "primary", "best", "owner")}
    with open(os.path.join(_runs_dir(), f"{rid}.json"), "w", encoding="utf-8") as f:
        json.dump(lean, f, ensure_ascii=False)
    return rec


def list_runs(*, owner: str | None = None) -> list:
    out = []
    base = _runs_dir()
    for fn in os.listdir(base):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(base, fn), encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            continue
        ro = rec.get("owner") or ""
        if ro and owner and ro != owner:
            continue
        out.append({"id": rec.get("id"), "name": rec.get("name"), "best": rec.get("best"),
                    "primary": rec.get("primary"), "epochs": rec.get("epochs")})
    out.sort(key=lambda r: (r["best"] is None, -(r["best"] or 0)))
    return out


def compare_runs(run_dirs: list[str], names: list[str] | None = None) -> dict:
    """Ingest several runs and assemble overlay curves + final table + config diff."""
    recs = []
    for i, d in enumerate(run_dirs):
        r = ingest_run(d, names[i] if names and i < len(names) else None)
        if not r.get("error"):
            recs.append(r)
    if not recs:
        return {"error": "no readable runs"}
    # Config diff: keys whose values differ across runs.
    all_keys = set()
    for r in recs:
        all_keys |= set(r.get("config", {}).keys())
    diff = {}
    for k in sorted(all_keys):
        vals = [r.get("config", {}).get(k) for r in recs]
        if len(set(str(v) for v in vals)) > 1:
            diff[k] = {recs[i]["name"]: vals[i] for i in range(len(recs))}
    # Final metrics table over the union of metric keys.
    metric_keys = []
    for r in recs:
        for k in r["final"]:
            if k != "epoch" and k not in metric_keys:
                metric_keys.append(k)
    primary = next((p for p in _PRIMARY if any(p in r["final"] for r in recs)), None)
    return {
        "runs": [{"name": r["name"], "final": r["final"], "curves": r["curves"],
                  "epochs": r["epochs"], "best": r["best"]} for r in recs],
        "metric_keys": metric_keys,
        "primary": primary,
        "config_diff": diff,
    }
