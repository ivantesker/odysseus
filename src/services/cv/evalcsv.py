"""Aggregate scattered evaluation CSVs into one comparison table.

The user's eval results live in dated folders (eval_results_09_02, Mar12_190705)
as CSVs with varying columns. This walks a root, reads every CSV, pulls its
numeric metrics (the final/summary row), and builds one comparison keyed by the
file/folder name — so model selection stops being "open 12 folders by hand".
Pure stdlib (csv).
"""

from __future__ import annotations

import csv
from pathlib import Path

# Columns we treat as "the headline metric" for the comparison bar chart, in order.
_PRIMARY = ["map50-95", "map5095", "map50_95", "map", "map50", "map_50", "ap",
            "accuracy", "acc", "f1", "mota", "precision", "recall"]


def _num(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def _summarize_csv(path: Path) -> dict | None:
    try:
        rows = list(csv.DictReader(path.read_text(encoding="utf-8", errors="replace").splitlines()))
    except Exception:
        return None
    if not rows:
        return None
    # Numeric columns = those parseable as float in the LAST row (final epoch /
    # summary line). Fall back to the column max if the last row is blank.
    metrics = {}
    for col in rows[0].keys():
        if not col:
            continue
        key = col.strip().lower().replace("(", "").replace(")", "")
        if key.startswith("metrics/"):
            key = key[len("metrics/"):]
        last = _num(rows[-1].get(col))
        if last is None:
            vals = [_num(r.get(col)) for r in rows]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            last = max(vals)
        metrics[key] = round(last, 5)
    return metrics or None


def aggregate_csvs(root, *, limit: int = 200) -> dict:
    """Walk `root`, summarize every CSV, return a comparison table."""
    base = Path(root)
    if not base.exists():
        return {"error": f"path not found: {root}"}
    files = sorted(base.rglob("*.csv"))[:limit]
    runs = []
    all_metrics: list[str] = []
    for f in files:
        m = _summarize_csv(f)
        if not m:
            continue
        # Name by the parent folder if it looks like a run dir, else the file stem.
        name = f.parent.name if f.name in ("results.csv", "metrics.csv", "eval.csv") else f.stem
        runs.append({"name": name, "path": str(f), "metrics": m})
        for k in m:
            if k not in all_metrics:
                all_metrics.append(k)
    if not runs:
        return {"error": f"no readable CSVs under {root}", "files_seen": len(files)}
    primary = next((p for p in _PRIMARY if p in all_metrics), all_metrics[0] if all_metrics else None)
    # Sort runs by the primary metric desc (best first) when present.
    if primary:
        runs.sort(key=lambda r: r["metrics"].get(primary, float("-inf")), reverse=True)
    return {"runs": runs, "metrics": all_metrics, "primary": primary, "n_files": len(files)}
