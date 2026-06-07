"""CV ↔ Odysseus memory bridge.

Best-effort write-back of CV facts (model baselines, per-class thresholds,
device specs, calibration choices) into the shared memory store, and recall of
device constraints for the deploy guard. Failures never break the calling tool.
"""

from __future__ import annotations


def _manager():
    from core.constants import DATA_DIR
    from src.memory import MemoryManager
    return MemoryManager(DATA_DIR)


def remember_cv(text: str, *, owner: str | None = None, category: str = "project") -> bool:
    """Store a CV fact. Returns True on success, never raises."""
    if not text:
        return False
    try:
        _manager().add_entry(text, source="cv_pipeline", category=category, owner=owner or "")
        return True
    except Exception:
        return False


def recall_cv(query: str, *, owner: str | None = None, max_items: int = 5) -> list:
    """Return CV-relevant memory snippets for the query (best-effort)."""
    try:
        mgr = _manager()
        allm = mgr.load_all()
        scoped = [m for m in allm if not (m.get("owner") or "") or not owner or (m.get("owner") == owner)]
        hits = mgr.get_relevant_memories(query, scoped, max_items=max_items)
        return [h.get("text", "") if isinstance(h, dict) else str(h) for h in hits]
    except Exception:
        return []


def device_constraints(device: str, *, owner: str | None = None) -> list:
    """Recall stored constraints for a target device (e.g. 'rk3588')."""
    if not device:
        return []
    return recall_cv(f"device {device} latency memory constraints", owner=owner, max_items=4)
