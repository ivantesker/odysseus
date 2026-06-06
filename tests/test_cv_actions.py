"""Scheduled CV actions + memory write-back."""
import asyncio
import json

import pytest

from src.builtin_actions import (
    BUILTIN_ACTIONS,
    TaskNoop,
    action_cv_dataset_health,
    action_cv_drift_check,
)


def test_actions_registered():
    assert "cv_dataset_health" in BUILTIN_ACTIONS
    assert "cv_drift_check" in BUILTIN_ACTIONS


def test_dataset_health_action(tmp_path):
    lbl = tmp_path / "labels"; lbl.mkdir()
    for i in range(10):
        (lbl / f"{i}.txt").write_text("0 .5 .5 .2 .2\n")
    msg, ok = asyncio.run(action_cv_dataset_health("o", command=json.dumps({"labels_dir": str(lbl)})))
    assert ok is True
    assert "health" in msg.lower()


def test_dataset_health_noop_without_config():
    with pytest.raises(TaskNoop):
        asyncio.run(action_cv_dataset_health("o", command="{}"))


def test_drift_action_alerts(tmp_path):
    b = tmp_path / "b"; n = tmp_path / "n"
    b.mkdir(); n.mkdir()
    for i in range(30):
        (b / f"{i}.txt").write_text("0 .5 .5 .1 .1 0.9\n")
        (n / f"{i}.txt").write_text("1 .5 .5 .1 .1 0.3\n1 .2 .2 .1 .1 0.25\n")
    msg, ok = asyncio.run(action_cv_drift_check("o", command=json.dumps({
        "baseline_preds": str(b), "new_preds": str(n)})))
    assert ok is True
    assert "drift" in msg.lower()


def test_drift_action_noop_without_config():
    with pytest.raises(TaskNoop):
        asyncio.run(action_cv_drift_check("o", command="{}"))


def test_cv_memory_write_back(tmp_path):
    from src.services.cv import cv_memory
    assert cv_memory.remember_cv("model vX mAP 0.9", owner="t", category="project") is True
    assert cv_memory.remember_cv("", owner="t") is False  # empty → no-op
