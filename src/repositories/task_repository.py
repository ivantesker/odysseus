"""Owner-scoped access to ScheduledTasks."""

from __future__ import annotations

from core.database import ScheduledTask

from .base import BaseRepository


class ScheduledTaskRepository(BaseRepository):
    model = ScheduledTask
