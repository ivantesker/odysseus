"""Path confinement for CV tools.

CV routes + agent tools take filesystem paths (labels/images/out dirs, ONNX
models, calibration lists). Without confinement a prompt-injected admin agent
could read/write anywhere. Route every user-supplied path through the same
allowlist the read_file/write_file tools use: the project data dir, /tmp, and
the user's opt-in ``tool_path_extra_roots`` setting (where the user adds their
dataset drives, e.g. D:/rider_dome). Sensitive dirs (.ssh, …) stay blocked.
"""

from __future__ import annotations


class CvPathError(ValueError):
    """Raised when a CV path is empty or outside the allowed roots."""


def confine(raw, *, required: bool = True, field: str = "path") -> str | None:
    """Resolve + confine a user-supplied path. Returns the realpath, or None
    when optional and empty. Raises CvPathError otherwise."""
    s = (str(raw).strip() if raw is not None else "")
    if not s:
        if required:
            raise CvPathError(f"{field} is required")
        return None
    try:
        from src.tool_execution import _resolve_tool_path
        return _resolve_tool_path(s)
    except ValueError as e:
        raise CvPathError(f"{field}: {e}") from e
