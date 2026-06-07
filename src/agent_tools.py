"""
agent_tools.py — Facade module.

Re-exports tool parsing, schemas, execution, and implementations
for backward compatibility. All importers continue to work unchanged.

Sub-modules:
  - tool_parsing.py: regex patterns, parse/strip functions
  - tool_schemas.py: FUNCTION_TOOL_SCHEMAS, function_call_to_tool_block
  - tool_execution.py: execute_tool_block, format_tool_result, MCP helpers
  - tool_implementations.py: all do_* tool functions
"""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (kept here — sub-modules import from here)
# ---------------------------------------------------------------------------
MAX_AGENT_ROUNDS = 50
SHELL_TIMEOUT = 60
PYTHON_TIMEOUT = 30
MAX_OUTPUT_CHARS = 10_000
MAX_READ_CHARS = 20_000

# Tool primitives moved to the leaf module src.tool_types to break the
# agent_tools ↔ tool_parsing circular import. Re-exported here so the many
# `from src.agent_tools import ToolBlock, TOOL_TAGS` call sites keep working.
# TOOL_TAGS stays a single shared mutable set (register_extra_tool_tags updates
# it in place).
from src.tool_types import TOOL_TAGS, ToolBlock  # noqa: F401

# ---------------------------------------------------------------------------
# MCP Manager (kept here — used by execution and agent_loop)
# ---------------------------------------------------------------------------
_mcp_manager = None

def set_mcp_manager(manager):
    """Set the global MCP manager instance."""
    global _mcp_manager
    _mcp_manager = manager

def get_mcp_manager():
    """Get the global MCP manager instance."""
    return _mcp_manager

# ---------------------------------------------------------------------------
# Helpers (kept here — used by sub-modules)
# ---------------------------------------------------------------------------
def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    # Callers treat the result as text, so always return a string: coerce a
    # non-string (None -> "", otherwise str(...)) instead of returning it raw,
    # which would just move the crash downstream.
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    if len(text) > limit:
        return text[:limit] + f"\n... (truncated, {len(text)} chars total)"
    return text

# ---------------------------------------------------------------------------
# Re-exports from sub-modules
# ---------------------------------------------------------------------------

# Parsing
from src.tool_parsing import (  # noqa: E402, F401
    parse_tool_blocks,
    strip_tool_blocks,
    _TOOL_NAME_MAP,
    _TOOL_BLOCK_RE,
    _TOOL_CALL_RE,
    _XML_TOOL_CALL_RE,
    _XML_INVOKE_RE,
    _XML_PARAM_RE,
)

# Schemas
from src.tool_schemas import (  # noqa: E402, F401
    FUNCTION_TOOL_SCHEMAS,
    function_call_to_tool_block,
)

# Execution
from src.tool_execution import (  # noqa: E402, F401
    execute_tool_block,
    format_tool_result,
)

# Implementations
from src.tool_implementations import (  # noqa: E402, F401
    set_active_document,
    set_active_model,
    get_active_document,
    do_create_document,
    do_update_document,
    do_edit_document,
    do_suggest_document,
    do_search_chats,
    do_manage_skills,
    do_manage_tasks,
    do_manage_endpoints,
    do_manage_mcp,
    do_manage_webhooks,
    do_manage_tokens,
    do_manage_documents,
    do_manage_settings,
    do_api_call,
)
