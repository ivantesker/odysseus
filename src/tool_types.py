"""Shared tool primitives — the leaf both agent_tools and tool_parsing import.

Extracted to break the agent_tools ↔ tool_parsing circular import: tool_parsing
needs ``ToolBlock`` + ``TOOL_TAGS`` while agent_tools re-exports tool_parsing's
parsers. Putting these two dependency-free primitives in a leaf module lets both
import from here, so neither has to import the other at module-load time.

``TOOL_TAGS`` is a shared MUTABLE set — tool_parsing.register_extra_tool_tags()
updates it in place (e.g. for plugin tools) and the change is visible to every
module that imported the same object.
"""

from __future__ import annotations

from collections import namedtuple

# Tool types that trigger execution.
TOOL_TAGS = {"bash", "python", "web_search", "web_fetch", "read_file", "write_file", "edit_file",
             "grep", "glob", "ls",
             "create_document", "update_document", "edit_document",
             "search_chats",
             "chat_with_model", "create_session", "list_sessions",
             "send_to_session",
             "pipeline",
             "manage_session", "manage_memory", "list_models",
             "ui_control", "generate_image", "ask_user", "update_plan",
             "manage_tasks", "api_call", "ask_teacher", "manage_skills",
             "suggest_document",
             "manage_endpoints", "manage_mcp", "manage_webhooks",
             "manage_tokens", "manage_documents", "manage_settings",
             "manage_notes", "manage_calendar",
             "resolve_contact", "manage_contact", "list_email_accounts", "send_email", "list_emails",
             "read_email", "reply_to_email", "bulk_email", "archive_email",
             "delete_email", "mark_email_read",
             # Cookbook tools (LLM serving + downloads). Without these
             # entries, native function calls to e.g. list_served_models
             # are rejected as "Unknown function call" before reaching
             # the dispatcher — silent failure for the whole cookbook
             # surface.
             "download_model", "serve_model",
             "list_served_models", "stop_served_model",
             "list_downloads", "cancel_download",
             "search_hf_models", "list_cached_models",
             "list_serve_presets", "serve_preset", "adopt_served_model",
             "list_cookbook_servers",
             # Other tools the agent reaches for that were also missing.
             "edit_image", "trigger_research", "manage_research",
             # Generic loopback to any UI-button endpoint (cookbook,
             # gallery, email folders, etc.) — agent uses this when
             # there's no named tool wrapper for the action.
             "app_api"}

ToolBlock = namedtuple("ToolBlock", ["tool_type", "content"])
