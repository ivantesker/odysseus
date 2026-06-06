"""Pure text helpers for the agent tools — parsing, truncation, document sniffing.

Split out of src/tool_implementations.py: no DB, no globals, no I/O. Just string
logic the tool handlers rely on, so it can be unit-tested directly.
"""

import json
import re

DEFAULT_OUTPUT_LIMIT = 10_000


def truncate(text: str, limit: int = DEFAULT_OUTPUT_LIMIT) -> str:
    if len(text) > limit:
        return text[:limit] + f"\n... (truncated, {len(text)} chars total)"
    return text


def parse_tool_args(content):
    """Parse a tool-call argument blob (JSON string or dict).

    Unwraps the common ``{"body": {...}}`` envelope smaller models emit when they
    read "Body is JSON: {...}" literally. Returns a dict; raises ValueError on
    bad JSON.
    """
    if isinstance(content, str):
        try:
            args = json.loads(content) if content.strip() else {}
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(str(e)) from e
    elif isinstance(content, dict):
        args = content
    else:
        args = {}
    if (
        isinstance(args, dict)
        and len(args) == 1
        and "body" in args
        and isinstance(args["body"], dict)
        and "action" in args["body"]
    ):
        args = args["body"]
    return args


def looks_like_email_document(text: str = "", title: str = "") -> bool:
    title_l = (title or "").strip().lower()
    if title_l in {"new email", "new mail", "new message"}:
        return True
    s = (text or "").lstrip()
    if "\n---\n" in s and re.search(r"(?im)^To:\s*", s) and re.search(r"(?im)^Subject:\s*", s):
        return True
    return bool(re.search(r"(?im)^To:\s*", s) and re.search(r"(?im)^Subject:\s*", s))


def sniff_doc_language(text: str) -> str:
    """Best-effort language detection when the model didn't specify one.

    Defaults to 'markdown' (prose); recognizes the common markup/code types the
    editor supports so e.g. an SVG isn't saved as markdown.
    """
    s = (text or "").strip()
    if not s:
        return "markdown"
    head = s[:600]
    hl = head.lower()
    if looks_like_email_document(s):
        return "email"
    if "<svg" in hl:
        return "svg"
    if hl.startswith("<?xml"):
        return "xml"
    if (hl.startswith("<!doctype html") or hl.startswith("<html")
            or re.search(r"<(div|body|head|p|span|table|button|h[1-6]|ul|ol|li|img)\b", hl)):
        return "html"
    if s[0] in "{[":
        try:
            json.loads(s)
            return "json"
        except Exception:
            pass
    first = s.split("\n", 1)[0].strip().lower()
    if first.startswith("#!"):
        return "python" if "python" in first else "bash"
    if re.search(r"(?m)^\s*(def \w|class \w|import \w|from \w[\w.]* import )", s):
        return "python"
    if re.search(r"(?m)^\s*(function \w|const \w|let \w|export |import .* from )", s):
        return "javascript"
    if re.search(r"(?mi)^\s*(select .* from |create table |insert into |update \w)", s):
        return "sql"
    if re.search(r"(?m)^[.#]?[\w-]+\s*\{[^{}]*:[^{}]*;", s):
        return "css"
    return "markdown"


def coerce_email_document_content(existing: str, incoming: str) -> str:
    """Keep email docs in the To/Subject/---/body shape even if a model writes
    only the body or dumps header labels without the separator."""
    old = existing or ""
    new = (incoming or "").strip()
    if "\n---\n" in new:
        return new
    header = old.split("\n---\n", 1)[0] if "\n---\n" in old else "To: \nSubject: "
    if looks_like_email_document(new):
        lines = new.splitlines()
        last_header_idx = -1
        header_re = re.compile(
            r"^(To|Cc|Bcc|Subject|In-Reply-To|References|X-Source-UID|X-Source-Folder|X-Attachments):",
            re.I,
        )
        for i, line in enumerate(lines):
            if header_re.match(line.strip()):
                last_header_idx = i
        body_lines = lines[last_header_idx + 1:] if last_header_idx >= 0 else lines
        while body_lines and not body_lines[0].strip():
            body_lines.pop(0)
        body = "\n".join(body_lines).strip()
    else:
        body = new
    return header.rstrip() + "\n---\n" + body
