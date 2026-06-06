"""Model-endpoint pure logic — curation, parsing, classification helpers.

Split out of routes/model_routes.py: no DB, no HTTP, no Request. Host/connection
and ORM-bound helpers stay in the route module; everything here is pure functions
over strings/lists so model curation and id handling are unit-testable.
"""

import json
import re
from typing import Any

# Curated, ordered model lists per provider key. Used to partition a probed
# model list into "curated" (surfaced first) vs "extra".
PROVIDER_CURATED = {
    "openai": [
        "gpt-5.2", "gpt-5.2-pro", "gpt-5", "gpt-5-pro", "gpt-5-mini", "gpt-5-nano",
        "gpt-4o", "gpt-4o-mini", "o3", "o4-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
        "gpt-image-1.5", "gpt-image-1", "dall-e-3", "tts-1", "whisper-1",
    ],
    "anthropic": [
        "claude-sonnet-4", "claude-opus-4", "claude-haiku-4",
        "claude-sonnet-4-5", "claude-haiku-3-5",
    ],
    "zai": [
        "glm-5", "glm-5.1", "glm-5v-turbo", "glm-4.7", "glm-4.7-flash",
        "glm-4.6", "glm-4.6v",
        "glm-4.5", "glm-4.5v", "glm-4.5-air", "glm-4.5-flash",
    ],
    "zai-coding": [
        "glm-5.1", "glm-5v-turbo", "glm-5-turbo", "glm-4.7", "glm-4.5-air",
    ],
    "deepseek": [
        "deepseek-chat", "deepseek-reasoner",
    ],
    "groq": [
        "openai/gpt-oss-120b", "openai/gpt-oss-20b",
        "groq/compound", "groq/compound-mini",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "llama-4-scout-17b-16e-instruct",
        "llama-4-maverick-17b-128e-instruct",
    ],
    "mistral": [
        "mistral-large-latest", "mistral-medium-latest", "mistral-small-latest",
    ],
    "together": [
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        "deepseek-ai/DeepSeek-R1",
        "Qwen/Qwen2.5-72B-Instruct-Turbo",
    ],
    "fireworks": [
        "accounts/fireworks/models/llama4-scout-instruct-basic",
        "accounts/fireworks/models/llama4-maverick-instruct-basic",
        "accounts/fireworks/models/deepseek-r1",
    ],
    "google": [
        "gemini-3.5", "gemini-3.1", "gemini-3",
        "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash",
    ],
    "xai": [
        "grok-4.3", "grok-4", "grok-4-fast", "grok-3", "grok-3-fast",
    ],
}


def curate_models(model_ids, provider):
    """Partition model_ids into (curated, extra) by the provider's curated list.

    No curated list (or "openrouter") → (model_ids, []). Curated entries are
    matched by exact id or longest-prefix and sorted by curated priority.
    """
    if provider == "openrouter":
        return model_ids, []
    curated_list = PROVIDER_CURATED.get(provider)
    if not curated_list:
        return model_ids, []
    curated, extra = [], []

    def _best_match_idx(mid):
        best_i, best_len = -1, 0
        for i, entry in enumerate(curated_list):
            if (mid == entry or mid.startswith(entry)) and len(entry) > best_len:
                best_i, best_len = i, len(entry)
        return best_i

    for mid in model_ids:
        (curated if _best_match_idx(mid) >= 0 else extra).append(mid)
    curated.sort(key=lambda mid: (_best_match_idx(mid), mid))
    return curated, extra


def truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("true", "1", "yes", "on")


ENDPOINT_KINDS = {"auto", "local", "api", "proxy"}
REFRESH_MODES = {"auto", "manual", "disabled"}


def normalize_endpoint_kind(value: Any) -> str:
    kind = str(value or "auto").strip().lower()
    return kind if kind in ENDPOINT_KINDS else "auto"


def normalize_refresh_mode(value: Any, endpoint_kind: str = "auto") -> str:
    mode = str(value or "").strip().lower()
    kind = normalize_endpoint_kind(endpoint_kind)
    if mode in ("manual", "disabled"):
        return mode
    if mode == "auto" and kind != "proxy":
        return "auto"
    return "manual" if kind == "proxy" else "auto"


def parse_positive_int(raw: Any, *, minimum: int = 1, maximum: int = 86400) -> int | None:
    try:
        val = int(str(raw).strip())
    except Exception:
        return None
    if val < minimum:
        return None
    return min(val, maximum)


def parse_model_list(raw: Any) -> list[str]:
    """Sanitized model-id list from JSON / list / comma-or-newline text."""
    if raw is None:
        return []
    value = raw
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            value = parsed if isinstance(parsed, list) else re.split(r"[\n,]+", text)
        except Exception:
            value = re.split(r"[\n,]+", text)
    if not isinstance(value, list):
        return []
    out, seen = [], set()
    for item in value:
        mid = str(item or "").strip()
        if not mid or mid in seen:
            continue
        seen.add(mid)
        out.append(mid)
    return out


def normalize_model_ids(value):
    """Coerce a model-id input (list / JSON string / comma-newline string) to a
    clean, de-duplicated, order-preserving list of non-empty strings."""
    if value is None:
        return []
    items = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        items = parsed if isinstance(parsed, list) else re.split(r"[,\n]", text)
    if not isinstance(items, list):
        return []
    out, seen = [], set()
    for item in items:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def merge_model_ids(*lists):
    """Concatenate model-id lists, de-duplicating and preserving order."""
    out, seen = [], set()
    for ids in lists:
        for m in (ids or []):
            if not isinstance(m, str) or m in seen:
                continue
            seen.add(m)
            out.append(m)
    return out


def visible_models(cached_models, hidden_models, pinned_models=None):
    """Merge cached + pinned ids, then drop hidden ones (all inputs normalized)."""
    merged = merge_model_ids(
        normalize_model_ids(cached_models),
        normalize_model_ids(pinned_models),
    )
    if not hidden_models:
        return merged
    hidden = set(normalize_model_ids(hidden_models))
    return [m for m in merged if m not in hidden]


# Models that are NOT chat/completions-capable.
NON_CHAT_PREFIXES = (
    "dall-e", "tts-", "whisper", "text-embedding", "embedding",
    "davinci", "babbage", "moderation", "omni-moderation",
    "sora", "gpt-image", "chatgpt-image",
)
NON_CHAT_CONTAINS = (
    "-realtime", "-transcribe", "-tts", "-codex", "codex-",
)
NON_CHAT_EXACT_PREFIXES = (
    "gpt-audio",
    "gpt-3.5-turbo-instruct",
)


def is_chat_model(model_id: str) -> bool:
    """True if the model id looks chat/completions-capable."""
    mid = model_id.lower()
    if any(mid.startswith(p) for p in NON_CHAT_PREFIXES):
        return False
    if any(mid.startswith(p) for p in NON_CHAT_EXACT_PREFIXES):
        return False
    if any(s in mid for s in NON_CHAT_CONTAINS):
        return False
    return True
