# Re-export everything from the canonical core.constants module so that
# `from src.constants import X` keeps working while there is a single source
# of truth. (Previously this file was a near-duplicate that drifted from
# core.constants — they disagreed on APP_VERSION.)
from core.constants import *  # noqa: F401,F403
from core.constants import (  # explicit re-exports for IDE/type-checker visibility
    APP_VERSION,
    BASE_DIR,
    STATIC_DIR,
    DATA_DIR,
    SESSIONS_FILE,
    MEMORY_FILE,
    MEMORY_DOC,
    PERSONAL_DIR,
    RUNBOOK_DIR,
    UPLOAD_DIR,
    FEATURES_FILE,
    SETTINGS_FILE,
    MAX_CONTEXT_MESSAGES,
    REQUEST_TIMEOUT,
    OPENAI_COMPAT_PATH,
    DEFAULT_HOST,
    LLM_HOSTS,
    SEARXNG_INSTANCE,
    CLEANUP_ENABLED,
    CLEANUP_INTERVAL_HOURS,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
)
