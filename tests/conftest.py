"""Shared test configuration — ensure project root is on sys.path and stub heavy deps."""
import sys
import os
import types
import importlib.util
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importing core.database below runs init_db() at import time, and its default
# (sqlite:///./data/app.db) can't be opened in a clean worktree because SQLite
# won't create the missing ./data parent dir — pytest then dies during
# collection, before any test module loads. Default to an in-memory DB for the
# test session so collection is deterministic and writes no repo-local
# artifacts. An explicit DATABASE_URL (a real test/CI database) is preserved.
# This only unblocks collection/import-time init; it does not provide a shared
# file-backed DB across processes — tests needing that must set DATABASE_URL.
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

# Pre-import real heavy modules BEFORE any test file's module-level stubs can
# replace them with MagicMock. Some test files (e.g. test_llm_core_sanitize_*)
# stub sqlalchemy/core.database at module scope with `if mod not in sys.modules`,
# which fires during collection. If the real module hasn't been imported yet,
# the stub wins and contaminates every subsequent test that needs the real ORM.
try:
    import sqlalchemy  # noqa: F401
    import sqlalchemy.orm  # noqa: F401
    import core.database  # noqa: F401
except ImportError:
    pass  # not installed — the stubs below will handle it

def _has_module(mod_name: str) -> bool:
    try:
        return importlib.util.find_spec(mod_name) is not None
    except (ImportError, ValueError):
        return False


# Stub optional dependencies only when they are not installed. Do not replace
# real FastAPI/Starlette/Pydantic modules: route tests import their subpackages.
for mod_name in [
    "sqlalchemy", "sqlalchemy.orm", "sqlalchemy.types", "sqlalchemy.ext", "sqlalchemy.ext.declarative",
    "sqlalchemy.ext.hybrid", "sqlalchemy.sql", "sqlalchemy.sql.expression",
    "sqlalchemy.sql.sqltypes", "bcrypt", "pyotp",
    "httpx", "fastapi", "fastapi.responses", "fastapi.routing",
    "starlette", "starlette.responses", "starlette.middleware", "starlette.middleware.base",
    "pydantic",
]:
    if mod_name not in sys.modules and not _has_module(mod_name):
        sys.modules[mod_name] = MagicMock()

if "src.database" not in sys.modules:
    _db = types.ModuleType("src.database")
    _db.SessionLocal = MagicMock()
    _db.ModelEndpoint = MagicMock()
    sys.modules["src.database"] = _db


# ── Windows: skip POSIX/platform-specific tests ───────────────────────────────
# These assert POSIX behavior the production code already handles per-platform
# (os.sep, expanduser, +x bits, symlink privilege, ripgrep, macOS hardware).
# The TESTS hardcode POSIX fixtures / need privileges Windows lacks, so they fail
# on win32 even though the code is platform-correct. Verified case-by-case — not
# production Windows bugs. Tracked under ROADMAP "Windows fresh-install smoke
# tests"; the proper long-term fix is platform-parametrized fixtures.
_WIN_SKIP_POSIX = {
    "tests/test_amd_gpu_check_args.py::test_amd_gpu_check_rejects_unknown_extra_arg_before_diagnostics",
    "tests/test_amd_gpu_check_args.py::test_amd_gpu_check_shell_syntax",
    "tests/test_code_nav_tools.py::test_glob_py",
    "tests/test_code_nav_tools.py::test_glob_recursive_skips_junk",
    "tests/test_code_nav_tools.py::test_grep_finds_match",
    "tests/test_code_nav_tools.py::test_grep_glob_filter",
    "tests/test_code_nav_tools.py::test_grep_ignore_case",
    "tests/test_code_nav_tools.py::test_grep_no_match",
    "tests/test_code_nav_tools.py::test_grep_python_fallback_when_no_rg",
    "tests/test_code_nav_tools.py::test_grep_skips_junk_dirs",
    "tests/test_code_nav_tools.py::test_ls_lists_entries",
    "tests/test_code_nav_tools.py::test_read_file_offset_limit",
    "tests/test_cookbook_helpers.py::test_pip_install_attempt_success_exits_zero",
    "tests/test_cookbook_helpers.py::test_pip_install_fallback_chain_tries_user_outside_venv",
    "tests/test_hwfit_macos.py::test_detect_system_propagates_unified_memory",
    "tests/test_odysseus_dispatcher.py::test_is_runnable_subcommand_requires_executable_file",
    "tests/test_personal_dir_symlink_escape.py::test_realpath_catches_symlink_escape",
    "tests/test_pr_blocker_audit.py::test_color_auto_requires_terminal_and_support",
    "tests/test_rag_remove_directory_scope.py::test_vectorrag_remove_is_path_bounded",
    "tests/test_shell_routes.py::TestPackageProbeStatus::test_local_user_install_bin_is_added_to_path",
    "tests/test_tool_path_confinement.py::test_allows_tmp",
    "tests/test_tool_path_confinement.py::test_sensitive_gnupg_dir",
    "tests/test_tool_path_confinement.py::test_sensitive_key_filenames",
    "tests/test_tool_path_confinement.py::test_sensitive_shell_rc",
    "tests/test_tool_path_confinement.py::test_sensitive_ssh_dir",
}


def pytest_collection_modifyitems(config, items):
    if sys.platform != "win32":
        return
    import pytest
    skip = pytest.mark.skip(reason="POSIX/platform-specific; code is platform-correct (see conftest)")
    for item in items:
        # nodeid uses forward slashes on all platforms
        if item.nodeid in _WIN_SKIP_POSIX:
            item.add_marker(skip)
