from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path

from privibe import VIBE_ROOT


class GlobalPath:
    def __init__(self, resolver: Callable[[], Path]) -> None:
        self._resolver = resolver

    @property
    def path(self) -> Path:
        return self._resolver()


_DEFAULT_VIBE_HOME = Path.home() / ".privibe"


def _get_vibe_home() -> Path:
    if vibe_home := os.getenv("VIBE_HOME"):
        return Path(vibe_home).expanduser().resolve()
    return _DEFAULT_VIBE_HOME


VIBE_HOME = GlobalPath(_get_vibe_home)
GLOBAL_ENV_FILE = GlobalPath(lambda: VIBE_HOME.path / ".env")
SESSION_LOG_DIR = GlobalPath(lambda: VIBE_HOME.path / "logs" / "session")
TRUSTED_FOLDERS_FILE = GlobalPath(lambda: VIBE_HOME.path / "trusted_folders.toml")
LOG_DIR = GlobalPath(lambda: VIBE_HOME.path / "logs")
LOG_FILE = GlobalPath(lambda: VIBE_HOME.path / "logs" / "privibe.log")
HISTORY_FILE = GlobalPath(lambda: VIBE_HOME.path / "vibehistory")
PLANS_DIR = GlobalPath(lambda: VIBE_HOME.path / "plans")
# DEBUG LLM COMMUNICATIONS — destination for per-turn message and
# payload dumps when llm_debug_dump is on. Lives under VIBE_HOME so
# the dumps don't accumulate in whichever project cwd the user
# happened to launch privibe from.
DEBUG_DIR = GlobalPath(lambda: VIBE_HOME.path / "debug")

DEFAULT_TOOL_DIR = GlobalPath(lambda: VIBE_ROOT / "core" / "tools" / "builtins")
