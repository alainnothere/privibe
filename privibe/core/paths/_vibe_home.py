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


# privibe is a fork of Mistral Vibe, which read VIBE_HOME (pointing at ~/.vibe).
# We deliberately do NOT honor VIBE_HOME: a stale value left over from Vibe would
# silently redirect privibe into ~/.vibe and load an incompatible upstream config
# (mismatched transcribe/tts `client` enums, etc). PRIVIBE_HOME is the sole
# supported override; legacy_home_env_notice() surfaces the deprecation.
HOME_ENV_VAR = "PRIVIBE_HOME"
LEGACY_HOME_ENV_VAR = "VIBE_HOME"

_DEFAULT_VIBE_HOME = Path.home() / ".privibe"


def _get_vibe_home() -> Path:
    if home := os.getenv(HOME_ENV_VAR):
        return Path(home).expanduser().resolve()
    return _DEFAULT_VIBE_HOME


def legacy_home_env_notice() -> str | None:
    """Return a one-line warning when the deprecated VIBE_HOME is set but
    PRIVIBE_HOME is not. VIBE_HOME is a Mistral Vibe holdover that privibe
    ignores, so a stale value can't quietly point us at ~/.vibe. Returns None
    when there is nothing to warn about."""
    if os.getenv(LEGACY_HOME_ENV_VAR) and not os.getenv(HOME_ENV_VAR):
        return (
            f"Ignoring deprecated {LEGACY_HOME_ENV_VAR} environment variable "
            f"(a Mistral Vibe holdover). privibe is using {_get_vibe_home()}; "
            f"set {HOME_ENV_VAR} to override its home directory."
        )
    return None


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
