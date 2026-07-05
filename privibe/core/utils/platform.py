from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
import os
from pathlib import Path
import shutil
import subprocess
import sys

# Standard Git for Windows install locations. Mirrored into BashToolConfig's
# bash_search_paths default so they ship visibly in the user's config.toml;
# if the user removes the field entirely, this tuple is what applies.
DEFAULT_BASH_SEARCH_PATHS: tuple[str, ...] = (
    "C:\\Program Files\\Git\\bin\\bash.exe",
    "C:\\Program Files (x86)\\Git\\bin\\bash.exe",
    "%LOCALAPPDATA%\\Programs\\Git\\bin\\bash.exe",
)

_SMOKE_TEST_TIMEOUT = 10


def is_windows() -> bool:
    return sys.platform == "win32"


def _is_wsl_bash(path: Path) -> bool:
    # C:\Windows\System32\bash.exe is the WSL launcher: it runs commands inside
    # a Linux VM with a different filesystem view (/mnt/c/...), not Git Bash.
    return "system32" in (part.lower() for part in path.parts)


def _bash_works(path: Path) -> bool:
    try:
        result = subprocess.run(
            [str(path), "-c", "echo ok"],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=_SMOKE_TEST_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and "ok" in result.stdout


def _candidates_from_entry(entry: str) -> list[Path]:
    expanded = Path(os.path.expandvars(entry))
    if expanded.suffix.lower() == ".exe":
        return [expanded]
    # Directory entry: prefer the bin\bash.exe wrapper — it bootstraps the
    # MSYS PATH so ls/grep resolve even when Git's usr/bin is not on the
    # Windows PATH. A bare bash.exe directly inside is the fallback.
    return [expanded / "bin" / "bash.exe", expanded / "bash.exe"]


def _candidates_from_git() -> list[Path]:
    git = shutil.which("git")
    if not git:
        return []
    git_path = Path(git).resolve()
    # git.exe lives at <root>\cmd\git.exe, <root>\bin\git.exe, or
    # <root>\mingw64\bin\git.exe — try both plausible roots.
    candidates = []
    for root in (git_path.parent.parent, git_path.parent.parent.parent):
        candidate = root / "bin" / "bash.exe"
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


@lru_cache(maxsize=8)
def _resolve_windows_bash_cached(search_paths: tuple[str, ...]) -> str | None:
    checked: list[str] = []

    def try_candidate(path: Path) -> str | None:
        checked.append(str(path))
        if path.is_file() and _bash_works(path):
            return str(path)
        return None

    # Static paths (config or defaults) are trusted as-is — no WSL rejection;
    # an explicitly configured path is the user saying "I know what I'm doing".
    for entry in search_paths:
        for candidate in _candidates_from_entry(entry):
            if found := try_candidate(candidate):
                return found

    for candidate in _candidates_from_git():
        if not _is_wsl_bash(candidate) and (found := try_candidate(candidate)):
            return found

    if which_bash := shutil.which("bash"):
        candidate = Path(which_bash)
        if not _is_wsl_bash(candidate) and (found := try_candidate(candidate)):
            return found

    # Lazy import: logger pulls in privibe.core.paths and this module must
    # stay importable from anywhere without circular-import risk.
    from privibe.core.logger import logger

    logger.warning(
        "Git Bash not found (checked: %s); the bash tool will run commands "
        "via cmd.exe. Fine-grained command permissions are unavailable in "
        "this mode. Set bash_search_paths under [tools.bash] in config.toml "
        "to point at your Git installation.",
        ", ".join(checked) if checked else "no candidates",
    )
    return None


def resolve_windows_bash(search_paths: Sequence[str] | None = None) -> str | None:
    """Locate a working Git Bash on Windows; None elsewhere or when not found.

    Walks the static search paths first (entries may be a bash.exe path or a
    Git install directory), then derives candidates from the git executable on
    PATH, then falls back to `which bash` — rejecting the WSL launcher in
    System32 for the auto-detected steps. Each candidate is smoke-tested with
    `bash -c "echo ok"`. The result is cached per search-path list, so the
    smoke test runs once per process.

    `search_paths=None` means "use DEFAULT_BASH_SEARCH_PATHS"; an empty
    sequence is an explicit opt-out of the static path search.
    """
    if not is_windows():
        return None
    if search_paths is None:
        search_paths = DEFAULT_BASH_SEARCH_PATHS
    return _resolve_windows_bash_cached(tuple(search_paths))
