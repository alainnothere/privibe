"""Detect the current path dialect (Windows native, Git Bash, WSL, Cygwin, POSIX)
and translate between drive-letter forms so file tools accept whichever shape
the model picks.

Translation is best-effort: we only rewrite when the input clearly does not
exist in its raw form on the running interpreter, and a translated form does.

The translation pipeline can be configured via configure_path_translation():
- enabled=False makes translate_path() a pass-through. Aliases are also
  disabled in that mode (they feed into the same pipeline; one knob controls
  both). dialect_hint() still emits the environment fact — telling the model
  where Windows drives live is independent of whether we translate for it.
- aliases is a dict of prefix -> replacement applied BEFORE auto-rules.
  Longest matching prefix wins. Used for unusual mounts the auto-detector
  cannot probe its way to via the filesystem.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
import os
from pathlib import Path
import re
import sys


class PathDialect(StrEnum):
    WINDOWS_NATIVE = "windows_native"
    GIT_BASH = "git_bash"
    WSL = "wsl"
    CYGWIN = "cygwin"
    POSIX = "posix"


_DRIVE_LETTER = re.compile(r"^([A-Za-z]):[\\/](.*)$")
_SLASH_DRIVE = re.compile(r"^/([A-Za-z])(/.*)?$")
_MNT_DRIVE = re.compile(r"^/mnt/([A-Za-z])(/.*)?$")
_CYGDRIVE = re.compile(r"^/cygdrive/([A-Za-z])(/.*)?$")


def _detect() -> PathDialect:
    """Probe environment + filesystem to classify the running shell."""
    if sys.platform == "win32":
        if os.environ.get("MSYSTEM"):
            return PathDialect.GIT_BASH
        return PathDialect.WINDOWS_NATIVE

    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return PathDialect.WSL
    if sys.platform == "linux" and os.path.isdir("/mnt/c"):
        return PathDialect.WSL
    if sys.platform.startswith("cygwin") or os.environ.get("CYGWIN"):
        return PathDialect.CYGWIN
    if os.path.isdir("/cygdrive/c"):
        return PathDialect.CYGWIN

    return PathDialect.POSIX


@lru_cache(maxsize=1)
def detect_path_dialect() -> PathDialect:
    """Return the cached path dialect for this process.

    Eagerly call once at startup (the system prompt depends on this) and the
    rest of the codebase reuses the cached result.
    """
    return _detect()


def reset_dialect_cache() -> None:
    """Test hook: clear the cached dialect so the next call re-detects."""
    detect_path_dialect.cache_clear()


# Module-level translation config, populated by configure_path_translation().
# Defaults are "enabled with no aliases" so tests and tools that don't load
# VibeConfig still get the auto-translation behaviour.
_translation_enabled: bool = True
_aliases: dict[str, str] = {}


def configure_path_translation(
    *, enabled: bool, aliases: dict[str, str] | None = None
) -> None:
    """Apply VibeConfig.paths to the translation pipeline.

    Called once at startup after VibeConfig.load(). Subsequent calls override
    earlier settings — useful for tests.
    """
    global _translation_enabled, _aliases
    _translation_enabled = enabled
    _aliases = dict(aliases or {})


def reset_translation_config() -> None:
    """Test hook: restore the default 'enabled, no aliases' state."""
    configure_path_translation(enabled=True, aliases={})


def _apply_aliases(path_str: str) -> str:
    """Replace the longest matching alias prefix; first call returns input
    unchanged when no alias matches."""
    if not _aliases:
        return path_str
    best: str | None = None
    for prefix in _aliases:
        if path_str == prefix or path_str.startswith(prefix):
            if best is None or len(prefix) > len(best):
                best = prefix
    if best is None:
        return path_str
    return _aliases[best] + path_str[len(best):]


def _split_drive(path_str: str) -> tuple[str, str] | None:
    """Match any of the known drive-letter forms.

    Returns (drive_letter_lower, posix_rest_with_leading_slash) or None.
    """
    if m := _DRIVE_LETTER.match(path_str):
        rest = m.group(2).replace("\\", "/")
        return m.group(1).lower(), "/" + rest if rest else ""
    if m := _MNT_DRIVE.match(path_str):
        return m.group(1).lower(), m.group(2) or ""
    if m := _CYGDRIVE.match(path_str):
        return m.group(1).lower(), m.group(2) or ""
    if m := _SLASH_DRIVE.match(path_str):
        # Bare /c, /c/, /c/foo — only treat as drive if next char is "/" or end.
        # The regex already enforces that.
        return m.group(1).lower(), m.group(2) or ""
    return None


def _to_form(drive: str, rest: str, dialect: PathDialect) -> str:
    """Render (drive, rest) into the canonical form for the given dialect."""
    rest = rest.lstrip("/")
    if dialect == PathDialect.WINDOWS_NATIVE:
        sep = "\\"
        body = rest.replace("/", sep)
        return f"{drive.upper()}:{sep}{body}" if body else f"{drive.upper()}:{sep}"
    if dialect == PathDialect.GIT_BASH:
        # Git Bash + native Python: backslash form is what pathlib understands.
        sep = "\\"
        body = rest.replace("/", sep)
        return f"{drive.upper()}:{sep}{body}" if body else f"{drive.upper()}:{sep}"
    if dialect == PathDialect.WSL:
        return f"/mnt/{drive}/{rest}" if rest else f"/mnt/{drive}"
    if dialect == PathDialect.CYGWIN:
        return f"/cygdrive/{drive}/{rest}" if rest else f"/cygdrive/{drive}"
    # POSIX: there's no drive concept, so the most useful recovery for a
    # Windows-shaped path like `c:\home\user\foo` is to drop the drive and
    # treat the rest as an absolute POSIX path. Backslashes have already
    # been normalised to forward slashes by _split_drive. The existence
    # check in translate_path still gates this — a real /c directory keeps
    # its raw form so we don't mangle legitimate paths.
    return f"/{rest}" if rest else "/"


def translate_path(path_str: str, dialect: PathDialect | None = None) -> str:
    """Translate a Windows-style or unix-mounted path to the running dialect.

    Leaves the input alone when:
    - translation is disabled in VibeConfig.paths (one knob disables both
      auto-rules and aliases — see configure_path_translation), or
    - the input has no recognizable drive prefix and no alias matches, or
    - the raw form already exists on disk (don't be too clever), or
    - the dialect is POSIX (no Windows drives to translate).
    """
    if not path_str:
        return path_str
    if not _translation_enabled:
        return path_str

    # Aliases run first: the substituted form then flows through auto-rules,
    # so a user mapping like "/weird-mount" -> "C:\\weird-mount" still gets
    # converted to /mnt/c/weird-mount automatically when running on WSL.
    aliased = _apply_aliases(path_str)

    if dialect is None:
        dialect = detect_path_dialect()

    if dialect == PathDialect.POSIX:
        # On POSIX, only recover unambiguously Windows-shaped inputs (drive
        # letter + colon, e.g. `c:\foo` or `c:/foo`). Slash-drive forms like
        # `/c/foo` or `/B/x` are valid POSIX paths in their own right — most
        # of the time they're real directories or alias outputs, not Windows
        # paths in disguise. Treating them as drives causes false rewrites.
        m = _DRIVE_LETTER.match(aliased)
        if m is None:
            return aliased
        rest = m.group(2).replace("\\", "/")
        canonical = f"/{rest}" if rest else "/"
        if os.path.lexists(aliased):
            return aliased
        return canonical

    split = _split_drive(aliased)
    if split is None:
        return aliased
    drive, rest = split

    canonical = _to_form(drive, rest, dialect)
    if canonical == aliased:
        return aliased

    # Only rewrite if the raw form does NOT exist but the translated one does,
    # OR the raw form is in a shape that pathlib on this platform can't resolve.
    raw_exists = os.path.lexists(aliased)
    if raw_exists:
        return aliased

    canonical_exists = os.path.lexists(canonical)
    if canonical_exists:
        return canonical

    # Neither exists: still rewrite into the canonical form so the eventual
    # "not found" error message uses the dialect's natural shape (matches what
    # the user / model will see in tracebacks and terminals).
    return canonical


def to_posix_for_match(path_str: str) -> str:
    """Return path with forward slashes, suitable for PurePosixPath glob matching.

    Drive letter `C:\\foo` becomes `/c/foo` so `**/*.env` patterns still match.
    """
    split = _split_drive(path_str)
    if split is not None:
        drive, rest = split
        return f"/{drive}{rest or ''}"
    return path_str.replace("\\", "/")


def normalize_to_path(path_str: str) -> Path:
    """Translate then return a Path. Handles ~ expansion and relative paths.

    The returned path is NOT yet resolved — callers that need .resolve() should
    do it themselves (some tools use it for security checks, others don't).
    """
    translated = translate_path(path_str)
    p = Path(translated).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def dialect_hint() -> str | None:
    """One-line system-prompt hint about the local filesystem layout.

    Always emitted when the dialect is not pure POSIX, regardless of the
    translation flag — the model still needs to know where Windows drives
    live so it picks the canonical form on its own. The per-call path_note
    on tool results does the teaching when a wrong path comes in; the system
    prompt's job is just stating the environment.
    """
    dialect = detect_path_dialect()
    if dialect == PathDialect.WINDOWS_NATIVE:
        return "Filesystem: Windows. Use `C:\\path\\to\\file` for absolute paths."
    if dialect == PathDialect.GIT_BASH:
        return (
            "Filesystem: Windows + Git Bash. Use `C:\\path\\to\\file` or "
            "`c:/path` for absolute paths."
        )
    if dialect == PathDialect.WSL:
        return "Filesystem: WSL. Use `/mnt/c/path/to/file` for Windows drives."
    if dialect == PathDialect.CYGWIN:
        return "Filesystem: Cygwin. Use `/cygdrive/c/path` for Windows drives."
    return None
