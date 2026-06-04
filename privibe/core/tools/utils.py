from __future__ import annotations

import fnmatch
import os
from pathlib import Path, PurePosixPath

from privibe.core.paths.dialect import to_posix_for_match, translate_path
from privibe.core.tools.base import ToolPermission
from privibe.core.tools.permissions import (
    PermissionContext,
    PermissionScope,
    RequiredPermission,
)


def wildcard_match(text: str, pattern: str) -> bool:
    """Match text against a wildcard pattern using fnmatch.

    If pattern ends with " *", trailing part is optional (matches with or without args).
    """
    if fnmatch.fnmatch(text, pattern):
        return True
    if pattern.endswith(" *") and fnmatch.fnmatch(text, pattern[:-2]):
        return True
    return False


def _make_absolute(path_str: str) -> Path:
    """Translate cross-dialect drive letters then expand + absolutize.

    This is the single entry point all file tools use to interpret a model-
    supplied path. It does NOT call .resolve() — callers do that themselves
    when they need symlink + canonicalization, since some places only want the
    pre-resolve form (snapshots, error messages).
    """
    path = Path(translate_path(path_str)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def normalize_tool_path(path_str: str) -> Path:
    """Public alias for tools that previously inlined the expanduser/cwd dance."""
    return _make_absolute(path_str)


def display_path(path_str: str) -> str:
    """A short, readable form of a path for tool result messages.

    Relative to the current working directory when the file lives under it,
    otherwise the path as given. Keeps result lines from showing long absolute
    paths while still identifying where the file is.
    """
    try:
        rel = os.path.relpath(path_str)
    except ValueError:
        return path_str
    return path_str if rel.startswith("..") else rel


def normalization_note(original: str, resolved: Path) -> str | None:
    """Return a one-line hint for the model when the input path was rewritten.

    Compares the model's raw input against the canonical absolute form. We
    only emit the note when they differ in a way the model can learn from
    (drive-letter dialect translation), not for trivial differences like
    relative-to-absolute promotion.
    """
    if not original:
        return None
    translated = translate_path(original)
    if translated == original:
        return None
    return (
        f"Note: input path '{original}' was translated to "
        f"'{resolved}'. Use that form next time to avoid retries."
    )


def resolve_path_permission(
    path_str: str, *, allowlist: list[str], denylist: list[str]
) -> PermissionContext | None:
    """Resolve permission for a file path against glob patterns.

    Returns NEVER on denylist match, ALWAYS on allowlist match, None otherwise.
    """
    file_str = str(_make_absolute(path_str).resolve())

    for pattern in denylist:
        if fnmatch.fnmatch(file_str, pattern):
            return PermissionContext(permission=ToolPermission.NEVER)

    for pattern in allowlist:
        if fnmatch.fnmatch(file_str, pattern):
            return PermissionContext(permission=ToolPermission.ALWAYS)

    return None


def is_path_within_workdir(path_str: str) -> bool:
    """Return True if the resolved path is inside cwd.

    Both sides are run through translate_path first so a model-supplied
    `/c/repo/foo` is compared against a `C:\\repo` cwd as the same root.
    """
    try:
        _make_absolute(path_str).resolve().relative_to(Path.cwd().resolve())
        return True
    except ValueError:
        return False


def resolve_file_tool_permission(
    path_str: str,
    *,
    tool_name: str,
    allowlist: list[str],
    denylist: list[str],
    config_permission: ToolPermission,
    sensitive_patterns: list[str],
) -> PermissionContext | None:
    """Resolve permission for a file-based tool invocation.

    Checks allowlist/denylist, then sensitive patterns, then workdir boundary.
    Returns PermissionContext with granular required_permissions when applicable.
    """
    if (
        result := resolve_path_permission(
            path_str, allowlist=allowlist, denylist=denylist
        )
    ) is not None:
        return result

    required: list[RequiredPermission] = []

    file_path = _make_absolute(path_str)
    file_str = str(file_path.resolve())
    posix_for_glob = to_posix_for_match(file_str)

    for pattern in sensitive_patterns:
        if PurePosixPath(posix_for_glob).match(pattern):
            required.append(
                RequiredPermission(
                    scope=PermissionScope.FILE_PATTERN,
                    invocation_pattern=file_path.name,
                    session_pattern="*",
                    label=f"accessing sensitive files ({tool_name})",
                )
            )
            break

    if not is_path_within_workdir(path_str):
        if config_permission == ToolPermission.NEVER:
            return PermissionContext(permission=ToolPermission.NEVER)
        resolved = file_path.resolve()
        parent_dir = str(resolved.parent)
        glob = str(Path(parent_dir) / "*")
        required.append(
            RequiredPermission(
                scope=PermissionScope.OUTSIDE_DIRECTORY,
                invocation_pattern=glob,
                session_pattern=glob,
                label=f"outside workdir ({glob})",
            )
        )

    if required:
        return PermissionContext(
            permission=ToolPermission.ASK, required_permissions=required
        )

    return None
