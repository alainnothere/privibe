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


def line_range(start_line: int, lines_read: int) -> str:
    """`lines 100 to 120 (21 lines)` for a read's result line; `0 lines` when
    nothing came back so the range never lies about a line that isn't there."""
    if lines_read <= 0:
        return "0 lines"
    if lines_read == 1:
        return f"line {start_line}"
    return f"lines {start_line} to {start_line + lines_read - 1} ({lines_read} lines)"


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


def is_protected_path(path_str: str, protected_paths: list[str]) -> bool:
    """Return True when the path is a protected entry or lives inside one.

    Plain entries protect the path itself and its whole subtree; entries
    containing glob characters are matched with fnmatch against the resolved
    absolute path. Entries may use ~ and cross-dialect drive forms.
    """
    if not protected_paths:
        return False

    resolved = _make_absolute(path_str).resolve()
    resolved_str = str(resolved)

    for entry in protected_paths:
        if any(ch in entry for ch in "*?["):
            if fnmatch.fnmatch(resolved_str, os.path.expanduser(entry)):
                return True
            continue
        entry_resolved = _make_absolute(entry).resolve()
        if resolved == entry_resolved or entry_resolved in resolved.parents:
            return True
    return False


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
    protected_paths: list[str] | None = None,
    protect_outside_workdir: bool = False,
    outside_workdir_exempt: list[str] | None = None,
) -> PermissionContext | None:
    """Resolve permission for a file-based tool invocation.

    Checks protected paths, then allowlist/denylist, then sensitive patterns,
    then workdir boundary. Returns PermissionContext with granular
    required_permissions when applicable.
    """
    if protected_paths and is_protected_path(path_str, protected_paths):
        return PermissionContext(
            permission=ToolPermission.NEVER,
            reason=(
                f"Access denied: the path is protected by configuration "
                f"({tool_name}). Do not attempt to access it."
            ),
        )

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
        # With protect_outside_workdir on, a non-exempt outside path escalates:
        # the ask must reach a human even under auto_approve. Exempt entries
        # share is_protected_path's matching semantics: plain entries cover
        # their subtree, glob entries go through fnmatch.
        escalated = protect_outside_workdir and not is_protected_path(
            path_str, outside_workdir_exempt or []
        )
        resolved = file_path.resolve()
        # A directory target globs its own subtree, not its parent's — so an
        # approval for reading a file in a dir also covers grepping the dir.
        base_dir = resolved if resolved.is_dir() else resolved.parent
        glob = str(Path(base_dir) / "*")
        required.append(
            RequiredPermission(
                scope=PermissionScope.OUTSIDE_DIRECTORY,
                invocation_pattern=glob,
                session_pattern=glob,
                label=f"outside workdir ({glob})",
                escalated=escalated,
            )
        )

    if required:
        return PermissionContext(
            permission=ToolPermission.ASK, required_permissions=required
        )

    return None


def large_file_advisory(
    size_bytes: int, preview_bytes: int, preview_lines: int, threshold_kb: int
) -> str:
    """Message returned with the head preview of an over-threshold naive read.

    Delivered in the tool result (not the prompt) because that is the moment
    the model is about to page through a huge file, and result-level notes
    are what actually change its next step.
    """
    size_kb = size_bytes / 1024
    estimate = ""
    if preview_lines > 0 and preview_bytes > 0:
        est_total = int(size_bytes / (preview_bytes / preview_lines))
        estimate = f", roughly {est_total:,} lines"
    return (
        f"LARGE FILE: this file is {size_kb:,.0f} KB{estimate} - only the "
        f"first {preview_lines} lines are shown (the file exceeds the "
        f"{threshold_kb} KB large-file threshold). Do NOT page through the "
        "rest sequentially; that floods the context. Instead: locate what "
        "you need with grep or find_symbol, then read only the relevant "
        "ranges by passing start_line and limit."
    )
