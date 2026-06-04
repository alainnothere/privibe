from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import anyio
from pydantic import BaseModel, Field

from privibe.core.tools.base import ToolError
from privibe.core.tools.builtins.hashed_read import _line_hash, format_hashed_lines
from privibe.core.tools.utils import normalization_note, normalize_tool_path
from privibe.core.utils.io import read_safe_async

_ERROR_CONTEXT_LINES = 2
_SUCCESS_CONTEXT_LINES = 5

# Matches a leaked hashed_read line prefix at the start of a replacement line,
# e.g. "   11 b1c4  ": optional left-pad, the line number, one space, the 4-char
# hex hash, then exactly two spaces. Kept strict so it only fires on the literal
# read format, not on incidentally-similar content.
_LEAKED_PREFIX_RE = re.compile(r"^ *\d+ [0-9a-f]{4}  ")


def strip_leaked_prefix(new_content: str) -> tuple[str, int]:
    """Remove an accidentally-copied hashed_read prefix from each line.

    Returns the cleaned content and how many lines had a prefix removed.
    """
    out: list[str] = []
    stripped = 0
    for line in new_content.splitlines():
        match = _LEAKED_PREFIX_RE.match(line)
        if match:
            out.append(line[match.end() :])
            stripped += 1
        else:
            out.append(line)
    return "\n".join(out), stripped


class LineReplacement(BaseModel):
    line: int = Field(description="1-based line number from hashed_read.")
    hash: str = Field(description="4-char hash from hashed_read for that line.")
    new_content: str = Field(
        description="Replacement text. May be multiline. Do not include a trailing newline."
    )
    end_line: int | None = Field(
        default=None,
        description="Last line of the range to replace (1-based, inclusive). Omit for single-line.",
    )
    end_hash: str | None = Field(
        default=None,
        description="Hash of end_line from hashed_read. Required when end_line is provided.",
    )


@dataclass
class ApplyResult:
    path: str
    total_ops: int
    total_lines_changed: int
    context: str
    path_note: str | None
    content_note: str | None = None


def resolve_file_path(path_str: str) -> Path:
    if not path_str.strip():
        raise ToolError("Path cannot be empty")
    resolved = normalize_tool_path(path_str).resolve()
    if not resolved.exists():
        raise ToolError(f"File not found: {path_str}")
    if resolved.is_dir():
        raise ToolError(f"Path is a directory: {path_str}")
    return resolved


async def read_file_lines(file_path: Path) -> list[str]:
    try:
        content = await read_safe_async(file_path, raise_on_error=True)
        lines = content.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        return lines
    except OSError as exc:
        raise ToolError(f"Error reading {file_path}: {exc}") from exc


async def write_file_lines(file_path: Path, lines: list[str]) -> None:
    try:
        async with await anyio.Path(file_path).open(mode="w", encoding="utf-8") as f:
            await f.write("".join(lines))
    except OSError as exc:
        raise ToolError(f"Error writing {file_path}: {exc}") from exc


def build_replacement_lines(new_content: str) -> list[str]:
    if not new_content:
        return []
    return [line + "\n" for line in new_content.splitlines()]


def _context_around(file_lines: list[str], idx: int) -> str:
    start = max(0, idx - _ERROR_CONTEXT_LINES)
    end = min(len(file_lines), idx + _ERROR_CONTEXT_LINES + 1)
    return format_hashed_lines(file_lines[start:end], start + 1)


def resolve_replacements(
    replacements: list[LineReplacement],
    file_lines: list[str],
) -> list[tuple[int, int, LineReplacement]]:
    total = len(file_lines)
    resolved: list[tuple[int, int, LineReplacement]] = []
    for r in replacements:
        if r.line < 1 or r.line > total:
            raise ToolError(f"Line {r.line} is out of range (file has {total} lines).")
        if r.end_line is not None and r.end_hash is None:
            raise ToolError(f"end_hash is required when end_line is set (line {r.line}).")
        start_idx = r.line - 1
        end_idx = (r.end_line - 1) if r.end_line is not None else start_idx
        if end_idx >= total:
            raise ToolError(
                f"end_line {r.end_line} is out of range (file has {total} lines)."
            )
        if end_idx < start_idx:
            raise ToolError(
                f"end_line ({r.end_line}) must be >= line ({r.line})."
            )
        resolved.append((start_idx, end_idx, r))

    sorted_by_start = sorted(resolved, key=lambda x: x[0])
    for i in range(len(sorted_by_start) - 1):
        _, end_a, r_a = sorted_by_start[i]
        start_b, _, r_b = sorted_by_start[i + 1]
        if start_b <= end_a:
            raise ToolError(
                f"Replacements overlap: line {r_a.line} and line {r_b.line} target the same region."
            )
    return resolved


def validate_all_hashes(
    resolved: list[tuple[int, int, LineReplacement]],
    file_lines: list[str],
) -> None:
    errors: list[str] = []
    for start_idx, end_idx, r in resolved:
        start_content = file_lines[start_idx].rstrip("\r\n")
        actual = _line_hash(start_content)
        if actual != r.hash:
            context = _context_around(file_lines, start_idx)
            errors.append(
                f"Hash mismatch at line {r.line}: expected {r.hash!r}, got {actual!r}.\n"
                f"Current content:\n{context}"
            )
        if r.end_line is not None and r.end_hash is not None:
            end_content = file_lines[end_idx].rstrip("\r\n")
            actual_end = _line_hash(end_content)
            if actual_end != r.end_hash:
                context = _context_around(file_lines, end_idx)
                errors.append(
                    f"Hash mismatch at end_line {r.end_line}: expected {r.end_hash!r}, got {actual_end!r}.\n"
                    f"Current content:\n{context}"
                )
    if errors:
        raise ToolError(
            "File may have changed since last hashed_read. Re-read and retry.\n\n"
            + "\n\n".join(errors)
        )


def prepare_replacements(
    resolved: list[tuple[int, int, LineReplacement]],
    file_lines: list[str],
    *,
    allow_literal: bool,
    keep_duplicate: bool,
) -> tuple[list[tuple[int, int, LineReplacement, list[str]]], list[str]]:
    """Turn each replacement's new_content into the lines to splice, applying
    the two hallucination corrections and recording what was done.

    - Leaked hashed_read prefixes are stripped from new_content (unless
      ``allow_literal``).
    - A first/last new line that exactly duplicates the untouched original line
      immediately outside the region is dropped (unless ``keep_duplicate``).
      Only edit-induced boundary duplicates are touched: neighbours that are
      themselves being edited in this batch are left alone, and duplicates that
      already existed or live inside new_content are never removed.
    """
    covered: set[int] = set()
    for start_idx, end_idx, _ in resolved:
        covered.update(range(start_idx, end_idx + 1))

    total = len(file_lines)
    prepared: list[tuple[int, int, LineReplacement, list[str]]] = []
    notes: list[str] = []

    for start_idx, end_idx, r in resolved:
        content = r.new_content
        if not allow_literal:
            content, stripped = strip_leaked_prefix(content)
            if stripped:
                notes.append(
                    f"line {r.line}: stripped a hashed_read prefix from {stripped} "
                    f"replacement line{'s' if stripped != 1 else ''} "
                    "(pass allow_literal=true to keep it verbatim)"
                )

        lines = build_replacement_lines(content)

        if not keep_duplicate and lines:
            before = start_idx - 1
            if before >= 0 and before not in covered and lines[0] == file_lines[before]:
                lines.pop(0)
                notes.append(
                    f"line {r.line}: removed a new line that duplicated the line "
                    "immediately before it (pass keep_duplicate=true to keep it)"
                )
            after = end_idx + 1
            if (
                lines
                and after < total
                and after not in covered
                and lines[-1] == file_lines[after]
            ):
                lines.pop()
                notes.append(
                    f"line {r.line}: removed a new line that duplicated the line "
                    "immediately after it (pass keep_duplicate=true to keep it)"
                )

        prepared.append((start_idx, end_idx, r, lines))

    return prepared, notes


def build_success_context(
    new_lines: list[str],
    prepared_asc: list[tuple[int, int, LineReplacement, list[str]]],
) -> str:
    regions: list[tuple[int, int]] = []
    offset = 0
    for start_idx, end_idx, _r, lines in prepared_asc:
        replacement_lines_count = len(lines)
        new_start = start_idx + offset
        new_end = (
            new_start + replacement_lines_count - 1
            if replacement_lines_count
            else new_start - 1
        )
        regions.append((new_start, new_end))
        offset += replacement_lines_count - (end_idx - start_idx + 1)

    total = len(new_lines)
    windows: list[tuple[int, int]] = [
        (max(0, ns - _SUCCESS_CONTEXT_LINES), min(total - 1, ne + _SUCCESS_CONTEXT_LINES))
        for ns, ne in regions
    ]

    merged: list[list[int]] = []
    for ws, we in sorted(windows):
        if merged and ws <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], we)
        else:
            merged.append([ws, we])

    parts: list[str] = []
    for i, (ws, we) in enumerate(merged):
        if i > 0:
            parts.append("...")
        parts.append(format_hashed_lines(new_lines[ws : we + 1], ws + 1))
    return "\n".join(parts)


async def apply_replacements_to_file(
    path_str: str,
    replacements: list[LineReplacement],
    *,
    allow_literal: bool = False,
    keep_duplicate: bool = False,
) -> ApplyResult:
    file_path = resolve_file_path(path_str)
    file_lines = await read_file_lines(file_path)

    resolved = resolve_replacements(replacements, file_lines)
    validate_all_hashes(resolved, file_lines)

    prepared, notes = prepare_replacements(
        resolved,
        file_lines,
        allow_literal=allow_literal,
        keep_duplicate=keep_duplicate,
    )

    prepared_asc = sorted(prepared, key=lambda p: p[0])
    prepared_desc = list(reversed(prepared_asc))

    new_lines = list(file_lines)
    total_lines_changed = 0
    for start_idx, end_idx, _replacement, replacement_lines in prepared_desc:
        new_lines[start_idx : end_idx + 1] = replacement_lines
        total_lines_changed += end_idx - start_idx + 1

    await write_file_lines(file_path, new_lines)
    context = build_success_context(new_lines, prepared_asc)
    return ApplyResult(
        path=str(file_path),
        total_ops=len(replacements),
        total_lines_changed=total_lines_changed,
        context=context,
        path_note=normalization_note(path_str, file_path),
        content_note="\n".join(notes) if notes else None,
    )
