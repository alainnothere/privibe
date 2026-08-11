from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import zlib

import anyio
from pydantic import BaseModel, Field

from privibe.core.tools.base import ToolError
from privibe.core.tools.builtins.hashed_read import _line_hash, format_hashed_lines
from privibe.core.tools.utils import normalization_note, normalize_tool_path
from privibe.core.utils.io import read_safe_async

_ERROR_CONTEXT_LINES = 2
_SUCCESS_CONTEXT_LINES = 5

# Matches a leaked hashed_read line prefix at the start of a replacement line,
# e.g. "11|b1c4|": the line number, a pipe, the 4-char hex hash, a pipe. Kept
# strict so it only fires on the literal read format, not on incidentally-
# similar content. The second alternative is the pre-pipe space-padded format
# ("   11 b1c4  "); resumed sessions carry it in their read history and the
# model may paste it back, so it stays strippable.
_LEAKED_PREFIX_RE = re.compile(r"^(?:\d+\|[0-9a-f]{4}\|| *\d+ [0-9a-f]{4}  )")


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


_INDENT_UNIT_FALLBACK = 4
_INDENT_CONTEXT_LINES = 5
# Ordered: "///" must win over "//". "*" is handled separately because bare
# "*word" is usually code (pointers, unpacking), not a comment continuation.
_COMMENT_MARKERS = ("///", "//", "#")


def _indent_of(content: str) -> str:
    return content[: len(content) - len(content.lstrip(" \t"))]


def _comment_marker(content: str) -> str | None:
    body = content.lstrip(" \t")
    for marker in _COMMENT_MARKERS:
        if body.startswith(marker):
            return marker
    if body == "*" or body.startswith("* ") or body.startswith("*/"):
        return "*"
    return None


_CHAIN_WALK_LIMIT = 30


def _is_dot_continuation(content: str) -> bool:
    """True for a method-chain continuation line: leading "." then an
    identifier (.HasComment, .Where). Excludes ranges (..) and decimals (.5)."""
    body = content.lstrip(" \t")
    return (
        len(body) > 1
        and body[0] == "."
        and (body[1].isalpha() or body[1] == "_")
    )


def _chain_context(
    contents: list[str],
    idx: int,
    file_lines: list[str],
    start_idx: int,
    covered: set[int],
) -> tuple[str | None, str | None]:
    """(sibling indent, head indent) for the chain containing contents[idx].

    Sibling: the nearest preceding "." line in the same run. Head: the
    non-"." statement line that opens the chain. Searches the replacement
    lines first, then continues into the untouched file lines above the
    region (bounded walk; stops at lines edited by another replacement).
    Returns None for whichever was not found.
    """
    sibling: str | None = None
    for j in range(idx - 1, -1, -1):
        c = contents[j]
        if not c.strip():
            continue
        if _is_dot_continuation(c):
            if sibling is None:
                sibling = _indent_of(c)
        else:
            return sibling, _indent_of(c)
    for i in range(start_idx - 1, max(-1, start_idx - 1 - _CHAIN_WALK_LIMIT), -1):
        if i in covered:
            return sibling, None
        c = file_lines[i].rstrip("\r\n")
        if not c.strip():
            continue
        if _is_dot_continuation(c):
            if sibling is None:
                sibling = _indent_of(c)
        else:
            return sibling, _indent_of(c)
    return sibling, None


def _infer_indent_unit(file_lines: list[str], start_idx: int, end_idx: int) -> int:
    """Most common indent step among the nearby lines, defaulting to 4.

    Only used to decide whether a base-indent delta is on the file's grid;
    a wrong guess degrades to not correcting, never to a bad correction of
    interior structure.
    """

    def nearby_widths(indices: range) -> list[int]:
        widths: list[int] = []
        for i in indices:
            content = file_lines[i].rstrip("\r\n")
            if not content.strip():
                continue
            indent = _indent_of(content)
            if "\t" in indent:
                continue
            widths.append(len(indent))
            if len(widths) == _INDENT_CONTEXT_LINES:
                break
        return widths

    before = nearby_widths(range(start_idx - 1, -1, -1))
    before.reverse()
    after = nearby_widths(range(end_idx + 1, len(file_lines)))

    diffs: dict[int, int] = {}
    prev: int | None = None
    for width in before + after:
        if prev is not None and width != prev:
            step = abs(width - prev)
            if 1 <= step <= 8:
                diffs[step] = diffs.get(step, 0) + 1
        prev = width
    if not diffs:
        return _INDENT_UNIT_FALLBACK
    return min(diffs, key=lambda d: (-diffs[d], d))


def correct_indentation(
    lines: list[str],
    file_lines: list[str],
    start_idx: int,
    end_idx: int,
    covered: set[int],
    line_no: int,
) -> tuple[list[str], list[str]]:
    """Best-effort indent correction of replacement lines.

    Two obvious-mistake corrections, nothing cleverer: a uniform comment
    block snaps to the adjacent comment's indent, and a base indent sitting
    off the file's indent grid is shifted onto it. Relative indentation
    between the new lines is always preserved as written; nesting is never
    inferred from braces. Tab-indented context is left untouched.
    """
    notes: list[str] = []
    contents = [line.rstrip("\r\n") for line in lines]
    non_blank = [c for c in contents if c.strip()]
    if not non_blank:
        return lines, notes

    anchor = file_lines[start_idx].rstrip("\r\n")
    anchor_indent = _indent_of(anchor)
    if "\t" in anchor_indent or any("\t" in _indent_of(c) for c in non_blank):
        notes.append(
            f"line {line_no}: indent correction skipped (tab-indented context or content)"
        )
        return lines, notes

    # A block of same-marker comment lines aligns to the adjacent comment it
    # extends: comment lines at differing indents within one block are
    # essentially never intentional. Preference order: the untouched line
    # above (extending an existing block), the line being replaced, the
    # untouched line below.
    markers = {_comment_marker(c) for c in non_blank}
    if len(markers) == 1 and (marker := next(iter(markers))) is not None:
        target: str | None = None
        candidates = []
        if start_idx - 1 >= 0 and start_idx - 1 not in covered:
            candidates.append(file_lines[start_idx - 1].rstrip("\r\n"))
        candidates.append(anchor)
        if end_idx + 1 < len(file_lines) and end_idx + 1 not in covered:
            candidates.append(file_lines[end_idx + 1].rstrip("\r\n"))
        for candidate in candidates:
            if _comment_marker(candidate) == marker and "\t" not in _indent_of(candidate):
                target = _indent_of(candidate)
                break
        if target is not None:
            aligned = 0
            out = []
            for c in contents:
                if c.strip() and _indent_of(c) != target:
                    out.append(target + c.lstrip(" "))
                    aligned += 1
                else:
                    out.append(c)
            if aligned:
                notes.append(
                    f"line {line_no}: aligned {aligned} '{marker}' comment "
                    f"line{'s' if aligned != 1 else ''} to the adjacent comment's "
                    "indent (pass keep_indent=true to keep as written)"
                )
                return [c + "\n" for c in out], notes
            return lines, notes

    # A method-chain continuation line (leading ".") that has fallen to or
    # below its statement head's indent is a broken line, not a style: real
    # chains keep continuations deeper than their head and level with their
    # "." siblings. Snap such a line to the sibling's literal indent. Both
    # conditions are required; sibling-only would corrupt the legitimate
    # dedent of an outer chain resuming after a nested one.
    snapped = 0
    for i, c in enumerate(contents):
        if not c.strip() or not _is_dot_continuation(c):
            continue
        sibling, head = _chain_context(contents, i, file_lines, start_idx, covered)
        if sibling is None or head is None or "\t" in sibling or "\t" in head:
            continue
        indent = _indent_of(c)
        if len(indent) < len(sibling) and len(indent) <= len(head):
            contents[i] = sibling + c.lstrip(" ")
            snapped += 1
    if snapped:
        notes.append(
            f"line {line_no}: aligned {snapped} method-chain continuation "
            f"line{'s' if snapped != 1 else ''} ('.') to the chain's indent "
            "(pass keep_indent=true to keep as written)"
        )

    corrected = [c + "\n" for c in contents] if snapped else lines
    non_blank = [c for c in contents if c.strip()]
    base = min(len(_indent_of(c)) for c in non_blank)
    delta = len(anchor_indent) - base
    if delta == 0:
        return corrected, notes

    unit = _infer_indent_unit(file_lines, start_idx, end_idx)
    if delta % unit == 0:
        notes.append(
            f"line {line_no}: new content's base indent differs from the replaced "
            f"line's by {delta:+d} spaces; kept as written (a full indent step, "
            "assumed intentional)"
        )
        return corrected, notes

    # Off-grid delta: near-certain miscount. Round it to the nearest on-grid
    # delta (ties toward zero) and shift the whole block uniformly.
    lower = delta - (delta % unit)
    upper = lower + unit
    if delta - lower < upper - delta:
        rounded = lower
    elif upper - delta < delta - lower:
        rounded = upper
    else:
        rounded = lower if abs(lower) <= abs(upper) else upper
    adjustment = delta - rounded

    if adjustment < 0 and any(len(_indent_of(c)) < -adjustment for c in non_blank):
        notes.append(
            f"line {line_no}: indent correction skipped (a line would shift past column 0)"
        )
        return corrected, notes

    out = []
    for c in contents:
        if not c.strip():
            out.append(c)
        elif adjustment > 0:
            out.append(" " * adjustment + c)
        else:
            out.append(c[-adjustment:])
    notes.append(
        f"line {line_no}: shifted new content's indent by {adjustment:+d} "
        f"space{'s' if abs(adjustment) != 1 else ''} onto the {unit}-space indent "
        "grid relative to the replaced line (pass keep_indent=true to keep as written)"
    )
    return [c + "\n" for c in out], notes


# ---------------------------------------------------------------------------
# Shift map: per-file bookkeeping of the line-number shifts this session's
# hashed edits introduced, so a stale (line, hash) address from a pre-edit
# read can be deterministically re-pointed at its moved line instead of
# forcing a re-read/re-write round trip. Hashes are content-only, so an edit
# above a line changes nothing about its address except the number, and by
# exactly the delta recorded here. Translation is positional bookkeeping, not
# content search: it is only attempted when the given address fails, and only
# applied when the hash verifies at the translated position.
#
# Lifecycle: cleared when the model re-reads the file (hashed_read, which
# re-baselines its addresses to current coordinates), discarded when the file
# changes outside the hashed tools (fingerprint mismatch) or when a
# translation fails (the map has diverged), never persisted across sessions.
# ---------------------------------------------------------------------------

# (start_idx, end_idx, line_delta) of one applied edit, in the coordinates of
# the file as it was just before that edit's call.
_ShiftEdit = tuple[int, int, int]


@dataclass
class _ShiftMap:
    fingerprint: int
    # One generation per successful apply call; edits within a generation
    # share that call's pre-edit coordinates.
    generations: list[list[_ShiftEdit]] = field(default_factory=list)


_shift_maps: dict[str, _ShiftMap] = {}


def _content_fingerprint(lines: list[str]) -> int:
    return zlib.crc32("".join(lines).encode("utf-8"))


def reset_shift_map(path: str) -> None:
    """Forget recorded shifts for a file (keyed by resolved path string)."""
    _shift_maps.pop(path, None)


def clear_all_shift_maps() -> None:
    """Drop every recorded shift map. Test isolation hook."""
    _shift_maps.clear()


def _translate_interval(
    smap: _ShiftMap, start_idx: int, end_idx: int
) -> tuple[int, int] | None:
    """Walk [start_idx, end_idx] through the recorded generations.

    The interval shifts rigidly past edits entirely above it and ignores
    edits entirely below. Any recorded edit that intersects it means the
    addressed content itself was rewritten, so translation refuses (None)
    rather than risk silently overwriting an earlier edit. Returns None too
    when nothing moved: the given address was simply wrong, not stale.
    """
    cs, ce = start_idx, end_idx
    for generation in smap.generations:
        shift = 0
        for s, e, delta in generation:
            if e < cs:
                shift += delta
            elif s > ce:
                continue
            else:
                return None
        cs += shift
        ce += shift
    if (cs, ce) == (start_idx, end_idx):
        return None
    return cs, ce


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


def _address_ok(r: LineReplacement, start_idx: int, end_idx: int, file_lines: list[str]) -> bool:
    """True when the address is in range and its endpoint hashes match."""
    if start_idx < 0 or end_idx >= len(file_lines) or end_idx < start_idx:
        return False
    if _line_hash(file_lines[start_idx].rstrip("\r\n")) != r.hash:
        return False
    if r.end_line is not None and r.end_hash is not None:
        if _line_hash(file_lines[end_idx].rstrip("\r\n")) != r.end_hash:
            return False
    return True


def translate_stale_addresses(
    key: str,
    replacements: list[LineReplacement],
    file_lines: list[str],
) -> tuple[list[LineReplacement], list[str]]:
    """Re-point stale addresses at their shifted lines via the file's map.

    Addresses whose hashes match as given are current and pass untouched. A
    mismatched address is walked through the recorded shifts and adopted only
    if its hashes verify at the translated position, with a note. Any failure
    (fingerprint mismatch, translation refused, hash still wrong) discards the
    map and returns the replacements unchanged, so the normal validation error
    reports the lines the model actually sent and the re-read re-baselines.
    """
    smap = _shift_maps.get(key)
    if smap is None:
        return replacements, []
    if smap.fingerprint != _content_fingerprint(file_lines):
        # The file changed outside the hashed tools; the map is untrustworthy.
        reset_shift_map(key)
        return replacements, []

    out: list[LineReplacement] = []
    notes: list[str] = []
    for r in replacements:
        start_idx = r.line - 1
        end_idx = (r.end_line - 1) if r.end_line is not None else start_idx
        if r.end_line is not None and r.end_hash is None:
            # Malformed address; let resolve_replacements report it.
            out.append(r)
            continue
        if _address_ok(r, start_idx, end_idx, file_lines):
            out.append(r)
            continue
        translated = (
            _translate_interval(smap, start_idx, end_idx)
            if 0 <= start_idx <= end_idx
            else None
        )
        if translated is not None:
            new_start, new_end = translated
            update: dict[str, int] = {"line": new_start + 1}
            if r.end_line is not None:
                update["end_line"] = new_end + 1
            candidate = r.model_copy(update=update)
            if _address_ok(candidate, new_start, new_end, file_lines):
                shift = new_start - start_idx
                notes.append(
                    f"line {r.line}: address shifted {shift:+d} "
                    f"line{'s' if abs(shift) != 1 else ''} to line {new_start + 1} "
                    "to follow this session's earlier edits "
                    "(hashes verified at the new position)"
                )
                out.append(candidate)
                continue
        reset_shift_map(key)
        return replacements, []
    return out, notes


def record_shift_generation(
    key: str,
    prepared_asc: list[tuple[int, int, LineReplacement, list[str]]],
    new_lines: list[str],
) -> None:
    """Append this call's edits to the file's map and refresh the fingerprint.

    Zero-delta edits are recorded too: they shift nothing, but a later stale
    interval overlapping them must still be refused (its content changed).
    """
    edits: list[_ShiftEdit] = [
        (start_idx, end_idx, len(lines) - (end_idx - start_idx + 1))
        for start_idx, end_idx, _r, lines in prepared_asc
    ]
    smap = _shift_maps.get(key)
    if smap is None:
        smap = _ShiftMap(fingerprint=0)
        _shift_maps[key] = smap
    smap.fingerprint = _content_fingerprint(new_lines)
    smap.generations.append(edits)


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
    keep_indent: bool = False,
) -> tuple[list[tuple[int, int, LineReplacement, list[str]]], list[str]]:
    """Turn each replacement's new_content into the lines to splice, applying
    the hallucination corrections and recording what was done.

    - Leaked hashed_read prefixes are stripped from new_content (unless
      ``allow_literal``).
    - Indentation is best-effort corrected (unless ``keep_indent``): uniform
      comment blocks align to the adjacent comment, and an off-grid base
      indent is shifted onto the file's indent grid. Runs before duplicate
      detection so a corrected line can still be recognized as a duplicate.
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

        if not keep_indent and lines:
            lines, indent_notes = correct_indentation(
                lines, file_lines, start_idx, end_idx, covered, r.line
            )
            notes.extend(indent_notes)

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
    keep_indent: bool = False,
) -> ApplyResult:
    file_path = resolve_file_path(path_str)
    file_lines = await read_file_lines(file_path)

    replacements, shift_notes = translate_stale_addresses(
        str(file_path), replacements, file_lines
    )

    resolved = resolve_replacements(replacements, file_lines)
    validate_all_hashes(resolved, file_lines)

    prepared, notes = prepare_replacements(
        resolved,
        file_lines,
        allow_literal=allow_literal,
        keep_duplicate=keep_duplicate,
        keep_indent=keep_indent,
    )

    prepared_asc = sorted(prepared, key=lambda p: p[0])
    prepared_desc = list(reversed(prepared_asc))

    new_lines = list(file_lines)
    total_lines_changed = 0
    for start_idx, end_idx, _replacement, replacement_lines in prepared_desc:
        new_lines[start_idx : end_idx + 1] = replacement_lines
        total_lines_changed += end_idx - start_idx + 1

    await write_file_lines(file_path, new_lines)
    record_shift_generation(str(file_path), prepared_asc, new_lines)
    context = build_success_context(new_lines, prepared_asc)
    all_notes = shift_notes + notes
    return ApplyResult(
        path=str(file_path),
        total_ops=len(replacements),
        total_lines_changed=total_lines_changed,
        context=context,
        path_note=normalization_note(path_str, file_path),
        content_note="\n".join(all_notes) if all_notes else None,
    )
