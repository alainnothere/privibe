from __future__ import annotations

import difflib

from privibe.core.types import FileDiff

# Catastrophe guards: above these we skip difflib and show a head/tail sample
# instead. The ceiling is on FILE size, not change size, so keep it generous --
# a small edit to a large file should still get a real diff.
_MAX_LINES = 10_000
_MAX_BYTES = 1_000_000
_BINARY_SNIFF_BYTES = 8192

# Unified-diff context lines on each side of a change.
_CONTEXT = 2
# A contiguous run of removed/added lines longer than this is cropped to
# head + "N more lines cut" + tail.
_RUN_HEAD = 2
_RUN_TAIL = 2
_RUN_THRESHOLD = 5

Row = tuple[str, str]  # (css_class, text)


def _looks_binary(data: bytes | None) -> bool:
    return bool(data) and b"\x00" in data[:_BINARY_SNIFF_BYTES]


def _decode(data: bytes | None) -> list[str]:
    if not data:
        return []
    return data.decode("utf-8", errors="replace").splitlines()


def _crop_run(texts: list[str], css_class: str) -> list[Row]:
    """Crop a contiguous run of same-kind lines to head/tail with a cut marker."""
    n = len(texts)
    if n <= _RUN_THRESHOLD:
        return [(css_class, t) for t in texts]
    cut = n - _RUN_HEAD - _RUN_TAIL
    rows: list[Row] = [(css_class, t) for t in texts[:_RUN_HEAD]]
    rows.append(("diff-context", f"    {cut} more lines cut"))
    rows.extend((css_class, t) for t in texts[-_RUN_TAIL:])
    return rows


def _build_hunks(old_lines: list[str], new_lines: list[str]) -> list[list[Row]]:
    """Run difflib and split into hunks, cropping long +/- runs within each."""
    diff = difflib.unified_diff(old_lines, new_lines, n=_CONTEXT, lineterm="")
    hunks: list[list[Row]] = []
    current: list[Row] | None = None
    run_class: str | None = None
    run: list[str] = []

    def flush_run() -> None:
        nonlocal run_class, run
        if run_class is not None and current is not None:
            current.extend(_crop_run(run, run_class))
        run_class, run = None, []

    for line in diff:
        if line.startswith("--- ") or line.startswith("+++ "):
            continue
        if line.startswith("@@"):
            flush_run()
            if current is not None:
                hunks.append(current)
            current = [("diff-range", line)]
            continue
        if current is None:
            continue
        if line.startswith("-"):
            if run_class != "diff-removed":
                flush_run()
                run_class = "diff-removed"
            run.append(line)
        elif line.startswith("+"):
            if run_class != "diff-added":
                flush_run()
                run_class = "diff-added"
            run.append(line)
        else:  # context line (starts with a space)
            flush_run()
            current.append(("diff-context", line))

    flush_run()
    if current is not None:
        hunks.append(current)
    return hunks


def _sample_side(lines: list[str], prefix: str, css_class: str) -> list[Row]:
    n = len(lines)
    if n <= _RUN_HEAD + _RUN_TAIL + 1:
        return [(css_class, f"{prefix}{t}") for t in lines]
    cut = n - _RUN_HEAD - _RUN_TAIL
    rows: list[Row] = [(css_class, f"{prefix}{t}") for t in lines[:_RUN_HEAD]]
    rows.append(("diff-context", f"    {cut} more lines cut"))
    rows.extend((css_class, f"{prefix}{t}") for t in lines[-_RUN_TAIL:])
    return rows


def _sample(
    path: str,
    old_bytes: bytes | None,
    new_bytes: bytes | None,
    old_lines: list[str],
    new_lines: list[str],
    tool_name: str | None,
) -> FileDiff:
    if old_bytes is None:
        header = f"Created file ({len(new_lines)} lines)"
    elif new_bytes is None:
        header = f"Deleted file ({len(old_lines)} lines)"
    elif tool_name == "write_file":
        header = f"Replaced entire file ({len(new_lines)} lines)"
    else:
        header = f"Large change - preview only ({len(new_lines)} lines)"

    rows: list[Row] = [("diff-header", header)]
    if old_lines:
        rows.extend(_sample_side(old_lines, "-", "diff-removed"))
    if new_lines:
        rows.extend(_sample_side(new_lines, "+", "diff-added"))
    return FileDiff(path=path, kind="sample", hunks=[rows])


def build_file_diff(
    path: str,
    old_bytes: bytes | None,
    new_bytes: bytes | None,
    tool_name: str | None = None,
) -> FileDiff | None:
    """Build a display-only red/green diff for a file edit.

    old_bytes/new_bytes are the pre- and post-edit file contents (None = the
    file did not exist on that side: a create or a delete). Returns None for a
    no-op. Four outcomes: None (no-op), binary, sample (too large), diff.
    """
    if old_bytes == new_bytes:
        return None

    if _looks_binary(old_bytes) or _looks_binary(new_bytes):
        return FileDiff(path=path, kind="binary", note="Binary file - no diff shown")

    old_big = old_bytes is not None and len(old_bytes) > _MAX_BYTES
    new_big = new_bytes is not None and len(new_bytes) > _MAX_BYTES
    old_lines = _decode(old_bytes)
    new_lines = _decode(new_bytes)

    if old_big or new_big or max(len(old_lines), len(new_lines)) > _MAX_LINES:
        return _sample(path, old_bytes, new_bytes, old_lines, new_lines, tool_name)

    hunks = _build_hunks(old_lines, new_lines)
    if not hunks:
        return None
    return FileDiff(path=path, kind="diff", hunks=hunks)
