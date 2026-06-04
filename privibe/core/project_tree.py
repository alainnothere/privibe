from __future__ import annotations

import os
from pathlib import Path

from privibe.core.autocompletion.file_indexer.ignore_rules import (
    IgnoreRules,
    WALK_SKIP_DIR_NAMES,
)


def build_tree(
    root: Path,
    max_depth: int = 2,
    max_lines: int = 80,
) -> str:
    """Generate a compact directory tree for agent context.

    Uses IgnoreRules (defaults + .gitignore) to filter entries.
    Truncates with summary when max_lines would be exceeded.
    """
    root = root.resolve()
    rules = IgnoreRules()
    rules.ensure_for_root(root)

    lines: list[str] = []
    _collect(root, root, 0, max_depth, max_lines, lines, rules, prefix="")
    return "\n".join(lines)


def _collect(
    root: Path,
    current: Path,
    depth: int,
    max_depth: int,
    max_lines: int,
    lines: list[str],
    rules: IgnoreRules,
    prefix: str,
) -> None:
    """Walk directory, collecting entries with proper tree indentation."""
    entries: list[tuple[str, bool]] = []

    try:
        with os.scandir(current) as it:
            for entry in it:
                name = entry.name
                is_dir = entry.is_dir(follow_symlinks=False)

                if is_dir and name in WALK_SKIP_DIR_NAMES:
                    continue

                rel_str = str((current / name).relative_to(root))
                if rules.should_ignore(rel_str, name, is_dir):
                    continue

                entries.append((name, is_dir))
    except PermissionError:
        return

    if not entries:
        return

    # Sort: files first, then dirs, both alphabetically
    files = sorted((n, d) for n, d in entries if not d)
    dirs = sorted((n, d) for n, d in entries if d)
    all_entries = files + dirs
    total = len(all_entries)

    # Budget check
    remaining = max_lines - len(lines)
    if remaining <= 1:
        return

    # Estimate: each file costs 1 line, each dir costs ~3 (itself + avg children)
    estimated = len(files) + len(dirs) * 3

    if estimated > remaining:
        # Truncate: show what fits, summarize the rest
        budget = max(remaining - 1, 2)
        shown = 0
        for idx, (name, is_dir) in enumerate(all_entries):
            if shown >= budget:
                break
            is_last = idx == budget - 1
            connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
            lines.append(f"{prefix}{connector}{name}{'/' if is_dir else ''}")
            shown += 1
        skipped = total - shown
        if skipped > 0:
            lines.append(f"{prefix}  ... ({skipped} more)")
        return

    # Normal output with proper tree chars
    for idx, (name, is_dir) in enumerate(all_entries):
        is_last = idx == total - 1
        connector = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
        lines.append(f"{prefix}{connector}{name}{'/' if is_dir else ''}")

        # Recurse into dirs if budget and depth allow
        if is_dir and depth < max_depth:
            child_prefix = prefix + ("      " if is_last else "  \u2502   ")
            _collect(
                root, current / name, depth + 1, max_depth,
                max_lines, lines, rules, prefix=child_prefix
            )
