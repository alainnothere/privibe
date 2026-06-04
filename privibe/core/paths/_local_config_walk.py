from __future__ import annotations

from collections import deque
import logging
import os
from pathlib import Path

from privibe.core.autocompletion.file_indexer.ignore_rules import WALK_SKIP_DIR_NAMES

logger = logging.getLogger("privibe")

_VIBE_DIR = ".privibe"
_TOOLS_SUBDIR = Path(_VIBE_DIR) / "tools"
_VIBE_SKILLS_SUBDIR = Path(_VIBE_DIR) / "skills"
_AGENTS_SUBDIR = Path(_VIBE_DIR) / "agents"
_AGENTS_DIR = ".agents"
_AGENTS_SKILLS_SUBDIR = Path(_AGENTS_DIR) / "skills"

WALK_MAX_DEPTH = 0  # default: current directory only
_MAX_DIRS = 2000


def _collect_config_dirs_at(
    path: Path,
    entries: set[str],
    tools: list[Path],
    skills: list[Path],
    agents: list[Path],
) -> None:
    """Check a single directory for .privibe/ and .agents/ config subdirs."""
    if _VIBE_DIR in entries:
        if (candidate := path / _TOOLS_SUBDIR).is_dir():
            tools.append(candidate)
        if (candidate := path / _VIBE_SKILLS_SUBDIR).is_dir():
            skills.append(candidate)
        if (candidate := path / _AGENTS_SUBDIR).is_dir():
            agents.append(candidate)
    if _AGENTS_DIR in entries:
        if (candidate := path / _AGENTS_SKILLS_SUBDIR).is_dir():
            skills.append(candidate)


def _iter_child_dirs(path: Path, entries: set[str]) -> list[Path]:
    """Return sorted child directories to descend into, skipping ignored and dot-dirs."""
    children: list[Path] = []
    for name in sorted(entries):
        if name in WALK_SKIP_DIR_NAMES or name.startswith("."):
            continue
        child = path / name
        try:
            if child.is_dir():
                children.append(child)
        except OSError:
            continue
    return children


def walk_local_config_dirs_all(
    root: Path,
    max_depth: int = WALK_MAX_DEPTH,
) -> tuple[tuple[Path, ...], tuple[Path, ...], tuple[Path, ...]]:
    """Discover .privibe/ and .agents/ config directories under *root*.

    Uses breadth-first search bounded by ``max_depth`` and ``_MAX_DIRS``
    to avoid unbounded traversal in large repositories.
    ``max_depth=0`` means only the root directory is checked.
    """
    tools_dirs: list[Path] = []
    skills_dirs: list[Path] = []
    agents_dirs: list[Path] = []

    resolved_root = root.resolve()
    queue: deque[tuple[Path, int]] = deque([(resolved_root, 0)])
    visited = 0

    while queue and visited < _MAX_DIRS:
        current, depth = queue.popleft()
        visited += 1

        try:
            entries = set(os.listdir(current))
        except OSError:
            continue

        _collect_config_dirs_at(current, entries, tools_dirs, skills_dirs, agents_dirs)

        if depth < max_depth:
            queue.extend(
                (child, depth + 1) for child in _iter_child_dirs(current, entries)
            )

    if visited >= _MAX_DIRS:
        logger.warning(
            "Config directory scan reached directory limit (%d dirs) at %s",
            _MAX_DIRS,
            resolved_root,
        )

    return (tuple(tools_dirs), tuple(skills_dirs), tuple(agents_dirs))


def has_config_dirs_nearby(
    root: Path, *, max_depth: int = WALK_MAX_DEPTH, max_dirs: int = 50
) -> bool:
    """Quick check for .privibe/ or .agents/ config dirs in the near subtree.

    Returns ``True`` as soon as any config directory is found, without
    enumerating all of them.
    """
    resolved = root.resolve()
    queue: deque[tuple[Path, int]] = deque([(resolved, 0)])
    visited = 0
    found: list[Path] = []

    while queue and visited < max_dirs:
        current, depth = queue.popleft()
        visited += 1

        try:
            entries = set(os.listdir(current))
        except OSError:
            continue

        _collect_config_dirs_at(current, entries, found, found, found)
        if found:
            return True

        if depth < max_depth:
            queue.extend(
                (child, depth + 1) for child in _iter_child_dirs(current, entries)
            )

    return False
