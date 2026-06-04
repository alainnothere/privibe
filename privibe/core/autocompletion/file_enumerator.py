from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess

from privibe.core.autocompletion.file_indexer.ignore_rules import IgnoreRules

ASCII_CODEPOINT_LIMIT = 128

# Bounds the subprocess calls so a hung git/rg can never block the completion
# worker indefinitely.
_SUBPROCESS_TIMEOUT_SECONDS = 5.0

# Test/debug seam: set to "walk" (or "git"/"rg") to force a single backend.
# Read at call time so tests can toggle it with monkeypatch.setenv.
_BACKEND_ENV_VAR = "PRIVIBE_FILE_ENUMERATOR"


@dataclass(slots=True)
class IndexEntry:
    rel: str
    rel_lower: str
    name: str
    path: Path
    is_dir: bool
    ascii_mask: int


def build_ascii_mask(value: str) -> int:
    mask = 0
    for char in value:
        codepoint = ord(char)
        if codepoint >= ASCII_CODEPOINT_LIMIT:
            continue
        mask |= 1 << codepoint
    return mask


def enumerate_entries(root: Path) -> list[IndexEntry]:
    """Return the completion candidate set for ``root``.

    Files come from a fresh, stateless listing (git -> ripgrep -> walk). There is
    no persistent index and no watcher: each call reflects the tree as it is now.
    Directory entries are derived from the file list, so empty directories (which
    git does not track) are intentionally not suggested.
    """
    resolved_root = root.resolve()
    rel_files = _list_files(resolved_root)
    return _build_entries(resolved_root, rel_files)


def _list_files(root: Path) -> list[str]:
    forced = os.environ.get(_BACKEND_ENV_VAR)

    if forced == "walk":
        return _walk_files(root)
    if forced == "git":
        return _git_files(root) or []
    if forced == "rg":
        return _rg_files(root) or []

    git_files = _git_files(root)
    if git_files is not None:
        return git_files

    rg_files = _rg_files(root)
    if rg_files is not None:
        return rg_files

    return _walk_files(root)


def _git_files(root: Path) -> list[str] | None:
    if not shutil.which("git"):
        return None
    try:
        inside = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        )
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            return None

        # --cached + --others --exclude-standard => tracked files plus untracked
        # files that are not ignored. -z keeps paths with odd characters intact.
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        )
        if result.returncode != 0:
            return None
        # ls-files can list a tracked path twice (cached + others); dedupe.
        seen: dict[str, None] = {}
        for path in result.stdout.split("\0"):
            if path:
                seen.setdefault(path, None)
        return list(seen)
    except (OSError, subprocess.SubprocessError):
        return None


def _rg_files(root: Path) -> list[str] | None:
    rg = shutil.which("rg")
    if not rg:
        return None
    try:
        result = subprocess.run(
            [rg, "--files", "--hidden", "--glob", "!.git"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        )
        # 0 = files found, 1 = none found; both are valid, 2 = real error.
        if result.returncode not in {0, 1}:
            return None
        return [line for line in result.stdout.splitlines() if line]
    except (OSError, subprocess.SubprocessError):
        return None


def _walk_files(root: Path) -> list[str]:
    rules = IgnoreRules()
    rules.ensure_for_root(root)
    files: list[str] = []
    _walk_into(root, "", rules, files)
    return files


def _walk_into(
    directory: Path, rel_prefix: str, rules: IgnoreRules, out: list[str]
) -> None:
    try:
        with os.scandir(directory) as iterator:
            for entry in iterator:
                is_dir = entry.is_dir(follow_symlinks=False)
                name = entry.name
                rel = f"{rel_prefix}/{name}" if rel_prefix else name
                if rules.should_ignore(rel, name, is_dir):
                    continue
                if is_dir:
                    _walk_into(Path(entry.path), rel, rules, out)
                else:
                    out.append(rel)
    except (PermissionError, OSError):
        pass


def _build_entries(root: Path, rel_files: list[str]) -> list[IndexEntry]:
    entries: list[IndexEntry] = []
    seen_dirs: set[str] = set()

    for rel in rel_files:
        rel = rel.replace("\\", "/")
        parts = rel.split("/")
        name = parts[-1]
        entries.append(_make_entry(root, rel, name, is_dir=False))

        # Derive each parent directory once.
        for depth in range(1, len(parts)):
            dir_rel = "/".join(parts[:depth])
            if dir_rel in seen_dirs:
                continue
            seen_dirs.add(dir_rel)
            entries.append(_make_entry(root, dir_rel, parts[depth - 1], is_dir=True))

    return entries


def _make_entry(root: Path, rel: str, name: str, *, is_dir: bool) -> IndexEntry:
    rel_lower = rel.lower()
    return IndexEntry(
        rel=rel,
        rel_lower=rel_lower,
        name=name,
        path=root / rel,
        is_dir=is_dir,
        ascii_mask=build_ascii_mask(rel_lower),
    )
