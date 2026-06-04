from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

from privibe.core.autocompletion.file_enumerator import enumerate_entries


def _rels(root: Path, *, dirs: bool | None = None) -> set[str]:
    entries = enumerate_entries(root)
    if dirs is True:
        entries = [e for e in entries if e.is_dir]
    elif dirs is False:
        entries = [e for e in entries if not e.is_dir]
    return {e.rel for e in entries}


def _make_tree(root: Path) -> None:
    (root / "src" / "core").mkdir(parents=True)
    (root / "src" / "core" / "logger.py").write_text("", encoding="utf-8")
    (root / "src" / "main.py").write_text("", encoding="utf-8")
    (root / "README.md").write_text("", encoding="utf-8")
    (root / ".env").write_text("", encoding="utf-8")
    # An empty directory: git does not track these, so it must not be suggested.
    (root / "src" / "empty").mkdir()


@pytest.fixture(autouse=True)
def _force_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    # Default every test here to the walk backend; git/rg tests opt back in.
    monkeypatch.setenv("PRIVIBE_FILE_ENUMERATOR", "walk")


def test_walk_lists_files_and_derives_parent_directories(tmp_path: Path) -> None:
    _make_tree(tmp_path)

    files = _rels(tmp_path, dirs=False)
    dirs = _rels(tmp_path, dirs=True)

    assert files == {"src/core/logger.py", "src/main.py", "README.md", ".env"}
    assert dirs == {"src", "src/core"}


def test_walk_does_not_suggest_empty_directories(tmp_path: Path) -> None:
    _make_tree(tmp_path)

    assert "src/empty" not in _rels(tmp_path, dirs=True)


def test_walk_includes_dotfiles(tmp_path: Path) -> None:
    _make_tree(tmp_path)

    assert ".env" in _rels(tmp_path, dirs=False)


def test_walk_skips_default_ignored_directories(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text("", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_text("", encoding="utf-8")

    rels = _rels(tmp_path)

    assert not any(r.startswith("node_modules") for r in rels)
    assert not any(r.startswith("__pycache__") for r in rels)


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_git_backend_lists_untracked_and_respects_gitignore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PRIVIBE_FILE_ENUMERATOR", "git")
    # Neutralise global/system git config so --exclude-standard is deterministic.
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")

    _make_tree(tmp_path)
    (tmp_path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("", encoding="utf-8")
    subprocess.run(
        ["git", "init"], cwd=tmp_path, check=True, capture_output=True
    )

    files = _rels(tmp_path, dirs=False)

    # untracked-but-not-ignored files show up without needing a commit
    assert "src/main.py" in files
    assert ".env" in files
    # gitignored files do not
    assert "ignored.txt" not in files
    # still files-only with derived dirs, no empty dir
    assert "src/empty" not in _rels(tmp_path, dirs=True)


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
def test_rg_backend_lists_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PRIVIBE_FILE_ENUMERATOR", "rg")
    _make_tree(tmp_path)

    files = _rels(tmp_path, dirs=False)

    assert "src/main.py" in files
    assert "README.md" in files
    assert "src/empty" not in _rels(tmp_path, dirs=True)
