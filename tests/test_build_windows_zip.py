"""build-windows-zip.sh ships the working tree: tracked files as they are
on disk and untracked files git does not ignore. It once shipped a
modified agent_loop.py without the new, untracked auto_continue.py it
imported, and the Windows machine crashed on import.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest

SCRIPT = Path(__file__).parents[1] / "build-windows-zip.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("zip") is None or shutil.which("git") is None,
    reason="needs git and zip on PATH",
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def _build(repo: Path, home: Path) -> zipfile.ZipFile:
    (home / "sharedViaLlama.cpp").mkdir(parents=True)
    env = {**os.environ, "HOME": str(home)}
    subprocess.run(
        ["bash", str(repo / "build-windows-zip.sh")],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return zipfile.ZipFile(home / "sharedViaLlama.cpp" / "privibe-source-latest.zip")


def test_zip_ships_untracked_and_modified_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    shutil.copy(SCRIPT, repo / "build-windows-zip.sh")
    (repo / ".gitignore").write_text("privibe-source-*.zip\n.venv/\n")
    (repo / "privibe").mkdir()
    (repo / "privibe" / "tracked.py").write_text("old = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")

    (repo / "privibe" / "tracked.py").write_text("from privibe.fresh import x\n")
    (repo / "privibe" / "fresh.py").write_text("x = 1\n")
    (repo / ".venv").mkdir()
    (repo / ".venv" / "junk.py").write_text("no\n")

    with _build(repo, tmp_path / "home") as zf:
        names = set(zf.namelist())
        assert "privibe/privibe/fresh.py" in names
        assert zf.read("privibe/privibe/tracked.py") == b"from privibe.fresh import x\n"
        assert not any(".venv" in n for n in names)
        assert "privibe/privibe/_build_info.py" in names

    # The snapshot never touched the real index: fresh.py is still untracked.
    assert "?? privibe/fresh.py" in _git(repo, "status", "--short")
