"""The new-file ladder: repo files first, extension rule, siblings, platform."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from privibe.core.utils import newfile_conventions as nc
from privibe.core.utils.newfile_conventions import (
    clear_caches,
    conventions_for,
    editorconfig_for,
    extension_ending,
)


@pytest.fixture(autouse=True)
def _fresh_caches():
    clear_caches()
    yield
    clear_caches()


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _init_repo(path: Path) -> None:
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@t")
    _git(path, "config", "user.name", "t")
    _git(path, "config", "core.autocrlf", "false")


# ---------------------------------------------------------------------------
# .editorconfig
# ---------------------------------------------------------------------------


def test_editorconfig_basic_and_glob_sections(tmp_path: Path):
    (tmp_path / ".editorconfig").write_text(
        "root = true\n\n"
        "[*]\nend_of_line = lf\ncharset = utf-8\ninsert_final_newline = true\n"
        "trim_trailing_whitespace = true\nindent_style = space\nindent_size = 4\n\n"
        "[*.md]\ntrim_trailing_whitespace = false\n\n"
        "[*.{bat,cmd}]\nend_of_line = crlf\n\n"
        "[Makefile]\nindent_style = tab\n"
    )
    py = editorconfig_for(tmp_path / "src" / "a.py")
    assert py.end_of_line == "\n"
    assert py.trim_trailing_whitespace is True
    assert py.indent_style == "space"
    assert py.indent_size == 4
    md = editorconfig_for(tmp_path / "docs" / "x.md")
    assert md.trim_trailing_whitespace is False
    assert md.end_of_line == "\n"
    bat = editorconfig_for(tmp_path / "run.cmd")
    assert bat.end_of_line == "\r\n"
    mk = editorconfig_for(tmp_path / "Makefile")
    assert mk.indent_style == "tab"


def test_editorconfig_nearer_file_overrides_and_root_stops(tmp_path: Path):
    (tmp_path / ".editorconfig").write_text("[*]\nend_of_line = crlf\ncharset = utf-8-bom\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / ".editorconfig").write_text("[*]\nend_of_line = lf\n")
    ec = editorconfig_for(sub / "f.txt")
    assert ec.end_of_line == "\n"  # nearer wins
    assert ec.wants_bom is True  # inherited from the parent
    (sub / ".editorconfig").write_text("root = true\n[*]\nend_of_line = lf\n")
    clear_caches()
    ec = editorconfig_for(sub / "f.txt")
    assert ec.wants_bom is False  # root = true stops the walk


def test_editorconfig_charset_maps_to_python_codec(tmp_path: Path):
    (tmp_path / ".editorconfig").write_text("[*]\ncharset = latin1\n")
    assert editorconfig_for(tmp_path / "f").python_encoding == "latin-1"


# ---------------------------------------------------------------------------
# extension rule
# ---------------------------------------------------------------------------


def test_extension_rule():
    assert extension_ending(Path("x.sh")) == "\n"
    assert extension_ending(Path("x.BAT")) == "\r\n"
    assert extension_ending(Path("x.py")) is None


# ---------------------------------------------------------------------------
# the ladder
# ---------------------------------------------------------------------------


def test_ladder_defaults_outside_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(nc, "platform_ending", lambda _d: "\n")
    conv = conventions_for(tmp_path / "new.py")
    assert conv.newline == "\n"
    assert conv.encoding == "utf-8"
    assert conv.bom is False
    assert conv.final_newline is True
    assert conv.source == "platform"


def test_ladder_gitattributes_beats_everything(tmp_path: Path):
    _init_repo(tmp_path)
    (tmp_path / ".gitattributes").write_text("*.txt text eol=crlf\n")
    (tmp_path / ".editorconfig").write_text("[*]\nend_of_line = lf\n")
    conv = conventions_for(tmp_path / "new.txt")
    assert conv.newline == "\r\n"
    assert conv.source == ".gitattributes"


def test_ladder_editorconfig_beats_extension(tmp_path: Path):
    (tmp_path / ".editorconfig").write_text("[*.sh]\nend_of_line = crlf\n")
    conv = conventions_for(tmp_path / "odd.sh")
    assert conv.newline == "\r\n"
    assert conv.source == ".editorconfig"


def test_ladder_extension_beats_siblings(tmp_path: Path):
    _init_repo(tmp_path)
    (tmp_path / "a.txt").write_bytes(b"x\r\n")
    (tmp_path / "b.txt").write_bytes(b"x\r\n")
    _git(tmp_path, "add", ".")
    conv = conventions_for(tmp_path / "run.sh")
    assert conv.newline == "\n"
    assert conv.source == "extension"


def test_ladder_siblings_vote_on_endings(tmp_path: Path):
    _init_repo(tmp_path)
    (tmp_path / "a.cs").write_bytes(b"x\r\n")
    (tmp_path / "b.cs").write_bytes(b"x\r\n")
    (tmp_path / "c.cs").write_bytes(b"x\n")
    _git(tmp_path, "add", ".")
    conv = conventions_for(tmp_path / "new.cs")
    assert conv.newline == "\r\n"
    assert conv.source == "siblings"


def test_ladder_siblings_never_vote_on_bom(tmp_path: Path):
    _init_repo(tmp_path)
    (tmp_path / "a.cs").write_bytes(b"\xef\xbb\xbfx\r\n")
    (tmp_path / "b.cs").write_bytes(b"\xef\xbb\xbfx\r\n")
    _git(tmp_path, "add", ".")
    conv = conventions_for(tmp_path / "new.cs")
    assert conv.bom is False


def test_ladder_editorconfig_bom(tmp_path: Path):
    (tmp_path / ".editorconfig").write_text("[*.ps1]\ncharset = utf-8-bom\n")
    conv = conventions_for(tmp_path / "x.ps1")
    assert conv.bom is True
    assert conv.encoding == "utf-8"


def test_platform_ending_windows_autocrlf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _init_repo(tmp_path)
    _git(tmp_path, "config", "core.autocrlf", "true")
    monkeypatch.setattr(nc.sys, "platform", "win32")
    assert nc.platform_ending(tmp_path) == "\r\n"
    _git(tmp_path, "config", "core.autocrlf", "input")
    clear_caches()
    assert nc.platform_ending(tmp_path) == "\n"
    monkeypatch.setattr(nc.sys, "platform", "linux")
    _git(tmp_path, "config", "core.autocrlf", "true")
    clear_caches()
    assert nc.platform_ending(tmp_path) == "\n"
