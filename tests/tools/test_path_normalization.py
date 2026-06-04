from __future__ import annotations

from pathlib import Path

import pytest

from privibe.core.paths import dialect as dialect_mod
from privibe.core.paths.dialect import PathDialect, reset_dialect_cache
from privibe.core.tools.utils import (
    is_path_within_workdir,
    normalization_note,
    normalize_tool_path,
    resolve_file_tool_permission,
)
from privibe.core.tools.base import ToolPermission


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    reset_dialect_cache()
    yield
    reset_dialect_cache()


def _force_dialect(monkeypatch: pytest.MonkeyPatch, dialect: PathDialect) -> None:
    monkeypatch.setattr(dialect_mod, "_detect", lambda: dialect)
    reset_dialect_cache()


class TestNormalizeToolPath:
    def test_relative_against_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = normalize_tool_path("foo.txt")
        assert result.is_absolute()
        assert result.name == "foo.txt"

    def test_absolute_unchanged(self, tmp_path: Path) -> None:
        result = normalize_tool_path(str(tmp_path / "x"))
        assert result == tmp_path / "x"

    def test_user_expansion(self) -> None:
        # ~ expands; result is absolute
        result = normalize_tool_path("~/foo")
        assert result.is_absolute()
        assert "foo" in str(result)

    def test_translation_runs_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _force_dialect(monkeypatch, PathDialect.WSL)
        monkeypatch.setattr(dialect_mod.os.path, "lexists", lambda _: False)
        result = normalize_tool_path("C:\\foo\\bar")
        # On POSIX the WSL form passes through pathlib as a normal path.
        assert "/mnt/c/foo/bar" in str(result).replace("\\", "/")


class TestNormalizationNote:
    def test_no_note_when_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _force_dialect(monkeypatch, PathDialect.POSIX)
        assert normalization_note("/home/u/x", Path("/home/u/x")) is None

    def test_note_when_translated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _force_dialect(monkeypatch, PathDialect.WSL)
        monkeypatch.setattr(dialect_mod.os.path, "lexists", lambda _: False)
        note = normalization_note("C:\\foo", Path("/mnt/c/foo"))
        assert note is not None
        assert "C:\\foo" in note
        assert "/mnt/c/foo" in note

    def test_no_note_for_relative_promotion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Promoting `foo.txt` to `<cwd>/foo.txt` is not a dialect translation
        and shouldn't produce a note."""
        _force_dialect(monkeypatch, PathDialect.POSIX)
        monkeypatch.chdir(tmp_path)
        assert normalization_note("foo.txt", tmp_path / "foo.txt") is None


class TestIsPathWithinWorkdir:
    def test_inside(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "a").mkdir()
        assert is_path_within_workdir("a") is True
        assert is_path_within_workdir(str(tmp_path / "a" / "b")) is True

    def test_outside(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        # /etc is outside any tmp_path
        assert is_path_within_workdir("/etc") is False


class TestSensitiveGlob:
    """Verify that sensitive-pattern matching works after path translation
    (regression for PurePath defaulting to PureWindowsPath on Windows)."""

    def test_dotenv_match_via_posix_form(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("X=1")
        ctx = resolve_file_tool_permission(
            ".env",
            tool_name="read_file",
            allowlist=[],
            denylist=[],
            config_permission=ToolPermission.ALWAYS,
            sensitive_patterns=["**/.env", "**/.env.*"],
        )
        assert ctx is not None
        assert ctx.permission == ToolPermission.ASK
