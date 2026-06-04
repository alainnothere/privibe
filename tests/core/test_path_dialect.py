from __future__ import annotations

from pathlib import Path

import pytest

from privibe.core.paths import dialect as dialect_mod
from privibe.core.paths.dialect import (
    PathDialect,
    configure_path_translation,
    detect_path_dialect,
    dialect_hint,
    reset_dialect_cache,
    reset_translation_config,
    to_posix_for_match,
    translate_path,
)


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    reset_dialect_cache()
    reset_translation_config()
    yield
    reset_dialect_cache()
    reset_translation_config()


def _force_dialect(monkeypatch: pytest.MonkeyPatch, dialect: PathDialect) -> None:
    """Stub out detect_path_dialect to return a specific dialect."""
    monkeypatch.setattr(dialect_mod, "_detect", lambda: dialect)
    reset_dialect_cache()


class TestDetectDialect:
    def test_posix_when_linux_and_no_wsl(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dialect_mod.sys, "platform", "linux")
        monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
        monkeypatch.delenv("WSL_INTEROP", raising=False)
        monkeypatch.delenv("CYGWIN", raising=False)
        monkeypatch.setattr(dialect_mod.os.path, "isdir", lambda p: False)
        reset_dialect_cache()
        assert detect_path_dialect() == PathDialect.POSIX

    def test_wsl_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dialect_mod.sys, "platform", "linux")
        monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
        reset_dialect_cache()
        assert detect_path_dialect() == PathDialect.WSL

    def test_wsl_via_mnt_c(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dialect_mod.sys, "platform", "linux")
        monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
        monkeypatch.delenv("WSL_INTEROP", raising=False)
        monkeypatch.setattr(
            dialect_mod.os.path, "isdir", lambda p: p == "/mnt/c"
        )
        reset_dialect_cache()
        assert detect_path_dialect() == PathDialect.WSL

    def test_git_bash_via_msystem(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dialect_mod.sys, "platform", "win32")
        monkeypatch.setenv("MSYSTEM", "MINGW64")
        reset_dialect_cache()
        assert detect_path_dialect() == PathDialect.GIT_BASH

    def test_windows_native_when_no_msystem(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dialect_mod.sys, "platform", "win32")
        monkeypatch.delenv("MSYSTEM", raising=False)
        reset_dialect_cache()
        assert detect_path_dialect() == PathDialect.WINDOWS_NATIVE

    def test_cached_across_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"n": 0}

        def fake_detect() -> PathDialect:
            calls["n"] += 1
            return PathDialect.POSIX

        monkeypatch.setattr(dialect_mod, "_detect", fake_detect)
        reset_dialect_cache()
        detect_path_dialect()
        detect_path_dialect()
        detect_path_dialect()
        assert calls["n"] == 1


class TestTranslatePath:
    """Translation rules per detected dialect.

    We don't depend on real filesystem state — translate_path falls through to
    the canonical form when neither raw nor canonical exist.
    """

    def test_posix_passthrough_no_drive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _force_dialect(monkeypatch, PathDialect.POSIX)
        assert translate_path("/home/u/x") == "/home/u/x"

    def test_posix_recovers_windows_shaped_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On Linux the model sometimes hands us `c:\\home\\user\\foo`. We
        drop the drive letter and normalise separators so the file actually
        gets read instead of failing with 'File not found'.

        Only the unambiguous colon-drive forms (`X:\\` and `X:/`) trigger the
        recovery — slash-drive forms like `/c/foo` are legitimate POSIX paths
        in their own right and pass through (see test_posix_passes_slash_drive_through)."""
        _force_dialect(monkeypatch, PathDialect.POSIX)
        monkeypatch.setattr(dialect_mod.os.path, "lexists", lambda _: False)
        assert translate_path("c:\\home\\user\\foo") == "/home/user/foo"
        assert translate_path("c:/home/user/foo") == "/home/user/foo"
        assert translate_path("D:\\projects\\bar") == "/projects/bar"

    def test_posix_passes_slash_drive_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`/c/foo`, `/B/x`, etc. are valid POSIX paths. Don't pretend they
        are Windows drives just because they look like the Git Bash form —
        on plain Linux there's no Git Bash translation layer underneath, and
        rewriting them would silently mangle alias outputs and real paths."""
        _force_dialect(monkeypatch, PathDialect.POSIX)
        assert translate_path("/c/foo") == "/c/foo"
        assert translate_path("/B/x") == "/B/x"
        assert translate_path("/mnt/c/foo") == "/mnt/c/foo"

    def test_to_windows_native(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _force_dialect(monkeypatch, PathDialect.WINDOWS_NATIVE)
        monkeypatch.setattr(dialect_mod.os.path, "lexists", lambda _: False)
        assert translate_path("/c/git/foo") == "C:\\git\\foo"
        assert translate_path("/mnt/c/git/foo") == "C:\\git\\foo"
        assert translate_path("/cygdrive/c/git/foo") == "C:\\git\\foo"
        # already in target form: untouched
        assert translate_path("C:\\git\\foo") == "C:\\git\\foo"

    def test_to_wsl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _force_dialect(monkeypatch, PathDialect.WSL)
        monkeypatch.setattr(dialect_mod.os.path, "lexists", lambda _: False)
        assert translate_path("C:\\git\\foo") == "/mnt/c/git/foo"
        assert translate_path("c:/git/foo") == "/mnt/c/git/foo"
        assert translate_path("/c/git/foo") == "/mnt/c/git/foo"
        assert translate_path("/cygdrive/c/git/foo") == "/mnt/c/git/foo"

    def test_to_cygwin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _force_dialect(monkeypatch, PathDialect.CYGWIN)
        monkeypatch.setattr(dialect_mod.os.path, "lexists", lambda _: False)
        assert translate_path("C:\\git\\foo") == "/cygdrive/c/git/foo"
        assert translate_path("c:/git/foo") == "/cygdrive/c/git/foo"
        assert translate_path("/c/git/foo") == "/cygdrive/c/git/foo"

    def test_drive_root_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _force_dialect(monkeypatch, PathDialect.WSL)
        monkeypatch.setattr(dialect_mod.os.path, "lexists", lambda _: False)
        assert translate_path("C:\\") == "/mnt/c"
        assert translate_path("/c") == "/mnt/c"

    def test_no_drive_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _force_dialect(monkeypatch, PathDialect.WINDOWS_NATIVE)
        monkeypatch.setattr(dialect_mod.os.path, "lexists", lambda _: False)
        assert translate_path("/home/u/x") == "/home/u/x"
        assert translate_path("./relative") == "./relative"
        assert translate_path("") == ""

    def test_real_path_wins_over_translation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """If the raw path actually exists, don't rewrite it."""
        _force_dialect(monkeypatch, PathDialect.WSL)
        real = tmp_path / "x"
        real.write_text("ok")
        # The raw form exists, so even though it has no drive prefix it stays.
        assert translate_path(str(real)) == str(real)


class TestToPosixForMatch:
    def test_drive_letter_to_slash_form(self) -> None:
        assert to_posix_for_match("C:\\foo\\bar") == "/c/foo/bar"
        assert to_posix_for_match("c:/foo/bar") == "/c/foo/bar"

    def test_already_posix(self) -> None:
        assert to_posix_for_match("/home/u/.env") == "/home/u/.env"

    def test_mnt_form_normalized(self) -> None:
        assert to_posix_for_match("/mnt/c/secrets/.env") == "/c/secrets/.env"


class TestDialectHint:
    def test_none_for_posix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _force_dialect(monkeypatch, PathDialect.POSIX)
        assert dialect_hint() is None

    def test_present_for_each_windows_dialect(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for d in (
            PathDialect.WINDOWS_NATIVE,
            PathDialect.GIT_BASH,
            PathDialect.WSL,
            PathDialect.CYGWIN,
        ):
            _force_dialect(monkeypatch, d)
            hint = dialect_hint()
            assert hint is not None
            assert "Filesystem" in hint

    def test_emitted_even_when_translation_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The hint states the environment (where Windows drives live), which
        the model still needs whether or not we translate for it. Per-call
        path_note on tool results handles the teaching when translation is on;
        the system prompt's job is just the environment fact."""
        _force_dialect(monkeypatch, PathDialect.WSL)
        configure_path_translation(enabled=False)
        hint = dialect_hint()
        assert hint is not None
        assert "/mnt/c" in hint

    def test_hint_does_not_promise_translation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: earlier the hint advertised auto-translation. That role
        belongs to the per-call path_note now, so the system-prompt hint must
        not duplicate it."""
        for d in (
            PathDialect.WINDOWS_NATIVE,
            PathDialect.GIT_BASH,
            PathDialect.WSL,
            PathDialect.CYGWIN,
        ):
            _force_dialect(monkeypatch, d)
            hint = dialect_hint()
            assert hint is not None
            assert "translate" not in hint.lower()
            assert "auto" not in hint.lower()


class TestTranslationDisabled:
    def test_pass_through_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _force_dialect(monkeypatch, PathDialect.WSL)
        configure_path_translation(enabled=False)
        assert translate_path("C:\\foo\\bar") == "C:\\foo\\bar"
        assert translate_path("/c/foo") == "/c/foo"

    def test_aliases_disabled_when_translation_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One-knob design: disabling translation also disables aliases,
        because aliases feed into the same pipeline."""
        _force_dialect(monkeypatch, PathDialect.WSL)
        configure_path_translation(
            enabled=False, aliases={"/IDontExists": "C:\\IDontExists"}
        )
        assert translate_path("/IDontExists/sub") == "/IDontExists/sub"


class TestAliases:
    def test_alias_applied_before_auto(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _force_dialect(monkeypatch, PathDialect.WSL)
        monkeypatch.setattr(dialect_mod.os.path, "lexists", lambda _: False)
        configure_path_translation(
            enabled=True, aliases={"/IDontExists": "C:\\IDontExists"}
        )
        # Alias maps to a Windows path, then auto-translation converts to
        # /mnt/c form on WSL.
        assert translate_path("/IDontExists/sub") == "/mnt/c/IDontExists/sub"

    def test_longest_prefix_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _force_dialect(monkeypatch, PathDialect.POSIX)
        configure_path_translation(
            enabled=True,
            aliases={"/data": "/A", "/data/team": "/B"},
        )
        assert translate_path("/data/team/x") == "/B/x"
        assert translate_path("/data/other") == "/A/other"

    def test_no_alias_match_passes_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _force_dialect(monkeypatch, PathDialect.POSIX)
        configure_path_translation(
            enabled=True, aliases={"/foo": "/bar"}
        )
        assert translate_path("/baz/qux") == "/baz/qux"

    def test_alias_only_no_auto_on_posix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On POSIX dialect, auto-translation is a no-op but aliases still apply."""
        _force_dialect(monkeypatch, PathDialect.POSIX)
        configure_path_translation(
            enabled=True, aliases={"/old": "/new"}
        )
        assert translate_path("/old/path") == "/new/path"
