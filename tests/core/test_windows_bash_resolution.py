"""Tests for the Git Bash resolution chain in privibe.core.utils.platform.

The chain itself is pure path-walking and is exercised here on any platform by
forcing sys.platform to win32 and stubbing the smoke test; only the real
subprocess smoke test is Windows-specific and is not run here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import privibe.core.utils.platform as platform_mod
from privibe.core.utils.platform import resolve_windows_bash


@pytest.fixture(autouse=True)
def clear_resolver_cache():
    platform_mod._resolve_windows_bash_cached.cache_clear()
    yield
    platform_mod._resolve_windows_bash_cached.cache_clear()


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(platform_mod.sys, "platform", "win32")


@pytest.fixture
def no_auto_detection(monkeypatch):
    """Make the dynamic steps (git on PATH, which bash) find nothing, so tests
    only exercise the static search-path walk.
    """
    monkeypatch.setattr(platform_mod.shutil, "which", lambda _name: None)


def _working_bash(monkeypatch, *paths: Path) -> None:
    """Stub the smoke test: only the given paths count as a working bash."""
    allowed = {str(p) for p in paths}
    monkeypatch.setattr(
        platform_mod, "_bash_works", lambda p: str(p) in allowed
    )


def _make_exe(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return path


def test_returns_none_on_non_windows(monkeypatch):
    monkeypatch.setattr(platform_mod.sys, "platform", "linux")
    assert resolve_windows_bash() is None


def test_finds_exe_entry(windows, no_auto_detection, monkeypatch, tmp_path):
    bash = _make_exe(tmp_path / "Git" / "bin" / "bash.exe")
    _working_bash(monkeypatch, bash)

    assert resolve_windows_bash([str(bash)]) == str(bash)


def test_directory_entry_prefers_bin_wrapper(
    windows, no_auto_detection, monkeypatch, tmp_path
):
    root = tmp_path / "Git"
    wrapper = _make_exe(root / "bin" / "bash.exe")
    direct = _make_exe(root / "bash.exe")
    _working_bash(monkeypatch, wrapper, direct)

    assert resolve_windows_bash([str(root)]) == str(wrapper)


def test_directory_entry_falls_back_to_direct_bash(
    windows, no_auto_detection, monkeypatch, tmp_path
):
    root = tmp_path / "Git"
    direct = _make_exe(root / "bash.exe")
    _working_bash(monkeypatch, direct)

    assert resolve_windows_bash([str(root)]) == str(direct)


def test_failing_smoke_test_falls_through_to_next_entry(
    windows, no_auto_detection, monkeypatch, tmp_path
):
    broken = _make_exe(tmp_path / "broken" / "bash.exe")
    good = _make_exe(tmp_path / "good" / "bash.exe")
    _working_bash(monkeypatch, good)

    assert resolve_windows_bash([str(broken), str(good)]) == str(good)


def test_missing_file_falls_through(
    windows, no_auto_detection, monkeypatch, tmp_path
):
    missing = tmp_path / "nope" / "bash.exe"
    good = _make_exe(tmp_path / "good" / "bash.exe")
    _working_bash(monkeypatch, good)

    assert resolve_windows_bash([str(missing), str(good)]) == str(good)


def test_nothing_found_returns_none(windows, no_auto_detection, monkeypatch, tmp_path):
    _working_bash(monkeypatch)  # nothing works

    assert resolve_windows_bash([str(tmp_path / "nope" / "bash.exe")]) is None


def test_derives_bash_from_git_on_path(windows, monkeypatch, tmp_path):
    root = tmp_path / "Git"
    git = _make_exe(root / "cmd" / "git.exe")
    bash = _make_exe(root / "bin" / "bash.exe")
    _working_bash(monkeypatch, bash)
    monkeypatch.setattr(
        platform_mod.shutil,
        "which",
        lambda name: str(git) if name == "git" else None,
    )

    # Empty static list: opt-out of the static search, dynamic steps still run.
    assert resolve_windows_bash([]) == str(bash)


def test_which_bash_rejects_wsl_launcher(windows, monkeypatch, tmp_path):
    wsl = _make_exe(tmp_path / "Windows" / "System32" / "bash.exe")
    _working_bash(monkeypatch, wsl)
    monkeypatch.setattr(
        platform_mod.shutil,
        "which",
        lambda name: str(wsl) if name == "bash" else None,
    )

    assert resolve_windows_bash([]) is None


def test_explicit_entry_is_trusted_even_under_system32(
    windows, no_auto_detection, monkeypatch, tmp_path
):
    wsl = _make_exe(tmp_path / "Windows" / "System32" / "bash.exe")
    _working_bash(monkeypatch, wsl)

    assert resolve_windows_bash([str(wsl)]) == str(wsl)


def test_none_means_default_search_paths(windows, no_auto_detection, monkeypatch):
    seen: list[str] = []

    def record(p: Path) -> bool:
        seen.append(str(p))
        return False

    monkeypatch.setattr(platform_mod, "_bash_works", record)
    monkeypatch.setattr(platform_mod.Path, "is_file", lambda _self: True)

    assert resolve_windows_bash(None) is None
    expanded = [
        str(Path(platform_mod.os.path.expandvars(e)))
        for e in platform_mod.DEFAULT_BASH_SEARCH_PATHS
    ]
    assert seen == expanded
