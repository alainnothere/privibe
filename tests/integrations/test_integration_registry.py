from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import pytest

from privibe.core.integration_registry import (
    _backends,
    _config_defaults,
    _tool_overrides,
    discover_integrations,
    get_all_config_defaults,
    get_backend,
    get_registered_backend_keys,
    get_registered_tool_overrides,
    get_tool_override,
    register_backend,
    register_config_defaults,
    register_tool_override,
)


@pytest.fixture(autouse=True)
def reset_registry():
    saved_backends = dict(_backends)
    saved_tools = dict(_tool_overrides)
    saved_defaults = list(_config_defaults)
    _backends.clear()
    _tool_overrides.clear()
    _config_defaults.clear()
    yield
    _backends.clear()
    _backends.update(saved_backends)
    _tool_overrides.clear()
    _tool_overrides.update(saved_tools)
    _config_defaults.clear()
    _config_defaults.extend(saved_defaults)


class _FakeBackend:
    pass


class _OtherBackend:
    pass


class _FakeTool:
    pass


class _OtherTool:
    pass


def test_register_backend_stores_under_key():
    register_backend("foo", _FakeBackend)
    assert get_backend("foo") is _FakeBackend
    assert get_registered_backend_keys() == ["foo"]


def test_get_backend_unknown_key_returns_none():
    assert get_backend("nope") is None


def test_register_backend_collision_keeps_first(caplog):
    register_backend("foo", _FakeBackend)
    with caplog.at_level(logging.WARNING, logger="privibe"):
        register_backend("foo", _OtherBackend)
    assert get_backend("foo") is _FakeBackend
    assert any(
        "foo" in r.getMessage() and "ignored" in r.getMessage()
        for r in caplog.records
    )


def test_register_tool_override_stores_under_name():
    register_tool_override("web_search", _FakeTool)
    assert get_tool_override("web_search") is _FakeTool
    assert get_registered_tool_overrides() == ["web_search"]


def test_register_tool_override_collision_keeps_first(caplog):
    register_tool_override("web_search", _FakeTool)
    with caplog.at_level(logging.WARNING, logger="privibe"):
        register_tool_override("web_search", _OtherTool)
    assert get_tool_override("web_search") is _FakeTool
    assert any(
        "web_search" in r.getMessage() and "ignored" in r.getMessage()
        for r in caplog.records
    )


def test_register_config_defaults_accumulates_in_order():
    register_config_defaults({"providers": [{"name": "a"}]})
    register_config_defaults({"providers": [{"name": "b"}]})
    assert get_all_config_defaults() == [
        {"providers": [{"name": "a"}]},
        {"providers": [{"name": "b"}]},
    ]


def test_discover_missing_directory_is_noop(tmp_path):
    discover_integrations(tmp_path / "nonexistent")
    assert get_registered_backend_keys() == []


def test_discover_empty_directory_is_noop(tmp_path):
    discover_integrations(tmp_path)
    assert get_registered_backend_keys() == []


def test_discover_skips_non_packages_and_underscore_prefixed(tmp_path):
    (tmp_path / "loose_file.py").write_text("")
    (tmp_path / "no_init_dir").mkdir()
    (tmp_path / "_underscore").mkdir()
    (tmp_path / "_underscore" / "__init__.py").write_text(
        "raise RuntimeError('should not be imported')"
    )
    discover_integrations(tmp_path, module_prefix="should_not_be_used")
    assert get_registered_backend_keys() == []


def _write_pkg(parent: Path, name: str, body: str) -> Path:
    pkg = parent / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text(textwrap.dedent(body).lstrip())
    return pkg


def test_discover_imports_integration_and_runs_registration(tmp_path, monkeypatch):
    fake_root = tmp_path / "fake_root"
    fake_root.mkdir()
    (fake_root / "__init__.py").write_text("")
    _write_pkg(
        fake_root,
        "fake_int",
        """
        from privibe.core.integration_registry import register_backend

        class FakeBackend:
            pass

        register_backend("fake", FakeBackend)
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    discover_integrations(fake_root, module_prefix="fake_root")
    assert "fake" in get_registered_backend_keys()


def test_discover_soft_fails_on_import_error(tmp_path, monkeypatch, caplog):
    fake_root = tmp_path / "fake_root_broken"
    fake_root.mkdir()
    (fake_root / "__init__.py").write_text("")
    _write_pkg(fake_root, "broken_int", "raise RuntimeError('boom')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="privibe"):
        discover_integrations(fake_root, module_prefix="fake_root_broken")
    assert any(
        "broken_int" in r.getMessage() and "boom" in r.getMessage()
        for r in caplog.records
    )
    assert get_registered_backend_keys() == []
