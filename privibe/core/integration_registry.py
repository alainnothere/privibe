from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from privibe.core.logger import logger


_backends: dict[str, Any] = {}
_tool_overrides: dict[str, Any] = {}
_config_defaults: list[dict[str, Any]] = []


def register_backend(key: str, backend_class: Any) -> None:
    if key in _backends:
        existing_module = getattr(_backends[key], "__module__", "<unknown>")
        new_module = getattr(backend_class, "__module__", "<unknown>")
        logger.warning(
            "integration %r tried to register backend key %r "
            "(already registered by %r); ignored — pick a unique key",
            new_module,
            key,
            existing_module,
        )
        return
    _backends[key] = backend_class


def register_tool_override(tool_name: str, tool_class: Any) -> None:
    if tool_name in _tool_overrides:
        existing_module = getattr(_tool_overrides[tool_name], "__module__", "<unknown>")
        new_module = getattr(tool_class, "__module__", "<unknown>")
        logger.warning(
            "integration %r tried to override tool %r "
            "(already overridden by %r); ignored — only one override allowed per tool",
            new_module,
            tool_name,
            existing_module,
        )
        return
    _tool_overrides[tool_name] = tool_class


def register_config_defaults(defaults: dict[str, Any]) -> None:
    _config_defaults.append(defaults)


def get_backend(key: str) -> Any | None:
    return _backends.get(key)


def get_tool_override(tool_name: str) -> Any | None:
    return _tool_overrides.get(tool_name)


def get_all_config_defaults() -> list[dict[str, Any]]:
    return list(_config_defaults)


def get_registered_backend_keys() -> list[str]:
    return list(_backends.keys())


def get_registered_tool_overrides() -> list[str]:
    return list(_tool_overrides.keys())


def _clear_registry() -> None:
    _backends.clear()
    _tool_overrides.clear()
    _config_defaults.clear()


INTEGRATIONS_PATH = Path(__file__).parent.parent / "integrations"


def discover_integrations(
    path: Path | None = None,
    *,
    module_prefix: str = "privibe.integrations",
) -> None:
    if path is None:
        path = INTEGRATIONS_PATH
    if not path.exists() or not path.is_dir():
        return
    for entry in sorted(path.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("_") or entry.name.startswith("."):
            continue
        if not (entry / "__init__.py").exists():
            continue
        module_name = f"{module_prefix}.{entry.name}"
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            logger.warning(
                "failed to load integration %r: %s: %s",
                entry.name,
                type(exc).__name__,
                exc,
            )
