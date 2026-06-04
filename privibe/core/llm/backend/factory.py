from __future__ import annotations

from typing import Any

from privibe.core.integration_registry import (
    _backends,
    discover_integrations,
    get_backend,
    get_registered_backend_keys,
    register_backend,
)
from privibe.core.llm.backend.generic import GenericBackend


register_backend("generic", GenericBackend)
discover_integrations()


def get_backend_class(backend_key: str) -> Any:
    cls = get_backend(backend_key)
    if cls is None:
        available = get_registered_backend_keys()
        raise KeyError(
            f"Unknown backend {backend_key!r}; available: {available}. "
            "Install or enable the integration that provides this backend."
        )
    return cls


class _BackendFactoryProxy:
    def __getitem__(self, key: str) -> Any:
        return get_backend_class(str(key))

    def __setitem__(self, key: str, value: Any) -> None:
        _backends[str(key)] = value

    def __contains__(self, key: str) -> bool:
        return get_backend(str(key)) is not None


BACKEND_FACTORY = _BackendFactoryProxy()
