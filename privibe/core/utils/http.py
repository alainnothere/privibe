from __future__ import annotations

import re

from privibe import __version__


def get_user_agent(backend_key: str | None) -> str:
    user_agent = f"Privibe/{__version__}"
    if not backend_key:
        return user_agent
    # Lazy import to avoid a cycle with paths/logger/integration_registry
    # at package-init time.
    from privibe.core.integration_registry import get_backend

    cls = get_backend(backend_key)
    if cls is None:
        return user_agent
    prefix = getattr(cls, "USER_AGENT_PREFIX", "")
    if prefix:
        return f"{prefix}{user_agent}"
    return user_agent


def get_server_url_from_api_base(api_base: str) -> str | None:
    match = re.match(r"(https?://[^/]+)(/v.*)", api_base)
    return match.group(1) if match else None
