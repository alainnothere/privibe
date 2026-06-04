from __future__ import annotations

from privibe.core.integration_registry import (
    register_backend,
    register_config_defaults,
    register_tool_override,
)
from privibe.integrations.mistral.backend import MistralBackend
from privibe.integrations.mistral.web_search import MistralWebSearchImpl

register_backend("mistral", MistralBackend)
register_tool_override("web_search", MistralWebSearchImpl)

register_config_defaults(
    {
        "providers": [
            {
                "name": "mistral",
                "api_base": "https://api.mistral.ai/v1",
                "api_key_env_var": "MISTRAL_API_KEY",
                "backend": "mistral",
                "stream_tool_calls": True,
            }
        ],
        "models": [
            {
                "name": "devstral-latest",
                "provider": "mistral",
                "alias": "devstral-2",
                "input_price": 0.4,
                "output_price": 2.0,
            },
            {
                "name": "devstral-small-latest",
                "provider": "mistral",
                "alias": "devstral-small",
                "input_price": 0.1,
                "output_price": 0.3,
            },
        ],
    }
)
