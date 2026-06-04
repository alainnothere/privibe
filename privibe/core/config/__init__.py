from __future__ import annotations

from privibe.core.config._settings import (
    DEFAULT_TRANSCRIBE_MODELS,
    DEFAULT_TRANSCRIBE_PROVIDERS,
    DEFAULT_TTS_MODELS,
    DEFAULT_TTS_PROVIDERS,
    PATHS_TEMPLATE_FILE,
    MCPHttp,
    MCPServer,
    MCPStdio,
    MCPStreamableHttp,
    MissingAPIKeyError,
    MissingPromptFileError,
    ModelConfig,
    PathConfig,
    ProjectContextConfig,
    ProviderConfig,
    SessionLoggingConfig,
    TomlFileSettingsSource,
    TranscribeClient,
    TranscribeModelConfig,
    TranscribeProviderConfig,
    TTSClient,
    TTSModelConfig,
    TTSProviderConfig,
    VibeConfig,
    cycle_message_prune_rows,
    cycle_preview_lines,
    load_dotenv_values,
)


def __getattr__(name: str):  # noqa: D401 — PEP 562 lazy re-export
    if name in ("DEFAULT_PROVIDERS", "DEFAULT_MODELS"):
        from privibe.core.config import _settings

        return getattr(_settings, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_MODELS",
    "DEFAULT_PROVIDERS",
    "DEFAULT_TRANSCRIBE_MODELS",
    "DEFAULT_TRANSCRIBE_PROVIDERS",
    "DEFAULT_TTS_MODELS",
    "DEFAULT_TTS_PROVIDERS",
    "PATHS_TEMPLATE_FILE",
    "MCPHttp",
    "MCPServer",
    "MCPStdio",
    "MCPStreamableHttp",
    "MissingAPIKeyError",
    "MissingPromptFileError",
    "ModelConfig",
    "PathConfig",
    "ProjectContextConfig",
    "ProviderConfig",
    "SessionLoggingConfig",
    "TTSClient",
    "TTSModelConfig",
    "TTSProviderConfig",
    "TomlFileSettingsSource",
    "TranscribeClient",
    "TranscribeModelConfig",
    "TranscribeProviderConfig",
    "VibeConfig",
    "cycle_message_prune_rows",
    "cycle_preview_lines",
    "load_dotenv_values",
]
