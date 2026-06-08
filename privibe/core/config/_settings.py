from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from enum import StrEnum, auto
import os
from pathlib import Path
import re
import shlex
import tomllib
from typing import Annotated, Any, Literal, NamedTuple
from dotenv import dotenv_values
from pydantic import (
    BaseModel,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic.fields import FieldInfo
from pydantic_core import to_jsonable_python
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
import tomli_w

from privibe.core.config.harness_files import get_harness_files_manager
from privibe.core.logger import logger
from privibe.core.paths import GLOBAL_ENV_FILE, SESSION_LOG_DIR
from privibe.core.prompts import SystemPrompt
from privibe.core.utils.io import read_safe

# Path to the documented [paths] template. Bootstrap copies its contents
# verbatim into the user's config.toml so the comments and example aliases
# survive (tomli_w does not preserve comments on its own).
PATHS_TEMPLATE_FILE = Path(__file__).parent / "default_config.toml"


def load_dotenv_values(
    env_path: Path = GLOBAL_ENV_FILE.path,
    environ: MutableMapping[str, str] = os.environ,
) -> None:
    # We allow FIFO path to support some environment management solutions (e.g. https://developer.1password.com/docs/environments/local-env-file/)
    if not env_path.is_file() and not env_path.is_fifo():
        return

    env_vars = dotenv_values(env_path)
    for key, value in env_vars.items():
        if not value:
            continue
        environ.update({key: value})


class ModelApiKeyStatus(NamedTuple):
    api_key_set: bool
    env_var_name: str
    provider_name: str


class MissingAPIKeyError(RuntimeError):
    def __init__(self, env_key: str, provider_name: str) -> None:
        super().__init__(
            f"Missing {env_key} environment variable for {provider_name} provider"
        )
        self.env_key = env_key
        self.provider_name = provider_name


class MissingPromptFileError(RuntimeError):
    def __init__(self, system_prompt_id: str, *prompt_dirs: str) -> None:
        dirs_str = " or ".join(prompt_dirs) if prompt_dirs else "<no prompt dirs>"
        super().__init__(
            f"Invalid system_prompt_id value: '{system_prompt_id}'. "
            f"Must be one of the available prompts ({', '.join(f'{p.name.lower()}' for p in SystemPrompt)}), "
            f"or correspond to a .md file in {dirs_str}"
        )
        self.system_prompt_id = system_prompt_id


class TomlFileSettingsSource(PydanticBaseSettingsSource):
    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        self.toml_data = self._load_toml()

    def _load_toml(self) -> dict[str, Any]:
        file = get_harness_files_manager().config_file
        if file is None:
            return {}
        try:
            with file.open("rb") as f:
                return tomllib.load(f)
        except FileNotFoundError:
            return {}
        except tomllib.TOMLDecodeError as e:
            raise RuntimeError(f"Invalid TOML in {file}: {e}") from e
        except OSError as e:
            raise RuntimeError(f"Cannot read {file}: {e}") from e

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        return self.toml_data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return self.toml_data


class ProjectContextConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    default_commit_count: int = 5
    timeout_seconds: float = 2.0


class PathConfig(BaseSettings):
    """Path translation behaviour for cross-dialect file tools.

    WHY THIS DOC LIVES IN TWO PLACES:
        These defaults are mirrored in privibe/core/config/default_config.toml,
        which is the user-facing template (it ships with comments and example
        aliases on first install). The roundtrip test
        tests/core/test_path_config_roundtrip.py loads that template and
        asserts it produces an equal PathConfig() — so if you change a
        default here, update the template too, and vice versa, or the test
        will fail loudly.

    WHAT enable_translation DOES:
        When true (the default), the read/write/grep/edit tools translate
        Windows / WSL / Git Bash / Cygwin path forms into whatever the
        running interpreter expects. When false, paths are passed verbatim
        and `aliases` below also stop applying — alias substitution feeds
        into the same translation pipeline, so disabling one disables both.

    WHAT aliases ARE:
        Manual prefix substitutions applied BEFORE auto-translation. They
        exist for paths the auto-detector can't probe its way to via the
        filesystem: unusual mounts, mapped drives, etc.
    """

    model_config = SettingsConfigDict(extra="ignore")

    enable_translation: bool = True
    aliases: dict[str, str] = Field(default_factory=dict)


class SessionLoggingConfig(BaseSettings):
    save_dir: str = ""
    session_prefix: str = "session"
    enabled: bool = True
    resume_preview_lines: int = 2

    @field_validator("save_dir", mode="before")
    @classmethod
    def set_default_save_dir(cls, v: str) -> str:
        if not v:
            return str(SESSION_LOG_DIR.path)
        return v

    @field_validator("save_dir", mode="after")
    @classmethod
    def expand_save_dir(cls, v: str) -> str:
        return str(Path(v).expanduser().resolve())


class ProviderConfig(BaseModel):
    name: str
    api_base: str
    api_key_env_var: str = ""
    api_style: str = "openai"
    backend: str = "generic"
    reasoning_field_name: str = "reasoning_content"
    project_id: str = ""
    region: str = ""
    stream_tool_calls: bool = False


class TranscribeClient(StrEnum):
    WHISPERLIVE = auto()


class TranscribeProviderConfig(BaseModel):
    name: str
    api_base: str = "ws://localhost:9090"
    api_key_env_var: str = ""
    client: TranscribeClient = TranscribeClient.WHISPERLIVE


class _MCPBase(BaseModel):
    name: str = Field(description="Short alias used to prefix tool names")
    prompt: str | None = Field(
        default=None, description="Optional usage hint appended to tool descriptions"
    )
    startup_timeout_sec: float = Field(
        default=10.0,
        gt=0,
        description="Timeout in seconds for the server to start and initialize.",
    )
    tool_timeout_sec: float = Field(
        default=60.0, gt=0, description="Timeout in seconds for tool execution."
    )
    sampling_enabled: bool = Field(
        default=True,
        description="Allow this MCP server to request LLM completions via sampling/createMessage.",
    )

    @field_validator("name", mode="after")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9_-]", "_", v)
        normalized = normalized.strip("_-")
        return normalized[:256]


class _MCPHttpFields(BaseModel):
    url: str = Field(description="Base URL of the MCP HTTP server")
    headers: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Additional HTTP headers when using 'http' transport (e.g., Authorization or X-API-Key)."
        ),
    )
    api_key_env: str = Field(
        default="",
        description=(
            "Environment variable name containing an API token to send for HTTP transport."
        ),
    )
    api_key_header: str = Field(
        default="Authorization",
        description=(
            "HTTP header name to carry the token when 'api_key_env' is set (e.g., 'Authorization' or 'X-API-Key')."
        ),
    )
    api_key_format: str = Field(
        default="Bearer {token}",
        description=(
            "Format string for the header value when 'api_key_env' is set. Use '{token}' placeholder."
        ),
    )

    def http_headers(self) -> dict[str, str]:
        hdrs = dict(self.headers or {})
        env_var = (self.api_key_env or "").strip()
        if env_var and (token := os.getenv(env_var)):
            target = (self.api_key_header or "").strip() or "Authorization"
            if not any(h.lower() == target.lower() for h in hdrs):
                try:
                    value = (self.api_key_format or "{token}").format(token=token)
                except Exception:
                    value = token
                hdrs[target] = value
        return hdrs


class MCPHttp(_MCPBase, _MCPHttpFields):
    transport: Literal["http"]


class MCPStreamableHttp(_MCPBase, _MCPHttpFields):
    transport: Literal["streamable-http"]


class MCPStdio(_MCPBase):
    transport: Literal["stdio"]
    command: str | list[str]
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables to set for the MCP server process.",
    )

    def argv(self) -> list[str]:
        base = (
            shlex.split(self.command)
            if isinstance(self.command, str)
            else list(self.command or [])
        )
        return [*base, *self.args] if self.args else base


MCPServer = Annotated[
    MCPHttp | MCPStreamableHttp | MCPStdio, Field(discriminator="transport")
]


def _default_alias_to_name(data: Any) -> Any:
    if isinstance(data, dict):
        if "alias" not in data or data["alias"] is None:
            data["alias"] = data.get("name")
    return data


class ModelConfig(BaseModel):
    name: str
    provider: str
    alias: str
    temperature: float = 0.2
    input_price: float = 0.0  # Price per million input tokens
    output_price: float = 0.0  # Price per million output tokens
    thinking: Literal["off", "low", "medium", "high"] = "off"
    auto_compact_threshold: int = 200_000

    _default_alias_to_name = model_validator(mode="before")(_default_alias_to_name)


class TranscribeModelConfig(BaseModel):
    name: str
    provider: str
    alias: str
    sample_rate: int = 16000
    encoding: Literal["pcm_s16le"] = "pcm_s16le"
    language: str = "en"
    target_streaming_delay_ms: int = 500

    _default_alias_to_name = model_validator(mode="before")(_default_alias_to_name)


class TTSClient(StrEnum):
    OPENAI = auto()


class TTSProviderConfig(BaseModel):
    name: str
    api_base: str = ""
    api_key_env_var: str = ""
    client: TTSClient = TTSClient.OPENAI


class TTSModelConfig(BaseModel):
    name: str
    provider: str
    alias: str
    voice: str = "alloy"
    response_format: str = "wav"

    _default_alias_to_name = model_validator(mode="before")(_default_alias_to_name)


_CORE_DEFAULT_PROVIDERS = [
    ProviderConfig(
        name="llamacpp",
        api_base="http://127.0.0.1:8080/v1",
        api_key_env_var="",  # NOTE: if you wish to use --api-key in llama-server, change this value
        stream_tool_calls=True,
    ),
]

_CORE_DEFAULT_MODELS = [
    ModelConfig(
        name="devstral",
        provider="llamacpp",
        alias="local",
        input_price=0.0,
        output_price=0.0,
    ),
]


def _build_default_providers() -> list[ProviderConfig]:
    from privibe.core.integration_registry import (
        discover_integrations,
        get_all_config_defaults,
    )

    discover_integrations()
    integration_providers: list[ProviderConfig] = []
    for defaults in get_all_config_defaults():
        for entry in defaults.get("providers", []):
            integration_providers.append(ProviderConfig(**entry))
    return integration_providers + list(_CORE_DEFAULT_PROVIDERS)


def _build_default_models() -> list[ModelConfig]:
    from privibe.core.integration_registry import (
        discover_integrations,
        get_all_config_defaults,
    )

    discover_integrations()
    integration_models: list[ModelConfig] = []
    for defaults in get_all_config_defaults():
        for entry in defaults.get("models", []):
            integration_models.append(ModelConfig(**entry))
    return integration_models + list(_CORE_DEFAULT_MODELS)


def __getattr__(name: str):  # noqa: D401 — PEP 562 module-level lazy attrs
    if name == "DEFAULT_PROVIDERS":
        return _build_default_providers()
    if name == "DEFAULT_MODELS":
        return _build_default_models()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

DEFAULT_TRANSCRIBE_PROVIDERS = [
    TranscribeProviderConfig(
        name="whisperlive", api_base="ws://localhost:9090", api_key_env_var=""
    )
]

DEFAULT_TRANSCRIBE_MODELS = [
    TranscribeModelConfig(name="small", provider="whisperlive", alias="whisper")
]

DEFAULT_TTS_PROVIDERS = [
    TTSProviderConfig(name="tts", api_base="", api_key_env_var="")
]

DEFAULT_TTS_MODELS = [
    TTSModelConfig(name="tts-1", provider="tts", alias="tts")
]


TOOL_RESULT_PREVIEW_OPTIONS = (3, 5, 10)
MESSAGE_PRUNE_KEEP_OPTIONS = (50, 100, 250, 500, 1000)
# The single /detect-context-size control cycles through poll cadences. 0 here
# means "auto" (detect on model change + retry, no polling); 1/2/5/10 also force
# a re-detect every N turns. "off" is a separate state held by
# auto_detect_context_size=False (see cycle_context_size_mode).
_CONTEXT_SIZE_POLL_CYCLE = (0, 1, 2, 5, 10)


def _next_in_cycle(current: int, options: tuple[int, ...]) -> int:
    """Next value after *current* in *options*, wrapping; first if not present."""
    try:
        idx = options.index(current)
    except ValueError:
        return options[0]
    return options[(idx + 1) % len(options)]


def sanitize_cycle_options(value: Any, default: tuple[int, ...]) -> list[int]:
    """Keep the usable options from a user-supplied cycle list, falling back to
    *default* when nothing usable remains.

    Accepts a list/tuple and keeps positive ints, preserving order while dropping
    bools, non-ints, non-positive values, and duplicates. Returns ``list(default)``
    when the input isn't a sequence or has no usable entries — so a malformed
    config value degrades to the built-in cycle instead of breaking startup.
    """
    if isinstance(value, (list, tuple)):
        cleaned: list[int] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int):
                continue
            if item > 0 and item not in cleaned:
                cleaned.append(item)
        if cleaned:
            return cleaned
    return list(default)


def sanitize_positive_int(value: Any, default: int) -> int:
    """Return *value* if it's a positive int, else *default*."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return default
    return value


def cycle_preview_lines(
    current: int, options: Sequence[int] = TOOL_RESULT_PREVIEW_OPTIONS
) -> int:
    """Next value in the tool-result preview-line cycle (default 3 -> 5 -> 10)."""
    return _next_in_cycle(current, tuple(options))


def cycle_message_prune_rows(
    current: int, options: Sequence[int] = MESSAGE_PRUNE_KEEP_OPTIONS
) -> int:
    """Next value in the kept-message-rows cycle (default 50 -> ... -> 1000)."""
    return _next_in_cycle(current, tuple(options))


def cycle_context_size_mode(auto_detect: bool, every: int) -> tuple[bool, int]:
    """Advance the single /detect-context-size control one step.

    One control, two fields: returns the next (auto_detect_context_size,
    context_size_redetect_every) pair. Cycle is
    off -> auto (detect on model change, no polling) -> every 1 -> 2 -> 5 -> 10
    -> back to off, so no confusing master/cadence mismatch is reachable.
    """
    if not auto_detect:
        return (True, 0)  # off -> auto
    try:
        idx = _CONTEXT_SIZE_POLL_CYCLE.index(every)
    except ValueError:
        return (True, 0)  # unknown cadence -> auto
    nxt = idx + 1
    if nxt >= len(_CONTEXT_SIZE_POLL_CYCLE):
        return (False, 0)  # past 'every 10' -> off
    return (True, _CONTEXT_SIZE_POLL_CYCLE[nxt])


def context_size_mode_label(auto_detect: bool, every: int) -> str:
    """Human-readable label for the current /detect-context-size state."""
    if not auto_detect:
        return "off"
    if every == 0:
        return "auto (re-detect when the model changes)"
    return f"re-detect every {every} turn(s)"


class VibeConfig(BaseSettings):
    active_model: str = "local"
    vim_keybindings: bool = False
    disable_welcome_banner_animation: bool = False
    autocopy_to_clipboard: bool = True
    auto_detect_context_size: bool = True
    context_size_redetect_every: int = 0
    tool_result_preview_lines: int = 3
    tool_result_preview_options: list[int] = Field(
        default_factory=lambda: list(TOOL_RESULT_PREVIEW_OPTIONS),
        description="Values /preview-lines cycles through (positive ints).",
    )
    message_prune_keep_rows: int = 250
    message_prune_keep_options: list[int] = Field(
        default_factory=lambda: list(MESSAGE_PRUNE_KEEP_OPTIONS),
        description="Values /scrollback cycles through (positive ints).",
    )
    file_watcher_for_autocomplete: bool = False
    displayed_workdir: str = ""
    context_warnings: bool = False
    voice_mode_enabled: bool = False
    narrator_enabled: bool = False
    active_transcribe_model: str = "whisper"
    active_tts_model: str = "tts"
    auto_approve: bool = False
    # `preflight_warmup` was removed in 0.1.0 — it actively harmed
    # first-turn prefill on hybrid/recurrent (Qwen3.5 SSM) models. Old
    # configs with this key still load (extra="ignore" on the model).
    llm_debug_dump: bool = False  # DEBUG LLM COMMUNICATIONS — When True, dumps messages + payload to ~/.privibe/debug/ before each LLM call
    system_prompt_id: str = "cli"
    include_commit_signature: bool = True
    include_model_info: bool = True
    include_project_context: bool = True
    include_prompt_detail: bool = True
    # Experimental: keep the volatile datetime + project context (git/tree) OUT
    # of the system prompt and send them as a separate first injected message, so
    # the large static system prompt stays a stable prefix the server can keep
    # KV-cached across sessions. Opt-in; default off preserves current behavior.
    stable_system_prefix: bool = False
    enable_notifications: bool = True
    api_timeout: float = 720.0
    auto_compact_threshold: int = 200_000

    project_scan_depth: int = Field(
        default=0,
        ge=0,
        description=(
            "How many directory levels deep privibe will scan for .privibe/ and .agents/ "
            "project config dirs. 0 = current directory only (fastest). "
            "The original default was 4."
        ),
    )

    providers: list[ProviderConfig] = Field(default_factory=_build_default_providers)
    models: list[ModelConfig] = Field(default_factory=_build_default_models)
    compaction_model: ModelConfig | None = None

    transcribe_providers: list[TranscribeProviderConfig] = Field(
        default_factory=lambda: list(DEFAULT_TRANSCRIBE_PROVIDERS)
    )
    transcribe_models: list[TranscribeModelConfig] = Field(
        default_factory=lambda: list(DEFAULT_TRANSCRIBE_MODELS)
    )

    tts_providers: list[TTSProviderConfig] = Field(
        default_factory=lambda: list(DEFAULT_TTS_PROVIDERS)
    )
    tts_models: list[TTSModelConfig] = Field(
        default_factory=lambda: list(DEFAULT_TTS_MODELS)
    )

    project_context: ProjectContextConfig = Field(default_factory=ProjectContextConfig)
    session_logging: SessionLoggingConfig = Field(default_factory=SessionLoggingConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    tools: dict[str, dict[str, Any]] = Field(default_factory=dict)
    tool_paths: list[Path] = Field(
        default_factory=list,
        description=(
            "Additional directories or files to explore for custom tools. "
            "Paths may be absolute or relative to the current working directory. "
            "Directories are shallow-searched for tool definition files, "
            "while files are loaded directly if valid."
        ),
    )

    mcp_servers: list[MCPServer] = Field(
        default_factory=list, description="Preferred MCP server configuration entries."
    )

    enabled_tools: list[str] = Field(
        default_factory=list,
        description=(
            "An explicit list of tool names/patterns to enable. If set, only these"
            " tools will be active. Supports glob patterns (e.g., 'serena_*') and"
            " regex with 're:' prefix (e.g., 're:^serena_.*')."
        ),
    )
    disabled_tools: list[str] = Field(
        default_factory=list,
        description=(
            "A list of tool names/patterns to disable. Ignored if 'enabled_tools'"
            " is set. Supports glob patterns and regex with 're:' prefix."
        ),
    )
    agent_paths: list[Path] = Field(
        default_factory=list,
        description=(
            "Additional directories to search for custom agent profiles. "
            "Each path may be absolute or relative to the current working directory."
        ),
    )
    extra_instruction_files: list[Path] = Field(
        default_factory=list,
        description=(
            "Additional files whose contents are injected into the model's instructions. "
            "Paths may be absolute or relative to the current working directory. "
            "Silently ignored if a file does not exist or is empty."
        ),
    )
    enabled_agents: list[str] = Field(
        default_factory=list,
        description=(
            "An explicit list of agent names/patterns to enable. If set, only these"
            " agents will be available. Supports glob patterns (e.g., 'custom-*')"
            " and regex with 're:' prefix."
        ),
    )
    disabled_agents: list[str] = Field(
        default_factory=list,
        description=(
            "A list of agent names/patterns to disable. Ignored if 'enabled_agents'"
            " is set. Supports glob patterns and regex with 're:' prefix."
        ),
    )
    installed_agents: list[str] = Field(
        default_factory=list,
        description=(
            "A list of opt-in builtin agent names that have been explicitly installed."
        ),
    )
    skill_paths: list[Path] = Field(
        default_factory=list,
        description=(
            "Additional directories to search for skills. "
            "Each path may be absolute or relative to the current working directory."
        ),
    )
    enabled_skills: list[str] = Field(
        default_factory=list,
        description=(
            "An explicit list of skill names/patterns to enable. If set, only these"
            " skills will be active. Supports glob patterns (e.g., 'search-*') and"
            " regex with 're:' prefix."
        ),
    )
    disabled_skills: list[str] = Field(
        default_factory=list,
        description=(
            "A list of skill names/patterns to disable. Ignored if 'enabled_skills'"
            " is set. Supports glob patterns and regex with 're:' prefix."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="VIBE_", case_sensitive=False, extra="ignore"
    )

    @field_validator(
        "tool_result_preview_options", "message_prune_keep_options", mode="before"
    )
    @classmethod
    def _sanitize_cycle_option_lists(
        cls, value: Any, info: ValidationInfo
    ) -> list[int]:
        defaults = {
            "tool_result_preview_options": TOOL_RESULT_PREVIEW_OPTIONS,
            "message_prune_keep_options": MESSAGE_PRUNE_KEEP_OPTIONS,
        }
        return sanitize_cycle_options(value, defaults[info.field_name])

    @field_validator(
        "tool_result_preview_lines", "message_prune_keep_rows", mode="before"
    )
    @classmethod
    def _sanitize_cycle_current_values(cls, value: Any, info: ValidationInfo) -> int:
        defaults = {"tool_result_preview_lines": 3, "message_prune_keep_rows": 250}
        return sanitize_positive_int(value, defaults[info.field_name])

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)

    @property
    def system_prompt(self) -> str:
        try:
            return SystemPrompt[self.system_prompt_id.upper()].read()
        except KeyError:
            pass

        mgr = get_harness_files_manager()
        prompt_dirs = mgr.project_prompts_dirs + mgr.user_prompts_dirs
        for current_prompt_dir in prompt_dirs:
            custom_sp_path = (current_prompt_dir / self.system_prompt_id).with_suffix(
                ".md"
            )
            if custom_sp_path.is_file():
                return read_safe(custom_sp_path)

        raise MissingPromptFileError(
            self.system_prompt_id, *(str(d) for d in prompt_dirs)
        )

    def get_active_model(self) -> ModelConfig:
        matched: ModelConfig | None = None
        for model in self.models:
            if model.alias == self.active_model:
                matched = model
                break

        if matched is not None:
            status = self._is_environment_variable_set_for_model(matched)
            if status.api_key_set:
                return matched

        for model in self.models:
            status = self._is_environment_variable_set_for_model(model)
            if status.api_key_set:
                logger.warning(
                    "Active model '%s' unusable (not found or missing API key), falling back to '%s'",
                    self.active_model,
                    model.alias,
                )
                self.active_model = model.alias
                return model

        raise ValueError(
            f"Tried to load active model '{self.active_model}' but its API key environment "
            f"variable is not set, and no other configured model has its API key environment "
            f"variable set either."
        )

    def get_compaction_model(self) -> ModelConfig:
        if self.compaction_model is not None:
            return self.compaction_model
        return self.get_active_model()

    def get_provider_for_model(self, model: ModelConfig) -> ProviderConfig:
        for provider in self.providers:
            if provider.name == model.provider:
                return provider
        raise ValueError(
            f"Provider '{model.provider}' for model '{model.name}' not found in configuration."
        )

    def get_active_transcribe_model(self) -> TranscribeModelConfig:
        for model in self.transcribe_models:
            if model.alias == self.active_transcribe_model:
                return model
        raise ValueError(
            f"Active transcribe model '{self.active_transcribe_model}' not found in configuration."
        )

    def get_transcribe_provider_for_model(
        self, model: TranscribeModelConfig
    ) -> TranscribeProviderConfig:
        for provider in self.transcribe_providers:
            if provider.name == model.provider:
                return provider
        raise ValueError(
            f"Transcribe provider '{model.provider}' for transcribe model '{model.name}' not found in configuration."
        )

    def get_active_tts_model(self) -> TTSModelConfig:
        for model in self.tts_models:
            if model.alias == self.active_tts_model:
                return model
        raise ValueError(
            f"Active TTS model '{self.active_tts_model}' not found in configuration."
        )

    def get_tts_provider_for_model(self, model: TTSModelConfig) -> TTSProviderConfig:
        for provider in self.tts_providers:
            if provider.name == model.provider:
                return provider
        raise ValueError(
            f"TTS provider '{model.provider}' for TTS model '{model.name}' not found in configuration."
        )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Define the priority of settings sources.

        Note: dotenv_settings is intentionally excluded. API keys and other
        non-config environment variables are stored in .env but loaded manually
        into os.environ for use by providers. Only VIBE_* prefixed environment
        variables (via env_settings) and TOML config are used for Pydantic settings.
        """
        return (
            init_settings,
            env_settings,
            TomlFileSettingsSource(settings_cls),
            file_secret_settings,
        )

    @model_validator(mode="after")
    def _apply_global_auto_compact_threshold(self) -> VibeConfig:
        self.models = [
            model
            if "auto_compact_threshold" in model.model_fields_set
            else model.model_copy(
                update={"auto_compact_threshold": self.auto_compact_threshold}
            )
            for model in self.models
        ]
        return self

    @model_validator(mode="after")
    def _check_compaction_model_provider(self) -> VibeConfig:
        if self.compaction_model is None:
            return self

        compaction_provider = self.get_provider_for_model(self.compaction_model)
        try:
            active_provider = self.get_provider_for_model(self.get_active_model())
        except ValueError:
            return self
        if active_provider.name != compaction_provider.name:
            raise ValueError(
                f"Compaction model '{self.compaction_model.alias}' uses provider "
                f"'{compaction_provider.name}' but active model uses provider "
                f"'{active_provider.name}'. They must share the same provider."
            )
        return self

    @model_validator(mode="after")
    def _check_api_key(self) -> VibeConfig:
        try:
            active_model = self.get_active_model()
            status = self._is_environment_variable_set_for_model(active_model)
            if not status.api_key_set:
                from privibe.core.logger import logger

                logger.warning(
                    "API key not set for provider '%s' (env var: %s). "
                    "Set %s or switch to a different model with /model.",
                    status.provider_name,
                    status.env_var_name,
                    status.env_var_name,
                )
        except ValueError:
            pass
        return self

    def _is_environment_variable_set_for_model(
        self, model: ModelConfig
    ) -> ModelApiKeyStatus:
        provider = self.get_provider_for_model(model)
        api_key_env = provider.api_key_env_var
        api_key_set = not api_key_env or bool(os.getenv(api_key_env))
        return ModelApiKeyStatus(
            api_key_set=api_key_set,
            env_var_name=api_key_env,
            provider_name=provider.name,
        )

    @field_validator("tool_paths", mode="before")
    @classmethod
    def _expand_tool_paths(cls, v: Any) -> list[Path]:
        if not v:
            return []
        return [Path(p).expanduser().resolve() for p in v]

    @field_validator("skill_paths", mode="before")
    @classmethod
    def _expand_skill_paths(cls, v: Any) -> list[Path]:
        if not v:
            return []
        return [Path(p).expanduser().resolve() for p in v]

    @field_validator("tools", mode="before")
    @classmethod
    def _normalize_tool_configs(cls, v: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(v, dict):
            return {}

        normalized: dict[str, dict[str, Any]] = {}
        for tool_name, tool_config in v.items():
            if isinstance(tool_config, dict):
                normalized[tool_name] = tool_config
            else:
                normalized[tool_name] = {}

        return normalized

    @model_validator(mode="after")
    def _validate_model_uniqueness(self) -> VibeConfig:
        seen_aliases: set[str] = set()
        for model in self.models:
            if model.alias in seen_aliases:
                raise ValueError(
                    f"Duplicate model alias found: '{model.alias}'. Aliases must be unique."
                )
            seen_aliases.add(model.alias)
        return self

    @model_validator(mode="after")
    def _validate_transcribe_model_uniqueness(self) -> VibeConfig:
        seen_aliases: set[str] = set()
        for model in self.transcribe_models:
            if model.alias in seen_aliases:
                raise ValueError(
                    f"Duplicate transcribe model alias found: '{model.alias}'. Aliases must be unique."
                )
            seen_aliases.add(model.alias)
        return self

    @model_validator(mode="after")
    def _validate_tts_model_uniqueness(self) -> VibeConfig:
        seen_aliases: set[str] = set()
        for model in self.tts_models:
            if model.alias in seen_aliases:
                raise ValueError(
                    f"Duplicate TTS model alias found: '{model.alias}'. Aliases must be unique."
                )
            seen_aliases.add(model.alias)
        return self

    @model_validator(mode="after")
    def _check_system_prompt(self) -> VibeConfig:
        _ = self.system_prompt
        return self

    @classmethod
    def save_updates(cls, updates: dict[str, Any]) -> None:
        if not get_harness_files_manager().persist_allowed:
            return
        current_config = TomlFileSettingsSource(cls).toml_data

        def deep_merge(target: dict, source: dict) -> None:
            for key, value in source.items():
                if (
                    key in target
                    and isinstance(target.get(key), dict)
                    and isinstance(value, dict)
                ):
                    deep_merge(target[key], value)
                elif (
                    key in target
                    and isinstance(target.get(key), list)
                    and isinstance(value, list)
                ):
                    if key in {
                        "providers",
                        "models",
                        "transcribe_providers",
                        "transcribe_models",
                        "tts_providers",
                        "tts_models",
                        "installed_agents",
                    }:
                        target[key] = value
                    else:
                        target[key] = list(set(value + target[key]))
                else:
                    target[key] = value

        deep_merge(current_config, updates)
        cls.dump_config(current_config)

    @classmethod
    def dump_config(cls, config: dict[str, Any]) -> None:
        mgr = get_harness_files_manager()
        if not mgr.persist_allowed:
            return
        target = mgr.config_file or mgr.user_config_file
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as f:
            tomli_w.dump(to_jsonable_python(config, exclude_none=True, fallback=str), f)

    @classmethod
    def _migrate(cls) -> None:
        mgr = get_harness_files_manager()
        if not mgr.persist_allowed:
            return
        file = mgr.config_file
        if file is None:
            return
        try:
            with file.open("rb") as f:
                data = tomllib.load(f)
        except (FileNotFoundError, tomllib.TOMLDecodeError, OSError):
            return

        bash_tools = data.get("tools", {}).get("bash", {})
        allowlist = bash_tools.get("allowlist")
        if allowlist is None or "find" not in allowlist:
            return

        allowlist.remove("find")
        cls.dump_config(data)

    @classmethod
    def load(cls, **overrides: Any) -> VibeConfig:
        cls._migrate()
        instance = cls(**(overrides or {}))
        # Apply the configured scan depth to the harness manager now that config is loaded.
        from privibe.core.config.harness_files import set_project_scan_depth
        set_project_scan_depth(instance.project_scan_depth)
        # Push path-translation knobs into the dialect module so the file
        # tools see them on the very first call. Done here (not lazily) so
        # the system prompt's dialect_hint matches what tools will actually do.
        from privibe.core.paths import configure_path_translation
        configure_path_translation(
            enabled=instance.paths.enable_translation,
            aliases=instance.paths.aliases,
        )
        return instance

    @classmethod
    def create_default(cls) -> dict[str, Any]:
        config = cls.model_construct()
        config_dict = config.model_dump(mode="json")

        from privibe.core.tools.manager import ToolManager

        tool_defaults = ToolManager.discover_tool_defaults()
        if tool_defaults:
            config_dict["tools"] = tool_defaults

        config_dict["extra_instruction_files"] = ["./someFolder/anotherFileWithInstructionsToRead.md"]

        return config_dict
