from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pytest
import tomli_w

from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_voice_manager import FakeVoiceManager
from privibe.cli.textual_ui.app import StartupOptions, VibeApp
from privibe.core.agent_loop import AgentLoop
from privibe.core.agents.models import BuiltinAgentName
from privibe.core.config import (
    DEFAULT_MODELS,
    ModelConfig,
    SessionLoggingConfig,
    VibeConfig,
)
from privibe.core.config.harness_files import (
    init_harness_files_manager,
    reset_harness_files_manager,
)
from privibe.core.llm.types import BackendLike


def get_base_config() -> dict[str, Any]:
    return {
        "active_model": "devstral-latest",
        "providers": [
            {
                "name": "mistral",
                "api_base": "https://api.mistral.ai/v1",
                "api_key_env_var": "MISTRAL_API_KEY",
                "backend": "mistral",
            }
        ],
        "models": [
            {
                "name": "mistral-privibe-cli-latest",
                "provider": "mistral",
                "alias": "devstral-latest",
            }
        ],
    }


@pytest.fixture(autouse=True)
def tmp_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    tmp_working_directory = tmp_path_factory.mktemp("test_cwd")
    monkeypatch.chdir(tmp_working_directory)
    return tmp_working_directory


@pytest.fixture(autouse=True)
def config_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    tmp_path = tmp_path_factory.mktemp("privibe")
    config_dir = tmp_path / ".privibe"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.toml"
    config_file.write_text(tomli_w.dumps(get_base_config()), encoding="utf-8")

    monkeypatch.setattr("privibe.core.paths._vibe_home._DEFAULT_VIBE_HOME", config_dir)
    return config_dir


@pytest.fixture(autouse=True)
def _reset_trusted_folders_manager(config_dir: Path) -> None:
    """Prevent the singleton from writing to the real ~/.privibe/trusted_folders.toml.

    The module-level ``trusted_folders_manager`` captures its file path at import
    time (before any monkeypatch), so it would otherwise target the real home
    directory.  Redirect it to the temp config dir used by the ``config_dir``
    fixture.
    """
    from privibe.core.trusted_folders import trusted_folders_manager

    trusted_folders_manager._file_path = config_dir / "trusted_folders.toml"
    trusted_folders_manager._trusted = []
    trusted_folders_manager._untrusted = []


@pytest.fixture(autouse=True)
def _init_harness_files_manager():
    reset_harness_files_manager()
    init_harness_files_manager("user", "project")
    yield
    reset_harness_files_manager()


@pytest.fixture(autouse=True)
def _mock_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "mock")


@pytest.fixture(autouse=True)
def _mock_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock platform to be Linux with /bin/sh shell for consistent test behavior.

    This ensures that platform-specific system prompt generation is consistent
    across all tests regardless of the actual platform running the tests.
    """
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("SHELL", "/bin/sh")


@pytest.fixture(autouse=True)
def _disable_feedback_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "privibe.cli.textual_ui.widgets.feedback_bar.FEEDBACK_PROBABILITY", 0
    )


@pytest.fixture(autouse=True)
def _mock_clipboard_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent startup clipboard warning from appearing in tests by default."""
    monkeypatch.setattr(
        "privibe.cli.textual_ui.app.is_reliable_clipboard_available",
        lambda: True,
    )



@pytest.fixture
def vibe_app() -> VibeApp:
    return build_test_vibe_app()


@pytest.fixture
def agent_loop() -> AgentLoop:
    return build_test_agent_loop()


@pytest.fixture
def vibe_config() -> VibeConfig:
    return build_test_vibe_config()


def make_test_models(auto_compact_threshold: int) -> list[ModelConfig]:
    return [
        m.model_copy(update={"auto_compact_threshold": auto_compact_threshold})
        for m in DEFAULT_MODELS
    ]


def build_test_vibe_config(**kwargs) -> VibeConfig:
    session_logging = kwargs.pop("session_logging", None)
    resolved_session_logging = (
        SessionLoggingConfig(enabled=False)
        if session_logging is None
        else session_logging
    )
    if kwargs.get("models"):
        kwargs.setdefault("active_model", kwargs["models"][0].alias)
    return VibeConfig(
        session_logging=resolved_session_logging,
        **kwargs,
    )


def build_test_agent_loop(
    *,
    config: VibeConfig | None = None,
    agent_name: str = BuiltinAgentName.DEFAULT,
    backend: BackendLike | None = None,
    enable_streaming: bool = False,
    **kwargs,
) -> AgentLoop:

    resolved_config = config or build_test_vibe_config()

    return AgentLoop(
        config=resolved_config,
        agent_name=agent_name,
        backend=backend or FakeBackend(),
        enable_streaming=enable_streaming,
        **kwargs,
    )


def build_test_vibe_app(
    *, config: VibeConfig | None = None, agent_loop: AgentLoop | None = None, **kwargs
) -> VibeApp:
    app_config = config or build_test_vibe_config()

    resolved_agent_loop = agent_loop or build_test_agent_loop(config=app_config)

    voice_manager = kwargs.pop("voice_manager", FakeVoiceManager())

    return VibeApp(
        agent_loop=resolved_agent_loop,
        startup=StartupOptions(initial_prompt=kwargs.pop("initial_prompt", None)),
        voice_manager=voice_manager,
        **kwargs,
    )
