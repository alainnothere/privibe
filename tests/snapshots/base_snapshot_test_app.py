from __future__ import annotations

from rich.style import Style
from textual.widgets.text_area import TextAreaTheme

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.stubs.fake_backend import FakeBackend
from privibe.cli.textual_ui.app import VibeApp
from privibe.cli.textual_ui.widgets.chat_input import ChatTextArea
from privibe.core.agents.models import BuiltinAgentName
from privibe.core.config import VibeConfig


def default_config() -> VibeConfig:
    """Default configuration for snapshot testing.
    Remove as much interference as possible from the snapshot comparison, in order to get a clean pixel-to-pixel comparison.
    - Injects a fake backend to prevent (or stub) LLM calls.
    - Disables the banner animation.
    - Forces a value for the displayed workdir
    - Hides the chat input cursor (as the blinking animation is not deterministic).
    """
    return build_test_vibe_config(
        disable_welcome_banner_animation=True, displayed_workdir="/test/workdir"
    )


class BaseSnapshotTestApp(VibeApp):
    CSS_PATH = "../../privibe/cli/textual_ui/app.tcss"
    _current_agent_name: str = BuiltinAgentName.DEFAULT

    def __init__(
        self,
        config: VibeConfig | None = None,
        backend: FakeBackend | None = None,
        **kwargs,
    ):
        agent_loop = build_test_agent_loop(
            config=config or default_config(),
            agent_name=self._current_agent_name,
            enable_streaming=bool(kwargs.get("enable_streaming", False)),
            backend=backend or FakeBackend(),
        )

        super().__init__(agent_loop=agent_loop, **kwargs)

    async def on_mount(self) -> None:
        await super().on_mount()
        self._hide_chat_input_cursor()

    def _hide_chat_input_cursor(self) -> None:
        text_area = self.query_one(ChatTextArea)
        hidden_cursor_theme = TextAreaTheme(name="hidden_cursor", cursor_style=Style())
        text_area.register_theme(hidden_cursor_theme)
        text_area.theme = "hidden_cursor"
