"""/set-conversation-name in the TUI: the tab title, the loud one-line bar
above the chat, and the input-box border title next to the agent name.
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

import pytest

from privibe.cli.commands import CommandRegistry
from privibe.cli.textual_ui.notifications import (
    NotificationContext,
    TextualNotificationAdapter,
)
from privibe.cli.textual_ui.widgets.chat_input import ChatInputContainer
from privibe.cli.textual_ui.widgets.messages import UserCommandMessage
from privibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from tests.conftest import build_test_vibe_app


def _fake_app() -> MagicMock:
    app = MagicMock()
    type(app).is_headless = PropertyMock(return_value=False)
    return app


def _titles(app: MagicMock) -> list[str]:
    return [
        call.args[0].removeprefix("\x1b]0;").removesuffix("\x07")
        for call in app._driver.write.call_args_list
    ]


class TestAdapterDefaultTitle:
    def test_set_default_title_applies_immediately_when_focused(self) -> None:
        app = _fake_app()
        adapter = TextualNotificationAdapter(
            app, get_enabled=lambda: True, default_title="Privibe"
        )
        adapter.set_default_title("Privibe - parser")
        assert _titles(app)[-1] == "Privibe - parser"

    def test_set_default_title_waits_for_focus_when_blurred(self) -> None:
        app = _fake_app()
        adapter = TextualNotificationAdapter(
            app, get_enabled=lambda: True, default_title="Privibe"
        )
        adapter.on_blur()
        adapter.notify(NotificationContext.ACTION_REQUIRED)
        assert _titles(app)[-1] == "Privibe - Action Required"

        adapter.set_default_title("Privibe - parser")
        # Still showing the notification: the user has not come back yet.
        assert _titles(app)[-1] == "Privibe - Action Required"

        adapter.on_focus()
        assert _titles(app)[-1] == "Privibe - parser"

    def test_notification_suffix_follows_new_title(self) -> None:
        app = _fake_app()
        adapter = TextualNotificationAdapter(
            app, get_enabled=lambda: True, default_title="Privibe"
        )
        adapter.set_default_title("Privibe - parser")
        adapter.on_blur()
        adapter.notify(NotificationContext.COMPLETE)
        assert _titles(app)[-1] == "Privibe - parser - Task Complete"


class TestCommandRegistry:
    def test_registered_with_args(self) -> None:
        registry = CommandRegistry()
        command, args = registry.parse_command("/set-conversation-name fix the parser")
        assert command is not None
        assert command.handler == "_set_conversation_name"
        assert command.takes_args is True
        assert args == "fix the parser"

    def test_short_alias(self) -> None:
        registry = CommandRegistry()
        command, args = registry.parse_command("/name hi")
        assert command is not None
        assert command.handler == "_set_conversation_name"
        assert args == "hi"


@pytest.mark.asyncio
async def test_command_paints_every_surface_and_clears() -> None:
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        top_bar = app.query_one("#conversation-title-bar", NoMarkupStatic)
        input_box = app.query_one("#input-box")
        container = app.query_one(ChatInputContainer)
        agent_name = container._agent_name

        # Unnamed: nothing painted, top bar hidden.
        assert not top_bar.has_class("has-name")
        assert not top_bar.display
        assert input_box.border_title == agent_name

        # Typing "/..." into the input opens the completion popup and Enter
        # accepts a completion, so drive the command the way the other
        # command tests do.
        assert await app._handle_command("/set-conversation-name fix the parser")
        await pilot.pause(0.1)

        assert app.agent_loop.conversation_name == "fix the parser"
        assert top_bar.has_class("has-name")
        assert top_bar.display
        assert str(top_bar.content) == "fix the parser"
        # Loud on purpose: white bar, black text, one row.
        assert top_bar.styles.background.hex.lower() == "#ffffff"
        assert top_bar.styles.color.hex.lower() == "#000000"
        assert top_bar.size.height == 1
        assert input_box.border_title == f"fix the parser | {agent_name}"
        notices = [str(m._content) for m in app.query(UserCommandMessage)]
        assert any("fix the parser" in n for n in notices)

        # Bare command loads the current name for editing, clears nothing.
        assert await app._handle_command("/set-conversation-name")
        await pilot.pause(0.1)
        assert app.agent_loop.conversation_name == "fix the parser"
        input_text = app.query_one(
            ChatInputContainer
        )._body.input_widget.get_full_text()
        assert input_text == "/set-conversation-name fix the parser"
        notices = [str(m._content) for m in app.query(UserCommandMessage)]
        assert any("empty the line" in n for n in notices)

        # The user emptied the loaded line and pressed Enter: that clears.
        assert await app._handle_command("/set-conversation-name")
        await pilot.pause(0.1)

        assert app.agent_loop.conversation_name is None
        assert not top_bar.has_class("has-name")
        assert not top_bar.display
        assert input_box.border_title == agent_name


@pytest.mark.asyncio
async def test_tab_title_follows_name() -> None:
    notifier = MagicMock()
    app = build_test_vibe_app(terminal_notifier=notifier)
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        notifier.set_default_title.assert_called_with("Privibe")

        assert await app._handle_command("/name parser")
        notifier.set_default_title.assert_called_with("parser")

        assert await app._handle_command("/name")
        notifier.set_default_title.assert_called_with("parser")
        assert await app._handle_command("/name")
        notifier.set_default_title.assert_called_with("Privibe")


@pytest.mark.asyncio
async def test_custom_border_label_wins_over_name() -> None:
    """A recording/remote label replaces the title; the name returns after."""
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        container = app.query_one(ChatInputContainer)
        input_box = app.query_one("#input-box")

        container.set_conversation_name("parser")
        container.set_custom_border("recording", "border-recording")
        assert input_box.border_title == "recording"

        container.set_custom_border(None)
        assert input_box.border_title == f"parser | {container._agent_name}"


@pytest.mark.asyncio
async def test_current_value_shown_in_completion_entries() -> None:
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app.agent_loop.set_conversation_name("parser")
        entries = dict(app.query_one(ChatInputContainer)._get_slash_entries())
        assert "(current: parser)" in entries["/set-conversation-name"]


@pytest.mark.asyncio
async def test_anything_else_between_two_bare_calls_disarms_the_clear() -> None:
    """Bare, then some other command, then bare again: the second bare is a
    fresh load, not the clear the first one armed.
    """
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        assert await app._handle_command("/name keep me")
        assert await app._handle_command("/name")
        assert await app._handle_command("/help")
        assert await app._handle_command("/name")
        await pilot.pause(0.1)
        assert app.agent_loop.conversation_name == "keep me"
        input_text = app.query_one(
            ChatInputContainer
        )._body.input_widget.get_full_text()
        assert (
            input_text == "/name keep me"
            or input_text == "/set-conversation-name keep me"
        )


@pytest.mark.asyncio
async def test_bare_with_no_name_set_just_clears() -> None:
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        assert await app._handle_command("/name")
        await pilot.pause(0.1)
        assert app.agent_loop.conversation_name is None
        notices = [str(m._content) for m in app.query(UserCommandMessage)]
        assert any("cleared" in n for n in notices)
