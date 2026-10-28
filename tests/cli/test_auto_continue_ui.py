"""Auto-continue in the two UIs: the dim nudge bubble, /continue,
/auto-continue.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from privibe.cli.commands import CommandRegistry
from privibe.cli.console_ui.app import ConsoleUI
from privibe.cli.textual_ui.app import BottomApp
from privibe.cli.textual_ui.widgets.messages import ErrorMessage, UserMessage
from privibe.cli.textual_ui.widgets.option_picker import OptionPickerApp
from privibe.core.types import UserMessageEvent
from tests.cli.test_console_ui import StubAgentLoop
from tests.conftest import build_test_vibe_app


def test_commands_registered() -> None:
    registry = CommandRegistry()
    cont, _ = registry.parse_command("/continue")
    assert cont is not None and cont.handler == "_continue_turn"
    auto, args = registry.parse_command("/auto-continue heuristic")
    assert auto is not None and auto.handler == "_select_auto_continue" and args == "heuristic"


@pytest.mark.asyncio
async def test_nudge_event_mounts_a_dim_user_bubble() -> None:
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        assert app.event_handler is not None
        await app.event_handler.handle_event(
            UserMessageEvent(
                content="Your previous reply was cut off.",
                message_id="m1",
                message_index=3,
                auto_continue=True,
            )
        )
        await pilot.pause(0.2)
        bubbles = list(app.query(UserMessage))
        assert len(bubbles) == 1
        assert bubbles[0].has_class("auto-continue")
        assert bubbles[0].auto_continue
        assert bubbles[0].message_index == 3
        assert bubbles[0].get_content() == "Your previous reply was cut off."


@pytest.mark.asyncio
async def test_continue_sends_a_plain_message_unless_running() -> None:
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        with patch.object(app, "_handle_user_message", new=AsyncMock()) as send:
            assert await app._handle_command("/continue")
            await pilot.pause(0.1)
            send.assert_awaited_once_with("Continue.")

        app._agent_running = True
        with patch.object(app, "_handle_user_message", new=AsyncMock()) as send:
            assert await app._handle_command("/continue")
            await pilot.pause(0.1)
            send.assert_not_awaited()
        errors = [str(m._error) for m in app.query(ErrorMessage)]
        assert any("still running" in e for e in errors)


@pytest.mark.asyncio
async def test_auto_continue_picker_and_typed_value() -> None:
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        assert await app._handle_command("/auto-continue")
        await pilot.pause(0.2)
        assert app._current_bottom_app == BottomApp.OptionPicker
        assert app.query_one(OptionPickerApp)._setting == "auto_continue"
        await pilot.press("escape")
        await pilot.pause(0.2)

        with patch("privibe.cli.textual_ui.app.VibeConfig.save_updates") as save:
            assert await app._handle_command("/auto-continue heuristic")
            await pilot.pause(0.2)
            save.assert_called_once_with({"auto_continue": "heuristic"})

        assert await app._handle_command("/auto-continue sideways")
        await pilot.pause(0.2)
        assert app._current_bottom_app == BottomApp.OptionPicker


@pytest.mark.asyncio
async def test_console_prints_the_nudge_and_takes_the_mode(capsys) -> None:
    ui = ConsoleUI(StubAgentLoop())
    ui._handle_event(
        UserMessageEvent(content="Go on and do it now.", message_id="m", auto_continue=True)
    )
    assert "[auto-continue] Go on and do it now." in capsys.readouterr().out

    ui.agent_loop.config.auto_continue = "off"
    ui.agent_loop.config.auto_continue_max = 3
    ui.agent_loop.refresh_config = lambda: None
    ui._command_auto_continue("")
    assert "Auto-continue: off (up to 3 nudges" in capsys.readouterr().out
    ui._command_auto_continue("sideways")
    assert "not an auto-continue mode" in capsys.readouterr().out
    with patch("privibe.cli.console_ui.app.VibeConfig.save_updates") as save:
        ui._command_auto_continue("truncated")
        save.assert_called_once_with({"auto_continue": "truncated"})
    assert "Auto-continue: truncated." in capsys.readouterr().out
