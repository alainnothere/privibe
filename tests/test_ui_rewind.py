from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_app
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from privibe.cli.textual_ui.app import BottomApp, VibeApp
from privibe.cli.textual_ui.widgets.chat_input.container import ChatInputContainer
from privibe.cli.textual_ui.widgets.messages import UserMessage
from privibe.cli.textual_ui.widgets.rewind_app import RewindApp


def _make_app(num_responses: int = 3) -> VibeApp:
    backend = FakeBackend([
        mock_llm_chunk(content=f"Response {i + 1}") for i in range(num_responses)
    ])
    agent_loop = build_test_agent_loop(backend=backend)
    return build_test_vibe_app(agent_loop=agent_loop)


async def _send_messages(pilot, messages: list[str]) -> None:
    for msg in messages:
        await pilot.press(*msg)
        await pilot.press("enter")
        await pilot.pause(0.4)


async def _enter_rewind_with_changes(pilot, app: VibeApp) -> tuple[RewindApp, AsyncMock]:
    """Open the rewind panel with the restore option present.

    Returns the panel and a mock standing in for the real rewind, so tests can
    assert whether a restore was requested without touching the filesystem.
    """
    rm = app.agent_loop.rewind_manager
    patch.object(rm, "has_file_changes_at", return_value=True).start()
    rewind_mock = AsyncMock(return_value=("hello", []))
    patch.object(rm, "rewind_to_message", rewind_mock).start()

    await pilot.press("alt+up")
    await pilot.app.workers.wait_for_complete()
    await pilot.pause(0.1)

    return app.query_one(RewindApp), rewind_mock


@pytest.mark.asyncio
async def test_rewind_mode_activates_on_alt_up() -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        await _send_messages(pilot, ["hello", "world"])

        await pilot.press("alt+up")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause(0.1)

        assert app._rewind_mode is True
        assert app._current_bottom_app == BottomApp.Rewind


@pytest.mark.asyncio
async def test_rewind_highlights_last_user_message() -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        await _send_messages(pilot, ["hello", "world"])

        await pilot.press("alt+up")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause(0.1)

        assert app._rewind_highlighted_widget is not None
        assert app._rewind_highlighted_widget.get_content() == "world"


@pytest.mark.asyncio
async def test_rewind_navigates_to_previous_message() -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        await _send_messages(pilot, ["hello", "world"])

        await pilot.press("alt+up")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause(0.1)
        await pilot.press("alt+up")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause(0.1)

        assert app._rewind_highlighted_widget is not None
        assert app._rewind_highlighted_widget.get_content() == "hello"


@pytest.mark.asyncio
async def test_rewind_navigates_down() -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        await _send_messages(pilot, ["hello", "world"])

        # Go up twice, then down once
        await pilot.press("alt+up")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause(0.1)
        await pilot.press("alt+up")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause(0.1)
        await pilot.press("alt+down")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause(0.1)

        assert app._rewind_highlighted_widget is not None
        assert app._rewind_highlighted_widget.get_content() == "world"


@pytest.mark.asyncio
async def test_rewind_escape_exits_mode() -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        await _send_messages(pilot, ["hello", "world"])

        await pilot.press("alt+up")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause(0.1)

        await pilot.press("escape")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause(0.1)

        assert app._rewind_mode is False
        assert app._rewind_highlighted_widget is None
        assert app._current_bottom_app == BottomApp.Input


@pytest.mark.asyncio
async def test_rewind_ctrl_p_n_alternate_bindings() -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        await _send_messages(pilot, ["hello", "world"])

        # ctrl+p should enter rewind mode
        await pilot.press("ctrl+p")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause(0.1)

        assert app._rewind_mode is True
        assert app._rewind_highlighted_widget is not None
        assert app._rewind_highlighted_widget.get_content() == "world"

        # ctrl+p again goes to previous
        await pilot.press("ctrl+p")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause(0.1)

        assert app._rewind_highlighted_widget is not None
        assert app._rewind_highlighted_widget.get_content() == "hello"

        # ctrl+n goes back
        await pilot.press("ctrl+n")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause(0.1)

        assert app._rewind_highlighted_widget is not None
        assert app._rewind_highlighted_widget.get_content() == "world"


@pytest.mark.asyncio
async def test_rewind_confirm_edits_message_and_prefills_input() -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        await _send_messages(pilot, ["hello", "world"])

        await pilot.press("alt+up")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause(0.1)

        # Confirm with enter (selects "Edit message from here")
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause(0.2)

        assert app._rewind_mode is False
        assert app._current_bottom_app == BottomApp.Input

        # Input should be pre-filled with the rewound message
        chat_input = app.query_one(ChatInputContainer)
        assert chat_input.value == "world"


@pytest.mark.asyncio
async def test_rewind_removes_messages_after_selected() -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        await _send_messages(pilot, ["first", "second", "third"])

        # Navigate to "second"
        await pilot.press("alt+up")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause(0.1)
        await pilot.press("alt+up")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause(0.1)

        assert app._rewind_highlighted_widget is not None
        assert app._rewind_highlighted_widget.get_content() == "second"

        # Confirm
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause(0.2)

        # Only "first" should remain as a UserMessage
        messages_area = app.query_one("#messages")
        user_widgets = [
            child for child in messages_area.children if isinstance(child, UserMessage)
        ]
        assert len(user_widgets) == 1
        assert user_widgets[0].get_content() == "first"


@pytest.mark.asyncio
async def test_rewind_skips_command_messages() -> None:
    """Slash-command echo messages (message_index=None) are not rewind-selectable."""
    app = _make_app()
    async with app.run_test() as pilot:
        await _send_messages(pilot, ["hello"])

        # Simulate a slash command inserting a UserMessage without message_index
        await app._mount_and_scroll(UserMessage("/model"))
        await pilot.pause(0.1)

        await _send_messages(pilot, ["world"])

        # First alt+up should land on "world", not the command message
        await pilot.press("alt+up")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause(0.1)

        assert app._rewind_highlighted_widget is not None
        assert app._rewind_highlighted_widget.get_content() == "world"

        # Second alt+up should land on "hello", skipping the command message
        await pilot.press("alt+up")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause(0.1)

        assert app._rewind_highlighted_widget is not None
        assert app._rewind_highlighted_widget.get_content() == "hello"


@pytest.mark.asyncio
async def test_rewind_does_not_activate_while_agent_running() -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        await _send_messages(pilot, ["hello"])

        app._agent_running = True

        await pilot.press("alt+up")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause(0.1)

        assert app._rewind_mode is False


@pytest.mark.asyncio
async def test_rewind_number_keys_do_not_commit() -> None:
    """Number keys are not shortcuts: committing must be a deliberate Enter."""
    app = _make_app()
    async with app.run_test() as pilot:
        await _send_messages(pilot, ["hello"])

        await pilot.press("alt+up")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause(0.1)

        await pilot.press("1")
        await pilot.press("2")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause(0.2)

        assert app._rewind_mode is True
        assert app._current_bottom_app == BottomApp.Rewind

        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause(0.2)

        assert app._rewind_mode is False
        assert app._current_bottom_app == BottomApp.Input


@pytest.mark.asyncio
async def test_rewind_defaults_to_not_restoring_files() -> None:
    """Enter on a freshly opened panel must never touch files."""
    app = _make_app()
    async with app.run_test() as pilot:
        await _send_messages(pilot, ["hello"])
        _panel, rewind_mock = await _enter_rewind_with_changes(pilot, app)

        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause(0.2)

        rewind_mock.assert_awaited_once()
        assert rewind_mock.await_args.kwargs["restore_files"] is False


@pytest.mark.asyncio
async def test_rewind_restore_requires_typed_code() -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        await _send_messages(pilot, ["hello"])
        panel, rewind_mock = await _enter_rewind_with_changes(pilot, app)

        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause(0.1)

        assert panel.is_confirming is True
        rewind_mock.assert_not_awaited()

        code = panel._confirm_code
        assert code is not None
        wrong = "99" if code != "99" else "10"
        await pilot.press(*wrong)
        await pilot.press("enter")
        await pilot.pause(0.1)

        # Wrong code: still gated, and the entry box is cleared for a new number.
        assert panel.is_confirming is True
        rewind_mock.assert_not_awaited()
        assert panel._confirm_input is not None
        assert panel._confirm_input.value == ""

        await pilot.press(*panel._confirm_code)
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause(0.2)

        rewind_mock.assert_awaited_once()
        assert rewind_mock.await_args.kwargs["restore_files"] is True


@pytest.mark.asyncio
async def test_rewind_escape_from_confirm_returns_to_options() -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        await _send_messages(pilot, ["hello"])
        panel, rewind_mock = await _enter_rewind_with_changes(pilot, app)

        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert panel.is_confirming is True

        await pilot.press("escape")
        await pilot.pause(0.1)

        assert panel.is_confirming is False
        assert app._rewind_mode is True
        rewind_mock.assert_not_awaited()

        await pilot.press("escape")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause(0.2)

        assert app._rewind_mode is False
        rewind_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_rewind_ctrl_c_exits_rewind_mode() -> None:
    app = _make_app()
    async with app.run_test() as pilot:
        await _send_messages(pilot, ["hello"])
        _panel, rewind_mock = await _enter_rewind_with_changes(pilot, app)

        await pilot.press("ctrl+c")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause(0.2)

        assert app._rewind_mode is False
        assert app._current_bottom_app == BottomApp.Input
        assert app.is_running is True
        rewind_mock.assert_not_awaited()
