from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from privibe.cli.commands import CommandRegistry
from privibe.cli.textual_ui.app import BottomApp, _options_with_current
from privibe.cli.textual_ui.widgets.chat_input.container import ChatInputContainer
from privibe.cli.textual_ui.widgets.option_picker import OptionPickerApp
from tests.conftest import build_test_vibe_app


# --- bare command opens the picker ---


@pytest.mark.asyncio
async def test_bare_effort_opens_picker() -> None:
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        assert await app._handle_command("/effort")
        await pilot.pause(0.2)

        assert app._current_bottom_app == BottomApp.OptionPicker
        assert len(app.query(OptionPickerApp)) == 1


@pytest.mark.asyncio
async def test_picker_escape_returns_to_input_without_saving() -> None:
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        assert await app._handle_command("/scrollback")
        await pilot.pause(0.2)

        with patch("privibe.cli.textual_ui.app.VibeConfig.save_updates") as mock_save:
            await pilot.press("escape")
            await pilot.pause(0.2)

            mock_save.assert_not_called()

        assert app._current_bottom_app == BottomApp.Input
        assert len(app.query(OptionPickerApp)) == 0


@pytest.mark.asyncio
async def test_effort_picker_select_applies_value() -> None:
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        assert await app._handle_command("/effort")
        await pilot.pause(0.2)

        # Preselected on "off"; one down lands on "low".
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause(0.2)

        assert app.agent_loop.current_reasoning_effort() == "low"
        assert app._current_bottom_app == BottomApp.Input


@pytest.mark.asyncio
async def test_scrollback_picker_select_saves_value() -> None:
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        assert await app._handle_command("/scrollback")
        await pilot.pause(0.2)

        # Preselected on the current value (default 250); Enter re-saves it.
        with patch("privibe.cli.textual_ui.app.VibeConfig.save_updates") as mock_save:
            await pilot.press("enter")
            await pilot.pause(0.2)

            mock_save.assert_called_once_with({"message_prune_keep_rows": 250})

        assert app._current_bottom_app == BottomApp.Input


# --- direct argument sets without a picker ---


@pytest.mark.asyncio
async def test_effort_with_argument_sets_directly() -> None:
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        assert await app._handle_command("/effort low")
        await pilot.pause(0.2)

        assert app.agent_loop.current_reasoning_effort() == "low"
        assert app._current_bottom_app == BottomApp.Input
        assert len(app.query(OptionPickerApp)) == 0


@pytest.mark.asyncio
async def test_preview_lines_with_argument_sets_directly() -> None:
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        with patch("privibe.cli.textual_ui.app.VibeConfig.save_updates") as mock_save:
            assert await app._handle_command("/preview-lines 5")
            await pilot.pause(0.2)

            mock_save.assert_called_once_with({"tool_result_preview_lines": 5})

        assert len(app.query(OptionPickerApp)) == 0


# --- bad argument shows the error and the picker ---


@pytest.mark.asyncio
async def test_bad_effort_value_shows_error_and_picker() -> None:
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        with patch("privibe.cli.textual_ui.app.VibeConfig.save_updates") as mock_save:
            assert await app._handle_command("/effort turbo")
            await pilot.pause(0.2)

            mock_save.assert_not_called()

        assert app._current_bottom_app == BottomApp.OptionPicker
        picker = app.query_one(OptionPickerApp)
        error = picker.query_one(".optionpicker-error")
        assert "turbo" in str(error.render())


@pytest.mark.asyncio
async def test_bad_scrollback_value_shows_error_and_picker() -> None:
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        with patch("privibe.cli.textual_ui.app.VibeConfig.save_updates") as mock_save:
            assert await app._handle_command("/scrollback lots")
            await pilot.pause(0.2)

            mock_save.assert_not_called()

        assert app._current_bottom_app == BottomApp.OptionPicker


# --- any positive number is accepted, not just the presets ---


@pytest.mark.asyncio
async def test_preview_lines_accepts_arbitrary_positive_number() -> None:
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        with patch("privibe.cli.textual_ui.app.VibeConfig.save_updates") as mock_save:
            assert await app._handle_command("/preview-lines 7")
            await pilot.pause(0.2)

            mock_save.assert_called_once_with({"tool_result_preview_lines": 7})

        assert len(app.query(OptionPickerApp)) == 0


@pytest.mark.asyncio
async def test_scrollback_accepts_arbitrary_positive_number() -> None:
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        with patch("privibe.cli.textual_ui.app.VibeConfig.save_updates") as mock_save:
            assert await app._handle_command("/scrollback 123")
            await pilot.pause(0.2)

            mock_save.assert_called_once_with({"message_prune_keep_rows": 123})


@pytest.mark.asyncio
async def test_negative_number_shows_error_and_picker() -> None:
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        with patch("privibe.cli.textual_ui.app.VibeConfig.save_updates") as mock_save:
            assert await app._handle_command("/preview-lines -3")
            await pilot.pause(0.2)

            mock_save.assert_not_called()

        assert app._current_bottom_app == BottomApp.OptionPicker


def test_options_with_current_inserts_custom_value_sorted() -> None:
    assert _options_with_current([3, 5, 10], 7) == [3, 5, 7, 10]
    assert _options_with_current([3, 5, 10], 5) == [3, 5, 10]


# --- /detect-context-size dual mode ---


@pytest.mark.asyncio
async def test_detect_context_size_bare_opens_picker() -> None:
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        assert await app._handle_command("/detect-context-size")
        await pilot.pause(0.2)

        assert app._current_bottom_app == BottomApp.OptionPicker


@pytest.mark.asyncio
async def test_detect_context_size_off_argument() -> None:
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        with patch("privibe.cli.textual_ui.app.VibeConfig.save_updates") as mock_save:
            assert await app._handle_command("/detect-context-size off")
            await pilot.pause(0.2)

            mock_save.assert_called_once_with(
                {"auto_detect_context_size": False, "context_size_redetect_every": 0}
            )

        assert len(app.query(OptionPickerApp)) == 0


@pytest.mark.asyncio
async def test_detect_context_size_arbitrary_cadence_argument() -> None:
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        with (
            patch("privibe.cli.textual_ui.app.VibeConfig.save_updates") as mock_save,
            patch.object(
                app.agent_loop, "resolve_context_size", AsyncMock(return_value=None)
            ),
        ):
            assert await app._handle_command("/detect-context-size 3")
            await pilot.pause(0.2)

            mock_save.assert_called_once_with(
                {"auto_detect_context_size": True, "context_size_redetect_every": 3}
            )


@pytest.mark.asyncio
async def test_detect_context_size_bad_value_shows_error_and_picker() -> None:
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        with patch("privibe.cli.textual_ui.app.VibeConfig.save_updates") as mock_save:
            assert await app._handle_command("/detect-context-size banana")
            await pilot.pause(0.2)

            mock_save.assert_not_called()

        assert app._current_bottom_app == BottomApp.OptionPicker
        picker = app.query_one(OptionPickerApp)
        error = picker.query_one(".optionpicker-error")
        assert "banana" in str(error.render())


# --- completion popup shows current values ---


def test_slash_entries_include_current_values() -> None:
    container = ChatInputContainer(
        command_registry=CommandRegistry(),
        current_values_getter=lambda: {"/effort": "low"},
    )
    entries = dict(container._get_slash_entries())
    assert entries["/effort"].endswith("(current: low)")
    assert "(current:" not in entries["/help"]


def test_slash_entries_without_getter_stay_plain() -> None:
    container = ChatInputContainer(command_registry=CommandRegistry())
    entries = dict(container._get_slash_entries())
    assert "(current:" not in entries["/effort"]
