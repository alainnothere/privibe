"""/ensure-commands-safety in the TUI: picker when off, actions when on."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from privibe.cli.textual_ui.app import BottomApp
from privibe.cli.textual_ui.widgets.nanny_status import NannyStatus
from privibe.cli.textual_ui.widgets.option_picker import OptionPickerApp
from tests.conftest import build_test_vibe_app


@pytest.mark.asyncio
async def test_command_when_off_opens_the_model_picker_and_attaches() -> None:
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        assert app._nanny is None
        assert app._nanny_status_label() == "off"
        assert await app._handle_command("/ensure-commands-safety")
        await pilot.pause(0.2)
        assert app._current_bottom_app == BottomApp.OptionPicker
        picker = app.query_one(OptionPickerApp)
        assert picker._setting == "nanny_model"

        with patch("privibe.cli.textual_ui.app.VibeConfig.save_updates") as mock_save:
            await pilot.press("enter")
            await pilot.pause(0.4)
            mock_save.assert_called_once()
            assert mock_save.call_args.args[0] == {"nanny_model": "devstral-latest"}

        assert app._nanny is not None
        assert app.agent_loop.nanny is app._nanny
        assert app._nanny_status_label() == "devstral-latest"
        assert app.query_one(NannyStatus)._nanny is app._nanny
        assert app._current_bottom_app == BottomApp.Input


@pytest.mark.asyncio
async def test_command_when_on_offers_actions_and_off_detaches() -> None:
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        with patch("privibe.cli.textual_ui.app.VibeConfig.save_updates"):
            await app._apply_nanny_model("devstral-latest")
            await pilot.pause(0.2)
            assert app._nanny is not None

            assert await app._handle_command("/ensure-commands-safety")
            await pilot.pause(0.2)
            picker = app.query_one(OptionPickerApp)
            assert picker._setting == "nanny_action"
            await pilot.press("escape")
            await pilot.pause(0.2)
            assert app._nanny is not None

            assert await app._handle_command("/ensure-commands-safety off")
            await pilot.pause(0.3)
        assert app._nanny is None
        assert app.agent_loop.nanny is None
        assert app.query_one(NannyStatus)._nanny is None


@pytest.mark.asyncio
async def test_unknown_alias_is_reported() -> None:
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        with patch("privibe.cli.textual_ui.app.VibeConfig.save_updates") as mock_save:
            assert await app._handle_command("/ensure-commands-safety ghost")
            await pilot.pause(0.2)
            mock_save.assert_not_called()
        assert app._nanny is None
