from __future__ import annotations

from unittest.mock import patch

import pytest
from textual.selection import Selection
from textual.widget import Widget

from privibe.cli.clipboard import copy_selection_to_clipboard
from privibe.cli.textual_ui.app import VibeApp
from privibe.cli.textual_ui.widgets.messages import WarningMessage
from tests.conftest import build_test_vibe_app, build_test_vibe_config


class ClipboardSelectionWidget(Widget):
    def __init__(self, selected_text: str) -> None:
        super().__init__()
        self._selected_text = selected_text

    @property
    def text_selection(self) -> Selection | None:
        return Selection(None, None)

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        return (self._selected_text, "\n")


@pytest.mark.asyncio
async def test_startup_warning_shown_when_no_clipboard_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup WarningMessage is mounted when autocopy is on and no reliable tool found."""
    monkeypatch.setattr("privibe.cli.textual_ui.app.is_reliable_clipboard_available", lambda: False)
    app = build_test_vibe_app(config=build_test_vibe_config(autocopy_to_clipboard=True))
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        warnings = app.query(WarningMessage)
        assert len(warnings) > 0
        assert any("xclip" in w._message for w in warnings)


@pytest.mark.asyncio
async def test_startup_warning_not_shown_when_clipboard_tool_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No clipboard WarningMessage when a reliable tool is present."""
    monkeypatch.setattr("privibe.cli.textual_ui.app.is_reliable_clipboard_available", lambda: True)
    app = build_test_vibe_app(config=build_test_vibe_config(autocopy_to_clipboard=True))
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        warnings = app.query(WarningMessage)
        assert len(warnings) == 0


@pytest.mark.asyncio
async def test_startup_warning_not_shown_when_autocopy_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No clipboard warning when autocopy is disabled, even without clipboard tools."""
    monkeypatch.setattr("privibe.cli.textual_ui.app.is_reliable_clipboard_available", lambda: False)
    app = build_test_vibe_app(config=build_test_vibe_config(autocopy_to_clipboard=False))
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        warnings = app.query(WarningMessage)
        assert len(warnings) == 0


@pytest.mark.asyncio
async def test_ui_clipboard_notification_does_not_crash_on_markup_text(
    monkeypatch: pytest.MonkeyPatch, vibe_app: VibeApp
) -> None:
    async with vibe_app.run_test(notifications=True) as pilot:
        await vibe_app.mount(ClipboardSelectionWidget("[/]"))
        with patch("privibe.cli.clipboard._copy_to_clipboard"):
            copy_selection_to_clipboard(vibe_app)

        await pilot.pause(0.1)
        notifications = list(vibe_app._notifications)
        assert notifications
        notification = notifications[-1]
        assert notification.markup is False
        assert "copied to clipboard" in notification.message
