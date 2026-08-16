from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Markdown

from privibe.cli.textual_ui.widgets.messages import ReasoningMessage, cap_open_block
from privibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from tests.conftest import build_test_vibe_config


# --- cap_open_block: bounds the still-open markdown block ---


def test_cap_accumulates_unbroken_text():
    text, open_len = cap_open_block("aaaa", 0, 100)
    assert text == "aaaa"
    assert open_len == 4


def test_cap_resets_on_paragraph_break():
    text, open_len = cap_open_block("aaa\n\nbb", 90, 100)
    assert text == "aaa\n\nbb"
    assert open_len == 2


def test_cap_injects_break_at_cap():
    text, open_len = cap_open_block("aaaa", 97, 100)
    assert text == "aaaa\n\n"
    assert open_len == 0


def test_cap_injects_when_tail_after_break_is_too_long():
    text, open_len = cap_open_block("a\n\n" + "b" * 100, 0, 100)
    assert text.endswith("\n\n")
    assert open_len == 0


def test_cap_trailing_break_resets_to_zero():
    text, open_len = cap_open_block("aaa\n\n", 50, 100)
    assert text == "aaa\n\n"
    assert open_len == 0


# --- config flag ---


def test_render_reasoning_markdown_defaults_off():
    assert build_test_vibe_config().render_reasoning_markdown is False


# --- widget behavior in both modes ---


class _Harness(App):
    def __init__(self, widget: ReasoningMessage) -> None:
        super().__init__()
        self._widget = widget

    def compose(self) -> ComposeResult:
        yield self._widget


@pytest.mark.asyncio
async def test_plain_mode_streams_into_static_not_markdown():
    msg = ReasoningMessage("", collapsed=False, markdown=False)
    app = _Harness(msg)
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        assert len(msg.query(Markdown)) == 0
        plain = msg.query_one(".reasoning-message-content-plain", NoMarkupStatic)

        await msg.append_content("first thoughts ")
        await msg.append_content("more thoughts")
        await msg.stop_stream()
        await pilot.pause(0.1)

        assert "first thoughts more thoughts" in str(plain.render())


@pytest.mark.asyncio
async def test_plain_mode_collapsed_defers_render_until_expanded():
    msg = ReasoningMessage("", collapsed=True, markdown=False)
    app = _Harness(msg)
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        plain = msg.query_one(".reasoning-message-content-plain", NoMarkupStatic)

        await msg.append_content("hidden thinking")
        await pilot.pause(0.1)
        assert plain.display is False
        assert "hidden thinking" not in str(plain.render())

        await msg.set_collapsed(False)
        await pilot.pause(0.1)
        assert plain.display is True
        assert "hidden thinking" in str(plain.render())


@pytest.mark.asyncio
async def test_markdown_mode_still_uses_markdown_widget():
    msg = ReasoningMessage("", collapsed=False, markdown=True)
    app = _Harness(msg)
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        assert len(msg.query(Markdown)) == 1
        assert len(msg.query(".reasoning-message-content-plain")) == 0


@pytest.mark.asyncio
async def test_markdown_mode_caps_open_block():
    msg = ReasoningMessage("", collapsed=False, markdown=True)
    app = _Harness(msg)
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        # Stream an unbroken ramble past the cap; the counter resetting to a
        # small value proves a paragraph break was injected along the way.
        chunk = "x" * 1024
        for _ in range(6):
            await msg.append_content(chunk)
            await pilot.pause(0.06)
        await msg.stop_stream()

        assert msg._open_block_len < msg.OPEN_BLOCK_CAP
