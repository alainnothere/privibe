from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from privibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from privibe.cli.textual_ui.widgets.tools import ToolResultMessage
from privibe.core.types import ToolResultEvent


class _Host(App):
    """Minimal app that mounts a single ToolResultMessage."""

    def __init__(self, event: ToolResultEvent) -> None:
        super().__init__()
        self._event = event

    def compose(self) -> ComposeResult:
        yield ToolResultMessage(self._event)


def _err_event(error: str) -> ToolResultEvent:
    return ToolResultEvent(
        tool_name="bash", tool_class=None, error=error, tool_call_id="t"
    )


def _skip_event(reason: str) -> ToolResultEvent:
    return ToolResultEvent(
        tool_name="bash",
        tool_class=None,
        skipped=True,
        skip_reason=reason,
        tool_call_id="t",
    )


async def _render_texts(event: ToolResultEvent) -> list[str]:
    """Mount the message and return the text of each rendered static."""
    app = _Host(event)
    async with app.run_test():
        msg = app.query_one(ToolResultMessage)
        # _render_result runs in on_mount; default preview is 3 lines (no config).
        # Scope to the content container so the border widget is excluded.
        assert msg._content_container is not None
        statics = msg._content_container.query(NoMarkupStatic)
        return [str(s.render()) for s in statics]


@pytest.mark.asyncio
async def test_long_error_is_truncated_to_preview_lines():
    error = "\n".join(f"line {i}" for i in range(10))
    texts = await _render_texts(_err_event(error))

    # First static holds the clamped body: "Error: line 0" plus lines 1-2 = 3 lines.
    assert texts[0].count("\n") == 2
    assert texts[0].startswith("Error: line 0")
    # A hint static announces the remainder (10 lines, 3 shown -> 7 more).
    assert any("more lines" in t for t in texts[1:])


@pytest.mark.asyncio
async def test_short_error_renders_without_hint():
    texts = await _render_texts(_err_event("boom: file not found"))
    assert texts == ["Error: boom: file not found"]


@pytest.mark.asyncio
async def test_long_skip_reason_is_truncated():
    reason = "\n".join(f"reason {i}" for i in range(10))
    texts = await _render_texts(_skip_event(reason))

    assert texts[0].count("\n") == 2
    assert texts[0].startswith("Skipped: reason 0")
    assert any("more lines" in t for t in texts[1:])
