"""Layout regression tests for capped tool-result regions.

Textual skips a non-cells max-height when measuring an auto-height parent
(Widget._get_box_model), so a tcss `max-height: 50vh` on the scrollable
content let .tool-result-container size itself to the unclamped content
height: trailing blank rows and a stretched border below the region. The
cap is now applied in cells from Python (half_viewport_cap); these tests
pin the invariant that container, border, and content agree on height.
"""

from __future__ import annotations

import pytest

from privibe.cli.textual_ui.widgets.tools import ToolResultMessage
from privibe.core.types import FileDiff, ToolResultEvent
from tests.conftest import build_test_vibe_app


def _tall_diff_event(rows: int) -> ToolResultEvent:
    return ToolResultEvent(
        tool_name="hashed_replace_block",
        tool_class=None,
        tool_call_id="tc-layout-test",
        file_diff=FileDiff(
            path="x.py",
            kind="diff",
            hunks=[[("diff-context", f"line {i}") for i in range(rows)]],
        ),
    )


async def _mount_result(app, pilot, rows: int) -> None:
    chat = app.query_one("#chat")
    await chat.mount(ToolResultMessage(_tall_diff_event(rows)))
    await pilot.pause()
    await pilot.pause()


def _heights(app) -> tuple[int, int, int, int]:
    container = app.query_one(".tool-result-container")
    border = app.query_one(".tool-result-border")
    content = app.query_one(".tool-result-content")
    return (
        container.size.height,
        border.size.height,
        content.size.height,
        content.virtual_size.height,
    )


@pytest.mark.asyncio
async def test_tall_result_container_matches_clamped_content() -> None:
    """Content taller than the cap: container and border stop at the cap too."""
    app = build_test_vibe_app()
    async with app.run_test(size=(80, 40)) as pilot:
        await _mount_result(app, pilot, rows=60)
        container_h, border_h, content_h, virtual_h = _heights(app)
        assert container_h == content_h
        assert border_h == content_h
        assert content_h <= app.size.height // 2
        # The overflow is still reachable by scrolling inside the region.
        assert virtual_h == 60


@pytest.mark.asyncio
async def test_short_result_is_not_clamped() -> None:
    """Content below the cap renders at its natural height, no scrolling."""
    app = build_test_vibe_app()
    async with app.run_test(size=(80, 40)) as pilot:
        await _mount_result(app, pilot, rows=5)
        container_h, border_h, content_h, virtual_h = _heights(app)
        assert container_h == content_h == border_h == 5
        assert virtual_h == 5


@pytest.mark.asyncio
async def test_cap_tracks_terminal_resize() -> None:
    """Shrinking the terminal re-caps already-mounted results."""
    app = build_test_vibe_app()
    async with app.run_test(size=(80, 40)) as pilot:
        await _mount_result(app, pilot, rows=60)
        await pilot.resize_terminal(80, 20)
        await pilot.pause()
        container_h, border_h, content_h, _ = _heights(app)
        assert content_h <= 10
        assert container_h == content_h
        assert border_h == content_h
