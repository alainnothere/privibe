from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from privibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from privibe.cli.textual_ui.widgets.tools import ToolResultMessage
from privibe.core.rewind.diffing import build_file_diff
from privibe.core.types import FileDiff, ToolResultEvent


class _Host(App):
    def __init__(self, event: ToolResultEvent) -> None:
        super().__init__()
        self._event = event

    def compose(self) -> ComposeResult:
        yield ToolResultMessage(self._event)


def _event(
    file_diff: FileDiff | None, tool_name: str = "hashed_replace_block"
) -> ToolResultEvent:
    return ToolResultEvent(
        tool_name=tool_name, tool_class=None, file_diff=file_diff, tool_call_id="t"
    )


async def _render(event: ToolResultEvent):
    app = _Host(event)
    async with app.run_test():
        msg = app.query_one(ToolResultMessage)
        assert msg._content_container is not None
        statics = msg._content_container.query(NoMarkupStatic)
        texts = [str(s.render()) for s in statics]
        removed = len(msg.query(".diff-removed"))
        added = len(msg.query(".diff-added"))
        return texts, removed, added


def _b(text: str) -> bytes:
    return text.encode("utf-8")


@pytest.mark.asyncio
async def test_edit_renders_red_and_green():
    diff = build_file_diff("f.py", _b("a\nb\nc\n"), _b("a\nB\nc\n"))
    _texts, removed, added = await _render(_event(diff))
    assert removed >= 1
    assert added >= 1


@pytest.mark.asyncio
async def test_hunk_budget_caps_and_shows_tail():
    # 4 well-separated edits; default preview (no config) = 3 hunks shown.
    lines = [str(i) for i in range(60)]
    old = "\n".join(lines) + "\n"
    for idx in (5, 20, 35, 50):
        lines[idx] = f"CHANGED{idx}"
    new = "\n".join(lines) + "\n"
    diff = build_file_diff("f.py", _b(old), _b(new))
    assert diff is not None and diff.kind == "diff" and len(diff.hunks) == 4
    texts, _removed, _added = await _render(_event(diff))
    assert any("1 more change below" in t for t in texts)


@pytest.mark.asyncio
async def test_binary_shows_note_no_colors():
    diff = build_file_diff("f.bin", b"a\x00b", _b("text\n"))
    texts, removed, added = await _render(_event(diff))
    assert any("Binary file - no diff shown" in t for t in texts)
    assert removed == 0 and added == 0


@pytest.mark.asyncio
async def test_search_replace_does_not_use_file_diff_widget():
    # search_replace is excluded (keeps its own diff widget); the FileDiffWidget
    # must not appear even when a file_diff is present on the event.
    from privibe.cli.textual_ui.widgets.tool_widgets import FileDiffWidget

    diff = build_file_diff("f.py", _b("a\nb\n"), _b("a\nB\n"))
    app = _Host(_event(diff, tool_name="search_replace"))
    async with app.run_test():
        msg = app.query_one(ToolResultMessage)
        assert len(msg.query(FileDiffWidget)) == 0
