from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets._markdown import MarkdownTableCellContents, MarkdownTableContent

from privibe.cli.textual_ui.ansi_markdown import AnsiMarkdown

TABLE = """| # | Link |
|---|---|
| 1 | [PR one](https://example.com/1) |
| 2 | [PR two](https://example.com/2) |
| 3 | [PR three](https://example.com/3) |
"""


class TableApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.clicked: list[str] = []

    def compose(self) -> ComposeResult:
        yield AnsiMarkdown("", open_links=False)

    def on_markdown_link_clicked(self, event: AnsiMarkdown.LinkClicked) -> None:
        self.clicked.append(event.href)


async def _stream_table(app: TableApp, pilot) -> MarkdownTableContent:
    stream = AnsiMarkdown.get_stream(app.query_one(AnsiMarkdown))
    for line in TABLE.splitlines(keepends=True):
        await stream.write(line)
        await pilot.pause(0.05)
    await stream.stop()
    await pilot.pause()
    return app.query_one(MarkdownTableContent)


@pytest.mark.asyncio
async def test_streamed_table_cells_are_link_capable() -> None:
    app = TableApp()
    async with app.run_test() as pilot:
        content = await _stream_table(app, pilot)
        kinds = {type(cell) for cell in content.children}
        assert kinds == {MarkdownTableCellContents}, kinds
        assert len(content.children) == 2 + 3 * 2


@pytest.mark.asyncio
async def test_streamed_table_link_click_posts_link_clicked() -> None:
    app = TableApp()
    async with app.run_test(size=(80, 24)) as pilot:
        content = await _stream_table(app, pilot)
        body_cells = [c for c in content.children if "cell" in c.classes]
        for cell in body_cells[1::2]:
            await pilot.click(cell, offset=(2, 0))
            await pilot.pause()
        assert app.clicked == [
            "https://example.com/1",
            "https://example.com/2",
            "https://example.com/3",
        ]
