from __future__ import annotations

from pygments.token import Token
from textual.app import ComposeResult
from textual.content import Content
from textual.highlight import HighlightTheme, highlight
from textual.widgets import Markdown
from textual.widgets._markdown import (
    MarkdownFence,
    MarkdownTable,
    MarkdownTableCellContents,
    MarkdownTableContent,
)


class AnsiHighlightTheme(HighlightTheme):
    STYLES = {
        Token.Comment: "ansi_bright_black italic",
        Token.Error: "ansi_red",
        Token.Generic.Strong: "bold",
        Token.Generic.Emph: "italic",
        Token.Generic.Error: "ansi_red",
        Token.Generic.Heading: "ansi_blue underline",
        Token.Generic.Subheading: "ansi_blue",
        Token.Keyword: "ansi_magenta",
        Token.Keyword.Constant: "ansi_cyan",
        Token.Keyword.Namespace: "ansi_magenta",
        Token.Keyword.Type: "ansi_cyan",
        Token.Literal.Number: "ansi_yellow",
        Token.Literal.String.Backtick: "ansi_bright_black",
        Token.Literal.String: "ansi_green",
        Token.Literal.String.Doc: "ansi_green italic",
        Token.Literal.String.Double: "ansi_green",
        Token.Name: "ansi_default",
        Token.Name.Attribute: "ansi_yellow",
        Token.Name.Builtin: "ansi_cyan",
        Token.Name.Builtin.Pseudo: "italic",
        Token.Name.Class: "ansi_yellow",
        Token.Name.Constant: "ansi_red",
        Token.Name.Decorator: "ansi_blue",
        Token.Name.Function: "ansi_blue",
        Token.Name.Function.Magic: "ansi_blue",
        Token.Name.Tag: "ansi_blue",
        Token.Name.Variable: "ansi_default",
        Token.Number: "ansi_yellow",
        Token.Operator: "ansi_default",
        Token.Operator.Word: "ansi_magenta",
        Token.String: "ansi_green",
        Token.Whitespace: "",
    }


class AnsiMarkdownFence(MarkdownFence):
    @classmethod
    def highlight(cls, code: str, language: str) -> Content:
        return highlight(code, language=language or None, theme=AnsiHighlightTheme)


class LinkableTableContent(MarkdownTableContent):
    """Streamed table rows that keep their links clickable.

    Textual 8.2.3's ``_update_rows`` mounts streamed rows as bare ``Static``
    cells, which have no ``action_link``; Textual does not bubble actions, so
    every link in a streamed table body is dead. Only the one-shot
    ``compose`` path uses ``MarkdownTableCellContents``. This mirrors
    upstream ``_update_rows`` with the right cell class. Delete once fixed
    upstream; tests/cli/textual_ui/test_markdown_table_links.py guards it.
    """

    async def _update_rows(self, updated_rows: list[list[Content]]) -> None:
        self.styles.grid_size_columns = len(self.headers)
        await self.query_children(f".cell.row{self.last_row}").remove()
        new_cells: list[MarkdownTableCellContents] = []
        row_index = self.last_row
        for row_index, row in enumerate(updated_rows, self.last_row):
            for cell in row:
                new_cells.append(
                    MarkdownTableCellContents(
                        cell, classes=f"row{row_index} cell"
                    ).with_tooltip(cell.plain)
                )
        self.last_row = row_index
        await self.mount_all(new_cells)


class LinkableMarkdownTable(MarkdownTable):
    def compose(self) -> ComposeResult:
        headers, rows = self._get_headers_and_rows()
        self._headers = headers
        self._rows = rows
        yield LinkableTableContent(headers, rows)


class AnsiMarkdown(Markdown):
    BLOCKS = {
        **Markdown.BLOCKS,
        "fence": AnsiMarkdownFence,
        "code_block": AnsiMarkdownFence,
        "table_open": LinkableMarkdownTable,
    }
