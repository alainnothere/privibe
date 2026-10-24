"""SEARCH text with non-ASCII falls back to its ASCII form, but only after an exact miss."""

from __future__ import annotations

from pathlib import Path

import pytest

from privibe.core.tools.base import BaseToolState, ToolError
from privibe.core.tools.builtins.search_replace import (
    SearchReplace,
    SearchReplaceArgs,
    SearchReplaceConfig,
)
from tests.mock.utils import collect_result

pytestmark = pytest.mark.asyncio


@pytest.fixture
def sr_tool() -> SearchReplace:
    return SearchReplace(config=SearchReplaceConfig(), state=BaseToolState())


def _block(search: str, replace: str) -> str:
    return f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE"


async def test_ascii_stored_file_matches_unicode_search(
    sr_tool: SearchReplace, tmp_path: Path
) -> None:
    f = tmp_path / "a.txt"
    f.write_text("intro\nwait - what\noutro\n", encoding="utf-8")
    args = SearchReplaceArgs(
        file_path=str(f), content=_block("wait — what", "wait, what")
    )
    result = await collect_result(sr_tool.run(args))
    assert f.read_text(encoding="utf-8") == "intro\nwait, what\noutro\n"
    assert result.blocks_applied == 1
    assert any(
        "block 1" in w and "line 2" in w and "allow_unicode" in w
        for w in result.warnings
    )


async def test_real_unicode_file_matches_exactly_without_note(
    sr_tool: SearchReplace, tmp_path: Path
) -> None:
    f = tmp_path / "b.txt"
    f.write_text("intro\nwait — what\noutro\n", encoding="utf-8")
    args = SearchReplaceArgs(
        file_path=str(f),
        content=_block("wait — what", "wait, what"),
        allow_unicode=True,
    )
    result = await collect_result(sr_tool.run(args))
    assert f.read_text(encoding="utf-8") == "intro\nwait, what\noutro\n"
    assert result.warnings == []
    assert not result.content_note


async def test_missing_search_still_fails(
    sr_tool: SearchReplace, tmp_path: Path
) -> None:
    f = tmp_path / "c.txt"
    f.write_text("intro\nwait - what\noutro\n", encoding="utf-8")
    args = SearchReplaceArgs(
        file_path=str(f), content=_block("nope — nothing", "x")
    )
    with pytest.raises(ToolError, match="Search text not found"):
        await collect_result(sr_tool.run(args))
    assert f.read_text(encoding="utf-8") == "intro\nwait - what\noutro\n"
