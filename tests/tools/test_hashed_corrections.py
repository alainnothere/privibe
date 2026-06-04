from __future__ import annotations

import pytest

from privibe.core.tools.base import BaseToolConfig, BaseToolState
from privibe.core.tools.builtins._hashed_core import strip_leaked_prefix
from privibe.core.tools.builtins.hashed_read import _line_hash, format_hashed_lines
from privibe.core.tools.builtins.hashed_replace_block import (
    HashedReplaceBlock,
    HashedReplaceBlockArgs,
    ReplaceBlockItem,
)
from privibe.core.tools.builtins.hashed_replace_line import (
    HashedReplaceLine,
    HashedReplaceLineArgs,
    ReplaceLineItem,
)
from tests.mock.utils import collect_result


@pytest.fixture
def replace_line_tool():
    return HashedReplaceLine(config=BaseToolConfig(), state=BaseToolState())


@pytest.fixture
def replace_block_tool():
    return HashedReplaceBlock(config=BaseToolConfig(), state=BaseToolState())


def _rli(line: int, current: str, new: str) -> ReplaceLineItem:
    return ReplaceLineItem(line=line, hash=_line_hash(current), new_content=new)


def _rbl(
    line: int, current: str, end_line: int, end_current: str, new: str
) -> ReplaceBlockItem:
    return ReplaceBlockItem(
        line=line,
        hash=_line_hash(current),
        end_line=end_line,
        end_hash=_line_hash(end_current),
        new_content=new,
    )


# ---------------------------------------------------------------------------
# strip_leaked_prefix (pure)
# ---------------------------------------------------------------------------


def test_strip_removes_authentic_read_prefix():
    # Build the exact text hashed_read would emit for a line, then confirm the
    # prefix round-trips off.
    leaked = format_hashed_lines(["    return x\n"], 12)
    cleaned, n = strip_leaked_prefix(leaked)
    assert cleaned == "    return x"
    assert n == 1


def test_strip_leaves_normal_content_untouched():
    cleaned, n = strip_leaked_prefix("    return x")
    assert cleaned == "    return x"
    assert n == 0


def test_strip_only_affects_prefixed_lines():
    leaked = format_hashed_lines(["alpha\n"], 3)
    content = f"{leaked}\nbeta\ngamma"
    cleaned, n = strip_leaked_prefix(content)
    assert cleaned == "alpha\nbeta\ngamma"
    assert n == 1


def test_strip_requires_four_hex_and_two_spaces():
    # 3-char hash, single trailing space, and no-hash variants must NOT match.
    assert strip_leaked_prefix("   12 abc  x") == ("   12 abc  x", 0)
    assert strip_leaked_prefix("   12 ab12 x") == ("   12 ab12 x", 0)
    assert strip_leaked_prefix("   1234  x") == ("   1234  x", 0)


def test_strip_does_not_match_uppercase_hash():
    # _line_hash emits lowercase hex; uppercase is not our prefix.
    assert strip_leaked_prefix("   12 AB12  x") == ("   12 AB12  x", 0)


# ---------------------------------------------------------------------------
# leak stripping through the tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replace_line_strips_leaked_prefix(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\n", encoding="utf-8")
    leaked_new = format_hashed_lines(["X\n"], 2)  # model pasted the read line back

    result = await collect_result(
        replace_line_tool.run(
            HashedReplaceLineArgs(path=str(f), replacements=[_rli(2, "b", leaked_new)])
        )
    )

    assert f.read_text(encoding="utf-8") == "a\nX\nc\n"
    assert result.content_note is not None
    assert "prefix" in result.content_note


@pytest.mark.asyncio
async def test_allow_literal_keeps_prefix_verbatim(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\n", encoding="utf-8")
    leaked_new = format_hashed_lines(["X\n"], 2)

    result = await collect_result(
        replace_line_tool.run(
            HashedReplaceLineArgs(
                path=str(f),
                replacements=[_rli(2, "b", leaked_new)],
                allow_literal=True,
            )
        )
    )

    assert f.read_text(encoding="utf-8") == f"a\n{leaked_new}\nc\n"
    assert result.content_note is None


# ---------------------------------------------------------------------------
# boundary-duplicate removal through the tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trailing_boundary_duplicate_removed(tmp_path, monkeypatch, replace_block_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\nd\n", encoding="utf-8")

    # Replace line 2 ("b") with "NEW\nc"; the trailing "c" duplicates line 3.
    result = await collect_result(
        replace_block_tool.run(
            HashedReplaceBlockArgs(
                path=str(f), replacements=[_rbl(2, "b", 2, "b", "NEW\nc")]
            )
        )
    )

    assert f.read_text(encoding="utf-8") == "a\nNEW\nc\nd\n"
    assert result.content_note is not None
    assert "after" in result.content_note


@pytest.mark.asyncio
async def test_leading_boundary_duplicate_removed(tmp_path, monkeypatch, replace_block_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\nd\n", encoding="utf-8")

    # Replace line 3 ("c") with "b\nNEW"; the leading "b" duplicates line 2.
    result = await collect_result(
        replace_block_tool.run(
            HashedReplaceBlockArgs(
                path=str(f), replacements=[_rbl(3, "c", 3, "c", "b\nNEW")]
            )
        )
    )

    assert f.read_text(encoding="utf-8") == "a\nb\nNEW\nd\n"
    assert result.content_note is not None
    assert "before" in result.content_note


@pytest.mark.asyncio
async def test_keep_duplicate_keeps_boundary_line(tmp_path, monkeypatch, replace_block_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\nd\n", encoding="utf-8")

    result = await collect_result(
        replace_block_tool.run(
            HashedReplaceBlockArgs(
                path=str(f),
                replacements=[_rbl(2, "b", 2, "b", "NEW\nc")],
                keep_duplicate=True,
            )
        )
    )

    assert f.read_text(encoding="utf-8") == "a\nNEW\nc\nc\nd\n"
    assert result.content_note is None


@pytest.mark.asyncio
async def test_preexisting_duplicate_not_touched(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nb\nc\n", encoding="utf-8")  # b is already duplicated

    # Edit an unrelated line; the existing b/b must survive untouched.
    result = await collect_result(
        replace_line_tool.run(
            HashedReplaceLineArgs(path=str(f), replacements=[_rli(1, "a", "X")])
        )
    )

    assert f.read_text(encoding="utf-8") == "X\nb\nb\nc\n"
    assert result.content_note is None


@pytest.mark.asyncio
async def test_duplicate_within_new_content_not_touched(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\n", encoding="utf-8")

    # Two identical lines inside new_content, neither matching a neighbour.
    result = await collect_result(
        replace_line_tool.run(
            HashedReplaceLineArgs(path=str(f), replacements=[_rli(2, "b", "P\nP")])
        )
    )

    assert f.read_text(encoding="utf-8") == "a\nP\nP\nc\n"
    assert result.content_note is None


@pytest.mark.asyncio
async def test_boundary_skipped_when_neighbor_is_also_edited(tmp_path, monkeypatch, replace_block_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\nd\n", encoding="utf-8")

    # Replace line 1 -> "X\nb" and line 2 -> "Y" in one batch. The first edit's
    # trailing "b" matches original line 2, but line 2 is itself being replaced,
    # so the duplicate must NOT be removed (it isn't actually duplicated in the
    # result).
    result = await collect_result(
        replace_block_tool.run(
            HashedReplaceBlockArgs(
                path=str(f),
                replacements=[
                    _rbl(1, "a", 1, "a", "X\nb"),
                    _rbl(2, "b", 2, "b", "Y"),
                ],
            )
        )
    )

    assert f.read_text(encoding="utf-8") == "X\nb\nY\nc\nd\n"
    assert result.content_note is None
