"""Shift map: stale (line, hash) addresses survive earlier same-session edits.

An edit above a line changes nothing about its hashed address except the
line number, by a knowable delta. The shift map records those deltas per
file so a later call holding pre-edit addresses is deterministically
re-pointed instead of failing into a re-read/re-write round trip.
Translation is only attempted when the given address fails, and only
applied when the hashes verify at the translated position.
"""

from __future__ import annotations

import pytest

from privibe.core.tools.base import BaseToolConfig, BaseToolState, ToolError
from privibe.core.tools.builtins._hashed_core import (
    _ShiftMap,
    _translate_interval,
    clear_all_shift_maps,
)
from privibe.core.tools.builtins.hashed_delete_block import (
    DeleteBlockItem,
    HashedDeleteBlock,
    HashedDeleteBlockArgs,
)
from privibe.core.tools.builtins.hashed_read import (
    HashedRead,
    HashedReadArgs,
    HashedReadConfig,
    _line_hash,
)
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


@pytest.fixture(autouse=True)
def _isolated_shift_maps():
    clear_all_shift_maps()
    yield
    clear_all_shift_maps()


@pytest.fixture
def replace_line_tool():
    return HashedReplaceLine(config=BaseToolConfig(), state=BaseToolState())


@pytest.fixture
def replace_block_tool():
    return HashedReplaceBlock(config=BaseToolConfig(), state=BaseToolState())


@pytest.fixture
def delete_block_tool():
    return HashedDeleteBlock(config=BaseToolConfig(), state=BaseToolState())


@pytest.fixture
def read_tool():
    return HashedRead(config=HashedReadConfig(), state=BaseToolState())


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


async def _replace_line(tool, path: str, *items: ReplaceLineItem):
    return await collect_result(
        tool.run(HashedReplaceLineArgs(path=path, replacements=list(items)))
    )


async def _replace_block(tool, path: str, *items: ReplaceBlockItem):
    return await collect_result(
        tool.run(HashedReplaceBlockArgs(path=path, replacements=list(items)))
    )


# ---------------------------------------------------------------------------
# _translate_interval (pure)
# ---------------------------------------------------------------------------


def _map_of(*generations: list[tuple[int, int, int]]) -> _ShiftMap:
    return _ShiftMap(fingerprint=0, generations=list(generations))


def test_interval_shifts_past_edit_above():
    # Edit replaced lines 1-2 with 4 lines (+2); old line 5 (idx 4) is now idx 6.
    smap = _map_of([(1, 2, 2)])
    assert _translate_interval(smap, 4, 6) == (6, 8)


def test_interval_ignores_edit_below():
    smap = _map_of([(10, 12, 5)])
    assert _translate_interval(smap, 2, 4) is None  # nothing moved: not stale


def test_interval_refused_when_edit_intersects():
    smap = _map_of([(3, 4, 2)])
    assert _translate_interval(smap, 2, 5) is None
    assert _translate_interval(smap, 4, 8) is None


def test_interval_composes_generations():
    # +2 above, then +3 above (in post-first-edit coordinates): net +5.
    smap = _map_of([(0, 0, 2)], [(4, 4, 3)])
    assert _translate_interval(smap, 8, 9) == (13, 14)


def test_interval_refused_by_zero_delta_overlap():
    # A same-size edit shifts nothing but rewrote lines 3-4: a stale interval
    # covering them must not translate.
    smap = _map_of([(0, 0, 2), (5, 6, 0)])
    assert _translate_interval(smap, 5, 6) is None


# ---------------------------------------------------------------------------
# Through the tools: the sequential-edit workflow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stale_block_after_insert_above_translates(
    tmp_path, monkeypatch, replace_line_tool, replace_block_tool
):
    """The reported failure: edit near the top, then a block edit below
    addressed with pre-edit line numbers.
    """
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\nd\ne\nf\n", encoding="utf-8")

    # Edit 1: replace line 2 ("b") with three lines (+2).
    await _replace_line(replace_line_tool, str(f), _rli(2, "b", "b1\nb2\nb3"))
    assert f.read_text(encoding="utf-8") == "a\nb1\nb2\nb3\nc\nd\ne\nf\n"

    # Edit 2 still addresses d-e as lines 4-5 (their pre-edit numbers).
    result = await _replace_block(
        replace_block_tool, str(f), _rbl(4, "d", 5, "e", "D\nE")
    )
    assert f.read_text(encoding="utf-8") == "a\nb1\nb2\nb3\nc\nD\nE\nf\n"
    assert result.content_note is not None
    assert "shifted +2" in result.content_note
    assert "line 6" in result.content_note


@pytest.mark.asyncio
async def test_stale_single_line_translates(
    tmp_path, monkeypatch, replace_line_tool
):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\nd\n", encoding="utf-8")

    await _replace_line(replace_line_tool, str(f), _rli(1, "a", "a1\na2"))
    result = await _replace_line(replace_line_tool, str(f), _rli(3, "c", "C"))

    assert f.read_text(encoding="utf-8") == "a1\na2\nb\nC\nd\n"
    assert result.content_note is not None
    assert "shifted +1" in result.content_note


@pytest.mark.asyncio
async def test_stale_delete_translates_and_reports(
    tmp_path, monkeypatch, replace_line_tool, delete_block_tool
):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")

    await _replace_line(replace_line_tool, str(f), _rli(1, "a", "a1\na2\na3"))
    result = await collect_result(
        delete_block_tool.run(
            HashedDeleteBlockArgs(
                path=str(f),
                deletions=[
                    DeleteBlockItem(
                        line=3,
                        hash=_line_hash("c"),
                        end_line=4,
                        end_hash=_line_hash("d"),
                    )
                ],
            )
        )
    )

    assert f.read_text(encoding="utf-8") == "a1\na2\na3\nb\ne\n"
    assert result.content_note is not None
    assert "shifted +2" in result.content_note


@pytest.mark.asyncio
async def test_shifts_compose_across_sequential_edits(
    tmp_path, monkeypatch, replace_line_tool
):
    """Three edits marching down the file off one read, each below the last."""
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")

    await _replace_line(replace_line_tool, str(f), _rli(1, "a", "a1\na2"))  # +1
    await _replace_line(replace_line_tool, str(f), _rli(2, "b", "b1\nb2\nb3"))  # +2
    result = await _replace_line(replace_line_tool, str(f), _rli(4, "d", "D"))

    assert f.read_text(encoding="utf-8") == "a1\na2\nb1\nb2\nb3\nc\nD\ne\n"
    assert "shifted +3" in (result.content_note or "")


@pytest.mark.asyncio
async def test_fresh_addresses_pass_without_note(
    tmp_path, monkeypatch, replace_line_tool
):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\n", encoding="utf-8")

    await _replace_line(replace_line_tool, str(f), _rli(1, "a", "a1\na2"))
    # Current coordinates: c is line 4 now.
    result = await _replace_line(replace_line_tool, str(f), _rli(4, "c", "C"))

    assert f.read_text(encoding="utf-8") == "a1\na2\nb\nC\n"
    assert result.content_note is None


@pytest.mark.asyncio
async def test_translation_refused_when_region_overlaps_prior_edit(
    tmp_path, monkeypatch, replace_line_tool, replace_block_tool
):
    """A stale block covering an earlier edit's region must error, never
    silently overwrite that edit with stale content.
    """
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\nd\n", encoding="utf-8")

    await _replace_line(replace_line_tool, str(f), _rli(2, "b", "B1\nB2"))
    with pytest.raises(ToolError, match="Hash mismatch"):
        await _replace_block(
            replace_block_tool, str(f), _rbl(2, "b", 3, "c", "X\nY")
        )
    # The overwrite never happened.
    assert f.read_text(encoding="utf-8") == "a\nB1\nB2\nc\nd\n"


@pytest.mark.asyncio
async def test_hashed_read_resets_map(
    tmp_path, monkeypatch, replace_line_tool, read_tool
):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\nd\n", encoding="utf-8")

    await _replace_line(replace_line_tool, str(f), _rli(1, "a", "a1\na2"))
    await collect_result(read_tool.run(HashedReadArgs(path=str(f))))

    # Post-read, pre-edit addresses are no longer translatable.
    with pytest.raises(ToolError, match="Hash mismatch"):
        await _replace_line(replace_line_tool, str(f), _rli(3, "c", "C"))


@pytest.mark.asyncio
async def test_foreign_write_discards_map(
    tmp_path, monkeypatch, replace_line_tool
):
    """An edit outside the hashed tools invalidates the map even when the
    translation would still have landed on a hash-matching line.
    """
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\nd\n", encoding="utf-8")

    await _replace_line(replace_line_tool, str(f), _rli(1, "a", "a1\na2"))
    # Foreign edit: line "b" changes, shifted positions of c/d stay intact.
    f.write_text("a1\na2\nB\nc\nd\n", encoding="utf-8")

    with pytest.raises(ToolError, match="Hash mismatch"):
        await _replace_line(replace_line_tool, str(f), _rli(3, "c", "C"))
    assert f.read_text(encoding="utf-8") == "a1\na2\nB\nc\nd\n"


@pytest.mark.asyncio
async def test_failed_translation_discards_map(
    tmp_path, monkeypatch, replace_line_tool
):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\nd\n", encoding="utf-8")

    await _replace_line(replace_line_tool, str(f), _rli(1, "a", "a1\na2"))

    # A hash that matches nowhere: translation fails and the map is dropped.
    with pytest.raises(ToolError, match="Hash mismatch"):
        await _replace_line(replace_line_tool, str(f), _rli(3, "not-there", "X"))

    # An address that WOULD have translated before now fails too.
    with pytest.raises(ToolError, match="Hash mismatch"):
        await _replace_line(replace_line_tool, str(f), _rli(3, "c", "C"))


@pytest.mark.asyncio
async def test_batch_call_records_one_generation(
    tmp_path, monkeypatch, replace_line_tool
):
    """Two replacements in one call share pre-call coordinates; a later stale
    address translates through their combined deltas.
    """
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")

    await _replace_line(
        replace_line_tool,
        str(f),
        _rli(1, "a", "a1\na2"),  # +1
        _rli(3, "c", "c1\nc2\nc3"),  # +2
    )
    result = await _replace_line(replace_line_tool, str(f), _rli(5, "e", "E"))

    assert f.read_text(encoding="utf-8") == "a1\na2\nb\nc1\nc2\nc3\nd\nE\n"
    assert "shifted +3" in (result.content_note or "")
