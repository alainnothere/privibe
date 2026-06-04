from __future__ import annotations

import pytest

from privibe.core.tools.base import BaseToolConfig, BaseToolState, ToolError
from privibe.core.tools.builtins.hashed_delete_block import (
    DeleteBlockItem,
    HashedDeleteBlock,
    HashedDeleteBlockArgs,
)
from privibe.core.tools.builtins.hashed_delete_line import (
    DeleteLineItem,
    HashedDeleteLine,
    HashedDeleteLineArgs,
)
from privibe.core.tools.builtins.hashed_read import (
    HashedRead,
    HashedReadArgs,
    HashedReadConfig,
    _line_hash,
    format_hashed_lines,
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

# ---------------------------------------------------------------------------
# _line_hash
# ---------------------------------------------------------------------------


def test_line_hash_length():
    assert len(_line_hash("hello")) == 4


def test_line_hash_hex():
    h = _line_hash("hello")
    assert all(c in "0123456789abcdef" for c in h)


def test_line_hash_stable():
    assert _line_hash("hello") == _line_hash("hello")


def test_line_hash_differs_on_trailing_space():
    assert _line_hash("hello") != _line_hash("hello ")


def test_line_hash_empty_line():
    h = _line_hash("")
    assert len(h) == 4


# ---------------------------------------------------------------------------
# format_hashed_lines
# ---------------------------------------------------------------------------


def test_format_hashed_lines_structure():
    lines = ["hello\n", "world\n"]
    out = format_hashed_lines(lines, 1)
    parts = out.splitlines()
    assert len(parts) == 2
    assert parts[0].startswith("    1 ")
    assert parts[0].endswith("  hello")
    assert parts[1].startswith("    2 ")


def test_format_hashed_lines_start_num():
    lines = ["x\n"]
    out = format_hashed_lines(lines, 10)
    assert out.startswith("   10 ")


def test_format_hashed_lines_strips_newline():
    lines = ["hello   \n"]
    out = format_hashed_lines(lines, 1)
    assert out.endswith("  hello   ")


# ---------------------------------------------------------------------------
# HashedRead.run
# ---------------------------------------------------------------------------


@pytest.fixture
def hashed_read_tool():
    return HashedRead(config=HashedReadConfig(), state=BaseToolState())


@pytest.mark.asyncio
async def test_hashed_read_basic(tmp_path, monkeypatch, hashed_read_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "sample.py"
    f.write_text("line one\nline two\nline three\n", encoding="utf-8")

    result = await collect_result(hashed_read_tool.run(HashedReadArgs(path=str(f))))

    lines = result.content.splitlines()
    assert len(lines) == 3
    assert "    1 " in lines[0]
    assert "line one" in lines[0]
    assert result.lines_read == 3
    assert result.start_line == 1
    assert not result.was_truncated


@pytest.mark.asyncio
async def test_hashed_read_offset_and_limit(tmp_path, monkeypatch, hashed_read_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "file.txt"
    f.write_text("a\nb\nc\nd\n", encoding="utf-8")

    result = await collect_result(
        hashed_read_tool.run(HashedReadArgs(path=str(f), offset=1, limit=2))
    )

    lines = result.content.splitlines()
    assert len(lines) == 2
    assert "    2 " in lines[0]
    assert "b" in lines[0]
    assert "c" in lines[1]


@pytest.mark.asyncio
async def test_hashed_read_missing_file_raises(tmp_path, monkeypatch, hashed_read_tool):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ToolError, match="File not found"):
        await collect_result(hashed_read_tool.run(HashedReadArgs(path="missing.py")))


@pytest.mark.asyncio
async def test_hashed_read_directory_raises(tmp_path, monkeypatch, hashed_read_tool):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ToolError, match="directory"):
        await collect_result(hashed_read_tool.run(HashedReadArgs(path=str(tmp_path))))


@pytest.mark.asyncio
async def test_hashed_read_truncates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tool = HashedRead(config=HashedReadConfig(max_read_bytes=10), state=BaseToolState())
    f = tmp_path / "big.txt"
    f.write_text("a" * 100 + "\n" + "b" * 100 + "\n", encoding="utf-8")

    result = await collect_result(tool.run(HashedReadArgs(path=str(f))))
    assert result.was_truncated


# ---------------------------------------------------------------------------
# Helpers for new tools
# ---------------------------------------------------------------------------


@pytest.fixture
def replace_line_tool():
    return HashedReplaceLine(config=BaseToolConfig(), state=BaseToolState())


@pytest.fixture
def replace_block_tool():
    return HashedReplaceBlock(config=BaseToolConfig(), state=BaseToolState())


@pytest.fixture
def delete_line_tool():
    return HashedDeleteLine(config=BaseToolConfig(), state=BaseToolState())


@pytest.fixture
def delete_block_tool():
    return HashedDeleteBlock(config=BaseToolConfig(), state=BaseToolState())


def _rli(line: int, current: str, new: str) -> ReplaceLineItem:
    return ReplaceLineItem(line=line, hash=_line_hash(current), new_content=new)


def _rbl(line: int, current: str, end_line: int, end_current: str, new: str) -> ReplaceBlockItem:
    return ReplaceBlockItem(
        line=line,
        hash=_line_hash(current),
        end_line=end_line,
        end_hash=_line_hash(end_current),
        new_content=new,
    )


def _dli(line: int, current: str) -> DeleteLineItem:
    return DeleteLineItem(line=line, hash=_line_hash(current))


def _dbl(line: int, current: str, end_line: int, end_current: str) -> DeleteBlockItem:
    return DeleteBlockItem(
        line=line,
        hash=_line_hash(current),
        end_line=end_line,
        end_hash=_line_hash(end_current),
    )


# ---------------------------------------------------------------------------
# HashedReplaceLine
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replace_line_single_to_single(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\n", encoding="utf-8")

    await collect_result(
        replace_line_tool.run(HashedReplaceLineArgs(path=str(f), replacements=[_rli(2, "b", "X")]))
    )

    assert f.read_text(encoding="utf-8") == "a\nX\nc\n"


@pytest.mark.asyncio
async def test_replace_line_expands_to_block(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\n", encoding="utf-8")

    await collect_result(
        replace_line_tool.run(HashedReplaceLineArgs(path=str(f), replacements=[_rli(2, "b", "X\nY\nZ")]))
    )

    assert f.read_text(encoding="utf-8") == "a\nX\nY\nZ\nc\n"


@pytest.mark.asyncio
async def test_replace_line_first_line(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\n", encoding="utf-8")

    await collect_result(
        replace_line_tool.run(HashedReplaceLineArgs(path=str(f), replacements=[_rli(1, "a", "Z")]))
    )

    assert f.read_text(encoding="utf-8") == "Z\nb\nc\n"


@pytest.mark.asyncio
async def test_replace_line_last_line(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\n", encoding="utf-8")

    await collect_result(
        replace_line_tool.run(HashedReplaceLineArgs(path=str(f), replacements=[_rli(3, "c", "Z")]))
    )

    assert f.read_text(encoding="utf-8") == "a\nb\nZ\n"


@pytest.mark.asyncio
async def test_replace_line_batch_no_count_change(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")

    await collect_result(
        replace_line_tool.run(
            HashedReplaceLineArgs(
                path=str(f),
                replacements=[_rli(2, "b", "BB"), _rli(4, "d", "DD")],
            )
        )
    )

    assert f.read_text(encoding="utf-8") == "a\nBB\nc\nDD\ne\n"


@pytest.mark.asyncio
async def test_replace_line_batch_with_count_change(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\nd\n", encoding="utf-8")

    await collect_result(
        replace_line_tool.run(
            HashedReplaceLineArgs(
                path=str(f),
                replacements=[_rli(1, "a", "a1\na2\na3"), _rli(3, "c", "Z")],
            )
        )
    )

    assert f.read_text(encoding="utf-8") == "a1\na2\na3\nb\nZ\nd\n"


@pytest.mark.asyncio
async def test_replace_line_hash_mismatch_raises(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\n", encoding="utf-8")

    with pytest.raises(ToolError, match="Hash mismatch"):
        await collect_result(
            replace_line_tool.run(
                HashedReplaceLineArgs(
                    path=str(f),
                    replacements=[ReplaceLineItem(line=1, hash="dead", new_content="X")],
                )
            )
        )


@pytest.mark.asyncio
async def test_replace_line_atomicity(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    original = "a\nb\nc\n"
    f.write_text(original, encoding="utf-8")

    with pytest.raises(ToolError, match="Hash mismatch"):
        await collect_result(
            replace_line_tool.run(
                HashedReplaceLineArgs(
                    path=str(f),
                    replacements=[
                        _rli(1, "a", "GOOD"),
                        ReplaceLineItem(line=3, hash="dead", new_content="BAD"),
                    ],
                )
            )
        )

    assert f.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# HashedReplaceBlock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replace_block_with_larger_block(tmp_path, monkeypatch, replace_block_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\nd\n", encoding="utf-8")

    await collect_result(
        replace_block_tool.run(
            HashedReplaceBlockArgs(
                path=str(f),
                replacements=[_rbl(2, "b", 3, "c", "X\nY\nZ")],
            )
        )
    )

    assert f.read_text(encoding="utf-8") == "a\nX\nY\nZ\nd\n"


@pytest.mark.asyncio
async def test_replace_block_with_smaller_block(tmp_path, monkeypatch, replace_block_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")

    await collect_result(
        replace_block_tool.run(
            HashedReplaceBlockArgs(
                path=str(f),
                replacements=[_rbl(2, "b", 4, "d", "X")],
            )
        )
    )

    assert f.read_text(encoding="utf-8") == "a\nX\ne\n"


@pytest.mark.asyncio
async def test_replace_block_same_size(tmp_path, monkeypatch, replace_block_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\nd\n", encoding="utf-8")

    await collect_result(
        replace_block_tool.run(
            HashedReplaceBlockArgs(
                path=str(f),
                replacements=[_rbl(2, "b", 3, "c", "X\nY")],
            )
        )
    )

    assert f.read_text(encoding="utf-8") == "a\nX\nY\nd\n"


@pytest.mark.asyncio
async def test_replace_block_at_file_start(tmp_path, monkeypatch, replace_block_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\n", encoding="utf-8")

    await collect_result(
        replace_block_tool.run(
            HashedReplaceBlockArgs(
                path=str(f),
                replacements=[_rbl(1, "a", 2, "b", "X")],
            )
        )
    )

    assert f.read_text(encoding="utf-8") == "X\nc\n"


@pytest.mark.asyncio
async def test_replace_block_at_file_end(tmp_path, monkeypatch, replace_block_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\n", encoding="utf-8")

    await collect_result(
        replace_block_tool.run(
            HashedReplaceBlockArgs(
                path=str(f),
                replacements=[_rbl(2, "b", 3, "c", "X")],
            )
        )
    )

    assert f.read_text(encoding="utf-8") == "a\nX\n"


@pytest.mark.asyncio
async def test_replace_block_hash_mismatch_start(tmp_path, monkeypatch, replace_block_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    original = "a\nb\nc\n"
    f.write_text(original, encoding="utf-8")

    with pytest.raises(ToolError, match="Hash mismatch"):
        await collect_result(
            replace_block_tool.run(
                HashedReplaceBlockArgs(
                    path=str(f),
                    replacements=[
                        ReplaceBlockItem(line=1, hash="dead", end_line=2, end_hash=_line_hash("b"), new_content="X")
                    ],
                )
            )
        )

    assert f.read_text(encoding="utf-8") == original


@pytest.mark.asyncio
async def test_replace_block_hash_mismatch_end(tmp_path, monkeypatch, replace_block_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    original = "a\nb\nc\n"
    f.write_text(original, encoding="utf-8")

    with pytest.raises(ToolError, match="Hash mismatch"):
        await collect_result(
            replace_block_tool.run(
                HashedReplaceBlockArgs(
                    path=str(f),
                    replacements=[
                        ReplaceBlockItem(line=1, hash=_line_hash("a"), end_line=2, end_hash="dead", new_content="X")
                    ],
                )
            )
        )

    assert f.read_text(encoding="utf-8") == original


@pytest.mark.asyncio
async def test_replace_block_batch(tmp_path, monkeypatch, replace_block_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\nd\ne\nf\ng\n", encoding="utf-8")

    await collect_result(
        replace_block_tool.run(
            HashedReplaceBlockArgs(
                path=str(f),
                replacements=[
                    _rbl(2, "b", 3, "c", "BC"),
                    _rbl(5, "e", 6, "f", "EF"),
                ],
            )
        )
    )

    assert f.read_text(encoding="utf-8") == "a\nBC\nd\nEF\ng\n"


@pytest.mark.asyncio
async def test_replace_block_overlapping_raises(tmp_path, monkeypatch, replace_block_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\nd\n", encoding="utf-8")

    with pytest.raises(ToolError, match="overlap"):
        await collect_result(
            replace_block_tool.run(
                HashedReplaceBlockArgs(
                    path=str(f),
                    replacements=[
                        _rbl(1, "a", 3, "c", "X"),
                        _rbl(2, "b", 4, "d", "Y"),
                    ],
                )
            )
        )


@pytest.mark.asyncio
async def test_replace_block_atomicity(tmp_path, monkeypatch, replace_block_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    original = "a\nb\nc\nd\n"
    f.write_text(original, encoding="utf-8")

    with pytest.raises(ToolError, match="Hash mismatch"):
        await collect_result(
            replace_block_tool.run(
                HashedReplaceBlockArgs(
                    path=str(f),
                    replacements=[
                        _rbl(1, "a", 2, "b", "GOOD"),
                        ReplaceBlockItem(line=3, hash="dead", end_line=4, end_hash=_line_hash("d"), new_content="BAD"),
                    ],
                )
            )
        )

    assert f.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# HashedDeleteLine
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_line_middle(tmp_path, monkeypatch, delete_line_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\n", encoding="utf-8")

    await collect_result(
        delete_line_tool.run(HashedDeleteLineArgs(path=str(f), deletions=[_dli(2, "b")]))
    )

    assert f.read_text(encoding="utf-8") == "a\nc\n"


@pytest.mark.asyncio
async def test_delete_line_first(tmp_path, monkeypatch, delete_line_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\n", encoding="utf-8")

    await collect_result(
        delete_line_tool.run(HashedDeleteLineArgs(path=str(f), deletions=[_dli(1, "a")]))
    )

    assert f.read_text(encoding="utf-8") == "b\nc\n"


@pytest.mark.asyncio
async def test_delete_line_last(tmp_path, monkeypatch, delete_line_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\n", encoding="utf-8")

    await collect_result(
        delete_line_tool.run(HashedDeleteLineArgs(path=str(f), deletions=[_dli(3, "c")]))
    )

    assert f.read_text(encoding="utf-8") == "a\nb\n"


@pytest.mark.asyncio
async def test_delete_line_batch(tmp_path, monkeypatch, delete_line_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")

    await collect_result(
        delete_line_tool.run(
            HashedDeleteLineArgs(path=str(f), deletions=[_dli(2, "b"), _dli(4, "d")])
        )
    )

    assert f.read_text(encoding="utf-8") == "a\nc\ne\n"


@pytest.mark.asyncio
async def test_delete_line_hash_mismatch_raises(tmp_path, monkeypatch, delete_line_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    original = "a\nb\n"
    f.write_text(original, encoding="utf-8")

    with pytest.raises(ToolError, match="Hash mismatch"):
        await collect_result(
            delete_line_tool.run(
                HashedDeleteLineArgs(
                    path=str(f),
                    deletions=[DeleteLineItem(line=1, hash="dead")],
                )
            )
        )

    assert f.read_text(encoding="utf-8") == original


@pytest.mark.asyncio
async def test_delete_line_atomicity(tmp_path, monkeypatch, delete_line_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    original = "a\nb\nc\n"
    f.write_text(original, encoding="utf-8")

    with pytest.raises(ToolError, match="Hash mismatch"):
        await collect_result(
            delete_line_tool.run(
                HashedDeleteLineArgs(
                    path=str(f),
                    deletions=[_dli(1, "a"), DeleteLineItem(line=3, hash="dead")],
                )
            )
        )

    assert f.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# HashedDeleteBlock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_block_middle(tmp_path, monkeypatch, delete_block_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")

    await collect_result(
        delete_block_tool.run(
            HashedDeleteBlockArgs(path=str(f), deletions=[_dbl(2, "b", 4, "d")])
        )
    )

    assert f.read_text(encoding="utf-8") == "a\ne\n"


@pytest.mark.asyncio
async def test_delete_block_at_start(tmp_path, monkeypatch, delete_block_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\nd\n", encoding="utf-8")

    await collect_result(
        delete_block_tool.run(
            HashedDeleteBlockArgs(path=str(f), deletions=[_dbl(1, "a", 2, "b")])
        )
    )

    assert f.read_text(encoding="utf-8") == "c\nd\n"


@pytest.mark.asyncio
async def test_delete_block_at_end(tmp_path, monkeypatch, delete_block_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\nd\n", encoding="utf-8")

    await collect_result(
        delete_block_tool.run(
            HashedDeleteBlockArgs(path=str(f), deletions=[_dbl(3, "c", 4, "d")])
        )
    )

    assert f.read_text(encoding="utf-8") == "a\nb\n"


@pytest.mark.asyncio
async def test_delete_block_hash_mismatch_start(tmp_path, monkeypatch, delete_block_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    original = "a\nb\nc\n"
    f.write_text(original, encoding="utf-8")

    with pytest.raises(ToolError, match="Hash mismatch"):
        await collect_result(
            delete_block_tool.run(
                HashedDeleteBlockArgs(
                    path=str(f),
                    deletions=[DeleteBlockItem(line=1, hash="dead", end_line=2, end_hash=_line_hash("b"))],
                )
            )
        )

    assert f.read_text(encoding="utf-8") == original


@pytest.mark.asyncio
async def test_delete_block_hash_mismatch_end(tmp_path, monkeypatch, delete_block_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    original = "a\nb\nc\n"
    f.write_text(original, encoding="utf-8")

    with pytest.raises(ToolError, match="Hash mismatch"):
        await collect_result(
            delete_block_tool.run(
                HashedDeleteBlockArgs(
                    path=str(f),
                    deletions=[DeleteBlockItem(line=1, hash=_line_hash("a"), end_line=2, end_hash="dead")],
                )
            )
        )

    assert f.read_text(encoding="utf-8") == original


@pytest.mark.asyncio
async def test_delete_block_batch(tmp_path, monkeypatch, delete_block_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\nd\ne\nf\ng\n", encoding="utf-8")

    await collect_result(
        delete_block_tool.run(
            HashedDeleteBlockArgs(
                path=str(f),
                deletions=[_dbl(2, "b", 3, "c"), _dbl(5, "e", 6, "f")],
            )
        )
    )

    assert f.read_text(encoding="utf-8") == "a\nd\ng\n"


@pytest.mark.asyncio
async def test_delete_block_overlapping_raises(tmp_path, monkeypatch, delete_block_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    f.write_text("a\nb\nc\nd\n", encoding="utf-8")

    with pytest.raises(ToolError, match="overlap"):
        await collect_result(
            delete_block_tool.run(
                HashedDeleteBlockArgs(
                    path=str(f),
                    deletions=[_dbl(1, "a", 3, "c"), _dbl(2, "b", 4, "d")],
                )
            )
        )


@pytest.mark.asyncio
async def test_delete_block_atomicity(tmp_path, monkeypatch, delete_block_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.py"
    original = "a\nb\nc\nd\ne\nf\ng\n"
    f.write_text(original, encoding="utf-8")

    with pytest.raises(ToolError, match="Hash mismatch"):
        await collect_result(
            delete_block_tool.run(
                HashedDeleteBlockArgs(
                    path=str(f),
                    deletions=[
                        _dbl(2, "b", 3, "c"),
                        DeleteBlockItem(line=5, hash="dead", end_line=6, end_hash=_line_hash("f")),
                    ],
                )
            )
        )

    assert f.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# HashedReplaceLine — counts, deletion, context windowing & validation
# (migrated from the former combined HashedReplace tool)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replace_line_reports_counts_and_context(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "code.py"
    f.write_text("line one\nline two\nline three\n", encoding="utf-8")

    result = await collect_result(
        replace_line_tool.run(
            HashedReplaceLineArgs(path=str(f), replacements=[_rli(2, "line two", "replaced two")])
        )
    )

    assert result.total_replacements == 1
    assert result.total_lines_replaced == 1
    assert f.read_text(encoding="utf-8") == "line one\nreplaced two\nline three\n"
    assert "replaced two" in result.context


@pytest.mark.asyncio
async def test_replace_line_empty_content_deletes_line(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "code.py"
    f.write_text("keep\ndelete me\nkeep too\n", encoding="utf-8")

    await collect_result(
        replace_line_tool.run(
            HashedReplaceLineArgs(path=str(f), replacements=[_rli(2, "delete me", "")])
        )
    )

    assert f.read_text(encoding="utf-8") == "keep\nkeep too\n"


@pytest.mark.asyncio
async def test_replace_line_hash_mismatch_includes_context(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "code.py"
    f.write_text("line one\nline two\nline three\n", encoding="utf-8")

    with pytest.raises(ToolError) as exc_info:
        await collect_result(
            replace_line_tool.run(
                HashedReplaceLineArgs(
                    path=str(f),
                    replacements=[ReplaceLineItem(line=2, hash="dead", new_content="x")],
                )
            )
        )

    assert "line two" in str(exc_info.value)


@pytest.mark.asyncio
async def test_replace_line_out_of_range_raises(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "code.py"
    f.write_text("only one line\n", encoding="utf-8")

    with pytest.raises(ToolError, match="out of range"):
        await collect_result(
            replace_line_tool.run(
                HashedReplaceLineArgs(
                    path=str(f),
                    replacements=[ReplaceLineItem(line=99, hash="0000", new_content="x")],
                )
            )
        )


@pytest.mark.asyncio
async def test_replace_line_context_is_windowed_not_full_file(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    lines = [f"line{i}\n" for i in range(1, 31)]
    f = tmp_path / "code.py"
    f.write_text("".join(lines), encoding="utf-8")

    result = await collect_result(
        replace_line_tool.run(
            HashedReplaceLineArgs(path=str(f), replacements=[_rli(15, "line15", "CHANGED")])
        )
    )

    assert "    1 " not in result.context
    assert "   30 " not in result.context
    assert "CHANGED" in result.context
    assert "   14 " in result.context
    assert "   16 " in result.context


@pytest.mark.asyncio
async def test_replace_line_context_has_separator_for_distant_changes(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    lines = [f"line{i}\n" for i in range(1, 51)]
    f = tmp_path / "code.py"
    f.write_text("".join(lines), encoding="utf-8")

    result = await collect_result(
        replace_line_tool.run(
            HashedReplaceLineArgs(
                path=str(f),
                replacements=[_rli(2, "line2", "FIRST"), _rli(49, "line49", "SECOND")],
            )
        )
    )

    assert "FIRST" in result.context
    assert "SECOND" in result.context
    assert "..." in result.context


@pytest.mark.asyncio
async def test_replace_line_context_merges_nearby_changes(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    lines = [f"line{i}\n" for i in range(1, 21)]
    f = tmp_path / "code.py"
    f.write_text("".join(lines), encoding="utf-8")

    result = await collect_result(
        replace_line_tool.run(
            HashedReplaceLineArgs(
                path=str(f),
                replacements=[_rli(5, "line5", "FIRST"), _rli(8, "line8", "SECOND")],
            )
        )
    )

    assert "FIRST" in result.context
    assert "SECOND" in result.context
    assert "..." not in result.context


@pytest.mark.asyncio
async def test_replace_line_context_has_valid_hashes(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "code.py"
    f.write_text("a\nb\nc\nd\n", encoding="utf-8")

    result = await collect_result(
        replace_line_tool.run(
            HashedReplaceLineArgs(path=str(f), replacements=[_rli(2, "b", "REPLACED")])
        )
    )

    assert "REPLACED" in result.context
    for line in result.context.splitlines():
        parts = line.split()
        assert len(parts) >= 2
        h_part = parts[1]
        assert len(h_part) == 4
        assert all(c in "0123456789abcdef" for c in h_part)
