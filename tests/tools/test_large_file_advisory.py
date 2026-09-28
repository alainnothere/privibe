"""Large-file advisory: naive whole-file reads of over-threshold files get a
head preview plus an advisory; targeted reads pass through untouched."""

from __future__ import annotations

from pathlib import Path

import pytest

from privibe.core.tools.base import BaseToolState
from privibe.core.tools.builtins.hashed_read import (
    HashedRead,
    HashedReadArgs,
    HashedReadConfig,
)
from privibe.core.tools.builtins.read_file import (
    ReadFile,
    ReadFileArgs,
    ReadFileState,
    ReadFileToolConfig,
)
from privibe.core.tools.utils import count_file_lines
from tests.mock.utils import collect_result

# 4 KB threshold / 1 KB preview keep the fixtures small.
THRESHOLD_KB = 4
PREVIEW_KB = 1

LINE = "x" * 99 + "\n"  # 100 bytes per line


def _write_big_file(tmp_path: Path) -> Path:
    f = tmp_path / "big.txt"
    f.write_text(LINE * 100, encoding="utf-8")  # ~10 KB, over threshold
    return f


def _write_small_file(tmp_path: Path) -> Path:
    f = tmp_path / "small.txt"
    f.write_text(LINE * 10, encoding="utf-8")  # ~1 KB, under threshold
    return f


def _write_long_line_file(tmp_path: Path) -> Path:
    """Ten short lines, then one 8 KB line with no trailing newline: 11 lines.

    The old size-over-preview-average estimator saw only the short lines and
    reported hundreds. The count must say 11 and honor the unterminated tail.
    """
    f = tmp_path / "longline.txt"
    f.write_text("short\n" * 10 + "y" * 8192, encoding="utf-8")
    return f


@pytest.fixture
def read_file_tool() -> ReadFile:
    config = ReadFileToolConfig(
        large_file_threshold_kb=THRESHOLD_KB, large_file_preview_kb=PREVIEW_KB
    )
    return ReadFile(config=config, state=ReadFileState())


@pytest.fixture
def hashed_read_tool() -> HashedRead:
    config = HashedReadConfig(
        large_file_threshold_kb=THRESHOLD_KB, large_file_preview_kb=PREVIEW_KB
    )
    return HashedRead(config=config, state=BaseToolState())


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_naive_read_of_big_file_returns_preview_and_advisory(
    tmp_path, monkeypatch, read_file_tool
):
    monkeypatch.chdir(tmp_path)
    f = _write_big_file(tmp_path)

    result = await collect_result(read_file_tool.run(ReadFileArgs(path=str(f))))

    assert result.advisory is not None
    assert "LARGE FILE" in result.advisory
    assert "grep" in result.advisory
    assert "100 lines" in result.advisory
    assert "roughly" not in result.advisory
    assert result.total_lines == 100
    assert result.was_truncated
    # Preview honors the preview cap, not max_read_bytes.
    assert len(result.content.encode("utf-8")) <= PREVIEW_KB * 1024
    assert result.lines_read == 10  # 1 KB / 100-byte lines


@pytest.mark.asyncio
async def test_targeted_read_of_big_file_is_untouched(
    tmp_path, monkeypatch, read_file_tool
):
    monkeypatch.chdir(tmp_path)
    f = _write_big_file(tmp_path)

    with_limit = await collect_result(
        read_file_tool.run(ReadFileArgs(path=str(f), limit=50))
    )
    assert with_limit.advisory is None
    assert with_limit.total_lines is None
    assert with_limit.lines_read == 50

    with_start_line = await collect_result(
        read_file_tool.run(ReadFileArgs(path=str(f), start_line=91))
    )
    assert with_start_line.advisory is None
    assert with_start_line.lines_read == 10


@pytest.mark.asyncio
async def test_small_file_naive_read_is_untouched(
    tmp_path, monkeypatch, read_file_tool
):
    monkeypatch.chdir(tmp_path)
    f = _write_small_file(tmp_path)

    result = await collect_result(read_file_tool.run(ReadFileArgs(path=str(f))))

    assert result.advisory is None
    assert not result.was_truncated
    assert result.lines_read == 10


@pytest.mark.asyncio
async def test_threshold_is_configurable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f = _write_big_file(tmp_path)  # ~10 KB

    generous = ReadFile(
        config=ReadFileToolConfig(large_file_threshold_kb=64),
        state=ReadFileState(),
    )
    result = await collect_result(generous.run(ReadFileArgs(path=str(f))))

    assert result.advisory is None
    assert result.lines_read == 100


# ---------------------------------------------------------------------------
# hashed_read
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hashed_naive_read_of_big_file_returns_preview_and_advisory(
    tmp_path, monkeypatch, hashed_read_tool
):
    monkeypatch.chdir(tmp_path)
    f = _write_big_file(tmp_path)

    result = await collect_result(hashed_read_tool.run(HashedReadArgs(path=str(f))))

    assert result.advisory is not None
    assert "LARGE FILE" in result.advisory
    assert "100 lines" in result.advisory
    assert "roughly" not in result.advisory
    assert result.total_lines == 100
    assert result.was_truncated
    assert result.lines_read == 10
    # Output still carries usable line|hash| addresses for the preview.
    assert result.content.splitlines()[0].startswith("1|")


@pytest.mark.asyncio
async def test_hashed_targeted_read_of_big_file_is_untouched(
    tmp_path, monkeypatch, hashed_read_tool
):
    monkeypatch.chdir(tmp_path)
    f = _write_big_file(tmp_path)

    result = await collect_result(
        hashed_read_tool.run(HashedReadArgs(path=str(f), start_line=41, limit=20))
    )

    assert result.advisory is None
    assert result.total_lines is None
    assert result.lines_read == 20
    assert result.start_line == 41


@pytest.mark.asyncio
async def test_hashed_small_file_naive_read_is_untouched(
    tmp_path, monkeypatch, hashed_read_tool
):
    monkeypatch.chdir(tmp_path)
    f = _write_small_file(tmp_path)

    result = await collect_result(hashed_read_tool.run(HashedReadArgs(path=str(f))))

    assert result.advisory is None
    assert not result.was_truncated


# ---------------------------------------------------------------------------
# exact line count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_advisory_counts_long_line_file_exactly(
    tmp_path, monkeypatch, read_file_tool, hashed_read_tool
):
    monkeypatch.chdir(tmp_path)
    f = _write_long_line_file(tmp_path)

    plain = await collect_result(read_file_tool.run(ReadFileArgs(path=str(f))))
    hashed = await collect_result(hashed_read_tool.run(HashedReadArgs(path=str(f))))

    for result in (plain, hashed):
        assert result.advisory is not None
        assert result.total_lines == 11
        assert "11 lines" in result.advisory


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"", 0),
        (b"a\n", 1),
        (b"a", 1),
        (b"a\nb", 2),
        (b"a\nb\n", 2),
        (b"a\r\nb\r\n", 2),
        (b"\n\n\n", 3),
        # Newline exactly at the 1 MB chunk boundary, then a partial tail.
        (b"x" * (1 << 20) + b"\ny", 2),
        (b"x" * (1 << 20) + b"\n", 1),
    ],
)
def test_count_file_lines_matches_reader_numbering(tmp_path, data, expected):
    f = tmp_path / "f.bin"
    f.write_bytes(data)

    assert count_file_lines(f) == expected
