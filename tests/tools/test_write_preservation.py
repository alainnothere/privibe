"""The alignment tests: edit one line, and git says one line in, one line out.

Every writer goes through the same codec, so every writer gets the same
round-trip property: bytes the model did not name are bytes that did not
change. CRLF, BOM, mixed endings, a missing final newline and a cp1252 file
each get a turn, and the ASCII policy gets exercised through each tool's
``prepare_args`` the way the agent loop calls it.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from privibe.core.rewind.diffing import build_file_diff
from privibe.core.tools.base import BaseToolConfig, BaseToolState, ToolError
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
from privibe.core.tools.builtins.read_file import (
    ReadFile,
    ReadFileArgs,
    ReadFileToolConfig,
)
from privibe.core.tools.builtins.search_replace import (
    SearchReplace,
    SearchReplaceArgs,
    SearchReplaceConfig,
)
from privibe.core.tools.builtins.write_file import (
    WriteFile,
    WriteFileArgs,
    WriteFileConfig,
)
from privibe.core.utils import newfile_conventions as nc
from privibe.core.utils.asciify import set_session_ascii_override
from tests.mock.utils import collect_result

pytestmark = pytest.mark.asyncio

CRLF_BOM_NOFINAL = b"\xef\xbb\xbfone\r\ntwo\r\nthree\r\nfour"
MIXED = b"a\nb\r\nc\r\nd\n"


@pytest.fixture(autouse=True)
def _isolation():
    nc.clear_caches()
    set_session_ascii_override(None)
    yield
    nc.clear_caches()
    set_session_ascii_override(None)


@pytest.fixture
def write_tool() -> WriteFile:
    return WriteFile(config=WriteFileConfig(), state=BaseToolState())


@pytest.fixture
def sr_tool() -> SearchReplace:
    return SearchReplace(config=SearchReplaceConfig(), state=BaseToolState())


@pytest.fixture
def line_tool() -> HashedReplaceLine:
    return HashedReplaceLine(config=BaseToolConfig(), state=BaseToolState())


@pytest.fixture
def block_tool() -> HashedReplaceBlock:
    return HashedReplaceBlock(config=BaseToolConfig(), state=BaseToolState())


@pytest.fixture
def delete_tool() -> HashedDeleteLine:
    return HashedDeleteLine(config=BaseToolConfig(), state=BaseToolState())


def _sr(search: str, replace: str) -> str:
    return f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE"


async def _run(tool, args):
    tool.prepare_args(args)
    return await collect_result(tool.run(args))


# ---------------------------------------------------------------------------
# hashed_replace_line
# ---------------------------------------------------------------------------


async def test_replace_line_keeps_crlf_bom_and_missing_final_newline(
    tmp_path: Path, line_tool: HashedReplaceLine
):
    f = tmp_path / "x.cs"
    f.write_bytes(CRLF_BOM_NOFINAL)
    await _run(
        line_tool,
        HashedReplaceLineArgs(
            path=str(f),
            replacements=[ReplaceLineItem(line=2, hash=_line_hash("two"), new_content="TWO")],
        ),
    )
    assert f.read_bytes() == b"\xef\xbb\xbfone\r\nTWO\r\nthree\r\nfour"


async def test_replace_line_expanding_block_follows_its_ending(
    tmp_path: Path, line_tool: HashedReplaceLine
):
    f = tmp_path / "x.txt"
    f.write_bytes(MIXED)
    await _run(
        line_tool,
        HashedReplaceLineArgs(
            path=str(f),
            replacements=[ReplaceLineItem(line=2, hash=_line_hash("b"), new_content="b1\nb2")],
        ),
    )
    assert f.read_bytes() == b"a\nb1\r\nb2\r\nc\r\nd\n"


async def test_replace_line_no_note_noise_on_plain_file(
    tmp_path: Path, line_tool: HashedReplaceLine
):
    f = tmp_path / "x.txt"
    f.write_bytes(b"a\r\nb\r\n")
    r = await _run(
        line_tool,
        HashedReplaceLineArgs(
            path=str(f),
            replacements=[ReplaceLineItem(line=1, hash=_line_hash("a"), new_content="A")],
        ),
    )
    assert r.content_note is None
    assert f.read_bytes() == b"A\r\nb\r\n"


async def test_replace_line_mixed_file_is_noted_once(
    tmp_path: Path, line_tool: HashedReplaceLine
):
    f = tmp_path / "x.txt"
    f.write_bytes(MIXED)
    r = await _run(
        line_tool,
        HashedReplaceLineArgs(
            path=str(f),
            replacements=[ReplaceLineItem(line=1, hash=_line_hash("a"), new_content="A")],
        ),
    )
    assert r.content_note is not None and "mixed line endings" in r.content_note


async def test_replace_line_transliterates_and_notes(
    tmp_path: Path, line_tool: HashedReplaceLine
):
    f = tmp_path / "x.py"
    f.write_bytes(b"x = 1\n")
    r = await _run(
        line_tool,
        HashedReplaceLineArgs(
            path=str(f),
            replacements=[
                ReplaceLineItem(line=1, hash=_line_hash("x = 1"), new_content="x = 1  # café → ok")
            ],
        ),
    )
    assert f.read_bytes() == b"x = 1  # cafe -> ok\n"
    assert r.content_note is not None
    assert "allow_unicode=true" in r.content_note
    assert "rightwards arrow" in r.content_note


async def test_replace_line_allow_unicode_writes_utf8(
    tmp_path: Path, line_tool: HashedReplaceLine
):
    f = tmp_path / "x.py"
    f.write_bytes(b"x = 1\n")
    r = await _run(
        line_tool,
        HashedReplaceLineArgs(
            path=str(f),
            allow_unicode=True,
            replacements=[ReplaceLineItem(line=1, hash=_line_hash("x = 1"), new_content="x = 'é'")],
        ),
    )
    assert f.read_bytes() == "x = 'é'\n".encode()
    assert r.content_note is None


async def test_replace_line_cp1252_file_written_back_as_cp1252(
    tmp_path: Path, line_tool: HashedReplaceLine
):
    f = tmp_path / "legacy.cs"
    f.write_bytes("// café\r\nint x;\r\n".encode("cp1252"))
    r = await _run(
        line_tool,
        HashedReplaceLineArgs(
            path=str(f),
            allow_unicode=True,
            replacements=[ReplaceLineItem(line=2, hash=_line_hash("int x;"), new_content="int niño;")],
        ),
    )
    assert f.read_bytes() == "// café\r\nint niño;\r\n".encode("cp1252")
    assert r.content_note is not None and "cp1252" in r.content_note


async def test_replace_line_cp1252_cannot_take_a_rocket(
    tmp_path: Path, line_tool: HashedReplaceLine
):
    f = tmp_path / "legacy.cs"
    f.write_bytes("// café\r\nint x;\r\n".encode("cp1252"))
    with pytest.raises(ToolError, match="cannot be written in this file's encoding"):
        await _run(
            line_tool,
            HashedReplaceLineArgs(
                path=str(f),
                allow_unicode=True,
                replacements=[ReplaceLineItem(line=2, hash=_line_hash("int x;"), new_content="// \U0001f680")],
            ),
        )
    assert f.read_bytes() == "// café\r\nint x;\r\n".encode("cp1252")


async def test_replace_line_binary_refused(tmp_path: Path, line_tool: HashedReplaceLine):
    f = tmp_path / "blob.bin"
    f.write_bytes(b"\x00\x01\x02")
    with pytest.raises(ToolError, match="binary"):
        await _run(
            line_tool,
            HashedReplaceLineArgs(
                path=str(f), replacements=[ReplaceLineItem(line=1, hash="0000", new_content="x")]
            ),
        )


async def test_replace_line_blends_tabs(tmp_path: Path, line_tool: HashedReplaceLine):
    f = tmp_path / "Makefile"
    f.write_bytes(b"all:\n\techo one\n\techo two\n")
    r = await _run(
        line_tool,
        HashedReplaceLineArgs(
            path=str(f),
            replacements=[ReplaceLineItem(line=2, hash=_line_hash("\techo one"), new_content="    echo ONE")],
        ),
    )
    assert f.read_bytes() == b"all:\n\techo ONE\n\techo two\n"
    assert r.content_note is not None and "tabs" in r.content_note


async def test_replace_line_editorconfig_trims_only_new_lines(
    tmp_path: Path, line_tool: HashedReplaceLine
):
    (tmp_path / ".editorconfig").write_text("[*]\ntrim_trailing_whitespace = true\n")
    f = tmp_path / "x.py"
    f.write_bytes(b"a   \nb\nc   \n")
    await _run(
        line_tool,
        HashedReplaceLineArgs(
            path=str(f),
            replacements=[ReplaceLineItem(line=2, hash=_line_hash("b"), new_content="B   ")],
        ),
    )
    assert f.read_bytes() == b"a   \nB\nc   \n"


# ---------------------------------------------------------------------------
# hashed_replace_block / hashed_delete_line
# ---------------------------------------------------------------------------


async def test_replace_block_crlf(tmp_path: Path, block_tool: HashedReplaceBlock):
    f = tmp_path / "x.txt"
    f.write_bytes(b"1\r\n2\r\n3\r\n4\r\n")
    await _run(
        block_tool,
        HashedReplaceBlockArgs(
            path=str(f),
            replacements=[
                ReplaceBlockItem(
                    line=2, hash=_line_hash("2"), end_line=3, end_hash=_line_hash("3"), new_content="two\nthree"
                )
            ],
        ),
    )
    assert f.read_bytes() == b"1\r\ntwo\r\nthree\r\n4\r\n"


async def test_delete_line_crlf_bom(tmp_path: Path, delete_tool: HashedDeleteLine):
    f = tmp_path / "x.txt"
    f.write_bytes(CRLF_BOM_NOFINAL)
    await _run(
        delete_tool,
        HashedDeleteLineArgs(path=str(f), deletions=[DeleteLineItem(line=2, hash=_line_hash("two"))]),
    )
    assert f.read_bytes() == b"\xef\xbb\xbfone\r\nthree\r\nfour"


# ---------------------------------------------------------------------------
# search_replace
# ---------------------------------------------------------------------------


async def test_search_replace_splices_into_mixed_file(tmp_path: Path, sr_tool: SearchReplace):
    f = tmp_path / "x.txt"
    f.write_bytes(MIXED)
    await _run(sr_tool, SearchReplaceArgs(file_path=str(f), content=_sr("b\nc", "B\nC")))
    assert f.read_bytes() == b"a\nB\r\nC\r\nd\n"


async def test_search_replace_crlf_search_text_matches_lf_normalized(
    tmp_path: Path, sr_tool: SearchReplace
):
    f = tmp_path / "x.txt"
    f.write_bytes(b"\xef\xbb\xbfone\r\ntwo\r\nthree")
    r = await _run(
        sr_tool, SearchReplaceArgs(file_path=str(f), content=_sr("one\r\ntwo", "ONE\r\nTWO"))
    )
    assert r.blocks_applied == 1
    assert f.read_bytes() == b"\xef\xbb\xbfONE\r\nTWO\r\nthree"


async def test_search_replace_transliterates_replace_only(
    tmp_path: Path, sr_tool: SearchReplace
):
    f = tmp_path / "x.md"
    f.write_bytes("café — old\n".encode())
    args = SearchReplaceArgs(
        file_path=str(f), content=_sr("café — old", "café — new")
    )
    r = await _run(sr_tool, args)
    assert f.read_bytes() == b"cafe - new\n"
    assert "block 1 REPLACE" in (r.content_note or "")
    # The echoed content is what was applied: SEARCH untouched, REPLACE ASCII.
    assert "café — old" in r.content
    assert "cafe - new" in r.content


async def test_search_replace_no_write_when_nothing_changes(
    tmp_path: Path, sr_tool: SearchReplace
):
    f = tmp_path / "x.txt"
    f.write_bytes(b"a\r\nb\r\n")
    before = f.stat().st_mtime_ns
    r = await _run(sr_tool, SearchReplaceArgs(file_path=str(f), content=_sr("a", "a")))
    assert r.lines_changed == 0
    assert f.read_bytes() == b"a\r\nb\r\n"
    assert f.stat().st_mtime_ns == before


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


async def test_write_file_new_file_follows_editorconfig_crlf_and_bom(
    tmp_path: Path, write_tool: WriteFile
):
    (tmp_path / ".editorconfig").write_text("[*.ps1]\nend_of_line = crlf\ncharset = utf-8-bom\n")
    f = tmp_path / "s.ps1"
    r = await _run(write_tool, WriteFileArgs(path=str(f), content="Write-Host hi\nexit 0"))
    assert f.read_bytes() == b"\xef\xbb\xbfWrite-Host hi\r\nexit 0\r\n"
    assert r.bytes_written == len(f.read_bytes())
    assert r.content == "Write-Host hi\nexit 0"


async def test_write_file_new_shell_script_is_lf_even_when_siblings_are_crlf(
    tmp_path: Path, write_tool: WriteFile
):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "core.autocrlf", "false"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_bytes(b"x\r\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    f = tmp_path / "run.sh"
    await _run(write_tool, WriteFileArgs(path=str(f), content="#!/bin/sh\necho hi\n"))
    assert f.read_bytes() == b"#!/bin/sh\necho hi\n"


async def test_write_file_overwrite_blends_with_the_file(
    tmp_path: Path, write_tool: WriteFile
):
    f = tmp_path / "x.cs"
    f.write_bytes(CRLF_BOM_NOFINAL)
    await _run(
        write_tool, WriteFileArgs(path=str(f), content="one\nTWO\nthree\nfour", overwrite=True)
    )
    assert f.read_bytes() == b"\xef\xbb\xbfone\r\nTWO\r\nthree\r\nfour"


async def test_write_file_overwrite_with_explicit_newline_converts(
    tmp_path: Path, write_tool: WriteFile
):
    f = tmp_path / "x.txt"
    f.write_bytes(b"a\r\nb\r\n")
    r = await _run(
        write_tool, WriteFileArgs(path=str(f), content="a\nb\n", overwrite=True, newline="lf")
    )
    assert f.read_bytes() == b"a\nb\n"
    assert "converted to LF" in (r.content_note or "")


async def test_write_file_explicit_encoding_utf8_sig(tmp_path: Path, write_tool: WriteFile):
    f = tmp_path / "x.csv"
    await _run(write_tool, WriteFileArgs(path=str(f), content="a,b\n", encoding="utf-8-sig"))
    assert f.read_bytes() == b"\xef\xbb\xbfa,b\n"


async def test_write_file_unknown_encoding_is_an_error(tmp_path: Path, write_tool: WriteFile):
    f = tmp_path / "x.txt"
    with pytest.raises(ToolError, match="Unknown encoding"):
        await _run(write_tool, WriteFileArgs(path=str(f), content="a", encoding="klingon"))


async def test_write_file_ascii_default_transliterates_and_echoes_landed_content(
    tmp_path: Path, write_tool: WriteFile
):
    f = tmp_path / "x.md"
    args = WriteFileArgs(path=str(f), content="Café — \U0001f680 launch\n")
    write_tool.prepare_args(args)
    # The approval preview reads args.content: it must already be the ASCII.
    assert args.content == "Cafe -  launch\n"
    r = await collect_result(write_tool.run(args))
    assert f.read_bytes() == b"Cafe -  launch\n"
    assert r.content == args.content
    assert "rocket dropped" in (r.content_note or "")


async def test_write_file_rocketman_mode_config(tmp_path: Path):
    tool = WriteFile(config=WriteFileConfig(ascii_default=False), state=BaseToolState())
    f = tmp_path / "x.md"
    r = await _run(tool, WriteFileArgs(path=str(f), content="\U0001f680\n"))
    assert f.read_bytes() == "\U0001f680\n".encode()
    assert r.content_note is None


async def test_write_file_session_override_beats_config(tmp_path: Path, write_tool: WriteFile):
    set_session_ascii_override(False)
    f = tmp_path / "x.md"
    await _run(write_tool, WriteFileArgs(path=str(f), content="é\n"))
    assert f.read_bytes() == "é\n".encode()
    # ...and an explicit false on the call beats the session override.
    g = tmp_path / "y.md"
    await _run(write_tool, WriteFileArgs(path=str(g), content="é\n", allow_unicode=False))
    assert g.read_bytes() == b"e\n"


async def test_write_file_ascii_file_gains_unicode_as_utf8_no_bom(
    tmp_path: Path, write_tool: WriteFile
):
    f = tmp_path / "x.py"
    f.write_bytes(b"x = 1\n")
    await _run(
        write_tool,
        WriteFileArgs(path=str(f), content="x = '\U0001f680'\n", overwrite=True, allow_unicode=True),
    )
    assert f.read_bytes() == "x = '\U0001f680'\n".encode()


# ---------------------------------------------------------------------------
# readers never show the BOM
# ---------------------------------------------------------------------------


async def test_hashed_read_hides_bom_and_hash_matches_edit_tools(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_bytes(CRLF_BOM_NOFINAL)
    tool = HashedRead(config=HashedReadConfig(), state=BaseToolState())
    r = await collect_result(tool.run(HashedReadArgs(path=str(f))))
    first = r.content.splitlines()[0]
    assert "﻿" not in first
    assert first == f"1|{_line_hash('one')}|one"


async def test_read_file_hides_bom(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_bytes(CRLF_BOM_NOFINAL)
    tool = ReadFile(config=ReadFileToolConfig(), state=BaseToolState())
    r = await collect_result(tool.run(ReadFileArgs(path=str(f))))
    assert "﻿" not in r.content
    assert r.content.startswith("one")


# ---------------------------------------------------------------------------
# the diff view says when only invisible bytes moved
# ---------------------------------------------------------------------------


def test_diff_header_names_an_ending_flip():
    d = build_file_diff("x", b"a\r\nb\r\n", b"a\nb\n")
    assert d is not None
    assert d.hunks[0][0] == ("diff-header", "line endings CRLF -> LF")


def test_diff_header_names_a_bom_change():
    d = build_file_diff("x", b"\xef\xbb\xbfa\n", b"a\n")
    assert d is not None
    assert d.hunks[0][0][1] == "BOM removed"


def test_diff_no_header_for_a_normal_edit():
    d = build_file_diff("x", b"a\r\nb\r\n", b"a\r\nB\r\n")
    assert d is not None
    assert d.hunks[0][0][0] != "diff-header"


# ---------------------------------------------------------------------------
# the one that matters: git counts one line in, one line out
# ---------------------------------------------------------------------------


async def test_git_numstat_one_line_in_one_line_out(tmp_path: Path, line_tool: HashedReplaceLine):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "core.autocrlf", "false"], cwd=tmp_path, check=True)
    f = tmp_path / "Migration.cs"
    f.write_bytes(CRLF_BOM_NOFINAL)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=tmp_path, check=True)

    await _run(
        line_tool,
        HashedReplaceLineArgs(
            path=str(f),
            replacements=[ReplaceLineItem(line=3, hash=_line_hash("three"), new_content="THREE")],
        ),
    )
    out = subprocess.run(
        ["git", "diff", "--numstat"], cwd=tmp_path, check=True, capture_output=True, text=True
    ).stdout
    assert out.split("\t")[:2] == ["1", "1"], out
