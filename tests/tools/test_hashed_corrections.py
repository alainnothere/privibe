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
    assert strip_leaked_prefix("12|AB12|x") == ("12|AB12|x", 0)


def test_strip_removes_legacy_space_padded_prefix():
    # Resumed sessions carry the pre-pipe format in their read history; a
    # prefix pasted back from there must still strip.
    assert strip_leaked_prefix("   12 ab12  x") == ("x", 1)


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


# ---------------------------------------------------------------------------
# indent correction
# ---------------------------------------------------------------------------

CSHARP = (
    "namespace N\n"
    "{\n"
    "    class C\n"
    "    {\n"
    "        /// <summary>Old</summary>\n"
    "        public void M()\n"
    "        {\n"
    "            var x = 1;\n"
    "        }\n"
    "    }\n"
    "}\n"
)


def _write_csharp(tmp_path):
    f = tmp_path / "f.cs"
    f.write_text(CSHARP, encoding="utf-8")
    return f


@pytest.mark.asyncio
async def test_off_by_one_under_indent_snaps_to_grid(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = _write_csharp(tmp_path)

    # 11 spaces where the replaced line sits at 12 in a 4-space file.
    result = await collect_result(
        replace_line_tool.run(
            HashedReplaceLineArgs(
                path=str(f),
                replacements=[_rli(8, "            var x = 1;", "           var y = 2;")],
            )
        )
    )

    assert "            var y = 2;\n" in f.read_text(encoding="utf-8")
    assert result.content_note is not None
    assert "keep_indent" in result.content_note


@pytest.mark.asyncio
async def test_off_by_one_over_indent_snaps_to_grid(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = _write_csharp(tmp_path)

    result = await collect_result(
        replace_line_tool.run(
            HashedReplaceLineArgs(
                path=str(f),
                replacements=[_rli(8, "            var x = 1;", "             var y = 2;")],
            )
        )
    )

    assert "            var y = 2;\n" in f.read_text(encoding="utf-8")
    assert result.content_note is not None


@pytest.mark.asyncio
async def test_full_indent_step_difference_kept_with_note(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = _write_csharp(tmp_path)

    # 8 spaces vs the replaced line's 12: a full step, assumed intentional.
    result = await collect_result(
        replace_line_tool.run(
            HashedReplaceLineArgs(
                path=str(f),
                replacements=[_rli(8, "            var x = 1;", "        var y = 2;")],
            )
        )
    )

    assert "        var y = 2;\n" in f.read_text(encoding="utf-8")
    assert result.content_note is not None
    assert "kept as written" in result.content_note


@pytest.mark.asyncio
async def test_comment_block_aligns_to_replaced_comment(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = _write_csharp(tmp_path)

    # Three /// lines at 7 spaces replacing the /// line at 8.
    new = "       /// <summary>\n       /// New.\n       /// </summary>"
    result = await collect_result(
        replace_line_tool.run(
            HashedReplaceLineArgs(
                path=str(f),
                replacements=[_rli(5, "        /// <summary>Old</summary>", new)],
            )
        )
    )

    text = f.read_text(encoding="utf-8")
    assert "        /// <summary>\n        /// New.\n        /// </summary>\n" in text
    assert result.content_note is not None
    assert "'///'" in result.content_note


@pytest.mark.asyncio
async def test_comment_alignment_beats_grid_rebase(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = _write_csharp(tmp_path)

    # 4 spaces is on-grid (a full step from 8), but the comment rule still
    # aligns it to the /// line it replaces.
    result = await collect_result(
        replace_line_tool.run(
            HashedReplaceLineArgs(
                path=str(f),
                replacements=[_rli(5, "        /// <summary>Old</summary>", "    /// <summary>New</summary>")],
            )
        )
    )

    assert "        /// <summary>New</summary>\n" in f.read_text(encoding="utf-8")
    assert result.content_note is not None
    assert "comment" in result.content_note


@pytest.mark.asyncio
async def test_marker_mismatch_falls_through_to_rebase(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = _write_csharp(tmp_path)

    # "//" does not match the "///" neighbours, so the comment rule stays out;
    # the off-grid base (7) still snaps to 8 via the rebase.
    result = await collect_result(
        replace_line_tool.run(
            HashedReplaceLineArgs(
                path=str(f),
                replacements=[_rli(5, "        /// <summary>Old</summary>", "       // plain")],
            )
        )
    )

    assert "        // plain\n" in f.read_text(encoding="utf-8")
    assert result.content_note is not None
    assert "shifted" in result.content_note


@pytest.mark.asyncio
async def test_rebase_preserves_relative_nesting(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = _write_csharp(tmp_path)

    new = "           if (a)\n           {\n               b();\n           }"
    result = await collect_result(
        replace_line_tool.run(
            HashedReplaceLineArgs(
                path=str(f),
                replacements=[_rli(8, "            var x = 1;", new)],
            )
        )
    )

    text = f.read_text(encoding="utf-8")
    assert "            if (a)\n            {\n                b();\n            }\n" in text
    assert result.content_note is not None


@pytest.mark.asyncio
async def test_keep_indent_writes_verbatim(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = _write_csharp(tmp_path)

    result = await collect_result(
        replace_line_tool.run(
            HashedReplaceLineArgs(
                path=str(f),
                replacements=[_rli(8, "            var x = 1;", "           var y = 2;")],
                keep_indent=True,
            )
        )
    )

    assert "           var y = 2;\n" in f.read_text(encoding="utf-8")
    assert result.content_note is None


@pytest.mark.asyncio
async def test_matching_indent_no_note(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = _write_csharp(tmp_path)

    result = await collect_result(
        replace_line_tool.run(
            HashedReplaceLineArgs(
                path=str(f),
                replacements=[_rli(8, "            var x = 1;", "            var z = 3;")],
            )
        )
    )

    assert "            var z = 3;\n" in f.read_text(encoding="utf-8")
    assert result.content_note is None


@pytest.mark.asyncio
async def test_tab_indented_context_skipped_with_note(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.c"
    f.write_text("void f()\n{\n\tint x;\n}\n", encoding="utf-8")

    result = await collect_result(
        replace_line_tool.run(
            HashedReplaceLineArgs(
                path=str(f),
                replacements=[_rli(3, "\tint x;", "  int y;")],
            )
        )
    )

    assert "  int y;\n" in f.read_text(encoding="utf-8")
    assert result.content_note is not None
    assert "tab" in result.content_note


@pytest.mark.asyncio
async def test_block_replace_off_grid_base_shifts_uniformly(tmp_path, monkeypatch, replace_block_tool):
    monkeypatch.chdir(tmp_path)
    f = _write_csharp(tmp_path)

    new = "       public void M2()\n       {\n           var y = 2;\n       }"
    result = await collect_result(
        replace_block_tool.run(
            HashedReplaceBlockArgs(
                path=str(f),
                replacements=[_rbl(6, "        public void M()", 9, "        }", new)],
            )
        )
    )

    text = f.read_text(encoding="utf-8")
    assert "        public void M2()\n        {\n            var y = 2;\n        }\n" in text
    assert result.content_note is not None
    assert "keep_indent" in result.content_note


# ---------------------------------------------------------------------------
# method-chain continuation snap
# ---------------------------------------------------------------------------

FLUENT = (
    "public void Configure()\n"
    "{\n"
    "    builder.Entity<X>()\n"
    "        .Property(k => k.CycleDate)\n"
    "        .HasMaxLength(9)\n"
    '        .HasComment("old");\n'
    "}\n"
)


@pytest.mark.asyncio
async def test_chain_continuation_at_column_zero_snaps(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.cs"
    f.write_text(FLUENT, encoding="utf-8")

    # The stray line is at column 0: shallower than its ".HasMaxLength"
    # sibling (8) and at-or-below its "builder..." head (4), so it snaps to 8.
    result = await collect_result(
        replace_line_tool.run(
            HashedReplaceLineArgs(
                path=str(f),
                replacements=[_rli(6, '        .HasComment("old");', '.HasComment("new");')],
            )
        )
    )

    assert '        .HasComment("new");\n' in f.read_text(encoding="utf-8")
    assert result.content_note is not None
    assert "method-chain" in result.content_note


@pytest.mark.asyncio
async def test_chain_stray_inside_block_snaps(tmp_path, monkeypatch, replace_block_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.cs"
    f.write_text(FLUENT, encoding="utf-8")

    new = (
        "    builder.Entity<X>()\n"
        "        .Property(k => k.CorrespondentFiId)\n"
        '.HasComment("cc");'
    )
    result = await collect_result(
        replace_block_tool.run(
            HashedReplaceBlockArgs(
                path=str(f),
                replacements=[
                    _rbl(3, "    builder.Entity<X>()", 6, '        .HasComment("old");', new)
                ],
            )
        )
    )

    text = f.read_text(encoding="utf-8")
    assert '        .HasComment("cc");\n' in text
    assert "        .Property(k => k.CorrespondentFiId)\n" in text
    assert result.content_note is not None
    assert "method-chain" in result.content_note


@pytest.mark.asyncio
async def test_nested_chain_dedent_not_touched(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.cs"
    f.write_text(
        "var q = query\n"
        "    .Select(x => x.Items\n"
        "        .Where(i => i.Active)\n"
        "        .ToList())\n"
        "    .OrderBy(x => x.Id);\n",
        encoding="utf-8",
    )

    # ".OrderBy" at 4 is shallower than the inner ".ToList())" at 8 but still
    # deeper than the "var q" head at 0: a legitimate outer-chain dedent that
    # must never be "corrected".
    result = await collect_result(
        replace_line_tool.run(
            HashedReplaceLineArgs(
                path=str(f),
                replacements=[_rli(5, "    .OrderBy(x => x.Id);", "    .OrderBy(x => x.Name);")],
            )
        )
    )

    assert "    .OrderBy(x => x.Name);\n" in f.read_text(encoding="utf-8")
    assert result.content_note is None


@pytest.mark.asyncio
async def test_chain_without_sibling_left_alone(tmp_path, monkeypatch, replace_line_tool):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "f.cs"
    f.write_text("var a = b\n.C();\n", encoding="utf-8")

    # First continuation of the chain: no "." sibling to copy an indent from,
    # so the rule stays out rather than guessing.
    result = await collect_result(
        replace_line_tool.run(
            HashedReplaceLineArgs(path=str(f), replacements=[_rli(2, ".C();", ".D();")])
        )
    )

    assert ".D();\n" in f.read_text(encoding="utf-8")
    assert result.content_note is None
