from __future__ import annotations

from privibe.core.tools.builtins.grep import Grep, GrepResult
from privibe.core.tools.builtins.hashed_read import HashedRead, HashedReadResult
from privibe.core.tools.builtins.read_file import ReadFile, ReadFileResult
from privibe.core.tools.builtins.search_replace import (
    SearchReplace,
    SearchReplaceResult,
)
from privibe.core.tools.builtins.websearch import WebSearch, WebSearchResult
from privibe.core.tools.builtins.write_file import WriteFile, WriteFileResult
from privibe.core.tools.ui import ToolUIDataAdapter, prefix_tool_name
from privibe.core.tools.utils import line_range
from privibe.core.types import ToolResultEvent


def _evt(tool_cls, result=None, **kw) -> ToolResultEvent:
    return ToolResultEvent(
        tool_name=tool_cls.get_name(),
        tool_class=tool_cls,
        result=result,
        tool_call_id="t",
        **kw,
    )


def _line(tool_cls, result=None, **kw) -> str:
    """What every UI shows: the adapter line, tool name first."""
    return (
        ToolUIDataAdapter(tool_cls)
        .get_result_display(_evt(tool_cls, result, **kw))
        .message
    )


# ---------------------------------------------------------------------------
# adapter — the tool name leads every line, exactly once
# ---------------------------------------------------------------------------


def test_prefix_tool_name_adds_once():
    assert (
        prefix_tool_name("grep", '"x" in /p: 3 matches') == 'grep "x" in /p: 3 matches'
    )
    assert prefix_tool_name("grep", "grep already here") == "grep already here"
    assert prefix_tool_name("grep", "grep") == "grep"
    assert prefix_tool_name("grep", "") == "grep"
    # "grepping" is not the tool name
    assert prefix_tool_name("grep", "grepping x") == "grep grepping x"


def test_adapter_error_and_skip_lines_lead_with_tool_name():
    assert _line(Grep, error="boom") == "grep error: boom"
    assert _line(Grep, skipped=True, skip_reason="nope") == "grep skipped: nope"
    assert _line(Grep, skipped=True) == "grep skipped: Skipped"


def test_adapter_success_line_leads_with_tool_name():
    r = GrepResult(
        matches="", match_count=3, was_truncated=False, pattern="x", path="/src"
    )
    assert _line(Grep, r) == 'grep "x" in /src: 3 matches'


# ---------------------------------------------------------------------------
# line_range
# ---------------------------------------------------------------------------


def test_line_range():
    assert line_range(100, 21) == "lines 100 to 120 (21 lines)"
    assert line_range(7, 1) == "line 7"
    assert line_range(7, 0) == "0 lines"


# ---------------------------------------------------------------------------
# grep — pattern, where, how many; model exclusion
# ---------------------------------------------------------------------------


def test_grep_display_includes_pattern_path_and_count():
    r = GrepResult(
        matches="", match_count=20, was_truncated=True, pattern="bla", path="/abs/src"
    )
    assert _line(Grep, r) == 'grep "bla" in /abs/src: 20 matches (truncated)'


def test_grep_display_singular_match():
    r = GrepResult(
        matches="", match_count=1, was_truncated=False, pattern="x", path="/p"
    )
    assert _line(Grep, r) == 'grep "x" in /p: 1 match'


def test_grep_display_fields_excluded_from_model_result():
    r = GrepResult(
        matches="", match_count=1, was_truncated=False, pattern="x", path="y"
    )
    dumped = r.model_dump()
    assert "pattern" not in dumped
    assert "path" not in dumped
    assert "used_gnu_grep" not in dumped


def test_grep_display_warns_on_gnu_grep_fallback():
    r = GrepResult(
        matches="", match_count=1, was_truncated=False, pattern="x", used_gnu_grep=True
    )
    d = Grep.get_result_display(_evt(Grep, r))
    assert any("ripgrep (rg) not found" in w for w in d.warnings)


def test_grep_display_no_fallback_warning_on_ripgrep():
    r = GrepResult(matches="", match_count=1, was_truncated=False, pattern="x")
    d = Grep.get_result_display(_evt(Grep, r))
    assert d.warnings == []


# ---------------------------------------------------------------------------
# file tools — full path, so it can be copied straight off the screen
# ---------------------------------------------------------------------------


def test_hashed_read_display_has_full_path_and_line_range(tmp_path):
    path = str(tmp_path / "a.py")
    r = HashedReadResult(
        path=path, content="", start_line=100, lines_read=21, was_truncated=False
    )
    assert _line(HashedRead, r) == f"hashed_read {path} lines 100 to 120 (21 lines)"


def test_read_file_display_has_full_path_and_line_range(tmp_path):
    path = str(tmp_path / "pkg" / "a.py")
    r = ReadFileResult(
        path=path, content="", lines_read=5, was_truncated=False, start_line=3
    )
    assert _line(ReadFile, r) == f"read_file {path} lines 3 to 7 (5 lines)"


def test_read_file_start_line_excluded_from_model_result():
    r = ReadFileResult(path="/a", content="", lines_read=1, was_truncated=False)
    assert "start_line" not in r.model_dump()


def test_write_file_display_has_full_path(tmp_path):
    path = str(tmp_path / "a.py")
    r = WriteFileResult(path=path, bytes_written=3, file_existed=False, content="abc")
    assert _line(WriteFile, r) == f"write_file created {path}"


def test_search_replace_display_has_full_path(tmp_path):
    path = str(tmp_path / "a.py")
    r = SearchReplaceResult(file=path, blocks_applied=2, lines_changed=1, content="")
    assert _line(SearchReplace, r) == f"search_replace {path}: 2 blocks applied"


# ---------------------------------------------------------------------------
# websearch — query echoed, excluded from model result
# ---------------------------------------------------------------------------


def test_websearch_display_includes_query():
    r = WebSearchResult(answer="", sources=[], query="cats")
    assert _line(WebSearch, r) == 'web_search "cats": 0 sources'


def test_websearch_query_excluded_from_model_result():
    r = WebSearchResult(answer="", query="cats")
    assert "query" not in r.model_dump()
