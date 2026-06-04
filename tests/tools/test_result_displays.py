from __future__ import annotations

import os

from privibe.core.tools.builtins.grep import Grep, GrepResult
from privibe.core.tools.builtins.hashed_read import HashedRead, HashedReadResult
from privibe.core.tools.builtins.read_file import ReadFile, ReadFileResult
from privibe.core.tools.builtins.search_replace import SearchReplace, SearchReplaceResult
from privibe.core.tools.builtins.websearch import WebSearch, WebSearchResult
from privibe.core.tools.builtins.write_file import WriteFile, WriteFileResult
from privibe.core.tools.utils import display_path
from privibe.core.types import ToolResultEvent


def _evt(tool_cls, result) -> ToolResultEvent:
    return ToolResultEvent(
        tool_name=tool_cls.get_name(),
        tool_class=tool_cls,
        result=result,
        tool_call_id="t",
    )


# ---------------------------------------------------------------------------
# display_path
# ---------------------------------------------------------------------------


def test_display_path_relative_under_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sub").mkdir()
    p = tmp_path / "sub" / "f.py"
    assert display_path(str(p)) == os.path.join("sub", "f.py")


def test_display_path_keeps_absolute_when_outside_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert display_path("/etc/hosts") == "/etc/hosts"


# ---------------------------------------------------------------------------
# grep — pattern + path, with the dot-path omission and model exclusion
# ---------------------------------------------------------------------------


def test_grep_display_includes_pattern_and_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = GrepResult(
        matches="", match_count=20, was_truncated=True, pattern="bla", path="src"
    )
    d = Grep.get_result_display(_evt(Grep, r))
    assert d.message == 'Found 20 matches for "bla" in src (truncated)'


def test_grep_display_omits_dot_path():
    r = GrepResult(matches="", match_count=3, was_truncated=False, pattern="x", path=".")
    d = Grep.get_result_display(_evt(Grep, r))
    assert d.message == 'Found 3 matches for "x"'


def test_grep_display_fields_excluded_from_model_result():
    r = GrepResult(matches="", match_count=1, was_truncated=False, pattern="x", path="y")
    dumped = r.model_dump()
    assert "pattern" not in dumped
    assert "path" not in dumped


# ---------------------------------------------------------------------------
# file tools — now name the file via display_path
# ---------------------------------------------------------------------------


def test_hashed_read_display_includes_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = HashedReadResult(
        path=str(tmp_path / "a.py"),
        content="",
        start_line=1,
        lines_read=20,
        was_truncated=False,
    )
    d = HashedRead.get_result_display(_evt(HashedRead, r))
    assert d.message == "Read 20 lines from a.py (hashed)"


def test_read_file_display_includes_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pkg").mkdir()
    r = ReadFileResult(
        path=str(tmp_path / "pkg" / "a.py"),
        content="",
        lines_read=5,
        was_truncated=False,
    )
    d = ReadFile.get_result_display(_evt(ReadFile, r))
    assert f"from {os.path.join('pkg', 'a.py')}" in d.message


def test_write_file_display_includes_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = WriteFileResult(
        path=str(tmp_path / "a.py"),
        bytes_written=3,
        file_existed=False,
        content="abc",
    )
    d = WriteFile.get_result_display(_evt(WriteFile, r))
    assert d.message == "Created a.py"


def test_search_replace_display_includes_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = SearchReplaceResult(
        file=str(tmp_path / "a.py"), blocks_applied=2, lines_changed=1, content=""
    )
    d = SearchReplace.get_result_display(_evt(SearchReplace, r))
    assert d.message == "Applied 2 blocks in a.py"


# ---------------------------------------------------------------------------
# websearch — query echoed, excluded from model result
# ---------------------------------------------------------------------------


def test_websearch_display_includes_query():
    r = WebSearchResult(answer="", sources=[], query="cats")
    d = WebSearch.get_result_display(_evt(WebSearch, r))
    assert d.message == '0 sources found for "cats"'


def test_websearch_query_excluded_from_model_result():
    r = WebSearchResult(answer="", query="cats")
    assert "query" not in r.model_dump()
