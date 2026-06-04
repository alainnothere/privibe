from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from privibe.core.tools.base import BaseToolState
from privibe.core.tools.builtins.find_symbol import (
    MAX_BODY_LINES,
    CONTEXT_BEFORE,
    FindSymbol,
    FindSymbolArgs,
    FindSymbolConfig,
    _build_search_pattern,
    _deduplicate,
    _extract_brace_body,
    _extract_indent_body,
    _format_match,
    _infer_extensions,
    _parse_rg_json,
)

# ---------------------------------------------------------------------------
# _parse_rg_json
# ---------------------------------------------------------------------------


def test_parse_rg_json_extracts_matches():
    lines = [
        json.dumps({"type": "begin", "data": {}}),
        json.dumps({"type": "match", "data": {"path": {"text": "foo.py"}, "line_number": 5}}),
        json.dumps({"type": "match", "data": {"path": {"text": "bar.py"}, "line_number": 10}}),
        json.dumps({"type": "summary", "data": {}}),
    ]
    result = _parse_rg_json("\n".join(lines))
    assert result == [{"file": "foo.py", "line": 5}, {"file": "bar.py", "line": 10}]


def test_parse_rg_json_ignores_non_match_types():
    lines = [
        json.dumps({"type": "begin", "data": {}}),
        json.dumps({"type": "context", "data": {"path": {"text": "x.py"}, "line_number": 1}}),
    ]
    assert _parse_rg_json("\n".join(lines)) == []


def test_parse_rg_json_tolerates_malformed_lines():
    stdout = "not json\n" + json.dumps(
        {"type": "match", "data": {"path": {"text": "ok.py"}, "line_number": 3}}
    )
    result = _parse_rg_json(stdout)
    assert result == [{"file": "ok.py", "line": 3}]


def test_parse_rg_json_empty_string():
    assert _parse_rg_json("") == []


# ---------------------------------------------------------------------------
# _deduplicate
# ---------------------------------------------------------------------------


def test_deduplicate_keeps_distant_matches_same_file():
    matches = [
        {"file": "a.py", "line": 1},
        {"file": "a.py", "line": 50},
    ]
    result = _deduplicate(matches)
    assert len(result) == 2


def test_deduplicate_drops_adjacent_matches_same_file():
    matches = [
        {"file": "a.py", "line": 10},
        {"file": "a.py", "line": 12},  # within 5 lines
    ]
    result = _deduplicate(matches)
    assert len(result) == 1
    assert result[0]["line"] == 10


def test_deduplicate_keeps_separate_files():
    matches = [
        {"file": "a.py", "line": 5},
        {"file": "b.py", "line": 5},
    ]
    result = _deduplicate(matches)
    assert len(result) == 2


def test_deduplicate_boundary_at_exactly_5():
    # distance of 5 is NOT > 5, so the second match is deduplicated
    matches = [
        {"file": "a.py", "line": 10},
        {"file": "a.py", "line": 15},
    ]
    assert len(_deduplicate(matches)) == 1

    # distance of 6 passes the threshold
    matches2 = [
        {"file": "a.py", "line": 10},
        {"file": "a.py", "line": 16},
    ]
    assert len(_deduplicate(matches2)) == 2


# ---------------------------------------------------------------------------
# _build_search_pattern
# ---------------------------------------------------------------------------


def test_find_symbol_args_strips_kind_quotes():
    args = FindSymbolArgs(symbol="Foo", kind='"class"')
    assert args.kind == "class"

    args2 = FindSymbolArgs(symbol="Foo", kind="'function'")
    assert args2.kind == "function"


def test_build_search_pattern_no_kind_is_word_boundary():
    pattern = _build_search_pattern("DoSomething", None, ["py"])
    assert pattern == r"\bDoSomething\b"


def test_build_search_pattern_python_function():
    pattern = _build_search_pattern("my_func", "function", ["py"])
    assert "def" in pattern
    assert "my_func" in pattern


def test_build_search_pattern_csharp_class():
    pattern = _build_search_pattern("MyClass", "class", ["cs"])
    assert "class" in pattern
    assert "MyClass" in pattern


def test_build_search_pattern_regex_passthrough():
    pattern = _build_search_pattern("I.*Service", "interface", ["cs"])
    assert "I.*Service" in pattern


def test_build_search_pattern_unknown_extension_falls_back():
    pattern = _build_search_pattern("Foo", "function", ["xyz"])
    assert pattern == r"\bFoo\b"


def test_build_search_pattern_multiple_extensions_alternates():
    pattern = _build_search_pattern("Foo", "class", ["cs", "java"])
    # Both C# and Java class patterns should be present
    assert "class" in pattern
    assert "Foo" in pattern


# ---------------------------------------------------------------------------
# _infer_extensions
# ---------------------------------------------------------------------------


def test_infer_extensions_explicit_override():
    result = _infer_extensions("src/", ["cs", "ts"])
    assert result == ["cs", "ts"]


def test_infer_extensions_from_glob():
    result = _infer_extensions("src/**/*.cs", None)
    assert result == ["cs"]


def test_infer_extensions_unknown_extension_returns_all():
    result = _infer_extensions("src/", None)
    assert "py" in result
    assert "cs" in result
    assert "ts" in result


def test_infer_extensions_from_file(tmp_path):
    f = tmp_path / "foo.go"
    f.write_text("package main")
    result = _infer_extensions(str(f), None)
    assert result == ["go"]


# ---------------------------------------------------------------------------
# _extract_brace_body
# ---------------------------------------------------------------------------


def _to_lines(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines


def test_extract_brace_body_simple_function():
    src = _to_lines(
        "void Foo()\n"
        "{\n"
        "    return;\n"
        "}\n"
    )
    end_idx, truncated = _extract_brace_body(src, 0)
    assert end_idx == 3
    assert not truncated


def test_extract_brace_body_nested_braces():
    src = _to_lines(
        "void Foo() {\n"
        "    if (x) {\n"
        "        y();\n"
        "    }\n"
        "}\n"
    )
    end_idx, truncated = _extract_brace_body(src, 0)
    assert end_idx == 4
    assert not truncated


def test_extract_brace_body_truncates_at_limit(tmp_path):
    # Function with more lines than MAX_BODY_LINES
    body = ["void Big() {\n"] + ["    x();\n"] * (MAX_BODY_LINES + 10) + ["}\n"]
    end_idx, truncated = _extract_brace_body(body, 0)
    assert truncated
    assert end_idx < len(body) - 1


# ---------------------------------------------------------------------------
# _extract_indent_body
# ---------------------------------------------------------------------------


def test_extract_indent_body_simple_function():
    src = _to_lines(
        "def foo():\n"
        "    x = 1\n"
        "    return x\n"
        "\n"
        "def bar():\n"
        "    pass\n"
    )
    end_idx, truncated = _extract_indent_body(src, 0)
    assert end_idx < 4  # stops before 'def bar'
    assert not truncated


def test_extract_indent_body_skips_blank_lines():
    src = _to_lines(
        "def foo():\n"
        "    x = 1\n"
        "\n"
        "    return x\n"
        "\n"
        "class Bar:\n"
    )
    end_idx, truncated = _extract_indent_body(src, 0)
    assert end_idx < 5  # stops before 'class Bar'
    assert not truncated


def test_extract_indent_body_multiline_signature():
    src = _to_lines(
        "def foo(\n"
        "    arg1,\n"
        "    arg2,\n"
        "):\n"
        "    return arg1\n"
        "\n"
        "def bar():\n"
        "    pass\n"
    )
    end_idx, truncated = _extract_indent_body(src, 0)
    # Body ends somewhere before 'def bar' (line index 6)
    assert end_idx < 6
    # The body content up to end_idx must include 'return arg1' (line index 4)
    assert any("return arg1" in src[i] for i in range(4, end_idx + 1))
    assert not truncated


# ---------------------------------------------------------------------------
# _format_match integration (no subprocess)
# ---------------------------------------------------------------------------


def test_format_match_python_function(tmp_path):
    f = tmp_path / "foo.py"
    f.write_text(
        "import os\n"
        "\n"
        "def my_func(x):\n"
        "    return x + 1\n"
        "\n"
        "def other():\n"
        "    pass\n"
    )
    result = _format_match(str(f), 3, "py")
    assert "=== " in result
    assert ":3 ===" in result
    assert "my_func" in result
    assert "return x + 1" in result
    assert "other" not in result


def test_format_match_includes_context_before(tmp_path):
    lines = [f"line{i}\n" for i in range(1, 20)]
    f = tmp_path / "foo.py"
    f.write_text("".join(lines))
    result = _format_match(str(f), 10, "py")
    # Should include up to 5 lines before line 10
    assert "line5" in result or "line6" in result


def test_format_match_csharp_method(tmp_path):
    f = tmp_path / "Foo.cs"
    f.write_text(
        "public class Foo\n"
        "{\n"
        "    public int Bar(int x)\n"
        "    {\n"
        "        return x * 2;\n"
        "    }\n"
        "}\n"
    )
    result = _format_match(str(f), 3, "cs")
    assert "Bar" in result
    assert "return x * 2" in result


def test_format_match_missing_file():
    result = _format_match("/nonexistent/path/foo.py", 1, "py")
    assert "could not read file" in result


def test_format_match_line_out_of_range(tmp_path):
    f = tmp_path / "tiny.py"
    f.write_text("x = 1\n")
    result = _format_match(str(f), 999, "py")
    assert "out of range" in result


def test_format_match_truncation_note(tmp_path):
    # Large function body
    body = "def big():\n" + "    x = 1\n" * (MAX_BODY_LINES + 5)
    f = tmp_path / "big.py"
    f.write_text(body)
    result = _format_match(str(f), 1, "py")
    assert "truncated" in result


# ---------------------------------------------------------------------------
# FindSymbol.run integration (requires ripgrep)
# ---------------------------------------------------------------------------

pytestmark_rg = pytest.mark.skipif(
    not shutil.which("rg"), reason="ripgrep not installed"
)


@pytest.fixture
def python_project(tmp_path):
    (tmp_path / "a.py").write_text(
        "def my_func(x):\n"
        "    return x + 1\n"
        "\n"
        "def other_func():\n"
        "    return 0\n"
    )
    (tmp_path / "b.py").write_text(
        "class MyClass:\n"
        "    def my_func(self):\n"
        "        pass\n"
    )
    return tmp_path


@pytest.fixture
def cs_project(tmp_path):
    (tmp_path / "IFoo.cs").write_text(
        "public interface IFoo\n"
        "{\n"
        "    void DoSomething();\n"
        "}\n"
    )
    (tmp_path / "FooImpl.cs").write_text(
        "public class FooImpl : IFoo\n"
        "{\n"
        "    public void DoSomething()\n"
        "    {\n"
        "        Console.WriteLine(\"done\");\n"
        "    }\n"
        "}\n"
    )
    return tmp_path


async def _run_find(args_dict):
    tool = FindSymbol(config=FindSymbolConfig(), state=BaseToolState())
    args = FindSymbolArgs(**args_dict)
    results = [r async for r in tool.run(args)]
    assert len(results) == 1
    return results[0]


@pytestmark_rg
@pytest.mark.asyncio
async def test_find_symbol_python_function(python_project):
    result = await _run_find({
        "symbol": "my_func",
        "path": str(python_project),
        "kind": "function",
        "extensions": ["py"],
    })
    assert result.total_found >= 1
    assert "my_func" in result.output


@pytestmark_rg
@pytest.mark.asyncio
async def test_find_symbol_returns_all_implementations(cs_project):
    result = await _run_find({
        "symbol": "DoSomething",
        "path": str(cs_project),
        "extensions": ["cs"],
    })
    assert result.total_found >= 2
    assert "IFoo.cs" in result.output or "FooImpl.cs" in result.output


@pytestmark_rg
@pytest.mark.asyncio
async def test_find_symbol_interface_pattern(cs_project):
    result = await _run_find({
        "symbol": "IFoo",
        "path": str(cs_project),
        "kind": "interface",
        "extensions": ["cs"],
    })
    assert result.total_found >= 1
    assert "IFoo" in result.output


@pytestmark_rg
@pytest.mark.asyncio
async def test_find_symbol_no_match(python_project):
    result = await _run_find({
        "symbol": "nonexistent_symbol_xyz",
        "path": str(python_project),
    })
    assert result.total_found == 0
    assert "No matches" in result.output


@pytestmark_rg
@pytest.mark.asyncio
async def test_find_symbol_regex_pattern(python_project):
    result = await _run_find({
        "symbol": r"\w+_func",
        "path": str(python_project),
        "kind": "function",
        "extensions": ["py"],
    })
    assert result.total_found >= 2  # my_func and other_func


@pytestmark_rg
@pytest.mark.asyncio
async def test_find_symbol_output_has_hashed_lines(python_project):
    result = await _run_find({
        "symbol": "my_func",
        "path": str(python_project),
        "kind": "function",
        "extensions": ["py"],
    })
    lines = result.output.splitlines()
    content_lines = [l for l in lines if not l.startswith("===") and l.strip()]
    assert any(len(parts := l.split()) >= 2 and len(parts[1]) == 4 for l in content_lines)
