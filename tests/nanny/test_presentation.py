from __future__ import annotations

from pathlib import Path

from privibe.core.nanny import present_call, present_permissions
from privibe.core.tools.builtins.bash import Bash, BashArgs
from privibe.core.tools.builtins.search_replace import SearchReplaceArgs
from privibe.core.tools.builtins.write_file import WriteFileArgs
from privibe.core.tools.permissions import PermissionScope, RequiredPermission


def test_bash_shows_the_whole_command() -> None:
    shown = present_call(
        "bash", Bash, BashArgs(command="rm -rf build && ls"), write_full_lines=80
    )
    assert shown.text == "bash command:\nrm -rf build && ls"
    assert shown.full_text is None


def test_short_write_is_shown_whole(tmp_path: Path) -> None:
    content = "\n".join(f"line {i}" for i in range(20))
    args = WriteFileArgs(path=str(tmp_path / "new.txt"), content=content)
    shown = present_call("write_file", None, args, write_full_lines=80)
    assert "does not exist yet" in shown.text
    assert "20 lines" in shown.text
    assert content in shown.text
    assert shown.full_text is None


def test_long_write_is_summarised_with_full_text_kept(tmp_path: Path) -> None:
    target = tmp_path / "big.txt"
    target.write_text("old")
    content = "\n".join(f"line {i}" for i in range(200))
    args = WriteFileArgs(path=str(target), content=content, overwrite=True)
    shown = present_call("write_file", None, args, write_full_lines=80)
    assert "exists, 3 bytes" in shown.text
    assert "overwrite=yes" in shown.text
    assert "line 0" in shown.text and "line 199" in shown.text
    assert "lines omitted" in shown.text
    assert "line 50" not in shown.text
    assert shown.full_text == content


def test_search_replace_block_is_shown_whole_when_small(tmp_path: Path) -> None:
    block = "<<<<<<< SEARCH\nfoo\n=======\nbar\n>>>>>>> REPLACE"
    args = SearchReplaceArgs(file_path=str(tmp_path / "a.py"), content=block)
    shown = present_call("search_replace", None, args, write_full_lines=80)
    assert block in shown.text
    assert shown.full_text is None


def test_generic_tool_falls_back_to_args_json() -> None:
    from pydantic import BaseModel

    class Args(BaseModel):
        query: str

    shown = present_call("mystery", None, Args(query="x"), write_full_lines=80)
    assert "mystery" in shown.text and '"query":"x"' in shown.text


def test_permissions_line() -> None:
    rp = RequiredPermission(
        scope=PermissionScope.OUTSIDE_DIRECTORY,
        invocation_pattern="/etc/*",
        session_pattern="/etc/*",
        label="access to /etc",
    )
    assert present_permissions([rp]) == "Permissions this call needs: access to /etc\n"
    assert present_permissions(None) == ""
