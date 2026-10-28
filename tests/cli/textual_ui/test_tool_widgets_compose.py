"""Every registered tool widget must compose against its real args/result model.

Regression guard for the read_file approval card that kept asking for
``offset`` after the arg model had been renamed to ``start_line``: the widget
blew up inside ``compose``, which killed the approval future and took the
tool loop down with it. Nothing exercised the typed widgets, so nothing
noticed for four commits.
"""

from __future__ import annotations

from pydantic import BaseModel
import pytest
from textual.app import App, ComposeResult
from textual.widget import Widget

from privibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from privibe.cli.textual_ui.widgets.tool_widgets import (
    APPROVAL_WIDGETS,
    RESULT_WIDGETS,
    ReadApprovalWidget,
    get_approval_widget,
    get_result_widget,
)
from privibe.core.tools.builtins.ask_user_question import Answer, AskUserQuestionResult
from privibe.core.tools.builtins.bash import BashArgs, BashResult
from privibe.core.tools.builtins.grep import GrepArgs, GrepResult
from privibe.core.tools.builtins.hashed_read import HashedReadArgs
from privibe.core.tools.builtins.orchestrate import OrchestrateArgs
from privibe.core.tools.builtins.read_file import ReadFileArgs, ReadFileResult
from privibe.core.tools.builtins.search_replace import (
    SearchReplaceArgs,
    SearchReplaceResult,
)
from privibe.core.tools.builtins.todo import TodoArgs, TodoItem, TodoResult
from privibe.core.tools.builtins.write_file import WriteFileArgs, WriteFileResult

# One real, fully-populated args instance per registered approval widget.
APPROVAL_ARGS: dict[str, BaseModel] = {
    "bash": BashArgs(command="ls -la"),
    "read_file": ReadFileArgs(path="/src/a.py", start_line=40, limit=20),
    "hashed_read": HashedReadArgs(path="/src/a.py", start_line=40, limit=20),
    "write_file": WriteFileArgs(path="/src/a.py", content="x = 1\n"),
    "search_replace": SearchReplaceArgs(
        file_path="/src/a.py",
        content="<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE\n",
    ),
    "grep": GrepArgs(pattern="TODO", path="src"),
    "todo": TodoArgs(action="write", todos=[TodoItem(id="1", content="do it")]),
    "orchestrate": OrchestrateArgs(
        goal="Make f return 2.\n\nKeep the tests green.",
        files=["/src/a.py"],
        scope=["/src"],
        done_command="pytest -q",
        max_rounds=3,
    ),
}

# One real result instance per registered result widget.
RESULTS: dict[str, BaseModel] = {
    "bash": BashResult(command="ls", stdout="a\nb\n", stderr="", returncode=0),
    "read_file": ReadFileResult(
        path="/src/a.py", content="x = 1\n", lines_read=1, was_truncated=False
    ),
    "write_file": WriteFileResult(
        path="/src/a.py", bytes_written=6, file_existed=False, content="x = 1\n"
    ),
    "search_replace": SearchReplaceResult(
        file="/src/a.py", blocks_applied=1, lines_changed=1, content="x = 2\n"
    ),
    "grep": GrepResult(matches="a.py:1:TODO", match_count=1, was_truncated=False),
    "todo": TodoResult(
        message="ok", todos=[TodoItem(id="1", content="do it")], total_count=1
    ),
    "ask_user_question": AskUserQuestionResult(
        answers=[Answer(question="Which?", answer="This one")]
    ),
}


class _Host(App):
    """Bare app that mounts one widget so its compose() actually runs."""

    def __init__(self, widget: Widget) -> None:
        super().__init__()
        self._widget = widget

    def compose(self) -> ComposeResult:
        yield self._widget


async def _mount(widget: Widget) -> list[str]:
    """Mount the widget and return the text of its NoMarkupStatic lines."""
    app = _Host(widget)
    async with app.run_test():
        assert widget.is_mounted
        return [str(w.content) for w in widget.query(NoMarkupStatic)]


def test_every_registered_approval_widget_has_sample_args() -> None:
    assert set(APPROVAL_WIDGETS) == set(APPROVAL_ARGS)


def test_every_registered_result_widget_has_sample_result() -> None:
    assert set(RESULT_WIDGETS) == set(RESULTS)


@pytest.mark.parametrize("tool_name", sorted(APPROVAL_WIDGETS))
@pytest.mark.asyncio
async def test_approval_widget_composes(tool_name: str) -> None:
    await _mount(get_approval_widget(tool_name, APPROVAL_ARGS[tool_name]))


@pytest.mark.parametrize("tool_name", sorted(RESULT_WIDGETS))
@pytest.mark.parametrize("collapsed", [True, False])
@pytest.mark.asyncio
async def test_result_widget_composes(tool_name: str, collapsed: bool) -> None:
    await _mount(
        get_result_widget(
            tool_name, RESULTS[tool_name], True, "done", collapsed=collapsed
        )
    )


def test_both_read_tools_share_one_approval_widget() -> None:
    assert APPROVAL_WIDGETS["read_file"] is ReadApprovalWidget
    assert APPROVAL_WIDGETS["hashed_read"] is ReadApprovalWidget


@pytest.mark.parametrize("args_cls", [ReadFileArgs, HashedReadArgs])
@pytest.mark.asyncio
async def test_read_approval_hides_defaults(
    args_cls: type[ReadFileArgs | HashedReadArgs],
) -> None:
    lines = await _mount(ReadApprovalWidget(args_cls(path="/src/a.py")))
    assert lines == ["path: /src/a.py"]


@pytest.mark.parametrize("args_cls", [ReadFileArgs, HashedReadArgs])
@pytest.mark.asyncio
async def test_read_approval_shows_range(
    args_cls: type[ReadFileArgs | HashedReadArgs],
) -> None:
    lines = await _mount(
        ReadApprovalWidget(args_cls(path="/src/a.py", start_line=40, limit=20))
    )
    assert lines == ["path: /src/a.py", "start_line: 40", "limit: 20"]
