"""The tool call line folds open like the thinking block: the call's
parameters (ToolCallDisplay.content) and every stream message it has shown,
not only the latest one. Orchestrate was the first tool whose progress was
worth reading back.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.css.stylesheet import Stylesheet

from privibe.cli.textual_ui.widgets.tools import ToolCallMessage
from privibe.core.tools.builtins.bash import Bash, BashArgs
from privibe.core.tools.builtins.orchestrate import Orchestrate, OrchestrateArgs
from privibe.core.tools.ui import ToolUIDataAdapter
from privibe.core.types import ToolCallEvent

_TCSS = Path(__file__).parents[3] / "privibe" / "cli" / "textual_ui" / "app.tcss"


def _orchestrate_event() -> ToolCallEvent:
    return ToolCallEvent(
        tool_call_id="call-1",
        tool_name="orchestrate",
        tool_class=Orchestrate,
        args=OrchestrateArgs(
            goal="Make f return 2.\nSecond line of the goal.",
            files=["/src/a.py"],
            scope=["/src"],
            done_command="pytest -q",
        ),
    )


class _Host(App[None]):
    def __init__(self, widget: ToolCallMessage) -> None:
        super().__init__()
        self.widget = widget

    def compose(self) -> ComposeResult:
        yield self.widget


@pytest.mark.asyncio
async def test_params_on_the_line_and_full_stream_log_in_the_body() -> None:
    widget = ToolCallMessage(_orchestrate_event())
    async with _Host(widget).run_test():
        line = str(widget._text_widget.render())
        assert line.startswith('orchestrate "Make f return 2."\n')
        assert "orchestrate orchestrate" not in line
        assert "scope: /src" in line
        assert "files: /src/a.py" in line
        assert "`pytest -q` exits 0" in line
        assert widget.collapsed
        # No ToolCallDisplay.content and no stream yet: nothing to unfold.
        assert widget._body is not None and not widget._body.display
        assert widget._triangle_widget is not None
        assert not widget._triangle_widget.display

        widget.set_stream_message("manager: planning")
        widget.set_stream_message("brainstorm")
        widget.set_stream_message("round 1/4: worker on task 1: change f")
        assert widget.stream_log == [
            "manager: planning",
            "brainstorm",
            "round 1/4: worker on task 1: change f",
        ]
        latest = str(widget._stream_widget.render())
        assert "worker on task 1" in latest and "planning" not in latest
        assert "planning" in str(widget._log_widget.render())

        await widget.set_collapsed(False)
        assert widget._body.display
        assert str(widget._triangle_widget.render()) == "▼"
        await widget.set_collapsed(True)
        assert not widget._body.display


@pytest.mark.asyncio
async def test_no_body_without_params_or_stream() -> None:
    event = ToolCallEvent(
        tool_call_id="call-2",
        tool_name="bash",
        tool_class=Bash,
        args=BashArgs(command="ls"),
    )
    widget = ToolCallMessage(event)
    async with _Host(widget).run_test():
        assert not widget._triangle_widget.display
        await widget.set_collapsed(False)
        assert not widget._body.display
        widget.set_stream_message("running")
        assert widget._triangle_widget.display
        assert widget._body.display


def test_orchestrate_summary_carries_the_goal() -> None:
    display = Orchestrate.format_call_display(
        OrchestrateArgs(goal="\n  Make f return 2.\nmore", max_rounds=2)
    )
    first, *rest = display.summary.splitlines()
    assert first == 'orchestrate "Make f return 2."'
    assert rest == [
        "  scope: (none)",
        "  files: (none)",
        "  done when: the manager says so (unverified)",
        "  max_rounds: 2",
    ]
    assert display.content is None
    long = Orchestrate.format_call_display(OrchestrateArgs(goal="x" * 200))
    assert long.summary.splitlines()[0] == 'orchestrate "' + "x" * 79 + '…"'


def test_adapter_does_not_double_the_tool_name() -> None:
    """The adapter prefixes the tool name unless the summary already starts
    with "<name> ". An "orchestrate: goal" summary once came out as
    "orchestrate orchestrate: goal" on the call line.
    """
    summary = (
        ToolUIDataAdapter(Orchestrate).get_call_display(_orchestrate_event()).summary
    )
    assert summary.startswith('orchestrate "Make f return 2."')
    assert not summary.startswith("orchestrate orchestrate")


def test_tool_call_rules_are_top_level_in_app_tcss() -> None:
    """A partial stage once landed these rules inside `.tool-result {`,
    where they style nothing a ToolCallMessage owns.
    """
    sheet = Stylesheet()
    sheet.add_source(_TCSS.read_text(), read_from=(str(_TCSS), ""))
    sheet.parse()
    top_level = {rule.selector_set[0].css for rule in sheet.rules if rule.selector_set}
    for selector in (
        ".tool-call-triangle",
        ".tool-call-body",
        ".tool-call-params",
        ".tool-call-log",
        ".tool-result",
    ):
        assert selector in top_level, f"{selector} is nested or missing"
