"""The tool call line folds open like the thinking block: the call's
parameters (ToolCallDisplay.content) and every stream message it has shown,
not only the latest one. Orchestrate was the first tool whose progress was
worth reading back.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from privibe.cli.textual_ui.widgets.tools import ToolCallMessage
from privibe.core.tools.builtins.bash import Bash, BashArgs
from privibe.core.tools.builtins.orchestrate import Orchestrate, OrchestrateArgs
from privibe.core.types import ToolCallEvent


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
async def test_body_holds_params_and_full_stream_log() -> None:
    widget = ToolCallMessage(_orchestrate_event())
    async with _Host(widget).run_test():
        assert widget.collapsed
        assert widget._body is not None and not widget._body.display
        assert widget._triangle_widget is not None and widget._triangle_widget.display
        params = str(widget._params_widget.render())
        assert "Second line of the goal." in params
        assert "scope: /src" in params
        assert "`pytest -q` exits 0" in params

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
        OrchestrateArgs(goal="\n  Make f return 2.\nmore")
    )
    assert display.summary == "orchestrate: Make f return 2."
    long = Orchestrate.format_call_display(OrchestrateArgs(goal="x" * 200))
    assert long.summary == "orchestrate: " + "x" * 79 + "…"
