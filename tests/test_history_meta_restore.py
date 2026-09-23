"""Resume rebuilds real tool events from the stored message plus its meta.

The call event is rebuilt from the JSON the model sent (already stored on
the assistant message) validated into the live args model. The result event
is rebuilt from meta: outcome, structured result, diff rows, and the two
instants whose difference is the duration. Old sessions (no meta) and tools
this process does not know fall back to the bare widgets.
"""

from __future__ import annotations

from privibe.cli.textual_ui.widgets.tools import finished_hint_text
from privibe.cli.textual_ui.windowing.history import (
    history_tool_call_event,
    history_tool_result_event,
)
from privibe.core.tools.builtins.bash import Bash, BashArgs, BashResult
from privibe.core.tools.ui import ToolUIDataAdapter
from privibe.core.types import (
    FileDiff,
    FunctionCall,
    LLMMessage,
    MessageMeta,
    Role,
    ToolCall,
    ToolOutcome,
)

TOOLS = {"bash": Bash}


def _call(arguments: str = '{"command": "ls -la"}') -> ToolCall:
    return ToolCall(
        id="c1", index=0, function=FunctionCall(name="bash", arguments=arguments)
    )


def _assistant(request_sent_at: float | None = 100.0) -> LLMMessage:
    meta = MessageMeta(request_sent_at=request_sent_at) if request_sent_at else None
    return LLMMessage(role=Role.assistant, content="", tool_calls=[_call()], meta=meta)


def _tool_msg(
    meta: MessageMeta | None, content: str = "command: ls -la\nstdout: x"
) -> LLMMessage:
    return LLMMessage(
        role=Role.tool, name="bash", tool_call_id="c1", content=content, meta=meta
    )


class TestCallEvent:
    def test_rebuilt_from_stored_arguments(self) -> None:
        event = history_tool_call_event(_call(), _assistant(), TOOLS)
        assert event is not None
        assert event.tool_class is Bash
        assert isinstance(event.args, BashArgs)
        assert event.args.command == "ls -la"
        assert event.tool_call_id == "c1"
        assert event.requested_at == 100.0
        # the same adapter the live path uses draws the same line
        summary = ToolUIDataAdapter(Bash).get_call_display(event).summary
        assert "ls -la" in summary

    def test_no_meta_means_no_request_instant(self) -> None:
        event = history_tool_call_event(_call(), _assistant(None), TOOLS)
        assert event is not None
        assert event.requested_at is None

    def test_unknown_tool_is_none(self) -> None:
        assert history_tool_call_event(_call(), _assistant(), {}) is None

    def test_arguments_the_live_model_rejects_leave_args_none(self) -> None:
        event = history_tool_call_event(_call('{"nope": 1}'), _assistant(), TOOLS)
        assert event is not None
        assert event.args is None
        event = history_tool_call_event(_call("not json"), _assistant(), TOOLS)
        assert event is not None
        assert event.args is None


class TestResultEvent:
    def test_success_rebuilds_the_typed_result_and_duration(self) -> None:
        meta = MessageMeta(
            tool_started_at=10.0,
            tool_finished_at=12.5,
            tool_outcome=ToolOutcome.success,
            tool_result={
                "command": "ls -la",
                "stdout": "x",
                "stderr": "",
                "returncode": 0,
            },
            file_diff=FileDiff(path="f", kind="diff", hunks=[[("diff-added", "+")]]),
        )
        event = history_tool_result_event(_tool_msg(meta), "bash", TOOLS)
        assert event is not None
        assert isinstance(event.result, BashResult)
        assert event.result.stdout == "x"
        assert event.error is None and not event.skipped
        assert event.duration == 2.5
        assert event.file_diff == meta.file_diff
        assert event.tool_call_id == "c1"
        display = ToolUIDataAdapter(Bash).get_result_display(event)
        assert display.success

    def test_failure_uses_the_stored_error_text(self) -> None:
        meta = MessageMeta(tool_finished_at=1.0, tool_outcome=ToolOutcome.failure)
        event = history_tool_result_event(
            _tool_msg(meta, content="<error>boom</error>"), "bash", TOOLS
        )
        assert event is not None
        assert event.error == "<error>boom</error>"
        assert event.result is None
        assert event.duration is None

    def test_skipped_uses_the_stored_reason(self) -> None:
        meta = MessageMeta(tool_finished_at=1.0, tool_outcome=ToolOutcome.skipped)
        event = history_tool_result_event(
            _tool_msg(meta, content="nope"), "bash", TOOLS
        )
        assert event is not None
        assert event.skipped and event.skip_reason == "nope"

    def test_result_the_live_model_rejects_is_none_not_a_crash(self) -> None:
        meta = MessageMeta(
            tool_outcome=ToolOutcome.success, tool_result={"garbage": True}
        )
        event = history_tool_result_event(_tool_msg(meta), "bash", TOOLS)
        assert event is not None
        assert event.result is None

    def test_no_meta_or_unknown_tool_is_none(self) -> None:
        assert history_tool_result_event(_tool_msg(None), "bash", TOOLS) is None
        meta = MessageMeta(tool_outcome=ToolOutcome.success, tool_result={})
        assert history_tool_result_event(_tool_msg(meta), "bash", {}) is None


class TestHint:
    def test_hint_is_derived_from_instants(self) -> None:
        assert finished_hint_text(2.5, None) == "(run 2.5s)"
        assert finished_hint_text(None, None) == ""
        text = finished_hint_text(0.04, 1_700_000_000.0)
        assert text.startswith("(run 0.0s · ") and text.endswith(")")
