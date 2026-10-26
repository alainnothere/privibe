"""Resumed history draws the nanny's stored decision after the call it judged."""

from __future__ import annotations

from weakref import WeakKeyDictionary

from privibe.cli.textual_ui.widgets.messages import NannyMessage
from privibe.cli.textual_ui.widgets.tools import ToolCallMessage, ToolResultMessage
from privibe.cli.textual_ui.windowing.history import build_history_widgets
from privibe.core.nanny import NannyDecision, TalkExchange
from privibe.core.tools.builtins.bash import Bash
from privibe.core.types import FunctionCall, LLMMessage, Role, ToolCall


def call(call_id: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=0,
        function=FunctionCall(name="bash", arguments='{"command": "ls"}'),
    )


def decision(call_id: str, verdict: str, **extra) -> NannyDecision:
    return NannyDecision(
        tool_call_id=call_id,
        tool_name="bash",
        agent="default",
        verdict=verdict,
        reason="lists files",
        ask_index=3,
        verdict_index=4,
        header_session_id="s",
        main_session_id="s",
        **extra,
    )


def test_entry_is_placed_between_call_and_result() -> None:
    batch = [
        LLMMessage(role=Role.assistant, content="", tool_calls=[call("a")]),
        LLMMessage(role=Role.tool, tool_call_id="a", name="bash", content="ok"),
        LLMMessage(role=Role.assistant, content="", tool_calls=[call("b")]),
        LLMMessage(role=Role.tool, tool_call_id="b", name="bash", content="ok"),
    ]
    refused = decision(
        "b",
        "refuse",
        human_decision="allow",
        talk=[TalkExchange(human="why?", nanny="because", ask_index=5, reply_index=6)],
    )
    widgets = build_history_widgets(
        batch,
        {},
        start_index=0,
        tools_collapsed=True,
        history_widget_indices=WeakKeyDictionary(),
        tool_classes={"bash": Bash},
        nanny_decisions={"a": decision("a", "approve"), "b": refused},
    )
    kinds = [type(w) for w in widgets]
    assert kinds == [
        ToolCallMessage,
        NannyMessage,
        ToolResultMessage,
        ToolCallMessage,
        NannyMessage,
        ToolResultMessage,
    ]
    approved, refused_widget = (w for w in widgets if isinstance(w, NannyMessage))
    assert approved.has_class("nanny-approved")
    assert NannyMessage.format_lines(approved.decision) == [
        "nanny approved: lists files"
    ]
    assert NannyMessage.format_lines(refused_widget.decision) == [
        "nanny refused: lists files",
        "you: why?",
        "nanny: because",
        "you allowed it once",
    ]


def test_no_decisions_means_no_entries() -> None:
    batch = [
        LLMMessage(role=Role.assistant, content="", tool_calls=[call("a")]),
        LLMMessage(role=Role.tool, tool_call_id="a", name="bash", content="ok"),
    ]
    widgets = build_history_widgets(
        batch,
        {},
        start_index=0,
        tools_collapsed=True,
        history_widget_indices=WeakKeyDictionary(),
        tool_classes={"bash": Bash},
    )
    assert not any(isinstance(w, NannyMessage) for w in widgets)
