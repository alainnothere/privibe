"""A sub-agent's calls are rows of the chat, indented under the call that
ran it, live and after resume.

Live: the tool yields the sub-agent's ToolCallEvent / ToolResultEvent stamped
with parent_call_id and depth, plus TraceLineEvent rows for headers; the
event handler mounts them after the parent call, in order, and updates a
header in place when its stamped version arrives. Resume: the stored result
names the sub-agent sessions (task: session_dir; orchestrate: the ledger's
calls.json) and the history builder loads them and builds the same rows one
level deeper.
"""

from __future__ import annotations

import json
from pathlib import Path
from weakref import WeakKeyDictionary

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widget import Widget

from privibe.cli.textual_ui.handlers.event_handler import EventHandler
from privibe.cli.textual_ui.widgets.tools import ToolCallMessage, TraceLineMessage
from privibe.cli.textual_ui.windowing.history import (
    build_history_widgets,
    build_tool_call_map,
    subagent_children,
)
from privibe.core.tools.base import BaseTool
from privibe.core.tools.builtins.bash import Bash, BashArgs, BashResult
from privibe.core.tools.builtins.orchestrate import (
    Orchestrate,
    OrchestrateArgs,
    OrchestrateResult,
)
from privibe.core.tools.builtins.task import Task, TaskArgs, TaskResult
from privibe.core.types import (
    BaseEvent,
    FunctionCall,
    LLMMessage,
    MessageMeta,
    Role,
    ToolCall,
    ToolCallEvent,
    ToolOutcome,
    ToolResultEvent,
    TraceLineEvent,
)
from privibe.core.utils.tags import is_user_cancellation_event


class _Host(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.area = Vertical(id="messages")

    def compose(self) -> ComposeResult:
        yield self.area

    async def mount_message(self, widget: Widget, after: Widget | None = None) -> None:
        if after is not None and after.parent is self.area:
            await self.area.mount(widget, after=after)
        else:
            await self.area.mount(widget)


def _parent_call() -> ToolCallEvent:
    return ToolCallEvent(
        tool_call_id="orch-1",
        tool_name="orchestrate",
        tool_class=Orchestrate,
        args=OrchestrateArgs(goal="make f return 2", scope=["/r"]),
    )


def _parent_result() -> ToolResultEvent:
    return ToolResultEvent(
        tool_call_id="orch-1",
        tool_name="orchestrate",
        tool_class=Orchestrate,
        result=OrchestrateResult(status="done", rounds=1, calls=4, ledger_dir="/l"),
    )


def _bash(call_id: str, command: str, *, parent: str, depth: int = 1):
    call = ToolCallEvent(
        tool_call_id=call_id,
        tool_name="bash",
        tool_class=Bash,
        args=BashArgs(command=command),
        parent_call_id=parent,
        depth=depth,
    )
    result = ToolResultEvent(
        tool_call_id=call_id,
        tool_name="bash",
        tool_class=Bash,
        result=BashResult(command=command, stdout="", stderr="", returncode=0),
        parent_call_id=parent,
        depth=depth,
    )
    return call, result


def _rows(app: _Host) -> list[Widget]:
    return list(app.area.children)


@pytest.mark.asyncio
async def test_live_children_land_under_the_parent_in_order_and_indented() -> None:
    app = _Host()
    async with app.run_test() as pilot:
        handler = EventHandler(app.mount_message, lambda: True)
        await handler.handle_event(_parent_call())
        header = TraceLineEvent(
            tool_call_id="orch-1",
            message="── worker · round 1/1 ──",
            entry_id="orch-1:1",
        )
        await handler.handle_event(header)
        call_a, result_a = _bash("call-1", "ls", parent="orch-1")
        call_b, result_b = _bash("call-2", "pwd", parent="orch-1")
        await handler.handle_event(call_a)
        await handler.handle_event(result_a)
        await handler.handle_event(call_b)
        await handler.handle_event(result_b)
        await handler.handle_event(
            header.model_copy(
                update={"message": "── worker · round 1/1 ── 3s · 2 calls"}
            )
        )
        await handler.handle_event(_parent_result())
        await pilot.pause()

        rows = _rows(app)
        kinds = [type(r).__name__ for r in rows]
        assert kinds == [
            "ToolCallMessage",  # orchestrate
            "TraceLineMessage",  # header, one widget, updated in place
            "ToolCallMessage",  # bash ls
            "ToolResultMessage",
            "ToolCallMessage",  # bash pwd
            "ToolResultMessage",
            "ToolResultMessage",  # orchestrate's own result, after every child
        ]
        header_row = rows[1]
        assert isinstance(header_row, TraceLineMessage)
        assert header_row.text == "── worker · round 1/1 ── 3s · 2 calls"
        for row in rows[1:6]:
            assert row.has_class("depth-1"), row
        assert not rows[0].has_class("depth-1") and not rows[6].has_class("depth-1")
        # The nested call lines are the same lines the top level prints.
        assert isinstance(rows[2], ToolCallMessage)
        assert rows[2].get_content() == "bash $ ls"
        # Registry is clean once the parent finished.
        assert handler.tool_calls == {} and handler.last_child == {}
        assert handler.trace_rows == {}


@pytest.mark.asyncio
async def test_a_workers_call_id_cannot_collide_with_the_parents() -> None:
    """llama.cpp numbers calls call_0, call_1 in every loop. A worker's
    call-1 must not be taken for the parent's live call-1.
    """
    app = _Host()
    async with app.run_test() as pilot:
        handler = EventHandler(app.mount_message, lambda: True)
        parent = ToolCallEvent(
            tool_call_id="call-1",
            tool_name="orchestrate",
            tool_class=Orchestrate,
            args=OrchestrateArgs(goal="g"),
        )
        await handler.handle_event(parent)
        call, result = _bash("call-1", "ls", parent="call-1")
        await handler.handle_event(call)
        await handler.handle_event(result)
        await pilot.pause()
        rows = _rows(app)
        assert [type(r).__name__ for r in rows] == [
            "ToolCallMessage",
            "ToolCallMessage",
            "ToolResultMessage",
        ]
        # The parent is still registered and still spinning; the child is done.
        assert "call-1" in handler.tool_calls
        assert rows[0].get_content().startswith("orchestrate")
        assert rows[1].get_content() == "bash $ ls"


def test_a_subagents_cancelled_result_is_not_the_parents_cancellation() -> None:
    top = ToolResultEvent(
        tool_call_id="x", tool_name="bash", tool_class=Bash, cancelled=True
    )
    nested = top.model_copy(update={"parent_call_id": "orch-1", "depth": 1})
    assert is_user_cancellation_event(top)
    assert not is_user_cancellation_event(nested)


def test_result_type_introspection_skips_every_event_type() -> None:
    """task and orchestrate yield any BaseEvent before their result; the
    result model is still found.
    """
    assert Task._get_tool_args_results() == (TaskArgs, TaskResult)
    assert Orchestrate._get_tool_args_results() == (OrchestrateArgs, OrchestrateResult)
    assert issubclass(TraceLineEvent, BaseEvent)
    assert issubclass(Bash, BaseTool)


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def _tool_message(call_id: str, name: str, result: dict) -> LLMMessage:
    return LLMMessage(
        role=Role.tool,
        name=name,
        tool_call_id=call_id,
        content="ok",
        meta=MessageMeta(tool_outcome=ToolOutcome.success, tool_result=result),
    )


def _assistant_call(call_id: str, name: str, args: dict) -> LLMMessage:
    return LLMMessage(
        role=Role.assistant,
        content="",
        tool_calls=(
            ToolCall(
                id=call_id, function=FunctionCall(name=name, arguments=json.dumps(args))
            ),
        ),
    )


def _child_session(tmp_path: Path, name: str) -> tuple[Path, list[LLMMessage]]:
    """A worker session on disk (messages.jsonl is what the loader reads)."""
    session_dir = tmp_path / "agents" / name
    session_dir.mkdir(parents=True)
    messages = [
        LLMMessage(role=Role.user, content="do it"),
        _assistant_call("w1", "bash", {"command": "ls"}),
        _tool_message(
            "w1",
            "bash",
            {"command": "ls", "stdout": "a", "stderr": "", "returncode": 0},
        ),
    ]
    (session_dir / "messages.jsonl").write_text("x\n")  # exists; the loader is faked
    return session_dir, messages


def test_subagent_children_from_task_and_orchestrate_results(tmp_path: Path) -> None:
    assert (
        subagent_children(
            "bash", BashResult(command="", stdout="", stderr="", returncode=0)
        )
        == []
    )
    task = TaskResult(response="r", turns_used=1, completed=True, session_dir="/s/one")
    assert subagent_children("task", task) == [("", Path("/s/one"))]
    assert (
        subagent_children(
            "task", TaskResult(response="r", turns_used=1, completed=True)
        )
        == []
    )
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    (ledger / "calls.json").write_text(
        json.dumps([
            {"label": "manager · plan", "session_dir": "/s/p"},
            {"label": "worker · round 1/2 · task 1: x", "session_dir": ""},
        ])
    )
    orch = OrchestrateResult(status="done", ledger_dir=str(ledger))
    assert subagent_children("orchestrate", orch) == [
        ("manager · plan", Path("/s/p")),
        ("worker · round 1/2 · task 1: x", Path("")),
    ]
    assert (
        subagent_children(
            "orchestrate", OrchestrateResult(status="done", ledger_dir="/gone")
        )
        == []
    )


def test_resume_rebuilds_task_children_under_the_call(tmp_path: Path) -> None:
    session_dir, child_messages = _child_session(tmp_path, "explore_1")
    history = [
        LLMMessage(role=Role.user, content="explore"),
        _assistant_call("t1", "task", {"task": "look", "agent": "explore"}),
        _tool_message(
            "t1",
            "task",
            {
                "response": "found",
                "turns_used": 1,
                "completed": True,
                "session_dir": str(session_dir),
            },
        ),
        LLMMessage(role=Role.assistant, content="done"),
    ]
    loaded: list[Path] = []

    def loader(path: Path) -> list[LLMMessage]:
        loaded.append(path)
        return child_messages

    indices: WeakKeyDictionary[Widget, int] = WeakKeyDictionary()
    widgets = build_history_widgets(
        history,
        build_tool_call_map(history),
        start_index=0,
        tools_collapsed=True,
        history_widget_indices=indices,
        tool_classes={"task": Task, "bash": Bash},
        child_loader=loader,
    )
    kinds = [type(w).__name__ for w in widgets]
    assert kinds == [
        "UserMessage",
        "ToolCallMessage",  # task
        "ToolCallMessage",  # the worker's bash, depth 1 (its prompt bubble is not shown)
        "ToolResultMessage",  # depth 1
        "ToolResultMessage",  # task result
        "AssistantMessage",
    ]
    assert loaded == [session_dir]
    assert widgets[2].has_class("depth-1") and widgets[3].has_class("depth-1")
    assert not widgets[1].has_class("depth-1") and not widgets[4].has_class("depth-1")
    # Children are not stamped into the parent's history indices.
    stamped = set(indices.values())
    assert stamped == {0, 1, 2, 3}
    assert widgets[2] not in indices and widgets[3] not in indices


def test_resume_rebuilds_orchestrate_role_calls_with_headers(tmp_path: Path) -> None:
    worker_dir, child_messages = _child_session(tmp_path, "orchestrate-worker_1")
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    (ledger / "calls.json").write_text(
        json.dumps([
            {"label": "manager · plan", "session_dir": str(tmp_path / "gone")},
            {"label": "worker · round 1/1 · task 1: x", "session_dir": str(worker_dir)},
        ])
    )
    history = [
        _assistant_call("o1", "orchestrate", {"goal": "g"}),
        _tool_message(
            "o1",
            "orchestrate",
            {"status": "done", "rounds": 1, "calls": 2, "ledger_dir": str(ledger)},
        ),
    ]

    def loader(path: Path) -> list[LLMMessage]:
        return child_messages if path == worker_dir else []

    widgets = build_history_widgets(
        history,
        build_tool_call_map(history),
        start_index=0,
        tools_collapsed=True,
        history_widget_indices=WeakKeyDictionary(),
        tool_classes={"orchestrate": Orchestrate, "bash": Bash},
        child_loader=loader,
    )
    kinds = [type(w).__name__ for w in widgets]
    assert kinds == [
        "ToolCallMessage",  # orchestrate
        "TraceLineMessage",  # manager · plan (its session is gone: header only)
        "TraceLineMessage",  # worker header
        "ToolCallMessage",
        "ToolResultMessage",
        "ToolResultMessage",  # orchestrate result
    ]
    assert widgets[1].text.startswith("── manager · plan ──")
    assert widgets[2].text.startswith("── worker · round 1/1 · task 1: x ──")
    assert all(w.has_class("depth-1") for w in widgets[1:5])
    assert not widgets[5].has_class("depth-1")


def test_without_a_loader_resume_is_what_it_was(tmp_path: Path) -> None:
    history = [
        _assistant_call("t1", "task", {"task": "look", "agent": "explore"}),
        _tool_message(
            "t1",
            "task",
            {"response": "r", "turns_used": 1, "completed": True, "session_dir": "/x"},
        ),
    ]
    widgets = build_history_widgets(
        history,
        build_tool_call_map(history),
        start_index=0,
        tools_collapsed=True,
        history_widget_indices=WeakKeyDictionary(),
        tool_classes={"task": Task},
    )
    assert [type(w).__name__ for w in widgets] == [
        "ToolCallMessage",
        "ToolResultMessage",
    ]
