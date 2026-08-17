from __future__ import annotations

import time

import pytest

from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_app,
    build_test_vibe_config,
)
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from privibe.cli.textual_ui.app import VibeApp
from privibe.cli.textual_ui.widgets.compact import CompactMessage
from privibe.cli.textual_ui.widgets.load_more import (
    HistoryLoadMoreMessage,
    HistoryLoadMoreRequested,
)
from privibe.cli.textual_ui.widgets.messages import UserMessage
from privibe.core.agents.models import BuiltinAgentName
from privibe.core.config import SessionLoggingConfig, VibeConfig
from privibe.core.types import FunctionCall, LLMMessage, Role, ToolCall


async def _wait_until(pause, predicate, timeout: float = 5.0) -> None:
    start = time.monotonic()
    while (time.monotonic() - start) < timeout:
        if predicate():
            return
        await pause(0.02)
    raise AssertionError("Condition was not met within the timeout")


@pytest.mark.asyncio
async def test_rewind_reaches_user_message_buried_in_backfill() -> None:
    """A user message windowed into the backfill behind a batch that contains
    no user message must still be reachable by rewind.

    Layout (50 messages, tail = 20, load-more batch = 10):
      index 0      : user  "deep-target"   <- in backfill, behind non-user batches
      indices 1-29 : assistant             <- 29 non-user messages
      index 30     : user  "recent"        <- mounted in the tail
      indices 31-49: assistant             <- fills the rest of the tail
    Walking up from "recent" must load through the all-assistant batches
    (indices 20-29, then 10-19) before "deep-target" (index 0) is mounted.
    """
    config = VibeConfig(session_logging=SessionLoggingConfig(enabled=False))
    agent_loop = build_test_agent_loop(config=config)
    history = [LLMMessage(role=Role.user, content="deep-target")]
    history += [LLMMessage(role=Role.assistant, content=f"a{i}") for i in range(29)]
    history += [LLMMessage(role=Role.user, content="recent")]
    history += [LLMMessage(role=Role.assistant, content=f"b{i}") for i in range(19)]
    agent_loop.messages._data.extend(history)

    app = VibeApp(agent_loop=agent_loop)
    async with app.run_test() as pilot:
        # Resume windows the history: only the tail mounts, the rest is backfill
        # behind a "load more" control.
        await _wait_until(
            pilot.pause, lambda: len(app.query(HistoryLoadMoreMessage)) == 1
        )

        # Enter rewind — highlights the only mounted user message.
        await pilot.press("alt+up")
        await app.workers.wait_for_complete()
        await pilot.pause(0.1)
        assert app._rewind_highlighted_widget is not None
        assert app._rewind_highlighted_widget.get_content() == "recent"

        # Up again: must load through the all-assistant batches to reach the
        # backfilled user message rather than getting stuck.
        await pilot.press("alt+up")
        await app.workers.wait_for_complete()
        await pilot.pause(0.1)
        assert app._rewind_highlighted_widget is not None
        assert app._rewind_highlighted_widget.get_content() == "deep-target"


def _make_todo_tool_call(call_id: str, index: int = 0) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=index,
        function=FunctionCall(name="todo", arguments='{"action": "read"}'),
    )


def _build_pruning_tool_turn_app(n_turns: int) -> VibeApp:
    """Live-session app whose turns each produce reasoning, an assistant
    message with several tool calls, tool results, and a tall final answer,
    with a prune threshold small enough that early turns fall off screen.
    """
    long_response = "\n\n".join(
        f"paragraph {i} with some text in it" for i in range(15)
    )
    responses = []
    for i in range(n_turns):
        responses.append(
            [
                mock_llm_chunk(
                    content=f"Checking todos for turn {i}.",
                    reasoning_content="Let me think about the todos here.",
                    tool_calls=[
                        _make_todo_tool_call(f"call_{i}_{j}", index=j)
                        for j in range(3)
                    ],
                )
            ]
        )
        responses.append(
            [
                mock_llm_chunk(
                    content=long_response,
                    reasoning_content="Now let me summarize what I found.",
                )
            ]
        )
    config = build_test_vibe_config(
        message_prune_keep_rows=40,
        enabled_tools=["todo"],
        tools={"todo": {"permission": "always"}},
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=False,
    )
    agent_loop = build_test_agent_loop(
        config=config,
        agent_name=BuiltinAgentName.AUTO_APPROVE,
        backend=FakeBackend(responses),
    )
    return build_test_vibe_app(config=config, agent_loop=agent_loop)


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_rewind_enters_and_walks_back_after_live_prune_of_tool_turns() -> None:
    """Regression: live pruning of tool-call/reasoning turns must not break
    rewind entry (bug: alt+up did nothing with zero mounted user messages)
    nor lose messages to a miscounted backfill boundary (bug: the old
    widget-count fallback assumed one widget per history message).
    """
    n_turns = 3
    app = _build_pruning_tool_turn_app(n_turns)

    async with app.run_test(size=(100, 30)) as pilot:
        for i in range(n_turns):
            for ch in f"msg{i}":
                await pilot.press(ch)
            await pilot.press("enter")
            await pilot.pause(0.4)
        await app.workers.wait_for_complete()
        await pilot.pause(0.2)

        messages_area = app.query_one("#messages")
        mounted_users = [
            c
            for c in messages_area.children
            if isinstance(c, UserMessage) and c.message_index is not None
        ]
        assert not mounted_users, (
            "Precondition: pruning should have removed every user message; "
            "shrink message_prune_keep_rows or grow the turns if this fails"
        )
        assert app._windowing.has_backfill

        # Bug 1: alt+up must enter rewind by loading from the backfill.
        seen: list[str] = []
        for _ in range(n_turns):
            await pilot.press("alt+up")
            await app.workers.wait_for_complete()
            await pilot.pause(0.15)
            widget = app._rewind_highlighted_widget
            seen.append(widget.get_content() if widget else "<none>")
        assert seen == ["msg2", "msg1", "msg0"], f"walk-up sequence: {seen}"


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_no_message_lost_to_backfill_hole_after_live_prune() -> None:
    """Regression: after live pruning, exhausting "Load more" must remount
    every user message; none may fall between the mounted tail and the
    backfill boundary.
    """
    n_turns = 3
    app = _build_pruning_tool_turn_app(n_turns)

    async with app.run_test(size=(100, 30)) as pilot:
        for i in range(n_turns):
            for ch in f"msg{i}":
                await pilot.press(ch)
            await pilot.press("enter")
            await pilot.pause(0.4)
        await app.workers.wait_for_complete()
        await pilot.pause(0.2)

        messages_area = app.query_one("#messages")
        for _ in range(20):
            if not app._windowing.has_backfill:
                break
            await app.on_history_load_more_requested(HistoryLoadMoreRequested())
            await pilot.pause(0.05)

        remounted = [
            c.get_content()
            for c in messages_area.children
            if isinstance(c, UserMessage) and c.message_index is not None
        ]
        assert remounted == [f"msg{i}" for i in range(n_turns)], (
            f"user messages after exhausting backfill: {remounted}"
        )


@pytest.mark.asyncio
async def test_compact_resets_windowing_state() -> None:
    """Regression: compaction rewinds history, so pre-compact backfill must
    not survive where "Load more" could resurface deleted messages.
    """
    config = VibeConfig(session_logging=SessionLoggingConfig(enabled=False))
    agent_loop = build_test_agent_loop(config=config)
    history = [LLMMessage(role=Role.user, content=f"m{i}") for i in range(40)]
    agent_loop.messages._data.extend(history)

    app = VibeApp(agent_loop=agent_loop)
    async with app.run_test() as pilot:
        await _wait_until(
            pilot.pause, lambda: len(app.query(HistoryLoadMoreMessage)) == 1
        )
        assert app._windowing.has_backfill

        messages_area = app.query_one("#messages")
        compact_widget = CompactMessage()
        await messages_area.mount(compact_widget)
        await app.on_compact_message_completed(
            CompactMessage.Completed(compact_widget)
        )
        await pilot.pause(0.1)

        assert not app._windowing.has_backfill
        assert app._load_more.widget is None
        assert len(app.query(HistoryLoadMoreMessage)) == 0


@pytest.mark.asyncio
async def test_load_more_survives_detachment_by_live_prune() -> None:
    """A bulk remove_children (live prune) can detach the load-more button
    without the manager noticing. The next load-more request must fall back
    to mounting at the top and remount the button, not crash with a
    MountError on the detached anchor."""
    config = VibeConfig(session_logging=SessionLoggingConfig(enabled=False))
    agent_loop = build_test_agent_loop(config=config)
    history = [LLMMessage(role=Role.user, content="deep-target")]
    history += [LLMMessage(role=Role.assistant, content=f"a{i}") for i in range(40)]
    agent_loop.messages._data.extend(history)

    app = VibeApp(agent_loop=agent_loop)
    async with app.run_test() as pilot:
        await _wait_until(
            pilot.pause, lambda: len(app.query(HistoryLoadMoreMessage)) == 1
        )
        remaining_before = app._windowing.remaining

        # Detach the button behind the manager's back, as the prune does, and
        # reinstate the stale reference: in the real race the rewind worker
        # runs in the window before _try_prune's deferred cleanup clears it.
        messages_area = app.query_one("#messages")
        stale = app._load_more.widget
        assert stale is not None
        await messages_area.remove_children([stale])
        await pilot.pause(0.1)
        assert stale.parent is None
        app._load_more.widget = stale

        await app.on_history_load_more_requested(HistoryLoadMoreRequested())
        await pilot.pause(0.1)

        assert app._windowing.remaining < remaining_before
        assert app._windowing.has_backfill
        healed = app._load_more.widget
        assert healed is not None
        assert healed is not stale
        assert healed.parent is not None
