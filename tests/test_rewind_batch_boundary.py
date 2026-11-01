"""Rewind into a turn, as a steering message at a closed tool batch.

The cut lands right after a batch whose every call has its result, the tail
is those results, and the next message goes in as a steering message: the
same slot and the same envelope as a message typed mid-turn. Nothing is
shimmed and the history stays what happened. Checkpoints are taken at every
such boundary so files can come back too.
"""

from __future__ import annotations

import pytest

from privibe.cli.textual_ui.widgets.chat_input.container import ChatInputContainer
from privibe.cli.textual_ui.widgets.messages import UserMessage
from privibe.cli.textual_ui.widgets.tools import ToolResultMessage
from privibe.cli.textual_ui.windowing.rewind_targets import (
    RewindKind,
    rewind_plan,
    rewind_targets,
)
from privibe.core.llm.message_utils import STEERING_TAG
from privibe.core.rewind.manager import RewindError
from privibe.core.types import FunctionCall, Role, ToolCall, UserMessageEvent
from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_app,
    build_test_vibe_config,
)
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


def _todo_call(call_id: str, index: int = 0) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=index,
        function=FunctionCall(name="todo", arguments='{"action": "read"}'),
    )


def _config():
    return build_test_vibe_config(
        enabled_tools=["todo"],
        tools={"todo": {"permission": "always"}},
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=False,
    )


def _two_batches_then_answer(*later: str) -> FakeBackend:
    """One turn: a two-call batch, a one-call batch, an answer; then plain
    answers for whatever comes after.
    """
    return FakeBackend([
        [
            mock_llm_chunk(
                content="", tool_calls=[_todo_call("a", 0), _todo_call("b", 1)]
            )
        ],
        [mock_llm_chunk(content="", tool_calls=[_todo_call("c", 0)])],
        [mock_llm_chunk(content="turn one done")],
        *([mock_llm_chunk(content=t)] for t in later),
    ])


async def _collect(loop, prompt: str, **kw):
    return [ev async for ev in loop.act(prompt, **kw)]


# ---------------------------------------------------------------------------
# Loop side
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_boundaries_and_checkpoints_at_each() -> None:
    loop = build_test_agent_loop(config=_config(), backend=_two_batches_then_answer())
    await _collect(loop, "go")
    roles = [m.role for m in loop.messages]
    # system, user, assistant(a,b), tool, tool, assistant(c), tool, assistant
    assert roles == [
        Role.system,
        Role.user,
        Role.assistant,
        Role.tool,
        Role.tool,
        Role.assistant,
        Role.tool,
        Role.assistant,
    ]
    rm = loop.rewind_manager
    assert rm.batch_boundaries() == [5, 7]
    # One checkpoint for the user message and one per boundary, no duplicates.
    assert [cp.message_index for cp in rm.checkpoints] == [1, 5, 7]


@pytest.mark.asyncio
async def test_rewind_to_index_cuts_at_a_boundary_and_refuses_elsewhere() -> None:
    loop = build_test_agent_loop(config=_config(), backend=_two_batches_then_answer())
    await _collect(loop, "go")
    rm = loop.rewind_manager
    with pytest.raises(RewindError, match="does not close a tool batch"):
        await rm.rewind_to_index(4, restore_files=False)  # inside batch a,b
    with pytest.raises(RewindError):
        await rm.rewind_to_index(6, restore_files=False)  # the assistant(c) slot
    errors = await rm.rewind_to_index(5, restore_files=False)
    assert errors == []
    assert [m.role for m in loop.messages] == [
        Role.system,
        Role.user,
        Role.assistant,
        Role.tool,
        Role.tool,
    ]
    assert [cp.message_index for cp in rm.checkpoints] == [1]
    # Still a boundary: the conversation now ends on the closed batch.
    assert rm.batch_boundaries() == [5]


@pytest.mark.asyncio
async def test_act_steering_appends_a_steering_message_wrapped_on_the_wire_only() -> (
    None
):
    backend = _two_batches_then_answer("steered answer")
    loop = build_test_agent_loop(config=_config(), backend=backend)
    await _collect(loop, "go")
    await loop.rewind_manager.rewind_to_index(5, restore_files=False)
    events = await _collect(loop, "do it differently", steering=True)
    bubble = next(e for e in events if isinstance(e, UserMessageEvent))
    assert bubble.steering is True and bubble.message_index == 5
    steer = loop.messages[5]
    assert steer.role == Role.user and steer.steering is True
    assert steer.content == "do it differently"  # raw words in storage
    wire = backend.requests_messages[-1]
    sent = [m for m in wire if m.role == Role.user][-1]
    assert sent.content is not None and f"<{STEERING_TAG}>" in sent.content
    assert "do it differently" in sent.content
    # The request is the old request up to the cut plus the steer: a prefix
    # extension, no shim anywhere.
    before = backend.requests_messages[1]  # the request that got answer "c"
    assert [m.role for m in wire[:5]] == [m.role for m in before[:5]]
    assert all("no response" not in (m.content or "").lower() for m in wire)


# ---------------------------------------------------------------------------
# Targets and plans (pure)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_targets_are_user_messages_and_mid_turn_batch_rows() -> None:
    backend = _two_batches_then_answer("two")
    app = build_test_vibe_app(
        agent_loop=build_test_agent_loop(config=_config(), backend=backend)
    )
    async with app.run_test() as pilot:
        await pilot.press(*"go")
        await pilot.press("enter")
        await pilot.pause(0.6)
        await pilot.press(*"two")
        await pilot.press("enter")
        await pilot.pause(0.4)
        children = list(app.query_one("#messages").children)
        targets = rewind_targets(
            children,
            list(app.agent_loop.messages),
            app.agent_loop.rewind_manager.batch_boundaries(),
        )
        kinds = [type(t).__name__ for t in targets]
        # user "go", the row closing batch (a,b), the row closing batch (c),
        # user "two". The batch rows are the LAST result of each batch.
        assert kinds == [
            "UserMessage",
            "ToolResultMessage",
            "ToolResultMessage",
            "UserMessage",
        ]
        plan = rewind_plan(
            targets[1],
            children,
            list(app.agent_loop.messages),
            app.agent_loop.rewind_manager.batch_boundaries(),
        )
        assert plan is not None and plan.kind == RewindKind.BATCH
        assert plan.cut_index == 5 and plan.steering is True
        assert plan.preview.startswith("after todo")
        # The target row stays; removal starts at the row after it.
        assert plan.first_removed is children[children.index(targets[1]) + 1]
        # Of the two results of batch (a,b), only the one that closed it (the
        # later message, whatever order the rows were mounted in) is a target.
        results = [c for c in children if isinstance(c, ToolResultMessage)]
        assert len(results) == 3
        assert sum(r in targets for r in results[:2]) == 1


# ---------------------------------------------------------------------------
# The whole gesture in the TUI
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_alt_right_steps_onto_a_batch_row_and_the_next_message_steers() -> None:
    backend = _two_batches_then_answer("steered answer")
    app = build_test_vibe_app(
        agent_loop=build_test_agent_loop(config=_config(), backend=backend)
    )
    async with app.run_test() as pilot:
        await pilot.press(*"go")
        await pilot.press("enter")
        await pilot.pause(0.6)
        await app.workers.wait_for_complete()

        # alt+up: the user message, as always. alt+right: the first batch row.
        await pilot.press("alt+up")
        await app.workers.wait_for_complete()
        await pilot.pause(0.1)
        assert isinstance(app._rewind_highlighted_widget, UserMessage)
        await pilot.press("alt+right")
        await app.workers.wait_for_complete()
        await pilot.pause(0.1)
        target = app._rewind_highlighted_widget
        assert isinstance(target, ToolResultMessage)
        rows_before = list(app.query_one("#messages").children)
        target_pos = rows_before.index(target)

        await pilot.press("enter")  # Edit from here
        await app.workers.wait_for_complete()
        await pilot.pause(0.2)
        rows = list(app.query_one("#messages").children)
        assert (
            rows == rows_before[: target_pos + 1]
        )  # the batch row stays, the rest went
        assert [m.role for m in app.agent_loop.messages][-2:] == [Role.tool, Role.tool]
        assert app.query_one(ChatInputContainer).value == ""
        assert app._steer_next_message is True

        await pilot.press(*"use a list")
        await pilot.press("enter")
        await pilot.pause(0.6)
        await app.workers.wait_for_complete()

        steer = app.agent_loop.messages[5]
        assert (
            steer.role == Role.user and steer.steering and steer.content == "use a list"
        )
        bubbles = [
            c for c in app.query_one("#messages").children if isinstance(c, UserMessage)
        ]
        assert [b.get_content() for b in bubbles] == ["go", "use a list"]
        assert bubbles[1].message_index == 5
        assert app._steer_next_message is False
        wire = backend.requests_messages[-1]
        assert f"<{STEERING_TAG}>" in (wire[-1].content or "")
        assert all(
            STEERING_TAG not in (m.content or "") for m in app.agent_loop.messages
        )
        assert app.agent_loop.messages[-1].content == "steered answer"
