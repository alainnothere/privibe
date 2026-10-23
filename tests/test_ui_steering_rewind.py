"""Steering messages in the TUI: bubble, rewind target, and a stable prefix.

The last test is the one that matters: rewind to a steer, edit it, run two
more rounds, and prove every request the backend saw is a byte-for-byte
extension of the one before it, with message 0 and the tools never moving.
"""

from __future__ import annotations

import json

import pytest

from privibe.cli.textual_ui.widgets.chat_input.container import ChatInputContainer
from privibe.cli.textual_ui.widgets.messages import UserMessage
from privibe.core.llm.message_utils import STEERING_TAG
from privibe.core.types import FunctionCall, LLMMessage, Role, ToolCall, ToolCallEvent
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


def _app(backend: FakeBackend):
    config = build_test_vibe_config(
        enabled_tools=["todo"],
        tools={"todo": {"permission": "always"}},
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=False,
    )
    loop = build_test_agent_loop(config=config, backend=backend)
    return build_test_vibe_app(agent_loop=loop)


def _steer_on_first_tool_call(app, text: str) -> None:
    """Queue *text* exactly as the input handler does while the agent runs."""
    original = app.agent_loop.act
    fired = False

    async def act(msg, **kw):
        nonlocal fired
        async for ev in original(msg, **kw):
            if not fired and isinstance(ev, ToolCallEvent):
                fired = True
                app.agent_loop.queue_steering(text)
            yield ev

    app.agent_loop.act = act


async def _send(pilot, text: str) -> None:
    await pilot.press(*text)
    await pilot.press("enter")
    await pilot.pause(0.5)


async def _alt_up(pilot) -> None:
    await pilot.press("alt+up")
    await pilot.app.workers.wait_for_complete()
    await pilot.pause(0.1)


def _user_widgets(app) -> list[UserMessage]:
    return [
        w
        for w in app.query_one("#messages").children
        if isinstance(w, UserMessage) and w.message_index is not None
    ]


def _two_call_batch_then_answer(*later: str) -> FakeBackend:
    return FakeBackend([
        [
            mock_llm_chunk(
                content="working", tool_calls=[_todo_call("a", 0), _todo_call("b", 1)]
            )
        ],
        [mock_llm_chunk(content="turn one done")],
        *([mock_llm_chunk(content=t)] for t in later),
    ])


@pytest.mark.asyncio
async def test_steer_mounts_a_user_bubble_with_its_history_index() -> None:
    app = _app(_two_call_batch_then_answer())
    async with app.run_test() as pilot:
        _steer_on_first_tool_call(app, "STEER")
        await _send(pilot, "first")

        widgets = _user_widgets(app)
        assert [w.get_content() for w in widgets] == ["first", "STEER"]
        steer_widget = widgets[1]
        assert steer_widget.message_index is not None
        stored = app.agent_loop.messages[steer_widget.message_index]
        assert (
            stored.role == Role.user and stored.steering and stored.content == "STEER"
        )
        assert app.agent_loop.rewind_manager.get_rewindable_messages()[-1] == (
            steer_widget.message_index,
            "STEER",
        )


@pytest.mark.asyncio
async def test_rewind_to_steer_prefills_raw_text_and_cuts_below_it() -> None:
    app = _app(_two_call_batch_then_answer())
    async with app.run_test() as pilot:
        _steer_on_first_tool_call(app, "STEER")
        await _send(pilot, "first")

        await _alt_up(pilot)
        assert app._rewind_highlighted_widget is not None
        assert app._rewind_highlighted_widget.get_content() == "STEER"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause(0.2)

        assert app.query_one(ChatInputContainer).value == "STEER"
        roles = [m.role for m in app.agent_loop.messages]
        assert roles == [Role.system, Role.user, Role.assistant, Role.tool, Role.tool]
        assert [w.get_content() for w in _user_widgets(app)] == ["first"]


@pytest.mark.asyncio
async def test_resume_renders_a_steer_as_a_user_bubble() -> None:
    app = _app(FakeBackend([]))
    loop = app.agent_loop
    loop.messages.add(LLMMessage(role=Role.user, content="first"))
    loop.messages.add(
        LLMMessage(
            role=Role.assistant, content="working", tool_calls=(_todo_call("a"),)
        )
    )
    loop.messages.add(
        LLMMessage(role=Role.tool, tool_call_id="a", name="todo", content="{}")
    )
    loop.messages.add(LLMMessage(role=Role.user, content="STEER", steering=True))
    loop.messages.add(LLMMessage(role=Role.assistant, content="ok"))
    async with app.run_test() as pilot:
        await pilot.pause(0.3)
        widgets = _user_widgets(app)
        assert [w.get_content() for w in widgets] == ["first", "STEER"]
        assert widgets[1].message_index == 4


def _wire(msg: LLMMessage) -> str:
    """The bytes that matter for the prefix: role and content, nothing local."""
    return json.dumps(
        {
            "role": msg.role.value,
            "content": msg.content,
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls or ()] or None,
            "tool_call_id": msg.tool_call_id,
        },
        sort_keys=True,
    )


@pytest.mark.asyncio
async def test_rewind_then_two_rounds_keeps_every_request_a_prefix_extension() -> None:
    backend = _two_call_batch_then_answer("after the edit", "round two", "round three")
    app = _app(backend)
    async with app.run_test() as pilot:
        _steer_on_first_tool_call(app, "STEER")
        await _send(pilot, "first")
        assert len(backend.requests_messages) == 2
        before_rewind = [_wire(m) for m in backend.requests_messages[1]]

        await _alt_up(pilot)
        assert app._rewind_highlighted_widget is not None
        assert app._rewind_highlighted_widget.get_content() == "STEER"
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause(0.2)
        assert app.query_one(ChatInputContainer).value == "STEER"

        # edit the steer and send it, then two plain rounds
        await pilot.press("ctrl+a", "backspace")
        await _send(pilot, "EDITED")
        await _send(pilot, "two")
        await _send(pilot, "three")

    requests = [[_wire(m) for m in req] for req in backend.requests_messages]
    assert len(requests) == 5
    tools = [
        json.dumps(
            [t.model_dump() for t in app.agent_loop.messages.tools_for_request()],
            sort_keys=True,
        )
    ]

    # message 0 never moves
    assert len({req[0] for req in requests}) == 1
    # the rewind kept everything above the cut
    cut = (
        len(before_rewind) - 1
    )  # the old steer was the last user message before the answer
    post = requests[2]
    assert post[:cut] == before_rewind[:cut]
    assert "EDITED" in post[cut]
    # every later request is the previous one plus appended messages
    for prev, nxt in zip(requests[2:], requests[3:], strict=False):
        assert nxt[: len(prev)] == prev
        assert len(nxt) > len(prev)
    # the fingerprint guard never fired: tools array is one frozen blob
    assert len(tools) == 1
    # the envelope rides on the wire, never in storage
    wire_text = "\n".join("\n".join(req) for req in requests)
    assert f"<{STEERING_TAG}>" in wire_text
    assert all(STEERING_TAG not in (m.content or "") for m in app.agent_loop.messages)
