"""Steering: a message typed while the agent runs becomes a real user message.

It lands after the running tool batch (never between its results), is stored
as the user's raw words, and reaches the model wrapped in the steering
envelope. The envelope is applied at request time only.
"""

from __future__ import annotations

import pytest

from privibe.core.agents.models import BuiltinAgentName
from privibe.core.llm.message_utils import STEERING_TAG, wrap_steering_messages
from privibe.core.types import (
    BaseEvent,
    FunctionCall,
    LLMMessage,
    Role,
    ToolCall,
    ToolCallEvent,
    UserMessageEvent,
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


def _todo_call(call_id: str, index: int) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=index,
        function=FunctionCall(name="todo", arguments='{"action": "read"}'),
    )


def _loop(backend: FakeBackend):
    config = build_test_vibe_config(
        enabled_tools=["todo"],
        tools={"todo": {"permission": "always"}},
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=False,
    )
    return build_test_agent_loop(
        config=config, agent_name=BuiltinAgentName.AUTO_APPROVE, backend=backend
    )


async def _act_steering_on_first_tool_call(loop, prompt: str, steer: str):
    events: list[BaseEvent] = []
    steered = False
    async for ev in loop.act(prompt):
        events.append(ev)
        if not steered and isinstance(ev, ToolCallEvent):
            steered = True
            loop.queue_steering(steer)
    return events


@pytest.mark.asyncio
async def test_steer_lands_after_the_whole_tool_batch() -> None:
    backend = FakeBackend([
        [
            mock_llm_chunk(
                content="two tools", tool_calls=[_todo_call("a", 0), _todo_call("b", 1)]
            )
        ],
        [mock_llm_chunk(content="done")],
    ])
    loop = _loop(backend)
    events = await _act_steering_on_first_tool_call(loop, "go", "STEER")

    roles = [m.role for m in loop.messages]
    # system, user, assistant(2 calls), tool, tool, user(steer), assistant
    assert roles == [
        Role.system,
        Role.user,
        Role.assistant,
        Role.tool,
        Role.tool,
        Role.user,
        Role.assistant,
    ]
    steer = loop.messages[5]
    assert steer.steering is True
    assert steer.injected is False
    assert steer.content == "STEER"
    assert STEERING_TAG not in (steer.content or "")
    assert not any(
        STEERING_TAG in (m.content or "") for m in loop.messages if m.role == Role.tool
    )

    steer_events = [e for e in events if isinstance(e, UserMessageEvent) and e.steering]
    assert len(steer_events) == 1
    assert steer_events[0].content == "STEER"
    assert steer_events[0].message_index == 5
    assert loop.messages[5].message_id == steer_events[0].message_id
    assert loop.drain_steering_queue() == []


@pytest.mark.asyncio
async def test_steer_is_wrapped_on_the_wire_only() -> None:
    backend = FakeBackend([
        [mock_llm_chunk(content="tool", tool_calls=[_todo_call("a", 0)])],
        [mock_llm_chunk(content="done")],
    ])
    loop = _loop(backend)
    await _act_steering_on_first_tool_call(loop, "go", "STEER")

    second_request = backend.requests_messages[1]
    wire_steer = [m for m in second_request if m.role == Role.user][-1]
    assert isinstance(wire_steer.content, str)
    assert wire_steer.content.startswith(f"<{STEERING_TAG}>")
    assert wire_steer.content.endswith(f"</{STEERING_TAG}>")
    assert "STEER" in wire_steer.content
    assert wire_steer.message_id == loop.messages[4].message_id
    # stored history untouched
    assert loop.messages[4].content == "STEER"


@pytest.mark.asyncio
async def test_steer_with_no_tool_call_stays_queued_for_the_ui() -> None:
    backend = FakeBackend([[mock_llm_chunk(content="no tools")]])
    loop = _loop(backend)
    queued = False
    async for _ in loop.act("go"):
        if not queued:
            queued = True
            loop.queue_steering("LATE")
    assert all(m.role != Role.user or m.content == "go" for m in loop.messages)
    assert loop.drain_steering_queue() == ["LATE"]


def test_wrap_steering_messages_is_pure_and_targeted() -> None:
    msgs = [
        LLMMessage(role=Role.system, content="sys"),
        LLMMessage(role=Role.user, content="plain"),
        LLMMessage(role=Role.user, content="steer", steering=True),
    ]
    once = wrap_steering_messages(msgs)
    twice = wrap_steering_messages(msgs)
    assert [m.content for m in once] == [m.content for m in twice]
    assert once[0] is msgs[0] and once[1] is msgs[1]
    wrapped = once[2].content
    assert isinstance(wrapped, str)
    assert wrapped.startswith(f"<{STEERING_TAG}>") and "steer" in wrapped
    assert msgs[2].content == "steer"
