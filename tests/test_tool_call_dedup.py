from __future__ import annotations

import pytest

from privibe.core.agent_loop import AgentLoop
from privibe.core.agents import AgentManager
from privibe.core.agents.models import BuiltinAgentName
from privibe.core.config import VibeConfig
from privibe.core.skills.manager import SkillManager
from privibe.core.system_prompt import (
    DUPLICATE_TOOL_CALL_NOTE,
    get_universal_system_prompt,
)
from privibe.core.tools.base import ToolPermission
from privibe.core.tools.manager import ToolManager
from privibe.core.types import BaseEvent, FunctionCall, Role, ToolCall, ToolResultEvent
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


async def act_and_collect_events(agent_loop: AgentLoop, prompt: str) -> list[BaseEvent]:
    return [ev async for ev in agent_loop.act(prompt)]


def make_config() -> VibeConfig:
    return build_test_vibe_config(
        enabled_tools=["todo"],
        tools={"todo": {"permission": ToolPermission.ALWAYS.value}},
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=False,
    )


def make_todo_tool_call(
    call_id: str, index: int = 0, arguments: str = '{"action": "read"}'
) -> ToolCall:
    return ToolCall(
        id=call_id, index=index, function=FunctionCall(name="todo", arguments=arguments)
    )


def make_agent_loop(backend: FakeBackend) -> AgentLoop:
    return build_test_agent_loop(
        config=make_config(),
        agent_name=BuiltinAgentName.AUTO_APPROVE,
        backend=backend,
    )


@pytest.mark.asyncio
async def test_same_round_duplicates_execute_once() -> None:
    # A and A-duplicate share tool name and identical arguments; B is the same
    # tool with different arguments. Only A and B should run for real.
    call_a = make_todo_tool_call("call_a", index=0, arguments='{"action": "read"}')
    call_a_dup = make_todo_tool_call("call_a_dup", index=1, arguments='{"action": "read"}')
    call_b = make_todo_tool_call(
        "call_b", index=2, arguments='{"action": "write", "todos": []}'
    )
    agent_loop = make_agent_loop(
        backend=FakeBackend([
            [
                mock_llm_chunk(
                    content="Three calls.",
                    tool_calls=[call_a, call_a_dup, call_b],
                )
            ],
            [mock_llm_chunk(content="Done.")],
        ])
    )

    events = await act_and_collect_events(agent_loop, "Go")

    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    executed = [e for e in result_events if not e.skipped and e.result is not None]
    skipped = [e for e in result_events if e.skipped]
    # Exactly two real executions: A and B.
    assert len(executed) == 2
    assert {e.tool_call_id for e in executed} == {"call_a", "call_b"}
    assert agent_loop.stats.tool_calls_succeeded == 2

    # The duplicate produced a skipped result naming the dedup behavior.
    assert len(skipped) == 1
    assert skipped[0].tool_call_id == "call_a_dup"
    assert skipped[0].skip_reason is not None
    assert "Duplicate call not executed" in skipped[0].skip_reason
    assert agent_loop.stats.tool_calls_rejected == 1

    # The duplicate has a tool response message carrying the dedup notice.
    tool_msgs = [m for m in agent_loop.messages if m.role == Role.tool]
    dup_msg = next(m for m in tool_msgs if m.tool_call_id == "call_a_dup")
    assert "Duplicate call not executed" in (dup_msg.content or "")

    # Every one of the three call ids has a tool response message.
    responded_ids = {m.tool_call_id for m in tool_msgs}
    assert {"call_a", "call_a_dup", "call_b"} <= responded_ids


@pytest.mark.asyncio
async def test_same_call_in_different_rounds_runs_both_times() -> None:
    # Identical calls in separate assistant messages are unaffected by dedup.
    round_one = make_todo_tool_call("call_r1", index=0, arguments='{"action": "read"}')
    round_two = make_todo_tool_call("call_r2", index=0, arguments='{"action": "read"}')
    agent_loop = make_agent_loop(
        backend=FakeBackend([
            [mock_llm_chunk(content="First read.", tool_calls=[round_one])],
            [mock_llm_chunk(content="Second read.", tool_calls=[round_two])],
            [mock_llm_chunk(content="Done.")],
        ])
    )

    events = await act_and_collect_events(agent_loop, "Read twice")

    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(result_events) == 2
    assert all(not e.skipped for e in result_events)
    assert {e.tool_call_id for e in result_events} == {"call_r1", "call_r2"}
    assert agent_loop.stats.tool_calls_succeeded == 2
    assert agent_loop.stats.tool_calls_rejected == 0


@pytest.mark.asyncio
async def test_different_args_same_round_both_run() -> None:
    # Same tool, different arguments in one message: both execute, none skipped.
    call_read = make_todo_tool_call("call_read", index=0, arguments='{"action": "read"}')
    call_write = make_todo_tool_call(
        "call_write", index=1, arguments='{"action": "write", "todos": []}'
    )
    agent_loop = make_agent_loop(
        backend=FakeBackend([
            [
                mock_llm_chunk(
                    content="Read and write.",
                    tool_calls=[call_read, call_write],
                )
            ],
            [mock_llm_chunk(content="Done.")],
        ])
    )

    events = await act_and_collect_events(agent_loop, "Go")

    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(result_events) == 2
    assert all(not e.skipped for e in result_events)
    assert {e.tool_call_id for e in result_events} == {"call_read", "call_write"}
    assert agent_loop.stats.tool_calls_succeeded == 2
    assert agent_loop.stats.tool_calls_rejected == 0


def _managers(config: VibeConfig):
    return (
        ToolManager(lambda: config),
        SkillManager(lambda: config),
        AgentManager(lambda: config),
    )


def _reorder(config: VibeConfig):
    tool_manager, skill_manager, agent_manager = _managers(config)
    return tool_manager, config, skill_manager, agent_manager


def test_system_prompt_mentions_duplicate_dedup() -> None:
    detail_config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=True,
    )
    detail_prompt = get_universal_system_prompt(*_reorder(detail_config))
    assert DUPLICATE_TOOL_CALL_NOTE in detail_prompt

    no_detail_config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=False,
    )
    no_detail_prompt = get_universal_system_prompt(*_reorder(no_detail_config))
    assert DUPLICATE_TOOL_CALL_NOTE not in no_detail_prompt
