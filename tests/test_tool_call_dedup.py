from __future__ import annotations

import json

import pytest

from privibe.core.agent_loop import AgentLoop
from privibe.core.agents import AgentManager
from privibe.core.agents.models import BuiltinAgentName
from privibe.core.config import VibeConfig
from privibe.core.skills.manager import SkillManager
from privibe.core.system_prompt import get_universal_system_prompt
from privibe.core.tools.base import ToolPermission
from privibe.core.tools.manager import ToolManager
from privibe.core.types import (
    AssistantEvent,
    BaseEvent,
    FunctionCall,
    Role,
    ToolCall,
    ToolResultEvent,
)
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


def make_bash_config(keywords: list[str] | None = None) -> VibeConfig:
    bash_tool: dict = {"permission": ToolPermission.ALWAYS.value}
    if keywords is not None:
        bash_tool["single_call_keywords"] = keywords
    return build_test_vibe_config(
        enabled_tools=["bash", "todo"],
        tools={
            "bash": bash_tool,
            "todo": {"permission": ToolPermission.ALWAYS.value},
        },
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=False,
    )


def make_bash_tool_call(
    call_id: str, command: str, index: int = 0
) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=index,
        function=FunctionCall(
            name="bash", arguments=json.dumps({"command": command})
        ),
    )


def make_bash_agent_loop(
    backend: FakeBackend, keywords: list[str] | None = None
) -> AgentLoop:
    return build_test_agent_loop(
        config=make_bash_config(keywords),
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
async def test_same_call_in_consecutive_rounds_second_is_refused() -> None:
    # The identical call in the very next assistant message is not run again:
    # its result is already in the context. The model is told so.
    round_one = make_todo_tool_call("call_r1", index=0, arguments='{"action": "read"}')
    round_two = make_todo_tool_call("call_r2", index=0, arguments='{"action":"read"}')
    agent_loop = make_agent_loop(
        backend=FakeBackend([
            [mock_llm_chunk(content="First read.", tool_calls=[round_one])],
            [mock_llm_chunk(content="Second read.", tool_calls=[round_two])],
            [mock_llm_chunk(content="Done.")],
        ])
    )

    events = await act_and_collect_events(agent_loop, "Read twice")

    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert [e.tool_call_id for e in result_events] == ["call_r1", "call_r2"]
    assert not result_events[0].skipped
    assert result_events[1].skipped
    assert "identical to one in your previous message" in (
        result_events[1].skip_reason or ""
    )
    assert agent_loop.stats.tool_calls_succeeded == 1
    assert agent_loop.stats.tool_calls_rejected == 1
    # The refusal reached the model as the tool response, and the turn went on
    # to the final text answer without being stopped.
    r2_msg = next(m for m in agent_loop.messages if m.tool_call_id == "call_r2")
    assert "identical to one in your previous message" in (r2_msg.content or "")
    assert agent_loop.messages[-1].content == "Done."
    assert not any(
        isinstance(e, AssistantEvent) and e.stopped_by_middleware for e in events
    )


@pytest.mark.asyncio
async def test_same_call_in_three_consecutive_rounds_stops_turn() -> None:
    # Refused once, repeated anyway: the turn ends and the prompt goes back to
    # the user. The fourth response is never requested.
    calls = [
        make_todo_tool_call(f"call_r{i}", index=0, arguments='{"action": "read"}')
        for i in range(1, 4)
    ]
    backend = FakeBackend([
        [mock_llm_chunk(content=f"Read {i}.", tool_calls=[tc])]
        for i, tc in enumerate(calls, start=1)
    ] + [[mock_llm_chunk(content="Never requested.")]])
    agent_loop = make_agent_loop(backend=backend)

    events = await act_and_collect_events(agent_loop, "Loop")

    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert [e.skipped for e in result_events] == [False, True, True]
    stops = [e for e in events if isinstance(e, AssistantEvent) and e.stopped_by_middleware]
    assert len(stops) == 1
    assert "identical `todo` call in 3 consecutive messages" in stops[0].content
    assert len(backend.requests_messages) == 3
    assert agent_loop.messages[-1].role == Role.tool


@pytest.mark.asyncio
async def test_same_call_with_another_call_between_runs_both_times() -> None:
    # Only the immediately previous assistant message counts as a repeat.
    read_one = make_todo_tool_call("call_r1", index=0, arguments='{"action": "read"}')
    write = make_todo_tool_call(
        "call_w", index=0, arguments='{"action": "write", "todos": []}'
    )
    read_two = make_todo_tool_call("call_r2", index=0, arguments='{"action": "read"}')
    agent_loop = make_agent_loop(
        backend=FakeBackend([
            [mock_llm_chunk(content="Read.", tool_calls=[read_one])],
            [mock_llm_chunk(content="Write.", tool_calls=[write])],
            [mock_llm_chunk(content="Read again.", tool_calls=[read_two])],
            [mock_llm_chunk(content="Done.")],
        ])
    )

    events = await act_and_collect_events(agent_loop, "Read, write, read")

    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(result_events) == 3
    assert all(not e.skipped for e in result_events)
    assert agent_loop.stats.tool_calls_succeeded == 3
    assert agent_loop.stats.tool_calls_rejected == 0


@pytest.mark.asyncio
async def test_same_call_after_new_user_message_runs_again() -> None:
    # A repeat across turns is the user asking again, not the model looping.
    round_one = make_todo_tool_call("call_r1", index=0, arguments='{"action": "read"}')
    round_two = make_todo_tool_call("call_r2", index=0, arguments='{"action": "read"}')
    agent_loop = make_agent_loop(
        backend=FakeBackend([
            [mock_llm_chunk(content="Read.", tool_calls=[round_one])],
            [mock_llm_chunk(content="Done.")],
            [mock_llm_chunk(content="Read.", tool_calls=[round_two])],
            [mock_llm_chunk(content="Done again.")],
        ])
    )

    first = await act_and_collect_events(agent_loop, "Read")
    second = await act_and_collect_events(agent_loop, "Read again")

    for events in (first, second):
        result_events = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(result_events) == 1
        assert not result_events[0].skipped
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


@pytest.mark.asyncio
async def test_bash_shared_keyword_collapses() -> None:
    # Two bash calls with different commands but a shared configured keyword:
    # the first runs, the second is skipped with a keyword message.
    call_a = make_bash_tool_call("bash_a", "echo alpha deploy.sh", index=0)
    call_b = make_bash_tool_call("bash_b", "echo beta deploy.sh", index=1)
    agent_loop = make_bash_agent_loop(
        backend=FakeBackend([
            [mock_llm_chunk(content="Two deploys.", tool_calls=[call_a, call_b])],
            [mock_llm_chunk(content="Done.")],
        ]),
        keywords=["deploy.sh"],
    )

    events = await act_and_collect_events(agent_loop, "Go")

    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    executed = [e for e in result_events if not e.skipped and e.result is not None]
    skipped = [e for e in result_events if e.skipped]
    assert {e.tool_call_id for e in executed} == {"bash_a"}
    assert len(skipped) == 1
    assert skipped[0].tool_call_id == "bash_b"
    assert skipped[0].skip_reason is not None
    assert "deploy.sh" in skipped[0].skip_reason
    assert "bash_a" in skipped[0].skip_reason
    assert agent_loop.stats.tool_calls_rejected == 1


@pytest.mark.asyncio
async def test_bash_keyword_match_is_case_insensitive() -> None:
    # Keyword "Foo.sh" collapses commands that spell it "foo.sh" and "FOO.SH".
    call_a = make_bash_tool_call("bash_a", "echo run foo.sh", index=0)
    call_b = make_bash_tool_call("bash_b", "echo run FOO.SH", index=1)
    agent_loop = make_bash_agent_loop(
        backend=FakeBackend([
            [mock_llm_chunk(content="Two casings.", tool_calls=[call_a, call_b])],
            [mock_llm_chunk(content="Done.")],
        ]),
        keywords=["Foo.sh"],
    )

    events = await act_and_collect_events(agent_loop, "Go")

    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    executed = [e for e in result_events if not e.skipped and e.result is not None]
    skipped = [e for e in result_events if e.skipped]
    assert {e.tool_call_id for e in executed} == {"bash_a"}
    assert len(skipped) == 1
    assert skipped[0].tool_call_id == "bash_b"
    assert skipped[0].skip_reason is not None
    assert "Foo.sh" in skipped[0].skip_reason
    assert agent_loop.stats.tool_calls_rejected == 1


@pytest.mark.asyncio
async def test_bash_different_keywords_both_run() -> None:
    # Two bash calls each match a different configured keyword: both execute.
    call_a = make_bash_tool_call("bash_a", "echo run deploy.sh", index=0)
    call_b = make_bash_tool_call("bash_b", "echo run build.sh", index=1)
    agent_loop = make_bash_agent_loop(
        backend=FakeBackend([
            [mock_llm_chunk(content="Two scripts.", tool_calls=[call_a, call_b])],
            [mock_llm_chunk(content="Done.")],
        ]),
        keywords=["deploy.sh", "build.sh"],
    )

    events = await act_and_collect_events(agent_loop, "Go")

    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(result_events) == 2
    assert all(not e.skipped for e in result_events)
    assert {e.tool_call_id for e in result_events} == {"bash_a", "bash_b"}
    assert agent_loop.stats.tool_calls_rejected == 0


@pytest.mark.asyncio
async def test_non_bash_calls_with_keyword_text_unaffected() -> None:
    # Non-bash tool calls whose arguments contain the keyword text are not
    # collapsed, even when bash keyword collapsing is configured.
    call_a = make_todo_tool_call(
        "todo_a",
        index=0,
        arguments=json.dumps(
            {"action": "write", "todos": [{"id": "1", "content": "deploy.sh one"}]}
        ),
    )
    call_b = make_todo_tool_call(
        "todo_b",
        index=1,
        arguments=json.dumps(
            {"action": "write", "todos": [{"id": "2", "content": "deploy.sh two"}]}
        ),
    )
    agent_loop = make_bash_agent_loop(
        backend=FakeBackend([
            [mock_llm_chunk(content="Two todos.", tool_calls=[call_a, call_b])],
            [mock_llm_chunk(content="Done.")],
        ]),
        keywords=["deploy.sh"],
    )

    events = await act_and_collect_events(agent_loop, "Go")

    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(result_events) == 2
    assert all(not e.skipped for e in result_events)
    assert {e.tool_call_id for e in result_events} == {"todo_a", "todo_b"}
    assert agent_loop.stats.tool_calls_rejected == 0


@pytest.mark.asyncio
async def test_bash_exact_duplicate_uses_exact_message_not_keyword() -> None:
    # Byte-identical bash calls collapse via the exact-duplicate path, keeping
    # the exact-duplicate message rather than the keyword message.
    call_a = make_bash_tool_call("bash_a", "echo run deploy.sh", index=0)
    call_a_dup = make_bash_tool_call("bash_a_dup", "echo run deploy.sh", index=1)
    agent_loop = make_bash_agent_loop(
        backend=FakeBackend([
            [mock_llm_chunk(content="Same twice.", tool_calls=[call_a, call_a_dup])],
            [mock_llm_chunk(content="Done.")],
        ]),
        keywords=["deploy.sh"],
    )

    events = await act_and_collect_events(agent_loop, "Go")

    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    executed = [e for e in result_events if not e.skipped and e.result is not None]
    skipped = [e for e in result_events if e.skipped]
    assert {e.tool_call_id for e in executed} == {"bash_a"}
    assert len(skipped) == 1
    assert skipped[0].tool_call_id == "bash_a_dup"
    assert skipped[0].skip_reason is not None
    assert "Duplicate call not executed" in skipped[0].skip_reason
    assert agent_loop.stats.tool_calls_rejected == 1


@pytest.mark.asyncio
async def test_bash_default_sentinel_keyword_does_not_collapse() -> None:
    # With the default config (only the inert sentinel keyword), two ordinary
    # differing bash commands both execute.
    call_a = make_bash_tool_call("bash_a", "echo alpha", index=0)
    call_b = make_bash_tool_call("bash_b", "echo beta", index=1)
    agent_loop = make_bash_agent_loop(
        backend=FakeBackend([
            [mock_llm_chunk(content="Two echoes.", tool_calls=[call_a, call_b])],
            [mock_llm_chunk(content="Done.")],
        ]),
    )

    events = await act_and_collect_events(agent_loop, "Go")

    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(result_events) == 2
    assert all(not e.skipped for e in result_events)
    assert {e.tool_call_id for e in result_events} == {"bash_a", "bash_b"}
    assert agent_loop.stats.tool_calls_rejected == 0


@pytest.mark.asyncio
async def test_bash_three_sharing_keyword_only_first_runs() -> None:
    # Three bash calls sharing a keyword: the first executes, the other two are
    # both skipped.
    call_a = make_bash_tool_call("bash_a", "echo one deploy.sh", index=0)
    call_b = make_bash_tool_call("bash_b", "echo two deploy.sh", index=1)
    call_c = make_bash_tool_call("bash_c", "echo three deploy.sh", index=2)
    agent_loop = make_bash_agent_loop(
        backend=FakeBackend([
            [
                mock_llm_chunk(
                    content="Three deploys.", tool_calls=[call_a, call_b, call_c]
                )
            ],
            [mock_llm_chunk(content="Done.")],
        ]),
        keywords=["deploy.sh"],
    )

    events = await act_and_collect_events(agent_loop, "Go")

    result_events = [e for e in events if isinstance(e, ToolResultEvent)]
    executed = [e for e in result_events if not e.skipped and e.result is not None]
    skipped = [e for e in result_events if e.skipped]
    assert {e.tool_call_id for e in executed} == {"bash_a"}
    assert {e.tool_call_id for e in skipped} == {"bash_b", "bash_c"}
    assert agent_loop.stats.tool_calls_rejected == 2


def _managers(config: VibeConfig):
    return (
        ToolManager(lambda: config),
        SkillManager(lambda: config),
        AgentManager(lambda: config),
    )


def _reorder(config: VibeConfig):
    tool_manager, skill_manager, agent_manager = _managers(config)
    return tool_manager, config, skill_manager, agent_manager


def test_system_prompt_omits_duplicate_dedup_note() -> None:
    # The runtime skip notice teaches dedup in-band; the system prompt
    # must not carry a note about it.
    detail_config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=True,
    )
    detail_prompt = get_universal_system_prompt(*_reorder(detail_config))
    assert "Duplicate tool calls" not in detail_prompt
