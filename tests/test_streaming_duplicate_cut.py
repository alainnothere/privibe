"""Streaming duplicate-call cut: when the model re-generates a tool call
byte-identical to one it already completed in the same assistant message, the
stream is aborted instead of letting the repetition loop run to its own
timeout (observed in the wild: minutes of identical bash calls at ~2s each).

The cut triggers at the earliest certain moment - when a THIRD call block
starts after two identical complete ones - and the in-flight partial call is
dropped. Execution-level dedup then skips the surviving duplicate with its
instructive reason, which is the model's feedback.
"""

from __future__ import annotations

import pytest

from privibe.core.agent_loop import AgentLoop, _streaming_duplicate_call
from privibe.core.tools.base import ToolPermission
from privibe.core.types import (
    BaseEvent,
    FunctionCall,
    LLMMessage,
    Role,
    ToolCall,
    ToolResultEvent,
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


def make_call(call_id: str, index: int, arguments: str = '{"action": "read"}') -> ToolCall:
    return ToolCall(
        id=call_id, index=index, function=FunctionCall(name="todo", arguments=arguments)
    )


def make_agent_loop(backend: FakeBackend) -> AgentLoop:
    return build_test_agent_loop(
        config=build_test_vibe_config(
            enabled_tools=["todo"],
            tools={"todo": {"permission": ToolPermission.ALWAYS.value}},
            system_prompt_id="tests",
            include_project_context=False,
            include_prompt_detail=False,
        ),
        backend=backend,
        enable_streaming=True,
    )


async def act_and_collect_events(agent_loop: AgentLoop, prompt: str) -> list[BaseEvent]:
    return [ev async for ev in agent_loop.act(prompt)]


# ---------------------------------------------------------------------------
# The helper: judge only complete calls, never the streaming tail
# ---------------------------------------------------------------------------


def _msg(calls: list[ToolCall]) -> LLMMessage:
    return LLMMessage(role=Role.assistant, content="", tool_calls=calls)


def test_no_duplicate_below_three_calls() -> None:
    # Two identical calls with nothing after them: the second may still be
    # streaming, and even complete it is execution-dedup's problem, not a
    # reason to cut a stream that may be about to end anyway.
    assert _streaming_duplicate_call(_msg([make_call("a", 0), make_call("b", 1)])) is None


def test_duplicate_detected_when_third_block_starts() -> None:
    dup = _streaming_duplicate_call(
        _msg([make_call("a", 0), make_call("b", 1), make_call("c", 2, '{"act')])
    )
    assert dup is not None
    assert dup.id == "b"


def test_distinct_complete_calls_do_not_trigger() -> None:
    calls = [
        make_call("a", 0, '{"action": "read"}'),
        make_call("b", 1, '{"action": "write"}'),
        make_call("c", 2, '{"act'),
    ]
    assert _streaming_duplicate_call(_msg(calls)) is None


def test_streaming_tail_is_never_judged() -> None:
    # The last call equals the first but may still be receiving deltas -
    # judging it would cut on a prefix coincidence.
    calls = [
        make_call("a", 0, '{"action": "read"}'),
        make_call("b", 1, '{"action": "write"}'),
        make_call("c", 2, '{"action": "read"}'),
    ]
    assert _streaming_duplicate_call(_msg(calls)) is None


# ---------------------------------------------------------------------------
# The cut, end to end through the streaming agent loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_cut_on_second_identical_call() -> None:
    flood = [
        mock_llm_chunk(content="", tool_calls=[make_call("a", 0)]),
        mock_llm_chunk(content="", tool_calls=[make_call("b", 1)]),
        # Third identical block starting: the cut fires here...
        mock_llm_chunk(content="", tool_calls=[make_call("c", 2)]),
        # ...so the rest of the flood must never be consumed.
        *(
            mock_llm_chunk(content="", tool_calls=[make_call(f"x{i}", 3 + i)])
            for i in range(30)
        ),
    ]
    backend = FakeBackend([flood, [mock_llm_chunk(content="done")]])
    agent = make_agent_loop(backend)

    events = await act_and_collect_events(agent, "go")

    assistant_msg = next(m for m in agent.messages if m.role == Role.assistant)
    calls = assistant_msg.tool_calls or []
    assert [tc.id for tc in calls] == ["a", "b"], (
        "the cut must keep the two complete identical calls and drop the "
        "in-flight third block plus everything after it"
    )

    results = [e for e in events if isinstance(e, ToolResultEvent)]
    skipped = [e for e in results if e.skipped]
    executed = [e for e in results if not e.skipped and e.tool_name == "todo"]
    assert len(executed) == 1
    assert len(skipped) == 1
    assert "Duplicate call not executed" in (skipped[0].skip_reason or "")


@pytest.mark.asyncio
async def test_distinct_calls_stream_uncut() -> None:
    stream = [
        mock_llm_chunk(content="", tool_calls=[make_call("a", 0, '{"action": "read"}')]),
        mock_llm_chunk(
            content="",
            tool_calls=[
                make_call(
                    "b", 1, '{"action": "write", "tasks": [{"id": 1, "content": "x", "status": "pending"}]}'
                )
            ],
        ),
        mock_llm_chunk(content="", tool_calls=[make_call("c", 2, '{"action": "read", "extra": 1}')]),
    ]
    backend = FakeBackend([stream, [mock_llm_chunk(content="done")]])
    agent = make_agent_loop(backend)

    await act_and_collect_events(agent, "go")

    assistant_msg = next(m for m in agent.messages if m.role == Role.assistant)
    assert [tc.id for tc in (assistant_msg.tool_calls or [])] == ["a", "b", "c"]
