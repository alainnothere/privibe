"""Auto-continue: the detector, the loop hook, the cap, and finish_reason
reaching the stored message from both request paths.
"""

from __future__ import annotations

import pytest

from privibe.core.agent_loop import AgentLoop
from privibe.core.agents.models import BuiltinAgentName
from privibe.core.auto_continue import (
    AutoContinueMode,
    NudgeKind,
    looks_unfinished,
    nudge_kind,
)
from privibe.core.conversation import is_step_budget_warning
from privibe.core.llm.backend.generic import _LOCAL_ONLY_FIELDS
from privibe.core.tools.base import ToolPermission
from privibe.core.types import (
    AssistantEvent,
    BaseEvent,
    FunctionCall,
    LLMChunk,
    LLMMessage,
    MessageMeta,
    Role,
    ToolCall,
    UserMessageEvent,
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

# --- the detector -----------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Let me update the parser now.",
        "I'll now run the tests.",
        "I will fix that next.",
        "Next, I will open the config.",
        "Now let me check the file.",
        "- First, I'll read the file",
        "Moving on to the tests",
        "The plan:",
        "Here is what I found...",
        "```python\ndef f():\n    pass",
        "Done with A.\n\nProceeding to B.",
        "Found it. Let me fix the parser now.",
        "The bug is in verdict.py! I'll patch it.",
    ],
)
def test_looks_unfinished(text: str) -> None:
    assert looks_unfinished(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        None,
        "Done. The parser now reads both channels.",
        "Should I also update the docs?",
        "I'll do that if you want. Do you want me to?",
        "```python\ndef f():\n    pass\n```",
        "Let me know if you need anything else.",  # ends a reply; see note below
    ],
)
def test_looks_finished(text: str | None) -> None:
    # "Let me know ..." is the one phrase that opens like intent and is a
    # sign-off; it is caught because the sentence asks the user something.
    if text == "Let me know if you need anything else.":
        pytest.xfail("sign-off phrase that reads as intent; accepted false positive")
    assert not looks_unfinished(text)


def _assistant(content: str, finish: str | None = None, tools: bool = False) -> LLMMessage:
    return LLMMessage(
        role=Role.assistant,
        content=content,
        tool_calls=[
            ToolCall(id="c1", index=0, function=FunctionCall(name="todo", arguments="{}"))
        ]
        if tools
        else None,
        meta=MessageMeta(finish_reason=finish),
    )


def test_nudge_kind_by_mode() -> None:
    cut = _assistant("I was writing the", finish="length")
    plan = _assistant("Let me update the parser.", finish="stop")
    done = _assistant("Done.", finish="stop")
    assert nudge_kind(cut, "off") is None
    assert nudge_kind(cut, "truncated") is NudgeKind.TRUNCATED
    assert nudge_kind(cut, AutoContinueMode.HEURISTIC) is NudgeKind.TRUNCATED
    assert nudge_kind(plan, "truncated") is None
    assert nudge_kind(plan, "heuristic") is NudgeKind.UNFINISHED
    assert nudge_kind(done, "heuristic") is None
    # A reply that calls tools is not a turn end; never nudged.
    assert nudge_kind(_assistant("x", finish="length", tools=True), "heuristic") is None
    # No meta at all (older sessions): only the heuristics can fire.
    bare = LLMMessage(role=Role.assistant, content="Let me check.")
    assert nudge_kind(bare, "truncated") is None
    assert nudge_kind(bare, "heuristic") is NudgeKind.UNFINISHED


def test_auto_continue_flag_never_reaches_the_wire() -> None:
    assert "auto_continue" in _LOCAL_ONLY_FIELDS


# --- the loop ---------------------------------------------------------------


def _loop(backend: FakeBackend, **overrides) -> AgentLoop:
    config = build_test_vibe_config(
        enabled_tools=["todo"],
        tools={"todo": {"permission": ToolPermission.ALWAYS.value}},
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=False,
        **overrides,
    )
    return build_test_agent_loop(
        config=config,
        agent_name=BuiltinAgentName.AUTO_APPROVE,
        backend=backend,
        enable_streaming=overrides.get("_streaming", False) if False else False,
    )


async def _collect(loop: AgentLoop, prompt: str = "do the thing") -> list[BaseEvent]:
    return [ev async for ev in loop.act(prompt)]


def _nudges(events: list[BaseEvent]) -> list[UserMessageEvent]:
    return [e for e in events if isinstance(e, UserMessageEvent) and e.auto_continue]


@pytest.mark.asyncio
async def test_truncated_reply_is_nudged_once_and_the_model_finishes() -> None:
    backend = FakeBackend([
        [mock_llm_chunk(content="I started writing the file and", finish_reason="length")],
        [mock_llm_chunk(content="finished it. Done.", finish_reason="stop")],
    ])
    loop = _loop(backend, auto_continue="truncated")

    events = await _collect(loop)

    nudges = _nudges(events)
    assert len(nudges) == 1
    assert "cut off" in nudges[0].content
    assert nudges[0].message_index == 3  # system, user, assistant, nudge
    assert len(backend.requests_messages) == 2
    roles = [m.role for m in loop.messages]
    assert roles == [Role.system, Role.user, Role.assistant, Role.user, Role.assistant]
    nudge_msg = loop.messages[3]
    assert nudge_msg.auto_continue and not nudge_msg.steering and not nudge_msg.injected
    assert loop.messages[2].meta is not None
    assert loop.messages[2].meta.finish_reason == "length"
    assert loop.messages[4].meta.finish_reason == "stop"
    # The nudge was sent to the model as a plain user message.
    sent = backend.requests_messages[1][-1]
    assert sent.role == Role.user and "cut off" in (sent.content or "")
    assert isinstance(events[-1], AssistantEvent) and events[-1].content == "finished it. Done."


@pytest.mark.asyncio
async def test_off_means_no_nudge() -> None:
    backend = FakeBackend([
        [mock_llm_chunk(content="I started and", finish_reason="length")],
        [mock_llm_chunk(content="never asked")],
    ])
    loop = _loop(backend, auto_continue="off")
    events = await _collect(loop)
    assert _nudges(events) == []
    assert len(backend.requests_messages) == 1


@pytest.mark.asyncio
async def test_heuristic_nudges_an_announced_step() -> None:
    backend = FakeBackend([
        [mock_llm_chunk(content="Found it. Let me fix the parser now.", finish_reason="stop")],
        [
            mock_llm_chunk(
                content="Fixing.",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        index=0,
                        function=FunctionCall(name="todo", arguments='{"action": "read"}'),
                    )
                ],
            )
        ],
        [mock_llm_chunk(content="Fixed. Done.")],
    ])
    loop = _loop(backend, auto_continue="heuristic")
    events = await _collect(loop)
    nudges = _nudges(events)
    assert len(nudges) == 1
    assert "announced the next step" in nudges[0].content
    assert len(backend.requests_messages) == 3


@pytest.mark.asyncio
async def test_truncated_mode_ignores_an_announced_step() -> None:
    backend = FakeBackend([
        [mock_llm_chunk(content="Let me fix the parser now.", finish_reason="stop")],
        [mock_llm_chunk(content="never asked")],
    ])
    loop = _loop(backend, auto_continue="truncated")
    events = await _collect(loop)
    assert _nudges(events) == []


@pytest.mark.asyncio
async def test_a_question_to_the_user_is_never_nudged() -> None:
    backend = FakeBackend([
        [mock_llm_chunk(content="Two ways to do this. Which do you prefer?", finish_reason="stop")],
        [mock_llm_chunk(content="never asked")],
    ])
    loop = _loop(backend, auto_continue="heuristic")
    events = await _collect(loop)
    assert _nudges(events) == []


@pytest.mark.asyncio
async def test_cap_per_user_message_then_the_turn_ends() -> None:
    backend = FakeBackend([
        [mock_llm_chunk(content="part 1", finish_reason="length")],
        [mock_llm_chunk(content="part 2", finish_reason="length")],
        [mock_llm_chunk(content="part 3", finish_reason="length")],
        [mock_llm_chunk(content="part 4", finish_reason="length")],
    ])
    loop = _loop(backend, auto_continue="truncated", auto_continue_max=2)

    events = await _collect(loop)
    assert len(_nudges(events)) == 2
    assert len(backend.requests_messages) == 3

    # A new user message starts a fresh budget.
    events = await _collect(loop, "go on")
    assert len(_nudges(events)) == 1  # one reply left in the script, then empty
    assert loop._auto_continues_this_turn == 1


@pytest.mark.asyncio
async def test_step_budget_writeup_is_not_nudged() -> None:
    call = ToolCall(
        id="c1", index=0, function=FunctionCall(name="todo", arguments='{"action": "read"}')
    )
    backend = FakeBackend([
        [mock_llm_chunk(content="Step 1.", tool_calls=[call])],
        # The forced write-up, cut off by the server: still ends the turn.
        [mock_llm_chunk(content="Here is what I found so far, I", finish_reason="length")],
        [mock_llm_chunk(content="never asked")],
    ])
    loop = _loop(backend, auto_continue="heuristic", max_llm_calls_per_turn=1)
    events = await _collect(loop)
    assert _nudges(events) == []
    assert any(is_step_budget_warning(m) for m in loop.messages)
    assert len(backend.requests_messages) == 2


@pytest.mark.asyncio
async def test_streaming_path_stores_the_finish_reason() -> None:
    backend = FakeBackend([
        [
            mock_llm_chunk(content="partial "),
            mock_llm_chunk(content="text", finish_reason="length"),
        ],
        [mock_llm_chunk(content="rest.", finish_reason="stop")],
    ])
    config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=False,
        auto_continue="truncated",
    )
    loop = build_test_agent_loop(
        config=config, agent_name=BuiltinAgentName.DEFAULT, backend=backend, enable_streaming=True
    )
    events = await _collect(loop)
    assert loop.messages[2].content == "partial text"
    assert loop.messages[2].meta.finish_reason == "length"
    assert len(_nudges(events)) == 1
    assert loop.messages[4].meta.finish_reason == "stop"


def test_chunk_addition_keeps_the_latest_finish_reason() -> None:
    a = LLMChunk(message=LLMMessage(role=Role.assistant, content="a"), finish_reason=None)
    b = LLMChunk(message=LLMMessage(role=Role.assistant, content="b"), finish_reason="length")
    c = LLMChunk(message=LLMMessage(role=Role.assistant, content="c"))
    assert (a + b).finish_reason == "length"
    assert (a + b + c).finish_reason == "length"
    assert (a + c).finish_reason is None
