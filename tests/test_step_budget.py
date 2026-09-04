"""StepBudgetMiddleware wired into the agent loop.

A turn gets max_llm_calls_per_turn LLM calls. The next call is a write-up
request with tools suspended; a tool call in that reply is refused and the
turn is stopped, a text reply ends the turn normally.
"""

from __future__ import annotations

import pytest

from privibe.cli.commands import CommandRegistry
from privibe.core.agent_loop import AgentLoop
from privibe.core.agents.models import BuiltinAgentName
from privibe.core.config import cycle_llm_calls_per_turn
from privibe.core.conversation import is_step_budget_warning
from privibe.core.tools.base import ToolPermission
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


def _read_call(call_id: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=0,
        function=FunctionCall(name="todo", arguments='{"action": "read"}'),
    )


def _write_call(call_id: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=0,
        function=FunctionCall(
            name="todo", arguments='{"action": "write", "todos": []}'
        ),
    )


def _alternating_calls(n: int) -> list[list]:
    """n tool-calling responses that never repeat the previous call."""
    return [
        [
            mock_llm_chunk(
                content=f"Step {i}.",
                tool_calls=[_read_call(f"c{i}") if i % 2 else _write_call(f"c{i}")],
            )
        ]
        for i in range(1, n + 1)
    ]


def _loop(backend: FakeBackend, budget: int) -> AgentLoop:
    config = build_test_vibe_config(
        enabled_tools=["todo"],
        tools={"todo": {"permission": ToolPermission.ALWAYS.value}},
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=False,
        max_llm_calls_per_turn=budget,
    )
    return build_test_agent_loop(
        config=config, agent_name=BuiltinAgentName.AUTO_APPROVE, backend=backend
    )


async def _collect(agent_loop: AgentLoop, prompt: str) -> list[BaseEvent]:
    return [ev async for ev in agent_loop.act(prompt)]


def _stops(events: list[BaseEvent]) -> list[AssistantEvent]:
    return [
        e for e in events if isinstance(e, AssistantEvent) and e.stopped_by_middleware
    ]


@pytest.mark.asyncio
async def test_model_that_keeps_calling_tools_is_stopped_after_writeup_call() -> None:
    backend = FakeBackend(_alternating_calls(10))
    agent_loop = _loop(backend, budget=2)

    events = await _collect(agent_loop, "Audit everything")

    # Two budgeted calls plus the write-up call; the fourth never happens.
    assert len(backend.requests_messages) == 3
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert [e.skipped for e in results] == [False, False, True]
    assert "step budget" in (results[2].skip_reason or "").lower()
    assert agent_loop.stats.tool_calls_succeeded == 2
    assert agent_loop.stats.tool_calls_rejected == 1

    # The write-up request went in as an injected user message after the
    # second tool response and before the third assistant message.
    roles = [(m.role, m.injected) for m in agent_loop.messages]
    assert roles[-4:] == [
        (Role.tool, False),
        (Role.user, True),
        (Role.assistant, False),
        (Role.tool, False),
    ]
    injected = [m for m in agent_loop.messages if m.role == Role.user and m.injected]
    assert len(injected) == 1
    assert "Step budget spent" in (injected[0].content or "")
    assert "2 tool-calling steps" in (injected[0].content or "")

    stops = _stops(events)
    assert len(stops) == 1
    assert "Step budget spent" in stops[0].content


@pytest.mark.asyncio
async def test_model_that_writes_up_ends_turn_normally() -> None:
    backend = FakeBackend(
        _alternating_calls(2) + [[mock_llm_chunk(content="Here is what I found.")]]
    )
    agent_loop = _loop(backend, budget=2)

    events = await _collect(agent_loop, "Audit everything")

    assert len(backend.requests_messages) == 3
    assert _stops(events) == []
    assert agent_loop.messages[-1].role == Role.assistant
    assert agent_loop.messages[-1].content == "Here is what I found."
    assert agent_loop.stats.tool_calls_rejected == 0


@pytest.mark.asyncio
async def test_new_user_message_grants_a_fresh_budget() -> None:
    backend = FakeBackend(
        _alternating_calls(2)
        + [[mock_llm_chunk(content="Round one.")]]
        + _alternating_calls(2)
        + [[mock_llm_chunk(content="Round two.")]]
    )
    agent_loop = _loop(backend, budget=2)

    first = await _collect(agent_loop, "Go")
    second = await _collect(agent_loop, "Continue")

    assert _stops(first) == [] and _stops(second) == []
    assert len(backend.requests_messages) == 6
    assert agent_loop.stats.tool_calls_succeeded == 4
    injected = [m for m in agent_loop.messages if m.role == Role.user and m.injected]
    assert len(injected) == 2


@pytest.mark.asyncio
async def test_zero_budget_disables_the_cap() -> None:
    backend = FakeBackend(_alternating_calls(8) + [[mock_llm_chunk(content="Done.")]])
    agent_loop = _loop(backend, budget=0)

    events = await _collect(agent_loop, "Go")

    assert len(backend.requests_messages) == 9
    assert _stops(events) == []
    assert agent_loop.stats.tool_calls_succeeded == 8
    assert not any(m.injected for m in agent_loop.messages)


def test_default_budget_is_thirty() -> None:
    assert build_test_vibe_config().max_llm_calls_per_turn == 30


# --- /llm-calls-per-turn plumbing ------------------------------------------


def test_command_registered() -> None:
    cmd = CommandRegistry().find_command("/llm-calls-per-turn")
    assert cmd is not None
    assert cmd.handler == "_select_llm_calls_per_turn"
    assert cmd.takes_args


def test_command_can_be_excluded() -> None:
    registry = CommandRegistry(excluded_commands=["llm_calls_per_turn"])
    assert registry.find_command("/llm-calls-per-turn") is None


def test_cycle_rotates_30_60_90() -> None:
    assert cycle_llm_calls_per_turn(30) == 60
    assert cycle_llm_calls_per_turn(60) == 90
    assert cycle_llm_calls_per_turn(90) == 30
    assert cycle_llm_calls_per_turn(45) == 30
    assert cycle_llm_calls_per_turn(10, [10, 20]) == 20


def test_options_default_and_sanitized() -> None:
    assert build_test_vibe_config().llm_calls_per_turn_options == [30, 60, 90]
    assert build_test_vibe_config(
        llm_calls_per_turn_options=[15, 0, "x", 15, 45]
    ).llm_calls_per_turn_options == [15, 45]
    assert build_test_vibe_config(
        llm_calls_per_turn_options=[]
    ).llm_calls_per_turn_options == [30, 60, 90]


# ---------------------------------------------------------------------------
# A cancelled write-up must not haunt the next turn (session e29828fd)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interrupted_writeup_warning_is_dropped_before_the_next_turn() -> None:
    """Esc during the write-up leaves the warning as the tail. The next user
    message must not land right after "do not call any tool now".
    """
    backend = FakeBackend(
        _alternating_calls(2)
        + [[mock_llm_chunk(content="Round one.")]]
        + _alternating_calls(1)
        + [[mock_llm_chunk(content="Round two.")]]
    )
    agent_loop = _loop(backend, budget=2)

    await _collect(agent_loop, "Go")
    # Simulate the interrupt: drop the write-up reply so the warning is the tail.
    agent_loop.messages.rewind(1)
    assert is_step_budget_warning(agent_loop.messages[-1])

    second = await _collect(agent_loop, "Do it")

    assert _stops(second) == []
    assert agent_loop.stats.tool_calls_succeeded == 3
    assert not any(is_step_budget_warning(m) for m in agent_loop.messages)
    # The new user message follows the last tool response directly, and the
    # turn's checkpoint points at it (the drop happened before the checkpoint).
    do_it = next(i for i, m in enumerate(agent_loop.messages) if m.content == "Do it")
    assert agent_loop.messages[do_it - 1].role == Role.tool
    assert agent_loop.rewind_manager._checkpoints[-1].message_index == do_it
    # What the model saw on the second turn: no warning anywhere near "Do it".
    sent = backend.requests_messages[-2]
    sent_do_it = next(i for i, m in enumerate(sent) if m.content == "Do it")
    assert not is_step_budget_warning(sent[sent_do_it - 1])


@pytest.mark.asyncio
async def test_resumed_session_drops_a_dangling_writeup_warning(tmp_path) -> None:
    """Quit while the warning is the tail, resume: restore() pops it before
    pushing the context refresh, so the model never sees it as a live order.
    """
    from unittest.mock import MagicMock

    from privibe.core.agents.models import AgentProfile, AgentSafety
    from privibe.core.config import SessionLoggingConfig
    from privibe.core.conversation import ConversationList
    from privibe.core.middleware import step_budget_summary_request
    from privibe.core.session.session_logger import SessionLogger
    from privibe.core.tools.manager import ToolManager
    from privibe.core.types import AgentStats, LLMMessage

    conv = ConversationList(tools_provider=list)
    conv.add(LLMMessage(role=Role.system, content="SYSTEM"))
    conv.add(LLMMessage(role=Role.user, content="Go"))
    conv.add(LLMMessage(role=Role.assistant, content="Working."))
    conv.add(
        LLMMessage(
            role=Role.user, content=step_budget_summary_request(2), injected=True
        )
    )
    assert is_step_budget_warning(conv[-1])

    logger = SessionLogger(
        SessionLoggingConfig(
            save_dir=str(tmp_path / "s"), session_prefix="t", enabled=True
        ),
        "resume-ghost",
    )
    profile = AgentProfile(
        name="t",
        display_name="T",
        description="t",
        safety=AgentSafety.NEUTRAL,
        overrides={},
    )
    tool_manager = MagicMock(spec=ToolManager)
    tool_manager.available_tools = {}
    await logger.save_interaction(
        conv, AgentStats(), build_test_vibe_config(), tool_manager, profile
    )

    resumed = ConversationList(tools_provider=list)
    resumed.restore(logger.session_dir)

    assert resumed[-1].role == Role.assistant
    assert not any(is_step_budget_warning(m) for m in resumed)


def test_step_budget_prompt_opens_with_the_marker() -> None:
    """The prompt file is editable; the drop keys off its first words."""
    from privibe.core.conversation import STEP_BUDGET_MARKER
    from privibe.core.middleware import step_budget_summary_request
    from privibe.core.utils import VIBE_WARNING_TAG

    text = step_budget_summary_request(30)
    assert text.startswith(f"<{VIBE_WARNING_TAG}>{STEP_BUDGET_MARKER}")
    assert "do not ask to be granted steps" in text
