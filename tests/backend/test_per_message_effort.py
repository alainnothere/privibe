"""Per-message reasoning_effort: field persistence, merge, adapter wire
format, knob cycling, and effort inheritance.

The effort value is stored on the message forever (it is rendered into the
KV-cache prefix by the llama.cpp chat template), and only providers that
opt in via per_message_reasoning_effort ever see it on the wire.
"""

from __future__ import annotations

import json

from privibe.cli.commands import effort_cycle_notice
from privibe.core.agent_loop import AgentLoop
from privibe.core.config import ProviderConfig, cycle_reasoning_effort
from privibe.core.llm.backend.generic import OpenAIAdapter
from privibe.core.llm.message_utils import merge_consecutive_user_messages
from privibe.core.types import LLMMessage, Role


def _provider(per_message_effort: bool) -> ProviderConfig:
    return ProviderConfig(
        name="llamacpp",
        api_base="http://localhost:8089/v1",
        per_message_reasoning_effort=per_message_effort,
    )


def _payload(provider: ProviderConfig, messages: list[LLMMessage]) -> dict:
    req = OpenAIAdapter().prepare_request(
        model_name="m",
        messages=messages,
        temperature=0.2,
        tools=None,
        max_tokens=None,
        tool_choice=None,
        enable_streaming=False,
        provider=provider,
    )
    return json.loads(req.body)


CONVERSATION = [
    LLMMessage(role=Role.system, content="sys"),
    LLMMessage(role=Role.user, content="q1", reasoning_effort="xhigh"),
    LLMMessage(role=Role.assistant, content="a1", reasoning_content="think"),
    LLMMessage(role=Role.user, content="q2", reasoning_effort="low"),
]


# ---------------------------------------------------------------------------
# Message model / persistence
# ---------------------------------------------------------------------------


def test_effort_round_trips_through_serialized_message() -> None:
    msg = LLMMessage(role=Role.user, content="hi", reasoning_effort="xhigh")
    line = json.dumps(msg.model_dump(exclude_none=True, mode="json"))
    restored = LLMMessage.model_validate(json.loads(line))
    assert restored.reasoning_effort == "xhigh"


def test_effort_absent_is_omitted_from_serialization() -> None:
    msg = LLMMessage(role=Role.user, content="hi")
    assert "reasoning_effort" not in msg.model_dump(exclude_none=True)


def test_merge_keeps_effort_from_either_side() -> None:
    real_then_injected = merge_consecutive_user_messages([
        LLMMessage(role=Role.user, content="real", reasoning_effort="low"),
        LLMMessage(role=Role.user, content="injected", injected=True),
    ])
    assert len(real_then_injected) == 1
    assert real_then_injected[0].reasoning_effort == "low"

    injected_then_real = merge_consecutive_user_messages([
        LLMMessage(role=Role.user, content="injected", injected=True),
        LLMMessage(role=Role.user, content="real", reasoning_effort="xhigh"),
    ])
    assert len(injected_then_real) == 1
    assert injected_then_real[0].reasoning_effort == "xhigh"


# ---------------------------------------------------------------------------
# Adapter wire format
# ---------------------------------------------------------------------------


def test_adapter_emits_effort_and_kwarg_when_provider_opted_in() -> None:
    body = _payload(_provider(per_message_effort=True), CONVERSATION)
    efforts = [m.get("reasoning_effort") for m in body["messages"]]
    assert efforts == [None, "xhigh", None, "low"]
    assert body["chat_template_kwargs"] == {"per_message_reasoning_effort": True}


def test_adapter_strips_effort_for_default_provider() -> None:
    body = _payload(_provider(per_message_effort=False), CONVERSATION)
    assert all("reasoning_effort" not in m for m in body["messages"])
    assert "chat_template_kwargs" not in body


def test_kwarg_not_sent_while_conversation_unstamped() -> None:
    # A conversation that never used /effort must render exactly as stock,
    # keeping its cache, even on an opted-in provider.
    body = _payload(
        _provider(per_message_effort=True),
        [LLMMessage(role=Role.user, content="plain")],
    )
    assert "chat_template_kwargs" not in body


def test_kwarg_appears_with_first_stamp_and_is_replay_stable() -> None:
    # Kwarg presence is a pure function of the stored conversation: once any
    # message is stamped it is sent on every request, so a resumed
    # conversation renders byte-identically.
    provider = _provider(per_message_effort=True)
    stamped = [
        LLMMessage(role=Role.user, content="q1"),
        LLMMessage(role=Role.assistant, content="a1"),
        LLMMessage(role=Role.user, content="q2", reasoning_effort="low"),
    ]
    body = _payload(provider, stamped)
    assert body["chat_template_kwargs"] == {"per_message_reasoning_effort": True}
    replayed = [LLMMessage.model_validate(m.model_dump()) for m in stamped]
    assert _payload(provider, replayed) == body


# ---------------------------------------------------------------------------
# Knob cycling and effort inheritance
# ---------------------------------------------------------------------------


def test_cycle_order() -> None:
    seen = [cycle_reasoning_effort(None)]
    for _ in range(4):
        seen.append(cycle_reasoning_effort(seen[-1]))
    assert seen == ["low", "medium", "xhigh", "off", "low"]


def _bare_loop(messages: list[LLMMessage], override: str | None) -> AgentLoop:
    loop = AgentLoop.__new__(AgentLoop)
    loop._reasoning_effort_override = override
    loop.messages = messages  # current_reasoning_effort only slices/iterates
    return loop


def test_current_effort_inherits_last_stamped_user_message() -> None:
    loop = _bare_loop(list(CONVERSATION), override=None)
    assert loop.current_reasoning_effort() == "low"


def test_current_effort_none_when_history_unstamped() -> None:
    loop = _bare_loop(
        [LLMMessage(role=Role.user, content="plain")], override=None
    )
    assert loop.current_reasoning_effort() is None


def test_current_effort_override_beats_history() -> None:
    loop = _bare_loop(list(CONVERSATION), override="xhigh")
    assert loop.current_reasoning_effort() == "xhigh"


def test_current_effort_off_disables_stamping_despite_history() -> None:
    loop = _bare_loop(list(CONVERSATION), override="off")
    assert loop.current_reasoning_effort() is None


# ---------------------------------------------------------------------------
# /effort cycle notice
# ---------------------------------------------------------------------------


def test_notice_off_has_no_warnings() -> None:
    notice = effort_cycle_notice("off", "llamacpp", False, False)
    assert "off" in notice
    assert "\n" not in notice


def test_notice_warns_when_provider_not_sending() -> None:
    notice = effort_cycle_notice("low", "openrouter", False, False)
    assert "Stamping locally only" in notice
    assert "openrouter" in notice
    assert "per_message_reasoning_effort" in notice


def test_notice_mentions_companion_build_once() -> None:
    first = effort_cycle_notice("low", "llamacpp", True, False)
    assert "companion llama-server" in first
    assert "silently ignores" in first
    later = effort_cycle_notice("medium", "llamacpp", True, True)
    assert "companion" not in later
    assert "\n" not in later
