"""Frozen driver contract: the request is a pure function of the stored context.

Two pillars:

1. Golden payloads — the exact bytes prepare_request produces for fixed
   conversations, committed under goldens/. Any code change that alters what
   llama.cpp receives for an existing context fails here, one test per
   fixture, naming every serialization that changed. Regenerating goldens
   (UPDATE_GOLDENS=1 pytest ...) is a deliberate migration that voids every
   live KV-cache prefix — never an incidental side effect.

2. The append-invariance law — growing a conversation may only append to
   "messages". Every other payload field, and the serialization of every
   previously sent message, must stay byte-identical, or the llama.cpp
   KV-prefix dies and the whole context reprocesses.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from privibe.core.config import ProviderConfig
from privibe.core.llm.backend.generic import OpenAIAdapter
from privibe.core.types import (
    AvailableFunction,
    AvailableTool,
    FunctionCall,
    LLMMessage,
    Role,
    ToolCall,
)

GOLDEN_DIR = Path(__file__).parent / "goldens"

_DOCTRINE = """
================================================================================
STOP. YOU ARE ABOUT TO BREAK THE KV CACHE. READ THIS BEFORE TOUCHING ANYTHING.
================================================================================

llama.cpp caches the conversation as a token PREFIX. On every request it
re-renders the whole prompt string from the payload and reuses the cache only
up to the first byte that differs from last time. One changed byte at position
N throws away everything after N. A changed byte near the top throws away
EVERYTHING.

This is not hypothetical. On 2026-08-18 a conditionally-sent
chat_template_kwargs flag appeared for the first time 180k tokens into a live
conversation. The rendered prefix diverged at byte ~19. The server kept 1
token and reprocessed 180k at 30-80 tok/s prompt speed: roughly TEN MINUTES
of dead air, on local hardware the user paid for, to answer one message.

THE LAW (settled, not open for debate):
  * The stored context is the single source of truth. Every decision that can
    reach the rendered prompt - flags, kwargs, efforts, tools, system text -
    is written into the context ONCE, at the moment it is made, and never
    rewritten.
  * The driver is a pure serializer: payload = f(stored context). It may
    depend on message bytes and session-birth constants. It may NEVER consult
    conversation-global properties (any(), counts, "is X present"), mutable
    runtime state, or live config. Those flip between requests, and a flip
    anywhere in the payload can rewrite bytes anywhere in the render.
  * Growing a conversation may only APPEND. What was already sent must
    re-serialize byte-identically, forever, including across resume.
  * Tools, system message, modes: whatever was there at session birth is set
    in stone. New capability mid-session arrives as an appended user message
    telling the model about it (tail-append, cache-safe), or you start a new
    session. Worst case for the append: ~1 minute to mention the new tool.
    The alternative you are contemplating costs a full-context reprocess.

WHAT TO DO INSTEAD OF SHIPPING THIS CHANGE:
  * Need new per-request behavior? Encode it in the NEW message (tail), like
    per-message reasoning_effort notes - those are cache-safe by design.
  * Need a new render-affecting constant? Store it in session state at birth;
    old sessions keep their stored value, new sessions get the new one.
  * Genuinely changing the serialization format? That is a MIGRATION: fix the
    code, regenerate goldens with UPDATE_GOLDENS=1, and accept in writing
    (your commit message) that every live KV-cache prefix dies once.

If this failure surprised you, the change you just made is exactly the kind
this suite exists to catch. Do not weaken the test. Fix the change.
================================================================================
"""


def _first_divergence(a: str, b: str) -> str:
    common = 0
    for ca, cb in zip(a, b, strict=False):
        if ca != cb:
            break
        common += 1
    lo = max(0, common - 60)
    return (
        f"first divergence at byte {common} of {len(a)} (was) / {len(b)} (now)\n"
        f"  was: ...{a[lo : common + 120]!r}\n"
        f"  now: ...{b[lo : common + 120]!r}"
    )


DEFAULT_PROVIDER = ProviderConfig(name="llamacpp", api_base="http://localhost:8089/v1")
PER_MSG_PROVIDER = ProviderConfig(
    name="llamacpp",
    api_base="http://localhost:8089/v1",
    per_message_reasoning_effort=True,
)

TOOLS = [
    AvailableTool(
        function=AvailableFunction(
            name="read_file",
            description="Read a file",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
    )
]


def _request(
    provider: ProviderConfig,
    messages: list[LLMMessage],
    tools: list[AvailableTool] | None = None,
    *,
    enable_streaming: bool = False,
    api_key: str | None = None,
    max_tokens: int | None = None,
    tool_choice: Any = None,
    wire_per_message_effort: bool | None = None,
):
    return OpenAIAdapter().prepare_request(
        model_name="m",
        messages=messages,
        temperature=0.2,
        tools=tools,
        max_tokens=max_tokens,
        tool_choice=tool_choice,
        enable_streaming=enable_streaming,
        provider=provider,
        api_key=api_key,
        wire_per_message_effort=wire_per_message_effort,
    )


def _payload(
    provider: ProviderConfig,
    messages: list[LLMMessage],
    tools: list[AvailableTool] | None = None,
) -> dict[str, Any]:
    return json.loads(_request(provider, messages, tools).body)


# ---------------------------------------------------------------------------
# Fixture conversations
# ---------------------------------------------------------------------------


def _basic() -> list[LLMMessage]:
    return [
        LLMMessage(role=Role.system, content="sys"),
        LLMMessage(role=Role.user, content="q1"),
    ]


def _tool_roundtrip() -> list[LLMMessage]:
    return [
        LLMMessage(role=Role.system, content="sys"),
        LLMMessage(role=Role.user, content="read a.py"),
        LLMMessage(
            role=Role.assistant,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    index=0,
                    function=FunctionCall(
                        name="read_file", arguments='{"path": "a.py"}'
                    ),
                )
            ],
        ),
        LLMMessage(
            role=Role.tool,
            name="read_file",
            tool_call_id="call_1",
            content="print('hi')",
        ),
        LLMMessage(role=Role.assistant, content="done"),
    ]


def _thinking_only_turn() -> list[LLMMessage]:
    return [
        LLMMessage(role=Role.system, content="sys"),
        LLMMessage(role=Role.user, content="q1"),
        LLMMessage(role=Role.assistant, reasoning_content="pondering"),
        LLMMessage(role=Role.user, content="q2"),
    ]


def _injected_merge() -> list[LLMMessage]:
    return [
        LLMMessage(role=Role.system, content="sys"),
        LLMMessage(role=Role.user, content="real question"),
        LLMMessage(role=Role.user, content="middleware note", injected=True),
    ]


def _consecutive_assistant() -> list[LLMMessage]:
    return [
        LLMMessage(role=Role.system, content="sys"),
        LLMMessage(role=Role.user, content="q1"),
        LLMMessage(role=Role.assistant, content="a1"),
        LLMMessage(role=Role.assistant, content="a2"),
    ]


def _stamped() -> list[LLMMessage]:
    return [
        LLMMessage(role=Role.system, content="sys"),
        LLMMessage(role=Role.user, content="q1", reasoning_effort="xhigh"),
        LLMMessage(role=Role.assistant, content="a1", reasoning_content="think"),
        LLMMessage(role=Role.user, content="q2", reasoning_effort="low"),
    ]


def _unstamped() -> list[LLMMessage]:
    return [
        LLMMessage(role=Role.system, content="sys"),
        LLMMessage(role=Role.user, content="q1"),
    ]


RENAMED_FIELD_PROVIDER = ProviderConfig(
    name="llamacpp",
    api_base="http://localhost:8089/v1",
    reasoning_field_name="reasoning",
)
STREAM_TOOL_CALLS_PROVIDER = ProviderConfig(
    name="llamacpp", api_base="http://localhost:8089/v1", stream_tool_calls=True
)
MAX_TOOL_CALLS_PROVIDER = ProviderConfig(
    name="llamacpp", api_base="http://localhost:8089/v1", max_tool_calls=3
)

# (name, provider, messages, tools, request_kwargs) - one entry per driver
# branch that can alter the bytes on the wire. If you add a branch to the
# driver, it gets a golden here or it does not ship.
GOLDEN_CASES = [
    ("basic", DEFAULT_PROVIDER, _basic(), None, {}),
    ("basic_streaming", DEFAULT_PROVIDER, _basic(), None, {"enable_streaming": True}),
    ("tools", DEFAULT_PROVIDER, _basic(), TOOLS, {}),
    ("tool_roundtrip", DEFAULT_PROVIDER, _tool_roundtrip(), TOOLS, {}),
    ("thinking_only_turn", DEFAULT_PROVIDER, _thinking_only_turn(), None, {}),
    ("injected_merge", DEFAULT_PROVIDER, _injected_merge(), None, {}),
    ("consecutive_assistant", DEFAULT_PROVIDER, _consecutive_assistant(), None, {}),
    ("effort_stamped", PER_MSG_PROVIDER, _stamped(), None, {}),
    ("effort_stamped_default_provider", DEFAULT_PROVIDER, _stamped(), None, {}),
    ("effort_unstamped_per_msg_provider", PER_MSG_PROVIDER, _unstamped(), None, {}),
    # Session-frozen flag overrides the provider config in both directions.
    (
        "wire_flag_forced_on",
        DEFAULT_PROVIDER,
        _stamped(),
        None,
        {"wire_per_message_effort": True},
    ),
    (
        "wire_flag_forced_off",
        PER_MSG_PROVIDER,
        _stamped(),
        None,
        {"wire_per_message_effort": False},
    ),
    # Reasoning field rename on outbound (provider.reasoning_field_name).
    ("reasoning_field_rename", RENAMED_FIELD_PROVIDER, _stamped(), None, {}),
    # Authorization header from the api key.
    ("auth_header", DEFAULT_PROVIDER, _basic(), None, {"api_key": "sk-golden"}),
    # max_tokens + string tool_choice branches of build_payload.
    (
        "sampling_knobs",
        DEFAULT_PROVIDER,
        _basic(),
        TOOLS,
        {"max_tokens": 512, "tool_choice": "auto"},
    ),
    # tool_choice as a structured object.
    (
        "tool_choice_object",
        DEFAULT_PROVIDER,
        _basic(),
        TOOLS,
        {"tool_choice": TOOLS[0]},
    ),
    # stream_tool_calls provider flag reaches stream_options.
    (
        "streaming_tool_calls",
        STREAM_TOOL_CALLS_PROVIDER,
        _basic(),
        TOOLS,
        {"enable_streaming": True},
    ),
    # Grammar-enforced tool-call cap; sent only when tools are advertised.
    ("max_tool_calls", MAX_TOOL_CALLS_PROVIDER, _basic(), TOOLS, {}),
    ("max_tool_calls_no_tools", MAX_TOOL_CALLS_PROVIDER, _basic(), None, {}),
]


@pytest.mark.parametrize(
    ("name", "provider", "messages", "tools", "request_kwargs"),
    GOLDEN_CASES,
    ids=[c[0] for c in GOLDEN_CASES],
)
def test_golden_payload(
    name: str,
    provider: ProviderConfig,
    messages: list[LLMMessage],
    tools: list[AvailableTool] | None,
    request_kwargs: dict[str, Any],
) -> None:
    req = _request(provider, messages, tools, **request_kwargs)
    actual = {
        "endpoint": req.endpoint,
        "headers": req.headers,
        "body": req.body.decode("utf-8"),
    }
    golden_path = GOLDEN_DIR / f"{name}.json"
    if os.environ.get("UPDATE_GOLDENS"):
        GOLDEN_DIR.mkdir(exist_ok=True)
        golden_path.write_text(json.dumps(actual, indent=2, ensure_ascii=False) + "\n")
    golden = json.loads(golden_path.read_text())
    if actual == golden:
        return
    problems: list[str] = []
    for field in ("endpoint", "headers"):
        if actual[field] != golden[field]:
            problems.append(
                f"{field} changed:\n  was: {golden[field]!r}\n  now: {actual[field]!r}"
            )
    if actual["body"] != golden["body"]:
        problems.append(
            "body changed: " + _first_divergence(golden["body"], actual["body"])
        )
    pytest.fail(
        f"GOLDEN PAYLOAD '{name}' NO LONGER MATCHES: the driver now sends "
        "different bytes to llama.cpp for a context that already exists on "
        "users' disks. Every live conversation serialized under the old bytes "
        "will re-render differently, the KV-prefix match will fail at the "
        "first changed byte, and the entire context reprocesses.\n\n"
        + "\n".join(problems)
        + "\n"
        + _DOCTRINE,
        pytrace=False,
    )


# ---------------------------------------------------------------------------
# The append-invariance law
# ---------------------------------------------------------------------------

LAW_CASES = [
    pytest.param(
        DEFAULT_PROVIDER,
        _basic(),
        [
            LLMMessage(role=Role.assistant, content="a1"),
            LLMMessage(role=Role.user, content="q2"),
        ],
        None,
        id="grow_basic",
    ),
    pytest.param(
        DEFAULT_PROVIDER,
        _tool_roundtrip(),
        [LLMMessage(role=Role.user, content="thanks")],
        TOOLS,
        id="grow_after_tool_roundtrip",
    ),
    pytest.param(
        PER_MSG_PROVIDER,
        _stamped(),
        [
            LLMMessage(role=Role.assistant, content="a2"),
            LLMMessage(role=Role.user, content="q3", reasoning_effort="medium"),
        ],
        None,
        id="grow_already_stamped",
    ),
    pytest.param(
        PER_MSG_PROVIDER,
        _stamped(),
        [
            LLMMessage(role=Role.assistant, content="a2"),
            LLMMessage(role=Role.user, content="q3"),
        ],
        None,
        id="grow_stamped_then_effort_off",
    ),
    pytest.param(
        PER_MSG_PROVIDER,
        [
            LLMMessage(role=Role.system, content="sys"),
            LLMMessage(role=Role.user, content="q1"),
        ],
        [
            LLMMessage(role=Role.assistant, content="a1"),
            LLMMessage(role=Role.user, content="q2", reasoning_effort="low"),
        ],
        None,
        # Historical bug marker: the kwarg used to be derived from
        # any(msg.reasoning_effort), so this exact growth - the first
        # /effort stamp of a conversation - made it appear mid-session and
        # voided a 180k-token prefix. Now the kwarg is a per-session
        # constant and this case must pass forever.
        id="first_stamp_flip",
    ),
]


@pytest.mark.parametrize(("provider", "base", "extension", "tools"), LAW_CASES)
def test_appending_messages_never_rewrites_what_was_already_sent(
    provider: ProviderConfig,
    base: list[LLMMessage],
    extension: list[LLMMessage],
    tools: list[AvailableTool] | None,
) -> None:
    before = _payload(provider, base, tools)
    after = _payload(provider, base + extension, tools)

    before_rest = {k: v for k, v in before.items() if k != "messages"}
    after_rest = {k: v for k, v in after.items() if k != "messages"}
    if before_rest != after_rest:
        changed = sorted(
            set(before_rest) ^ set(after_rest)
            | {
                k
                for k in set(before_rest) & set(after_rest)
                if before_rest[k] != after_rest[k]
            }
        )
        detail = "\n".join(
            f"  '{k}': was {before_rest.get(k, '<absent>')!r} -> now {after_rest.get(k, '<absent>')!r}"
            for k in changed
        )
        pytest.fail(
            "APPEND-INVARIANCE VIOLATED: merely growing the conversation "
            "changed payload field(s) OUTSIDE 'messages':\n"
            f"{detail}\n\n"
            "The chat template may consult these fields anywhere in the "
            "render, including the very top of the system region. A field "
            "that appears, disappears, or changes value between requests can "
            "therefore rewrite prefix bytes 180k tokens away from the message "
            "that triggered it. This exact shape of bug (chat_template_kwargs "
            "derived from any(msg.reasoning_effort)) once reduced a 180k-token "
            "cached prefix to 19 matching characters.\n" + _DOCTRINE,
            pytrace=False,
        )

    sent = before["messages"]
    grown = after["messages"][: len(sent)]
    if grown != sent:
        idx = next(
            i for i, (a, b) in enumerate(zip(sent, grown, strict=False)) if a != b
        )
        pytest.fail(
            "APPEND-INVARIANCE VIOLATED: growing the conversation "
            f"re-serialized already-sent message index {idx}:\n"
            f"  was: {json.dumps(sent[idx], ensure_ascii=False)}\n"
            f"  now: {json.dumps(grown[idx], ensure_ascii=False)}\n\n"
            "That message's bytes were already rendered into the KV-cache "
            "prefix on the previous request. Re-serializing it differently "
            "moves the first mismatching byte back to THIS message's position, "
            "and every token after it - potentially the entire context - "
            "reprocesses at prompt speed. History is frozen: a stored message "
            "is a write-once record, not a view.\n" + _DOCTRINE,
            pytrace=False,
        )
