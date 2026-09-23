"""Every atom of the bits leaving the house is the same with or without meta.

This does not test that `meta` is excluded. It builds the same conversation
twice, once with every meta field populated on every message and once with
none, runs both through each backend's request builder, and asserts the
serialized bodies are byte-identical. If a backend ever starts leaking
display data onto the wire, this is the test that fails.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from privibe.core.config import ProviderConfig
from privibe.core.llm.backend.anthropic import AnthropicAdapter
from privibe.core.llm.backend.generic import OpenAIAdapter
from privibe.core.llm.backend.reasoning_adapter import ReasoningAdapter
from privibe.core.types import (
    AvailableFunction,
    AvailableTool,
    FileDiff,
    FunctionCall,
    LLMMessage,
    MessageMeta,
    Role,
    ServerTimings,
    ToolCall,
    ToolOutcome,
)


def _meta() -> MessageMeta:
    return MessageMeta(
        submitted_at=1.0,
        request_sent_at=2.0,
        first_chunk_at=2.5,
        response_done_at=3.0,
        server_timings=ServerTimings(cache_n=1, prompt_n=2, prompt_ms=3.0),
        approval_asked_at=4.0,
        approval_answered_at=5.0,
        tool_started_at=6.0,
        tool_finished_at=7.0,
        tool_outcome=ToolOutcome.success,
        tool_result={"command": "ls", "stdout": "", "stderr": "", "returncode": 0},
        file_diff=FileDiff(path="x", kind="diff", hunks=[[("diff-added", "+")]]),
    )


def _conversation(with_meta: bool) -> list[LLMMessage]:
    meta = _meta() if with_meta else None
    return [
        LLMMessage(role=Role.system, content="sys", meta=meta),
        LLMMessage(role=Role.user, content="run ls", meta=meta, message_id="u1"),
        LLMMessage(
            role=Role.assistant,
            content="",
            reasoning_content="thinking",
            reasoning_signature="sig",
            tool_calls=[
                ToolCall(
                    id="c1",
                    index=0,
                    function=FunctionCall(name="bash", arguments='{"command":"ls"}'),
                )
            ],
            meta=meta,
            message_id="a1",
            reasoning_message_id="r1",
        ),
        LLMMessage(
            role=Role.tool,
            name="bash",
            tool_call_id="c1",
            content="command: ls\nstdout: \nstderr: \nreturncode: 0",
            meta=meta,
        ),
        LLMMessage(role=Role.assistant, content="done", meta=meta, message_id="a2"),
    ]


def _tools() -> list[AvailableTool]:
    return [
        AvailableTool(
            function=AvailableFunction(
                name="bash",
                description="run",
                parameters={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                },
            )
        )
    ]


def _openai_provider() -> ProviderConfig:
    return ProviderConfig(
        name="p", api_base="http://x", api_key_env_var="K", api_style="openai"
    )


def _anthropic_provider() -> ProviderConfig:
    return ProviderConfig(
        name="a",
        api_base="https://api.anthropic.com",
        api_key_env_var="ANTHROPIC_API_KEY",
        api_style="anthropic",
    )


def _body(adapter, provider: ProviderConfig, messages: list[LLMMessage], **kw) -> bytes:
    req = adapter.prepare_request(
        model_name="m",
        messages=messages,
        temperature=0.2,
        tools=_tools(),
        max_tokens=256,
        tool_choice=None,
        enable_streaming=kw.pop("enable_streaming", False),
        provider=provider,
        **kw,
    )
    body = req.body
    return body if isinstance(body, bytes) else str(body).encode()


BACKENDS: list[tuple[str, Callable[[], object], Callable[[], ProviderConfig], dict]] = [
    ("openai", OpenAIAdapter, _openai_provider, {}),
    ("openai-streaming", OpenAIAdapter, _openai_provider, {"enable_streaming": True}),
    (
        "openai-per-message-effort",
        OpenAIAdapter,
        _openai_provider,
        {"wire_per_message_effort": True},
    ),
    ("reasoning", ReasoningAdapter, _openai_provider, {}),
    (
        "reasoning-streaming",
        ReasoningAdapter,
        _openai_provider,
        {"enable_streaming": True},
    ),
    ("anthropic", AnthropicAdapter, _anthropic_provider, {}),
    (
        "anthropic-streaming",
        AnthropicAdapter,
        _anthropic_provider,
        {"enable_streaming": True},
    ),
]


@pytest.mark.parametrize(
    ("name", "adapter_cls", "provider_fn", "kw"), BACKENDS, ids=[b[0] for b in BACKENDS]
)
def test_wire_bytes_identical_with_and_without_meta(
    name: str, adapter_cls, provider_fn, kw: dict
) -> None:
    provider = provider_fn()
    with_meta = _body(adapter_cls(), provider, _conversation(True), **dict(kw))
    without = _body(adapter_cls(), provider, _conversation(False), **dict(kw))
    assert with_meta == without, f"{name}: meta leaked onto the wire"
    assert b"meta" not in with_meta
    assert b"tool_started_at" not in with_meta
