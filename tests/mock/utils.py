from __future__ import annotations

from collections.abc import AsyncGenerator
import json

from privibe.core.types import (
    LLMChunk,
    LLMMessage,
    LLMUsage,
    Role,
    ToolCall,
    ToolStreamEvent,
)

MOCK_DATA_ENV_VAR = "VIBE_MOCK_LLM_DATA"


def mock_llm_chunk(
    content: str = "Hello!",
    reasoning_content: str | None = None,
    role: Role = Role.assistant,
    tool_calls: list[ToolCall] | None = None,
    name: str | None = None,
    tool_call_id: str | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    # Fixed, server-reported speeds so tests are deterministic. Without these the
    # agent loop falls back to a wall-clock estimate (completion_tokens /
    # elapsed), which varies run-to-run and breaks exact-match UI snapshots.
    tokens_per_second: float = 50.0,
    prompt_tokens_per_second: float = 100.0,
) -> LLMChunk:
    message = LLMMessage(
        role=role,
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
        name=name,
        tool_call_id=tool_call_id,
    )
    return LLMChunk(
        message=message,
        usage=LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_per_second=tokens_per_second,
            prompt_tokens_per_second=prompt_tokens_per_second,
        ),
    )


def get_mocking_env(mock_chunks: list[LLMChunk] | None = None) -> dict[str, str]:
    if mock_chunks is None:
        mock_chunks = [mock_llm_chunk()]

    mock_data = [LLMChunk.model_dump(mock_chunk) for mock_chunk in mock_chunks]

    return {MOCK_DATA_ENV_VAR: json.dumps(mock_data, ensure_ascii=False)}


async def collect_result[T](gen: AsyncGenerator[ToolStreamEvent | T, None]) -> T:
    result = None
    async for item in gen:
        if not isinstance(item, ToolStreamEvent):
            result = item
    if result is None:
        raise RuntimeError("Generator did not yield a result")
    return result
