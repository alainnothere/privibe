"""finish_reason lifted out of each dialect, normalised to stop / length /
tool_calls.
"""

from __future__ import annotations

from privibe.core.config import ProviderConfig
from privibe.core.llm.backend.anthropic import AnthropicAdapter, AnthropicMapper
from privibe.core.llm.backend.finish import anthropic_finish_reason, finish_reason_of
from privibe.core.llm.backend.generic import OpenAIAdapter
from privibe.core.llm.backend.reasoning_adapter import ReasoningAdapter


def _provider(backend: str = "openai") -> ProviderConfig:
    return ProviderConfig(name="p", backend=backend, api_base="http://x", api_key="k")


def test_openai_non_stream_and_stream_shapes() -> None:
    assert finish_reason_of({"choices": [{"message": {}, "finish_reason": "length"}]}) == "length"
    assert finish_reason_of({"choices": [{"delta": {"content": "x"}, "finish_reason": None}]}) is None
    assert finish_reason_of({"choices": [{"delta": {}, "finish_reason": "stop"}]}) == "stop"
    assert finish_reason_of({"usage": {}}) is None
    assert finish_reason_of({"choices": []}) is None


def test_openai_adapter_puts_it_on_the_chunk() -> None:
    data = {
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    assert OpenAIAdapter().parse_response(data, _provider()).finish_reason == "length"
    delta = {"choices": [{"delta": {"content": "h"}, "finish_reason": None}]}
    assert OpenAIAdapter().parse_response(delta, _provider()).finish_reason is None


def test_reasoning_adapter_puts_it_on_the_chunk() -> None:
    data = {
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    assert ReasoningAdapter().parse_response(data, _provider("reasoning")).finish_reason == "tool_calls"


def test_anthropic_mapping() -> None:
    assert anthropic_finish_reason("end_turn") == "stop"
    assert anthropic_finish_reason("stop_sequence") == "stop"
    assert anthropic_finish_reason("max_tokens") == "length"
    assert anthropic_finish_reason("tool_use") == "tool_calls"
    assert anthropic_finish_reason("refusal") == "refusal"
    assert anthropic_finish_reason(None) is None
    assert anthropic_finish_reason("") is None


def test_anthropic_mapper_non_stream_and_message_delta() -> None:
    mapper = AnthropicMapper()
    data = {
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    assert mapper.parse_response(data).finish_reason == "length"

    delta_only = {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}}
    chunk, _ = mapper._handle_message_delta(delta_only, 0)
    assert chunk is not None and chunk.finish_reason == "stop"

    nothing = {"type": "message_delta", "delta": {}, "usage": {}}
    chunk, _ = mapper._handle_message_delta(nothing, 0)
    assert chunk is None


def test_anthropic_adapter_streaming_delta_carries_it() -> None:
    adapter = AnthropicAdapter()
    delta = {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 3}}
    chunk = adapter._parse_message_delta(delta)
    assert chunk.finish_reason == "length"
    assert chunk.usage is not None and chunk.usage.completion_tokens == 3
    bare = {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}
    assert adapter._parse_message_delta(bare).finish_reason == "stop"
