"""Finish reasons, normalised to the OpenAI words, plus the OpenAI-dialect
message parser the two OpenAI-shaped adapters share.

privibe stores one of three words: "stop" (the model chose to end),
"length" (the server cut it off at the token cap), "tool_calls" (it
stopped to call tools). Other dialects map onto these; anything unknown is
kept verbatim rather than dropped. None means the backend did not say.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from privibe.core.types import LLMMessage

FINISH_STOP = "stop"
FINISH_LENGTH = "length"
FINISH_TOOL_CALLS = "tool_calls"

# Anthropic stop_reason -> the words above.
_ANTHROPIC_FINISH = {
    "end_turn": FINISH_STOP,
    "stop_sequence": FINISH_STOP,
    "max_tokens": FINISH_LENGTH,
    "tool_use": FINISH_TOOL_CALLS,
}


def finish_reason_of(data: dict[str, Any]) -> str | None:
    """The finish_reason of the first choice of an OpenAI-style response, or
    None. Streaming chunks carry it only on the last delta.
    """
    choices = data.get("choices")
    if not choices:
        return None
    reason = choices[0].get("finish_reason")
    return str(reason) if reason else None


def anthropic_finish_reason(stop_reason: Any) -> str | None:
    if not stop_reason:
        return None
    return _ANTHROPIC_FINISH.get(str(stop_reason), str(stop_reason))


def parse_openai_message(
    data: dict[str, Any],
    field_name: str,
    reasoning_from_api: Callable[[dict[str, Any], str], dict[str, Any]],
) -> LLMMessage | None:
    """The message (or delta) of the first choice, with the provider's
    reasoning field folded in by `reasoning_from_api`.
    """
    if data.get("choices"):
        choice = data["choices"][0]
        if "message" in choice:
            return LLMMessage.model_validate(
                reasoning_from_api(choice["message"], field_name)
            )
        if "delta" in choice:
            return LLMMessage.model_validate(
                reasoning_from_api(choice["delta"], field_name)
            )
        raise ValueError("Invalid response data: missing message or delta")

    if "message" in data:
        return LLMMessage.model_validate(reasoning_from_api(data["message"], field_name))
    if "delta" in data:
        return LLMMessage.model_validate(reasoning_from_api(data["delta"], field_name))
    return None
