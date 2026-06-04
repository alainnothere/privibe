from __future__ import annotations

from collections.abc import Sequence

from privibe.core.types import LLMMessage, Role


def strip_reasoning(msg: LLMMessage) -> LLMMessage:
    if msg.role != Role.assistant or not msg.reasoning_content:
        return msg
    return msg.model_copy(
        update={"reasoning_content": None, "reasoning_signature": None}
    )


def insert_between_consecutive_assistant_messages(
    messages: Sequence[LLMMessage],
) -> list[LLMMessage]:
    """Insert an empty user message between consecutive assistant messages.

    Some backends (e.g. Devstral via llama.cpp) require strictly alternating
    user/assistant turns. An empty placeholder keeps the sequence valid without
    altering any real content.
    """
    result: list[LLMMessage] = []
    for msg in messages:
        if result and result[-1].role == Role.assistant and msg.role == Role.assistant:
            result.append(LLMMessage(role=Role.user, content=""))
        result.append(msg)
    return result


def merge_consecutive_user_messages(messages: Sequence[LLMMessage]) -> list[LLMMessage]:
    """Merge consecutive user messages into a single message.

    This handles cases where middleware injects messages resulting in
    consecutive user messages before sending to the API.
    """
    result: list[LLMMessage] = []
    for msg in messages:
        if result and result[-1].role == Role.user and msg.role == Role.user:
            prev_content = result[-1].content or ""
            curr_content = msg.content or ""
            merged_content = f"{prev_content}\n\n{curr_content}".strip()
            result[-1] = LLMMessage(
                role=Role.user, content=merged_content, message_id=result[-1].message_id
            )
        else:
            result.append(msg)

    return result
