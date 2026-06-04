from __future__ import annotations

from privibe.core.llm.message_utils import (
    insert_between_consecutive_assistant_messages,
    merge_consecutive_user_messages,
)
from privibe.core.types import LLMMessage, Role


def test_merge_consecutive_user_messages() -> None:
    messages = [
        LLMMessage(role=Role.system, content="System"),
        LLMMessage(role=Role.user, content="User 1"),
        LLMMessage(role=Role.user, content="User 2"),
        LLMMessage(role=Role.assistant, content="Assistant"),
    ]
    result = merge_consecutive_user_messages(messages)
    assert len(result) == 3
    assert result[1].content == "User 1\n\nUser 2"


def test_preserves_non_consecutive_user_messages() -> None:
    messages = [
        LLMMessage(role=Role.user, content="User 1"),
        LLMMessage(role=Role.assistant, content="Assistant"),
        LLMMessage(role=Role.user, content="User 2"),
    ]
    result = merge_consecutive_user_messages(messages)
    assert len(result) == 3


def test_empty_messages() -> None:
    assert merge_consecutive_user_messages([]) == []


def test_single_message() -> None:
    messages = [LLMMessage(role=Role.user, content="Only one")]
    result = merge_consecutive_user_messages(messages)
    assert len(result) == 1


def test_three_consecutive_user_messages() -> None:
    messages = [
        LLMMessage(role=Role.user, content="A"),
        LLMMessage(role=Role.user, content="B"),
        LLMMessage(role=Role.user, content="C"),
    ]
    result = merge_consecutive_user_messages(messages)
    assert len(result) == 1
    assert result[0].content == "A\n\nB\n\nC"


# --- insert_between_consecutive_assistant_messages ---


def test_no_consecutive_assistant_messages() -> None:
    messages = [
        LLMMessage(role=Role.user, content="Hello"),
        LLMMessage(role=Role.assistant, content="Hi"),
        LLMMessage(role=Role.user, content="Again"),
        LLMMessage(role=Role.assistant, content="Sure"),
    ]
    result = insert_between_consecutive_assistant_messages(messages)
    assert result == list(messages)


def test_two_consecutive_assistant_messages() -> None:
    messages = [
        LLMMessage(role=Role.user, content="Hello"),
        LLMMessage(role=Role.assistant, content="First"),
        LLMMessage(role=Role.assistant, content="Second"),
    ]
    result = insert_between_consecutive_assistant_messages(messages)
    assert len(result) == 4
    assert result[0].role == Role.user
    assert result[1].role == Role.assistant
    assert result[1].content == "First"
    assert result[2].role == Role.user
    assert result[2].content == ""
    assert result[3].role == Role.assistant
    assert result[3].content == "Second"


def test_three_consecutive_assistant_messages() -> None:
    messages = [
        LLMMessage(role=Role.user, content="Go"),
        LLMMessage(role=Role.assistant, content="A"),
        LLMMessage(role=Role.assistant, content="B"),
        LLMMessage(role=Role.assistant, content="C"),
    ]
    result = insert_between_consecutive_assistant_messages(messages)
    assert len(result) == 6
    roles = [m.role for m in result]
    assert roles == [Role.user, Role.assistant, Role.user, Role.assistant, Role.user, Role.assistant]


def test_empty_messages_assistant() -> None:
    assert insert_between_consecutive_assistant_messages([]) == []


def test_single_assistant_message() -> None:
    messages = [LLMMessage(role=Role.assistant, content="Only one")]
    result = insert_between_consecutive_assistant_messages(messages)
    assert result == list(messages)


def test_non_assistant_roles_not_affected() -> None:
    messages = [
        LLMMessage(role=Role.system, content="sys"),
        LLMMessage(role=Role.user, content="u1"),
        LLMMessage(role=Role.user, content="u2"),
        LLMMessage(role=Role.assistant, content="a1"),
    ]
    result = insert_between_consecutive_assistant_messages(messages)
    assert result == list(messages)
