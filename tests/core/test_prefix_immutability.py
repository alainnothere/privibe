"""Structural enforcement of the conversation-prefix immutability invariant.

Message 0 (the system prompt), stored messages, and the advertised tools
array form the llama.cpp KV-cache prefix. Once a session has started they
must never change:

- LLMMessage is frozen: in-place edits raise.
- ConversationList.rewind() can never remove a leading system message.
- The prefix tripwire hard-fails a request whose prefix bytes diverged.
"""

from __future__ import annotations

import pydantic
import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from privibe.core.conversation import ConversationList
from privibe.core.types import LLMMessage, Role


def _conversation(*user_contents: str) -> ConversationList:
    conv = ConversationList()
    conv.add(LLMMessage(role=Role.system, content="system prompt"))
    for content in user_contents:
        conv.add(LLMMessage(role=Role.user, content=content))
    return conv


class TestFrozenMessages:
    def test_stored_message_content_cannot_be_assigned(self) -> None:
        conv = _conversation("hello")
        with pytest.raises(pydantic.ValidationError):
            conv[0].content = "rewritten system prompt"
        with pytest.raises(pydantic.ValidationError):
            conv[1].role = Role.assistant

    def test_streaming_accumulation_still_works(self) -> None:
        a = LLMMessage(role=Role.assistant, content="Hel")
        b = LLMMessage(role=Role.assistant, content="lo")
        assert (a + b).content == "Hello"
        # The originals are untouched.
        assert a.content == "Hel"
        assert b.content == "lo"


class TestRewindFloor:
    def test_rewind_cannot_remove_message_zero(self) -> None:
        conv = _conversation("hello", "world")
        with pytest.raises(ValueError, match="may not remove the system message"):
            conv.rewind(len(conv))
        with pytest.raises(ValueError, match="may not remove the system message"):
            conv.rewind(999)

    def test_rewind_tail_still_works(self) -> None:
        conv = _conversation("hello", "world")
        conv.rewind(len(conv) - 1)
        assert len(conv) == 1
        assert conv[0].role == Role.system

    def test_rewind_without_system_message_is_unrestricted(self) -> None:
        conv = ConversationList()
        conv.add(LLMMessage(role=Role.user, content="hello"))
        conv.rewind(1)
        assert len(conv) == 0


class TestPrefixTripwire:
    def test_identical_prefix_passes(self) -> None:
        agent = build_test_agent_loop(
            config=build_test_vibe_config(
                include_project_context=False, include_prompt_detail=False
            )
        )
        tools = agent.messages.tools_for_request()
        agent._check_prefix_integrity(tools)
        agent._check_prefix_integrity(tools)

    def test_changed_tools_array_fails_hard(self) -> None:
        agent = build_test_agent_loop(
            config=build_test_vibe_config(
                include_project_context=False, include_prompt_detail=False
            )
        )
        tools = agent.messages.tools_for_request()
        agent._check_prefix_integrity(tools)
        with pytest.raises(RuntimeError, match="prefix changed mid-session"):
            agent._check_prefix_integrity(tools[:-1])

    def test_conversation_reset_rearms_the_tripwire(self) -> None:
        agent = build_test_agent_loop(
            config=build_test_vibe_config(
                include_project_context=False, include_prompt_detail=False
            )
        )
        tools = agent.messages.tools_for_request()
        agent._check_prefix_integrity(tools)
        # /clear-style tail cut fires the reset hooks.
        agent.messages.add(LLMMessage(role=Role.user, content="hello"))
        agent.messages.rewind(len(agent.messages) - 1)
        assert agent._prefix_fingerprint is None
        agent._check_prefix_integrity(tools)
