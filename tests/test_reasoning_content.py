from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
from mistralai.client.models import (
    AssistantMessage,
    ContentChunk,
    TextChunk,
    ThinkChunk,
)
import pytest
import respx

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from privibe.core.config import ModelConfig, ProviderConfig, VibeConfig
from privibe.core.llm.backend.generic import GenericBackend, OpenAIAdapter
from privibe.integrations.mistral.backend import MistralBackend, MistralMapper, ParsedContent
from privibe.core.llm.format import APIToolFormatHandler
from privibe.core.types import (
    AssistantEvent,
    FunctionCall,
    LLMMessage,
    ReasoningEvent,
    Role,
    ToolCall,
)


def make_config() -> VibeConfig:
    return build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=False,
        include_model_info=False,
        include_commit_signature=False,
        enabled_tools=[],
        tools={},
    )


class TestMistralMapperParseContent:
    def test_parse_content_string_returns_content_only(self):
        mapper = MistralMapper()
        result = mapper.parse_content("Hello, world!")

        assert result == ParsedContent(content="Hello, world!", reasoning_content=None)

    def test_parse_content_text_chunk_returns_content_only(self):
        mapper = MistralMapper()
        content: list[ContentChunk] = [
            TextChunk(type="text", text="Hello from text chunk")
        ]

        result = mapper.parse_content(content)

        assert result == ParsedContent(
            content="Hello from text chunk", reasoning_content=None
        )

    def test_parse_content_thinking_chunk_extracts_reasoning(self):
        mapper = MistralMapper()
        content: list[ContentChunk] = [
            ThinkChunk(
                type="thinking",
                thinking=[TextChunk(type="text", text="Let me think...")],
            ),
            TextChunk(type="text", text="The answer is 42."),
        ]

        result = mapper.parse_content(content)

        assert result == ParsedContent(
            content="The answer is 42.", reasoning_content="Let me think..."
        )

    def test_parse_content_multiple_thinking_chunks_concatenates(self):
        mapper = MistralMapper()
        content: list[ContentChunk] = [
            ThinkChunk(
                type="thinking",
                thinking=[TextChunk(type="text", text="First thought. ")],
            ),
            ThinkChunk(
                type="thinking",
                thinking=[TextChunk(type="text", text="Second thought.")],
            ),
            TextChunk(type="text", text="Final answer."),
        ]

        result = mapper.parse_content(content)

        assert result == ParsedContent(
            content="Final answer.", reasoning_content="First thought. Second thought."
        )

    def test_parse_content_thinking_only_returns_empty_content(self):
        mapper = MistralMapper()
        content: list[ContentChunk] = [
            ThinkChunk(
                type="thinking",
                thinking=[TextChunk(type="text", text="Just thinking...")],
            )
        ]

        result = mapper.parse_content(content)

        assert result == ParsedContent(content="", reasoning_content="Just thinking...")

    def test_parse_content_empty_list_returns_empty(self):
        mapper = MistralMapper()
        content: list[ContentChunk] = []

        result = mapper.parse_content(content)

        assert result == ParsedContent(content="", reasoning_content=None)


class TestMistralMapperPrepareMessage:
    def test_prepare_assistant_message_without_reasoning(self):
        mapper = MistralMapper()
        msg = LLMMessage(role=Role.assistant, content="Hello!")

        result = mapper.prepare_message(msg)

        assert isinstance(result, AssistantMessage)
        assert result.content == "Hello!"

    def test_prepare_assistant_message_with_reasoning_creates_chunks(self):
        mapper = MistralMapper()
        msg = LLMMessage(
            role=Role.assistant,
            content="The answer is 42.",
            reasoning_content="Let me calculate...",
        )

        result = mapper.prepare_message(msg)

        assert isinstance(result, AssistantMessage)
        assert isinstance(result.content, list)
        assert len(result.content) == 2

        think_chunk = result.content[0]
        assert isinstance(think_chunk, ThinkChunk)
        assert think_chunk.type == "thinking"
        assert len(think_chunk.thinking) == 1
        inner_chunk = think_chunk.thinking[0]
        assert isinstance(inner_chunk, TextChunk)
        assert inner_chunk.text == "Let me calculate..."

        text_chunk = result.content[1]
        assert isinstance(text_chunk, TextChunk)
        assert text_chunk.type == "text"
        assert text_chunk.text == "The answer is 42."

    def test_prepare_assistant_message_with_reasoning_and_none_content(self):
        mapper = MistralMapper()
        msg = LLMMessage(
            role=Role.assistant, content=None, reasoning_content="Just thinking..."
        )

        result = mapper.prepare_message(msg)

        assert isinstance(result, AssistantMessage)
        assert isinstance(result.content, list)
        assert len(result.content) == 1

        think_chunk = result.content[0]
        assert isinstance(think_chunk, ThinkChunk)
        assert think_chunk.type == "thinking"
        assert len(think_chunk.thinking) == 1
        inner_chunk = think_chunk.thinking[0]
        assert isinstance(inner_chunk, TextChunk)
        assert inner_chunk.text == "Just thinking..."


class TestGenericBackendReasoningContent:
    @pytest.mark.asyncio
    async def test_complete_extracts_reasoning_content(self):
        base_url = "https://api.example.com"
        json_response = {
            "id": "fake_id",
            "created": 1234567890,
            "model": "test-model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "The answer is 42.",
                        "reasoning_content": "Let me think step by step...",
                    },
                }
            ],
        }

        with respx.mock(base_url=base_url) as mock_api:
            mock_api.post("/v1/chat/completions").mock(
                return_value=httpx.Response(status_code=200, json=json_response)
            )
            provider = ProviderConfig(
                name="test", api_base=f"{base_url}/v1", api_key_env_var="API_KEY"
            )
            backend = GenericBackend(provider=provider)
            model = ModelConfig(name="test-model", provider="test", alias="test")
            messages = [LLMMessage(role=Role.user, content="What is the answer?")]

            result = await backend.complete(
                model=model,
                messages=messages,
                temperature=0.2,
                tools=None,
                max_tokens=None,
                tool_choice=None,
                extra_headers=None,
            )

            assert result.message.content == "The answer is 42."
            assert result.message.reasoning_content == "Let me think step by step..."

    @pytest.mark.asyncio
    async def test_complete_streaming_extracts_reasoning_content(self):
        base_url = "https://api.example.com"
        chunks = [
            b'data: {"id":"id1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Thinking..."},"finish_reason":null}]}',
            b'data: {"id":"id1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{"content":"Answer"},"finish_reason":null}]}',
            b'data: {"id":"id1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}',
            b"data: [DONE]",
        ]

        with respx.mock(base_url=base_url) as mock_api:
            mock_api.post("/v1/chat/completions").mock(
                return_value=httpx.Response(
                    status_code=200,
                    stream=httpx.ByteStream(stream=b"\n\n".join(chunks)),
                    headers={"Content-Type": "text/event-stream"},
                )
            )
            provider = ProviderConfig(
                name="test", api_base=f"{base_url}/v1", api_key_env_var="API_KEY"
            )
            backend = GenericBackend(provider=provider)
            model = ModelConfig(name="test-model", provider="test", alias="test")
            messages = [LLMMessage(role=Role.user, content="Stream please")]

            results = []
            async for chunk in backend.complete_streaming(
                model=model,
                messages=messages,
                temperature=0.2,
                tools=None,
                max_tokens=None,
                tool_choice=None,
                extra_headers=None,
            ):
                results.append(chunk)

            assert results[0].message.reasoning_content == "Thinking..."
            assert results[0].message.content == ""
            assert results[1].message.content == "Answer"
            assert results[1].message.reasoning_content is None


class TestAPIToolFormatHandlerReasoningContent:
    def test_process_api_response_message_extracts_reasoning_content(self):
        handler = APIToolFormatHandler()

        mock_message = MagicMock()
        mock_message.role = "assistant"
        mock_message.content = "The answer is 42."
        mock_message.reasoning_content = "Let me think..."
        mock_message.reasoning_signature = None
        mock_message.tool_calls = None

        result = handler.process_api_response_message(mock_message)

        assert result.content == "The answer is 42."
        assert result.reasoning_content == "Let me think..."

    def test_process_api_response_message_handles_missing_reasoning_content(self):
        handler = APIToolFormatHandler()

        mock_message = MagicMock(spec=["role", "content", "tool_calls"])
        mock_message.role = "assistant"
        mock_message.content = "Hello"
        mock_message.tool_calls = None

        result = handler.process_api_response_message(mock_message)

        assert result.content == "Hello"
        assert result.reasoning_content is None


class TestAgentLoopStreamingReasoningEvents:
    @pytest.mark.asyncio
    async def test_streaming_accumulates_reasoning_in_message(self):
        backend = FakeBackend([
            mock_llm_chunk(content="", reasoning_content="First thought. "),
            mock_llm_chunk(content="", reasoning_content="Second thought."),
            mock_llm_chunk(content="Final answer."),
        ])
        agent = build_test_agent_loop(
            config=make_config(), backend=backend, enable_streaming=True
        )

        [_ async for _ in agent.act("Think and answer")]

        assistant_msg = next(m for m in agent.messages if m.role == Role.assistant)
        assert assistant_msg.reasoning_content == "First thought. Second thought."
        assert assistant_msg.content == "Final answer."

    @pytest.mark.asyncio
    async def test_streaming_content_only_no_reasoning(self):
        backend = FakeBackend([
            mock_llm_chunk(content="Hello "),
            mock_llm_chunk(content="world!"),
        ])
        agent = build_test_agent_loop(
            config=make_config(), backend=backend, enable_streaming=True
        )

        events = [event async for event in agent.act("Say hello")]

        reasoning_events = [e for e in events if isinstance(e, ReasoningEvent)]
        assert len(reasoning_events) == 0

        assistant_events = [e for e in events if isinstance(e, AssistantEvent)]
        assert len(assistant_events) == 2

        assistant_msg = next(m for m in agent.messages if m.role == Role.assistant)
        assert assistant_msg.reasoning_content is None
        assert assistant_msg.content == "Hello world!"


class TestLLMMessageReasoningContent:
    def test_llm_message_from_dict_with_reasoning_content(self):
        data = {
            "role": "assistant",
            "content": "Answer",
            "reasoning_content": "Thinking...",
        }

        msg = LLMMessage.model_validate(data)

        assert msg.reasoning_content == "Thinking..."

    def test_llm_message_model_dump_includes_reasoning_content(self):
        msg = LLMMessage(
            role=Role.assistant, content="Answer", reasoning_content="Thinking..."
        )

        dumped = msg.model_dump(exclude_none=True)

        assert dumped["reasoning_content"] == "Thinking..."

    def test_llm_message_model_dump_excludes_none_reasoning_content(self):
        msg = LLMMessage(role=Role.assistant, content="Answer")

        dumped = msg.model_dump(exclude_none=True)

        assert "reasoning_content" not in dumped


class TestReasoningFieldNameConversion:
    def test_reasoning_to_api_keeps_default_field(self):
        adapter = OpenAIAdapter()
        msg_dict = {
            "role": "assistant",
            "content": "Answer",
            "reasoning_content": "Thinking...",
        }

        result = adapter._reasoning_to_api(msg_dict, "reasoning_content")

        assert result["reasoning_content"] == "Thinking..."
        assert "reasoning" not in result

    def test_reasoning_to_api_renames_to_custom_field(self):
        adapter = OpenAIAdapter()
        msg_dict = {
            "role": "assistant",
            "content": "Answer",
            "reasoning_content": "Thinking...",
        }

        result = adapter._reasoning_to_api(msg_dict, "reasoning")

        assert result["reasoning"] == "Thinking..."
        assert "reasoning_content" not in result

    def test_reasoning_from_api_converts_custom_field(self):
        adapter = OpenAIAdapter()
        msg_dict = {
            "role": "assistant",
            "content": "Answer",
            "reasoning": "Thinking...",
        }

        result = adapter._reasoning_from_api(msg_dict, "reasoning")

        assert result["reasoning_content"] == "Thinking..."
        assert "reasoning" not in result

    def test_reasoning_from_api_keeps_default_field(self):
        adapter = OpenAIAdapter()
        msg_dict = {
            "role": "assistant",
            "content": "Answer",
            "reasoning_content": "Thinking...",
        }

        result = adapter._reasoning_from_api(msg_dict, "reasoning_content")

        assert result["reasoning_content"] == "Thinking..."

    @pytest.mark.asyncio
    async def test_complete_with_custom_reasoning_field_name(self):
        base_url = "https://api.example.com"
        json_response = {
            "id": "fake_id",
            "created": 1234567890,
            "model": "test-model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "The answer is 42.",
                        "reasoning": "Let me think step by step...",
                    },
                }
            ],
        }

        with respx.mock(base_url=base_url) as mock_api:
            mock_api.post("/v1/chat/completions").mock(
                return_value=httpx.Response(status_code=200, json=json_response)
            )
            provider = ProviderConfig(
                name="test",
                api_base=f"{base_url}/v1",
                api_key_env_var="API_KEY",
                reasoning_field_name="reasoning",
            )
            backend = GenericBackend(provider=provider)
            model = ModelConfig(name="test-model", provider="test", alias="test")
            messages = [LLMMessage(role=Role.user, content="What is the answer?")]

            result = await backend.complete(
                model=model,
                messages=messages,
                temperature=0.2,
                tools=None,
                max_tokens=None,
                tool_choice=None,
                extra_headers=None,
            )

            assert result.message.content == "The answer is 42."
            assert result.message.reasoning_content == "Let me think step by step..."

    @pytest.mark.asyncio
    async def test_streaming_with_custom_reasoning_field_name(self):
        base_url = "https://api.example.com"
        chunks = [
            b'data: {"id":"id1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{"role":"assistant","reasoning":"Thinking..."},"finish_reason":null}]}',
            b'data: {"id":"id1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{"content":"Answer"},"finish_reason":null}]}',
            b'data: {"id":"id1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}',
            b"data: [DONE]",
        ]

        with respx.mock(base_url=base_url) as mock_api:
            mock_api.post("/v1/chat/completions").mock(
                return_value=httpx.Response(
                    status_code=200,
                    stream=httpx.ByteStream(stream=b"\n\n".join(chunks)),
                    headers={"Content-Type": "text/event-stream"},
                )
            )
            provider = ProviderConfig(
                name="test",
                api_base=f"{base_url}/v1",
                api_key_env_var="API_KEY",
                reasoning_field_name="reasoning",
            )
            backend = GenericBackend(provider=provider)
            model = ModelConfig(name="test-model", provider="test", alias="test")
            messages = [LLMMessage(role=Role.user, content="Stream please")]

            results = []
            async for chunk in backend.complete_streaming(
                model=model,
                messages=messages,
                temperature=0.2,
                tools=None,
                max_tokens=None,
                tool_choice=None,
                extra_headers=None,
            ):
                results.append(chunk)

            assert results[0].message.reasoning_content == "Thinking..."
            assert results[1].message.content == "Answer"


class TestMistralReasoningFieldNameValidation:
    def test_mistral_backend_rejects_custom_reasoning_field_name(self):
        provider = ProviderConfig(
            name="mistral",
            api_base="https://api.mistral.ai/v1",
            api_key_env_var="MISTRAL_API_KEY",
            reasoning_field_name="reasoning",
        )

        with pytest.raises(ValueError) as exc_info:
            MistralBackend(provider=provider)

        assert "does not support custom reasoning_field_name" in str(exc_info.value)
        assert "reasoning" in str(exc_info.value)

    def test_mistral_backend_accepts_default_reasoning_field_name(self):
        provider = ProviderConfig(
            name="mistral",
            api_base="https://api.mistral.ai/v1",
            api_key_env_var="MISTRAL_API_KEY",
            reasoning_field_name="reasoning_content",
        )

        backend = MistralBackend(provider=provider)
        assert backend is not None


class TestEnsureAssistantContent:
    """A thinking-only assistant turn (reasoning_content set, content None,
    no tool_calls) is dropped to a content-less dict by exclude_none, and
    llama.cpp rejects the whole request with 400 "Assistant message must
    contain either 'content' or 'tool_calls'!". The adapter must add an
    empty content to the outgoing dict, matching what the session loader
    produces on restore.
    """

    def _prepare_payload(self, messages: list[LLMMessage]) -> dict:
        adapter = OpenAIAdapter()
        provider = ProviderConfig(name="test", api_base="https://api.example.com/v1")
        req = adapter.prepare_request(
            model_name="test-model",
            messages=messages,
            temperature=0.2,
            tools=None,
            max_tokens=None,
            tool_choice=None,
            enable_streaming=False,
            provider=provider,
        )
        return json.loads(req.body)

    def test_adds_empty_content_when_content_and_tool_calls_missing(self):
        adapter = OpenAIAdapter()
        msg_dict = {"role": "assistant", "reasoning_content": "Thinking..."}

        result = adapter._ensure_assistant_content(msg_dict)

        assert result["content"] == ""

    def test_keeps_existing_content(self):
        adapter = OpenAIAdapter()
        msg_dict = {"role": "assistant", "content": "Answer"}

        result = adapter._ensure_assistant_content(msg_dict)

        assert result["content"] == "Answer"

    def test_skips_assistant_message_with_tool_calls(self):
        adapter = OpenAIAdapter()
        msg_dict = {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "type": "function"}],
        }

        result = adapter._ensure_assistant_content(msg_dict)

        assert "content" not in result

    def test_skips_non_assistant_roles(self):
        adapter = OpenAIAdapter()
        msg_dict = {"role": "tool", "tool_call_id": "c1"}

        result = adapter._ensure_assistant_content(msg_dict)

        assert "content" not in result

    def test_prepare_request_thinking_only_assistant_gets_empty_content(self):
        payload = self._prepare_payload(
            [
                LLMMessage(role=Role.user, content="Question?"),
                LLMMessage(
                    role=Role.assistant,
                    content=None,
                    reasoning_content="Thinking...",
                ),
                LLMMessage(role=Role.user, content="Still there?"),
            ]
        )

        assistant = payload["messages"][1]
        assert assistant["role"] == "assistant"
        assert assistant["content"] == ""
        assert assistant["reasoning_content"] == "Thinking..."

    def test_prepare_request_assistant_with_tool_calls_stays_content_less(self):
        tool_call = ToolCall(
            id="c1",
            index=0,
            function=FunctionCall(name="bash", arguments="{}"),
        )
        payload = self._prepare_payload(
            [
                LLMMessage(role=Role.user, content="Question?"),
                LLMMessage(
                    role=Role.assistant,
                    content=None,
                    reasoning_content="Thinking...",
                    tool_calls=[tool_call],
                ),
                LLMMessage(
                    role=Role.tool, tool_call_id="c1", name="bash", content="ok"
                ),
            ]
        )

        assistant = payload["messages"][1]
        assert "content" not in assistant
        assert assistant["tool_calls"]
