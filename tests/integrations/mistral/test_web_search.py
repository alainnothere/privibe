from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from mistralai.client import Mistral
from mistralai.client.errors import SDKError
from mistralai.client.models import (
    ConversationResponse,
    ConversationUsageInfo,
    MessageOutputEntry,
    TextChunk,
    ToolReferenceChunk,
)
import pytest

from privibe.core.config import ProviderConfig
from privibe.core.tools.base import InvokeContext, ToolError
from privibe.core.tools.builtins.websearch import WebSearchArgs, WebSearchConfig
from privibe.integrations.mistral.web_search import MistralWebSearchImpl
from tests.mock.utils import collect_result


def _make_response(
    content: list | None = None, outputs: list | None = None
) -> ConversationResponse:
    if outputs is None:
        outputs = [MessageOutputEntry(content=content or [])]
    return ConversationResponse(
        conversation_id="test",
        outputs=outputs,
        usage=ConversationUsageInfo(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        ),
    )


@pytest.fixture
def impl(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    return MistralWebSearchImpl(WebSearchConfig(), ctx=None)


def test_parse_text_chunks(impl):
    response = _make_response(
        content=[TextChunk(text="Hello "), TextChunk(text="world")]
    )
    result = impl._parse_response(response)
    assert result.answer == "Hello world"
    assert result.sources == []


def test_parse_sources_deduped(impl):
    response = _make_response(
        content=[
            TextChunk(text="Answer"),
            ToolReferenceChunk(tool="web_search", title="Site A", url="https://a.com"),
            ToolReferenceChunk(
                tool="web_search", title="Site A duplicate", url="https://a.com"
            ),
            ToolReferenceChunk(tool="web_search", title="Site B", url="https://b.com"),
        ]
    )
    result = impl._parse_response(response)
    assert result.answer == "Answer"
    assert len(result.sources) == 2
    assert result.sources[0].url == "https://a.com"
    assert result.sources[0].title == "Site A"
    assert result.sources[1].url == "https://b.com"


def test_parse_skips_source_without_url(impl):
    response = _make_response(
        content=[
            TextChunk(text="Answer"),
            ToolReferenceChunk(tool="web_search", title="No URL"),
        ]
    )
    result = impl._parse_response(response)
    assert result.sources == []


def test_parse_empty_text_raises(impl):
    response = _make_response(content=[])
    with pytest.raises(ToolError, match="No text in agent response"):
        impl._parse_response(response)


def test_parse_whitespace_only_raises(impl):
    response = _make_response(content=[TextChunk(text="   ")])
    with pytest.raises(ToolError, match="No text in agent response"):
        impl._parse_response(response)


def test_parse_skips_non_message_entries(impl):
    response = _make_response(
        outputs=[MessageOutputEntry(content=[TextChunk(text="Answer")])]
    )
    result = impl._parse_response(response)
    assert result.answer == "Answer"


@pytest.mark.asyncio
async def test_search_missing_api_key(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    impl = MistralWebSearchImpl(WebSearchConfig(), ctx=None)
    with pytest.raises(ToolError, match="MISTRAL_API_KEY"):
        await collect_result(impl.search(WebSearchArgs(query="test")))


@pytest.mark.asyncio
async def test_search_returns_parsed_result(impl):
    response = _make_response(
        content=[
            TextChunk(text="The answer"),
            ToolReferenceChunk(
                tool="web_search", title="Source", url="https://example.com"
            ),
        ]
    )

    mock_start = AsyncMock(return_value=response)
    with patch.object(Mistral, "beta", create=True) as mock_beta:
        mock_beta.conversations.start_async = mock_start
        with patch.object(Mistral, "__aenter__", return_value=None):
            with patch.object(Mistral, "__aexit__", return_value=None):
                result = await collect_result(
                    impl.search(WebSearchArgs(query="test query"))
                )

    assert result.answer == "The answer"
    assert len(result.sources) == 1
    assert result.sources[0].url == "https://example.com"


@pytest.mark.asyncio
async def test_search_sdk_error_wrapped(impl):
    from unittest.mock import Mock

    import httpx

    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 500
    mock_response.text = "error"
    mock_response.headers = httpx.Headers({"content-type": "application/json"})

    with patch.object(Mistral, "beta", create=True) as mock_beta:
        mock_beta.conversations.start_async = AsyncMock(
            side_effect=SDKError("API failed", mock_response)
        )
        with patch.object(Mistral, "__aenter__", return_value=None):
            with patch.object(Mistral, "__aexit__", return_value=None):
                with pytest.raises(ToolError, match="Mistral API error"):
                    await collect_result(impl.search(WebSearchArgs(query="test")))


def test_resolve_server_url_no_ctx():
    impl = MistralWebSearchImpl(WebSearchConfig(), ctx=None)
    assert impl._resolve_server_url() is None


def test_resolve_server_url_no_agent_manager():
    ctx = InvokeContext(tool_call_id="t1", agent_manager=None)
    impl = MistralWebSearchImpl(WebSearchConfig(), ctx=ctx)
    assert impl._resolve_server_url() is None


def test_resolve_server_url_with_mistral_provider():
    config = MagicMock()
    config.providers = [
        ProviderConfig(
            name="mistral",
            api_base="https://on-prem.example.com/v1",
            api_key_env_var="MISTRAL_API_KEY",
            backend="mistral",
        )
    ]
    agent_manager = MagicMock()
    agent_manager.config = config

    ctx = InvokeContext(tool_call_id="t1", agent_manager=agent_manager)
    impl = MistralWebSearchImpl(WebSearchConfig(), ctx=ctx)
    assert impl._resolve_server_url() == "https://on-prem.example.com"


def test_resolve_server_url_with_default_provider():
    config = MagicMock()
    config.providers = [
        ProviderConfig(
            name="mistral",
            api_base="https://api.mistral.ai/v1",
            api_key_env_var="MISTRAL_API_KEY",
            backend="mistral",
        )
    ]
    agent_manager = MagicMock()
    agent_manager.config = config

    ctx = InvokeContext(tool_call_id="t1", agent_manager=agent_manager)
    impl = MistralWebSearchImpl(WebSearchConfig(), ctx=ctx)
    assert impl._resolve_server_url() == "https://api.mistral.ai"


def test_resolve_server_url_no_mistral_provider():
    config = MagicMock()
    config.providers = [
        ProviderConfig(name="llamacpp", api_base="http://127.0.0.1:8080/v1")
    ]
    agent_manager = MagicMock()
    agent_manager.config = config

    ctx = InvokeContext(tool_call_id="t1", agent_manager=agent_manager)
    impl = MistralWebSearchImpl(WebSearchConfig(), ctx=ctx)
    assert impl._resolve_server_url() is None
