from __future__ import annotations

from collections.abc import AsyncGenerator
import os
from typing import TYPE_CHECKING

from mistralai.client import Mistral
from mistralai.client.errors import SDKError
from mistralai.client.models import (
    ConversationResponse,
    MessageOutputEntry,
    TextChunk,
    ToolReferenceChunk,
)

from privibe.core.tools.base import InvokeContext, ToolError
from privibe.core.tools.builtins.websearch import (
    WebSearchArgs,
    WebSearchConfig,
    WebSearchResult,
    WebSearchSource,
)
from privibe.core.types import ToolStreamEvent
from privibe.core.utils import get_server_url_from_api_base

if TYPE_CHECKING:
    pass


class MistralWebSearchImpl:
    def __init__(self, config: WebSearchConfig, ctx: InvokeContext | None) -> None:
        self.config = config
        self.ctx = ctx

    async def search(
        self, args: WebSearchArgs
    ) -> AsyncGenerator[ToolStreamEvent | WebSearchResult, None]:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ToolError("MISTRAL_API_KEY environment variable not set.")

        client = Mistral(
            api_key=api_key,
            server_url=self._resolve_server_url(),
            timeout_ms=self.config.timeout * 1000,
        )

        try:
            async with client:
                response = await client.beta.conversations.start_async(
                    model=self.config.model,
                    instructions="Always use the web_search tool to answer queries. Never answer from memory alone.",
                    tools=[{"type": "web_search"}],
                    inputs=args.query,
                    store=False,
                )

                yield self._parse_response(response)

        except SDKError as exc:
            raise ToolError(f"Mistral API error: {exc}") from exc

    def _resolve_server_url(self) -> str | None:
        if not self.ctx or not self.ctx.agent_manager:
            return None
        for provider in self.ctx.agent_manager.config.providers:
            if provider.backend == "mistral":
                return get_server_url_from_api_base(provider.api_base)
        return None

    def _parse_response(self, response: ConversationResponse) -> WebSearchResult:
        text_parts: list[str] = []
        sources: dict[str, WebSearchSource] = {}

        for entry in response.outputs:
            if not isinstance(entry, MessageOutputEntry):
                continue
            for chunk in entry.content:
                if isinstance(chunk, TextChunk):
                    text_parts.append(chunk.text)
                elif isinstance(chunk, ToolReferenceChunk) and chunk.url:
                    if chunk.url not in sources:
                        sources[chunk.url] = WebSearchSource(
                            title=chunk.title, url=chunk.url
                        )

        answer = "".join(text_parts).strip()
        if not answer:
            raise ToolError("No text in agent response.")

        return WebSearchResult(answer=answer, sources=list(sources.values()))
