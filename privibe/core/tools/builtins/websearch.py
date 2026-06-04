from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, ClassVar, final

from pydantic import BaseModel, Field

from privibe.core.integration_registry import get_tool_override
from privibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from privibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from privibe.core.types import ToolStreamEvent

if TYPE_CHECKING:
    from privibe.core.types import ToolCallEvent, ToolResultEvent


class WebSearchSource(BaseModel):
    title: str
    url: str


class WebSearchArgs(BaseModel):
    query: str = Field(min_length=1)


class WebSearchResult(BaseModel):
    answer: str
    sources: list[WebSearchSource] = Field(default_factory=list)
    # Echoed back for the user-facing result line; excluded from the model result.
    query: str = Field(default="", exclude=True)


class WebSearchConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    timeout: int = Field(default=120, description="HTTP timeout in seconds.")
    model: str = Field(
        default="mistral-small-latest",
        description="Model to use for web search (must support tool use).",
    )


_NOT_CONFIGURED_MESSAGE = (
    "web_search is not configured. Install or enable an integration that "
    "provides a web_search implementation (e.g., the Mistral integration)."
)


class WebSearch(
    BaseTool[WebSearchArgs, WebSearchResult, WebSearchConfig, BaseToolState],
    ToolUIData[WebSearchArgs, WebSearchResult],
):
    description: ClassVar[str] = "Search the web for current information."

    @classmethod
    def is_available(cls) -> bool:
        return get_tool_override("web_search") is not None

    @final
    async def run(
        self, args: WebSearchArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | WebSearchResult, None]:
        impl_class = get_tool_override("web_search")
        if impl_class is None:
            raise ToolError(_NOT_CONFIGURED_MESSAGE)
        impl = impl_class(self.config, ctx)
        async for event in impl.search(args):
            if isinstance(event, WebSearchResult):
                event.query = args.query
            yield event

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if event.args is None:
            return ToolCallDisplay(summary="websearch")
        if not isinstance(event.args, WebSearchArgs):
            return ToolCallDisplay(summary="websearch")
        return ToolCallDisplay(summary=f"Searching the web: '{event.args.query}'")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, WebSearchResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        r = event.result
        suffix = f' for "{r.query}"' if r.query else ""
        return ToolResultDisplay(
            success=True, message=f"{len(r.sources)} sources found{suffix}"
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Searching the web"
