from __future__ import annotations

import pytest

from privibe.core.integration_registry import (
    _tool_overrides,
    register_tool_override,
)
from privibe.core.tools.base import BaseToolState, ToolError
from privibe.core.tools.builtins.websearch import (
    WebSearch,
    WebSearchArgs,
    WebSearchConfig,
    WebSearchResult,
)
from tests.mock.utils import collect_result


@pytest.fixture
def websearch():
    config = WebSearchConfig()
    return WebSearch(config=config, state=BaseToolState())


@pytest.fixture
def isolated_overrides():
    saved = dict(_tool_overrides)
    _tool_overrides.clear()
    yield
    _tool_overrides.clear()
    _tool_overrides.update(saved)


def test_is_available_returns_false_without_integration(isolated_overrides):
    assert WebSearch.is_available() is False


def test_is_available_returns_true_with_integration(isolated_overrides):
    class FakeImpl:
        def __init__(self, config, ctx): pass
        async def search(self, args): pass
    register_tool_override("web_search", FakeImpl)
    assert WebSearch.is_available() is True


def test_get_status_text():
    assert WebSearch.get_status_text() == "Searching the web"


@pytest.mark.asyncio
async def test_run_raises_when_no_override_registered(websearch, isolated_overrides):
    with pytest.raises(ToolError, match="not configured"):
        await collect_result(websearch.run(WebSearchArgs(query="test")))


@pytest.mark.asyncio
async def test_run_delegates_to_registered_override(websearch, isolated_overrides):
    class FakeImpl:
        def __init__(self, config, ctx):
            self.config = config
            self.ctx = ctx

        async def search(self, args):
            yield WebSearchResult(answer=f"echo: {args.query}", sources=[])

    register_tool_override("web_search", FakeImpl)
    result = await collect_result(websearch.run(WebSearchArgs(query="hello")))
    assert isinstance(result, WebSearchResult)
    assert result.answer == "echo: hello"
