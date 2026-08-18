from __future__ import annotations

import httpx
import pytest
import respx

from privibe.core.tools.base import BaseToolState, ToolError
from privibe.core.tools.builtins.webfetch import WebFetch, WebFetchArgs, WebFetchConfig
from tests.mock.utils import collect_result


@pytest.fixture
def webfetch():
    config = WebFetchConfig()
    return WebFetch(config=config, state=BaseToolState())


@pytest.fixture
def webfetch_small():
    config = WebFetchConfig(max_content_bytes=100)
    return WebFetch(config=config, state=BaseToolState())


@pytest.mark.asyncio
@respx.mock
async def test_bare_domain_gets_https(webfetch):
    respx.get("https://example.com").mock(
        return_value=httpx.Response(
            200, text="ok", headers={"Content-Type": "text/plain"}
        )
    )
    result = await collect_result(webfetch.run(WebFetchArgs(url="example.com")))
    assert result.url == "https://example.com"
    assert result.content == "ok"


@pytest.mark.asyncio
@respx.mock
async def test_http_url_stays_http(webfetch):
    respx.get("http://example.com").mock(
        return_value=httpx.Response(
            200, text="ok", headers={"Content-Type": "text/plain"}
        )
    )
    result = await collect_result(webfetch.run(WebFetchArgs(url="http://example.com")))
    assert result.url == "http://example.com"


@pytest.mark.asyncio
@respx.mock
async def test_https_url_stays_https(webfetch):
    respx.get("https://example.com").mock(
        return_value=httpx.Response(
            200, text="ok", headers={"Content-Type": "text/plain"}
        )
    )
    result = await collect_result(webfetch.run(WebFetchArgs(url="https://example.com")))
    assert result.url == "https://example.com"


@pytest.mark.asyncio
@respx.mock
async def test_protocol_relative_url_normalized(webfetch):
    respx.get("https://example.com").mock(
        return_value=httpx.Response(
            200, text="ok", headers={"Content-Type": "text/plain"}
        )
    )
    result = await collect_result(webfetch.run(WebFetchArgs(url="//example.com")))
    assert result.url == "https://example.com"
    assert result.content == "ok"


@pytest.mark.asyncio
async def test_ftp_scheme_rejected(webfetch):
    with pytest.raises(ToolError, match="Invalid URL scheme: ftp"):
        await collect_result(webfetch.run(WebFetchArgs(url="ftp://example.com")))


@pytest.mark.asyncio
async def test_empty_url_rejected(webfetch):
    with pytest.raises(ToolError, match="URL cannot be empty"):
        await collect_result(webfetch.run(WebFetchArgs(url="   ")))


@pytest.mark.asyncio
@respx.mock
async def test_html_converted_to_markdown(webfetch):
    html = "<html><body><h1>Title</h1><p>Hello world</p></body></html>"
    respx.get("https://example.com").mock(
        return_value=httpx.Response(
            200, text=html, headers={"Content-Type": "text/html; charset=utf-8"}
        )
    )
    result = await collect_result(webfetch.run(WebFetchArgs(url="https://example.com")))
    assert "# Title" in result.content
    assert "Hello world" in result.content


@pytest.mark.asyncio
@respx.mock
async def test_plain_text_unchanged(webfetch):
    respx.get("https://example.com/file.txt").mock(
        return_value=httpx.Response(
            200, text="just text", headers={"Content-Type": "text/plain"}
        )
    )
    result = await collect_result(
        webfetch.run(WebFetchArgs(url="https://example.com/file.txt"))
    )
    assert result.content == "just text"


@pytest.mark.asyncio
@respx.mock
async def test_scripts_stripped_from_markdown(webfetch):
    html = "<html><body><script>alert('xss')</script><style>.x{}</style><p>Clean</p></body></html>"
    respx.get("https://example.com").mock(
        return_value=httpx.Response(
            200, text=html, headers={"Content-Type": "text/html"}
        )
    )
    result = await collect_result(webfetch.run(WebFetchArgs(url="https://example.com")))
    assert "alert" not in result.content
    assert ".x{}" not in result.content
    assert "Clean" in result.content


@pytest.mark.asyncio
@respx.mock
async def test_default_user_agent_identifies_privibe(webfetch):
    route = respx.get("https://example.com").mock(
        return_value=httpx.Response(
            200, text="ok", headers={"Content-Type": "text/plain"}
        )
    )
    await collect_result(webfetch.run(WebFetchArgs(url="https://example.com")))
    user_agent = route.calls[0].request.headers["User-Agent"]
    assert user_agent.startswith("privibe-cli/")
    assert "https://" in user_agent


@pytest.mark.asyncio
@respx.mock
async def test_403_retried_as_browser(webfetch):
    route = respx.get("https://example.com")
    route.side_effect = [
        httpx.Response(403),
        httpx.Response(200, text="success", headers={"Content-Type": "text/plain"}),
    ]
    result = await collect_result(webfetch.run(WebFetchArgs(url="https://example.com")))
    assert result.content == "success"
    assert route.call_count == 2

    second_request = route.calls[1].request
    assert second_request.headers["User-Agent"].startswith("Mozilla/5.0")


@pytest.mark.asyncio
@respx.mock
async def test_persistent_403_raises_after_retry(webfetch):
    route = respx.get("https://example.com").mock(
        return_value=httpx.Response(403, headers={"Content-Type": "text/plain"})
    )
    with pytest.raises(ToolError, match="HTTP error 403"):
        await collect_result(webfetch.run(WebFetchArgs(url="https://example.com")))
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_truncates_to_max_bytes_with_disclaimer(webfetch_small):
    body = "a" * 200
    respx.get("https://example.com").mock(
        return_value=httpx.Response(
            200, text=body, headers={"Content-Type": "text/plain"}
        )
    )
    result = await collect_result(
        webfetch_small.run(WebFetchArgs(url="https://example.com"))
    )
    assert result.content.startswith("a" * 100)
    assert "[Content truncated due to size limit]" in result.content


@pytest.mark.asyncio
@respx.mock
async def test_truncates_html_with_disclaimer(webfetch_small):
    html = (
        "<html><body><h2>first title</h2>"
        + "x" * 200
        + "<h2>second title</h2></body></html>"
    )
    respx.get("https://example.com").mock(
        return_value=httpx.Response(
            200, text=html, headers={"Content-Type": "text/html"}
        )
    )
    result = await collect_result(
        webfetch_small.run(WebFetchArgs(url="https://example.com"))
    )

    assert "## first title" in result.content
    assert "## second title" not in result.content
    assert "[Content truncated due to size limit]" in result.content


@pytest.mark.asyncio
@respx.mock
async def test_images_reduced_to_alt_text(webfetch):
    html = (
        "<html><body><p>Story</p>"
        '<img src="https://cdn.example.com/enormous-hash.webp" alt="A cat">'
        '<img src="https://cdn.example.com/no-alt.webp"></body></html>'
    )
    respx.get("https://example.com").mock(
        return_value=httpx.Response(
            200, text=html, headers={"Content-Type": "text/html"}
        )
    )
    result = await collect_result(webfetch.run(WebFetchArgs(url="https://example.com")))
    assert "cdn.example.com" not in result.content
    assert "![" not in result.content
    assert "A cat" in result.content


@pytest.mark.asyncio
@respx.mock
async def test_boilerplate_tags_stripped(webfetch):
    html = (
        "<html><body><nav><a href='/home'>Home</a></nav>"
        "<p>Article body</p>"
        "<aside>Related links</aside>"
        "<form><button>Subscribe</button></form>"
        "<footer>© 2026 MegaCorp</footer></body></html>"
    )
    respx.get("https://example.com").mock(
        return_value=httpx.Response(
            200, text=html, headers={"Content-Type": "text/html"}
        )
    )
    result = await collect_result(webfetch.run(WebFetchArgs(url="https://example.com")))
    assert "Article body" in result.content
    for noise in ("Home", "Related links", "Subscribe", "MegaCorp"):
        assert noise not in result.content


@pytest.mark.asyncio
@respx.mock
async def test_link_hover_titles_dropped(webfetch):
    html = (
        '<html><body><p><a href="/wiki/Main" title="Visit the main page [z]">'
        "Main page</a></p></body></html>"
    )
    respx.get("https://example.com").mock(
        return_value=httpx.Response(
            200, text=html, headers={"Content-Type": "text/html"}
        )
    )
    result = await collect_result(webfetch.run(WebFetchArgs(url="https://example.com")))
    assert "[Main page](/wiki/Main)" in result.content
    assert "Visit the main page" not in result.content


@pytest.mark.asyncio
@respx.mock
async def test_http_404_raises_tool_error(webfetch):
    respx.get("https://example.com").mock(return_value=httpx.Response(404))
    with pytest.raises(ToolError, match="HTTP error 404"):
        await collect_result(webfetch.run(WebFetchArgs(url="https://example.com")))


@pytest.mark.asyncio
@respx.mock
async def test_http_500_raises_tool_error(webfetch):
    respx.get("https://example.com").mock(return_value=httpx.Response(500))
    with pytest.raises(ToolError, match="HTTP error 500"):
        await collect_result(webfetch.run(WebFetchArgs(url="https://example.com")))


@pytest.mark.asyncio
@respx.mock
async def test_timeout_raises_tool_error(webfetch):
    respx.get("https://example.com").mock(side_effect=httpx.ReadTimeout("timed out"))
    with pytest.raises(ToolError, match="Request timed out"):
        await collect_result(webfetch.run(WebFetchArgs(url="https://example.com")))


@pytest.mark.asyncio
@respx.mock
async def test_network_error_raises_tool_error(webfetch):
    respx.get("https://example.com").mock(
        side_effect=httpx.ConnectError("connection refused")
    )
    with pytest.raises(ToolError, match="Failed to fetch URL"):
        await collect_result(webfetch.run(WebFetchArgs(url="https://example.com")))


@pytest.mark.asyncio
async def test_negative_timeout_rejected(webfetch):
    with pytest.raises(ToolError, match="Timeout must be a positive number"):
        await collect_result(
            webfetch.run(WebFetchArgs(url="https://example.com", timeout=-1))
        )


@pytest.mark.asyncio
async def test_zero_timeout_rejected(webfetch):
    with pytest.raises(ToolError, match="Timeout must be a positive number"):
        await collect_result(
            webfetch.run(WebFetchArgs(url="https://example.com", timeout=0))
        )


@pytest.mark.asyncio
async def test_over_max_timeout_rejected(webfetch):
    with pytest.raises(ToolError, match="Timeout cannot exceed"):
        await collect_result(
            webfetch.run(WebFetchArgs(url="https://example.com", timeout=999))
        )


def test_get_status_text():
    assert WebFetch.get_status_text() == "Fetching URL"
