from __future__ import annotations

from collections.abc import AsyncGenerator
import mimetypes
from pathlib import Path
import re
from typing import TYPE_CHECKING, ClassVar, final
from urllib.parse import urlparse

from bs4.element import Tag
import httpx
from markdownify import MarkdownConverter
from pydantic import BaseModel, Field

from privibe.core.paths import VIBE_HOME
from privibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from privibe.core.tools.permissions import (
    PermissionContext,
    PermissionScope,
    RequiredPermission,
)
from privibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from privibe.core.types import ToolStreamEvent

if TYPE_CHECKING:
    from privibe.core.types import ToolCallEvent, ToolResultEvent


# Some servers only serve mainstream browsers; used as a one-shot 403 fallback.
_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_HTTP_FORBIDDEN = 403


class _Converter(MarkdownConverter):
    convert_script = convert_style = convert_noscript = convert_iframe = (
        convert_object
    ) = convert_embed = convert_nav = convert_footer = convert_aside = convert_form = (
        convert_button
    ) = convert_svg = lambda *_, **__: ""

    def convert_img(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        # A text model can't see images; alt text is the only useful part.
        return str(el.attrs.get("alt") or "")

    def convert_a(self, el: Tag, text: str, parent_tags: set[str]) -> str:
        # Hover titles are tooltip text for a mouse that doesn't exist.
        el.attrs.pop("title", None)
        return str(super().convert_a(el, text, parent_tags))


class WebFetchArgs(BaseModel):
    url: str = Field(description="URL to fetch (http/https)")
    timeout: int | None = Field(
        default=None, description="Timeout in seconds (max 120)"
    )


class WebFetchResult(BaseModel):
    url: str
    content: str
    content_type: str
    # Set when the response was binary and written to disk instead of being
    # returned as content.
    saved_path: str | None = None


class WebFetchConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK

    default_timeout: int = Field(default=30, description="Default timeout in seconds.")
    max_timeout: int = Field(default=120, description="Maximum allowed timeout.")
    max_content_bytes: int = Field(
        default=512_000, description="Maximum text content size to return."
    )
    max_download_bytes: int = Field(
        default=50_000_000,
        description="Maximum size for binary responses saved to disk.",
    )
    # Robot-policy style (name/version + contact URL): Wikipedia and friends
    # fingerprint the TLS stack, so a fake browser UA reads as a liar and gets
    # 403'd where an identified client is served.
    user_agent: str = Field(
        default="privibe-cli/0.1 (+https://github.com/alainnothere/privibe) httpx",
        description="User agent string for requests.",
    )


class WebFetch(
    BaseTool[WebFetchArgs, WebFetchResult, WebFetchConfig, BaseToolState],
    ToolUIData[WebFetchArgs, WebFetchResult],
):
    description: ClassVar[str] = (
        "Fetch content from a URL. Converts HTML to markdown for readability."
    )

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalise a URL to always have an http(s) scheme.

        Handles protocol-relative URLs (//example.com) and bare URLs (example.com).
        """
        raw = url.lstrip("/") if url.startswith("//") else url
        return raw if raw.startswith(("http://", "https://")) else "https://" + raw

    def resolve_permission(self, args: WebFetchArgs) -> PermissionContext | None:
        if self.config.permission in {ToolPermission.ALWAYS, ToolPermission.NEVER}:
            return PermissionContext(permission=self.config.permission)

        parsed = urlparse(self._normalize_url(args.url))
        domain = parsed.netloc or parsed.path.split("/")[0]
        if not domain:
            return None

        return PermissionContext(
            permission=ToolPermission.ASK,
            required_permissions=[
                RequiredPermission(
                    scope=PermissionScope.URL_PATTERN,
                    invocation_pattern=domain,
                    session_pattern=domain,
                    label=f"fetching from {domain}",
                )
            ],
        )

    @final
    async def run(
        self, args: WebFetchArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | WebFetchResult, None]:
        self._validate_args(args)

        url = self._normalize_url(args.url)
        timeout = self._resolve_timeout(args.timeout)

        content, content_type, saved_path = await self._fetch_url(url, timeout)

        if saved_path is None and "text/html" in content_type:
            content = _html_to_markdown(content)

        yield WebFetchResult(
            url=url,
            content=content,
            content_type=content_type,
            saved_path=saved_path,
        )

    def _validate_args(self, args: WebFetchArgs) -> None:
        if not args.url.strip():
            raise ToolError("URL cannot be empty")

        parsed = urlparse(args.url)
        if parsed.scheme and parsed.scheme not in {"http", "https"}:
            raise ToolError(
                f"Invalid URL scheme: {parsed.scheme}. Must be http or https."
            )

        if args.timeout is not None:
            if args.timeout <= 0:
                raise ToolError("Timeout must be a positive number")
            if args.timeout > self.config.max_timeout:
                raise ToolError(
                    f"Timeout cannot exceed {self.config.max_timeout} seconds"
                )

    def _resolve_timeout(self, timeout: int | None) -> int:
        if timeout is None:
            return self.config.default_timeout
        return min(timeout, self.config.max_timeout)

    async def _fetch_url(self, url: str, timeout: int) -> tuple[str, str, str | None]:
        """Fetch the URL; returns (content, content_type, saved_path).

        Textual responses come back as decoded content with saved_path None.
        Binary responses are written to disk untouched, and content is a short
        notice pointing at the file — raw bytes decoded errors="ignore" are
        token soup that can flood the context (a 14MB PDF, famously).
        """
        headers = {
            "User-Agent": self.config.user_agent,
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,image/apng,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }

        try:
            response = await self._do_fetch(url, timeout, headers)
        except httpx.TimeoutException:
            raise ToolError(f"Request timed out after {timeout} seconds")
        except httpx.RequestError as e:
            raise ToolError(f"Failed to fetch URL: {e}")

        if response.is_error:
            raise ToolError(
                f"HTTP error {response.status_code}: {response.reason_phrase}"
            )

        content_type = response.headers.get("Content-Type", "text/plain")
        mime = content_type.split(";")[0].strip().lower()

        if not _is_textual(mime, response.content):
            saved = self._save_download(url, mime, response.content)
            notice = (
                f"The response is not a text page ({mime}, "
                f"{_format_size(len(response.content))}); its raw bytes were "
                f"not added to the context. Saved to: {saved}\n"
                f"Use local tools to inspect or convert it as needed."
            )
            return notice, content_type, str(saved)

        content_bytes = response.content[: self.config.max_content_bytes]
        content = content_bytes.decode("utf-8", errors="ignore")

        if len(response.content) > self.config.max_content_bytes:
            content += "[Content truncated due to size limit]"

        return content, content_type, None

    def _save_download(self, url: str, mime: str, body: bytes) -> Path:
        if len(body) > self.config.max_download_bytes:
            raise ToolError(
                f"Response is {_format_size(len(body))} of {mime}, over the "
                f"{_format_size(self.config.max_download_bytes)} download limit."
            )
        downloads = VIBE_HOME.path / "downloads"
        downloads.mkdir(parents=True, exist_ok=True)

        name = Path(urlparse(url).path).name
        name = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._") or "download"
        if "." not in name:
            name += mimetypes.guess_extension(mime) or ".bin"

        target = downloads / name
        stem, suffix = target.stem, target.suffix
        counter = 1
        while target.exists():
            target = downloads / f"{stem}-{counter}{suffix}"
            counter += 1
        target.write_bytes(body)
        return target

    async def _do_fetch(
        self, url: str, timeout: int, headers: dict[str, str]
    ) -> httpx.Response:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=httpx.Timeout(timeout)
        ) as client:
            response = await client.get(url, headers=headers)

            # Some servers 403 unfamiliar agents; retry once as a browser
            if response.status_code == _HTTP_FORBIDDEN:
                headers["User-Agent"] = _BROWSER_USER_AGENT
                response = await client.get(url, headers=headers)

            return response

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if event.args is None:
            return ToolCallDisplay(summary="webfetch")
        if not isinstance(event.args, WebFetchArgs):
            return ToolCallDisplay(summary="webfetch")

        parsed = urlparse(event.args.url)
        domain = parsed.netloc or event.args.url[:50]
        summary = f"Fetching: {domain}"

        if event.args.timeout:
            summary += f" (timeout {event.args.timeout}s)"

        return ToolCallDisplay(summary=summary)

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, WebFetchResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        parsed = urlparse(event.result.url)
        domain = parsed.netloc or event.result.url[:50]
        mime = event.result.content_type.split(";")[0]

        if event.result.saved_path:
            message = f"Saved {mime} from {domain} to {event.result.saved_path}"
        else:
            content_len = len(event.result.content)
            message = f"Fetched {content_len:,} chars from {domain} ({mime})"

        return ToolResultDisplay(success=True, message=message)

    @classmethod
    def get_status_text(cls) -> str:
        return "Fetching URL"


def _html_to_markdown(html: str) -> str:
    return _Converter(heading_style="ATX", bullets="-").convert(html)


_TEXTUAL_MIME_TYPES = {
    "application/json",
    "application/javascript",
    "application/x-javascript",
    "application/x-ndjson",
    "application/xml",
    "application/x-www-form-urlencoded",
}


def _is_textual(mime: str, body: bytes) -> bool:
    if (
        mime.startswith("text/")
        or mime in _TEXTUAL_MIME_TYPES
        or mime.endswith(("+json", "+xml"))
    ):
        # Servers that don't send a Content-Type land on text/plain, and some
        # mislabel binaries as text; a null byte early in the body outranks
        # whatever the header claims.
        return b"\x00" not in body[:8192]
    return False


def _format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB"):
        if value < 1024:
            return f"{value:,.0f} {unit}" if unit == "B" else f"{value:,.1f} {unit}"
        value /= 1024
    return f"{value:,.1f} GB"
