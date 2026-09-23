from __future__ import annotations

from datetime import datetime
from time import time

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from privibe.cli.textual_ui.widgets.loading import _format_elapsed
from privibe.cli.textual_ui.widgets.messages import ExpandingBorder, NonSelectableStatic
from privibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from privibe.cli.textual_ui.widgets.status_message import StatusMessage
from privibe.cli.textual_ui.widgets.tool_widgets import (
    FileDiffWidget,
    _truncate_lines,
    get_result_widget,
    half_viewport_cap,
)
from privibe.core.tools.ui import ToolUIDataAdapter
from privibe.core.types import MessageMeta, ToolCallEvent, ToolResultEvent


class ToolCallMessage(StatusMessage):
    def __init__(
        self,
        event: ToolCallEvent | None = None,
        *,
        tool_name: str | None = None,
        history: bool = False,
    ) -> None:
        """A tool call line. Live: built from the event as it streams in.
        History (resume / load-more): built from the stored message, with an
        event when the tool is known and without one otherwise; never spins.
        """
        if event is None and tool_name is None:
            raise ValueError("Either event or tool_name must be provided")

        self._event = event
        self._tool_name = tool_name or (event.tool_name if event else None) or "unknown"
        self._is_history = history or event is None
        self._stream_widget: NoMarkupStatic | None = None
        self._hint_widget: NoMarkupStatic | None = None
        self._start_time: float | None = None
        self._last_elapsed: int = -1

        super().__init__()
        self.add_class("tool-call")

        if self._is_history:
            self._is_spinning = False

    @property
    def history_key(self) -> str | None:
        """Stable id for windowing: the tool call id, prefixed to distinguish
        the assistant message holding the call from the tool result message.
        """
        if self._event is None or not self._event.tool_call_id:
            return None
        return f"call:{self._event.tool_call_id}"

    def compose(self) -> ComposeResult:
        with Vertical(classes="tool-call-container"):
            with Horizontal():
                self._indicator_widget = NonSelectableStatic(
                    self._spinner.current_frame(), classes="status-indicator-icon"
                )
                yield self._indicator_widget
                self._text_widget = NoMarkupStatic("", classes="status-indicator-text")
                yield self._text_widget
                self._hint_widget = NoMarkupStatic("", classes="loading-hint")
                yield self._hint_widget
            self._stream_widget = NoMarkupStatic("", classes="tool-stream-message")
            self._stream_widget.display = False
            yield self._stream_widget

    def update_display(self) -> None:
        super().update_display()
        if self._hint_widget and self._start_time is not None and self._is_spinning:
            elapsed = int(time() - self._start_time)
            if elapsed != self._last_elapsed:
                self._last_elapsed = elapsed
                timeout = self._event.timeout if self._event else None
                if timeout is not None:
                    self._hint_widget.update(
                        f"({_format_elapsed(elapsed)} / {timeout}s)"
                    )
                else:
                    self._hint_widget.update(f"({_format_elapsed(elapsed)})")

    def on_mount(self) -> None:
        if not self._is_history:
            self._start_time = time()
        super().on_mount()
        siblings = list(self.parent.children) if self.parent else []
        idx = siblings.index(self) if self in siblings else -1
        if idx > 0 and isinstance(
            siblings[idx - 1], (ToolCallMessage, ToolResultMessage)
        ):
            self.add_class("no-gap")

    @property
    def tool_call_id(self) -> str | None:
        return self._event.tool_call_id if self._event else None

    def get_content(self) -> str:
        if self._event:
            adapter = ToolUIDataAdapter(self._event.tool_class)
            display = adapter.get_call_display(self._event)
            return display.summary
        return self._tool_name

    def update_event(self, event: ToolCallEvent) -> None:
        self._event = event
        self._tool_name = event.tool_name
        if self._text_widget:
            self._text_widget.update(self.get_content())

    def set_stream_message(self, message: str) -> None:
        """Update the stream message displayed below the tool call indicator."""
        if self._stream_widget:
            self._stream_widget.update(f"→ {message}")
            self._stream_widget.display = True

    def stop_spinning(self, success: bool = True) -> None:
        """Stop the spinner while keeping stream row stable to avoid layout jumps."""
        super().stop_spinning(success)

    def set_result_text(self, text: str) -> None:
        if self._text_widget:
            self._text_widget.update(text)

    def set_finished_hint(self, result: ToolResultEvent | None) -> None:
        """Replace the frozen elapsed counter with where the time went: the
        LLM request behind this call, the approval wait, and the run.
        """
        if self._hint_widget is None:
            return
        call_requested_at = self._event.requested_at if self._event else None
        if result is None:
            self._hint_widget.update(finished_hint_text(None, call_requested_at))
            return
        self._hint_widget.update(
            finished_hint_text(
                result.duration,
                call_requested_at,
                request_meta=result.request_meta,
                tool_meta=result.tool_meta,
                include_request=result.first_in_message,
            )
        )


# Wall time of a request that the server's own clocks don't account for
# (queue, slot wait, template, network). Shown only above this, so noise
# stays off the line.
OVERHEAD_SHOWN_ABOVE_S = 0.1
_MINUTE = 60
_THOUSAND = 1000


def _secs(seconds: float) -> str:
    if seconds < _MINUTE:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(int(seconds), _MINUTE)
    return f"{minutes}m{rest:02d}s"


def _tokens(n: int) -> str:
    return str(n) if n < _THOUSAND else f"{n / _THOUSAND:.1f}k"


def _request_parts(meta: MessageMeta) -> list[str]:
    """Where the LLM request's time went. llama.cpp reports its own prefill
    and decode clocks; whatever the wall clock saw beyond them is overhead.
    Providers without server timings fall back to our own instants.
    """
    parts: list[str] = []
    wall = (
        meta.response_done_at - meta.request_sent_at
        if meta.response_done_at is not None and meta.request_sent_at is not None
        else None
    )
    t = meta.server_timings
    if t is not None and t.prompt_ms is not None:
        prefill = f"prefill {_secs(t.prompt_ms / 1000)} {_tokens(t.prompt_n or 0)} new"
        if t.cache_n:
            prefill += f" {_tokens(t.cache_n)} cached"
        parts.append(prefill)
        server_s = t.prompt_ms / 1000
        if t.predicted_ms is not None:
            parts.append(
                f"decode {_secs(t.predicted_ms / 1000)} "
                f"{_tokens(t.predicted_n or 0)} tok"
            )
            server_s += t.predicted_ms / 1000
        if wall is not None and wall - server_s > OVERHEAD_SHOWN_ABOVE_S:
            parts.append(f"overhead {_secs(wall - server_s)}")
    elif meta.first_chunk_at is not None and meta.request_sent_at is not None:
        parts.append(f"first token {_secs(meta.first_chunk_at - meta.request_sent_at)}")
        if meta.response_done_at is not None:
            parts.append(f"gen {_secs(meta.response_done_at - meta.first_chunk_at)}")
    elif wall is not None:
        parts.append(f"llm {_secs(wall)}")
    return parts


def finished_hint_text(
    duration: float | None,
    requested_at: float | None,
    *,
    request_meta: MessageMeta | None = None,
    tool_meta: MessageMeta | None = None,
    include_request: bool = False,
) -> str:
    """The hint after a finished call, in the order things happened: the LLM
    request (first call of a batch only), the approval wait, the run, and the
    wall-clock instant the request went out. Every number is a difference of
    stored instants or a server-reported clock; nothing here is stored.
    Empty when nothing is known (sessions written before meta existed).
    """
    parts: list[str] = []
    if include_request and request_meta is not None:
        parts.extend(_request_parts(request_meta))
    if (
        tool_meta is not None
        and tool_meta.approval_asked_at is not None
        and tool_meta.approval_answered_at is not None
    ):
        waited = tool_meta.approval_answered_at - tool_meta.approval_asked_at
        parts.append(f"approval {_secs(waited)}")
    if duration is not None:
        parts.append(f"run {_secs(duration)}")
    sent_at = (
        request_meta.request_sent_at
        if request_meta is not None and request_meta.request_sent_at is not None
        else requested_at
    )
    if sent_at is not None:
        parts.append(datetime.fromtimestamp(sent_at).strftime("%H:%M:%S"))
    return f"({' · '.join(parts)})" if parts else ""


class ToolResultMessage(Static):
    def __init__(
        self,
        event: ToolResultEvent | None = None,
        call_widget: ToolCallMessage | None = None,
        collapsed: bool = True,
        *,
        tool_name: str | None = None,
        content: str | None = None,
    ) -> None:
        if event is None and tool_name is None:
            raise ValueError("Either event or tool_name must be provided")

        self._event = event
        self._call_widget = call_widget
        self._tool_name = tool_name or (event.tool_name if event else "unknown")
        self._content = content
        self.collapsed = collapsed
        self._content_container: Vertical | None = None

        super().__init__()
        self.add_class("tool-result")

    @property
    def tool_name(self) -> str:
        return self._tool_name

    @property
    def history_key(self) -> str | None:
        """Stable id for windowing: the tool call id, prefixed to distinguish
        the tool result message from the assistant message holding the call.
        """
        if self._event is None or not self._event.tool_call_id:
            return None
        return f"result:{self._event.tool_call_id}"

    def compose(self) -> ComposeResult:
        with Horizontal(classes="tool-result-container"):
            yield ExpandingBorder(classes="tool-result-border")
            self._content_container = Vertical(classes="tool-result-content")
            yield self._content_container

    def apply_height_cap(self, screen_height: int | None = None) -> None:
        """Clamp the content region to half the screen, in cells (see
        half_viewport_cap). Re-applied from VibeApp.on_resize, which passes
        the new height explicitly since app.size may not be updated yet.
        """
        if self._content_container is not None:
            self._content_container.styles.max_height = half_viewport_cap(
                screen_height if screen_height is not None else self.app.size.height
            )

    async def on_mount(self) -> None:
        self.apply_height_cap()
        if self._call_widget:
            success = self._determine_success()
            self._call_widget.stop_spinning(success=success)
            result_text = self._get_result_text()
            self._call_widget.set_result_text(result_text)
            self._call_widget.set_finished_hint(self._event)
        await self._render_result()

    def _determine_success(self) -> bool:
        if self._event is None:
            return True
        if self._event.error or self._event.skipped:
            return False
        if self._event.tool_class:
            adapter = ToolUIDataAdapter(self._event.tool_class)
            display = adapter.get_result_display(self._event)
            return display.success
        return True

    def _get_result_text(self) -> str:
        if self._event is None:
            return f"{self._tool_name} completed"

        if self._event.error:
            return f"{self._tool_name} error"

        if self._event.skipped:
            return f"{self._tool_name} skipped"

        if self._event.tool_class:
            adapter = ToolUIDataAdapter(self._event.tool_class)
            display = adapter.get_result_display(self._event)
            return display.message

        return f"{self._tool_name} completed"

    def _preview_lines(self) -> int:
        app_config = getattr(self.app, "config", None)
        return getattr(app_config, "tool_result_preview_lines", 3)

    async def _mount_truncated(self, text: str, max_lines: int) -> None:
        """Mount text clamped to max_lines so it can't overdraw the input area."""
        if self._content_container is None:
            return
        content, truncation_info = _truncate_lines(text, max_lines)
        await self._content_container.mount(NoMarkupStatic(content))
        if truncation_info:
            await self._content_container.mount(
                NoMarkupStatic(truncation_info, classes="tool-result-hint")
            )

    async def _render_result(self) -> None:
        if self._content_container is None:
            return

        await self._content_container.remove_children()

        if self._event is None:
            if self._content:
                await self._content_container.mount(
                    NoMarkupStatic(self._content, classes="tool-result-detail")
                )
                self.display = not self.collapsed
            else:
                self.display = False
            return

        preview_lines = self._preview_lines()

        if self._event.error:
            self.add_class("error-text")
            await self._mount_truncated(f"Error: {self._event.error}", preview_lines)
            self.display = True
            return

        if self._event.skipped:
            self.add_class("warning-text")
            reason = self._event.skip_reason or "User skipped"
            await self._mount_truncated(f"Skipped: {reason}", preview_lines)
            self.display = True
            return

        self.remove_class("error-text")
        self.remove_class("warning-text")

        # File-mutating tools render a red/green diff instead of a field dump.
        # search_replace keeps its own diff widget for now (retiring that path is
        # a separate, decoupled change).
        if (
            self._event.file_diff is not None
            and self._event.tool_name != "search_replace"
        ):
            warnings: list[str] = []
            if self._event.tool_class is not None:
                adapter = ToolUIDataAdapter(self._event.tool_class)
                warnings = adapter.get_result_display(self._event).warnings
            for warning in warnings:
                await self._content_container.mount(
                    NoMarkupStatic(f"⚠ {warning}", classes="tool-result-warning")
                )
            await self._content_container.mount(
                FileDiffWidget(self._event.file_diff, max_hunks=preview_lines)
            )
            self.display = True
            return

        if self._event.tool_class is None:
            self.display = False
            return

        adapter = ToolUIDataAdapter(self._event.tool_class)
        display = adapter.get_result_display(self._event)

        widget = get_result_widget(
            self._event.tool_name,
            self._event.result,
            success=display.success,
            message=display.message,
            collapsed=self.collapsed,
            warnings=display.warnings,
            preview_lines=preview_lines,
        )
        await self._content_container.mount(widget)
        self.display = bool(widget.children)

    async def set_collapsed(self, collapsed: bool) -> None:
        if self.collapsed == collapsed:
            return
        self.collapsed = collapsed
        await self._render_result()

    async def toggle_collapsed(self) -> None:
        self.collapsed = not self.collapsed
        await self._render_result()
