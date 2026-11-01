from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from textual.widget import Widget

from privibe.cli.textual_ui.widgets.compact import CompactMessage
from privibe.cli.textual_ui.widgets.messages import (
    AssistantMessage,
    ReasoningMessage,
    UserMessage,
)
from privibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from privibe.cli.textual_ui.widgets.tools import (
    ToolCallMessage,
    ToolResultMessage,
    TraceLineMessage,
)
from privibe.core.tools.ui import ToolUIDataAdapter
from privibe.core.types import (
    AgentProfileChangedEvent,
    AssistantEvent,
    BaseEvent,
    CompactEndEvent,
    CompactStartEvent,
    ReasoningEvent,
    ToolCallEvent,
    ToolResultEvent,
    ToolStreamEvent,
    TraceLineEvent,
    UserMessageEvent,
    WaitingForInputEvent,
)
from privibe.core.utils import TaggedText

if TYPE_CHECKING:
    from privibe.cli.textual_ui.widgets.loading import LoadingWidget


class EventHandler:
    def __init__(
        self,
        mount_callback: Callable,
        get_tools_collapsed: Callable[[], bool],
        on_profile_changed: Callable[[], None] | None = None,
        get_reasoning_markdown: Callable[[], bool] | None = None,
    ) -> None:
        self.mount_callback = mount_callback
        self.get_tools_collapsed = get_tools_collapsed
        self.on_profile_changed = on_profile_changed
        self.get_reasoning_markdown = get_reasoning_markdown
        self.tool_calls: dict[str, ToolCallMessage] = {}
        # Nanny entries mounted after a call widget, so the result line
        # lands after the nanny's verdict rather than between call and verdict.
        self.nanny_entries: dict[str, Widget] = {}
        # Rows a sub-agent made under a running call, keyed by that parent
        # call's id: the last one mounted (the next child goes after it) and
        # the trace rows that can be updated in place, by entry id.
        self.last_child: dict[str, Widget] = {}
        self.trace_rows: dict[str, TraceLineMessage] = {}
        self.current_compact: CompactMessage | None = None
        self.current_streaming_message: AssistantMessage | None = None
        self.current_streaming_reasoning: ReasoningMessage | None = None

    async def handle_event(
        self,
        event: BaseEvent,
        loading_active: bool = False,
        loading_widget: LoadingWidget | None = None,
    ) -> ToolCallMessage | None:
        match event:
            case ReasoningEvent():
                await self._handle_reasoning_message(event)
            case AssistantEvent():
                await self._handle_assistant_message(event)
            case ToolCallEvent():
                await self.finalize_streaming()
                return await self._handle_tool_call(event, loading_widget)
            case ToolResultEvent():
                await self.finalize_streaming()
                sanitized_event = self._sanitize_event(event)
                await self._handle_tool_result(sanitized_event)
            case ToolStreamEvent():
                await self._handle_tool_stream(event)
            case TraceLineEvent():
                await self.finalize_streaming()
                await self._handle_trace_line(event)
            case CompactStartEvent():
                await self.finalize_streaming()
                await self._handle_compact_start()
            case CompactEndEvent():
                await self.finalize_streaming()
                await self._handle_compact_end(event)
            case AgentProfileChangedEvent():
                if self.on_profile_changed:
                    self.on_profile_changed()
            case UserMessageEvent():
                await self.finalize_streaming()
                if event.steering or event.auto_continue:
                    # The loop appended this one itself (typed mid-turn, or
                    # an auto-continue nudge); the app never mounted a bubble
                    # for it the way it does for a submitted message, so do
                    # it here with the rewind index.
                    await self.mount_callback(
                        UserMessage(
                            event.content,
                            message_index=event.message_index,
                            auto_continue=event.auto_continue,
                        )
                    )
            case WaitingForInputEvent():
                await self.finalize_streaming()
            case _:
                await self.finalize_streaming()
                await self._handle_unknown_event(event)
        return None

    def _sanitize_event(self, event: ToolResultEvent) -> ToolResultEvent:
        if isinstance(event, ToolResultEvent):
            # model_copy, not a field-by-field rebuild: every other field
            # (timing metas included) must reach the widget untouched.
            return event.model_copy(
                update={
                    "error": TaggedText.from_string(event.error).message
                    if event.error
                    else None,
                    "skip_reason": TaggedText.from_string(event.skip_reason).message
                    if event.skip_reason
                    else None,
                }
            )
        return event

    @staticmethod
    def _call_key(tool_call_id: str, parent_call_id: str | None) -> str:
        """The registry key of a call. A sub-agent's call is keyed under its
        parent, so a backend that numbers calls the same way in every loop
        cannot make a worker's call-1 collide with the parent's.
        """
        return f"{parent_call_id}/{tool_call_id}" if parent_call_id else tool_call_id

    async def _mount_child(self, parent_call_id: str, widget: Widget) -> None:
        """Mount a row a sub-agent made: after the last row of that parent,
        or right after the parent call itself (its nanny verdict included).
        """
        anchor = (
            self.last_child.get(parent_call_id)
            or self.nanny_entries.get(parent_call_id)
            or self.tool_calls.get(parent_call_id)
        )
        await self.mount_callback(widget, after=anchor)
        self.last_child[parent_call_id] = widget

    async def _handle_tool_call(
        self, event: ToolCallEvent, loading_widget: LoadingWidget | None = None
    ) -> ToolCallMessage | None:
        tool_call_id = event.tool_call_id
        key = self._call_key(tool_call_id, event.parent_call_id)
        existing_tool_call = self.tool_calls.get(key) if tool_call_id else None
        if existing_tool_call:
            existing_tool_call.update_event(event)
            tool_call = existing_tool_call
        else:
            tool_call = ToolCallMessage(event, collapsed=self.get_tools_collapsed())
            if tool_call_id:
                self.tool_calls[key] = tool_call
            if event.parent_call_id:
                await self._mount_child(event.parent_call_id, tool_call)
            else:
                await self.mount_callback(tool_call)

        if loading_widget and event.tool_class and not event.parent_call_id:
            adapter = ToolUIDataAdapter(event.tool_class)
            loading_widget.set_status(adapter.get_status_text())

        return tool_call

    async def _handle_tool_result(self, event: ToolResultEvent) -> None:
        tools_collapsed = self.get_tools_collapsed()
        key = self._call_key(event.tool_call_id, event.parent_call_id)
        call_widget = self.tool_calls.get(key) if event.tool_call_id else None

        tool_result = ToolResultMessage(event, call_widget, collapsed=tools_collapsed)
        if event.parent_call_id:
            # Under its own call, which is the parent's last child unless a
            # sibling ran concurrently and landed after it.
            await self.mount_callback(tool_result, after=call_widget)
            if (
                call_widget is None
                or self.last_child.get(event.parent_call_id) is call_widget
            ):
                self.last_child[event.parent_call_id] = tool_result
        else:
            anchor = (
                self.nanny_entries.pop(event.tool_call_id, None)
                or self.last_child.pop(event.tool_call_id, None)
                or call_widget
            )
            await self.mount_callback(tool_result, after=anchor)
            self._forget_children(event.tool_call_id)

        if event.tool_call_id and key in self.tool_calls:
            del self.tool_calls[key]

    def _forget_children(self, parent_call_id: str) -> None:
        self.last_child.pop(parent_call_id, None)
        prefix = f"{parent_call_id}/"
        for key in [k for k in self.tool_calls if k.startswith(prefix)]:
            del self.tool_calls[key]
        for key in [k for k in self.trace_rows if k.startswith(prefix)]:
            del self.trace_rows[key]

    async def _handle_trace_line(self, event: TraceLineEvent) -> None:
        """A sub-agent's non-call row: new, or an update of the row with the
        same entry id (a section header getting its timing).
        """
        if event.entry_id is not None:
            key = f"{event.tool_call_id}/{event.entry_id}"
            existing = self.trace_rows.get(key)
            if existing is not None:
                existing.set_text(event.message)
                return
        row = TraceLineMessage(
            event.message,
            depth=event.depth,
            entry_id=event.entry_id,
            parent_call_id=event.tool_call_id,
            resume_hint=event.resume_hint,
        )
        if event.entry_id is not None:
            self.trace_rows[f"{event.tool_call_id}/{event.entry_id}"] = row
        await self._mount_child(event.tool_call_id, row)

    async def _handle_tool_stream(self, event: ToolStreamEvent) -> None:
        tool_call = self.tool_calls.get(event.tool_call_id)
        if tool_call:
            tool_call.set_stream_message(event.message, event.entry_id)

    async def _handle_assistant_message(self, event: AssistantEvent) -> None:
        if self.current_streaming_reasoning is not None:
            self.current_streaming_reasoning.stop_spinning()
            await self.current_streaming_reasoning.stop_stream()
            self.current_streaming_reasoning = None

        if self.current_streaming_message is None:
            msg = AssistantMessage(event.content, history_key=event.message_id)
            self.current_streaming_message = msg
            await self.mount_callback(msg)
        else:
            await self.current_streaming_message.append_content(event.content)

    async def _handle_reasoning_message(self, event: ReasoningEvent) -> None:
        if self.current_streaming_message is not None:
            await self.current_streaming_message.stop_stream()
            if self.current_streaming_message.is_stripped_content_empty():
                await self.current_streaming_message.remove()
            self.current_streaming_message = None

        if self.current_streaming_reasoning is None:
            tools_collapsed = self.get_tools_collapsed()
            msg = ReasoningMessage(
                event.content,
                collapsed=tools_collapsed,
                history_key=event.message_id,
                markdown=self.get_reasoning_markdown()
                if self.get_reasoning_markdown
                else False,
            )
            self.current_streaming_reasoning = msg
            await self.mount_callback(msg)
        else:
            await self.current_streaming_reasoning.append_content(event.content)

    async def _handle_compact_start(self) -> None:
        compact_msg = CompactMessage()
        self.current_compact = compact_msg
        await self.mount_callback(compact_msg)

    async def _handle_compact_end(self, event: CompactEndEvent) -> None:
        if self.current_compact:
            self.current_compact.set_complete(
                old_tokens=event.old_context_tokens, new_tokens=event.new_context_tokens
            )
            self.current_compact = None

    async def _handle_unknown_event(self, event: BaseEvent) -> None:
        await self.mount_callback(NoMarkupStatic(str(event), classes="unknown-event"))

    async def finalize_streaming(self) -> None:
        if self.current_streaming_reasoning is not None:
            self.current_streaming_reasoning.stop_spinning()
            await self.current_streaming_reasoning.stop_stream()
            self.current_streaming_reasoning = None
        if self.current_streaming_message is not None:
            await self.current_streaming_message.stop_stream()
            self.current_streaming_message = None

    def stop_current_tool_call(self, success: bool = True) -> None:
        for tool_call in self.tool_calls.values():
            tool_call.stop_spinning(success=success)
        self.tool_calls.clear()
        self.last_child.clear()
        self.trace_rows.clear()

    def stop_current_compact(self) -> None:
        if self.current_compact:
            self.current_compact.stop_spinning(success=False)
            self.current_compact = None
