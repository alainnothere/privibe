from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any
from weakref import WeakKeyDictionary

from pydantic import BaseModel, ValidationError
from textual.widget import Widget

from privibe.cli.textual_ui.widgets.messages import AssistantMessage, UserMessage
from privibe.cli.textual_ui.widgets.tools import ToolCallMessage, ToolResultMessage
from privibe.core.types import (
    LLMMessage,
    MessageMeta,
    Role,
    ToolCall,
    ToolCallEvent,
    ToolOutcome,
    ToolResultEvent,
)

ToolClasses = Mapping[str, Any]


def _tool_models(tool_class: Any) -> tuple[type[BaseModel], type[BaseModel]] | None:
    introspect = getattr(tool_class, "_get_tool_args_results", None)
    if introspect is None:
        return None
    try:
        args_model, result_model = introspect()
    except Exception:
        return None
    return args_model, result_model


def history_tool_call_event(
    tool_call: ToolCall, assistant: LLMMessage, tool_classes: ToolClasses
) -> ToolCallEvent | None:
    """Rebuild the ToolCallEvent for a stored call, or None when the tool is
    not known to this process (MCP server down, tool retired).

    The arguments are the JSON the model sent, stored verbatim on the
    assistant message; validated into the live args model so the same
    adapter that drew the live line draws this one. Arguments the live
    model rejects (the tool changed shape) leave args=None, which the
    adapter renders as the bare tool name.
    """
    tool_name = tool_call.function.name or ""
    tool_class = tool_classes.get(tool_name)
    if tool_class is None:
        return None
    args: BaseModel | None = None
    models = _tool_models(tool_class)
    if models is not None:
        try:
            args = models[0].model_validate(
                json.loads(tool_call.function.arguments or "{}")
            )
        except (ValidationError, ValueError, TypeError):
            args = None
    meta = assistant.meta
    return ToolCallEvent(
        tool_call_id=tool_call.id or "",
        tool_name=tool_name,
        tool_class=tool_class,
        args=args,
        requested_at=meta.request_sent_at if meta else None,
    )


def history_tool_result_event(
    msg: LLMMessage,
    tool_name: str,
    tool_classes: ToolClasses,
    *,
    request_meta: MessageMeta | None = None,
    first_in_message: bool = False,
) -> ToolResultEvent | None:
    """Rebuild the ToolResultEvent for a stored tool message from its meta.

    None when the tool is unknown here or the message carries no meta
    (sessions written before meta existed): the caller falls back to the
    bare content widget. Duration is derived from the two stored instants,
    never stored itself.
    """
    meta = msg.meta
    tool_class = tool_classes.get(tool_name)
    if meta is None or tool_class is None or meta.tool_outcome is None:
        return None

    result: BaseModel | None = None
    error: str | None = None
    skipped = False
    skip_reason: str | None = None
    match meta.tool_outcome:
        case ToolOutcome.success:
            models = _tool_models(tool_class)
            if models is not None and meta.tool_result is not None:
                try:
                    result = models[1].model_validate(meta.tool_result)
                except ValidationError:
                    result = None
        case ToolOutcome.failure:
            error = msg.content or "error"
        case ToolOutcome.skipped:
            skipped = True
            skip_reason = msg.content or None

    duration: float | None = None
    if meta.tool_started_at is not None and meta.tool_finished_at is not None:
        duration = meta.tool_finished_at - meta.tool_started_at

    return ToolResultEvent(
        tool_name=tool_name,
        tool_class=tool_class,
        result=result,
        error=error,
        skipped=skipped,
        skip_reason=skip_reason,
        duration=duration,
        tool_call_id=msg.tool_call_id or "",
        file_diff=meta.file_diff,
        request_meta=request_meta,
        tool_meta=meta,
        first_in_message=first_in_message,
    )


def non_system_history_messages(messages: Sequence[LLMMessage]) -> list[LLMMessage]:
    return [msg for msg in messages if msg.role != Role.system]


def build_tool_call_map(messages: Sequence[LLMMessage]) -> dict[str, str]:
    tool_call_map: dict[str, str] = {}
    for msg in messages:
        if msg.role != Role.assistant or not msg.tool_calls:
            continue
        for tool_call in msg.tool_calls:
            if tool_call.id:
                tool_call_map[tool_call.id] = tool_call.function.name or "unknown"
    return tool_call_map


def build_history_widgets(
    batch: Sequence[LLMMessage],
    tool_call_map: dict[str, str],
    *,
    start_index: int,
    tools_collapsed: bool,
    history_widget_indices: WeakKeyDictionary[Widget, int],
    tool_classes: ToolClasses | None = None,
) -> list[Widget]:
    """Widgets for a batch of stored messages (resume and load-more).

    Tool calls and results are rebuilt into the same events the live path
    mounts, so the same adapter and per-tool widgets draw them. A call
    widget is linked to its result widget exactly like the live path, so
    the result line replaces the call line on mount. Without tool_classes
    (or for tools this process does not know) the bare widgets are used.
    """
    tool_classes = tool_classes or {}
    call_widgets: dict[str, ToolCallMessage] = {}
    # call id -> (meta of the assistant message that made it, first in batch)
    call_requests: dict[str, tuple[MessageMeta | None, bool]] = {}
    widgets: list[Widget] = []

    for history_index, msg in zip(
        range(start_index, start_index + len(batch)), batch, strict=True
    ):
        if msg.injected:
            continue
        match msg.role:
            case Role.user:
                if msg.content:
                    # history_index is 0-based in non-system messages;
                    # agent_loop.messages index = history_index + 1 (system msg at 0)
                    widget = UserMessage(msg.content, message_index=history_index + 1)
                    widgets.append(widget)
                    history_widget_indices[widget] = history_index

            case Role.assistant:
                if msg.content:
                    assistant_widget = AssistantMessage(msg.content)
                    widgets.append(assistant_widget)
                    history_widget_indices[assistant_widget] = history_index

                if msg.tool_calls:
                    for position, tool_call in enumerate(msg.tool_calls):
                        tool_name = tool_call.function.name or "unknown"
                        if tool_call.id:
                            tool_call_map[tool_call.id] = tool_name
                            call_requests[tool_call.id] = (msg.meta, position == 0)
                        call_event = history_tool_call_event(
                            tool_call, msg, tool_classes
                        )
                        widget = ToolCallMessage(
                            call_event, tool_name=tool_name, history=True
                        )
                        if tool_call.id:
                            call_widgets[tool_call.id] = widget
                        widgets.append(widget)
                        history_widget_indices[widget] = history_index

            case Role.tool:
                tool_name = msg.name or tool_call_map.get(
                    msg.tool_call_id or "", "tool"
                )
                request_meta, first = call_requests.get(
                    msg.tool_call_id or "", (None, False)
                )
                result_event = history_tool_result_event(
                    msg,
                    tool_name,
                    tool_classes,
                    request_meta=request_meta,
                    first_in_message=first,
                )
                if result_event is not None:
                    widget = ToolResultMessage(
                        result_event,
                        call_widgets.get(msg.tool_call_id or ""),
                        collapsed=tools_collapsed,
                    )
                else:
                    widget = ToolResultMessage(
                        tool_name=tool_name,
                        content=msg.content,
                        collapsed=tools_collapsed,
                    )
                widgets.append(widget)
                history_widget_indices[widget] = history_index

    return widgets


def split_history_tail(
    history_messages: list[LLMMessage], tail_size: int
) -> tuple[list[LLMMessage], list[LLMMessage], int]:
    tail_messages = history_messages[-tail_size:]
    backfill_messages = history_messages[:-tail_size]
    tail_start_index = len(history_messages) - len(tail_messages)
    return tail_messages, backfill_messages, tail_start_index


def build_history_id_index(messages: Sequence[LLMMessage]) -> dict[str, int]:
    """Map every stable message id to its non-system history index.

    Keys: message_id and reasoning_message_id verbatim, tool call ids as
    "call:<id>" (the assistant message holding the call) and "result:<id>"
    (the tool result message), matching the history_key each widget exposes.
    """
    id_index: dict[str, int] = {}
    for i, msg in enumerate(messages):
        if msg.message_id:
            id_index[msg.message_id] = i
        if msg.reasoning_message_id:
            id_index[msg.reasoning_message_id] = i
        if msg.role == Role.tool and msg.tool_call_id:
            id_index[f"result:{msg.tool_call_id}"] = i
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call.id:
                    id_index[f"call:{tool_call.id}"] = i

    return id_index


def resolve_visible_history_indices(
    children: list[Widget],
    history_widget_indices: WeakKeyDictionary[Widget, int],
    id_index: dict[str, int],
) -> list[int]:
    """Resolve each mounted widget to the history index of its message.

    Resolution order: the index stamped at build time (resume/load-more
    widgets), the message_index carried by live user messages, then the
    stable-id lookup for live-streamed widgets. Widgets that resolve nowhere
    (slash-command echoes, widgets whose message has not landed in history
    yet, decorative widgets) contribute nothing.
    """
    indices: list[int] = []
    for child in children:
        stamped = history_widget_indices.get(child)
        if stamped is not None:
            indices.append(stamped)
            continue
        if isinstance(child, UserMessage) and child.message_index is not None:
            # message_index is into agent_loop.messages (system message at 0)
            indices.append(child.message_index - 1)
            continue
        key = getattr(child, "history_key", None)
        if key is not None and (resolved := id_index.get(key)) is not None:
            indices.append(resolved)
    return indices
