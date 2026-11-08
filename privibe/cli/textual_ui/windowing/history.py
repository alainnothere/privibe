from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
from typing import Any
from weakref import WeakKeyDictionary

from pydantic import BaseModel, ValidationError
from textual.widget import Widget

from privibe.cli.textual_ui.widgets.messages import (
    AssistantMessage,
    NannyMessage,
    UserMessage,
)
from privibe.cli.textual_ui.widgets.tools import (
    ToolCallMessage,
    ToolResultMessage,
    TraceLineMessage,
)
from privibe.core.tools.subagent_trace import section_line
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
# Loads a sub-agent session's messages by its directory; [] when it is gone.
ChildLoader = Callable[[Path], Sequence[LLMMessage]]
# Tools whose result names the sub-agent sessions that ran under them.
_SUBAGENT_TOOLS = {"task", "orchestrate"}
_CALLS_FILENAME = "calls.json"


def subagent_children(
    tool_name: str, result: BaseModel | None
) -> list[tuple[str, str, Path]]:
    """(section label, row tag, session dir) for every sub-agent run a stored
    result points at: the task tool's one session, or each role call of an
    orchestrate run as its ledger's calls.json lists them. Labels are "" for
    a run with no header (task); the tag is what its rows carry before the
    tool name (the agent name, "worker 2/10 · task 3"). Missing paths are
    kept; the loader decides.
    """
    if result is None or tool_name not in _SUBAGENT_TOOLS:
        return []
    if tool_name == "task":
        session_dir = getattr(result, "session_dir", "")
        agent = str(getattr(result, "agent", "") or "")
        return [("", agent, Path(session_dir))] if session_dir else []
    ledger_dir = getattr(result, "ledger_dir", "")
    if not ledger_dir:
        return []
    calls_path = Path(ledger_dir) / _CALLS_FILENAME
    if not calls_path.is_file():
        return []
    try:
        raw = json.loads(calls_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    children: list[tuple[str, str, Path]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        session_dir = str(entry.get("session_dir", ""))
        children.append((
            str(entry.get("label", "")),
            str(entry.get("tag", "")),
            Path(session_dir),
        ))
    return children


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


def build_history_widgets(  # noqa: PLR0913
    batch: Sequence[LLMMessage],
    tool_call_map: dict[str, str],
    *,
    start_index: int,
    tools_collapsed: bool,
    history_widget_indices: WeakKeyDictionary[Widget, int],
    tool_classes: ToolClasses | None = None,
    nanny_decisions: Mapping[str, Any] | None = None,
    child_loader: ChildLoader | None = None,
    depth: int = 0,
    section: str | None = None,
) -> list[Widget]:
    """Widgets for a batch of stored messages (resume and load-more).
    `section` is the tag every call row of this batch carries before its
    tool name (a sub-agent session's rows).

    With `child_loader`, a task or orchestrate result also brings back the
    rows its sub-agents made: each worker session the result points at is
    loaded and built the same way, one level deeper, and placed under the
    parent call, so a resumed conversation shows what the live one showed.
    Child widgets are not stamped into history_widget_indices: rewind and
    windowing key on the parent's messages, and a child row follows its
    parent call wherever that goes.

    `nanny_decisions` maps a tool_call_id to the nanny's stored decision on
    it; a NannyMessage is placed right after that call's widget, rebuilt
    from the sidecar at render time, never from either conversation.

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
                    widget = UserMessage(
                        msg.content,
                        message_index=history_index + 1,
                        auto_continue=msg.auto_continue,
                    )
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
                            call_event,
                            tool_name=tool_name,
                            history=True,
                            depth=depth,
                            section=section,
                        )
                        if tool_call.id:
                            call_widgets[tool_call.id] = widget
                        widgets.append(widget)
                        history_widget_indices[widget] = history_index
                        decision = (
                            nanny_decisions.get(tool_call.id)
                            if nanny_decisions and tool_call.id
                            else None
                        )
                        if decision is not None:
                            entry = NannyMessage(decision)
                            widgets.append(entry)
                            history_widget_indices[entry] = history_index

            case Role.tool:
                _append_tool_result(
                    msg,
                    history_index,
                    widgets,
                    call_widgets,
                    call_requests,
                    tool_call_map,
                    tools_collapsed=tools_collapsed,
                    history_widget_indices=history_widget_indices,
                    tool_classes=tool_classes,
                    child_loader=child_loader,
                    depth=depth,
                )

    return widgets


def _append_tool_result(  # noqa: PLR0913  the builder's state, passed through
    msg: LLMMessage,
    history_index: int,
    widgets: list[Widget],
    call_widgets: dict[str, ToolCallMessage],
    call_requests: dict[str, tuple[MessageMeta | None, bool]],
    tool_call_map: dict[str, str],
    *,
    tools_collapsed: bool,
    history_widget_indices: WeakKeyDictionary[Widget, int],
    tool_classes: ToolClasses,
    child_loader: ChildLoader | None,
    depth: int,
) -> None:
    """The widget for a stored tool result, linked to its call widget, plus
    (with a child_loader) the rows of the sub-agents that ran under it,
    placed right under the parent call where the live path puts them.
    """
    tool_name = msg.name or tool_call_map.get(msg.tool_call_id or "", "tool")
    request_meta, first = call_requests.get(msg.tool_call_id or "", (None, False))
    result_event = history_tool_result_event(
        msg, tool_name, tool_classes, request_meta=request_meta, first_in_message=first
    )
    call_widget = call_widgets.get(msg.tool_call_id or "")
    if result_event is not None:
        widget = ToolResultMessage(
            result_event, call_widget, collapsed=tools_collapsed, depth=depth
        )
    else:
        widget = ToolResultMessage(
            tool_name=tool_name,
            content=msg.content,
            collapsed=tools_collapsed,
            depth=depth,
        )
    if child_loader is not None and result_event is not None:
        children = _child_widgets(
            tool_name,
            result_event.result,
            child_loader,
            tools_collapsed=tools_collapsed,
            tool_classes=tool_classes,
            depth=depth + 1,
        )
        if children:
            at = (
                widgets.index(call_widget) + 1
                if call_widget in widgets
                else len(widgets)
            )
            widgets[at:at] = children
    widgets.append(widget)
    history_widget_indices[widget] = history_index


def _child_widgets(
    tool_name: str,
    result: BaseModel | None,
    child_loader: ChildLoader,
    *,
    tools_collapsed: bool,
    tool_classes: ToolClasses | None,
    depth: int,
) -> list[Widget]:
    """The rows of every sub-agent session a stored result points at, each
    under its section header when the run had one (orchestrate's role calls).
    """
    widgets: list[Widget] = []
    for label, tag, session_dir in subagent_children(tool_name, result):
        if label:
            widgets.append(TraceLineMessage(section_line(label), depth=depth))
        if not str(session_dir):
            continue
        messages = child_loader(session_dir)
        if not messages:
            continue
        child_messages = non_system_history_messages(messages)
        rows = build_history_widgets(
            child_messages,
            build_tool_call_map(child_messages),
            start_index=0,
            tools_collapsed=tools_collapsed,
            history_widget_indices=WeakKeyDictionary(),
            tool_classes=tool_classes,
            child_loader=child_loader,
            depth=depth,
            section=tag or None,
        )
        # Tool rows only, as the live trace shows them: the worker's prompt
        # and its written answer stay in its own session.
        widgets.extend(
            w
            for w in rows
            if isinstance(w, (ToolCallMessage, ToolResultMessage, TraceLineMessage))
        )
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
