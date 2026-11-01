"""Where a rewind can land, and what landing there means.

Three kinds of target, all rows of #messages:

- a user message: the conversation is cut at it and its words come back to
  the input (the original rewind);
- a tool result row that closes a batch mid-turn: the cut is right after the
  batch, the tail is the batch's results, and the next message is a steering
  message into the turn, the same slot the loop uses for a message typed
  mid-turn; nothing is shimmed, nothing is lied about;
- a checkpoint row an orchestrate run left: the cut is before the call that
  ran it, and the input is pre-filled with the request to resume that run
  from that checkpoint. When the call was the first thing in its turn the cut
  is the user message that asked, with the request appended to its words.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import TYPE_CHECKING

from textual.widget import Widget

from privibe.cli.textual_ui.widgets.messages import UserMessage
from privibe.cli.textual_ui.widgets.tools import (
    ToolCallMessage,
    ToolResultMessage,
    TraceLineMessage,
)
from privibe.cli.textual_ui.windowing.history import (
    build_history_id_index,
    non_system_history_messages,
)
from privibe.core.types import LLMMessage, Role

if TYPE_CHECKING:
    from collections.abc import Sequence


class RewindKind(StrEnum):
    USER = auto()
    BATCH = auto()
    INSIDE = auto()


@dataclass(frozen=True)
class RewindPlan:
    kind: RewindKind
    # Index into agent_loop.messages: for USER the message itself, otherwise
    # the first message dropped.
    cut_index: int
    # The first row of #messages to remove (everything after it goes too).
    first_removed: Widget
    preview: str
    # Text for the input: the whole of it for BATCH/INSIDE, appended to the
    # user message's own words for USER.
    prefill: str
    steering: bool
    nanny_note: str


def _loop_index(widget: Widget, id_index: dict[str, int]) -> int | None:
    """The agent_loop.messages index of the message a tool widget stands for."""
    key = getattr(widget, "history_key", None)
    if key is None:
        return None
    non_system = id_index.get(key)
    return None if non_system is None else non_system + 1


def rewind_targets(
    children: Sequence[Widget],
    messages: Sequence[LLMMessage],
    boundaries: Sequence[int],
) -> list[Widget]:
    """The rows a rewind can land on, in screen order."""
    id_index = build_history_id_index(non_system_history_messages(messages))
    boundary_set = set(boundaries)
    targets: list[Widget] = []
    for child in children:
        if isinstance(child, UserMessage) and child.message_index is not None:
            targets.append(child)
        elif isinstance(child, ToolResultMessage) and child.depth == 0:
            index = _loop_index(child, id_index)
            if index is None or index + 1 not in boundary_set:
                continue
            # A boundary followed by the next turn's prompt is that user
            # message's target; the end of the conversation is a target of
            # its own (continue after a run that was cut short).
            if index + 1 < len(messages) and messages[index + 1].role == Role.user:
                continue
            targets.append(child)
        elif (
            isinstance(child, TraceLineMessage)
            and child.resume_hint
            and child.parent_call_id
            and f"call:{child.parent_call_id}" in id_index
        ):
            targets.append(child)
    return targets


def rewind_plan(
    widget: Widget,
    children: Sequence[Widget],
    messages: Sequence[LLMMessage],
    boundaries: Sequence[int],
) -> RewindPlan | None:
    """What rewinding to `widget` means, or None when it is not a target."""
    id_index = build_history_id_index(non_system_history_messages(messages))
    if isinstance(widget, UserMessage) and widget.message_index is not None:
        words = widget.get_content().strip().replace("\n", " ")[:80]
        return RewindPlan(
            kind=RewindKind.USER,
            cut_index=widget.message_index,
            first_removed=widget,
            preview=widget.get_content(),
            prefill="",
            steering=False,
            nanny_note=(
                f"the conversation was rewound to the user message '{words}'; "
                "calls judged after it were dropped and may be asked again"
            ),
        )
    if isinstance(widget, ToolResultMessage):
        index = _loop_index(widget, id_index)
        if index is None or index + 1 not in set(boundaries):
            return None
        position = children.index(widget) if widget in children else len(children)
        after = children[position + 1] if position + 1 < len(children) else widget
        summary = _result_summary(widget)
        return RewindPlan(
            kind=RewindKind.BATCH,
            cut_index=index + 1,
            first_removed=after if after is not widget else widget,
            preview=f"after {summary}",
            prefill="",
            steering=True,
            nanny_note=(
                f"the conversation was cut after the tool batch ending in '{summary}'; "
                "what follows is the user steering the turn from there"
            ),
        )
    if (
        isinstance(widget, TraceLineMessage)
        and widget.resume_hint
        and widget.parent_call_id
    ):
        call_non_system = id_index.get(f"call:{widget.parent_call_id}")
        if call_non_system is None:
            return None
        call_index = call_non_system + 1
        call_widget = next(
            (
                c
                for c in children
                if isinstance(c, ToolCallMessage)
                and c.tool_call_id == widget.parent_call_id
            ),
            None,
        )
        before = messages[call_index - 1] if call_index >= 1 else None
        if before is not None and before.role == Role.user:
            # The call was the turn's first move: land on the prompt itself.
            prompt_widget = _user_widget_before(children, call_widget)
            return RewindPlan(
                kind=RewindKind.USER,
                cut_index=call_index - 1,
                first_removed=prompt_widget or call_widget or widget,
                preview=widget.text,
                prefill="\n\n" + widget.resume_hint,
                steering=False,
                nanny_note=(
                    "the conversation was rewound into an orchestrate run "
                    f"({widget.text.strip()}); the request will be re-issued"
                ),
            )
        return RewindPlan(
            kind=RewindKind.INSIDE,
            cut_index=call_index,
            first_removed=call_widget or widget,
            preview=widget.text,
            prefill=widget.resume_hint,
            steering=True,
            nanny_note=(
                "the conversation was cut before an orchestrate call to resume it "
                f"({widget.text.strip()})"
            ),
        )
    return None


def _result_summary(widget: ToolResultMessage) -> str:
    return widget.get_content()


def _user_widget_before(
    children: Sequence[Widget], widget: Widget | None
) -> Widget | None:
    if widget is None or widget not in children:
        return None
    for child in reversed(children[: children.index(widget)]):
        if isinstance(child, UserMessage) and child.message_index is not None:
            return child
    return None
