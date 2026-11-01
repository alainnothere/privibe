"""What a tool that runs a sub-agent shows of it: events for the parent.

The parent transcript is append-only: every call leaves its line and its
result. A sub-agent used to run behind a one-line ticker and a text log in
the parent call's fold-out, re-rendered whole on every entry. The Tracer now
hands the sub-agent's tool calls and results straight through, stamped with
the parent call's id and a depth, so the UI draws them as the same rows it
draws for the top level, indented. What is not a tool call (a role header, a
verifier line, the closing summary) goes out as a TraceLineEvent, a row of
its own; rows carry an id so a header can be stamped with its timing once the
section ends. The console and ACP get the same events and print them as text.
"""

from __future__ import annotations

from itertools import count
import time

from privibe.core.types import (
    AssistantEvent,
    BaseEvent,
    ToolCallEvent,
    ToolResultEvent,
    TraceLineEvent,
)

SECTION_WIDTH = 60
# A section label longer than this is cut: the label is a heading, and a
# worker's whole task text in it wrapped the rule onto two or three rows.
LABEL_CHARS = 48
_MINUTE = 60
ANSWER_LINES = 40


def fit_label(label: str, limit: int = LABEL_CHARS) -> str:
    """The label's first non-empty line, cut to `limit` characters."""
    first = next((ln.strip() for ln in label.splitlines() if ln.strip()), "")
    if len(first) > limit:
        return first[: limit - 1].rstrip() + "…"
    return first


def section_line(label: str, stamp: str = "") -> str:
    head = f"── {fit_label(label)} "
    tail = f" {stamp}" if stamp else ""
    fill = max(2, SECTION_WIDTH - len(head) - len(tail))
    return head + "─" * fill + tail


def _secs(seconds: float) -> str:
    if seconds < _MINUTE:
        return f"{seconds:.0f}s"
    minutes, rest = divmod(int(seconds), _MINUTE)
    return f"{minutes}m{rest:02d}s"


def _cap(text: str, max_lines: int, more: str = "") -> str:
    lines = text.rstrip().splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    hidden = len(lines) - max_lines
    tail = f"({hidden} more lines{', ' + more if more else ''})"
    return "\n".join([*lines[:max_lines], tail])


def _indent(text: str, by: str = "  ") -> str:
    return "\n".join(by + line for line in text.splitlines())


class Tracer:
    """One per tool run. Call start_section when a role call begins, feed
    every event of the sub-agent loop to on_event, then end_section with
    the role's answer. Each method returns the events to emit.
    """

    def __init__(self, tool_name: str, tool_call_id: str, depth: int = 1) -> None:
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self.depth = depth
        self._ids = count(1)
        self._section_id: str | None = None
        self._section_label = ""
        self._section_started = 0.0
        self._section_calls = 0

    @property
    def section_label(self) -> str:
        return self._section_label

    def _row(self, message: str, entry_id: str | None = None) -> TraceLineEvent:
        return TraceLineEvent(
            tool_call_id=self.tool_call_id,
            message=message,
            depth=self.depth,
            entry_id=entry_id,
        )

    def _new_id(self) -> str:
        return f"{self.tool_call_id}:{next(self._ids)}"

    def line(self, text: str) -> list[BaseEvent]:
        """A plain row, no id: never updated."""
        return [self._row(text)]

    def start_section(self, label: str) -> list[BaseEvent]:
        self._section_id = self._new_id()
        self._section_label = label
        self._section_started = time.time()
        self._section_calls = 0
        return [self._row(section_line(label), self._section_id)]

    def on_event(self, event: BaseEvent) -> list[BaseEvent]:
        """The sub-agent's event as the parent should see it: its tool calls
        and results pass through under the parent call's id, one level deeper
        (a grandchild keeps its own depth on top). Everything else is dropped,
        except a middleware stop, which gets a row.
        """
        if isinstance(event, (ToolCallEvent, ToolResultEvent)):
            if isinstance(event, ToolCallEvent):
                self._section_calls += 1
            return [
                event.model_copy(
                    update={
                        "parent_call_id": self.tool_call_id,
                        "depth": self.depth + event.depth,
                    }
                )
            ]
        if isinstance(event, TraceLineEvent):
            return [event.model_copy(update={"depth": self.depth + event.depth})]
        if isinstance(event, AssistantEvent) and event.stopped_by_middleware:
            return self.line("✕ stopped by middleware")
        return []

    def end_section(self, answer: str, *, more: str = "") -> list[BaseEvent]:
        """Stamp the section header with its time and call count, then the
        role's written answer, capped, with `more` naming where the rest is.
        """
        events: list[BaseEvent] = []
        if self._section_id is not None:
            calls = self._section_calls
            stamp = f"{_secs(time.time() - self._section_started)}"
            if calls:
                stamp += f" · {calls} call{'s' if calls != 1 else ''}"
            events.append(
                self._row(section_line(self._section_label, stamp), self._section_id)
            )
            self._section_id = None
        if answer.strip():
            events.append(self._row(_indent(_cap(answer, ANSWER_LINES, more))))
        return events

    def close(self, label: str, stamp: str = "", detail: str = "") -> list[BaseEvent]:
        """The trace's last row: a rule naming how the run ended, with the
        detail under it.
        """
        text = section_line(label, stamp)
        if detail.strip():
            text += "\n" + _indent(detail.strip())
        return self.line(text)
