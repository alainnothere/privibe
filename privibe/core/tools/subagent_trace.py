"""What a tool that runs a sub-agent shows of it, as stream entries.

The parent transcript is append-only: every call leaves its line and its
result, and ctrl+o opens all of it. A sub-agent ran behind a one-line
ticker that overwrote itself, so by the time it finished nobody could say
what it had done. The Tracer turns the sub-agent's events into entries
for the parent's tool call body: a section line when a role call starts,
one line per nested tool call with the same text the top level prints,
a few lines of each result, and the role's written answer at the end.
Entries carry an id so the ticker's "running" line and the section
header can be updated in place when the result or the timing is known.
"""

from __future__ import annotations

from itertools import count
import time

from pydantic import BaseModel

from privibe.core.tools.ui import ToolUIDataAdapter
from privibe.core.types import (
    AssistantEvent,
    BaseEvent,
    ToolCallEvent,
    ToolResultEvent,
    ToolStreamEvent,
)

SECTION_WIDTH = 60
_MINUTE = 60
ANSWER_LINES = 40
# Tools whose result body is the file itself: the top level shows a diff
# or nothing for them, so the trace shows the one-liner only.
_NO_PREVIEW = {"read_file", "hashed_read", "write_file", "todo"}


def section_line(label: str, stamp: str = "") -> str:
    head = f"── {label} "
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


def _preview_text(result: BaseModel) -> str:
    """The result's biggest text field: stdout for bash, matches for grep,
    whatever a tool calls its body. Empty when nothing is text.
    """
    best = ""
    for name in type(result).model_fields:
        value = getattr(result, name, None)
        if isinstance(value, str) and len(value) > len(best):
            best = value
    return best


def _indent(text: str, by: str = "  ") -> str:
    return "\n".join(by + line for line in text.splitlines())


class Tracer:
    """One per tool run. Call start_section when a role call begins, feed
    every event of the sub-agent loop to on_event, then end_section with
    the role's answer. Each method returns the stream events to emit.
    """

    def __init__(self, tool_name: str, tool_call_id: str, preview_lines: int) -> None:
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self.preview_lines = preview_lines
        self._ids = count(1)
        self._section_id: str | None = None
        self._section_label = ""
        self._section_started = 0.0
        self._section_calls = 0
        self._open_calls: dict[str, tuple[str, str]] = {}  # call id -> (entry, line)

    def _event(self, message: str, entry_id: str | None = None) -> ToolStreamEvent:
        return ToolStreamEvent(
            tool_name=self.tool_name,
            message=message,
            tool_call_id=self.tool_call_id,
            entry_id=entry_id,
        )

    def _new_id(self) -> str:
        return f"{self.tool_call_id}:{next(self._ids)}"

    def line(self, text: str) -> list[ToolStreamEvent]:
        """A plain entry, no id: never updated."""
        return [self._event(text)]

    def start_section(self, label: str) -> list[ToolStreamEvent]:
        self._section_id = self._new_id()
        self._section_label = label
        self._section_started = time.time()
        self._section_calls = 0
        self._open_calls.clear()
        return [self._event(section_line(label), self._section_id)]

    def on_event(self, event: BaseEvent) -> list[ToolStreamEvent]:
        if isinstance(event, ToolCallEvent) and event.args is not None:
            summary = (
                ToolUIDataAdapter(event.tool_class).get_call_display(event).summary
            )
            entry = self._new_id()
            self._open_calls[event.tool_call_id] = (entry, summary)
            self._section_calls += 1
            return [self._event(f"⋯ {summary}", entry)]
        if isinstance(event, ToolResultEvent):
            entry, summary = self._open_calls.pop(
                event.tool_call_id, (None, event.tool_name)
            )
            return [self._event(self._result_entry(event, summary), entry)]
        if isinstance(event, AssistantEvent) and event.stopped_by_middleware:
            return self.line("✕ stopped by middleware")
        return []

    def _result_entry(self, event: ToolResultEvent, summary: str) -> str:
        if event.error:
            return f"✕ {summary}\n{_indent(_cap(event.error, self.preview_lines))}"
        if event.skipped:
            return f"✕ {summary} (skipped: {event.skip_reason or 'by the user'})"
        mark = "✓"
        body = ""
        if event.tool_class is not None and event.result is not None:
            display = ToolUIDataAdapter(event.tool_class).get_result_display(event)
            if not display.success:
                mark = "✕"
            # The one-liner says what happened; the call summary said what
            # was asked. Both when they differ, which is nearly always.
            summary = display.message if display.message else summary
            if event.tool_name not in _NO_PREVIEW:
                body = _preview_text(event.result)
        head = f"{mark} {summary}"
        if event.duration is not None and event.duration >= 1:
            head += f" ({_secs(event.duration)})"
        if body.strip():
            return head + "\n" + _indent(_cap(body, self.preview_lines))
        return head

    def end_section(self, answer: str, *, more: str = "") -> list[ToolStreamEvent]:
        """Stamp the section header with its time and call count, then the
        role's written answer, capped, with `more` naming where the rest is.
        """
        events: list[ToolStreamEvent] = []
        if self._section_id is not None:
            calls = self._section_calls
            stamp = f"{_secs(time.time() - self._section_started)}"
            if calls:
                stamp += f" · {calls} call{'s' if calls != 1 else ''}"
            events.append(
                self._event(section_line(self._section_label, stamp), self._section_id)
            )
            self._section_id = None
        if answer.strip():
            events.append(self._event(_cap(answer, ANSWER_LINES, more)))
        return events
