from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, cast

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical
from textual.message import Message
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from privibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from privibe.core.session.resume_sessions import (
    ResumeSessionInfo,
    ResumeSessionSource,
    short_session_id,
)

_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600
_SECONDS_PER_DAY = 86400
_SECONDS_PER_WEEK = 604800


def _format_relative_time(iso_time: str | None) -> str:
    if not iso_time:
        return "unknown"
    try:
        dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
        now = datetime.now(UTC)
        delta = now - dt
        seconds = int(delta.total_seconds())

        if seconds < _SECONDS_PER_MINUTE:
            return "just now"
        for threshold, divisor, unit in [
            (_SECONDS_PER_HOUR, _SECONDS_PER_MINUTE, "m"),
            (_SECONDS_PER_DAY, _SECONDS_PER_HOUR, "h"),
            (_SECONDS_PER_WEEK, _SECONDS_PER_DAY, "d"),
            (float("inf"), _SECONDS_PER_WEEK, "w"),
        ]:
            if seconds < threshold:
                return f"{seconds // divisor}{unit} ago"
    except (ValueError, OSError):
        pass
    return "unknown"


def _format_absolute_time(iso_time: str | None) -> str:
    if not iso_time:
        return ""
    try:
        dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%b %d %H:%M")
    except (ValueError, OSError):
        return ""


def _shorten_path(path_str: str) -> str:
    home = str(Path.home())
    if path_str.startswith(home):
        return "~" + path_str[len(home):]
    return path_str or "(unknown dir)"


def _build_option_text(
    session: ResumeSessionInfo, messages: list[tuple[str, str]]
) -> Text:
    text = Text(no_wrap=True)
    time_str = _format_relative_time(session.end_time)
    abs_time = _format_absolute_time(session.end_time)
    session_id = short_session_id(session.session_id, source=session.source)
    cwd_str = _shorten_path(session.cwd)

    text.append(f"{time_str}", style="dim")
    if abs_time:
        text.append(f" ({abs_time})", style="dim")
    text.append("  ")
    text.append(f"{session_id}  ", style="dim")
    text.append(cwd_str, style="bold")

    indent = " " * 14
    if session.session_path:
        text.append(f"\n{indent}")
        text.append(_shorten_path(session.session_path), style="dim")

    role_labels = {"user": "You", "assistant": "AI "}
    for role, msg_text in messages:
        label = role_labels.get(role, role[:3])
        truncated = msg_text if len(msg_text) <= 80 else msg_text[:80] + "…"
        text.append(f"\n{indent}")
        text.append(f"{label}: ", style="dim")
        text.append(truncated)

    return text


class SessionPickerApp(Container):
    """Session picker for /resume command."""

    can_focus_children = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False)
    ]

    class SessionSelected(Message):
        option_id: str
        source: ResumeSessionSource
        session_id: str

        def __init__(
            self, option_id: str, source: ResumeSessionSource, session_id: str
        ) -> None:
            self.option_id = option_id
            self.source = source
            self.session_id = session_id
            super().__init__()

    class Cancelled(Message):
        pass

    def __init__(
        self,
        sessions: list[ResumeSessionInfo],
        latest_messages: dict[str, list[tuple[str, str]]],
        **kwargs: Any,
    ) -> None:
        super().__init__(id="sessionpicker-app", **kwargs)
        self._sessions = sessions
        self._latest_messages = latest_messages

    def compose(self) -> ComposeResult:
        options = [
            Option(
                _build_option_text(
                    session,
                    self._latest_messages.get(session.option_id, []),
                ),
                id=session.option_id,
            )
            for session in self._sessions
        ]
        with Vertical(id="sessionpicker-content"):
            yield OptionList(*options, id="sessionpicker-options")
            yield NoMarkupStatic(
                "↑↓ Navigate  Enter Select  Esc Cancel", classes="sessionpicker-help"
            )

    def on_mount(self) -> None:
        self.query_one(OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id:
            option_id = event.option.id
            source, _, session_id = option_id.partition(":")
            self.post_message(
                self.SessionSelected(
                    option_id=option_id,
                    source=cast(ResumeSessionSource, source),
                    session_id=session_id,
                )
            )

    def action_cancel(self) -> None:
        self.post_message(self.Cancelled())
