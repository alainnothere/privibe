from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, cast

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical
from textual.message import Message
from textual.widgets import Input, OptionList
from textual.widgets.option_list import Option

from privibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from privibe.cli.textual_ui.widgets.vscode_compat import VscodeCompatInput
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

    can_focus = True
    can_focus_children = True

    # pageup/pagedown are app-level priority bindings, so they cannot be
    # rebound here; only up/down reach the picker while the search input
    # holds focus.
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
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
        search_index: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(id="sessionpicker-app", **kwargs)
        self._sessions = sessions
        self._latest_messages = latest_messages
        self._search_index = search_index or {}
        self._filtered = sessions

    def _matching_sessions(self, query: str) -> list[ResumeSessionInfo]:
        needle = query.lower()
        if not needle:
            return self._sessions
        return [
            session
            for session in self._sessions
            if needle in self._search_index.get(session.option_id, "")
        ]

    def _build_options(self) -> list[Option]:
        if not self._filtered:
            return [Option("No matching sessions", disabled=True)]
        return [
            Option(
                _build_option_text(
                    session,
                    self._latest_messages.get(session.option_id, []),
                ),
                id=session.option_id,
            )
            for session in self._filtered
        ]

    def compose(self) -> ComposeResult:
        with Vertical(id="sessionpicker-content"):
            yield VscodeCompatInput(
                placeholder="Search sessions...",
                id="sessionpicker-search",
                classes="sessionpicker-search",
            )
            yield OptionList(*self._build_options(), id="sessionpicker-options")
            yield NoMarkupStatic(
                "Type to search  ↑↓ Navigate  Enter Select  Esc Cancel",
                classes="sessionpicker-help",
            )

    def on_mount(self) -> None:
        self.focus()

    def focus(self, scroll_visible: bool = True) -> SessionPickerApp:
        """Override focus to focus the search input."""
        inputs = self.query(Input)
        if inputs:
            inputs.first().focus(scroll_visible=scroll_visible)
        else:
            super().focus(scroll_visible=scroll_visible)
        return self

    def on_input_changed(self, event: Input.Changed) -> None:
        self._filtered = self._matching_sessions(event.value)
        option_list = self.query_one(OptionList)
        option_list.set_options(self._build_options())
        option_list.highlighted = 0 if self._filtered else None

    def on_input_submitted(self, _event: Input.Submitted) -> None:
        if not self._filtered:
            return
        option_list = self.query_one(OptionList)
        highlighted = option_list.highlighted
        if highlighted is None:
            return
        self._post_selected(option_list.get_option_at_index(highlighted).id)

    def action_cursor_up(self) -> None:
        if self._filtered:
            self.query_one(OptionList).action_cursor_up()

    def action_cursor_down(self) -> None:
        if self._filtered:
            self.query_one(OptionList).action_cursor_down()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self._post_selected(event.option.id)

    def _post_selected(self, option_id: str | None) -> None:
        if not option_id:
            return
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
