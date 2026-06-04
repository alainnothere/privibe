from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from privibe.cli.textual_ui.widgets.session_picker import (
    SessionPickerApp,
    _format_absolute_time,
    _format_relative_time,
    _shorten_path,
)
from privibe.core.session.resume_sessions import ResumeSessionInfo


@pytest.fixture
def sample_sessions() -> list[ResumeSessionInfo]:
    return [
        ResumeSessionInfo(
            session_id="session-a",
            source="local",
            cwd="/test",
            title="Session A",
            end_time=(datetime.now(UTC) - timedelta(minutes=5)).isoformat(),
        ),
        ResumeSessionInfo(
            session_id="session-b",
            source="local",
            cwd="/test",
            title="Session B",
            end_time=(datetime.now(UTC) - timedelta(hours=1)).isoformat(),
        ),
        ResumeSessionInfo(
            session_id="session-c",
            source="remote",
            cwd="/test",
            title="Session C",
            end_time=(datetime.now(UTC) - timedelta(days=1)).isoformat(),
            status="RUNNING",
        ),
    ]


@pytest.fixture
def sample_latest_messages() -> dict[str, list[tuple[str, str]]]:
    return {
        "local:session-a": [("user", "Help me fix this bug"), ("assistant", "Sure, let me look at that.")],
        "local:session-b": [("user", "Refactor the authentication module")],
        "remote:session-c": [("user", "Add unit tests for the API"), ("assistant", "I'll start with the endpoints.")],
    }


class TestFormatRelativeTime:
    def test_just_now(self) -> None:
        now = datetime.now(UTC).isoformat()
        assert _format_relative_time(now) == "just now"

    def test_minutes_ago(self) -> None:
        time_5m_ago = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
        assert _format_relative_time(time_5m_ago) == "5m ago"

    def test_hours_ago(self) -> None:
        time_2h_ago = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        assert _format_relative_time(time_2h_ago) == "2h ago"

    def test_days_ago(self) -> None:
        time_3d_ago = (datetime.now(UTC) - timedelta(days=3)).isoformat()
        assert _format_relative_time(time_3d_ago) == "3d ago"

    def test_weeks_ago(self) -> None:
        time_2w_ago = (datetime.now(UTC) - timedelta(weeks=2)).isoformat()
        assert _format_relative_time(time_2w_ago) == "2w ago"

    def test_none_returns_unknown(self) -> None:
        assert _format_relative_time(None) == "unknown"

    def test_invalid_format_returns_unknown(self) -> None:
        assert _format_relative_time("not-a-date") == "unknown"

    def test_handles_z_suffix(self) -> None:
        time_str = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        assert _format_relative_time(time_str) == "just now"

    def test_boundary_59_seconds(self) -> None:
        time_59s_ago = (datetime.now(UTC) - timedelta(seconds=59)).isoformat()
        assert _format_relative_time(time_59s_ago) == "just now"

    def test_boundary_60_seconds(self) -> None:
        time_60s_ago = (datetime.now(UTC) - timedelta(seconds=60)).isoformat()
        assert _format_relative_time(time_60s_ago) == "1m ago"


class TestFormatAbsoluteTime:
    def test_returns_formatted_date(self) -> None:
        iso = "2026-03-30T09:14:00+00:00"
        result = _format_absolute_time(iso)
        assert result != ""
        assert ":" in result  # contains HH:MM

    def test_none_returns_empty(self) -> None:
        assert _format_absolute_time(None) == ""

    def test_invalid_returns_empty(self) -> None:
        assert _format_absolute_time("not-a-date") == ""

    def test_handles_z_suffix(self) -> None:
        iso = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        result = _format_absolute_time(iso)
        assert result != ""


class TestShortenPath:
    def test_replaces_home_with_tilde(self) -> None:
        from pathlib import Path
        home = str(Path.home())
        result = _shorten_path(f"{home}/projects/myapp")
        assert result == "~/projects/myapp"

    def test_leaves_non_home_path_unchanged(self) -> None:
        assert _shorten_path("/tmp/something") == "/tmp/something"

    def test_empty_string_returns_unknown_dir(self) -> None:
        assert _shorten_path("") == "(unknown dir)"


class TestSessionPickerAppInit:
    def test_init_sets_properties(
        self,
        sample_sessions: list[ResumeSessionInfo],
        sample_latest_messages: dict[str, list[tuple[str, str]]],
    ) -> None:
        picker = SessionPickerApp(
            sessions=sample_sessions, latest_messages=sample_latest_messages
        )
        assert picker._sessions == sample_sessions
        assert picker._latest_messages == sample_latest_messages

    def test_id_is_sessionpicker_app(self) -> None:
        picker = SessionPickerApp(sessions=[], latest_messages={})
        assert picker.id == "sessionpicker-app"

    def test_can_focus_children_is_true(self) -> None:
        assert SessionPickerApp.can_focus_children is True


class TestSessionPickerMessages:
    def test_session_selected_stores_option_id(self) -> None:
        msg = SessionPickerApp.SessionSelected(
            "local:test-session-id", "local", "test-session-id"
        )
        assert msg.option_id == "local:test-session-id"
        assert msg.source == "local"
        assert msg.session_id == "test-session-id"

    def test_session_selected_with_full_uuid(self) -> None:
        session_id = "abc12345-6789-0123-4567-89abcdef0123"
        option_id = f"remote:{session_id}"
        msg = SessionPickerApp.SessionSelected(option_id, "remote", session_id)
        assert msg.option_id == option_id
        assert msg.source == "remote"
        assert msg.session_id == session_id

    def test_cancelled_can_be_instantiated(self) -> None:
        msg = SessionPickerApp.Cancelled()
        assert isinstance(msg, SessionPickerApp.Cancelled)


class TestSessionPickerAppBindings:
    def _get_binding_keys(self) -> list[str]:
        keys = []
        for binding in SessionPickerApp.BINDINGS:
            if isinstance(binding, tuple) and len(binding) >= 1:
                keys.append(binding[0])
            else:
                keys.append(binding.key)
        return keys

    def test_has_escape_binding(self) -> None:
        assert "escape" in self._get_binding_keys()
