from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, OptionList

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

    def test_can_focus_is_true(self) -> None:
        assert SessionPickerApp.can_focus is True

    def test_search_index_defaults_to_empty(self) -> None:
        picker = SessionPickerApp(sessions=[], latest_messages={})
        assert picker._search_index == {}


class TestSessionPickerFiltering:
    @pytest.fixture
    def picker(
        self,
        sample_sessions: list[ResumeSessionInfo],
        sample_latest_messages: dict[str, list[tuple[str, str]]],
    ) -> SessionPickerApp:
        return SessionPickerApp(
            sessions=sample_sessions,
            latest_messages=sample_latest_messages,
            search_index={
                "local:session-a": "help me fix this bug\nsession a",
                "local:session-b": "refactor the authentication module\nsession b",
                "remote:session-c": "add unit tests for the api\nsession c",
            },
        )

    def test_empty_query_matches_all(self, picker: SessionPickerApp) -> None:
        assert picker._matching_sessions("") == picker._sessions

    def test_whitespace_only_query_matches_all(
        self, picker: SessionPickerApp
    ) -> None:
        assert picker._matching_sessions("   \t ") == picker._sessions

    def test_substring_match_is_case_insensitive(
        self, picker: SessionPickerApp
    ) -> None:
        matched = picker._matching_sessions("AUTHENTICATION")
        assert [s.option_id for s in matched] == ["local:session-b"]

    def test_words_match_out_of_order_and_far_apart(
        self, picker: SessionPickerApp
    ) -> None:
        matched = picker._matching_sessions("bug help")
        assert [s.option_id for s in matched] == ["local:session-a"]

    def test_more_matched_words_rank_first(self, picker: SessionPickerApp) -> None:
        matched = picker._matching_sessions("session api tests")
        assert [s.option_id for s in matched] == [
            "remote:session-c",
            "local:session-a",
            "local:session-b",
        ]

    def test_equal_scores_keep_incoming_order(
        self, picker: SessionPickerApp
    ) -> None:
        matched = picker._matching_sessions("session")
        assert [s.option_id for s in matched] == [
            s.option_id for s in picker._sessions
        ]

    def test_partial_word_matches(self, picker: SessionPickerApp) -> None:
        matched = picker._matching_sessions("sess")
        assert [s.option_id for s in matched] == [
            s.option_id for s in picker._sessions
        ]

    def test_duplicate_query_words_do_not_inflate_score(
        self, picker: SessionPickerApp
    ) -> None:
        # "api" only hits session-c and "bug" only hits session-a, so both score
        # 1 and stay in incoming order; counting "api" twice would flip them.
        matched = picker._matching_sessions("api api bug")
        assert [s.option_id for s in matched] == [
            "local:session-a",
            "remote:session-c",
        ]

    def test_session_without_any_query_word_is_excluded(
        self, picker: SessionPickerApp
    ) -> None:
        matched = picker._matching_sessions("authentication api")
        assert [s.option_id for s in matched] == [
            "local:session-b",
            "remote:session-c",
        ]

    def test_exclusion_term_drops_matching_session(
        self, picker: SessionPickerApp
    ) -> None:
        matched = picker._matching_sessions("-authentication")
        assert [s.option_id for s in matched] == [
            "local:session-a",
            "remote:session-c",
        ]

    def test_exclusion_beats_inclusion_score(self, picker: SessionPickerApp) -> None:
        # session-b matches "module" but is excluded; the rest rank as usual.
        matched = picker._matching_sessions("session tests -module")
        assert [s.option_id for s in matched] == [
            "remote:session-c",
            "local:session-a",
        ]

    def test_exclusion_only_query_keeps_incoming_order(
        self, picker: SessionPickerApp
    ) -> None:
        matched = picker._matching_sessions("-bug -api")
        assert [s.option_id for s in matched] == ["local:session-b"]

    def test_bare_dash_matches_all(self, picker: SessionPickerApp) -> None:
        assert picker._matching_sessions("-") == picker._sessions
        assert picker._matching_sessions("  -  ") == picker._sessions

    def test_word_both_included_and_excluded_drops_it(
        self, picker: SessionPickerApp
    ) -> None:
        assert picker._matching_sessions("bug -bug") == []

    def test_mid_word_hyphen_is_not_an_exclusion(
        self, sample_sessions: list[ResumeSessionInfo]
    ) -> None:
        picker = SessionPickerApp(
            sessions=sample_sessions,
            latest_messages={},
            search_index={
                "local:session-a": "context-refresh notes",
                "local:session-b": "unrelated notes",
                "remote:session-c": "more notes",
            },
        )
        matched = picker._matching_sessions("context-refresh")
        assert [s.option_id for s in matched] == ["local:session-a"]

    def test_no_match_returns_empty(self, picker: SessionPickerApp) -> None:
        assert picker._matching_sessions("zzzqqq wwwvvv") == []

    def test_session_missing_from_index_never_matches(
        self, sample_sessions: list[ResumeSessionInfo]
    ) -> None:
        picker = SessionPickerApp(
            sessions=sample_sessions, latest_messages={}, search_index={}
        )
        assert picker._matching_sessions("bug") == []

    def test_options_show_placeholder_when_no_matches(
        self, picker: SessionPickerApp
    ) -> None:
        picker._filtered = []
        options = picker._build_options()
        assert len(options) == 1
        assert options[0].disabled is True
        assert options[0].id is None

    def test_options_carry_option_ids(self, picker: SessionPickerApp) -> None:
        options = picker._build_options()
        assert [option.id for option in options] == [
            s.option_id for s in picker._sessions
        ]


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

    def test_has_arrow_bindings(self) -> None:
        keys = self._get_binding_keys()
        assert "up" in keys
        assert "down" in keys


class PickerHost(App):
    """Minimal host app so the picker can be driven by a Pilot."""

    def __init__(self, picker: SessionPickerApp) -> None:
        super().__init__()
        self._picker = picker
        self.selected_session_id: str | None = None

    def compose(self) -> ComposeResult:
        yield self._picker

    def on_session_picker_app_session_selected(
        self, event: SessionPickerApp.SessionSelected
    ) -> None:
        self.selected_session_id = event.session_id


@pytest.fixture
def picker_host(
    sample_sessions: list[ResumeSessionInfo],
    sample_latest_messages: dict[str, list[tuple[str, str]]],
) -> PickerHost:
    picker = SessionPickerApp(
        sessions=sample_sessions,
        latest_messages=sample_latest_messages,
        search_index={
            "local:session-a": "help me fix this bug\nsession a",
            "local:session-b": "refactor the authentication module\nsession b",
            "remote:session-c": "add unit tests for the api\nsession c",
        },
    )
    return PickerHost(picker)


class TestSessionPickerInteraction:
    @pytest.mark.asyncio
    async def test_search_input_has_focus_on_mount(
        self, picker_host: PickerHost
    ) -> None:
        async with picker_host.run_test():
            assert picker_host.query_one(Input).has_focus

    @pytest.mark.asyncio
    async def test_typing_filters_and_resets_highlight(
        self, picker_host: PickerHost
    ) -> None:
        async with picker_host.run_test() as pilot:
            option_list = picker_host.query_one(OptionList)

            await pilot.press("down")
            await pilot.pause()
            assert option_list.highlighted == 1

            await pilot.press("a", "u", "t", "h")
            await pilot.pause()

            assert option_list.option_count == 1
            assert option_list.get_option_at_index(0).id == "local:session-b"
            assert option_list.highlighted == 0

    @pytest.mark.asyncio
    async def test_clearing_query_restores_all_sessions(
        self, picker_host: PickerHost
    ) -> None:
        async with picker_host.run_test() as pilot:
            option_list = picker_host.query_one(OptionList)

            await pilot.press("a", "u", "t", "h")
            await pilot.pause()
            assert option_list.option_count == 1

            for _ in range(4):
                await pilot.press("backspace")
            await pilot.pause()

            assert option_list.option_count == 3

    @pytest.mark.asyncio
    async def test_space_reaches_the_search_input(
        self, picker_host: PickerHost
    ) -> None:
        async with picker_host.run_test() as pilot:
            await pilot.press("f", "i", "x", "space", "t", "h", "i", "s")
            await pilot.pause()

            assert picker_host.query_one(Input).value == "fix this"
            assert picker_host.query_one(OptionList).option_count == 1

    @pytest.mark.asyncio
    async def test_arrows_move_highlight_without_losing_input_focus(
        self, picker_host: PickerHost
    ) -> None:
        async with picker_host.run_test() as pilot:
            option_list = picker_host.query_one(OptionList)

            await pilot.press("down")
            await pilot.pause()
            assert option_list.highlighted == 1
            assert picker_host.query_one(Input).has_focus

            await pilot.press("up")
            await pilot.pause()
            assert option_list.highlighted == 0

    @pytest.mark.asyncio
    async def test_enter_on_empty_result_does_not_select(
        self, picker_host: PickerHost
    ) -> None:
        async with picker_host.run_test() as pilot:
            await pilot.press("z", "z", "z", "z")
            await pilot.pause()

            option_list = picker_host.query_one(OptionList)
            assert option_list.option_count == 1
            assert option_list.get_option_at_index(0).disabled is True
            assert option_list.highlighted is None

            await pilot.press("enter")
            await pilot.pause()

            assert picker_host.selected_session_id is None

    @pytest.mark.asyncio
    async def test_enter_selects_highlighted_session(
        self, picker_host: PickerHost
    ) -> None:
        async with picker_host.run_test() as pilot:
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause()

            assert picker_host.selected_session_id == "session-b"
