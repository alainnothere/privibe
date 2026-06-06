from __future__ import annotations

from typing import NamedTuple

from textual import events

from privibe.cli.autocompletion.base import CompletionResult, CompletionView
from privibe.cli.autocompletion.slash_command import SlashCommandController
from privibe.core.autocompletion.completers import CommandCompleter


class Suggestion(NamedTuple):
    alias: str
    description: str


class SuggestionEvent(NamedTuple):
    suggestions: list[Suggestion]
    selected_index: int


class Replacement(NamedTuple):
    start: int
    end: int
    replacement: str


class StubView(CompletionView):
    def __init__(self) -> None:
        self.suggestion_events: list[SuggestionEvent] = []
        self.reset_count = 0
        self.replacements: list[Replacement] = []

    def render_completion_suggestions(
        self, suggestions: list[tuple[str, str]], selected_index: int
    ) -> None:
        typed = [Suggestion(alias, description) for alias, description in suggestions]
        self.suggestion_events.append(SuggestionEvent(typed, selected_index))

    def clear_completion_suggestions(self) -> None:
        self.reset_count += 1

    def replace_completion_range(self, start: int, end: int, replacement: str) -> None:
        self.replacements.append(Replacement(start, end, replacement))


def key_event(key: str) -> events.Key:
    return events.Key(key, character=None)


def make_controller(
    *, prefix: str | None = None
) -> tuple[SlashCommandController, StubView]:
    commands = [
        ("/config", "Show current configuration"),
        ("/compact", "Compact history"),
        ("/help", "Display help"),
        ("/config", "Override description"),
        ("/summarize", "Summarize history"),
        ("/logpath", "Show log path"),
        ("/exit", "Exit application"),
        ("/vim", "Toggle vim keybindings"),
    ]
    completer = CommandCompleter(lambda: commands)
    view = StubView()
    controller = SlashCommandController(completer, view)

    if prefix is not None:
        controller.on_text_changed(prefix, cursor_index=len(prefix))
        view.suggestion_events.clear()

    return controller, view


def test_on_text_change_emits_matching_suggestions_in_insertion_order_and_ignores_duplicates() -> (
    None
):
    controller, view = make_controller(prefix="/c")

    controller.on_text_changed("/c", cursor_index=2)

    suggestions, selected = view.suggestion_events[-1]
    assert suggestions == [
        Suggestion("/config", "Override description"),
        Suggestion("/compact", "Compact history"),
    ]
    assert selected == 0


def test_on_text_change_filters_suggestions_case_insensitively() -> None:
    controller, view = make_controller(prefix="/c")

    controller.on_text_changed("/CO", cursor_index=3)

    suggestions, _ = view.suggestion_events[-1]
    assert [suggestion.alias for suggestion in suggestions] == ["/config", "/compact"]


def test_on_text_change_clears_suggestions_when_no_matches() -> None:
    controller, view = make_controller(prefix="/c")

    controller.on_text_changed("/c", cursor_index=2)
    controller.on_text_changed("config", cursor_index=6)

    assert view.reset_count >= 1


def test_on_text_change_limits_the_number_of_results_and_preserves_insertion_order() -> (
    None
):
    controller, view = make_controller(prefix="/")

    controller.on_text_changed("/", cursor_index=1)

    suggestions, selected_index = view.suggestion_events[-1]
    assert len(suggestions) == 7
    assert [suggestion.alias for suggestion in suggestions] == [
        "/config",
        "/compact",
        "/help",
        "/summarize",
        "/logpath",
        "/exit",
        "/vim",
    ]


def test_on_key_tab_applies_selected_completion() -> None:
    controller, view = make_controller(prefix="/c")

    result = controller.on_key(key_event("tab"), text="/c", cursor_index=2)

    assert result is CompletionResult.HANDLED
    assert view.replacements == [Replacement(0, 2, "/config")]
    assert view.reset_count == 1


def test_on_key_down_and_up_cycle_selection() -> None:
    controller, view = make_controller(prefix="/c")

    controller.on_key(key_event("down"), text="/c", cursor_index=2)
    suggestions, selected_index = view.suggestion_events[-1]
    assert selected_index == 1

    controller.on_key(key_event("down"), text="/c", cursor_index=2)
    suggestions, selected_index = view.suggestion_events[-1]
    assert selected_index == 0

    controller.on_key(key_event("up"), text="/c", cursor_index=2)
    suggestions, selected_index = view.suggestion_events[-1]
    assert selected_index == 1
    assert [suggestion.alias for suggestion in suggestions] == ["/config", "/compact"]


def test_on_key_enter_submits_selected_completion() -> None:
    controller, view = make_controller(prefix="/c")

    controller.on_key(key_event("down"), text="/c", cursor_index=2)

    result = controller.on_key(key_event("enter"), text="/c", cursor_index=2)

    assert result is CompletionResult.SUBMIT
    assert view.replacements == [Replacement(0, 2, "/compact")]
    assert view.reset_count == 1


def test_callable_entries_updates_completions_dynamically() -> None:
    """Test that CommandCompleter with a callable updates entries when the callable returns different values.

    This simulates config reload where available skills change.
    """
    available_skills: list[tuple[str, str]] = []

    def get_entries() -> list[tuple[str, str]]:
        base_commands = [("/help", "Display help"), ("/config", "Show configuration")]
        return base_commands + available_skills

    completer = CommandCompleter(get_entries)
    view = StubView()
    controller = SlashCommandController(completer, view)

    # Initially, only base commands are available
    controller.on_text_changed("/", cursor_index=1)
    suggestions, _ = view.suggestion_events[-1]
    assert [s.alias for s in suggestions] == ["/help", "/config"]

    # Simulate config reload: add a skill
    available_skills.append(("/summarize", "Summarize the conversation"))

    # Now completions should include the new skill
    controller.on_text_changed("/", cursor_index=1)
    suggestions, _ = view.suggestion_events[-1]
    assert [s.alias for s in suggestions] == ["/help", "/config", "/summarize"]

    # And searching for "/s" should find the new skill
    controller.on_text_changed("/s", cursor_index=2)
    suggestions, _ = view.suggestion_events[-1]
    assert [s.alias for s in suggestions] == ["/summarize"]
    assert suggestions[0].description == "Summarize the conversation"


def test_callable_entries_reflects_enabled_disabled_skills() -> None:
    """Test that skill enable/disable changes are reflected in completions.

    This simulates the scenario where a user changes enabled_skills in config
    and runs /reload.
    """
    enabled_skills: set[str] = {"commit", "review"}

    all_skills = [
        ("/commit", "Create a git commit"),
        ("/review", "Review code changes"),
        ("/deploy", "Deploy to production"),
    ]

    def get_entries() -> list[tuple[str, str]]:
        return [(name, desc) for name, desc in all_skills if name[1:] in enabled_skills]

    completer = CommandCompleter(get_entries)
    view = StubView()
    controller = SlashCommandController(completer, view)

    # Initially only commit and review are enabled
    controller.on_text_changed("/", cursor_index=1)
    suggestions, _ = view.suggestion_events[-1]
    assert [s.alias for s in suggestions] == ["/commit", "/review"]

    # Simulate config reload: enable deploy, disable commit
    enabled_skills.discard("commit")
    enabled_skills.add("deploy")

    # Now completions should reflect the change
    controller.on_text_changed("/", cursor_index=1)
    suggestions, _ = view.suggestion_events[-1]
    assert [s.alias for s in suggestions] == ["/review", "/deploy"]


# --- Scrolling window -------------------------------------------------------
# When more matches exist than fit in the popup, the list must not be truncated
# to the first N (which made later commands unreachable). Instead the popup
# shows a fixed-size window that scrolls to follow the selection, so every match
# can be reached by pressing down/up.

WINDOW = 10


def make_many_controller(
    n: int,
) -> tuple[SlashCommandController, StubView, list[tuple[str, str]]]:
    commands = [(f"/cmd{i:02d}", f"description {i}") for i in range(n)]
    completer = CommandCompleter(lambda: commands)
    view = StubView()
    controller = SlashCommandController(completer, view)
    return controller, view, commands


def test_short_list_is_shown_in_full_without_windowing() -> None:
    controller, view, commands = make_many_controller(6)

    controller.on_text_changed("/", cursor_index=1)

    suggestions, selected = view.suggestion_events[-1]
    assert [s.alias for s in suggestions] == [c[0] for c in commands]
    assert selected == 0


def test_long_list_shows_only_a_window_starting_at_the_top() -> None:
    controller, view, _ = make_many_controller(15)

    controller.on_text_changed("/", cursor_index=1)

    suggestions, selected = view.suggestion_events[-1]
    assert len(suggestions) == WINDOW
    assert selected == 0
    assert suggestions[0].alias == "/cmd00"
    assert suggestions[-1].alias == "/cmd09"


def test_window_scrolls_to_the_end_so_last_item_is_reachable() -> None:
    controller, view, commands = make_many_controller(15)
    controller.on_text_changed("/", cursor_index=1)

    for _ in range(len(commands) - 1):  # move to the last item
        controller.on_key(key_event("down"), text="/", cursor_index=1)

    suggestions, selected = view.suggestion_events[-1]
    assert len(suggestions) == WINDOW
    assert [s.alias for s in suggestions] == [f"/cmd{i:02d}" for i in range(5, 15)]
    assert suggestions[selected].alias == "/cmd14"
    assert selected == WINDOW - 1


def test_every_suggestion_is_reachable_by_scrolling_down() -> None:
    # The regression: previously the list was capped at 10, so /cmd10../cmd14
    # could never be selected. Walking the whole list must surface all of them.
    controller, view, commands = make_many_controller(15)
    controller.on_text_changed("/", cursor_index=1)

    seen = []
    suggestions, selected = view.suggestion_events[-1]
    seen.append(suggestions[selected].alias)
    for _ in range(len(commands) - 1):
        controller.on_key(key_event("down"), text="/", cursor_index=1)
        suggestions, selected = view.suggestion_events[-1]
        # The selected item is always inside the rendered window.
        assert 0 <= selected < len(suggestions)
        seen.append(suggestions[selected].alias)

    assert seen == [c[0] for c in commands]
    assert all(len(ev.suggestions) <= WINDOW for ev in view.suggestion_events)


def test_wraparound_from_top_jumps_window_to_the_bottom_and_back() -> None:
    controller, view, _ = make_many_controller(15)
    controller.on_text_changed("/", cursor_index=1)

    controller.on_key(key_event("up"), text="/", cursor_index=1)  # 0 -> last
    suggestions, selected = view.suggestion_events[-1]
    assert suggestions[selected].alias == "/cmd14"

    controller.on_key(key_event("down"), text="/", cursor_index=1)  # last -> 0
    suggestions, selected = view.suggestion_events[-1]
    assert suggestions[selected].alias == "/cmd00"
    assert selected == 0
