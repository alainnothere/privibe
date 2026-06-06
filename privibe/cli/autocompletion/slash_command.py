from __future__ import annotations

from textual import events

from privibe.cli.autocompletion.base import (
    CompletionResult,
    CompletionView,
    compute_visible_window,
)
from privibe.core.autocompletion.completers import CommandCompleter

# Number of suggestions visible in the popup at once. The full match list may be
# longer; the popup shows a window of this size that scrolls to follow the
# selected item (see _visible_window), so every match stays reachable.
MAX_VISIBLE_SUGGESTIONS = 10


class SlashCommandController:
    def __init__(self, completer: CommandCompleter, view: CompletionView) -> None:
        self._completer = completer
        self._view = view
        self._suggestions: list[tuple[str, str]] = []
        self._selected_index = 0

    def can_handle(self, text: str, cursor_index: int) -> bool:
        return text.startswith("/")

    def reset(self) -> None:
        if self._suggestions:
            self._suggestions.clear()
            self._selected_index = 0
            self._view.clear_completion_suggestions()

    def on_text_changed(self, text: str, cursor_index: int) -> None:
        if cursor_index < 0 or cursor_index > len(text):
            self.reset()
            return

        if not self.can_handle(text, cursor_index):
            self.reset()
            return

        suggestions = self._completer.get_completion_items(text, cursor_index)
        if suggestions:
            self._suggestions = suggestions
            self._selected_index = 0
            self._render_suggestions()
        else:
            self.reset()

    def on_key(
        self, event: events.Key, text: str, cursor_index: int
    ) -> CompletionResult:
        if not self._suggestions:
            return CompletionResult.IGNORED

        match event.key:
            case "tab":
                if self._apply_selected_completion(text, cursor_index):
                    result = CompletionResult.HANDLED
                else:
                    result = CompletionResult.IGNORED
            case "enter":
                if self._apply_selected_completion(text, cursor_index):
                    result = CompletionResult.SUBMIT
                else:
                    result = CompletionResult.HANDLED
            case "down":
                self._move_selection(1)
                result = CompletionResult.HANDLED
            case "up":
                self._move_selection(-1)
                result = CompletionResult.HANDLED
            case _:
                result = CompletionResult.IGNORED

        return result

    def _move_selection(self, delta: int) -> None:
        if not self._suggestions:
            return

        count = len(self._suggestions)
        self._selected_index = (self._selected_index + delta) % count
        self._render_suggestions()

    def _render_suggestions(self) -> None:
        window, selected_in_window = compute_visible_window(
            self._suggestions, self._selected_index, MAX_VISIBLE_SUGGESTIONS
        )
        self._view.render_completion_suggestions(window, selected_in_window)

    def _apply_selected_completion(self, text: str, cursor_index: int) -> bool:
        if not self._suggestions:
            return False

        alias, _ = self._suggestions[self._selected_index]
        replacement_range = self._completer.get_replacement_range(text, cursor_index)
        if replacement_range is None:
            self.reset()
            return False

        start, end = replacement_range
        self._view.replace_completion_range(start, end, alias)
        self.reset()
        return True
