from __future__ import annotations

from enum import StrEnum
from typing import Protocol


class CompletionResult(StrEnum):
    IGNORED = "ignored"
    HANDLED = "handled"
    SUBMIT = "submit"


class CompletionView(Protocol):
    def render_completion_suggestions(
        self, suggestions: list[tuple[str, str]], selected_index: int
    ) -> None: ...

    def clear_completion_suggestions(self) -> None: ...

    def replace_completion_range(
        self, start: int, end: int, replacement: str
    ) -> None: ...


def compute_visible_window(
    suggestions: list[tuple[str, str]],
    selected_index: int,
    max_visible: int,
) -> tuple[list[tuple[str, str]], int]:
    """Return up to ``max_visible`` items plus the selected item's index within
    that slice.

    The window scrolls to keep the selection visible (roughly centered) until it
    reaches the list's start or end, so every item stays reachable by pressing
    up/down — instead of truncating the list to the first N (which hid the rest).
    """
    count = len(suggestions)
    if count <= max_visible:
        return suggestions, selected_index

    half = max_visible // 2
    start = max(0, min(selected_index - half, count - max_visible))
    return suggestions[start : start + max_visible], selected_index - start
