from __future__ import annotations

from privibe.cli.autocompletion.base import compute_visible_window


def _items(n: int) -> list[tuple[str, str]]:
    return [(f"/c{i:02d}", "") for i in range(n)]


def test_returns_full_list_unchanged_when_within_window() -> None:
    data = _items(6)
    window, selected = compute_visible_window(data, selected_index=3, max_visible=10)
    assert window == data
    assert selected == 3


def test_window_anchored_at_top_when_selection_near_start() -> None:
    data = _items(20)
    window, selected = compute_visible_window(data, selected_index=0, max_visible=10)
    assert [a for a, _ in window] == [f"/c{i:02d}" for i in range(10)]
    assert selected == 0


def test_window_centers_selection_in_the_middle() -> None:
    data = _items(20)
    window, selected = compute_visible_window(data, selected_index=10, max_visible=10)
    assert [a for a, _ in window] == [f"/c{i:02d}" for i in range(5, 15)]
    assert window[selected][0] == "/c10"


def test_window_clamps_to_end_for_last_item() -> None:
    data = _items(20)
    window, selected = compute_visible_window(data, selected_index=19, max_visible=10)
    assert [a for a, _ in window] == [f"/c{i:02d}" for i in range(10, 20)]
    assert selected == 9
    assert window[selected][0] == "/c19"


def test_selected_item_is_always_visible_across_the_whole_list() -> None:
    data = _items(37)
    for i in range(len(data)):
        window, selected = compute_visible_window(data, selected_index=i, max_visible=10)
        assert len(window) == 10
        assert 0 <= selected < len(window)
        assert window[selected][0] == data[i][0]
