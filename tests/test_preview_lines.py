from __future__ import annotations

from privibe.cli.commands import CommandRegistry
from privibe.cli.textual_ui.widgets.tool_widgets import get_result_widget
from privibe.core.config import (
    cycle_message_prune_rows,
    cycle_preview_lines,
    sanitize_cycle_options,
    sanitize_positive_int,
)
from privibe.core.tools.builtins.grep import GrepResult
from tests.conftest import build_test_vibe_config


def test_cycle_rotates_3_5_10():
    assert cycle_preview_lines(3) == 5
    assert cycle_preview_lines(5) == 10
    assert cycle_preview_lines(10) == 3


def test_cycle_unknown_value_resets_to_first():
    assert cycle_preview_lines(7) == 3
    assert cycle_preview_lines(0) == 3


def test_config_default_is_3():
    assert build_test_vibe_config().tool_result_preview_lines == 3


def test_command_registered():
    cmd = CommandRegistry().find_command("/preview-lines")
    assert cmd is not None
    assert cmd.handler == "_cycle_preview_lines"


def test_command_can_be_excluded():
    registry = CommandRegistry(excluded_commands=["preview_lines"])
    assert registry.find_command("/preview-lines") is None


def test_get_result_widget_threads_preview_lines():
    r = GrepResult(matches="a\nb\nc", match_count=3, was_truncated=False)
    widget = get_result_widget("grep", r, True, "msg", collapsed=True, preview_lines=5)
    assert widget.preview_lines == 5


def test_get_result_widget_preview_lines_defaults_to_10():
    r = GrepResult(matches="", match_count=0, was_truncated=False)
    widget = get_result_widget("grep", r, True, "msg")
    assert widget.preview_lines == 10


def test_scrollback_cycle_rotates_through_options():
    assert cycle_message_prune_rows(50) == 100
    assert cycle_message_prune_rows(100) == 250
    assert cycle_message_prune_rows(250) == 500
    assert cycle_message_prune_rows(500) == 1000
    assert cycle_message_prune_rows(1000) == 50


def test_scrollback_cycle_unknown_value_resets_to_first():
    assert cycle_message_prune_rows(7) == 50


def test_scrollback_config_default_is_250():
    assert build_test_vibe_config().message_prune_keep_rows == 250


def test_scrollback_command_registered():
    cmd = CommandRegistry().find_command("/scrollback")
    assert cmd is not None
    assert cmd.handler == "_cycle_scrollback"


def test_scrollback_command_can_be_excluded():
    registry = CommandRegistry(excluded_commands=["scrollback"])
    assert registry.find_command("/scrollback") is None


# --- configurable cycle options --------------------------------------------
# The cycle lists are now config-driven; a malformed config value falls back to
# the built-in default instead of breaking startup.


def test_cycle_uses_custom_options():
    assert cycle_preview_lines(2, [2, 8, 20]) == 8
    assert cycle_preview_lines(20, [2, 8, 20]) == 2
    # current value outside the custom options resets to the first
    assert cycle_preview_lines(99, [2, 8, 20]) == 2


def test_cycle_message_prune_uses_custom_options():
    assert cycle_message_prune_rows(10, [10, 20, 30]) == 20
    assert cycle_message_prune_rows(30, [10, 20, 30]) == 10


def test_sanitize_cycle_options_keeps_valid_positive_ints_in_order():
    assert sanitize_cycle_options([5, 1, 10], (3, 5, 10)) == [5, 1, 10]


def test_sanitize_cycle_options_drops_bad_entries_and_dupes():
    # bools, non-ints, non-positive and duplicate values are dropped
    assert sanitize_cycle_options([5, True, "x", -2, 0, 5, 10], (3,)) == [5, 10]


def test_sanitize_cycle_options_falls_back_when_unusable():
    assert sanitize_cycle_options([], (3, 5, 10)) == [3, 5, 10]
    assert sanitize_cycle_options("nonsense", (3, 5, 10)) == [3, 5, 10]
    assert sanitize_cycle_options([-1, 0, "a"], (3, 5, 10)) == [3, 5, 10]
    assert sanitize_cycle_options(None, (50, 100)) == [50, 100]


def test_sanitize_positive_int_falls_back():
    assert sanitize_positive_int(7, 3) == 7
    assert sanitize_positive_int(0, 3) == 3
    assert sanitize_positive_int(-5, 3) == 3
    assert sanitize_positive_int(True, 3) == 3
    assert sanitize_positive_int("4", 3) == 3


def test_config_accepts_custom_cycle_options():
    config = build_test_vibe_config(
        tool_result_preview_options=[2, 8, 20],
        message_prune_keep_options=[10, 20, 30],
    )
    assert config.tool_result_preview_options == [2, 8, 20]
    assert config.message_prune_keep_options == [10, 20, 30]


def test_config_falls_back_to_defaults_for_bad_values():
    config = build_test_vibe_config(
        tool_result_preview_options=[],          # empty -> default
        message_prune_keep_options=[-1, "x", 0],  # no usable entries -> default
        tool_result_preview_lines=-5,            # invalid current -> default
        message_prune_keep_rows=0,               # invalid current -> default
    )
    assert config.tool_result_preview_options == [3, 5, 10]
    assert config.message_prune_keep_options == [50, 100, 250, 500, 1000]
    assert config.tool_result_preview_lines == 3
    assert config.message_prune_keep_rows == 250
