from __future__ import annotations

from privibe.cli.commands import CommandRegistry
from privibe.cli.textual_ui.widgets.tool_widgets import get_result_widget
from privibe.core.config import cycle_message_prune_rows, cycle_preview_lines
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
