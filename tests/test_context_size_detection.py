"""Tests for context-size auto-detection: bounded retry, flag-off on give-up,
re-detect cadence, and the served-model-change trigger.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from privibe.core import agent_loop as agent_loop_module
from privibe.core.agent_loop import (
    _CONTEXT_SIZE_MAX_ATTEMPTS,
    reset_context_size_detection_state,
)
from privibe.core.config import context_size_mode_label, cycle_context_size_mode
from tests.conftest import build_test_agent_loop, build_test_vibe_config


@pytest.fixture(autouse=True)
def _clear_latches():
    reset_context_size_detection_state()
    yield
    reset_context_size_detection_state()


def test_single_control_cycle_off_auto_then_cadence():
    # off -> auto (enabled, no polling)
    assert cycle_context_size_mode(False, 0) == (True, 0)
    # auto -> every 1 -> 2 -> 5 -> 10
    assert cycle_context_size_mode(True, 0) == (True, 1)
    assert cycle_context_size_mode(True, 1) == (True, 2)
    assert cycle_context_size_mode(True, 2) == (True, 5)
    assert cycle_context_size_mode(True, 5) == (True, 10)
    # past every-10 -> back to off
    assert cycle_context_size_mode(True, 10) == (False, 0)
    # any disabled state advances to auto regardless of stale cadence
    assert cycle_context_size_mode(False, 5) == (True, 0)


def test_context_size_mode_label():
    assert context_size_mode_label(False, 0) == "off"
    assert "auto" in context_size_mode_label(True, 0)
    assert context_size_mode_label(True, 5) == "re-detect every 5 turn(s)"


def test_reset_state_per_alias_is_scoped():
    agent_loop_module._context_size_resolved_for = "alpha"
    agent_loop_module._context_size_attempts.update({"alpha": 2, "beta": 1})

    reset_context_size_detection_state("alpha")

    assert agent_loop_module._context_size_resolved_for is None
    assert "alpha" not in agent_loop_module._context_size_attempts
    assert agent_loop_module._context_size_attempts.get("beta") == 1


def test_record_failure_retries_then_disables_flag_when_no_cadence():
    loop = build_test_agent_loop()
    model = loop.config.get_active_model()
    assert loop.config.auto_detect_context_size is True

    msg1 = loop._record_context_failure(model, "boom.")
    assert f"1/{_CONTEXT_SIZE_MAX_ATTEMPTS}" in msg1
    assert loop.config.auto_detect_context_size is True

    msg2 = loop._record_context_failure(model, "boom.")
    assert f"2/{_CONTEXT_SIZE_MAX_ATTEMPTS}" in msg2
    assert loop.config.auto_detect_context_size is True

    msg3 = loop._record_context_failure(model, "boom.")
    # On the final attempt with no cadence the flag is turned off for the run.
    assert "disabled" in msg3.lower()
    assert loop.config.auto_detect_context_size is False
    assert loop._base_config.auto_detect_context_size is False


def test_record_failure_keeps_flag_on_with_cadence():
    config = build_test_vibe_config(context_size_redetect_every=2)
    loop = build_test_agent_loop(config=config)
    model = loop.config.get_active_model()

    for _ in range(_CONTEXT_SIZE_MAX_ATTEMPTS):
        msg = loop._record_context_failure(model, "boom.")

    # With a cadence configured we never auto-disable; the cadence retries later.
    assert loop.config.auto_detect_context_size is True
    assert "every 2 turns" in msg


@pytest.mark.asyncio
async def test_note_served_model_triggers_redetect_on_change():
    loop = build_test_agent_loop()
    alias = loop.config.get_active_model().alias
    loop.resolve_context_size = AsyncMock(return_value=None)  # type: ignore[method-assign]

    # First observation only records the baseline — no re-detect.
    agent_loop_module._context_size_resolved_for = alias
    loop._note_served_model("model-a.gguf")
    assert agent_loop_module._context_size_resolved_for == alias
    loop.resolve_context_size.assert_not_called()

    # Same id again — still nothing.
    loop._note_served_model("model-a.gguf")
    loop.resolve_context_size.assert_not_called()

    # A different served id clears the latch and schedules a re-pull.
    loop._note_served_model("model-b.gguf")
    assert agent_loop_module._context_size_resolved_for is None
    loop.resolve_context_size.assert_called_once()


def test_note_served_model_ignores_empty():
    loop = build_test_agent_loop()
    loop.resolve_context_size = AsyncMock(return_value=None)  # type: ignore[method-assign]
    loop._note_served_model(None)
    loop._note_served_model("")
    loop.resolve_context_size.assert_not_called()
