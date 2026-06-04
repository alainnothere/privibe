from __future__ import annotations

import time

import pytest

from tests.conftest import build_test_agent_loop
from privibe.cli.textual_ui.app import VibeApp
from privibe.cli.textual_ui.widgets.load_more import HistoryLoadMoreMessage
from privibe.core.config import SessionLoggingConfig, VibeConfig
from privibe.core.types import LLMMessage, Role


async def _wait_until(pause, predicate, timeout: float = 5.0) -> None:
    start = time.monotonic()
    while (time.monotonic() - start) < timeout:
        if predicate():
            return
        await pause(0.02)
    raise AssertionError("Condition was not met within the timeout")


@pytest.mark.asyncio
async def test_rewind_reaches_user_message_buried_in_backfill() -> None:
    """A user message windowed into the backfill behind a batch that contains
    no user message must still be reachable by rewind.

    Layout (50 messages, tail = 20, load-more batch = 10):
      index 0      : user  "deep-target"   <- in backfill, behind non-user batches
      indices 1-29 : assistant             <- 29 non-user messages
      index 30     : user  "recent"        <- mounted in the tail
      indices 31-49: assistant             <- fills the rest of the tail
    Walking up from "recent" must load through the all-assistant batches
    (indices 20-29, then 10-19) before "deep-target" (index 0) is mounted.
    """
    config = VibeConfig(session_logging=SessionLoggingConfig(enabled=False))
    agent_loop = build_test_agent_loop(config=config)
    history = [LLMMessage(role=Role.user, content="deep-target")]
    history += [LLMMessage(role=Role.assistant, content=f"a{i}") for i in range(29)]
    history += [LLMMessage(role=Role.user, content="recent")]
    history += [LLMMessage(role=Role.assistant, content=f"b{i}") for i in range(19)]
    agent_loop.messages._data.extend(history)

    app = VibeApp(agent_loop=agent_loop)
    async with app.run_test() as pilot:
        # Resume windows the history: only the tail mounts, the rest is backfill
        # behind a "load more" control.
        await _wait_until(
            pilot.pause, lambda: len(app.query(HistoryLoadMoreMessage)) == 1
        )

        # Enter rewind — highlights the only mounted user message.
        await pilot.press("alt+up")
        await app.workers.wait_for_complete()
        await pilot.pause(0.1)
        assert app._rewind_highlighted_widget is not None
        assert app._rewind_highlighted_widget.get_content() == "recent"

        # Up again: must load through the all-assistant batches to reach the
        # backfilled user message rather than getting stuck.
        await pilot.press("alt+up")
        await app.workers.wait_for_complete()
        await pilot.pause(0.1)
        assert app._rewind_highlighted_widget is not None
        assert app._rewind_highlighted_widget.get_content() == "deep-target"
