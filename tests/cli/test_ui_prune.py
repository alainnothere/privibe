from __future__ import annotations

import time

import pytest

from privibe.cli.textual_ui.app import VibeApp
from privibe.cli.textual_ui.widgets.load_more import HistoryLoadMoreMessage
from privibe.core.config import SessionLoggingConfig, VibeConfig
from privibe.core.types import LLMMessage, Role
from tests.conftest import build_test_agent_loop


async def _wait_until(pause, predicate, timeout: float = 5.0) -> None:
    start = time.monotonic()
    while (time.monotonic() - start) < timeout:
        if predicate():
            return
        await pause(0.02)
    raise AssertionError("Condition was not met within the timeout")


def _message_widgets(messages_area) -> list:
    return [
        c
        for c in messages_area.children
        if not isinstance(c, HistoryLoadMoreMessage)
    ]


@pytest.mark.asyncio
async def test_prune_surfaces_load_more_button_for_pruned_messages() -> None:
    """Pruning the live transcript must not silently drop messages.

    Regression: /scrollback (and natural growth) pruned the oldest message
    widgets with no affordance to get them back. The messages are still in
    history, so a prune should now surface a "Load more" button that pages
    them back, instead of leaving them unreachable.
    """
    config = VibeConfig(
        session_logging=SessionLoggingConfig(enabled=False),
        # Tiny threshold so a prune actually triggers within the test.
        message_prune_keep_rows=2,
    )
    agent_loop = build_test_agent_loop(config=config)
    # 12 < HISTORY_RESUME_TAIL_MESSAGES (20): all mount, no load-more at startup.
    history = [
        LLMMessage(role=Role.assistant, content=f"message {i}") for i in range(12)
    ]
    agent_loop.messages._data.extend(history)
    history_len = len(agent_loop.messages)

    app = VibeApp(agent_loop=agent_loop)
    async with app.run_test() as pilot:
        messages_area = app.query_one("#messages")
        await _wait_until(pilot.pause, lambda: len(_message_widgets(messages_area)) >= 12)
        # Everything is mounted, so there's no load-more button yet.
        assert len(app.query(HistoryLoadMoreMessage)) == 0
        mounted_before = len(_message_widgets(messages_area))

        await app._try_prune()
        await _wait_until(
            pilot.pause, lambda: len(app.query(HistoryLoadMoreMessage)) == 1
        )

        # The prune trimmed mounted widgets...
        assert len(_message_widgets(messages_area)) < mounted_before
        # ...surfaced a "Load more" button...
        assert len(app.query(HistoryLoadMoreMessage)) == 1
        # ...and left the underlying conversation history untouched.
        assert len(agent_loop.messages) == history_len
