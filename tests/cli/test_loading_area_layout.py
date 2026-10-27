"""The loading area is a horizontal row: narrator status, nanny status, the
spinner (inside #loading-area-content, width 1fr), feedback bar.

Regression: NannyStatus shipped without a width rule, so as a Static in a
horizontal layout it took the full row and pushed the spinner off the right
edge of the screen (region x = screen width + 1, width 1). The spinner was
mounted, displayed, and invisible.
"""

from __future__ import annotations

import asyncio

import pytest

from privibe.cli.textual_ui.widgets.loading import LoadingWidget
from privibe.cli.textual_ui.widgets.nanny_status import NannyStatus
from privibe.core.types import LLMChunk, LLMMessage, Role
from tests.conftest import build_test_agent_loop, build_test_vibe_app
from tests.stubs.fake_backend import FakeBackend

SCREEN_WIDTH = 120


class _SlowBackend(FakeBackend):
    """Streams reasoning slowly so the turn is still running when we look."""

    async def complete_streaming(self, **kwargs):
        self._requests_messages.append(kwargs["messages"])
        self._requests_extra_headers.append(kwargs["extra_headers"])
        self._requests_metadata.append(kwargs.get("metadata"))
        for i in range(40):
            await asyncio.sleep(0.05)
            yield LLMChunk(
                message=LLMMessage(role=Role.assistant, reasoning_content=f"t{i} ")
            )
        yield LLMChunk(message=LLMMessage(role=Role.assistant, content="done"))


@pytest.mark.asyncio
@pytest.mark.parametrize("named", [False, True])
async def test_spinner_is_on_screen_while_the_model_thinks(named: bool) -> None:
    loop = build_test_agent_loop(backend=_SlowBackend(), enable_streaming=True)
    app = build_test_vibe_app(agent_loop=loop)
    async with app.run_test(size=(SCREEN_WIDTH, 40)) as pilot:
        await pilot.pause(0.1)
        if named:
            assert await app._handle_command("/name parser")
        await app._handle_user_message("hello")
        await pilot.pause(0.6)

        nanny = app.query_one(NannyStatus)
        # Idle nanny: an empty label, not a full-width one.
        assert nanny.region.width == 0

        [loading] = list(app.query(LoadingWidget))
        assert loading.display
        assert loading.region.x < SCREEN_WIDTH
        assert loading.region.width > 1
        assert loading.hint_widget is not None
        assert loading.hint_widget.region.right <= SCREEN_WIDTH
        assert loading.hint_widget.region.width > 0
