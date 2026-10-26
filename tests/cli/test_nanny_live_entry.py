"""A real turn through the TUI with a scripted nanny: the approval runs
without a prompt and the nanny's entry lands between the call and its result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from privibe.cli.textual_ui.widgets.messages import NannyMessage
from privibe.cli.textual_ui.widgets.tools import ToolCallMessage, ToolResultMessage
from privibe.core.agents.models import BuiltinAgentName
from privibe.core.tools.base import ToolPermission
from privibe.core.types import FunctionCall, ToolCall
from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_app,
    build_test_vibe_config,
)
from tests.mock.utils import mock_llm_chunk
from tests.nanny.conftest import APPROVE, READY, build_nanny
from tests.stubs.fake_backend import FakeBackend


@pytest.mark.asyncio
async def test_nanny_entry_mounts_between_call_and_result(tmp_path: Path) -> None:
    tool_call = ToolCall(
        id="call_1",
        index=0,
        function=FunctionCall(name="todo", arguments='{"action": "read"}'),
    )
    backend = FakeBackend([
        [mock_llm_chunk(content="Checking.", tool_calls=[tool_call])],
        [mock_llm_chunk(content="Done.")],
    ])
    config = build_test_vibe_config(
        enabled_tools=["todo"],
        tools={"todo": {"permission": ToolPermission.ASK.value}},
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=False,
    )
    loop = build_test_agent_loop(
        config=config, agent_name=BuiltinAgentName.DEFAULT, backend=backend
    )
    app = build_test_vibe_app(config=config, agent_loop=loop)
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        app._attach_nanny(build_nanny(tmp_path, [READY, APPROVE]))
        await app._handle_agent_loop_turn("what is on my list?")
        await pilot.pause(0.5)

        messages_area = app.query_one("#messages")
        kinds = [
            type(w)
            for w in messages_area.children
            if isinstance(w, (ToolCallMessage, NannyMessage, ToolResultMessage))
        ]
        assert kinds == [ToolCallMessage, NannyMessage, ToolResultMessage]
        entry = app.query_one(NannyMessage)
        assert entry.has_class("nanny-approved")
        assert app._nanny.decisions[-1].approved_by_nanny
        # No prompt was shown: the chat input is still the bottom app.
        assert app._pending_approval is None
