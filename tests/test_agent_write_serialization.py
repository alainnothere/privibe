from __future__ import annotations

import json

import pytest

from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from privibe.core.agents.models import BuiltinAgentName
from privibe.core.config import VibeConfig
from privibe.core.types import FunctionCall, ToolCall


def _write_call(call_id: str, index: int, path: str, content: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=index,
        function=FunctionCall(
            name="write_file",
            arguments=json.dumps({"path": path, "content": content, "overwrite": True}),
        ),
    )


def _restore_call(call_id: str, index: int, path: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=index,
        function=FunctionCall(
            name="restore_file", arguments=json.dumps({"path": path})
        ),
    )


def _config(*tools: str) -> VibeConfig:
    return build_test_vibe_config(
        enabled_tools=list(tools),
        tools={t: {"permission": "always"} for t in tools},
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=False,
    )


@pytest.mark.asyncio
async def test_same_file_writes_apply_in_model_order(tmp_path, monkeypatch):
    # Two writes to the same file in ONE turn must apply serially in the order
    # the model emitted them — never race to a nondeterministic result.
    monkeypatch.chdir(tmp_path)
    backend = FakeBackend([
        [
            mock_llm_chunk(
                content="writing",
                tool_calls=[
                    _write_call("c1", 0, "out.txt", "FIRST"),
                    _write_call("c2", 1, "out.txt", "SECOND"),
                ],
            )
        ],
        [mock_llm_chunk(content="done")],
    ])
    loop = build_test_agent_loop(
        config=_config("write_file"),
        agent_name=BuiltinAgentName.AUTO_APPROVE,
        backend=backend,
    )

    [ev async for ev in loop.act("write the file twice")]

    assert (tmp_path / "out.txt").read_text() == "SECOND"
    # The pre-edit states were captured into the per-agent undo stack.
    assert loop.undo_stack.has_versions("out.txt")


@pytest.mark.asyncio
async def test_write_then_restore_in_same_turn_reverts(tmp_path, monkeypatch):
    # End-to-end: a wrong overwrite followed by restore_file in the same turn
    # brings the file back to its pre-edit content (capture + serialize + restore).
    monkeypatch.chdir(tmp_path)
    (tmp_path / "doc.txt").write_text("ORIGINAL")

    backend = FakeBackend([
        [
            mock_llm_chunk(
                content="editing then undoing",
                tool_calls=[
                    _write_call("c1", 0, "doc.txt", "BROKEN"),
                    _restore_call("c2", 1, "doc.txt"),
                ],
            )
        ],
        [mock_llm_chunk(content="reverted")],
    ])
    loop = build_test_agent_loop(
        config=_config("write_file", "restore_file"),
        agent_name=BuiltinAgentName.AUTO_APPROVE,
        backend=backend,
    )

    [ev async for ev in loop.act("overwrite then undo")]

    assert (tmp_path / "doc.txt").read_text() == "ORIGINAL"
