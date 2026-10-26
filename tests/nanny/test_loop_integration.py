"""The agent loop with a nanny attached: approvals skip the callback,
refusals reach it with the review, failures fall back to the human, and
escalated permissions never see the nanny.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
import pytest

from privibe.core.agent_loop import AgentLoop, ToolExecutionResponse
from privibe.core.agents.models import BuiltinAgentName
from privibe.core.nanny import REFUSAL_PREFIX, NannyReview
from privibe.core.tools.base import ToolPermission
from privibe.core.tools.builtins.read_file import ReadFileArgs
from privibe.core.tools.permissions import RequiredPermission
from privibe.core.types import (
    ApprovalResponse,
    AssistantEvent,
    FunctionCall,
    ToolCall,
    ToolResultEvent,
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.nanny.conftest import APPROVE, READY, REFUSE, build_nanny, nanny_backend
from tests.stubs.fake_backend import FakeBackend


def todo_call(call_id: str = "call_1") -> ToolCall:
    return ToolCall(
        id=call_id,
        index=0,
        function=FunctionCall(name="todo", arguments='{"action": "read"}'),
    )


def worker_backend() -> FakeBackend:
    return FakeBackend([
        [
            mock_llm_chunk(
                content="Checking the todo list now.", tool_calls=[todo_call()]
            )
        ],
        [mock_llm_chunk(content="Done.")],
    ])


def make_loop(backend: FakeBackend) -> AgentLoop:
    config = build_test_vibe_config(
        enabled_tools=["todo"],
        tools={"todo": {"permission": ToolPermission.ASK.value}},
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=False,
    )
    return build_test_agent_loop(
        config=config, agent_name=BuiltinAgentName.DEFAULT, backend=backend
    )


class HumanCallback:
    """Records every call; accepts the review keyword the nanny path adds."""

    def __init__(self, answer: str = "allow") -> None:
        self.answer = answer
        self.calls: list[tuple[str, NannyReview | None]] = []
        self.talk_replies: list[str] = []

    async def __call__(
        self,
        tool_name: str,
        args: BaseModel,
        tool_call_id: str,
        required_permissions: list[RequiredPermission] | None,
        review: NannyReview | None = None,
    ) -> tuple[ApprovalResponse, str | None]:
        self.calls.append((tool_name, review))
        if review is not None and self.answer == "talk":
            self.talk_replies.append(await review.talk("why not?"))
            review.human_choice = "allow"
            return (ApprovalResponse.YES, None)
        if self.answer == "refuse":
            assert review is not None
            review.human_choice = "refuse"
            return (ApprovalResponse.NO, review.refusal_feedback())
        return (ApprovalResponse.YES, None)


async def run(loop: AgentLoop, prompt: str = "what is on my list?") -> list:
    return [event async for event in loop.act(prompt)]


@pytest.mark.asyncio
async def test_nanny_approval_runs_the_tool_without_the_callback(
    tmp_path: Path,
) -> None:
    loop = make_loop(worker_backend())
    human = HumanCallback()
    loop.set_approval_callback(human)
    nanny = build_nanny(tmp_path, [READY, APPROVE])
    loop.set_nanny(nanny, owner=True)

    events = await run(loop)

    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 1 and not results[0].skipped
    assert human.calls == []
    assert nanny.decisions[-1].approved_by_nanny
    ask_text = nanny.loop.backend.requests_messages[1][-1].content
    assert "what is on my list?" in ask_text  # the user prompt reached the nanny
    assert "Checking the todo list now." in ask_text  # the worker's words, as untrusted
    assert "Outcome: todo call call_1" in nanny._preamble[-1]


@pytest.mark.asyncio
async def test_nanny_refusal_reaches_the_human_and_the_worker_continues(
    tmp_path: Path,
) -> None:
    loop = make_loop(worker_backend())
    human = HumanCallback("refuse")
    loop.set_approval_callback(human)
    nanny = build_nanny(tmp_path, [READY, REFUSE])
    loop.set_nanny(nanny, owner=True)

    events = await run(loop)

    assert [name for name, _ in human.calls] == ["todo"]
    review = human.calls[0][1]
    assert review is not None and not review.approved
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert results[0].skipped and not results[0].cancelled
    assert results[0].skip_reason.startswith(REFUSAL_PREFIX)
    # Untagged feedback: the worker read it and answered again.
    assert isinstance(events[-1], AssistantEvent) and events[-1].content == "Done."
    assert nanny.decisions[-1].human_decision == "refuse"


@pytest.mark.asyncio
async def test_human_can_talk_then_allow(tmp_path: Path) -> None:
    loop = make_loop(worker_backend())
    human = HumanCallback("talk")
    loop.set_approval_callback(human)
    nanny = build_nanny(
        tmp_path, [READY, REFUSE, "Because it lists nothing I can see."]
    )
    loop.set_nanny(nanny, owner=True)

    events = await run(loop)

    assert human.talk_replies == ["Because it lists nothing I can see."]
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert not results[0].skipped
    assert nanny.decisions[-1].human_decision == "allow"
    assert len(nanny.decisions[-1].talk) == 1


@pytest.mark.asyncio
async def test_nanny_failure_falls_back_to_the_human(tmp_path: Path) -> None:
    loop = make_loop(worker_backend())
    human = HumanCallback()
    loop.set_approval_callback(human)
    nanny = build_nanny(
        tmp_path, backend=nanny_backend([], exception=RuntimeError("down"))
    )
    loop.set_nanny(nanny, owner=True)

    events = await run(loop)

    assert len(human.calls) == 1
    review = human.calls[0][1]
    assert review is not None and review.failed
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert not results[0].skipped


@pytest.mark.asyncio
async def test_escalated_permission_bypasses_the_nanny(
    tmp_path: Path, monkeypatch
) -> None:
    workdir = tmp_path / "repo"
    outside = tmp_path / "elsewhere"
    workdir.mkdir()
    outside.mkdir()
    (outside / "notes.md").write_text("x")
    monkeypatch.chdir(workdir)
    config = build_test_vibe_config(
        system_prompt_id="tests", include_project_context=False
    )
    config.protect_outside_workdir = True
    config.outside_workdir_exempt = []
    config.tools.setdefault("read_file", {})["outside_workdir_exempt"] = ["/dev/*"]
    loop = build_test_agent_loop(config=config, agent_name=BuiltinAgentName.DEFAULT)
    human = HumanCallback()
    loop.set_approval_callback(human)
    nanny = build_nanny(tmp_path, [READY, APPROVE])
    loop.set_nanny(nanny, owner=True)

    tool = loop.tool_manager.get("read_file")
    decision = await loop._should_execute_tool(
        tool, ReadFileArgs(path=str(outside / "notes.md")), "t1"
    )

    assert decision.verdict is ToolExecutionResponse.EXECUTE
    assert human.calls == [("read_file", None)]
    assert nanny.decisions == []
    assert nanny.loop.backend.requests_messages == []


@pytest.mark.asyncio
async def test_session_reset_of_the_owner_moves_the_nanny_files(tmp_path: Path) -> None:
    loop = make_loop(worker_backend())
    loop.set_approval_callback(HumanCallback())
    nanny = build_nanny(tmp_path, [READY, APPROVE])
    loop.set_nanny(nanny, owner=True)
    await run(loop)
    before = nanny.main_session_id
    loop._reset_session()
    assert nanny.main_session_id == loop.session_id != before


@pytest.mark.asyncio
async def test_borrowed_nanny_ignores_a_subloop_reset(tmp_path: Path) -> None:
    loop = make_loop(worker_backend())
    nanny = build_nanny(tmp_path, [READY, APPROVE])
    loop.set_nanny(nanny)  # borrowed, as a subagent would
    loop._reset_session()
    assert nanny.main_session_id == "main-session-1234"
