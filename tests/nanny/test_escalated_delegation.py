"""Escalated permissions and the nanny: off by default, on with
nanny_reviews_escalated, or handed over per call / per session by the human
answering NANNY / NANNY_ALWAYS.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
import pytest

from privibe.core.agent_loop import AgentLoop, ToolExecutionResponse
from privibe.core.agents.models import BuiltinAgentName
from privibe.core.nanny import NannyReview, present_permissions
from privibe.core.tools.builtins.read_file import ReadFileArgs
from privibe.core.tools.permissions import PermissionScope, RequiredPermission
from privibe.core.types import ApprovalResponse
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.nanny.conftest import APPROVE, READY, REFUSE, build_nanny


class Human:
    """Answers in order; records every call and the review it came with."""

    def __init__(self, *answers: ApprovalResponse) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, NannyReview | None]] = []

    async def __call__(
        self,
        tool_name: str,
        args: BaseModel,
        tool_call_id: str,
        required_permissions: list[RequiredPermission] | None,
        review: NannyReview | None = None,
    ) -> tuple[ApprovalResponse, str | None]:
        self.calls.append((tool_name, review))
        answer = self.answers.pop(0) if self.answers else ApprovalResponse.YES
        if review is not None:
            review.human_choice = "allow" if answer is ApprovalResponse.YES else "no"
        return (answer, None)


def outside_loop(tmp_path: Path, monkeypatch, **config_overrides) -> tuple[AgentLoop, Path]:
    workdir = tmp_path / "repo"
    outside = tmp_path / "elsewhere"
    workdir.mkdir()
    outside.mkdir()
    (outside / "notes.md").write_text("x")
    (outside / "other.md").write_text("y")
    monkeypatch.chdir(workdir)
    config = build_test_vibe_config(
        system_prompt_id="tests", include_project_context=False, **config_overrides
    )
    config.protect_outside_workdir = True
    config.outside_workdir_exempt = []
    config.tools.setdefault("read_file", {})["outside_workdir_exempt"] = ["/dev/*"]
    loop = build_test_agent_loop(config=config, agent_name=BuiltinAgentName.DEFAULT)
    return loop, outside


async def decide(loop: AgentLoop, path: Path, call_id: str = "t1"):
    tool = loop.tool_manager.get("read_file")
    return await loop._should_execute_tool(tool, ReadFileArgs(path=str(path)), call_id)


@pytest.mark.asyncio
async def test_default_escalated_goes_to_the_human(tmp_path: Path, monkeypatch) -> None:
    loop, outside = outside_loop(tmp_path, monkeypatch)
    human = Human(ApprovalResponse.YES)
    loop.set_approval_callback(human)
    nanny = build_nanny(tmp_path, [READY, APPROVE])
    loop.set_nanny(nanny, owner=True)

    decision = await decide(loop, outside / "notes.md")

    assert decision.verdict is ToolExecutionResponse.EXECUTE
    assert human.calls == [("read_file", None)]
    assert nanny.decisions == []


@pytest.mark.asyncio
async def test_config_flag_sends_escalated_to_the_nanny(tmp_path: Path, monkeypatch) -> None:
    loop, outside = outside_loop(tmp_path, monkeypatch, nanny_reviews_escalated=True)
    human = Human()
    loop.set_approval_callback(human)
    nanny = build_nanny(tmp_path, [READY, APPROVE])
    loop.set_nanny(nanny, owner=True)

    decision = await decide(loop, outside / "notes.md")

    assert decision.verdict is ToolExecutionResponse.EXECUTE
    assert human.calls == []
    assert nanny.decisions[-1].approved_by_nanny
    ask_text = nanny.loop.backend.requests_messages[1][-1].content
    assert "escalated: normally only the human may allow this" in ask_text


@pytest.mark.asyncio
async def test_human_hands_one_call_to_the_nanny(tmp_path: Path, monkeypatch) -> None:
    loop, outside = outside_loop(tmp_path, monkeypatch)
    human = Human(ApprovalResponse.NANNY, ApprovalResponse.YES)
    loop.set_approval_callback(human)
    nanny = build_nanny(tmp_path, [READY, APPROVE])
    loop.set_nanny(nanny, owner=True)

    first = await decide(loop, outside / "notes.md", "t1")
    assert first.verdict is ToolExecutionResponse.EXECUTE
    assert [r for _, r in human.calls] == [None]
    assert len(nanny.decisions) == 1

    # Once means once: the next escalated call asks the human again.
    second = await decide(loop, outside / "other.md", "t2")
    assert second.verdict is ToolExecutionResponse.EXECUTE
    assert len(human.calls) == 2 and human.calls[1][1] is None
    assert len(nanny.decisions) == 1


@pytest.mark.asyncio
async def test_human_hands_calls_like_this_to_the_nanny_for_the_session(
    tmp_path: Path, monkeypatch
) -> None:
    loop, outside = outside_loop(tmp_path, monkeypatch)
    human = Human(ApprovalResponse.NANNY_ALWAYS)
    loop.set_approval_callback(human)
    nanny = build_nanny(tmp_path, [READY, APPROVE, APPROVE])
    loop.set_nanny(nanny, owner=True)

    first = await decide(loop, outside / "notes.md", "t1")
    assert first.verdict is ToolExecutionResponse.EXECUTE
    assert len(human.calls) == 1
    assert loop._nanny_delegations and loop._nanny_delegations[0].tool_name == "read_file"

    # Same directory, different file: covered, straight to the nanny.
    second = await decide(loop, outside / "other.md", "t2")
    assert second.verdict is ToolExecutionResponse.EXECUTE
    assert len(human.calls) == 1
    assert len(nanny.decisions) == 2
    # Nothing was granted: no session approval rule exists.
    assert loop._session_rules == []


@pytest.mark.asyncio
async def test_delegated_call_the_nanny_refuses_comes_back_with_the_review(
    tmp_path: Path, monkeypatch
) -> None:
    loop, outside = outside_loop(tmp_path, monkeypatch)
    human = Human(ApprovalResponse.NANNY, ApprovalResponse.NO)
    loop.set_approval_callback(human)
    nanny = build_nanny(tmp_path, [READY, REFUSE])
    loop.set_nanny(nanny, owner=True)

    decision = await decide(loop, outside / "notes.md")

    assert decision.verdict is ToolExecutionResponse.SKIP
    assert len(human.calls) == 2
    assert human.calls[0][1] is None
    review = human.calls[1][1]
    assert review is not None and not review.approved
    assert nanny.decisions[-1].human_decision == "no"


@pytest.mark.asyncio
async def test_nanny_answer_without_a_nanny_is_a_refusal(tmp_path: Path, monkeypatch) -> None:
    loop, outside = outside_loop(tmp_path, monkeypatch)
    human = Human(ApprovalResponse.NANNY)
    loop.set_approval_callback(human)

    decision = await decide(loop, outside / "notes.md")

    assert decision.verdict is ToolExecutionResponse.SKIP
    assert "No nanny is attached" in (decision.feedback or "")


def test_present_permissions_marks_escalated() -> None:
    plain = RequiredPermission(
        scope=PermissionScope.OUTSIDE_DIRECTORY, invocation_pattern="/etc/x", session_pattern="/etc/*", label="access to /etc"
    )
    hot = plain.model_copy(update={"escalated": True, "label": "outside workdir (/tmp/*)"})
    assert present_permissions([plain, hot]) == (
        "Permissions this call needs: access to /etc; outside workdir (/tmp/*) "
        "(escalated: normally only the human may allow this)\n"
    )
