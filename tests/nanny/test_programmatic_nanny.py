from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
import pytest

from privibe.core.programmatic import _refuse_without_a_human
from privibe.core.types import ApprovalResponse, ToolResultEvent
from tests.nanny.conftest import APPROVE, READY, REFUSE, build_nanny, nanny_backend
from tests.nanny.test_loop_integration import make_loop, run, worker_backend


class Args(BaseModel):
    command: str = "echo hi"


async def headless_results(tmp_path: Path, nanny) -> list[ToolResultEvent]:
    loop = make_loop(worker_backend())
    loop.set_approval_callback(_refuse_without_a_human)
    loop.set_nanny(nanny, owner=True)
    events = await run(loop)
    return [e for e in events if isinstance(e, ToolResultEvent)]


@pytest.mark.asyncio
async def test_headless_run_executes_what_the_nanny_approves(tmp_path: Path) -> None:
    results = await headless_results(tmp_path, build_nanny(tmp_path, [READY, APPROVE]))

    assert len(results) == 1 and not results[0].skipped


@pytest.mark.asyncio
async def test_headless_run_refuses_what_the_nanny_refuses(tmp_path: Path) -> None:
    nanny = build_nanny(tmp_path, [READY, REFUSE])

    results = await headless_results(tmp_path, nanny)

    assert len(results) == 1 and results[0].skipped
    assert "deletes a directory outside the project" in (results[0].skip_reason or "")
    assert nanny.decisions[-1].human_decision == "no"


@pytest.mark.asyncio
async def test_headless_run_refuses_when_the_nanny_does_not_answer(
    tmp_path: Path,
) -> None:
    nanny = build_nanny(
        tmp_path, backend=nanny_backend([READY], exception=RuntimeError("boom"))
    )

    results = await headless_results(tmp_path, nanny)

    assert len(results) == 1 and results[0].skipped
    assert "nobody else to ask" in (results[0].skip_reason or "")


@pytest.mark.asyncio
async def test_an_escalated_call_with_no_review_is_refused() -> None:
    response, feedback = await _refuse_without_a_human("bash", Args(), "t1", [])

    assert response is ApprovalResponse.NO
    assert "programmatic mode has none" in (feedback or "")
