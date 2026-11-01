"""The loop against a scripted model: every exit path, every guard."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

from privibe.core.config import OrchestrateConfig
from privibe.core.orchestrate import (
    Bundle,
    JobSpec,
    Ledger,
    Outcome,
    OutcomeStatus,
    Progress,
    run_orchestration,
)
from privibe.core.orchestrate.prompt import Role

PLAN = "PLAN:\nEdit the file.\nTASKS:\n1. change return value\n2. check\n"
BRAINSTORM = "## Core difficulty\nnone\n## Candidate approaches\nedit\n"


class ScriptedModel:
    """Answers per role, in order; records every call it received."""

    def __init__(
        self, answers: dict[Role, list[str]], edit: Path | None = None
    ) -> None:
        self.answers = {k: list(v) for k, v in answers.items()}
        self.calls: list[tuple[Role, str, str, bool]] = []
        self.edit = edit

    async def __call__(
        self, role: Role, message: str, effort: str, answer_only: bool
    ) -> str:
        self.calls.append((role, message, effort, answer_only))
        if role is Role.EXECUTE and self.edit is not None:
            self.edit.write_text("def f():\n    return 2\n")
        queue = self.answers.get(role, [])
        if not queue:
            raise AssertionError(f"scripted model has no answer left for {role}")
        return queue.pop(0)


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path, Bundle, Ledger]:
    target = tmp_path / "t.py"
    target.write_text("def f():\n    return 1\n")
    bundle = Bundle.load([str(target)], max_bytes=10_000)
    ledger = Ledger(tmp_path / "ledger", plan_chars=4000, notes_chars=8000)
    return tmp_path, target, bundle, ledger


def _check_cmd(target: Path) -> str:
    # Exit 0 iff f() returns 2, no pytest involved.
    code = f"import importlib.util as u;s=u.spec_from_file_location('t',r'{target}');m=u.module_from_spec(s);s.loader.exec_module(m);raise SystemExit(0 if m.f()==2 else 1)"
    return f'"{sys.executable}" -c "{code}"'


async def _run(
    job: JobSpec, bundle: Bundle, ledger: Ledger, model: ScriptedModel, cwd: Path, **cfg
):
    config = OrchestrateConfig(**cfg)
    events: list[Progress | Outcome] = []
    async for ev in run_orchestration(
        job, bundle=bundle, ledger=ledger, call=model, config=config, cwd=cwd
    ):
        events.append(ev)
    outcome = events[-1]
    assert isinstance(outcome, Outcome)
    return outcome, events


@pytest.mark.asyncio
async def test_done_through_verifier(workspace) -> None:
    cwd, target, bundle, ledger = workspace
    model = ScriptedModel(
        {
            Role.PLAN: [PLAN],
            Role.BRAINSTORM: [BRAINSTORM],
            # The verifier passes right after the worker, so the loop ends
            # there and the manager is never asked a second time.
            Role.MANAGE: ["1. [todo] change return value\n2. [todo] check\nNEXT: 1"],
            Role.EXECUTE: ["edited\n## Notes\nf returns 2 now\n## Next steps\nnone"],
        },
        edit=target,
    )
    job = JobSpec(
        goal="make f return 2",
        files=[str(target)],
        scope=[str(cwd)],
        done_command=_check_cmd(target),
    )
    outcome, events = await _run(job, bundle, ledger, model, cwd)

    assert outcome.status is OutcomeStatus.DONE
    assert outcome.detail == "verifier passes after a worker"
    assert outcome.rounds == 1
    assert outcome.calls == 4
    assert [r for r, _m, _e, _a in model.calls] == [
        Role.PLAN,
        Role.BRAINSTORM,
        Role.MANAGE,
        Role.EXECUTE,
    ]
    assert outcome.verifier is not None and outcome.verifier.passed
    assert outcome.changed_files == [str(target)]
    assert ledger.read_notes() == "f returns 2 now"
    assert ledger.read_plan() == "Edit the file."
    # tasks.json is what the one manager answer wrote: nothing marked done.
    assert [t.done for t in ledger.read_tasks()] == [False, False]
    # Efforts and answer-only flags per role.
    flags = {(r, a) for r, _m, _e, a in model.calls}
    assert (Role.PLAN, True) in flags and (Role.MANAGE, True) in flags
    assert (Role.EXECUTE, False) in flags
    efforts = {r: e for r, _m, e, _a in model.calls}
    assert efforts[Role.PLAN] == "low" and efforts[Role.EXECUTE] == "medium"
    # The one manager call came before any worker or verifier run.
    manage_msgs = [m for r, m, _e, _a in model.calls if r is Role.MANAGE]
    assert len(manage_msgs) == 1
    assert "(not run yet)" in manage_msgs[0]
    assert "Last worker report" not in manage_msgs[0]
    assert (ledger.root / "calls" / "001-plan.md").is_file()
    assert "end done" in ledger.log_path.read_text()


@pytest.mark.asyncio
async def test_failing_verifier_vetoes_done(workspace) -> None:
    cwd, target, bundle, ledger = workspace
    model = ScriptedModel({
        Role.PLAN: [PLAN],
        Role.BRAINSTORM: [BRAINSTORM],
        # Manager claims done at once, twice; the file never changes. Each
        # DONE is vetoed by the failing verifier and the next open task runs.
        Role.MANAGE: [
            "1. [todo] change return value\n2. [todo] check\nNEXT: DONE",
            "1. [todo] change return value\n2. [todo] check\nNEXT: DONE",
        ],
        Role.EXECUTE: ["tried 1\n## Notes\nn1\n", "tried 2\n## Notes\nn2\n"],
        Role.FINALIZE: ["## Result\nI could not fix it."],
    })
    job = JobSpec(
        goal="g",
        files=[str(target)],
        scope=[],
        done_command=_check_cmd(target),
        max_rounds=2,
    )
    outcome, _ = await _run(job, bundle, ledger, model, cwd)
    assert outcome.status is OutcomeStatus.ROUNDS_EXHAUSTED
    assert "the finalizer ran and the verifier still fails" in outcome.detail
    assert [r for r, _m, _e, _a in model.calls] == [
        Role.PLAN,
        Role.BRAINSTORM,
        Role.MANAGE,
        Role.EXECUTE,
        Role.MANAGE,
        Role.EXECUTE,
        Role.FINALIZE,
    ]
    assert outcome.verifier is not None and not outcome.verifier.passed
    log = ledger.log_path.read_text()
    assert "DONE but the verifier fails" in log
    assert "DONE overruled" in log
    # Second manage call saw the FAIL.
    second = [m for r, m, _e, _a in model.calls if r is Role.MANAGE][1]
    assert "FAIL, exit code 1" in second


@pytest.mark.asyncio
async def test_done_without_verifier_is_unverified(workspace) -> None:
    cwd, target, bundle, ledger = workspace
    # DONE counts only after real work: a worker edits the file first.
    model = ScriptedModel(
        {
            Role.PLAN: [PLAN],
            Role.BRAINSTORM: [BRAINSTORM],
            Role.MANAGE: [
                "1. [todo] change return value\nNEXT: 1",
                "1. [done] change return value\nNEXT: DONE",
            ],
            Role.EXECUTE: ["edited\n## Notes\nn\n"],
        },
        edit=target,
    )
    job = JobSpec(goal="g", files=[str(target)], scope=[])
    outcome, _ = await _run(job, bundle, ledger, model, cwd)
    assert outcome.status is OutcomeStatus.DONE_UNVERIFIED
    assert outcome.detail == "manager declared done; no verifier"
    assert outcome.verifier is None
    assert outcome.changed_files == [str(target)]
    assert [r for r, _m, _e, _a in model.calls] == [
        Role.PLAN,
        Role.BRAINSTORM,
        Role.MANAGE,
        Role.EXECUTE,
        Role.MANAGE,
    ]


@pytest.mark.asyncio
async def test_same_task_twice_stalls(workspace) -> None:
    cwd, target, bundle, ledger = workspace
    manage = "1. [todo] change return value\nNEXT: 1"
    model = ScriptedModel({
        Role.PLAN: [PLAN],
        Role.BRAINSTORM: [BRAINSTORM],
        Role.MANAGE: [manage, manage],
        Role.EXECUTE: ["tried\n## Notes\nstill not\n"],
        Role.FINALIZE: ["## Result\nnothing more"],
    })
    job = JobSpec(goal="g", files=[str(target)], scope=[])
    outcome, _ = await _run(job, bundle, ledger, model, cwd)
    assert outcome.status is OutcomeStatus.STALLED
    assert "reissued" in outcome.detail
    assert "the finalizer ran, nothing verified it" in outcome.detail
    assert outcome.rounds == 2
    # plan, brainstorm, manage, execute, manage, finalize; the second execute
    # never ran
    assert outcome.calls == 6
    assert [r for r, _m, _e, _a in model.calls] == [
        Role.PLAN,
        Role.BRAINSTORM,
        Role.MANAGE,
        Role.EXECUTE,
        Role.MANAGE,
        Role.FINALIZE,
    ]


@pytest.mark.asyncio
async def test_manage_parse_error_retries_once_then_stalls(workspace) -> None:
    cwd, target, bundle, ledger = workspace
    model = ScriptedModel({
        Role.PLAN: [PLAN],
        Role.BRAINSTORM: [BRAINSTORM],
        # Every manage reply is prose: each round costs 2 manage calls and the
        # fallback takes the first open task, until no task is left.
        Role.MANAGE: [
            "I think we are good.",
            "Still prose, no list.",
            "Looks fine.",
            "Still fine.",
            "All good.",
            "Really good.",
        ],
        Role.EXECUTE: ["did 1\n## Notes\nn1\n", "did 2\n## Notes\nn2\n"],
        Role.FINALIZE: ["## Result\nnothing verified"],
    })
    job = JobSpec(goal="g", files=[str(target)], scope=[], max_rounds=5)
    outcome, _ = await _run(job, bundle, ledger, model, cwd)
    assert outcome.status is OutcomeStatus.STALLED
    assert "task list could not be parsed" in outcome.detail
    assert "the finalizer ran, nothing verified it" in outcome.detail
    assert [r for r, _m, _e, _a in model.calls] == [
        Role.PLAN,
        Role.BRAINSTORM,
        Role.MANAGE,
        Role.MANAGE,
        Role.EXECUTE,
        Role.MANAGE,
        Role.MANAGE,
        Role.EXECUTE,
        Role.MANAGE,
        Role.MANAGE,
        Role.FINALIZE,
    ]
    assert outcome.calls == 11
    retry = [m for r, m, _e, _a in model.calls if r is Role.MANAGE][1]
    assert "could not be parsed: no 'NEXT:' line" in retry
    assert "I think we are good." in retry


@pytest.mark.asyncio
async def test_plan_parse_error_recovers_on_retry(workspace) -> None:
    cwd, target, bundle, ledger = workspace
    model = ScriptedModel(
        {
            Role.PLAN: ["no shape at all", PLAN],
            Role.BRAINSTORM: [BRAINSTORM],
            Role.MANAGE: [
                "1. [todo] change return value\nNEXT: 1",
                "1. [done] change return value\nNEXT: DONE",
            ],
            Role.EXECUTE: ["edited\n## Notes\nn\n"],
        },
        edit=target,
    )
    job = JobSpec(goal="g", files=[str(target)], scope=[])
    outcome, _ = await _run(job, bundle, ledger, model, cwd)
    assert outcome.status is OutcomeStatus.DONE_UNVERIFIED
    assert [r for r, _m, _e, _a in model.calls] == [
        Role.PLAN,
        Role.PLAN,
        Role.BRAINSTORM,
        Role.MANAGE,
        Role.EXECUTE,
        Role.MANAGE,
    ]
    assert ledger.read_plan() == "Edit the file."
    assert "plan parse error" in ledger.log_path.read_text()


@pytest.mark.asyncio
async def test_rounds_exhausted_and_notes_kept_when_worker_omits_them(
    workspace,
) -> None:
    cwd, target, bundle, ledger = workspace
    model = ScriptedModel({
        Role.PLAN: [PLAN],
        Role.BRAINSTORM: [BRAINSTORM],
        Role.MANAGE: [
            "1. [todo] a\n2. [todo] b\nNEXT: 1",
            "1. [done] a\n2. [todo] b\nNEXT: 2",
        ],
        Role.EXECUTE: ["did a, no notes section", "did b, no notes section"],
        Role.FINALIZE: ["## Result\nnothing verified"],
    })
    job = JobSpec(goal="g", files=[str(target)], scope=[], max_rounds=2)
    outcome, events = await _run(job, bundle, ledger, model, cwd)
    assert outcome.status is OutcomeStatus.ROUNDS_EXHAUSTED
    assert "the finalizer ran, nothing verified it" in outcome.detail
    assert [r for r, _m, _e, _a in model.calls] == [
        Role.PLAN,
        Role.BRAINSTORM,
        Role.MANAGE,
        Role.EXECUTE,
        Role.MANAGE,
        Role.EXECUTE,
        Role.FINALIZE,
    ]
    assert ledger.read_notes() == BRAINSTORM.strip()
    assert ledger.log_path.read_text().count("notes kept") == 2
    progress = [e.message for e in events if isinstance(e, Progress)]
    assert progress[0] == "manager · plan"
    assert any(p.startswith("worker · round 2/2 · task 2: ") for p in progress)


@pytest.mark.asyncio
async def test_head_bytes_identical_in_every_call(workspace) -> None:
    """Wire-level: the text before the separator is the same in all 5 calls
    even though the worker changed the file and the ledger filled up. There
    is no done_command, so the loop does not stop after the worker and the
    last call (the manager's DONE) sees the edited file.
    """
    from privibe.core.orchestrate.prompt import HEAD_TAIL_SEPARATOR

    cwd, target, bundle, ledger = workspace
    model = ScriptedModel(
        {
            Role.PLAN: [PLAN],
            Role.BRAINSTORM: [BRAINSTORM],
            Role.MANAGE: ["1. [todo] a\nNEXT: 1", "1. [done] a\nNEXT: DONE"],
            Role.EXECUTE: ["x\n## Notes\nnew notes\n"],
        },
        edit=target,
    )
    job = JobSpec(goal="g", files=[str(target)], scope=[str(cwd)])
    outcome, _ = await _run(job, bundle, ledger, model, cwd)
    assert outcome.status is OutcomeStatus.DONE_UNVERIFIED
    assert [r for r, _m, _e, _a in model.calls] == [
        Role.PLAN,
        Role.BRAINSTORM,
        Role.MANAGE,
        Role.EXECUTE,
        Role.MANAGE,
    ]
    heads = {m.partition(HEAD_TAIL_SEPARATOR)[0] for _r, m, _e, _a in model.calls}
    assert len(heads) == 1
    tails = [m.partition(HEAD_TAIL_SEPARATOR)[2] for _r, m, _e, _a in model.calls]
    assert "return 2" in tails[-1] and "return 2" not in heads.pop()


@pytest.mark.asyncio
async def test_plan_and_manage_see_the_round_budget(workspace) -> None:
    cwd, target, bundle, ledger = workspace
    # The worker's edit makes the verifier pass, so the loop ends after round 1.
    model = ScriptedModel(
        {
            Role.PLAN: [PLAN],
            Role.BRAINSTORM: [BRAINSTORM],
            Role.MANAGE: ["1. [todo] a\n2. [todo] b\nNEXT: 1"],
            Role.EXECUTE: ["x\n## Notes\nn\n"],
        },
        edit=target,
    )
    job = JobSpec(
        goal="g",
        files=[str(target)],
        scope=[],
        max_rounds=3,
        done_command=_check_cmd(target),
    )
    outcome, _ = await _run(job, bundle, ledger, model, cwd)
    assert outcome.status is OutcomeStatus.DONE
    assert [r for r, _m, _e, _a in model.calls] == [
        Role.PLAN,
        Role.BRAINSTORM,
        Role.MANAGE,
        Role.EXECUTE,
    ]
    plan_msg = next(m for r, m, _e, _a in model.calls if r is Role.PLAN)
    manage_msg = next(m for r, m, _e, _a in model.calls if r is Role.MANAGE)
    assert "Plan at most 2 tasks" in plan_msg
    assert "round 1 of 3, 2 left" in manage_msg
