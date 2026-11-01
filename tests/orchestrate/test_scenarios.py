"""Scenario tests for the orchestrate loop, one per scenario in orchestrate-scenarios/out/scenarios.md."""

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
PLAN3 = "PLAN:\nEdit the file.\nTASKS:\n1. change return value\n2. check\n3. wrap up\n"
BRAINSTORM = "## Core difficulty\nnone\n## Candidate approaches\nedit\n"
WORKER = "did it\n## Notes\nnotes\n## Next steps\nnone"


class ScriptedModel:
    """Answers per role, in order; records every call it received.

    The loop swallows the AssertionError raised for a missing answer, so every
    test asserts the exact list of roles in `calls`.
    """

    def __init__(
        self,
        answers: dict[Role, list[str]],
        edit: Path | None = None,
        *,
        edit_roles: tuple[Role, ...] = (Role.EXECUTE,),
        edit_from: int = 1,
        failures: dict[Role, int] | None = None,
    ) -> None:
        self.answers = {k: list(v) for k, v in answers.items()}
        self.calls: list[tuple[Role, str, str, bool]] = []
        self.edit = edit
        # `edit` is written when a role in `edit_roles` is called, starting at
        # that role's `edit_from`-th call. `failures` makes the first n calls
        # of a role raise ConnectionError (each attempt counts).
        self.edit_roles = edit_roles
        self.edit_from = edit_from
        self.failures = dict(failures or {})
        self._seen: dict[Role, int] = {}

    async def __call__(
        self, role: Role, message: str, effort: str, answer_only: bool
    ) -> str:
        self.calls.append((role, message, effort, answer_only))
        if self.failures.get(role, 0) > 0:
            self.failures[role] -= 1
            raise ConnectionError("backend dropped the connection")
        self._seen[role] = self._seen.get(role, 0) + 1
        if (
            role in self.edit_roles
            and self.edit is not None
            and self._seen[role] >= self.edit_from
        ):
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
    # -B: no stale .pyc (same mtime and size) hides an edit made within a second.
    return f'"{sys.executable}" -B -c "{code}"'


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


def _roles(model: ScriptedModel) -> list[Role]:
    return [r for r, _m, _e, _a in model.calls]


def _messages(model: ScriptedModel, role: Role) -> list[str]:
    return [m for r, m, _e, _a in model.calls if r is role]


P, B, M, E, F = (
    Role.PLAN,
    Role.BRAINSTORM,
    Role.MANAGE,
    Role.EXECUTE,
    Role.FINALIZE,
)


@pytest.mark.asyncio
async def test_s1_2_manager_garbage_until_no_task_left(workspace) -> None:
    cwd, target, bundle, ledger = workspace
    model = ScriptedModel({
        Role.PLAN: [PLAN],
        Role.BRAINSTORM: [BRAINSTORM],
        Role.MANAGE: ["Looks good, let's continue"] * 6,
        Role.EXECUTE: [WORKER, WORKER],
        Role.FINALIZE: ["## Result\nnothing more"],
    })
    job = JobSpec(
        goal="g",
        files=[str(target)],
        scope=[],
        done_command=_check_cmd(target),
        max_rounds=5,
    )
    outcome, _ = await _run(job, bundle, ledger, model, cwd)
    assert outcome.status is OutcomeStatus.STALLED
    assert (
        "manager's task list could not be parsed and no task is left" in outcome.detail
    )
    assert "the finalizer ran and the verifier still fails" in outcome.detail
    assert _roles(model) == [P, B, M, M, E, M, M, E, M, M, F]
    assert outcome.calls == 11
    log = ledger.log_path.read_text()
    # The verifier ran after each of the 2 workers and after the finalizer.
    assert log.count("verifier exit 1") == 3
    assert log.count("manager unparseable; taking the first open task") == 2
    # Round 2 wrote task 1 as done; task 2 is marked done only in the harness'
    # own keys, the list on disk is not rewritten once no task is left.
    assert [t.done for t in ledger.read_tasks()] == [True, False]


@pytest.mark.asyncio
async def test_s2_1_verifier_passes_after_worker_manager_never_done(
    workspace,
) -> None:
    cwd, target, bundle, ledger = workspace
    model = ScriptedModel(
        {
            Role.PLAN: [PLAN3],
            Role.BRAINSTORM: [BRAINSTORM],
            Role.MANAGE: [
                "1. [todo] change return value\n2. [todo] check\n3. [todo] wrap up\nNEXT: 1"
            ],
            Role.EXECUTE: [WORKER],
        },
        edit=target,
    )
    job = JobSpec(
        goal="g",
        files=[str(target)],
        scope=[str(cwd)],
        done_command=_check_cmd(target),
        max_rounds=5,
    )
    outcome, _ = await _run(job, bundle, ledger, model, cwd)
    assert outcome.status is OutcomeStatus.DONE
    assert "verifier passes" in outcome.detail
    assert _roles(model) == [P, B, M, E]
    assert outcome.calls == 4
    assert outcome.rounds == 1
    assert outcome.verifier is not None and outcome.verifier.passed


@pytest.mark.asyncio
async def test_s2_2_bare_done_on_untouched_workspace_is_overruled(workspace) -> None:
    cwd, target, bundle, ledger = workspace
    lazy = "1. [todo] change return value\n2. [todo] check\n3. [todo] wrap up\nNEXT: DONE"
    model = ScriptedModel(
        {
            Role.PLAN: [PLAN3],
            Role.BRAINSTORM: [BRAINSTORM],
            Role.MANAGE: [lazy, lazy],
            Role.EXECUTE: [WORKER],
            Role.FINALIZE: ["## Result\nnothing"],
        },
        edit=target,
    )
    job = JobSpec(goal="g", files=[str(target)], scope=[], max_rounds=3)
    outcome, _ = await _run(job, bundle, ledger, model, cwd)
    # The first DONE did not close the job: a worker ran before anything ended.
    assert _roles(model) == [P, B, M, E, M]
    log = ledger.log_path.read_text()
    assert "manager said DONE before any work was done; overruled" in log
    assert "DONE overruled; running the first open task 1" in log
    # A DONE is reported done_unverified only with real changes behind it.
    assert outcome.status is OutcomeStatus.DONE_UNVERIFIED
    assert outcome.changed_files == [str(target)]
    assert outcome.verifier is None
    assert outcome.detail == "manager declared done; no verifier"


@pytest.mark.asyncio
async def test_s2_3_rounds_exhausted_failing_verifier_runs_finalizer_once(
    workspace,
) -> None:
    cwd, target, bundle, ledger = workspace
    model = ScriptedModel({
        Role.PLAN: [PLAN3],
        Role.BRAINSTORM: [BRAINSTORM],
        Role.MANAGE: [
            "1. [todo] a\n2. [todo] b\n3. [todo] c\nNEXT: 1",
            "1. [done] a\n2. [todo] b\n3. [todo] c\nNEXT: 2",
        ],
        Role.EXECUTE: [WORKER, WORKER],
        Role.FINALIZE: ["## Result\nSorry, could not fix it."],
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
    assert (
        outcome.detail
        == "2 rounds used; the finalizer ran and the verifier still fails"
    )
    assert _roles(model) == [P, B, M, E, M, E, F]
    assert outcome.calls == 7
    finalize_msg = _messages(model, Role.FINALIZE)[0]
    assert finalize_msg.rstrip().endswith("Why the loop ended: 2 rounds used.")
    assert model.calls[-1][3] is False  # finalizer may use tools
    log = ledger.log_path.read_text()
    assert "finalizer: 2 rounds used" in log
    # Verifier ran after each worker and once more after the finalizer.
    assert log.count("verifier exit 1") == 3
    assert outcome.verifier is not None and not outcome.verifier.passed
    for name in ("plan.md", "notes.md", "tasks.json", "rounds.log"):
        assert (ledger.root / name).is_file()
    calls_dir = ledger.root / "calls"
    assert sorted(p.name for p in calls_dir.iterdir()) == [
        "001-plan.md",
        "002-brainstorm.md",
        "003-manage.md",
        "004-execute.md",
        "005-manage.md",
        "006-execute.md",
        "007-finalize.md",
    ]


@pytest.mark.asyncio
async def test_s2_4_same_task_twice_goes_to_finalizer(workspace) -> None:
    cwd, target, bundle, ledger = workspace
    manage = (
        "1. [todo] change return value\n2. [todo] check\n3. [todo] wrap up\nNEXT: 1"
    )
    model = ScriptedModel({
        Role.PLAN: [PLAN3],
        Role.BRAINSTORM: [BRAINSTORM],
        Role.MANAGE: [manage, manage],
        Role.EXECUTE: [WORKER],
        Role.FINALIZE: ["## Result\nnothing more"],
    })
    job = JobSpec(
        goal="g",
        files=[str(target)],
        scope=[],
        done_command=_check_cmd(target),
        max_rounds=6,
    )
    outcome, _ = await _run(job, bundle, ledger, model, cwd)
    assert outcome.status is OutcomeStatus.STALLED
    assert outcome.detail.startswith(
        "manager reissued the same task: change return value"
    )
    assert "the finalizer ran and the verifier still fails" in outcome.detail
    # The second worker never ran; the loop did not spin on the repeated task.
    assert _roles(model) == [P, B, M, E, M, F]
    assert outcome.calls == 6
    finalize_msg = _messages(model, Role.FINALIZE)[0]
    assert finalize_msg.rstrip().endswith(
        "Why the loop ended: manager reissued the same task: change return value."
    )


@pytest.mark.asyncio
async def test_s2_5_same_task_with_different_punctuation_is_the_same(
    workspace,
) -> None:
    cwd, target, bundle, ledger = workspace
    model = ScriptedModel({
        Role.PLAN: [PLAN3],
        Role.BRAINSTORM: [BRAINSTORM],
        Role.MANAGE: [
            "1. [todo] Fix parse_manage.\n2. [todo] other\nNEXT: 1",
            "1. [todo] fix  PARSE_MANAGE\n2. [todo] other\nNEXT: 1",
        ],
        Role.EXECUTE: [WORKER],
        Role.FINALIZE: ["## Result\nnothing more"],
    })
    job = JobSpec(
        goal="g",
        files=[str(target)],
        scope=[],
        done_command=_check_cmd(target),
        max_rounds=6,
    )
    outcome, _ = await _run(job, bundle, ledger, model, cwd)
    assert outcome.status is OutcomeStatus.STALLED
    assert "manager reissued the same task" in outcome.detail
    # No second worker for the reworded task; the finalizer runs at once.
    assert _roles(model) == [P, B, M, E, M, F]
    assert outcome.rounds == 2
    assert outcome.calls == 6


LIST3 = "1. [todo] a\n2. [todo] b\n3. [todo] c\n"


@pytest.mark.asyncio
async def test_s3_1_manager_keeps_saying_done_while_verifier_fails(workspace) -> None:
    cwd, target, bundle, ledger = workspace
    done = "1. [todo] change return value\n2. [todo] check\nNEXT: DONE"
    model = ScriptedModel({
        Role.PLAN: [PLAN],
        Role.BRAINSTORM: [BRAINSTORM],
        Role.MANAGE: [done] * 4,
        Role.EXECUTE: [WORKER, WORKER],
        Role.FINALIZE: ["## Result\nnothing more"],
    })
    job = JobSpec(
        goal="g", files=[str(target)], scope=[], done_command="false", max_rounds=4
    )
    outcome, _ = await _run(job, bundle, ledger, model, cwd)
    roles = _roles(model)
    # Workers did run: the loop did not burn the budget on manage calls alone.
    assert roles.count(E) >= 1
    assert roles.count(M) < 4 or roles.count(E) >= 1
    assert roles.count(F) == 1 and roles[-1] is F
    assert roles.count(P) == 1 and roles.count(B) == 1
    # Never succeeded, and ended with a status and a reason.
    assert outcome.status in {OutcomeStatus.STALLED, OutcomeStatus.ROUNDS_EXHAUSTED}
    assert outcome.detail
    assert "the finalizer ran and the verifier still fails" in outcome.detail
    assert outcome.verifier is not None and not outcome.verifier.passed
    log = ledger.log_path.read_text()
    assert "manager said DONE but the verifier fails; continuing" in log
    assert "DONE overruled" in log


@pytest.mark.asyncio
async def test_s3_2_verifier_passes_at_done_without_any_worker(workspace) -> None:
    cwd, target, bundle, ledger = workspace
    model = ScriptedModel({
        Role.PLAN: [PLAN],
        Role.BRAINSTORM: [BRAINSTORM],
        Role.MANAGE: ["1. [todo] change return value\n2. [todo] check\nNEXT: DONE"],
    })
    job = JobSpec(
        goal="g", files=[str(target)], scope=[], done_command="true", max_rounds=3
    )
    outcome, _ = await _run(job, bundle, ledger, model, cwd)
    assert outcome.status is OutcomeStatus.DONE
    assert _roles(model) == [P, B, M]
    assert outcome.calls == 3
    assert outcome.verifier is not None and outcome.verifier.exit_code == 0
    assert outcome.changed_files == []


@pytest.mark.asyncio
async def test_s4_1_loop_ends_with_passing_verifier_finalizer_skipped(
    workspace,
) -> None:
    cwd, target, bundle, ledger = workspace
    model = ScriptedModel(
        {
            Role.PLAN: [PLAN3],
            Role.BRAINSTORM: [BRAINSTORM],
            Role.MANAGE: [
                LIST3 + "NEXT: 1",
                "1. [done] a\n2. [todo] b\n3. [todo] c\nNEXT: 2",
            ],
            Role.EXECUTE: [WORKER, WORKER],
        },
        edit=target,
        edit_from=2,  # the file is fixed by worker 2, not worker 1
    )
    job = JobSpec(
        goal="g",
        files=[str(target)],
        scope=[str(cwd)],
        done_command=_check_cmd(target),
        max_rounds=2,
    )
    outcome, events = await _run(job, bundle, ledger, model, cwd)
    assert outcome.status is OutcomeStatus.DONE
    assert "verifier passes" in outcome.detail
    assert _roles(model) == [P, B, M, E, M, E]
    assert outcome.calls == 6
    assert outcome.rounds == 2
    assert outcome.verifier is not None and outcome.verifier.passed
    assert "finalizer" not in ledger.log_path.read_text()
    progress = [e.message for e in events if isinstance(e, Progress)]
    assert "finalizer" not in progress


@pytest.mark.asyncio
async def test_s4_2_finalizer_repairs_failing_job_and_verifier_says_done(
    workspace,
) -> None:
    cwd, target, bundle, ledger = workspace
    manage = LIST3 + "NEXT: 1"
    model = ScriptedModel(
        {
            Role.PLAN: [PLAN3],
            Role.BRAINSTORM: [BRAINSTORM],
            Role.MANAGE: [manage, manage],
            Role.EXECUTE: [WORKER],
            Role.FINALIZE: ["## Result\nfixed f to return 2"],
        },
        edit=target,
        edit_roles=(F,),  # only the finalizer repairs the file
    )
    job = JobSpec(
        goal="g",
        files=[str(target)],
        scope=[str(cwd)],
        done_command=_check_cmd(target),
        max_rounds=6,
    )
    outcome, _ = await _run(job, bundle, ledger, model, cwd)
    assert outcome.status is OutcomeStatus.DONE
    assert outcome.detail == "finalizer completed the job: manager reissued the same task: a"
    assert _roles(model) == [P, B, M, E, M, F]
    assert outcome.calls == 6
    assert outcome.changed_files == [str(target)]
    assert outcome.verifier is not None and outcome.verifier.passed
    fin = [c for c in model.calls if c[0] is F]
    assert len(fin) == 1
    assert fin[0][3] is False  # answer_only False: the finalizer may edit
    assert fin[0][1].rstrip().endswith(
        "Why the loop ended: manager reissued the same task: a."
    )
    assert "FAIL, exit code 1" in fin[0][1]  # the last verifier run


@pytest.mark.asyncio
async def test_s4_3_finalizer_without_done_command_keeps_original_status(
    workspace,
) -> None:
    cwd, target, bundle, ledger = workspace
    model = ScriptedModel(
        {
            Role.PLAN: [PLAN3],
            Role.BRAINSTORM: [BRAINSTORM],
            Role.MANAGE: [
                LIST3 + "NEXT: 1",
                "1. [done] a\n2. [todo] b\n3. [todo] c\nNEXT: 2",
            ],
            Role.EXECUTE: [WORKER, WORKER],
            Role.FINALIZE: ["## Result\neverything is done"],
        },
        edit=target,
        edit_roles=(F,),
    )
    job = JobSpec(goal="g", files=[str(target)], scope=[], max_rounds=2)
    outcome, _ = await _run(job, bundle, ledger, model, cwd)
    assert outcome.status is OutcomeStatus.ROUNDS_EXHAUSTED
    assert outcome.detail == "2 rounds used; the finalizer ran, nothing verified it"
    assert outcome.verifier is None
    assert _roles(model) == [P, B, M, E, M, E, F]
    assert outcome.calls == 7
    assert outcome.changed_files == [str(target)]
    assert "verifier exit" not in ledger.log_path.read_text()


def _exhausted_job(target: Path, cwd: Path) -> JobSpec:
    return JobSpec(
        goal="g",
        files=[str(target)],
        scope=[str(cwd)],
        done_command=_check_cmd(target),
        max_rounds=2,
    )


def _two_round_script(finalize: str) -> dict[Role, list[str]]:
    return {
        Role.PLAN: [PLAN3],
        Role.BRAINSTORM: [BRAINSTORM],
        Role.MANAGE: [
            LIST3 + "NEXT: 1",
            "1. [done] a\n2. [todo] b\n3. [todo] c\nNEXT: 2",
        ],
        Role.EXECUTE: [WORKER, WORKER],
        Role.FINALIZE: [finalize],
    }


@pytest.mark.asyncio
async def test_s4_4_finalizer_garbage_is_not_retried_status_stands(workspace) -> None:
    cwd, target, bundle, ledger = workspace
    model = ScriptedModel(_two_round_script(""))  # empty finalizer answer
    outcome, _ = await _run(_exhausted_job(target, cwd), bundle, ledger, model, cwd)
    assert outcome.status is OutcomeStatus.ROUNDS_EXHAUSTED
    assert "the finalizer ran and the verifier still fails" in outcome.detail
    assert _roles(model) == [P, B, M, E, M, E, F]
    assert _roles(model).count(F) == 1  # never parsed, never retried
    transcript = ledger.root / "calls" / "007-finalize.md"
    assert transcript.is_file() and transcript.read_text() == ""
    assert outcome.verifier is not None and not outcome.verifier.passed


@pytest.mark.asyncio
async def test_s4_4_repair_with_garbage_answer_still_done(workspace) -> None:
    cwd, target, bundle, ledger = workspace
    model = ScriptedModel(
        _two_round_script("fixed!"),  # no '## Result' heading
        edit=target,
        edit_roles=(F,),
    )
    outcome, _ = await _run(_exhausted_job(target, cwd), bundle, ledger, model, cwd)
    assert outcome.status is OutcomeStatus.DONE
    assert outcome.detail == "finalizer completed the job: 2 rounds used"
    assert _roles(model) == [P, B, M, E, M, E, F]
    assert (ledger.root / "calls" / "007-finalize.md").read_text() == "fixed!"


@pytest.mark.asyncio
async def test_s5_5_worker_call_raises_is_retried_then_empty_answer(workspace) -> None:
    cwd, target, bundle, ledger = workspace
    model = ScriptedModel(
        _two_round_script("## Result\nnothing"),
        failures={Role.EXECUTE: 2},  # both attempts of worker 1 raise
    )
    outcome, _ = await _run(_exhausted_job(target, cwd), bundle, ledger, model, cwd)
    # One retry, then an empty answer; the job still ends with a status.
    assert _roles(model) == [P, B, M, E, E, M, E, F]
    assert outcome.calls == 8
    assert outcome.status is OutcomeStatus.ROUNDS_EXHAUSTED
    assert "the finalizer ran and the verifier still fails" in outcome.detail
    log = ledger.log_path.read_text()
    assert "execute call failed (ConnectionError" in log
    assert "retrying once" in log and "empty answer" in log
    # Empty worker answer: notes kept, and the verifier still ran after it.
    assert "worker gave no '## Notes' section; notes kept" in log
    assert log.count("verifier exit 1") == 3


@pytest.mark.asyncio
async def test_s5_5_finalizer_call_raises_verifier_still_decides(workspace) -> None:
    cwd, target, bundle, ledger = workspace
    model = ScriptedModel(
        _two_round_script("## Result\nnothing"),
        failures={Role.FINALIZE: 2},
    )
    outcome, _ = await _run(_exhausted_job(target, cwd), bundle, ledger, model, cwd)
    assert _roles(model) == [P, B, M, E, M, E, F, F]
    assert outcome.status is OutcomeStatus.ROUNDS_EXHAUSTED
    assert "the finalizer ran and the verifier still fails" in outcome.detail
    assert outcome.verifier is not None and not outcome.verifier.passed
    assert "finalize call failed (ConnectionError" in ledger.log_path.read_text()
