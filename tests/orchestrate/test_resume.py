"""The ledger is the run's only state: a stopped run continues from its last
clean boundary, an earlier boundary can be returned to, and a word from the
user reaches every later call.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from privibe.core.config import OrchestrateConfig
from privibe.core.orchestrate import (
    Bundle,
    JobSpec,
    Ledger,
    LedgerStateError,
    Outcome,
    OutcomeStatus,
    Progress,
    run_orchestration,
)
from privibe.core.orchestrate.ledger import (
    PHASE_BRAINSTORMED,
    PHASE_PLANNED,
    PHASE_ROUND_END,
    write_atomic,
)
from privibe.core.orchestrate.prompt import HEAD_TAIL_SEPARATOR, Role
from tests.orchestrate.test_loop import BRAINSTORM, PLAN, _check_cmd

CANCEL = object()


class Model:
    """Scripted answers per role. An EXECUTE answer may be paired with the file
    content the worker writes; the CANCEL marker raises CancelledError from
    inside the call, which is what a user's ctrl+c looks like to the loop.
    """

    def __init__(self, answers: dict[Role, list[object]], target: Path) -> None:
        self.answers = {k: list(v) for k, v in answers.items()}
        self.target = target
        self.calls: list[tuple[Role, str]] = []

    async def __call__(
        self, role: Role, message: str, effort: str, answer_only: bool
    ) -> str:
        self.calls.append((role, message))
        queue = self.answers.get(role, [])
        assert queue, f"no scripted answer left for {role}"
        answer = queue.pop(0)
        if isinstance(answer, tuple):
            text, content = answer
            self.target.write_text(content)
            answer = text
        if answer is CANCEL:
            raise asyncio.CancelledError
        assert isinstance(answer, str)
        return answer

    def roles(self) -> list[Role]:
        return [r for r, _ in self.calls]

    def heads(self) -> list[str]:
        return [m.partition(HEAD_TAIL_SEPARATOR)[0] for _, m in self.calls]

    def tails(self) -> list[str]:
        return [m.partition(HEAD_TAIL_SEPARATOR)[2] for _, m in self.calls]


WORKER_OK = "edited\n## Notes\nf returns 2\n## Next steps\nnone"
# Intermediate values are multi-digit on purpose: a same-size edit within the
# same second leaves Python's .pyc for the verifier's import looking fresh.
RETURN = "def f():\n    return {}\n"


@pytest.fixture
def ws(tmp_path: Path) -> tuple[Path, Path, Ledger, JobSpec]:
    target = tmp_path / "t.py"
    target.write_text(RETURN.format(1))
    ledger = Ledger(tmp_path / "ledger", plan_chars=4000, notes_chars=8000)
    job = JobSpec(
        goal="make f return 2",
        files=[str(target)],
        scope=[str(tmp_path)],
        done_command=_check_cmd(target),
        max_rounds=3,
    )
    return tmp_path, target, ledger, job


def _bundle(ledger: Ledger, job: JobSpec) -> Bundle:
    """What the tool does on resume: the bundle from snapshot.json, or a fresh
    load when there is none yet.
    """
    saved = ledger.read_snapshot()
    if saved is None:
        return Bundle.load(job.files, max_bytes=100_000)
    paths, contents = saved
    return Bundle(paths=paths, contents=contents)


async def _drive(ws, model: Model, **kw) -> list[Progress | Outcome]:
    cwd, _target, ledger, job = ws
    events: list[Progress | Outcome] = []
    async for ev in run_orchestration(
        job,
        bundle=_bundle(ledger, job),
        ledger=ledger,
        call=model,
        config=OrchestrateConfig(),
        cwd=cwd,
        **kw,
    ):
        events.append(ev)
    return events


def _outcome(events) -> Outcome:
    last = events[-1]
    assert isinstance(last, Outcome)
    return last


def _messages(events) -> list[str]:
    return [e.message for e in events if isinstance(e, Progress)]


@pytest.mark.asyncio
async def test_fresh_run_writes_job_snapshot_and_checkpoints(ws) -> None:
    cwd, target, ledger, job = ws
    model = Model(
        {
            Role.PLAN: [PLAN],
            Role.BRAINSTORM: [BRAINSTORM],
            Role.MANAGE: ["1. [todo] change return value\nNEXT: 1"],
            Role.EXECUTE: [(WORKER_OK, RETURN.format(2))],
        },
        target,
    )
    outcome = _outcome(await _drive(ws, model))
    assert outcome.status is OutcomeStatus.DONE

    record = ledger.read_job()
    assert record is not None
    assert record["goal"] == "make f return 2"
    assert record["max_rounds"] == 3  # the effective cap, resolved
    assert record["cwd"] == str(cwd)
    assert record["files"] == [str(target)]
    paths, contents = ledger.read_snapshot() or ([], {})
    assert paths == [str(target)]
    assert contents[str(target)] == RETURN.format(1)  # as it was at the start

    phases = [(c.number, c.phase, c.rounds) for c in ledger.checkpoints()]
    # The verifier passed in round 1, so the round never reached its end
    # boundary: the last checkpoint is the brainstorm.
    assert phases == [(1, PHASE_PLANNED, 0), (2, PHASE_BRAINSTORMED, 0)]
    assert outcome.checkpoint == 2
    assert not ledger.inflight_path.exists()


@pytest.mark.asyncio
async def test_round_end_checkpoint_holds_files_notes_verifier_and_keys(ws) -> None:
    cwd, target, ledger, job = ws
    model = Model(
        {
            Role.PLAN: [PLAN],
            Role.BRAINSTORM: [BRAINSTORM],
            Role.MANAGE: [
                "1. [todo] change return value\nNEXT: 1",
                "1. [todo] change return value\n2. [todo] check\nNEXT: 2",
            ],
            Role.EXECUTE: [
                ("edited\n## Notes\nnot yet\n## Next steps\nagain", RETURN.format(30)),
                (WORKER_OK, RETURN.format(2)),
            ],
        },
        target,
    )
    outcome = _outcome(await _drive(ws, model))
    assert outcome.status is OutcomeStatus.DONE
    assert outcome.rounds == 2

    cps = ledger.checkpoints()
    assert [c.phase for c in cps] == [
        PHASE_PLANNED,
        PHASE_BRAINSTORMED,
        PHASE_ROUND_END,
    ]
    end = cps[-1]
    assert end.rounds == 1
    assert end.files[str(target)] == RETURN.format(30)  # the tree after round 1
    assert end.notes == "not yet"
    assert end.verifier is not None and end.verifier["exit_code"] == 1
    assert end.last_task_key == "change return value"
    assert end.done_keys == ["change return value"]
    assert [t.text for t in end.tasks] == ["change return value"]
    assert end.calls == 4  # plan, brainstorm, manage, execute


@pytest.mark.asyncio
async def test_resume_skips_plan_and_brainstorm_and_keeps_the_head(ws) -> None:
    cwd, target, ledger, job = ws
    first = Model(
        {Role.PLAN: [PLAN], Role.BRAINSTORM: [BRAINSTORM], Role.MANAGE: [CANCEL]},
        target,
    )
    with pytest.raises(asyncio.CancelledError):
        await _drive(ws, first)
    assert ledger.latest_checkpoint() is not None
    assert ledger.latest_checkpoint().phase == PHASE_BRAINSTORMED  # type: ignore[union-attr]

    second = Model(
        {
            Role.MANAGE: ["1. [todo] change return value\nNEXT: 1"],
            Role.EXECUTE: [(WORKER_OK, RETURN.format(2))],
        },
        target,
    )
    events = await _drive(ws, second)
    outcome = _outcome(events)
    assert outcome.status is OutcomeStatus.DONE
    assert second.roles() == [Role.MANAGE, Role.EXECUTE]
    assert _messages(events)[0] == "resumed from checkpoint 2 (brainstormed, round 0/3)"
    # The frozen head is rebuilt from job.json and snapshot.json: same bytes.
    assert second.heads()[0] == first.heads()[0]
    # Counters continue: the manage call is call 3, as it would have been.
    assert outcome.calls == 4
    assert (ledger.root / "calls" / "003-manage.md").is_file()
    log = ledger.log_path.read_text()
    assert "resume from checkpoint 2 (brainstormed)" in log


@pytest.mark.asyncio
async def test_cancel_mid_worker_keeps_its_edits_and_no_stall_on_reissue(ws) -> None:
    cwd, target, ledger, job = ws
    first = Model(
        {
            Role.PLAN: [PLAN],
            Role.BRAINSTORM: [BRAINSTORM],
            Role.MANAGE: ["1. [todo] change return value\nNEXT: 1"],
        },
        target,
    )

    # The worker had edited the file when the cancel landed.
    async def _cancel_in_worker(role, message, effort, answer_only):
        if role is Role.EXECUTE:
            target.write_text(RETURN.format(50))
            raise asyncio.CancelledError
        return await Model.__call__(first, role, message, effort, answer_only)

    with pytest.raises(asyncio.CancelledError):
        await _drive(ws, _cancel_in_worker)  # type: ignore[arg-type]
    inflight = ledger.read_inflight()
    assert inflight == {"round": 1, "id": 1, "text": "change return value"}
    assert target.read_text() == RETURN.format(50)

    second = Model(
        {
            # The manager reissues the very task that was in flight.
            Role.MANAGE: ["1. [todo] change return value\nNEXT: 1"],
            Role.EXECUTE: [(WORKER_OK, RETURN.format(2))],
        },
        target,
    )
    events = await _drive(ws, second)
    outcome = _outcome(events)
    assert outcome.status is OutcomeStatus.DONE, outcome.detail
    assert outcome.rounds == 1  # the unfinished round is run again, not counted
    assert ledger.read_inflight() is None
    # The manager saw the half-done edit as a changed file, not as a snapshot.
    tail = second.tails()[0]
    assert "# Changed files" in tail and "return 50" in tail
    log = ledger.log_path.read_text()
    assert "a worker was running task 1 in round 1 when the run stopped" in log


@pytest.mark.asyncio
async def test_go_back_to_an_earlier_checkpoint_restores_files_and_drops_later(
    ws,
) -> None:
    cwd, target, ledger, job = ws
    model = Model(
        {
            Role.PLAN: [PLAN],
            Role.BRAINSTORM: [BRAINSTORM],
            Role.MANAGE: [
                "1. [todo] a\nNEXT: 1",
                "1. [done] a\n2. [todo] b\nNEXT: 2",
                "1. [done] a\n2. [done] b\n3. [todo] c\nNEXT: 3",
            ],
            Role.EXECUTE: [
                ("r1\n## Notes\nafter one\n## Next steps\n-", RETURN.format(30)),
                ("r2\n## Notes\nafter two\n## Next steps\n-", RETURN.format(400)),
                (WORKER_OK, RETURN.format(2)),
            ],
        },
        target,
    )
    outcome = _outcome(await _drive(ws, model))
    assert outcome.status is OutcomeStatus.DONE and outcome.rounds == 3
    assert [c.number for c in ledger.checkpoints()] == [1, 2, 3, 4]
    assert target.read_text() == RETURN.format(2)

    back = Model(
        {
            Role.MANAGE: ["1. [done] a\n2. [todo] b2\nNEXT: 2"],
            Role.EXECUTE: [(WORKER_OK, RETURN.format(2))],
        },
        target,
    )
    cwd, _t, ledger, job = ws
    events = []
    async for ev in run_orchestration(
        job,
        bundle=_bundle(ledger, job),
        ledger=ledger,
        call=back,
        config=OrchestrateConfig(),
        cwd=cwd,
        checkpoint=3,
    ):
        events.append(ev)
        if _messages(events) and len(events) == 1:
            # Right after the resume line the tree is back at round 1's end.
            assert target.read_text() == RETURN.format(30)
    outcome = _outcome(events)
    assert _messages(events)[0] == "resumed from checkpoint 3 (round_end, round 1/3)"
    assert outcome.status is OutcomeStatus.DONE
    assert outcome.rounds == 2  # continued as round 2
    assert ledger.read_notes() != "after two"  # notes went back too
    assert [c.number for c in ledger.checkpoints()] == [1, 2, 3]
    log = ledger.log_path.read_text()
    assert "dropped checkpoints [4]" in log and "restored 1 listed file" in log
    # The manager's tail on resume shows round 1's verifier, not round 2's.
    assert "# Last verifier run" in back.tails()[0]


@pytest.mark.asyncio
async def test_direction_reaches_every_later_call_and_never_the_head(ws) -> None:
    cwd, target, ledger, job = ws
    first = Model(
        {Role.PLAN: [PLAN], Role.BRAINSTORM: [BRAINSTORM], Role.MANAGE: [CANCEL]},
        target,
    )
    with pytest.raises(asyncio.CancelledError):
        await _drive(ws, first)
    assert "# Direction from the user" not in first.tails()[0]

    second = Model(
        {
            Role.MANAGE: ["1. [todo] change return value\nNEXT: 1"],
            Role.EXECUTE: [(WORKER_OK, RETURN.format(2))],
        },
        target,
    )
    await _drive(ws, second, direction="Use tabs, not spaces.")
    for tail in second.tails():
        assert "# Direction from the user" in tail
        assert "Use tabs, not spaces." in tail
        # Just before the role text, after the state the harness rewrites.
        assert tail.index("# Direction from the user") > tail.index(
            "# Last verifier run"
        )
        assert tail.index("# Direction from the user") < tail.index("# Your role")
    assert second.heads()[0] == first.heads()[0]
    assert "Use tabs, not spaces." in ledger.read_steer()

    # A later resume with a second direction keeps the first.
    third = Model({Role.MANAGE: ["1. [done] x\nNEXT: DONE"]}, target)
    await _drive(ws, third, direction="And keep the docstrings.")
    steer = ledger.read_steer()
    assert steer.index("Use tabs") < steer.index("keep the docstrings")
    assert "Use tabs" in third.tails()[0] and "keep the docstrings" in third.tails()[0]


@pytest.mark.asyncio
async def test_checkpoint_on_a_fresh_ledger_is_refused(ws) -> None:
    cwd, target, ledger, job = ws
    model = Model({}, target)
    with pytest.raises(LedgerStateError, match="no checkpoint to resume from"):
        await _drive(ws, model, checkpoint=1)
    assert model.calls == []
    with pytest.raises(LedgerStateError, match="no checkpoint 7"):
        ledger.checkpoint(7)


def test_write_atomic_replaces_whole_and_leaves_no_temp(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    write_atomic(path, "one\n")
    write_atomic(path, "two\n")
    assert path.read_text() == "two\n"
    assert [p.name for p in tmp_path.iterdir()] == ["tasks.json"]
