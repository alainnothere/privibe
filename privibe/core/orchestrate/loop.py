"""plan -> brainstorm -> [manage -> execute -> verify] x N, in plain Python.

The model is reached only through `RoleCall`, one awaitable per role call, so
the loop is testable with a scripted caller and the orchestrate tool supplies
the real one (a fresh AgentLoop per call).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
import re

from privibe.core.config import OrchestrateConfig
from privibe.core.orchestrate.ledger import (
    PHASE_BRAINSTORMED,
    PHASE_PLANNED,
    PHASE_ROUND_END,
    Checkpoint,
    Ledger,
    LedgerStateError,
    TaskItem,
)
from privibe.core.orchestrate.parse import (
    ManageAnswer,
    ParseError,
    PlanAnswer,
    parse_manage,
    parse_plan,
    parse_worker_notes,
)
from privibe.core.orchestrate.prompt import (
    Role,
    build_head,
    build_message,
    build_tail,
    role_text,
)
from privibe.core.orchestrate.snapshot import Bundle
from privibe.core.orchestrate.verifier import VerifierResult, run_verifier

# One model call: (role, message, effort, answer_only) -> the answer text.
# `answer_only` means every tool call in the turn is refused.
type RoleCall = Callable[[Role, str, str, bool], Awaitable[str]]

_WORKER_REPORT_CHARS = 4000
_WHOLE_JOB_TASK = "Do the whole task described above, every edit it needs."
_PARSE_RETRIES = 1
_CALL_RETRIES = 1


@dataclass(frozen=True)
class JobSpec:
    goal: str
    files: list[str]
    scope: list[str]
    done_command: str = ""
    max_rounds: int | None = None


class OutcomeStatus(StrEnum):
    DONE = "done"
    DONE_UNVERIFIED = "done_unverified"
    ROUNDS_EXHAUSTED = "rounds_exhausted"
    STALLED = "stalled"
    CANCELLED = "cancelled"


@dataclass
class Outcome:
    status: OutcomeStatus
    rounds: int
    calls: int
    verifier: VerifierResult | None
    changed_files: list[str]
    ledger_dir: Path
    detail: str = ""
    # The last clean boundary the ledger holds; what a later run can resume from.
    checkpoint: int | None = None


def job_record(job: JobSpec, *, max_rounds: int, cwd: Path) -> dict[str, object]:
    """The job as job.json stores it: everything a resume needs that the caller
    would otherwise have to repeat, with the round cap already resolved.
    """
    return {
        "goal": job.goal,
        "files": list(job.files),
        "scope": list(job.scope),
        "done_command": job.done_command,
        "max_rounds": max_rounds,
        "cwd": str(cwd),
    }


def job_from_record(raw: dict[str, object]) -> JobSpec:
    return JobSpec(
        goal=str(raw["goal"]),
        files=[str(f) for f in raw.get("files", [])],  # type: ignore[union-attr]
        scope=[str(s) for s in raw.get("scope", [])],  # type: ignore[union-attr]
        done_command=str(raw.get("done_command", "")),
        max_rounds=int(raw["max_rounds"]),  # type: ignore[call-overload]
    )


@dataclass(frozen=True)
class Progress:
    message: str
    # A role call is starting: the tool opens a section in its trace.
    section: bool = False


@dataclass
class _Run:
    job: JobSpec
    bundle: Bundle
    ledger: Ledger
    call: RoleCall
    config: OrchestrateConfig
    cwd: Path
    head: str = ""
    tasks: list[TaskItem] = field(default_factory=list)
    verifier: VerifierResult | None = None
    last_report: str = ""
    last_task_key: str | None = None
    # Keys of every task a worker has run, kept by the harness because a manager
    # that answers badly can reset the list's [done] marks.
    done_keys: set[str] = field(default_factory=set)
    calls: int = 0
    rounds: int = 0
    last_checkpoint: int | None = None

    @property
    def max_rounds(self) -> int:
        return self.job.max_rounds or self.config.max_rounds

    def checkpoint(self, phase: str) -> Checkpoint:
        """Write the run as it stands to state/NNN.json: a clean boundary a
        later run can continue from. The listed files go in as they are now,
        so an earlier boundary can be returned to, files included.
        """
        cp = Checkpoint(
            number=self.ledger.next_checkpoint_number(),
            phase=phase,
            rounds=self.rounds,
            calls=self.calls,
            verifier=asdict(self.verifier) if self.verifier is not None else None,
            last_report=self.last_report,
            last_task_key=self.last_task_key,
            done_keys=sorted(self.done_keys),
            tasks=list(self.tasks),
            notes=self.ledger.read_notes(),
            files=self.bundle.current_contents(),
        )
        self.ledger.write_checkpoint(cp)
        self.last_checkpoint = cp.number
        self.ledger.log(f"checkpoint {cp.number} ({phase}, round {cp.rounds})")
        return cp

    def restore(self, cp: Checkpoint) -> None:
        """Take the run back to `cp`: counters, stall keys, verifier, report,
        task list and notes. The notes and the task list are written back to
        the ledger too, so the files the roles read match the checkpoint.
        """
        self.rounds = cp.rounds
        self.calls = cp.calls
        self.verifier = VerifierResult(**cp.verifier) if cp.verifier else None
        self.last_report = cp.last_report
        self.last_task_key = cp.last_task_key
        self.done_keys = set(cp.done_keys)
        self.tasks = list(cp.tasks)
        self.last_checkpoint = cp.number
        self.ledger.write_tasks(self.tasks)
        self.ledger.write_notes(cp.notes)

    def plan_budget_text(self) -> str:
        # One task per round, and the last round must still run the verifier.
        return (
            f"\n\nBudget: the harness runs {self.max_rounds} rounds, one worker "
            f"task per round. Plan at most {max(1, self.max_rounds - 1)} tasks, so the "
            "last round is left for the check. Merge small steps into one task.\n"
        )

    def rounds_left_text(self) -> str:
        left = self.max_rounds - self.rounds
        return (
            f"\n\nBudget: round {self.rounds} of {self.max_rounds}, {left} left "
            "after this one. Say DONE only when the work is finished; a DONE "
            "that the verifier rejects still spends a round.\n"
        )

    async def ask(
        self,
        role: Role,
        effort: str,
        *,
        answer_only: bool,
        task: str = "",
        extra: str = "",
    ) -> str:
        self.calls += 1
        message = build_message(
            self.head,
            build_tail(
                plan=self.ledger.read_plan(),
                notes=self.ledger.read_notes(),
                tasks=self.tasks,
                rendered_changed=self.bundle.render_changed(),
                verifier=self.verifier,
                last_report=(
                    self.last_report if role in {Role.MANAGE, Role.FINALIZE} else ""
                ),
                report_chars=_WORKER_REPORT_CHARS,
                role_text=role_text(role, task=task) + extra,
                direction=self.ledger.read_steer(),
            ),
        )
        answer = await self._call_with_retry(role, message, effort, answer_only)
        self.ledger.save_transcript(f"{self.calls:03d}-{role.value}", answer)
        return answer

    async def _call_with_retry(
        self, role: Role, message: str, effort: str, answer_only: bool
    ) -> str:
        """One model call that never raises. A dropped connection or a backend
        error is retried once; if that fails too the answer is empty, and the
        loop's own fallbacks (parse retry, fallback task, finalizer) take over.
        Cancellation is not an error and passes through.
        """
        for attempt in range(1 + _CALL_RETRIES):
            try:
                return await self.call(role, message, effort, answer_only)
            except Exception as e:  # the backend can fail in any way
                self.ledger.log(
                    f"{role.value} call failed ({type(e).__name__}: {e})"
                    + (
                        "; retrying once"
                        if attempt < _CALL_RETRIES
                        else "; empty answer"
                    )
                )
                if attempt < _CALL_RETRIES:
                    self.calls += 1
        return ""

    def outcome(self, status: OutcomeStatus, detail: str = "") -> Outcome:
        self.ledger.log(f"end {status.value}" + (f": {detail}" if detail else ""))
        return Outcome(
            status=status,
            rounds=self.rounds,
            calls=self.calls,
            verifier=self.verifier,
            changed_files=self.bundle.changed_files(),
            ledger_dir=self.ledger.root,
            detail=detail,
            checkpoint=self.last_checkpoint,
        )

    async def plan(self) -> PlanAnswer:
        """The manager's plan. Never fails: when the answer cannot be parsed
        even after the retry, the job runs as one task and the manager's own
        text (or the goal) becomes the plan.
        """
        error = ""
        text = ""
        for _attempt in range(1 + _PARSE_RETRIES):
            text = await self.ask(
                Role.PLAN,
                self.config.effort_plan,
                answer_only=True,
                extra=self.plan_budget_text() + error,
            )
            try:
                return parse_plan(text)
            except ParseError as e:
                error = _retry_note(str(e), text)
                self.ledger.log(f"plan parse error: {e}")
        self.ledger.log("plan unparseable; falling back to one task for the whole job")
        return PlanAnswer(
            plan=text.strip() or self.job.goal.strip(),
            tasks=[TaskItem(id=1, text=_WHOLE_JOB_TASK)],
        )

    async def manage(self) -> ManageAnswer | None:
        error = ""
        for _attempt in range(1 + _PARSE_RETRIES):
            text = await self.ask(
                Role.MANAGE,
                self.config.effort_manage,
                answer_only=True,
                extra=self.rounds_left_text() + error,
            )
            try:
                return parse_manage(text)
            except ParseError as e:
                error = _retry_note(str(e), text)
                self.ledger.log(f"manage parse error: {e}")
        return None

    async def verify(self) -> VerifierResult:
        result = await run_verifier(
            self.job.done_command,
            cwd=self.cwd,
            timeout_s=self.config.verifier_timeout_s,
            output_chars=self.config.verifier_output_chars,
        )
        self.verifier = result
        suffix = " (timeout)" if result.timed_out else ""
        self.ledger.log(f"verifier exit {result.exit_code}{suffix}")
        return result


_PHASE_ORDER = {PHASE_PLANNED: 1, PHASE_BRAINSTORMED: 2, PHASE_ROUND_END: 3}


async def run_orchestration(
    job: JobSpec,
    *,
    bundle: Bundle,
    ledger: Ledger,
    call: RoleCall,
    config: OrchestrateConfig,
    cwd: Path,
    checkpoint: int | None = None,
    direction: str = "",
) -> AsyncGenerator[Progress | Outcome, None]:
    """Run the job, or continue it. The ledger says which: with no checkpoint
    on disk this is a fresh start and job.json plus snapshot.json are written;
    with checkpoints the run is restored from the latest one, or from
    `checkpoint` when given, and every phase that boundary already covers is
    skipped. `direction` is a word from the user, kept in the ledger and shown
    to every later role call.
    """
    run = _Run(job=job, bundle=bundle, ledger=ledger, call=call, config=config, cwd=cwd)
    latest = ledger.latest_checkpoint()
    if latest is None:
        if checkpoint is not None:
            raise LedgerStateError(f"no checkpoint to resume from in {ledger.root}")
        ledger.write_job(job_record(job, max_rounds=run.max_rounds, cwd=cwd))
        ledger.write_snapshot(bundle.paths, bundle.contents)
        ledger.write_task(job.goal)
        phase_done = 0
    else:
        target = latest if checkpoint is None else ledger.checkpoint(checkpoint)
        for line in _resume(run, target, latest):
            ledger.log(line)
        phase_done = _PHASE_ORDER[target.phase]
        yield Progress(
            f"resumed from checkpoint {target.number} ({target.phase}, "
            f"round {target.rounds}/{run.max_rounds})"
        )
    if direction.strip():
        ledger.append_steer(direction)
        ledger.log("direction from the user added to steer.md")
    run.head = build_head(
        goal=job.goal,
        scope=job.scope,
        done_command=job.done_command,
        rendered_files=bundle.render_frozen(),
    )

    # 1. plan
    if phase_done < _PHASE_ORDER[PHASE_PLANNED]:
        yield Progress("manager · plan", section=True)
        plan_answer = await run.plan()
        ledger.write_plan(plan_answer.plan)
        run.tasks = plan_answer.tasks
        ledger.write_tasks(run.tasks)
        ledger.log(f"plan written, {len(run.tasks)} seed tasks")
        run.checkpoint(PHASE_PLANNED)

    # 2. brainstorm
    if phase_done < _PHASE_ORDER[PHASE_BRAINSTORMED]:
        yield Progress("brainstorm", section=True)
        notes = await run.ask(
            Role.BRAINSTORM, config.effort_brainstorm, answer_only=True
        )
        ledger.write_notes(notes)
        ledger.log("brainstorm written to notes")
        run.checkpoint(PHASE_BRAINSTORMED)

    # 3. manage -> 4. execute -> 5. verify, up to max_rounds times
    async for event in _rounds(run):
        yield event


def _resume(run: _Run, target: Checkpoint, latest: Checkpoint) -> list[str]:
    """Put the run back at `target`. Continuing from the latest boundary keeps
    the tree as it is, a worker's unfinished edits included, since they are
    progress the next manager call will see as changed files. Going back to an
    earlier boundary puts the listed files back as they were there and drops
    the later checkpoints, the way a rewind does. Returns the log lines.
    """
    lines = [f"resume from checkpoint {target.number} ({target.phase})"]
    run.restore(target)
    inflight = run.ledger.read_inflight()
    if inflight is not None:
        lines.append(
            f"a worker was running task {inflight['id']} in round "
            f"{inflight['round']} when the run stopped; its edits stay in the tree"
        )
        run.ledger.clear_inflight()
    if target.number != latest.number:
        restored = run.bundle.restore(target.files)
        dropped = run.ledger.drop_checkpoints_after(target.number)
        lines.append(
            f"went back to checkpoint {target.number}: restored "
            f"{len(restored)} listed file(s), dropped checkpoints {dropped}; "
            f"transcripts after call {target.calls} will be overwritten"
        )
    return lines


async def _rounds(run: _Run) -> AsyncGenerator[Progress | Outcome, None]:
    job, ledger, config = run.job, run.ledger, run.config
    while run.rounds < run.max_rounds:
        run.rounds += 1
        tag = f"round {run.rounds}/{run.max_rounds}"
        yield Progress(f"manager · {tag}", section=True)
        decision = await run.manage() or _fallback_decision(run)
        if decision is None:
            async for event in _finalize(
                run,
                OutcomeStatus.STALLED,
                "manager's task list could not be parsed and no task is left",
            ):
                yield event
            return
        run.tasks = decision.tasks
        ledger.write_tasks(run.tasks)

        task = decision.next_task()
        if decision.done:
            events, resolved = await _resolve_done(run, tag)
            for event in events:
                yield event
            if isinstance(resolved, Outcome):
                yield resolved
                return
            task = resolved
        stall = _stall_reason(run, task, overruled=decision.done)
        if stall is not None:
            async for event in _finalize(run, OutcomeStatus.STALLED, stall):
                yield event
            return
        assert task is not None  # _stall_reason returns a reason for None
        run.last_task_key = _task_key(task.text)
        run.done_keys.add(run.last_task_key)
        ledger.write_inflight(round_no=run.rounds, task_id=task.id, text=task.text)

        yield Progress(f"worker · {tag} · task {task.id}: {task.text}", section=True)
        report = await run.ask(
            Role.EXECUTE, config.effort_execute, answer_only=False, task=task.render()
        )
        run.last_report = report
        new_notes = parse_worker_notes(report)
        if new_notes is not None:
            ledger.write_notes(new_notes)
        else:
            ledger.log("worker gave no '## Notes' section; notes kept")

        if job.done_command:
            yield Progress(f"verifier · {tag}")
            await run.verify()
            yield _verifier_progress(run.verifier)
            if run.verifier is not None and run.verifier.passed:
                # Ground truth: the job is done whatever the task list says.
                ledger.clear_inflight()
                yield run.outcome(OutcomeStatus.DONE, "verifier passes after a worker")
                return
        ledger.clear_inflight()
        run.checkpoint(PHASE_ROUND_END)

    async for event in _finalize(
        run, OutcomeStatus.ROUNDS_EXHAUSTED, f"{run.max_rounds} rounds used"
    ):
        yield event


_PUNCT_EDGES = " \t\r\n.,;:!?'\"`*_-"


def _task_key(text: str) -> str:
    """A task's text with case, spacing and edge punctuation ignored, so a small
    model rewording 'Fix x.' as 'fix  x' is still the same task.
    """
    return re.sub(r"\s+", " ", text).strip(_PUNCT_EDGES).lower()


async def _resolve_done(
    run: _Run, tag: str
) -> tuple[list[Progress], Outcome | TaskItem | None]:
    """The manager said DONE. Returns the progress lines to show, and either the
    Outcome that ends the job, the task to run instead because the DONE was
    overruled, or None when it was overruled and no task is left.
    """
    events: list[Progress] = []
    if run.job.done_command and run.verifier is None:
        events.append(Progress(f"verifier · {tag} (manager declared done)"))
        await run.verify()
        events.append(_verifier_progress(run.verifier))
    accepted = _done_outcome(run)
    if accepted is not None:
        return events, accepted
    return events, _overrule_done(run)


def _stall_reason(run: _Run, task: TaskItem | None, *, overruled: bool) -> str | None:
    """Why the loop cannot run `task`, or None when it can."""
    if task is None:
        if overruled:
            return "manager said DONE, nothing confirms it, and no other task is left"
        return "manager named no task"  # parse_manage rules this out; belt and braces
    if _task_key(task.text) == run.last_task_key:
        return f"manager reissued the same task: {task.text}"
    return None


def _done_outcome(run: _Run) -> Outcome | None:
    """The manager said DONE. An Outcome when that stands, None when it must be
    overruled. A verifier is ground truth: pass is done, fail vetoes. Without
    one the manager's word stands, except a DONE before any worker has run and
    with nothing changed, which is a lazy answer, not a finished job.
    """
    if run.job.done_command:
        if run.verifier is not None and run.verifier.passed:
            return run.outcome(OutcomeStatus.DONE)
        run.ledger.log("manager said DONE but the verifier fails; continuing")
        return None
    if run.last_task_key is None and not run.bundle.changed_files():
        run.ledger.log("manager said DONE before any work was done; overruled")
        return None
    return run.outcome(
        OutcomeStatus.DONE_UNVERIFIED, "manager declared done; no verifier"
    )


def _advance(run: _Run) -> tuple[list[TaskItem], TaskItem | None]:
    """Count the task the last worker ran as done and find the first one still
    open. The manager's own list is not trusted here: it was unparseable or it
    was overruled.
    """
    tasks = [
        TaskItem(
            id=t.id, text=t.text, done=t.done or _task_key(t.text) in run.done_keys
        )
        for t in run.tasks
    ]
    return tasks, next((t for t in tasks if not t.done), None)


def _overrule_done(run: _Run) -> TaskItem | None:
    """A DONE the verifier (or common sense) rejects: do the next open task
    instead of asking the manager again. None when nothing is open.
    """
    run.last_report = ""
    tasks, first_open = _advance(run)
    run.tasks = tasks
    run.ledger.write_tasks(tasks)
    if first_open is not None:
        run.ledger.log(f"DONE overruled; running the first open task {first_open.id}")
    return first_open


def _fallback_decision(run: _Run) -> ManageAnswer | None:
    """The manager's answer could not be parsed, even after the retry. Keep the
    job moving the way a person would: count the task the worker just ran as
    done and take the first one still open. None when nothing is open.
    """
    tasks, first_open = _advance(run)
    if first_open is None:
        return None
    run.ledger.log(f"manager unparseable; taking the first open task {first_open.id}")
    return ManageAnswer(tasks=tasks, next_id=first_open.id, done=False)


async def _finalize(
    run: _Run, status: OutcomeStatus, reason: str
) -> AsyncGenerator[Progress | Outcome, None]:
    """The loop is ending without a clean sign-off (rounds spent, a task
    reissued, no task left). One last worker finishes the job from the ledger,
    then the verifier decides. A pass is DONE whatever the reason; otherwise the
    original status stands, with what the finalizer did added to the detail.
    """
    if run.verifier is not None and run.verifier.passed:
        # The verifier runs after every worker call, so this is the current
        # tree: the job is done whatever the manager said, nothing to finish.
        run.ledger.log(f"loop ended ({reason}) but the verifier passes; done")
        yield run.outcome(OutcomeStatus.DONE, f"verifier passes: {reason}")
        return
    run.ledger.log(f"finalizer: {reason}")
    yield Progress("finalizer", section=True)
    report = await run.ask(
        Role.FINALIZE,
        run.config.effort_finalize,
        answer_only=False,
        extra=f"\n\nWhy the loop ended: {reason}.\n",
    )
    run.last_report = report
    if not run.job.done_command:
        yield run.outcome(status, f"{reason}; the finalizer ran, nothing verified it")
        return
    yield Progress("verifier · finalizer")
    await run.verify()
    yield _verifier_progress(run.verifier)
    if run.verifier is not None and run.verifier.passed:
        yield run.outcome(OutcomeStatus.DONE, f"finalizer completed the job: {reason}")
        return
    yield run.outcome(
        status, f"{reason}; the finalizer ran and the verifier still fails"
    )


def _verifier_progress(result: VerifierResult | None) -> Progress:
    if result is None:
        return Progress("verifier: not run")
    mark = "✓" if result.passed else "✕"
    head = f"{mark} verifier exit {result.exit_code}"
    if result.timed_out:
        head += " (timeout)"
    tail = result.output.rstrip().splitlines()[-3:]
    body = "\n".join("  " + line for line in tail)
    return Progress(head + ("\n" + body if body else ""))


def _retry_note(error: str, previous: str) -> str:
    return (
        "\n\nYour previous answer could not be parsed: "
        f"{error}. Answer again in exactly the shape above.\n"
        "Previous answer:\n"
        f"{previous.strip()[:2000]}"
    )
