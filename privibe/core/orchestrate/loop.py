"""plan -> brainstorm -> [manage -> execute -> verify] x N, in plain Python.

The model is reached only through `RoleCall`, one awaitable per role call, so
the loop is testable with a scripted caller and the orchestrate tool supplies
the real one (a fresh AgentLoop per call).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from privibe.core.config import OrchestrateConfig
from privibe.core.orchestrate.ledger import Ledger, TaskItem
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
_PARSE_RETRIES = 1


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


@dataclass(frozen=True)
class Progress:
    message: str


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
    last_task_text: str | None = None
    calls: int = 0
    rounds: int = 0

    @property
    def max_rounds(self) -> int:
        return self.job.max_rounds or self.config.max_rounds

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
                last_report=self.last_report if role == Role.MANAGE else "",
                report_chars=_WORKER_REPORT_CHARS,
                role_text=role_text(role, task=task) + extra,
            ),
        )
        answer = await self.call(role, message, effort, answer_only)
        self.ledger.save_transcript(f"{self.calls:03d}-{role.value}", answer)
        return answer

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
        )

    async def plan(self) -> PlanAnswer | None:
        error = ""
        for _attempt in range(1 + _PARSE_RETRIES):
            text = await self.ask(
                Role.PLAN, self.config.effort_plan, answer_only=True, extra=error
            )
            try:
                return parse_plan(text)
            except ParseError as e:
                error = _retry_note(str(e), text)
                self.ledger.log(f"plan parse error: {e}")
        return None

    async def manage(self) -> ManageAnswer | None:
        error = ""
        for _attempt in range(1 + _PARSE_RETRIES):
            text = await self.ask(
                Role.MANAGE, self.config.effort_manage, answer_only=True, extra=error
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


async def run_orchestration(
    job: JobSpec,
    *,
    bundle: Bundle,
    ledger: Ledger,
    call: RoleCall,
    config: OrchestrateConfig,
    cwd: Path,
) -> AsyncGenerator[Progress | Outcome, None]:
    run = _Run(job=job, bundle=bundle, ledger=ledger, call=call, config=config, cwd=cwd)
    ledger.write_task(job.goal)
    run.head = build_head(
        goal=job.goal,
        scope=job.scope,
        done_command=job.done_command,
        rendered_files=bundle.render_frozen(),
    )

    # 1. plan
    yield Progress("manager: planning")
    plan_answer = await run.plan()
    if plan_answer is None:
        yield run.outcome(OutcomeStatus.STALLED, "manager's plan could not be parsed")
        return
    ledger.write_plan(plan_answer.plan)
    run.tasks = plan_answer.tasks
    ledger.write_tasks(run.tasks)
    ledger.log(f"plan written, {len(run.tasks)} seed tasks")
    yield Progress(f"plan: {len(run.tasks)} tasks")
    for item in run.tasks:
        yield Progress(f"  {item.render()}")

    # 2. brainstorm
    yield Progress("brainstorm")
    notes = await run.ask(Role.BRAINSTORM, config.effort_brainstorm, answer_only=True)
    ledger.write_notes(notes)
    ledger.log("brainstorm written to notes")

    # 3. manage -> 4. execute -> 5. verify, up to max_rounds times
    async for event in _rounds(run):
        yield event


async def _rounds(run: _Run) -> AsyncGenerator[Progress | Outcome, None]:
    job, ledger, config = run.job, run.ledger, run.config
    while run.rounds < run.max_rounds:
        run.rounds += 1
        tag = f"round {run.rounds}/{run.max_rounds}"
        yield Progress(f"{tag}: manager")
        decision = await run.manage()
        if decision is None:
            yield run.outcome(
                OutcomeStatus.STALLED, "manager's task list could not be parsed"
            )
            return
        run.tasks = decision.tasks
        ledger.write_tasks(run.tasks)

        if decision.done:
            if not job.done_command:
                yield run.outcome(
                    OutcomeStatus.DONE_UNVERIFIED, "manager declared done; no verifier"
                )
                return
            if run.verifier is None:
                yield Progress(f"{tag}: verifier (manager declared done)")
                await run.verify()
            if run.verifier is not None and run.verifier.passed:
                yield run.outcome(OutcomeStatus.DONE)
                return
            # A failing run vetoes DONE: the manager sees it next round.
            ledger.log("manager said DONE but the verifier fails; continuing")
            run.last_report = ""
            continue

        task = decision.next_task()
        if task is None:  # parse_manage guarantees otherwise; belt and braces
            yield run.outcome(OutcomeStatus.STALLED, "manager named no task")
            return
        if task.text == run.last_task_text:
            yield run.outcome(
                OutcomeStatus.STALLED, f"manager reissued the same task: {task.text}"
            )
            return
        run.last_task_text = task.text

        yield Progress(f"{tag}: worker on task {task.id}: {task.text}")
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
            yield Progress(f"{tag}: verifier")
            await run.verify()

    yield run.outcome(OutcomeStatus.ROUNDS_EXHAUSTED, f"{run.max_rounds} rounds used")


def _retry_note(error: str, previous: str) -> str:
    return (
        "\n\nYour previous answer could not be parsed: "
        f"{error}. Answer again in exactly the shape above.\n"
        "Previous answer:\n"
        f"{previous.strip()[:2000]}"
    )
