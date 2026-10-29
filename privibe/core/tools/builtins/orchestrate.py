from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
import tempfile
from typing import ClassVar

from pydantic import BaseModel, Field

from privibe.core.agent_loop import AgentLoop
from privibe.core.agents.models import MUTATING_TOOLS, BuiltinAgentName
from privibe.core.config import SessionLoggingConfig, VibeConfig
from privibe.core.orchestrate import (
    Bundle,
    BundleTooLargeError,
    JobSpec,
    Ledger,
    Outcome,
    OutcomeStatus,
    Progress,
    run_orchestration,
)
from privibe.core.orchestrate.prompt import Role
from privibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from privibe.core.tools.builtins.bash import _extract_commands, _find_protected_token
from privibe.core.tools.permissions import PermissionContext
from privibe.core.tools.subagent_trace import Tracer
from privibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from privibe.core.tools.utils import is_path_within_workdir, is_protected_path
from privibe.core.types import Role as MessageRole, ToolStreamEvent
from privibe.core.utils.tags import VIBE_STOP_EVENT_TAG

# The file tools whose writes the job's scope limits. bash is left to its own
# permission config: its allowlist is command patterns, not paths.
_SCOPED_TOOLS = [t for t in MUTATING_TOOLS if t != "bash"]

# The call line shows this much of the goal; the body shows all of it.
_SUMMARY_GOAL_CHARS = 80


def summarize_goal(goal: str, limit: int = _SUMMARY_GOAL_CHARS) -> str:
    """The goal's first non-empty line, cut to `limit` characters."""
    first = next((ln.strip() for ln in goal.splitlines() if ln.strip()), "")
    if len(first) > limit:
        return first[: limit - 1].rstrip() + "…"
    return first


class OrchestrateArgs(BaseModel):
    goal: str = Field(
        description="What needs to be done, in full. The workers see nothing else about the request."
    )
    files: list[str] = Field(
        default_factory=list,
        description=(
            "Files to pre-feed to every call. Order: read-only context first, the "
            "files expected to be edited last."
        ),
    )
    scope: list[str] = Field(
        default_factory=list,
        description="Directories the workers may edit. Edits anywhere else are refused.",
    )
    done_command: str = Field(
        default="",
        description=(
            "Shell command that exits 0 when the job is done (a test run). Without it "
            "the job ends on the manager's word, unverified."
        ),
    )
    max_rounds: int | None = Field(
        default=None, ge=1, description="Manager/worker rounds; default from config."
    )


class OrchestrateResult(BaseModel):
    status: str = Field(
        description="done / done_unverified / rounds_exhausted / stalled / cancelled"
    )
    detail: str = ""
    rounds: int = 0
    calls: int = Field(default=0, description="Model calls made")
    verifier_exit_code: int | None = None
    verifier_output: str = ""
    changed_files: list[str] = Field(default_factory=list)
    ledger_dir: str = Field(
        description="Where plan.md, notes.md, tasks.json and the call transcripts are"
    )


class OrchestrateToolConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK


class Orchestrate(
    BaseTool[OrchestrateArgs, OrchestrateResult, OrchestrateToolConfig, BaseToolState],
    ToolUIData[OrchestrateArgs, OrchestrateResult],
):
    description: ClassVar[str] = (
        "Hand a coding job to a manager/worker loop of fresh model calls that "
        "coordinate through files on disk and stop when a verifier command passes."
    )

    @classmethod
    def format_call_display(cls, args: OrchestrateArgs) -> ToolCallDisplay:
        """The job on the call line itself, like every other tool prints its
        parameters: the goal's first line, then scope, files and what ends
        the job on indented lines. The fold-out body is for progress.
        """
        goal = summarize_goal(args.goal)
        lines = [f'orchestrate "{goal}"' if goal else "orchestrate"]
        lines.append("  scope: " + (", ".join(args.scope) if args.scope else "(none)"))
        lines.append("  files: " + (", ".join(args.files) if args.files else "(none)"))
        lines.append(
            "  done when: "
            + (
                f"`{args.done_command}` exits 0"
                if args.done_command
                else "the manager says so (unverified)"
            )
        )
        if args.max_rounds is not None:
            lines.append(f"  max_rounds: {args.max_rounds}")
        return ToolCallDisplay(summary="\n".join(lines))

    @classmethod
    def format_result_display(cls, result: OrchestrateResult) -> ToolResultDisplay:
        ok = result.status in {OutcomeStatus.DONE, OutcomeStatus.DONE_UNVERIFIED}
        message = f"orchestrate {result.status} after {result.rounds} rounds, {result.calls} calls"
        if result.detail:
            message += f": {result.detail}"
        return ToolResultDisplay(success=ok, message=message)

    @classmethod
    def get_status_text(cls) -> str:
        return "Orchestrating"

    def resolve_permission(self, args: OrchestrateArgs) -> PermissionContext | None:
        protected = self.config.protected_paths
        for p in [*args.files, *args.scope]:
            if protected and is_protected_path(p, protected):
                return PermissionContext(
                    permission=ToolPermission.NEVER,
                    reason=f"Access denied: {p} is protected by configuration.",
                )
            if (
                self.config.protect_outside_workdir
                and not is_path_within_workdir(p)
                and not is_protected_path(p, self.config.outside_workdir_exempt)
            ):
                return PermissionContext(
                    permission=ToolPermission.NEVER,
                    reason=f"Access denied: {p} is outside the working directory.",
                )
        if args.done_command and protected:
            hit = _find_protected_token(_extract_commands(args.done_command), protected)
            if hit:
                return PermissionContext(
                    permission=ToolPermission.NEVER,
                    reason=f"done_command denied: '{hit}' refers to a protected path.",
                )
        return None

    async def run(  # noqa: PLR0914, PLR0915
        self, args: OrchestrateArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | OrchestrateResult, None]:
        if not ctx or not ctx.agent_manager:
            raise ToolError("orchestrate requires agent_manager in context")
        parent_config = ctx.agent_manager.config
        settings = parent_config.orchestrate

        files = [str(Path(f).expanduser().resolve()) for f in args.files]
        scope = [str(Path(s).expanduser().resolve()) for s in args.scope]
        for f in files:
            if not Path(f).is_file():
                raise ToolError(f"not a file: {f}")
        for s in scope:
            if not Path(s).is_dir():
                raise ToolError(f"scope entry is not a directory: {s}")
        try:
            bundle = Bundle.load(files, max_bytes=settings.bundle_bytes)
        except BundleTooLargeError as e:
            raise ToolError(str(e)) from e
        except (UnicodeDecodeError, ValueError, OSError) as e:
            raise ToolError(f"cannot read a pre-fed file as text: {e}") from e

        ledger_root = (
            ctx.session_dir / "orchestrate" / ctx.tool_call_id
            if ctx.session_dir is not None
            else Path(tempfile.mkdtemp(prefix="privibe-orchestrate-"))
        )
        ledger = Ledger(
            ledger_root,
            plan_chars=settings.plan_chars,
            notes_chars=settings.notes_chars,
        )
        job = JobSpec(
            goal=args.goal,
            files=files,
            scope=scope,
            done_command=args.done_command,
            max_rounds=args.max_rounds,
        )
        worker_config = self._worker_config(ctx, parent_config, scope)
        queue: asyncio.Queue[ToolStreamEvent | Outcome] = asyncio.Queue()
        tracer = Tracer(
            self.get_name(), ctx.tool_call_id, parent_config.tool_result_preview_lines
        )
        transcripts = str(ledger.root / "calls")

        async def call(role: Role, message: str, effort: str, answer_only: bool) -> str:
            return await self._call_role(
                ctx,
                worker_config,
                queue,
                tracer,
                transcripts,
                message,
                effort,
                answer_only,
            )

        async def drive() -> None:
            # Progress is turned into trace entries here, on the producer
            # side, so a section is open before the role call it announces
            # starts feeding the tracer; the consumer only forwards.
            async for event in run_orchestration(
                job,
                bundle=bundle,
                ledger=ledger,
                call=call,
                config=settings,
                cwd=Path.cwd(),
            ):
                if isinstance(event, Progress):
                    entries = (
                        tracer.start_section(event.message)
                        if event.section
                        else tracer.line(event.message)
                    )
                    for entry in entries:
                        await queue.put(entry)
                else:
                    await queue.put(event)

        driver = asyncio.create_task(drive())
        try:
            while True:
                getter = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait(
                    {getter, driver}, return_when=asyncio.FIRST_COMPLETED
                )
                if getter in done:
                    item = getter.result()
                else:
                    getter.cancel()
                    # The driver ended without an Outcome: it raised.
                    driver.result()
                    raise ToolError("orchestration ended without a result")
                if isinstance(item, ToolStreamEvent):
                    yield item
                else:
                    await driver
                    yield _to_result(item)
                    return
        except asyncio.CancelledError:
            driver.cancel()
            ledger.log("end cancelled")
            yield OrchestrateResult(
                status=OutcomeStatus.CANCELLED,
                detail="cancelled by the user",
                changed_files=bundle.changed_files(),
                ledger_dir=str(ledger.root),
            )
            raise

    @staticmethod
    def _worker_config(
        ctx: InvokeContext, parent_config: VibeConfig, scope: list[str]
    ) -> VibeConfig:
        """The config every role call's AgentLoop is built from: the user's
        config file, plus the job's scope as per-tool allowlists, the worker
        LLM-call budget, and the parent's auto-approve state so a loop run from
        an auto-approve session does not prompt on every edit.
        """
        session_logging = SessionLoggingConfig(
            save_dir=str(ctx.session_dir / "agents") if ctx.session_dir else "",
            session_prefix=BuiltinAgentName.ORCHESTRATE_WORKER,
            enabled=ctx.session_dir is not None,
        )
        base = VibeConfig.load(session_logging=session_logging)
        globs = [str(Path(s) / "*") for s in scope]
        tools = dict(base.tools)
        for name in _SCOPED_TOOLS:
            tools[name] = {
                **tools.get(name, {}),
                "permission": "never",
                "allowlist": globs,
            }
        return base.model_copy(
            update={
                "tools": tools,
                "max_llm_calls_per_turn": parent_config.orchestrate.worker_llm_calls,
                "auto_approve": parent_config.auto_approve,
            }
        )

    async def _call_role(
        self,
        ctx: InvokeContext,
        worker_config: VibeConfig,
        queue: asyncio.Queue[ToolStreamEvent | Outcome],
        tracer: Tracer,
        transcripts: str,
        message: str,
        effort: str,
        answer_only: bool,
    ) -> str:
        """One role call: a fresh AgentLoop, one act(), the final answer text.
        Everything the loop does is traced into the parent's tool call body.
        """
        loop = AgentLoop(
            config=worker_config,
            agent_name=BuiltinAgentName.ORCHESTRATE_WORKER,
            entrypoint_metadata=ctx.entrypoint_metadata,
        )
        if ctx.approval_callback:
            loop.set_approval_callback(ctx.approval_callback)
        if ctx.nanny is not None:
            loop.set_nanny(ctx.nanny)
        for rule in ctx.session_rules:
            loop.add_session_rule(rule)
        loop.set_reasoning_effort(effort)

        async for event in loop.act(message, answer_only=answer_only):
            for entry in tracer.on_event(event):
                await queue.put(entry)

        answer = ""
        for msg in reversed(loop.messages[:]):
            if msg.role == MessageRole.assistant and msg.content:
                text = str(msg.content)
                if f"<{VIBE_STOP_EVENT_TAG}>" not in text:
                    answer = text
                    break
        for entry in tracer.end_section(answer, more=f"full text in {transcripts}"):
            await queue.put(entry)
        return answer


def _to_result(outcome: Outcome) -> OrchestrateResult:
    return OrchestrateResult(
        status=outcome.status,
        detail=outcome.detail,
        rounds=outcome.rounds,
        calls=outcome.calls,
        verifier_exit_code=outcome.verifier.exit_code if outcome.verifier else None,
        verifier_output=outcome.verifier.output if outcome.verifier else "",
        changed_files=outcome.changed_files,
        ledger_dir=str(outcome.ledger_dir),
    )
