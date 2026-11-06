from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Sequence
from pathlib import Path
import shlex
import signal
import sys
from typing import Any

from pydantic import BaseModel

from privibe import __version__
from privibe.core.agent_loop import AgentLoop
from privibe.core.agents.models import BuiltinAgentName
from privibe.core.config import VibeConfig
from privibe.core.conversation import is_step_budget_warning
from privibe.core.logger import logger
from privibe.core.output_formatters import create_formatter
from privibe.core.session.resume_sessions import short_session_id
from privibe.core.session.session_loader import SessionLoader
from privibe.core.types import (
    ApprovalResponse,
    AssistantEvent,
    ClientMetadata,
    EntrypointMetadata,
    LLMMessage,
    OutputFormat,
    Role,
)
from privibe.core.utils import ConversationLimitException
from privibe.core.utils.tags import CONTEXT_REFRESH_TAG

__all__ = ["ProgrammaticRunTerminated", "run_programmatic"]

# What a bare `-p --resume <id>` sends when no prompt comes with it.
CONTINUE_PROMPT = "continue"


class ProgrammaticRunTerminated(Exception):
    """SIGTERM ended the run (a `timeout`, a `kill`). The conversation was
    saved and the resume line printed before this was raised.
    """

    exit_code = 128 + 15


def _notice(text: str) -> None:
    # stderr: stdout belongs to the answer and to --output json consumers.
    print(f"[privibe] {text}", file=sys.stderr, flush=True)


def resume_command(session_ref: str, resume_flags: Sequence[str] = ()) -> str:
    """The command that picks a programmatic session up where it stopped.
    `-p` comes last and bare: with a session and no prompt it sends
    CONTINUE_PROMPT.
    """
    return shlex.join(["privibe", "--resume", session_ref, *resume_flags, "-p"])


def _announce_end(
    agent_loop: AgentLoop, problem: str | None, resume_flags: Sequence[str]
) -> None:
    """The last stderr line of a run: finished, or why not and how to go on."""
    session_logger = agent_loop.session_logger
    if not session_logger.enabled:
        return
    session_ref = short_session_id(session_logger.session_id)
    if problem is None:
        _notice(f"session {session_ref} finished")
        return
    on_disk = SessionLoader.does_session_exist(
        session_logger.session_id, session_logger.session_config
    )
    if on_disk is None:
        _notice(
            f"session {session_ref} did not finish ({problem}); nothing was "
            "saved yet, so there is nothing to resume"
        )
        return
    _notice(
        f"session {session_ref} did not finish ({problem}). "
        f"To continue it: {resume_command(session_ref, resume_flags)}"
    )

_DEFAULT_CLIENT_METADATA = ClientMetadata(name="vibe_programmatic", version=__version__)


async def _refuse_without_a_human(
    tool_name: str,
    args: BaseModel,
    tool_call_id: str,
    required_permissions: Any,
    review: Any = None,
) -> tuple[ApprovalResponse, str | None]:
    """The approval callback of a nanny-reviewed programmatic run. The loop
    only gets here with a call the nanny did not approve, and there is nobody
    at the keyboard to overrule it, so every such call is refused and the
    worker reads why as the tool result.
    """
    if review is None:
        # An escalated permission the nanny may not review (see
        # nanny_reviews_escalated): it is the human's call, and there is none.
        return (
            ApprovalResponse.NO,
            "This call needs a person to approve it (a path outside the "
            "working directory) and programmatic mode has none. It was not run.",
        )
    if review.failed:
        return (
            ApprovalResponse.NO,
            f"The nanny could not review this call ({review.notice or 'no answer'}) "
            "and programmatic mode has nobody else to ask. It was not run.",
        )
    return ApprovalResponse.NO, review.refusal_feedback()


def _adopt_session(agent_loop: AgentLoop, session_path: Path | None) -> bool:
    """Make the loop continue the session at `session_path` where it lives:
    same id, same directory. Without this the restored messages were saved
    under the fresh id the loop was born with, and every resume left one
    more copy behind. Returns whether a session was adopted.
    """
    session_logger = agent_loop.session_logger
    if session_path is None or not session_logger.enabled:
        return False
    session_id = SessionLoader.load_metadata(session_path).session_id
    agent_loop.session_id = session_id
    session_logger.resume_existing_session(session_id, session_path)
    return True


def _cancel_on_sigterm(terminated: list[bool]) -> None:
    """SIGTERM cancels the run like Ctrl-C does, so the loop's own `finally`
    still saves the conversation. Windows event loops have no signal
    handlers, and neither does a thread that is not the main one: there it
    stays a plain kill and the id printed at the start is all there is.
    """
    main_task = asyncio.current_task()

    def _on_sigterm() -> None:
        terminated.append(True)
        if main_task is not None:
            main_task.cancel()

    try:
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, _on_sigterm)
    except (NotImplementedError, RuntimeError, ValueError):
        pass


def _preload_previous_messages(
    agent_loop: AgentLoop, previous_messages: list[LLMMessage]
) -> None:
    from privibe.core.system_prompt import build_context_refresh_content

    non_system = [m for m in previous_messages if m.role != Role.system]
    for msg in non_system:
        agent_loop.messages.add(msg)
    last = agent_loop.messages[-1] if agent_loop.messages else None
    if last and f"<{CONTEXT_REFRESH_TAG}>" in (last.content or ""):
        agent_loop.messages.rewind(1)
    if agent_loop.messages and is_step_budget_warning(agent_loop.messages[-1]):
        # A write-up request whose call never happened; see
        # conversation._drop_dangling_step_budget_warning.
        agent_loop.messages.rewind(1)
    content = build_context_refresh_content(agent_loop.config)
    agent_loop.messages.add(LLMMessage(role=Role.user, content=content, injected=True))
    logger.info("Loaded %d previous messages", len(non_system))


async def _attach_headless_nanny(
    agent_loop: AgentLoop, alias: str, *, resumed: bool
) -> Any:
    from privibe.core.nanny import NannyConfigError, NannyManager

    try:
        nanny = NannyManager.for_loop(agent_loop, alias=alias)
    except NannyConfigError as exc:
        raise RuntimeError(f"nanny not started: {exc}") from exc
    session_dir = agent_loop.session_logger.session_dir
    if resumed and session_dir is not None:
        # The nanny's own conversation is stored beside the session: reload
        # it, so it reviews the rest knowing what it already decided. A
        # manager built on a session with decisions on file believes its
        # header was sent; without this reload it would review with an
        # empty context.
        nanny.resume(session_dir, agent_loop.session_id)
    agent_loop.set_nanny(nanny, owner=True)
    agent_loop.set_approval_callback(_refuse_without_a_human)
    await agent_loop.set_session_nanny_model(nanny.model_alias)
    return nanny


def _run_and_announce(
    run: Callable[[], Coroutine[Any, Any, str | None]],
    agent_loop: AgentLoop,
    terminated: list[bool],
    resume_flags: Sequence[str],
) -> str | None:
    """Run to the end and say on stderr how it ended. Every way out that is
    not a finished answer names the session and the command that resumes it.
    """
    try:
        result = asyncio.run(run())
    except KeyboardInterrupt:
        _announce_end(agent_loop, "interrupted", resume_flags)
        raise
    except asyncio.CancelledError:
        if not terminated:
            raise
        _announce_end(agent_loop, "terminated by SIGTERM", resume_flags)
        raise ProgrammaticRunTerminated() from None
    except ConversationLimitException:
        _announce_end(agent_loop, "stopped at a limit", resume_flags)
        raise
    except Exception as exc:
        _announce_end(agent_loop, f"failed: {type(exc).__name__}", resume_flags)
        raise
    _announce_end(agent_loop, None, resume_flags)
    return result


def run_programmatic(  # noqa: PLR0913
    config: VibeConfig,
    prompt: str,
    max_turns: int | None = None,
    max_price: float | None = None,
    output_format: OutputFormat = OutputFormat.TEXT,
    session_path: Path | None = None,
    previous_messages: list[LLMMessage] | None = None,
    agent_name: str = BuiltinAgentName.AUTO_APPROVE,
    client_metadata: ClientMetadata = _DEFAULT_CLIENT_METADATA,
    *,
    nanny_alias: str | None = None,
    resume_flags: Sequence[str] = (),
    pinned: frozenset[str] = frozenset(),
) -> str | None:
    """`nanny_alias` (--nanny) puts that model in the approver's seat: calls
    the agent profile would ask about go to the nanny, and whatever it does
    not approve is refused. Without it nothing here answers an approval
    prompt, which is why -p defaults to the auto-approve agent.

    `session_path` continues that session in place: same id, same directory,
    on the model and agent profile it had saved, except the parts named in
    `pinned` ("model" for --model, "agent" for --agent), which this run
    chose explicitly.

    The session id goes to stderr when the run starts, so it is known even
    if the process is killed outright, and again when it ends; a run that
    did not finish (an error, a limit, Ctrl-C, SIGTERM) also gets the
    command that continues it, with `resume_flags` (the flags this run
    needs again, e.g. --nanny) in it.
    """
    formatter = create_formatter(output_format)

    agent_loop = AgentLoop(
        config,
        agent_name=agent_name,
        message_observer=formatter.on_message_added,
        max_turns=max_turns,
        max_price=max_price,
        enable_streaming=False,
        entrypoint_metadata=EntrypointMetadata(
            agent_entrypoint="programmatic",
            agent_version=__version__,
            client_name=client_metadata.name,
            client_version=client_metadata.version,
        ),
    )
    logger.info("USER: %s", prompt)

    resumed = _adopt_session(agent_loop, session_path)
    if agent_loop.session_logger.enabled:
        session_ref = short_session_id(agent_loop.session_logger.session_id)
        _notice(f"session {session_ref}" + (" (resumed)" if resumed else ""))

    terminated: list[bool] = []

    async def _load_history() -> None:
        if session_path is not None:
            agent_loop.messages.restore(session_path)
            logger.info(
                "Loaded %d messages from previous session", len(agent_loop.messages)
            )
            if resumed:
                # The session goes on as it was: its saved model and agent
                # profile win over this run's defaults. Falling back to
                # -p's auto-approve here would hand a session that used to
                # ask (an interactive one, a nanny-reviewed one) a blank
                # cheque; a profile that asks with nobody to answer only
                # refuses.
                note = await agent_loop.restore_run_state(pinned)
                if note:
                    _notice(note)
        elif previous_messages is not None:
            _preload_previous_messages(agent_loop, previous_messages)

    async def _async_run() -> str | None:
        _cancel_on_sigterm(terminated)
        ctx_msg = await agent_loop.resolve_context_size()
        if ctx_msg:
            logger.warning(ctx_msg)
        from privibe.core.config.migration import pop_pending_message

        migration_msg = pop_pending_message()
        if migration_msg:
            logger.warning(migration_msg)
        await _load_history()

        nanny = None
        if nanny_alias:
            nanny = await _attach_headless_nanny(
                agent_loop, nanny_alias, resumed=resumed
            )

        try:
            async for event in agent_loop.act(prompt):
                formatter.on_event(event)
                if isinstance(event, AssistantEvent) and event.stopped_by_middleware:
                    raise ConversationLimitException(event.content)
        finally:
            if nanny is not None:
                await nanny.close()

        return formatter.finalize()

    return _run_and_announce(_async_run, agent_loop, terminated, resume_flags)
