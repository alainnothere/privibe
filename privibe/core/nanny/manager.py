"""The nanny manager: one reviewer conversation per session, one lock.

The nanny is a second AgentLoop (profile `nanny`: no tools, answer-only) whose
conversation receives, as user messages, the calls the worker wants approved
and what happened in between. It answers VERDICT/REASON. An approval runs the
call; anything else reaches the human with the reason attached.

One lock covers the whole review of a call: the ask, the human prompt if the
nanny did not approve, and every talk round. The lock is held by the
NannyReview handle, so talk turns run inside the review without re-acquiring
it (asyncio locks are not re-entrant). Lock order is always nanny first, then
the UI interaction lock, never the reverse.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path
from string import Template
import time
from typing import TYPE_CHECKING, Any

from privibe.core.agents.models import BuiltinAgentName
from privibe.core.config import SessionLoggingConfig, VibeConfig
from privibe.core.logger import logger
from privibe.core.nanny.presentation import NannyCallPresentation
from privibe.core.nanny.sidecar import (
    DECISIONS_FILENAME,
    NannyDecision,
    TalkExchange,
    append_decision,
    read_decisions,
)
from privibe.core.nanny.verdict import NannyAnswer, NannyVerdict, parse_verdict
from privibe.core.prompts import UtilityPrompt
from privibe.core.types import ApprovalResponse, AssistantEvent, LLMMessage, Role

if TYPE_CHECKING:
    from privibe.core.agent_loop import AgentLoop
    from privibe.core.llm.types import BackendLike

NANNY_SUBDIR = "nanny"
REFUSAL_PREFIX = "Refused by the reviewer: "
# Longest user prompt echoed into the nanny preamble, in characters.
PROMPT_ECHO_CAP = 2000


class NannyState(StrEnum):
    IDLE = auto()
    REVIEWING = auto()
    TALKING = auto()


class NannyListener:
    def on_nanny_state_change(self, state: NannyState) -> None:
        pass

    def on_nanny_decision(self, decision: NannyDecision) -> None:
        pass


class NannyConfigError(Exception):
    """The nanny could not be built from the config (bad alias, fallback)."""


class NannyUnavailable(Exception):
    """The nanny did not answer (timeout, backend error, empty reply)."""


@dataclass
class NannyAskRequest:
    tool_name: str
    tool_call_id: str
    agent: str
    presentation: NannyCallPresentation
    permissions_text: str = ""
    worker_text: str = ""
    siblings: list[str] = field(default_factory=list)


class NannyReview:
    """Handle for one reviewed call. Lives for the whole review and owns the
    manager lock for that time; the UI uses it to read the answer, talk to the
    nanny and record what the human decided.
    """

    def __init__(
        self,
        manager: NannyManager,
        request: NannyAskRequest,
        decision: NannyDecision,
        answer: NannyAnswer,
        *,
        failed: bool = False,
        notice: str | None = None,
    ) -> None:
        self._manager = manager
        self.request = request
        self.decision = decision
        self.answer = answer
        self.failed = failed
        self.notice = notice
        # Set by the front end to the option the human picked; the loop fills
        # in a coarse value from the callback result when it is left unset.
        self.human_choice: str | None = None

    @property
    def approved(self) -> bool:
        return self.answer.verdict is NannyVerdict.APPROVE and not self.failed

    @property
    def reason(self) -> str:
        return self.answer.reason

    @property
    def tool_name(self) -> str:
        return self.request.tool_name

    def refusal_feedback(self) -> str:
        """Untagged feedback for the worker when the human confirms a refusal:
        the worker reads it as the tool result and continues the turn.
        """
        return f"{REFUSAL_PREFIX}{self.answer.reason}"

    async def talk(self, text: str) -> str:
        """One human message to the nanny and its reply, inside this review.
        Runs under the lock this review already holds; never re-acquires it.
        """
        return await self._manager._talk(self, text)

    def human_decided(self, response: ApprovalResponse, feedback: str | None) -> None:
        if self.human_choice is None:
            self.human_choice = "allow" if response is ApprovalResponse.YES else "no"
        self.decision.human_decision = self.human_choice
        self.decision.human_feedback = feedback


class NannyManager:
    def __init__(
        self,
        *,
        config: VibeConfig,
        session_dir: Path | None,
        main_session_id: str,
        profile_name: str,
        cwd: Path | None = None,
        backend: BackendLike | None = None,
        loop_factory: Callable[[VibeConfig], AgentLoop] | None = None,
        config_loader: Callable[..., VibeConfig] | None = None,
        enable_streaming: bool = False,
    ) -> None:
        alias = config.nanny_model
        if not alias:
            raise NannyConfigError("nanny_model is not set")
        self.model_alias: str = alias
        self.timeout: float = config.nanny_timeout_seconds
        self.write_full_lines: int = config.nanny_write_full_lines
        self._effort = config.nanny_reasoning_effort
        self._backend = backend
        self._enable_streaming = enable_streaming
        self._loop_factory = loop_factory
        self._config_loader = config_loader or VibeConfig.load
        self._cwd = cwd or Path.cwd()
        self._profile_name = profile_name
        self.main_session_id = main_session_id
        self.header_session_id = main_session_id
        self._session_dir: Path | None = session_dir
        self._lock = asyncio.Lock()
        self._listeners: list[NannyListener] = []
        self._state = NannyState.IDLE
        self._preamble: list[str] = []
        self._header_sent = False
        self.decisions: list[NannyDecision] = []
        self._loop: AgentLoop = self._build_loop(alias)
        if self.decisions_path is not None:
            self.decisions = read_decisions(self.decisions_path)
            if self.decisions:
                self.header_session_id = self.decisions[-1].header_session_id
                self._header_sent = True

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def for_loop(
        cls, loop: AgentLoop, alias: str | None = None, **kwargs: Any
    ) -> NannyManager:
        """Build the nanny that reviews `loop` (the main TUI/console loop)
        from that loop's base config, or for `alias` when the choice was
        just made and is not in the config yet. Raises NannyConfigError when
        the alias is unset or unusable.
        """
        session_logger = loop.session_logger
        config = loop.base_config
        if alias is not None:
            config = config.model_copy(update={"nanny_model": alias})
        kwargs.setdefault("enable_streaming", getattr(loop, "enable_streaming", False))
        return cls(
            config=config,
            session_dir=session_logger.session_dir if session_logger.enabled else None,
            main_session_id=loop.session_id,
            profile_name=loop.agent_profile.name,
            **kwargs,
        )

    def _base_config(self, alias: str) -> VibeConfig:
        session_logging = SessionLoggingConfig(
            save_dir=str(self._session_dir / NANNY_SUBDIR) if self._session_dir else "",
            session_prefix=NANNY_SUBDIR,
            enabled=self._session_dir is not None,
        )
        try:
            base = self._config_loader(
                session_logging=session_logging, active_model=alias
            )
            base.get_active_model()
        except ValueError as exc:
            raise NannyConfigError(str(exc)) from exc
        if base._fallback_from_model is not None or base.active_model != alias:
            raise NannyConfigError(
                f"nanny model '{alias}' is unusable (not configured, or its API "
                f"key is missing); privibe would silently substitute "
                f"'{base.active_model}', so the nanny stays off."
            )
        return base

    def _build_loop(self, alias: str) -> AgentLoop:
        base = self._base_config(alias)
        if self._loop_factory is not None:
            loop = self._loop_factory(base)
        else:
            from privibe.core.agent_loop import AgentLoop

            loop = AgentLoop(
                config=base,
                agent_name=BuiltinAgentName.NANNY,
                backend=self._backend,
                enable_streaming=self._enable_streaming,
            )
        if loop.config._fallback_from_model is not None:
            raise NannyConfigError(
                f"nanny model '{alias}' fell back to '{loop.config.active_model}'"
            )
        loop.set_reasoning_effort(self._effort)
        return loop

    @property
    def loop(self) -> AgentLoop:
        return self._loop

    @property
    def enabled(self) -> bool:
        return True

    @property
    def state(self) -> NannyState:
        return self._state

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    @property
    def decisions_path(self) -> Path | None:
        if self._session_dir is None:
            return None
        return self._session_dir / NANNY_SUBDIR / DECISIONS_FILENAME

    # ------------------------------------------------------------------
    # Listeners
    # ------------------------------------------------------------------

    def add_listener(self, listener: NannyListener) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: NannyListener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _set_state(self, state: NannyState) -> None:
        self._state = state
        for listener in list(self._listeners):
            try:
                listener.on_nanny_state_change(state)
            except Exception as exc:
                logger.warning("nanny listener failed: %s", exc)

    def _emit_decision(self, decision: NannyDecision) -> None:
        for listener in list(self._listeners):
            try:
                listener.on_nanny_decision(decision)
            except Exception as exc:
                logger.warning("nanny listener failed: %s", exc)

    # ------------------------------------------------------------------
    # Preamble: what happened since the previous ask
    # ------------------------------------------------------------------

    def note_user_prompt(
        self, text: str, agent: str, *, steering: bool = False
    ) -> None:
        kind = "Steering message" if steering else "User prompt"
        shown = (
            text if len(text) <= PROMPT_ECHO_CAP else text[:PROMPT_ECHO_CAP] + " ..."
        )
        self._preamble.append(f'{kind} to agent "{agent}":\n{shown}')

    def note_outcome(
        self, tool_call_id: str, tool_name: str, status: str, agent: str
    ) -> None:
        self._preamble.append(
            f'Outcome: {tool_name} call {tool_call_id[:8]} by "{agent}": {status}'
        )

    def note_human_decision(self, decision: NannyDecision) -> None:
        choice = decision.human_decision or "undecided"
        line = (
            f"Human decision on {decision.tool_name} call "
            f"{decision.tool_call_id[:8]} you refused: {choice}"
        )
        if decision.human_feedback and choice in {"no", "refuse"}:
            line += f" (told the agent: {decision.human_feedback})"
        self._preamble.append(line)

    def note_rewind(self, text: str) -> None:
        self._preamble.append(f"Rewind: {text}")

    def _drain_preamble(self) -> str:
        if not self._preamble:
            return ""
        lines = "\n".join(f"- {item}" for item in self._preamble)
        self._preamble = []
        return f"Since the previous call:\n{lines}\n"

    # ------------------------------------------------------------------
    # Session directory
    # ------------------------------------------------------------------

    def on_main_session_reset(
        self, new_session_dir: Path | None, main_session_id: str
    ) -> None:
        """The main loop got a new session id (rewind, compaction, clear). The
        nanny's files follow it so resume finds them next to the conversation
        they belong to. The nanny conversation itself is untouched.
        """
        self.main_session_id = main_session_id
        if new_session_dir is None or self._session_dir == new_session_dir:
            return
        self._session_dir = new_session_dir
        session_logger = self._loop.session_logger
        if session_logger.enabled and session_logger.session_dir is not None:
            folder = session_logger.session_dir.name
            session_logger.save_dir = new_session_dir / NANNY_SUBDIR
            session_logger.session_dir = session_logger.save_dir / folder
            session_logger._forget_persisted_count()
        if self.decisions_path is not None:
            for decision in self.decisions:
                append_decision(self.decisions_path, decision)

    def resume(self, session_dir: Path, main_session_id: str) -> bool:
        """Reload the nanny conversation stored beside a resumed session.
        Returns False when that session has no nanny transcript.
        """
        nanny_dir = session_dir / NANNY_SUBDIR
        candidates = sorted(
            (p for p in nanny_dir.glob(f"{NANNY_SUBDIR}_*") if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
        )
        self._session_dir = session_dir
        self.main_session_id = main_session_id
        self.decisions = (
            read_decisions(self.decisions_path) if self.decisions_path else []
        )
        if not candidates:
            self._header_sent = False
            self.header_session_id = main_session_id
            return False
        latest = candidates[-1]
        session_logger = self._loop.session_logger
        if session_logger.enabled:
            session_logger.save_dir = nanny_dir
            session_logger.resume_existing_session(self._loop.session_id, latest)
        self._loop.messages.restore(latest)
        self._header_sent = True
        if self.decisions:
            self.header_session_id = self.decisions[-1].header_session_id
        self._preamble = []
        return True

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    async def wait_idle(self) -> None:
        """Returns once no review is in flight. Turn off and change model
        wait here rather than tearing the loop down under a running turn.
        """
        async with self._lock:
            pass

    async def change_model(self, alias: str) -> None:
        async with self._lock:
            base = self._base_config(alias)
            await self._loop.apply_runtime_config(base_config=base)
            if self._loop.config._fallback_from_model is not None:
                raise NannyConfigError(f"nanny model '{alias}' fell back")
            self.model_alias = alias
            self.timeout = base.nanny_timeout_seconds
            self.write_full_lines = base.nanny_write_full_lines

    # ------------------------------------------------------------------
    # Turns
    # ------------------------------------------------------------------

    def _close_dangling_user_turn(self, note: str) -> None:
        """A cancelled or failed turn leaves the ask as the tail with no
        answer. Chat templates reject two user turns in a row, so a stand-in
        assistant message closes it. Append-only: nothing is edited.
        """
        messages = self._loop.messages
        if len(messages) > 1 and messages[-1].role == Role.user:
            with messages.silent():
                messages.add(
                    LLMMessage(role=Role.assistant, content=f"(no answer: {note})")
                )
            # Nothing else will flush this before the next turn; a session
            # that ends here must still store a closed transcript.
            try:
                asyncio.ensure_future(messages.save())
            except RuntimeError:
                pass

    async def _turn(self, text: str, state: NannyState) -> tuple[str, int, int]:
        """One nanny turn. Returns (reply, ask_index, reply_index)."""
        loop = self._loop
        ask_index = len(loop.messages)
        pieces: list[str] = []

        async def run() -> None:
            async for event in loop.act(text, answer_only=True):
                if (
                    isinstance(event, AssistantEvent)
                    and event.content
                    and not event.stopped_by_middleware
                ):
                    pieces.append(event.content)

        self._set_state(state)
        try:
            await asyncio.wait_for(run(), timeout=self.timeout)
        except asyncio.CancelledError:
            self._close_dangling_user_turn("cancelled")
            raise
        except TimeoutError as exc:
            self._close_dangling_user_turn("timeout")
            raise NannyUnavailable(f"no answer within {self.timeout:.0f}s") from exc
        except Exception as exc:
            self._close_dangling_user_turn(f"{type(exc).__name__}")
            raise NannyUnavailable(f"{type(exc).__name__}: {exc}") from exc
        finally:
            self._set_state(NannyState.IDLE)

        reply = "".join(pieces).strip()
        reply_index = len(loop.messages) - 1
        if not reply:
            raise NannyUnavailable("empty answer")
        return reply, ask_index, reply_index

    async def _ensure_header(self) -> None:
        if self._header_sent:
            return
        text = Template(UtilityPrompt.NANNY_HEADER.read()).safe_substitute(
            session_id=self.header_session_id,
            cwd=str(self._cwd),
            profile=self._profile_name,
        )
        await self._turn(text, NannyState.REVIEWING)
        self._header_sent = True

    def _ask_text(self, request: NannyAskRequest) -> str:
        siblings = ""
        if request.siblings:
            siblings = (
                "Also pending in the same batch (judged separately): "
                + "; ".join(request.siblings)
                + "\n"
            )
        return Template(UtilityPrompt.NANNY_ASK.read()).safe_substitute(
            preamble=self._drain_preamble(),
            agent=request.agent,
            call=request.presentation.text,
            permissions=request.permissions_text,
            siblings=siblings,
            worker_text=request.worker_text.strip() or "(none)",
        )

    async def _ask(self, request: NannyAskRequest) -> NannyReview:
        decision = NannyDecision(
            tool_call_id=request.tool_call_id,
            tool_name=request.tool_name,
            agent=request.agent,
            verdict=NannyVerdict.REFUSE.value,
            reason="",
            ask_index=-1,
            verdict_index=-1,
            header_session_id=self.header_session_id,
            main_session_id=self.main_session_id,
        )
        try:
            await self._ensure_header()
            reply, ask_index, reply_index = await self._turn(
                self._ask_text(request), NannyState.REVIEWING
            )
            answer = parse_verdict(reply)
            if (
                answer.verdict is NannyVerdict.MORE
                and request.presentation.full_text is not None
            ):
                more = Template(UtilityPrompt.NANNY_MORE.read()).safe_substitute(
                    content=request.presentation.full_text
                )
                reply, _, reply_index = await self._turn(more, NannyState.REVIEWING)
                answer = parse_verdict(reply)
            if answer.verdict is NannyVerdict.MORE:
                # Nothing more to show, or asked twice: a refusal.
                answer = NannyAnswer(
                    verdict=NannyVerdict.REFUSE,
                    reason=answer.reason or "asked for more with nothing more to show",
                    raw=answer.raw,
                )
        except NannyUnavailable as exc:
            notice = f"nanny did not answer ({exc}); asking you instead"
            logger.warning("nanny: %s", notice)
            answer = NannyAnswer(
                verdict=NannyVerdict.REFUSE, reason=str(exc), raw="", readable=False
            )
            decision.failed = True
            decision.notice = notice
            decision.verdict = answer.verdict.value
            decision.reason = answer.reason
            decision.ask_index = len(self._loop.messages) - 1
            decision.verdict_index = decision.ask_index
            return NannyReview(
                self, request, decision, answer, failed=True, notice=notice
            )

        decision.verdict = answer.verdict.value
        decision.reason = answer.reason
        decision.ask_index = ask_index
        decision.verdict_index = reply_index
        if not answer.readable:
            decision.notice = "nanny answer could not be read; asking you instead"
        return NannyReview(self, request, decision, answer, notice=decision.notice)

    async def _talk(self, review: NannyReview, text: str) -> str:
        if not self._lock.locked():
            raise RuntimeError("talk() outside of a review")
        prompt = Template(UtilityPrompt.NANNY_TALK.read()).safe_substitute(text=text)
        try:
            reply, ask_index, reply_index = await self._turn(prompt, NannyState.TALKING)
        except NannyUnavailable as exc:
            reply, ask_index, reply_index = (
                f"(nanny did not answer: {exc})",
                len(self._loop.messages) - 1,
                len(self._loop.messages) - 1,
            )
        review.decision.talk.append(
            TalkExchange(
                human=text, nanny=reply, ask_index=ask_index, reply_index=reply_index
            )
        )
        return reply

    @asynccontextmanager
    async def review(self, request: NannyAskRequest) -> AsyncIterator[NannyReview]:
        """Review one call. The lock is held from the ask until the block
        exits, human prompt and talk rounds included, so no other call's ask
        can land in the nanny conversation mid-discussion.
        """
        async with self._lock:
            review = await self._ask(request)
            try:
                yield review
            finally:
                decision = review.decision
                decision.timestamp = time.time()
                if not review.approved:
                    self.note_human_decision(decision)
                self.decisions.append(decision)
                if self.decisions_path is not None:
                    append_decision(self.decisions_path, decision)
                self._emit_decision(decision)

    async def close(self) -> None:
        await self.wait_idle()


def nanny_session_dir(session_dir: Path | None) -> Path | None:
    return None if session_dir is None else session_dir / NANNY_SUBDIR


def decisions_for(session_dir: Path | None) -> dict[str, NannyDecision]:
    """Decisions stored beside a session, keyed by tool_call_id. The last
    record for an id wins (a call is reviewed once, but a resumed-then-moved
    session may carry duplicates from on_main_session_reset).
    """
    path = (
        None if session_dir is None else session_dir / NANNY_SUBDIR / DECISIONS_FILENAME
    )
    if path is None:
        return {}
    result: dict[str, NannyDecision] = {}
    for decision in read_decisions(path):
        result[decision.tool_call_id] = decision
    return result


__all__ = [
    "NANNY_SUBDIR",
    "REFUSAL_PREFIX",
    "NannyAskRequest",
    "NannyConfigError",
    "NannyListener",
    "NannyManager",
    "NannyReview",
    "NannyState",
    "NannyUnavailable",
    "decisions_for",
    "nanny_session_dir",
]
