from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Callable, Generator
import contextlib
from enum import StrEnum, auto
from http import HTTPStatus
from pathlib import Path
from threading import Thread
import time
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from pydantic import BaseModel

from privibe.core.agents.manager import AgentManager
from privibe.core.agents.models import AgentProfile, BuiltinAgentName
from privibe.core.config import ModelConfig, ProviderConfig, VibeConfig
from privibe.core.conversation import ConversationList
from privibe.core.llm.backend.factory import BACKEND_FACTORY
from privibe.core.llm.backend.generic import GenericBackend
from privibe.core.logger import logger
from privibe.core.llm.exceptions import BackendError
from privibe.core.llm.format import (
    APIToolFormatHandler,
    FailedToolCall,
    ResolvedMessage,
    ResolvedToolCall,
)
from privibe.core.llm.types import BackendLike
from privibe.core.middleware import (
    CHAT_AGENT_EXIT,
    CHAT_AGENT_REMINDER,
    PLAN_AGENT_EXIT,
    AutoCompactMiddleware,
    ContextWarningMiddleware,
    ConversationContext,
    MiddlewareAction,
    MiddlewarePipeline,
    MiddlewareResult,
    PriceLimitMiddleware,
    ReadOnlyAgentMiddleware,
    ResetReason,
    TurnLimitMiddleware,
    make_plan_agent_reminder,
)
from privibe.core.plan_session import PlanSession
from privibe.core.prompts import UtilityPrompt
from privibe.core.rewind import RewindManager
from privibe.core.rewind.undo_stack import FileUndoStack
from privibe.core.session.session_logger import SessionLogger
from privibe.core.session.session_migration import migrate_sessions_entrypoint
from privibe.core.skills.manager import SkillManager
from privibe.core.system_prompt import get_universal_system_prompt
from privibe.core.tools.base import (
    BaseTool,
    InvokeContext,
    ToolError,
    ToolPermission,
    ToolPermissionError,
)
from privibe.core.tools.manager import ToolManager
from privibe.core.tools.mcp import MCPRegistry
from privibe.core.tools.mcp_sampling import MCPSamplingHandler
from privibe.core.tools.permissions import (
    ApprovedRule,
    PermissionContext,
    RequiredPermission,
)
from privibe.core.tools.utils import wildcard_match
from privibe.core.types import (
    AgentProfileChangedEvent,
    AgentStats,
    ApprovalCallback,
    ApprovalResponse,
    AssistantEvent,
    BaseEvent,
    CompactEndEvent,
    CompactStartEvent,
    EntrypointMetadata,
    LLMChunk,
    LLMMessage,
    LLMUsage,
    RateLimitError,
    ReasoningEvent,
    Role,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
    ToolStreamEvent,
    UserInputCallback,
    UserMessageEvent,
)
from privibe.core.utils import (
    CANCELLATION_TAG,
    TOOL_ERROR_TAG,
    VIBE_STOP_EVENT_TAG,
    CancellationReason,
    get_user_agent,
    get_user_cancellation_message,
    is_user_cancellation_event,
)

class ToolExecutionResponse(StrEnum):
    SKIP = auto()
    EXECUTE = auto()


class ToolDecision(BaseModel):
    verdict: ToolExecutionResponse
    approval_type: ToolPermission
    feedback: str | None = None


class AgentLoopError(Exception):
    """Base exception for AgentLoop errors."""


class AgentLoopStateError(AgentLoopError):
    """Raised when agent loop is in an invalid state."""


class AgentLoopLLMResponseError(AgentLoopError):
    """Raised when LLM response is malformed or missing expected data."""


def _should_raise_rate_limit_error(e: Exception) -> bool:
    return isinstance(e, BackendError) and e.status == HTTPStatus.TOO_MANY_REQUESTS


# Wall-clock budget for context-size auto-detection. A healthy /v1/models or
# /props lookup is sub-100ms; anything slower is a configuration problem and
# we'd rather give up once than block startup or every turn.
_CONTEXT_SIZE_TIMEOUT_S: float = 2.0

# Per-model, per-process detection latches, keyed by model alias. act() fires
# detection on *every* turn, but it does real work only until it resolves or
# fails for the *current* model; later turns short-circuit. Keying by alias
# means switching models re-detects automatically (the new alias matches
# neither latch). This is *separate* from the persistent
# VibeConfig.auto_detect_context_size flag: that flag is the user's intent
# ("should we auto-detect at all?"); these latches are the in-process "we
# already settled this model this run, don't keep hammering." The
# /detect-context-size toggle clears both so a manual off/on re-pulls.
_context_size_resolved_for: str | None = None
_context_size_failed_for: str | None = None


def reset_context_size_detection_state() -> None:
    """Clear the per-model detection latches so the next call re-detects.

    Called when the user toggles auto-detection back on via /detect-context-size,
    so a manual off/on re-pulls context size and the cosmetic model name.
    """
    global _context_size_resolved_for, _context_size_failed_for
    _context_size_resolved_for = None
    _context_size_failed_for = None


class AgentLoop:
    def __init__(
        self,
        config: VibeConfig,
        agent_name: str = BuiltinAgentName.DEFAULT,
        message_observer: Callable[[LLMMessage], None] | None = None,
        max_turns: int | None = None,
        max_price: float | None = None,
        backend: BackendLike | None = None,
        enable_streaming: bool = False,
        entrypoint_metadata: EntrypointMetadata | None = None,
    ) -> None:
        self._base_config = config
        self.mcp_registry = MCPRegistry()
        self.agent_manager = AgentManager(
            lambda: self._base_config, initial_agent=agent_name
        )
        self.tool_manager = ToolManager(
            lambda: self.config, mcp_registry=self.mcp_registry
        )
        self.skill_manager = SkillManager(lambda: self.config)
        self.message_observer = message_observer
        self._max_turns = max_turns
        self._max_price = max_price
        self._plan_session = PlanSession()

        self.format_handler = APIToolFormatHandler()

        self.backend_factory = lambda: backend or self._select_backend()
        self.backend = self.backend_factory()
        self._sampling_handler = MCPSamplingHandler(
            backend_getter=lambda: self.backend, config_getter=lambda: self.config
        )

        self.enable_streaming = enable_streaming
        self.middleware_pipeline = MiddlewarePipeline()
        self._setup_middleware()

        self.messages = ConversationList(
            observer=message_observer,
            config_getter=lambda: self.config,
        )

        self.stats = AgentStats()
        self.approval_callback: ApprovalCallback | None = None
        self.user_input_callback: UserInputCallback | None = None
        self.entrypoint_metadata = entrypoint_metadata
        self.session_id = str(uuid4())

        try:
            active_model = config.get_active_model()
            self.stats.input_price_per_million = active_model.input_price
            self.stats.output_price_per_million = active_model.output_price
        except ValueError:
            pass

        self._current_user_message_id: str | None = None
        self._is_user_prompt_call: bool = False
        self._act_call_count: int = 0
        # Cosmetic model name detected from the endpoint (display-only; never
        # written to config). Stored with the alias it was detected for so a
        # stale name is never shown next to a different active model.
        self._detected_model_name: str | None = None
        self._detected_model_alias: str | None = None
        # agent steering — Steering message queue. When the user types a message
        # while the agent is running (instead of pressing Esc to cancel), the
        # message is queued here. It will be injected into the next tool result
        # as a user instruction, allowing the user to steer the conversation
        # without breaking the LLM/harness cycle and forcing a full reprocess.
        # If the loop exits (no more tool calls) with queued messages, they are
        # drained and delivered as a new user turn via _handle_user_message().
        self._steering_queue: list[str] = []

        self._session_rules: list[ApprovedRule] = []

        self.session_logger = SessionLogger(config.session_logging, self.session_id)
        self.rewind_manager = RewindManager(
            messages=self.messages,
            save_messages=self._save_messages,
            reset_session=self._reset_session,
        )
        # Per-agent in-memory undo history for file edits. Cleared on session
        # reset/clear/compact (via the messages reset hook) and on rewind (via
        # _reset_session). Each subagent gets its own AgentLoop, hence its own
        # stack, which dies when that agent is torn down.
        self.undo_stack = FileUndoStack()
        self.messages.on_reset(self.undo_stack.clear)
        self.messages.set_save_fn(self._save_messages)

        # Populate the initial system message
        system_prompt = get_universal_system_prompt(
            self.tool_manager, self.config, self.skill_manager, self.agent_manager
        )
        self.messages.add(LLMMessage(role=Role.system, content=system_prompt))

        thread = Thread(
            target=migrate_sessions_entrypoint,
            args=(config.session_logging,),
            daemon=True,
            name="migrate_sessions",
        )
        thread.start()

    # agent steering — Public API for the UI to queue a steering message.
    def queue_steering(self, message: str) -> None:
        """Queue a user message to be injected into the next tool result."""
        self._steering_queue.append(message)

    # agent steering — Drain and return all queued steering messages.
    def drain_steering_queue(self) -> list[str]:
        """Return and clear the steering queue. Called after the agent loop exits."""
        msgs = list(self._steering_queue)
        self._steering_queue.clear()
        return msgs

    @property
    def agent_profile(self) -> AgentProfile:
        return self.agent_manager.active_profile

    @property
    def base_config(self) -> VibeConfig:
        return self._base_config

    @property
    def config(self) -> VibeConfig:
        return self.agent_manager.config

    @property
    def auto_approve(self) -> bool:
        return self.config.auto_approve

    def refresh_config(self) -> None:
        self._base_config = VibeConfig.load()
        self.agent_manager.invalidate_config()

    def set_approval_callback(self, callback: ApprovalCallback) -> None:
        self.approval_callback = callback

    def set_user_input_callback(self, callback: UserInputCallback) -> None:
        self.user_input_callback = callback

    def set_tool_permission(
        self, tool_name: str, permission: ToolPermission, save_permanently: bool = False
    ) -> None:
        if save_permanently:
            VibeConfig.save_updates({
                "tools": {tool_name: {"permission": permission.value}}
            })

        if tool_name not in self.config.tools:
            self.config.tools[tool_name] = {}

        self.config.tools[tool_name]["permission"] = permission.value
        self.tool_manager.invalidate_tool(tool_name)

    def add_session_rule(self, rule: ApprovedRule) -> None:
        self._session_rules.append(rule)

    def _is_permission_covered(self, tool_name: str, rp: RequiredPermission) -> bool:
        return any(
            rule.tool_name == tool_name
            and rule.scope == rp.scope
            and wildcard_match(rp.invocation_pattern, rule.session_pattern)
            for rule in self._session_rules
        )

    def approve_always(
        self,
        tool_name: str,
        required_permissions: list[RequiredPermission] | None,
        save_permanently: bool = False,
    ) -> None:
        """Handle 'Allow Always' approval: add session rules or set tool-level permission."""
        if required_permissions:
            for rp in required_permissions:
                self.add_session_rule(
                    ApprovedRule(
                        tool_name=tool_name,
                        scope=rp.scope,
                        session_pattern=rp.session_pattern,
                    )
                )
        else:
            self.set_tool_permission(
                tool_name, ToolPermission.ALWAYS, save_permanently=save_permanently
            )

    def _select_backend(self) -> BackendLike:
        active_model = self.config.get_active_model()
        provider = self.config.get_provider_for_model(active_model)
        timeout = self.config.api_timeout
        return BACKEND_FACTORY[provider.backend](provider=provider, timeout=timeout)

    def detected_model_display_name(self) -> str | None:
        """Server-reported name for the *current* active model, detected from the
        endpoint, or None when nothing was detected for it. Cosmetic/display-only;
        never affects config or matching."""
        active = self.config.get_active_model()
        if self._detected_model_alias == active.alias:
            return self._detected_model_name
        return None

    async def resolve_context_size(self) -> str | None:
        """For generic (OpenAI-compatible) backends, query the model's actual context
        window size via /props or /v1/models and update auto_compact_threshold; also
        capture the server-reported model name for cosmetic display.

        Runs on every turn but does real work only until it resolves or fails for
        the current model (per-alias latches), so repeated turns short-circuit.
        Returns a user-facing message string when this attempt failed (so the
        configured value is used). Returns None on success, when the backend
        doesn't support detection, when the user has turned auto-detection off via
        /detect-context-size, or when this model was already settled this run. The
        2s wall-clock cap exists because a healthy /props or /v1/models lookup is
        sub-100ms; anything slower is a configuration problem.
        """
        global _context_size_resolved_for, _context_size_failed_for
        if not self.config.auto_detect_context_size:
            return None

        active_model = self.config.get_active_model()
        alias = active_model.alias
        if alias in (_context_size_resolved_for, _context_size_failed_for):
            return None

        provider = self.config.get_provider_for_model(active_model)
        if provider.backend != "generic":
            logger.info(
                "Context size auto-detection skipped: backend '%s' is not generic.",
                provider.backend,
            )
            # Nothing to detect for this model; latch so we don't retry every turn.
            _context_size_resolved_for = alias
            return None

        models_url = f"{provider.api_base.rstrip('/')}/models"
        logger.info(
            "Querying context size for model '%s' from %s ...",
            active_model.name,
            models_url,
        )
        backend = GenericBackend(provider=provider, timeout=_CONTEXT_SIZE_TIMEOUT_S)
        try:
            async with asyncio.timeout(_CONTEXT_SIZE_TIMEOUT_S):
                info = await backend.fetch_model_endpoint_info(active_model.name)
        except TimeoutError:
            _context_size_failed_for = alias
            logger.warning(
                "Context size lookup at %s exceeded %.0fs — disabling auto-detection "
                "for this model this run; using configured value (%d tokens).",
                models_url,
                _CONTEXT_SIZE_TIMEOUT_S,
                active_model.auto_compact_threshold,
            )
            return (
                f"Could not retrieve context size from {models_url} within "
                f"{_CONTEXT_SIZE_TIMEOUT_S:.0f}s. Auto-detection is disabled for "
                f"the rest of this run; the configured value "
                f"({active_model.auto_compact_threshold} tokens) will be used. "
                f"Retry with /detect-context-size."
            )
        except Exception as exc:
            _context_size_failed_for = alias
            logger.warning(
                "Context size lookup at %s failed (%s) — disabling auto-detection "
                "for this model this run; using configured value (%d tokens).",
                models_url,
                exc,
                active_model.auto_compact_threshold,
                exc_info=True,
            )
            return (
                f"Could not retrieve context size from {models_url} "
                f"({type(exc).__name__}). Auto-detection is disabled for the rest "
                f"of this run; the configured value "
                f"({active_model.auto_compact_threshold} tokens) will be used. "
                f"Retry with /detect-context-size."
            )

        # Cosmetic model name (display only — never touches config or matching).
        if info.display_name:
            self._detected_model_name = info.display_name
            self._detected_model_alias = alias

        ctx_size = info.context_size
        if ctx_size is None:
            _context_size_failed_for = alias
            logger.info(
                "Context size not found in endpoint response for model '%s' "
                "at %s — disabling auto-detection for this model this run; "
                "falling back to configured value (%d tokens).",
                active_model.name,
                provider.api_base,
                active_model.auto_compact_threshold,
            )
            return (
                f"Context size not exposed by {models_url} for model "
                f"'{active_model.name}'. Auto-detection is disabled for the rest "
                f"of this run; the configured value "
                f"({active_model.auto_compact_threshold} tokens) will be used. "
                f"Retry with /detect-context-size."
            )

        logger.info(
            "Detected context size for model '%s': %d tokens (was %d). "
            "Updating auto_compact_threshold.",
            active_model.name,
            ctx_size,
            active_model.auto_compact_threshold,
        )

        # Update every config layer that holds this model so the context bar,
        # auto-compact middleware, and compaction all see the live value.
        for cfg in (self._base_config, self.agent_manager.config):
            for model in cfg.models:
                if model.alias == active_model.alias:
                    model.auto_compact_threshold = ctx_size
                    break
        _context_size_resolved_for = alias
        return None

    async def _save_messages(self) -> None:
        await self.session_logger.save_interaction(
            self.messages,  # type: ignore[arg-type]
            self.stats,
            self._base_config,
            self.tool_manager,
            self.agent_profile,
        )

    def start_preflight_warmup_if_enabled(self) -> None:
        pass

    def _cancel_preflight_warmup(self) -> None:
        pass

    # DEBUG LLM COMMUNICATIONS
    def _dump_messages_state(
        self,
        active_model: ModelConfig,
        provider: ProviderConfig,
        *,
        kind: str = "real",
        messages_override: list[LLMMessage] | None = None,
    ) -> None:
        """Persist the message list to ~/.privibe/debug/ before each LLM call.

        Filename: {ts}_{seq}_{session8}_{kind}_msgs{N}.json — same scheme
        as the payload dumper so a single `ls $VIBE_HOME/debug | sort`
        groups by time and each (session, turn, kind) triple is
        unambiguous regardless of which cwd privibe was launched from.
        """
        import datetime as _dt

        from privibe.core.llm.backend._debug_dump import _seq
        from privibe.core.paths import DEBUG_DIR

        try:
            msgs = messages_override if messages_override is not None else list(self.messages)
            ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
            seq = next(_seq)
            session8 = (self.session_id or "nosess").replace("-", "")[:8]
            fname = f"{ts}_{seq:04d}_{session8}_{kind}_msgs{len(msgs)}.json"
            _DEBUG_DIR = DEBUG_DIR.path
            _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            payload: dict[str, Any] = {
                "ts": ts,
                "seq": seq,
                "session_id": self.session_id,
                "turn": self._act_call_count,
                "step": self.stats.steps,
                "kind": kind,
                "model": active_model.name,
                "provider": provider.name,
                "total_messages": len(msgs),
                "messages": [
                    {
                        "idx": i,
                        "role": m.role.value,
                        "content": m.content,
                        "content_len": len(m.content or ""),
                        "reasoning_content": m.reasoning_content,
                        "reasoning_len": len(m.reasoning_content or ""),
                        "tool_calls": [
                            (tc.id, tc.function.name if tc.function else None)
                            for tc in (m.tool_calls or [])
                        ],
                        "tool_call_id": m.tool_call_id,
                        "name": m.name,
                    }
                    for i, m in enumerate(msgs)
                ],
            }
            (_DEBUG_DIR / fname).write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
        except Exception as exc:
            # Never block the request path on a debug-dump failure.
            logger.debug("debug dump failed: %s", exc)

    async def act(
        self, msg: str, client_message_id: str | None = None
    ) -> AsyncGenerator[BaseEvent, None]:
        self._cancel_preflight_warmup()
        self.rewind_manager.create_checkpoint()
        self._act_call_count += 1
        # Auto-detect context size (and the cosmetic model name) every turn. The
        # call self-guards: it short-circuits when disabled or already settled for
        # the current model, so this does real work only until it first resolves.
        asyncio.ensure_future(self.resolve_context_size())
        self._append_shims_for_dangling_tool_calls()
        async for event in self._conversation_loop(
            msg, client_message_id=client_message_id
        ):
            yield event

    def _append_shims_for_dangling_tool_calls(self) -> None:
        """Append placeholder tool responses for unanswered calls at the conversation tail.

        When a session is interrupted mid-tool-execution, the last assistant message may
        have tool_calls that never received responses. We add shim results so the message
        history is valid before sending to the LLM. Only acts when the gap is at the tail
        (nothing but tool responses follows the last assistant-with-tools message).
        """
        last_tool_assistant_idx = -1
        for i, msg in enumerate(self.messages):
            if msg.role == Role.assistant and msg.tool_calls:
                last_tool_assistant_idx = i

        if last_tool_assistant_idx == -1:
            return

        tail = list(self.messages[last_tool_assistant_idx + 1:])
        # If any non-tool message follows the assistant, the gap is mid-history; skip.
        if any(m.role != Role.tool for m in tail):
            return

        responded_ids = {m.tool_call_id for m in tail if m.tool_call_id}
        last_tool_assistant = self.messages[last_tool_assistant_idx]
        for tc in last_tool_assistant.tool_calls:
            if (tc.id or "") not in responded_ids:
                self.messages.add(
                    LLMMessage(
                        role=Role.tool,
                        tool_call_id=tc.id or "",
                        name=(tc.function.name or "") if tc.function else "",
                        content=str(
                            get_user_cancellation_message(CancellationReason.TOOL_NO_RESPONSE)
                        ),
                    )
                )
    def _setup_middleware(self) -> None:
        """Configure middleware pipeline for this conversation."""
        self.middleware_pipeline.clear()

        if self._max_turns is not None:
            self.middleware_pipeline.add(TurnLimitMiddleware(self._max_turns))

        if self._max_price is not None:
            self.middleware_pipeline.add(PriceLimitMiddleware(self._max_price))

        self.middleware_pipeline.add(AutoCompactMiddleware())
        if self.config.context_warnings:
            self.middleware_pipeline.add(ContextWarningMiddleware(0.5))

        self.middleware_pipeline.add(
            ReadOnlyAgentMiddleware(
                lambda: self.agent_profile,
                BuiltinAgentName.PLAN,
                lambda: make_plan_agent_reminder(self._plan_session.plan_file_path_str),
                PLAN_AGENT_EXIT,
            )
        )
        self.middleware_pipeline.add(
            ReadOnlyAgentMiddleware(
                lambda: self.agent_profile,
                BuiltinAgentName.CHAT,
                CHAT_AGENT_REMINDER,
                CHAT_AGENT_EXIT,
            )
        )

    async def _handle_middleware_result(
        self, result: MiddlewareResult
    ) -> AsyncGenerator[BaseEvent]:
        match result.action:
            case MiddlewareAction.STOP:
                yield AssistantEvent(
                    content=f"<{VIBE_STOP_EVENT_TAG}>{result.reason}</{VIBE_STOP_EVENT_TAG}>",
                    stopped_by_middleware=True,
                )

            case MiddlewareAction.INJECT_MESSAGE:
                if result.message:
                    injected_message = LLMMessage(
                        role=Role.user, content=result.message, injected=True
                    )
                    self.messages.add(injected_message)

            case MiddlewareAction.COMPACT:
                old_tokens = result.metadata.get(
                    "old_tokens", self.stats.context_tokens
                )
                threshold = result.metadata.get(
                    "threshold", self.config.get_active_model().auto_compact_threshold
                )
                tool_call_id = str(uuid4())

                yield CompactStartEvent(
                    tool_call_id=tool_call_id,
                    current_context_tokens=old_tokens,
                    threshold=threshold,
                )
                summary = await self.compact()

                yield CompactEndEvent(
                    tool_call_id=tool_call_id,
                    old_context_tokens=old_tokens,
                    new_context_tokens=self.stats.context_tokens,
                    summary_length=len(summary),
                )

            case MiddlewareAction.CONTINUE:
                pass

    def _get_context(self) -> ConversationContext:
        return ConversationContext(
            messages=self.messages, stats=self.stats, config=self.config
        )

    def _build_metadata(self) -> dict[str, str]:
        base = self.entrypoint_metadata.model_dump() if self.entrypoint_metadata else {}
        metadata = base | {
            "session_id": self.session_id,
            "is_user_prompt": "true" if self._is_user_prompt_call else "false",
            "call_type": (
                "main_call" if self._is_user_prompt_call else "secondary_call"
            ),
        }
        if self._current_user_message_id is not None:
            metadata["message_id"] = self._current_user_message_id
        return metadata

    def _get_extra_headers(self, provider: ProviderConfig) -> dict[str, str]:
        headers: dict[str, str] = {
            "user-agent": get_user_agent(provider.backend),
            "x-affinity": self.session_id,
        }
        return headers

    async def _conversation_loop(
        self, user_msg: str, client_message_id: str | None = None
    ) -> AsyncGenerator[BaseEvent]:
        user_message = LLMMessage(
            role=Role.user, content=user_msg, message_id=client_message_id
        )
        self.messages.add(user_message)
        self.stats.steps += 1
        self._current_user_message_id = user_message.message_id

        if user_message.message_id is None:
            raise AgentLoopError("User message must have a message_id")

        yield UserMessageEvent(content=user_msg, message_id=user_message.message_id)

        try:
            should_break_loop = False
            first_llm_turn = True
            while not should_break_loop:
                self._is_user_prompt_call = False
                result = await self.middleware_pipeline.run_before_turn(
                    self._get_context()
                )
                async for event in self._handle_middleware_result(result):
                    yield event

                if result.action == MiddlewareAction.STOP:
                    return

                self.stats.steps += 1
                user_cancelled = False
                if first_llm_turn:
                    self._is_user_prompt_call = True
                    first_llm_turn = False
                async for event in self._perform_llm_turn():
                    if is_user_cancellation_event(event):
                        user_cancelled = True
                    yield event
                    await self._save_messages()
                self._is_user_prompt_call = False

                last_message = self.messages[-1]
                should_break_loop = last_message.role != Role.tool

                if user_cancelled:
                    return

        finally:
            await self._save_messages()

    async def _perform_llm_turn(self) -> AsyncGenerator[BaseEvent, None]:
        if self.enable_streaming:
            async for event in self._stream_assistant_events():
                yield event
        else:
            assistant_event = await self._get_assistant_event()
            if assistant_event.content:
                yield assistant_event

        last_message = self.messages[-1]

        parsed = self.format_handler.parse_message(last_message)
        resolved = self.format_handler.resolve_tool_calls(parsed, self.tool_manager)

        if not resolved.tool_calls and not resolved.failed_calls:
            return

        profile_before = self.agent_profile.name
        async for event in self._handle_tool_calls(resolved):
            yield event
        if self.agent_profile.name != profile_before:
            yield AgentProfileChangedEvent(agent_name=self.agent_profile.name)

    def _build_tool_call_events(
        self, tool_calls: list[ToolCall] | None, emitted_ids: set[str]
    ) -> Generator[ToolCallEvent, None, None]:
        for tc in tool_calls or []:
            if tc.id is None or not tc.function.name:
                continue
            if tc.id in emitted_ids:
                continue

            tool_class = self.tool_manager.available_tools.get(tc.function.name)
            if tool_class is None:
                continue

            tool_config = self.tool_manager.get_tool_config(tc.function.name)
            yield ToolCallEvent(
                tool_call_id=tc.id,
                tool_call_index=tc.index,
                tool_name=tc.function.name,
                tool_class=tool_class,
                timeout=getattr(tool_config, "default_timeout", None),
            )

    async def _stream_assistant_events(
        self,
    ) -> AsyncGenerator[AssistantEvent | ReasoningEvent | ToolCallEvent]:
        message_id: str | None = None
        reasoning_message_id: str | None = None
        emitted_tool_call_ids = set[str]()

        async for chunk in self._chat_streaming():
            if message_id is None:
                message_id = chunk.message.message_id
            if reasoning_message_id is None:
                reasoning_message_id = chunk.message.reasoning_message_id

            for event in self._build_tool_call_events(
                chunk.message.tool_calls, emitted_tool_call_ids
            ):
                emitted_tool_call_ids.add(event.tool_call_id)
                yield event

            if chunk.message.reasoning_content:
                yield ReasoningEvent(
                    content=chunk.message.reasoning_content,
                    message_id=reasoning_message_id,
                )

            if chunk.message.content:
                yield AssistantEvent(
                    content=chunk.message.content, message_id=message_id
                )

    async def _get_assistant_event(self) -> AssistantEvent:
        llm_result = await self._chat()
        return AssistantEvent(
            content=llm_result.message.content or "",
            message_id=llm_result.message.message_id,
        )

    async def _emit_failed_tool_events(
        self, failed_calls: list[FailedToolCall]
    ) -> AsyncGenerator[ToolResultEvent]:
        for failed in failed_calls:
            error_msg = f"<{TOOL_ERROR_TAG}>{failed.tool_name}: {failed.error}</{TOOL_ERROR_TAG}>"
            yield ToolResultEvent(
                tool_name=failed.tool_name,
                tool_class=None,
                error=error_msg,
                tool_call_id=failed.call_id,
            )
            self.stats.tool_calls_failed += 1
            self.messages.add(
                self.format_handler.create_failed_tool_response_message(
                    failed, error_msg
                )
            )

    async def _process_one_tool_call(
        self, tool_call: ResolvedToolCall
    ) -> AsyncGenerator[ToolResultEvent | ToolStreamEvent]:
        async for event in self._execute_tool_call(tool_call):
            yield event

    async def _execute_tool_call(
        self, tool_call: ResolvedToolCall
    ) -> AsyncGenerator[ToolResultEvent | ToolStreamEvent]:
        try:
            tool_instance = self.tool_manager.get(tool_call.tool_name)
        except Exception as exc:
            error_msg = f"Error getting tool '{tool_call.tool_name}': {exc}"
            yield self._tool_failure_event(tool_call, error_msg)
            return

        decision: ToolDecision | None = None
        try:
            decision = await self._should_execute_tool(
                tool_instance, tool_call.validated_args, tool_call.call_id
            )

            if decision.verdict == ToolExecutionResponse.SKIP:
                self.stats.tool_calls_rejected += 1
                skip_reason = decision.feedback or str(
                    get_user_cancellation_message(
                        CancellationReason.TOOL_SKIPPED, tool_call.tool_name
                    )
                )
                yield ToolResultEvent(
                    tool_name=tool_call.tool_name,
                    tool_class=tool_call.tool_class,
                    skipped=True,
                    skip_reason=skip_reason,
                    cancelled=f"<{CANCELLATION_TAG}>" in skip_reason,
                    tool_call_id=tool_call.call_id,
                )
                self._handle_tool_response(
                    tool_call, skip_reason, "skipped", decision
                )
                return

            self.stats.tool_calls_agreed += 1

            snapshot = tool_instance.get_file_snapshot(tool_call.validated_args)
            if snapshot is not None:
                self.rewind_manager.add_snapshot(snapshot)
                self.undo_stack.capture(snapshot)

            start_time = time.perf_counter()
            result_model = None
            async for item in tool_instance.invoke(
                ctx=InvokeContext(
                    tool_call_id=tool_call.call_id,
                    agent_manager=self.agent_manager,
                    session_dir=self.session_logger.session_dir,
                    entrypoint_metadata=self.entrypoint_metadata,
                    approval_callback=self.approval_callback,
                    user_input_callback=self.user_input_callback,
                    sampling_callback=self._sampling_handler,
                    plan_file_path=self._plan_session.plan_file_path,
                    switch_agent_callback=self.switch_agent,
                    skill_manager=self.skill_manager,
                    undo_stack=self.undo_stack,
                ),
                **tool_call.args_dict,
            ):
                if isinstance(item, ToolStreamEvent):
                    yield item
                else:
                    result_model = item

            duration = time.perf_counter() - start_time
            if result_model is None:
                raise ToolError("Tool did not yield a result")

            result_dict = result_model.model_dump()
            text = "\n".join(f"{k}: {v}" for k, v in result_dict.items())
            extra = tool_instance.get_result_extra(result_model)
            if extra:
                text += "\n\n" + extra
            self._handle_tool_response(
                tool_call, text, "success", decision, result_dict
            )
            yield ToolResultEvent(
                tool_name=tool_call.tool_name,
                tool_class=tool_call.tool_class,
                result=result_model,
                cancelled=getattr(result_model, "cancelled", False),
                duration=duration,
                tool_call_id=tool_call.call_id,
            )
            self.stats.tool_calls_succeeded += 1

        except asyncio.CancelledError:
            if result_model is not None:
                result_dict = result_model.model_dump()
                text = "\n".join(f"{k}: {v}" for k, v in result_dict.items())
                extra = tool_instance.get_result_extra(result_model)
                if extra:
                    text += "\n\n" + extra
                self._handle_tool_response(tool_call, text, "success", decision, result_dict)
                self.stats.tool_calls_succeeded += 1
                yield ToolResultEvent(
                    tool_name=tool_call.tool_name,
                    tool_class=tool_call.tool_class,
                    result=result_model,
                    cancelled=True,
                    tool_call_id=tool_call.call_id,
                )
            else:
                cancel = str(
                    get_user_cancellation_message(CancellationReason.TOOL_INTERRUPTED)
                )
                self.stats.tool_calls_failed += 1
                yield self._tool_failure_event(tool_call, cancel, decision, cancelled=True)
            raise

        except Exception as exc:
            error_msg = self._classify_tool_failure(
                tool_call, tool_instance, exc, decision
            )
            yield self._tool_failure_event(tool_call, error_msg, decision)

    def _classify_tool_failure(
        self,
        tool_call: ResolvedToolCall,
        tool_instance: BaseTool,
        exc: Exception,
        decision: ToolDecision | None,
    ) -> str:
        """Build the model-facing error for a failed tool call and update stats.

        Distinguishes an internal permission/approval fault (``decision is None``
        — the exception fired before the tool ran) from a genuine tool-execution
        failure. Both log a traceback so a bug can never hide as a bare
        "<tool> failed" message again.
        """
        name = tool_instance.get_name()
        if decision is None:
            logger.exception(
                "Internal error during permission/approval for tool '%s'",
                tool_call.tool_name,
            )
            self.stats.tool_calls_failed += 1
            return (
                f"<{TOOL_ERROR_TAG}>{name}: internal error during permission "
                f"check (see logs): {exc}</{TOOL_ERROR_TAG}>"
            )
        if isinstance(exc, ToolPermissionError):
            self.stats.tool_calls_agreed -= 1
            self.stats.tool_calls_rejected += 1
            return f"<{TOOL_ERROR_TAG}>{name} failed: {exc}</{TOOL_ERROR_TAG}>"
        logger.exception("Tool '%s' failed during execution", tool_call.tool_name)
        self.stats.tool_calls_failed += 1
        return f"<{TOOL_ERROR_TAG}>{name} failed: {exc}</{TOOL_ERROR_TAG}>"

    async def _handle_tool_calls(
        self, resolved: ResolvedMessage
    ) -> AsyncGenerator[ToolCallEvent | ToolResultEvent | ToolStreamEvent]:
        async for event in self._emit_failed_tool_events(resolved.failed_calls):
            yield event
        if not resolved.tool_calls:
            return

        for tool_call in resolved.tool_calls:
            tool_config = self.tool_manager.get_tool_config(tool_call.tool_name)
            yield ToolCallEvent(
                tool_name=tool_call.tool_name,
                tool_class=tool_call.tool_class,
                args=tool_call.validated_args,
                tool_call_id=tool_call.call_id,
                timeout=getattr(tool_config, "default_timeout", None),
            )

        async for event in self._run_tools_concurrently(resolved.tool_calls):
            yield event

    async def _execute_tool_to_queue(
        self,
        tc: ResolvedToolCall,
        queue: asyncio.Queue[ToolCallEvent | ToolResultEvent | ToolStreamEvent | None],
    ) -> None:
        """Run a single tool call, sending events to the queue."""
        async for event in self._process_one_tool_call(tc):
            await queue.put(event)

    async def _run_tools_concurrently(
        self, tool_calls: list[ResolvedToolCall]
    ) -> AsyncGenerator[ToolCallEvent | ToolResultEvent | ToolStreamEvent]:
        """Execute a batch of tool calls.

        Read-only tools run concurrently in arbitrary order. File-mutating tools
        run sequentially in the order the model emitted them, so two writes to
        the same file can never race or be applied out of order. bash counts as
        read-only here: its prompt forbids file edits and routes them to the
        dedicated tools.
        """
        readonly = [tc for tc in tool_calls if not tc.tool_class.mutates_files]
        mutating = [tc for tc in tool_calls if tc.tool_class.mutates_files]

        if readonly:
            async for event in self._run_concurrent_group(readonly):
                yield event
        for tc in mutating:
            async for event in self._process_one_tool_call(tc):
                yield event

    async def _run_concurrent_group(
        self, tool_calls: list[ResolvedToolCall]
    ) -> AsyncGenerator[ToolCallEvent | ToolResultEvent | ToolStreamEvent]:
        """Execute multiple tool calls concurrently, yielding events as they arrive."""
        queue: asyncio.Queue[
            ToolCallEvent | ToolResultEvent | ToolStreamEvent | None
        ] = asyncio.Queue()

        tasks = [
            asyncio.create_task(self._execute_tool_to_queue(tc, queue))
            for tc in tool_calls
        ]

        async def _signal_when_all_done() -> None:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                await queue.put(None)

        monitor = asyncio.create_task(_signal_when_all_done())

        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        except GeneratorExit:
            for t in tasks:
                if not t.done():
                    t.cancel()
            raise
        except asyncio.CancelledError:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            if not monitor.done():
                monitor.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await monitor

    def _handle_tool_response(
        self,
        tool_call: ResolvedToolCall,
        text: str,
        status: Literal["success", "failure", "skipped"],
        decision: ToolDecision | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        # agent steering — Before appending the tool result, check if there are
        # queued steering messages. If so, prepend them to the tool result text
        # in a prominent format:
        #   <user_steering>
        #   The user has provided the following instruction. Acknowledge it and
        #   incorporate it into your work: {steering message}
        #   </user_steering>
        # Then clear the queue. This injects user direction mid-conversation
        # without breaking the LLM/harness cycle.
        if self._steering_queue:
            steering_block = "\n".join(
                f"<user_steering>The user has provided the following instruction. Acknowledge it and incorporate it into your work: {msg}</user_steering>"
                for msg in self._steering_queue
            )
            text = f"{steering_block}\n\n{text}"
            self._steering_queue.clear()

        self.messages.add(
            LLMMessage.model_validate(
                self.format_handler.create_tool_response_message(tool_call, text)
            )
        )

    def _tool_failure_event(
        self,
        tool_call: ResolvedToolCall,
        error_msg: str,
        decision: ToolDecision | None = None,
        cancelled: bool = False,
    ) -> ToolResultEvent:
        """Create a ToolResultEvent for a failed tool and record the failure."""
        self._handle_tool_response(tool_call, error_msg, "failure", decision)
        return ToolResultEvent(
            tool_name=tool_call.tool_name,
            tool_class=tool_call.tool_class,
            error=error_msg,
            cancelled=cancelled,
            tool_call_id=tool_call.call_id,
        )

    async def _chat(
        self, max_tokens: int | None = None, model_override: ModelConfig | None = None
    ) -> LLMChunk:
        active_model = model_override or self.config.get_active_model()
        provider = self.config.get_provider_for_model(active_model)

        available_tools = self.format_handler.get_available_tools(self.tool_manager)
        tool_choice = self.format_handler.get_tool_choice()

        # DEBUG LLM COMMUNICATIONS
        if self.config.llm_debug_dump:
            self._dump_messages_state(active_model, provider, kind="real")
        from privibe.core.llm.backend._debug_dump import (
            set_debug_dump_payload,
            set_dump_context,
        )
        set_debug_dump_payload(self.config.llm_debug_dump)
        set_dump_context(session_id=self.session_id, kind="real")
        logger.info(
            "_chat: about to send — session_id=%s msg_count=%d model=%s provider=%s",
            self.session_id, len(self.messages), active_model.name, provider.name,
        )
        try:
            start_time = time.perf_counter()
            result = await self.backend.complete(
                model=active_model,
                messages=self.messages,
                temperature=active_model.temperature,
                tools=available_tools,
                tool_choice=tool_choice,
                extra_headers=self._get_extra_headers(provider),
                max_tokens=max_tokens,
                metadata=self._build_metadata(),
            )
            end_time = time.perf_counter()

            if result.usage is None:
                raise AgentLoopLLMResponseError(
                    "Usage data missing in non-streaming completion response"
                )
            self._update_stats(usage=result.usage, time_seconds=end_time - start_time)

            processed_message = self.format_handler.process_api_response_message(
                result.message
            )
            self.messages.add(processed_message)
            return LLMChunk(message=processed_message, usage=result.usage)

        except Exception as e:
            if _should_raise_rate_limit_error(e):
                raise RateLimitError(provider.name, active_model.name) from e

            raise RuntimeError(
                f"API error from {provider.name} (model: {active_model.name}): {e}"
            ) from e

    async def _chat_streaming(
        self, max_tokens: int | None = None
    ) -> AsyncGenerator[LLMChunk]:
        active_model = self.config.get_active_model()
        provider = self.config.get_provider_for_model(active_model)

        available_tools = self.format_handler.get_available_tools(self.tool_manager)
        tool_choice = self.format_handler.get_tool_choice()
        # DEBUG LLM COMMUNICATIONS
        # Captures the full message list state before each LLM call.
        # Filenames include session_id + timestamp + monotonic seq to keep
        # ordering correct across rapid turns AND prevent dumps from
        # different privibe instances colliding on `turn{N}_step{N}`.
        # Full message content is dumped (not 200-char preview) so
        # token-level divergence between turns is visible.
        # Controlled by `llm_debug_dump` in config.toml or `/llm-debug` toggle.
        if self.config.llm_debug_dump:
            self._dump_messages_state(active_model, provider, kind="real")
        # DEBUG LLM COMMUNICATIONS
        # Signal the backend dumper to fire on the next prepare_request
        # (covers Anthropic + OpenAI/llamacpp). The session_id + kind tag
        # are read by _debug_dump.dump_payload to namespace the file.
        from privibe.core.llm.backend._debug_dump import (
            set_debug_dump_payload,
            set_dump_context,
        )
        set_debug_dump_payload(self.config.llm_debug_dump)
        set_dump_context(session_id=self.session_id, kind="real")
        # DEBUG LLM COMMUNICATIONS
        logger.info(
            "_chat_streaming: about to send — session_id=%s msg_count=%d model=%s provider=%s",
            self.session_id, len(self.messages), active_model.name, provider.name,
        )
        try:
            start_time = time.perf_counter()
            usage = LLMUsage()
            chunk_agg: LLMChunk | None = None
            async for chunk in self.backend.complete_streaming(
                model=active_model,
                messages=self.messages,
                temperature=active_model.temperature,
                tools=available_tools,
                tool_choice=tool_choice,
                extra_headers=self._get_extra_headers(provider),
                max_tokens=max_tokens,
                metadata=self._build_metadata(),
            ):
                processed_message = self.format_handler.process_api_response_message(
                    chunk.message
                )
                processed_chunk = LLMChunk(message=processed_message, usage=chunk.usage)
                chunk_agg = (
                    processed_chunk
                    if chunk_agg is None
                    else chunk_agg + processed_chunk
                )
                usage += chunk.usage or LLMUsage()
                yield processed_chunk
            end_time = time.perf_counter()

            if chunk_agg is None or chunk_agg.usage is None:
                raise AgentLoopLLMResponseError(
                    "Usage data missing in final chunk of streamed completion"
                )
            self._update_stats(usage=usage, time_seconds=end_time - start_time)

            self.messages.add(chunk_agg.message)

        except Exception as e:
            if _should_raise_rate_limit_error(e):
                raise RateLimitError(provider.name, active_model.name) from e

            raise RuntimeError(
                f"API error from {provider.name} (model: {active_model.name}): {e}"
            ) from e

    def _update_stats(self, usage: LLMUsage, time_seconds: float) -> None:
        self.stats.last_turn_duration = time_seconds
        self.stats.last_turn_prompt_tokens = usage.prompt_tokens
        self.stats.last_turn_completion_tokens = usage.completion_tokens
        self.stats.session_prompt_tokens += usage.prompt_tokens
        self.stats.session_completion_tokens += usage.completion_tokens
        # Prefer the server-reported generation speed (llama.cpp); fall back to the
        # local wall-clock estimate for providers that don't expose timings.
        if usage.tokens_per_second is not None:
            self.stats.tokens_per_second = usage.tokens_per_second
        elif time_seconds > 0 and usage.completion_tokens > 0:
            self.stats.tokens_per_second = usage.completion_tokens / time_seconds
        # Prompt-processing (prefill) speed is server-reported only — there is no
        # local equivalent since we don't separate prefill from decode time.
        if usage.prompt_tokens_per_second is not None:
            self.stats.prompt_tokens_per_second = usage.prompt_tokens_per_second
        # Assign context_tokens last: it triggers the context-bar listener, which
        # should observe the freshly-updated speeds above.
        self.stats.context_tokens = usage.prompt_tokens + usage.completion_tokens

    async def _should_execute_tool(
        self, tool: BaseTool, args: BaseModel, tool_call_id: str
    ) -> ToolDecision:
        if self.auto_approve:
            return ToolDecision(
                verdict=ToolExecutionResponse.EXECUTE,
                approval_type=ToolPermission.ALWAYS,
            )

        tool_name = tool.get_name()
        ctx = tool.resolve_permission(args)

        if ctx is None:
            config_perm = self.tool_manager.get_tool_config(tool_name).permission
            ctx = PermissionContext(permission=config_perm)

        match ctx.permission:
            case ToolPermission.ALWAYS:
                return ToolDecision(
                    verdict=ToolExecutionResponse.EXECUTE,
                    approval_type=ToolPermission.ALWAYS,
                )
            case ToolPermission.NEVER:
                return ToolDecision(
                    verdict=ToolExecutionResponse.SKIP,
                    approval_type=ToolPermission.NEVER,
                    feedback=ctx.reason
                    or f"Tool '{tool_name}' is permanently disabled",
                )
            case _:
                uncovered = [
                    rp
                    for rp in ctx.required_permissions
                    if not self._is_permission_covered(tool_name, rp)
                ]
                if ctx.required_permissions and not uncovered:
                    return ToolDecision(
                        verdict=ToolExecutionResponse.EXECUTE,
                        approval_type=ToolPermission.ALWAYS,
                    )
                return await self._ask_approval(
                    tool_name, args, tool_call_id, uncovered
                )

    async def _ask_approval(
        self,
        tool_name: str,
        args: BaseModel,
        tool_call_id: str,
        required_permissions: list[RequiredPermission],
    ) -> ToolDecision:
        if not self.approval_callback:
            return ToolDecision(
                verdict=ToolExecutionResponse.SKIP,
                approval_type=ToolPermission.ASK,
                feedback="Tool execution not permitted.",
            )
        response, feedback = await self.approval_callback(
            tool_name, args, tool_call_id, required_permissions
        )

        match response:
            case ApprovalResponse.YES:
                verdict = ToolExecutionResponse.EXECUTE
            case _:
                verdict = ToolExecutionResponse.SKIP

        return ToolDecision(
            verdict=verdict, approval_type=ToolPermission.ASK, feedback=feedback
        )

    def _reset_session(self) -> None:
        self.session_id = str(uuid4())
        self.session_logger.reset_session(self.session_id)
        # Covers the rewind path, which suppresses message reset hooks but ends
        # by calling _reset_session; stale undo versions must not survive it.
        self.undo_stack.clear()

    async def clear_history(self) -> None:
        await self.session_logger.save_interaction(
            self.messages,
            self.stats,
            self._base_config,
            self.tool_manager,
            self.agent_profile,
        )
        self.messages.rewind(len(self.messages) - 1)

        self.stats = AgentStats.create_fresh(self.stats)
        self.stats.trigger_listeners()

        try:
            active_model = self.config.get_active_model()
            self.stats.update_pricing(
                active_model.input_price, active_model.output_price
            )
        except ValueError:
            pass

        self.middleware_pipeline.reset()
        self.tool_manager.reset_all()
        self._reset_session()

    async def compact(self) -> str:
        try:
            await self.messages.save()
            summary_request = UtilityPrompt.COMPACT.read()
            self.stats.steps += 1

            with self.messages.silent():
                self.messages.add(
                    LLMMessage(role=Role.user, content=summary_request)
                )
                summary_result = await self._chat(
                    model_override=self.config.get_compaction_model()
                )

            if summary_result.usage is None:
                raise AgentLoopLLMResponseError(
                    "Usage data missing in compaction summary response"
                )
            summary_content = summary_result.message.content or ""

            # After the silent block the list is [system, ...history..., request, response].
            # Rewind everything except the system message, then add the compact summary.
            # The summary is added silently: it is internal state, not a user turn.
            self.messages.rewind(len(self.messages) - 1)
            with self.messages.silent():
                self.messages.add(LLMMessage(role=Role.user, content=summary_content))

            active_model = self.config.get_active_model()
            provider = self.config.get_provider_for_model(active_model)

            actual_context_tokens = await self.backend.count_tokens(
                model=active_model,
                messages=self.messages,
                tools=self.format_handler.get_available_tools(self.tool_manager),
                extra_headers={"user-agent": get_user_agent(provider.backend)},
                metadata=self._build_metadata(),
            )

            self.stats.context_tokens = actual_context_tokens

            self._reset_session()
            await self.session_logger.save_interaction(
                self.messages,
                self.stats,
                self._base_config,
                self.tool_manager,
                self.agent_profile,
            )

            self.middleware_pipeline.reset(reset_reason=ResetReason.COMPACT)

            return summary_content or ""

        except Exception:
            await self.session_logger.save_interaction(
                self.messages,
                self.stats,
                self._base_config,
                self.tool_manager,
                self.agent_profile,
            )
            raise

    async def switch_agent(self, agent_name: str) -> None:
        if agent_name == self.agent_profile.name:
            return
        self.agent_manager.switch_profile(agent_name)
        await self.reload_with_initial_messages(reset_middleware=False)

    async def reload_with_initial_messages(
        self,
        base_config: VibeConfig | None = None,
        max_turns: int | None = None,
        max_price: float | None = None,
        reset_middleware: bool = True,
    ) -> None:
        self._cancel_preflight_warmup()

        # Force an immediate yield to allow the UI to update before heavy sync work.
        # When there are no messages, save_interaction returns early without any await,
        # so the coroutine would run synchronously through ToolManager, SkillManager,
        # and system prompt generation without yielding control to the event loop.
        await asyncio.sleep(0)

        await self.session_logger.save_interaction(
            self.messages,
            self.stats,
            self._base_config,
            self.tool_manager,
            self.agent_profile,
        )

        if base_config is not None:
            self._base_config = base_config
            self.agent_manager.invalidate_config()

        self.backend = self.backend_factory()

        if max_turns is not None:
            self._max_turns = max_turns
        if max_price is not None:
            self._max_price = max_price

        self.tool_manager = ToolManager(
            lambda: self.config, mcp_registry=self.mcp_registry
        )
        self.skill_manager = SkillManager(lambda: self.config)

        # Rebuild the system prompt and replay the conversation silently so the
        # observer (UI) is not re-notified for already-displayed messages.
        # Suppress reset hooks so that rewind checkpoints survive agent/model switches.
        non_system = list(self.messages[1:])
        if not non_system:
            self.stats.reset_context_state()

        with self.messages.silent(), self.messages.no_reset_hooks():
            self.messages.rewind(len(self.messages))
            system_prompt = get_universal_system_prompt(
                self.tool_manager, self.config, self.skill_manager, self.agent_manager
            )
            self.messages.add(LLMMessage(role=Role.system, content=system_prompt))
            for msg in non_system:
                self.messages.add(msg)

        try:
            active_model = self.config.get_active_model()
            self.stats.update_pricing(
                active_model.input_price, active_model.output_price
            )
        except ValueError:
            pass

        if reset_middleware:
            self._setup_middleware()

        self.start_preflight_warmup_if_enabled()
