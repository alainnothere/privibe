from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum, auto
import json
from string import Template
from typing import TYPE_CHECKING, Any, Protocol

from privibe.core.agents import AgentProfile
from privibe.core.prompts import UtilityPrompt
from privibe.core.types import LLMMessage, Role
from privibe.core.utils import VIBE_WARNING_TAG

if TYPE_CHECKING:
    from privibe.core.config import VibeConfig
    from privibe.core.conversation import ConversationList
    from privibe.core.types import AgentStats

# Metadata key on an INJECT_MESSAGE result: the agent loop must not execute any
# tool call the model makes in its very next reply. Set by StepBudgetMiddleware
# when it asks the model to write up instead of calling tools.
SUSPEND_TOOLS_METADATA_KEY = "suspend_tools"


class MiddlewareAction(StrEnum):
    CONTINUE = auto()
    STOP = auto()
    COMPACT = auto()
    INJECT_MESSAGE = auto()


class ResetReason(StrEnum):
    STOP = auto()
    COMPACT = auto()


@dataclass
class ConversationContext:
    messages: ConversationList
    stats: AgentStats
    config: VibeConfig
    # Id of the user message that started the current turn. Changes exactly
    # when the user sends something; compaction and injected reminders leave
    # it alone, so per-turn middleware state keys off it.
    turn_id: str | None = None


@dataclass
class MiddlewareResult:
    action: MiddlewareAction = MiddlewareAction.CONTINUE
    message: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ConversationMiddleware(Protocol):
    async def before_turn(self, context: ConversationContext) -> MiddlewareResult: ...

    def reset(self, reset_reason: ResetReason = ResetReason.STOP) -> None: ...


class TurnLimitMiddleware:
    def __init__(self, max_turns: int) -> None:
        self.max_turns = max_turns

    async def before_turn(self, context: ConversationContext) -> MiddlewareResult:
        if context.stats.steps - 1 >= self.max_turns:
            return MiddlewareResult(
                action=MiddlewareAction.STOP,
                reason=f"Turn limit of {self.max_turns} reached",
            )
        return MiddlewareResult()

    def reset(self, reset_reason: ResetReason = ResetReason.STOP) -> None:
        pass


def tool_call_key(name: str | None, arguments: str | None) -> tuple[str, str]:
    """(tool name, canonical arguments) for comparing tool calls across messages.

    Arguments are the raw JSON string the model emitted. Parsing and re-dumping
    with sorted keys makes `{"a":1,"b":2}` and `{"b": 2, "a": 1}` the same call;
    unparseable arguments compare as the raw string.
    """
    raw = arguments or ""
    try:
        canonical = json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":"))
    except (ValueError, TypeError):
        canonical = raw
    return (name or "", canonical)


def message_tool_call_keys(message: LLMMessage) -> set[tuple[str, str]]:
    return {
        tool_call_key(tc.function.name, tc.function.arguments)
        for tc in message.tool_calls or []
        if tc.function is not None
    }


def assistant_messages_this_turn(
    messages: Sequence[LLMMessage], limit: int
) -> list[LLMMessage]:
    """The last `limit` assistant messages of the current turn, newest first.

    Walks back from the tail and stops at the first real (non-injected) user
    message: once the user has spoken, earlier repeats are their business.
    """
    found: list[LLMMessage] = []
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.role == Role.user and not msg.injected:
            break
        if msg.role == Role.assistant:
            found.append(msg)
            if len(found) >= limit:
                break
    return found


class StepBudgetMiddleware:
    """Cap the LLM calls one user message may trigger.

    before_turn runs once per LLM call. Calls 1..max_llm_calls proceed. On the
    next one the model is told, via an injected message, to stop calling tools
    and write up what it has; the result carries SUSPEND_TOOLS_METADATA_KEY so
    the agent loop refuses to execute any tool call in that reply. If the loop
    comes back for yet another call (the model called tools anyway), the turn
    is stopped and the prompt returns to the user. A new user message resets
    the count; compaction does not.

    The limit is read from config on every call, so /llm-calls-per-turn takes
    effect mid-session without rebuilding the pipeline.
    """

    def __init__(self) -> None:
        self._turn_id: str | None = None
        self._calls = 0
        self._summary_requested = False

    async def before_turn(self, context: ConversationContext) -> MiddlewareResult:
        max_llm_calls = context.config.max_llm_calls_per_turn
        if max_llm_calls <= 0:
            return MiddlewareResult()
        if context.turn_id != self._turn_id:
            self._turn_id = context.turn_id
            self._calls = 0
            self._summary_requested = False

        self._calls += 1
        if self._calls <= max_llm_calls:
            return MiddlewareResult()

        if not self._summary_requested:
            self._summary_requested = True
            return MiddlewareResult(
                action=MiddlewareAction.INJECT_MESSAGE,
                message=step_budget_summary_request(max_llm_calls),
                metadata={SUSPEND_TOOLS_METADATA_KEY: True},
            )

        return MiddlewareResult(
            action=MiddlewareAction.STOP,
            reason=(
                f"Step budget spent: {max_llm_calls} LLM calls on this message, "
                "and when asked to write up its findings the model called tools "
                "instead. Send a message to grant it another "
                f"{max_llm_calls}."
            ),
        )

    def reset(self, reset_reason: ResetReason = ResetReason.STOP) -> None:
        # Compaction mid-turn is plumbing, not a new turn: the count survives it.
        if reset_reason == ResetReason.COMPACT:
            return
        self._turn_id = None
        self._calls = 0
        self._summary_requested = False


class RepeatedToolCallMiddleware:
    """Stop the turn when the model issues the same tool call `streak` messages in a row.

    The agent loop already refuses to execute a call identical to one in the
    previous assistant message and tells the model so. A model that repeats it
    anyway is looping, not thinking; this middleware ends the turn before the
    next LLM call so the user gets the prompt back. Only assistant messages of
    the current turn count, so a user re-asking never trips it.
    """

    def __init__(self, streak: int = 3) -> None:
        self.streak = streak

    async def before_turn(self, context: ConversationContext) -> MiddlewareResult:
        recent = assistant_messages_this_turn(context.messages, self.streak)
        if len(recent) < self.streak:
            return MiddlewareResult()
        repeated = set.intersection(*(message_tool_call_keys(m) for m in recent))
        if not repeated:
            return MiddlewareResult()
        name, _args = sorted(repeated)[0]
        return MiddlewareResult(
            action=MiddlewareAction.STOP,
            reason=(
                f"Stopped: the model issued the identical `{name}` call in "
                f"{self.streak} consecutive messages. Send a message to continue."
            ),
        )

    def reset(self, reset_reason: ResetReason = ResetReason.STOP) -> None:
        pass


class PriceLimitMiddleware:
    def __init__(self, max_price: float) -> None:
        self.max_price = max_price

    async def before_turn(self, context: ConversationContext) -> MiddlewareResult:
        if context.stats.session_cost > self.max_price:
            return MiddlewareResult(
                action=MiddlewareAction.STOP,
                reason=f"Price limit exceeded: ${context.stats.session_cost:.4f} > ${self.max_price:.2f}",
            )
        return MiddlewareResult()

    def reset(self, reset_reason: ResetReason = ResetReason.STOP) -> None:
        pass


class AutoCompactMiddleware:
    async def before_turn(self, context: ConversationContext) -> MiddlewareResult:
        threshold = context.config.get_active_model().auto_compact_threshold
        if threshold > 0 and context.stats.context_tokens >= threshold:
            return MiddlewareResult(
                action=MiddlewareAction.COMPACT,
                metadata={
                    "old_tokens": context.stats.context_tokens,
                    "threshold": threshold,
                },
            )
        return MiddlewareResult()

    def reset(self, reset_reason: ResetReason = ResetReason.STOP) -> None:
        pass


class ContextWarningMiddleware:
    def __init__(self, threshold_percent: float = 0.5) -> None:
        self.threshold_percent = threshold_percent
        self.has_warned = False

    async def before_turn(self, context: ConversationContext) -> MiddlewareResult:
        if self.has_warned:
            return MiddlewareResult()

        max_context = context.config.get_active_model().auto_compact_threshold
        if max_context <= 0:
            return MiddlewareResult()

        if context.stats.context_tokens >= max_context * self.threshold_percent:
            self.has_warned = True

            percentage_used = (context.stats.context_tokens / max_context) * 100
            warning_msg = f"<{VIBE_WARNING_TAG}>You have used {percentage_used:.0f}% of your total context ({context.stats.context_tokens:,}/{max_context:,} tokens)</{VIBE_WARNING_TAG}>"

            return MiddlewareResult(
                action=MiddlewareAction.INJECT_MESSAGE, message=warning_msg
            )

        return MiddlewareResult()

    def reset(self, reset_reason: ResetReason = ResetReason.STOP) -> None:
        self.has_warned = False


# Mode reminder/exit texts live in privibe/core/prompts/*.md (locally
# editable via ~/.privibe/prompts/); the warning-tag wrapper is protocol and
# stays here. Read lazily so an edited local copy applies without a restart.


def _wrap_warning(body: str) -> str:
    return f"<{VIBE_WARNING_TAG}>{body}</{VIBE_WARNING_TAG}>"


def make_plan_agent_reminder(plan_file_path: str) -> str:
    template = UtilityPrompt.PLAN_REMINDER.read()
    return _wrap_warning(
        Template(template).safe_substitute(plan_file_path=plan_file_path)
    )


def plan_agent_exit() -> str:
    return _wrap_warning(UtilityPrompt.PLAN_EXIT.read())


def chat_agent_reminder() -> str:
    return _wrap_warning(UtilityPrompt.CHAT_REMINDER.read())


def chat_agent_exit() -> str:
    return _wrap_warning(UtilityPrompt.CHAT_EXIT.read())


def step_budget_summary_request(max_llm_calls: int) -> str:
    template = UtilityPrompt.STEP_BUDGET.read()
    return _wrap_warning(
        Template(template).safe_substitute(max_llm_calls=str(max_llm_calls))
    )


class ReadOnlyAgentMiddleware:
    def __init__(
        self,
        profile_getter: Callable[[], AgentProfile],
        agent_name: str,
        reminder: str | Callable[[], str],
        exit_message: str | Callable[[], str],
    ) -> None:
        self._profile_getter = profile_getter
        self._agent_name = agent_name
        self._reminder = reminder
        self._exit_message = exit_message
        self._was_active = False

    @property
    def reminder(self) -> str:
        return self._reminder() if callable(self._reminder) else self._reminder

    @property
    def exit_message(self) -> str:
        return (
            self._exit_message()
            if callable(self._exit_message)
            else self._exit_message
        )

    def _is_active(self) -> bool:
        return self._profile_getter().name == self._agent_name

    async def before_turn(self, context: ConversationContext) -> MiddlewareResult:
        is_active = self._is_active()
        was_active = self._was_active

        if was_active and not is_active:
            self._was_active = False
            return MiddlewareResult(
                action=MiddlewareAction.INJECT_MESSAGE, message=self.exit_message
            )

        if is_active and not was_active:
            self._was_active = True
            return MiddlewareResult(
                action=MiddlewareAction.INJECT_MESSAGE, message=self.reminder
            )

        self._was_active = is_active
        return MiddlewareResult()

    def reset(self, reset_reason: ResetReason = ResetReason.STOP) -> None:
        self._was_active = False


class MiddlewarePipeline:
    def __init__(self) -> None:
        self.middlewares: list[ConversationMiddleware] = []

    def add(self, middleware: ConversationMiddleware) -> MiddlewarePipeline:
        self.middlewares.append(middleware)
        return self

    def clear(self) -> None:
        self.middlewares.clear()

    def reset(self, reset_reason: ResetReason = ResetReason.STOP) -> None:
        for mw in self.middlewares:
            mw.reset(reset_reason)

    async def run_before_turn(self, context: ConversationContext) -> MiddlewareResult:
        messages_to_inject = []
        # Injected results are merged into one message; their metadata merges
        # too, so a flag like SUSPEND_TOOLS_METADATA_KEY survives the merge.
        merged_metadata: dict[str, Any] = {}

        for mw in self.middlewares:
            result = await mw.before_turn(context)
            if result.action == MiddlewareAction.INJECT_MESSAGE and result.message:
                messages_to_inject.append(result.message)
                merged_metadata.update(result.metadata)
            elif result.action in {MiddlewareAction.STOP, MiddlewareAction.COMPACT}:
                return result
        if messages_to_inject:
            combined_message = "\n\n".join(messages_to_inject)
            return MiddlewareResult(
                action=MiddlewareAction.INJECT_MESSAGE,
                message=combined_message,
                metadata=merged_metadata,
            )

        return MiddlewareResult()
