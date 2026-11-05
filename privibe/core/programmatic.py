from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from privibe import __version__
from privibe.core.agent_loop import AgentLoop
from privibe.core.agents.models import BuiltinAgentName
from privibe.core.config import VibeConfig
from privibe.core.conversation import is_step_budget_warning
from privibe.core.logger import logger
from privibe.core.output_formatters import create_formatter
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

__all__ = ["run_programmatic"]

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
) -> str | None:
    """`nanny_alias` (--nanny) puts that model in the approver's seat: calls
    the agent profile would ask about go to the nanny, and whatever it does
    not approve is refused. Without it nothing here answers an approval
    prompt, which is why -p defaults to the auto-approve agent.
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

    async def _async_run() -> str | None:
        ctx_msg = await agent_loop.resolve_context_size()
        if ctx_msg:
            logger.warning(ctx_msg)
        from privibe.core.config.migration import pop_pending_message
        migration_msg = pop_pending_message()
        if migration_msg:
            logger.warning(migration_msg)
        if session_path is not None:
            agent_loop.messages.restore(session_path)
            logger.info(
                "Loaded %d messages from previous session", len(agent_loop.messages)
            )
        elif previous_messages is not None:
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
            agent_loop.messages.add(
                LLMMessage(role=Role.user, content=content, injected=True)
            )
            logger.info(
                "Loaded %d previous messages", len(non_system)
            )

        nanny = None
        if nanny_alias:
            from privibe.core.nanny import NannyConfigError, NannyManager

            try:
                nanny = NannyManager.for_loop(agent_loop, alias=nanny_alias)
            except NannyConfigError as exc:
                raise RuntimeError(f"nanny not started: {exc}") from exc
            agent_loop.set_nanny(nanny, owner=True)
            agent_loop.set_approval_callback(_refuse_without_a_human)
            await agent_loop.set_session_nanny_model(nanny.model_alias)

        try:
            async for event in agent_loop.act(prompt):
                formatter.on_event(event)
                if isinstance(event, AssistantEvent) and event.stopped_by_middleware:
                    raise ConversationLimitException(event.content)
        finally:
            if nanny is not None:
                await nanny.close()

        return formatter.finalize()

    return asyncio.run(_async_run())
