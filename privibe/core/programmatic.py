from __future__ import annotations

import asyncio
from pathlib import Path

from privibe import __version__
from privibe.core.agent_loop import AgentLoop
from privibe.core.agents.models import BuiltinAgentName
from privibe.core.config import VibeConfig
from privibe.core.logger import logger
from privibe.core.output_formatters import create_formatter
from privibe.core.types import (
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


def run_programmatic(
    config: VibeConfig,
    prompt: str,
    max_turns: int | None = None,
    max_price: float | None = None,
    output_format: OutputFormat = OutputFormat.TEXT,
    session_path: Path | None = None,
    previous_messages: list[LLMMessage] | None = None,
    agent_name: str = BuiltinAgentName.AUTO_APPROVE,
    client_metadata: ClientMetadata = _DEFAULT_CLIENT_METADATA,
) -> str | None:
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
            content = build_context_refresh_content(agent_loop.config)
            agent_loop.messages.add(
                LLMMessage(role=Role.user, content=content, injected=True)
            )
            logger.info(
                "Loaded %d previous messages", len(non_system)
            )

        async for event in agent_loop.act(prompt):
            formatter.on_event(event)
            if isinstance(event, AssistantEvent) and event.stopped_by_middleware:
                raise ConversationLimitException(event.content)

        return formatter.finalize()

    return asyncio.run(_async_run())
