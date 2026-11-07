"""`privibe --context` with --continue / --resume ID: how much of the model's
context a saved session uses, from its last save. Nothing is sent to the
model. The only way to ask a server how big a conversation is would be to
send it the conversation; a full one is exactly the case this exists for,
and the server would refuse it. The saved number is what the server said
the last time it did read the conversation.

The ceiling comes from the same probe startup uses (/props, /v1/models) when
the model's backend is generic and auto-detection is on; otherwise it is the
configured auto_compact_threshold, and the line says so.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

from privibe.core.config import VibeConfig
from privibe.core.llm.backend.generic import GenericBackend
from privibe.core.session.resume_sessions import short_session_id

_PROBE_TIMEOUT_S = 2.0

OVERFLOW_ADVICE = (
    "Does not fit: rewind it or raise the server's context size; "
    "compaction cannot summarize what the server cannot read."
)


def format_context_report(
    *,
    session_id: str,
    name: str | None,
    model_alias: str | None,
    context_tokens: int | None,
    max_tokens: int | None,
    max_from_config: bool,
    saved_at: str | None,
) -> str:
    """Two lines: which session, and the number. Pure; tested directly."""
    label = short_session_id(session_id) if session_id else "unknown"
    head = f"Session {label}"
    if name:
        head += f" \"{name}\""
    if model_alias:
        head += f" on {model_alias}"

    when = _format_when(saved_at)
    if context_tokens is None:
        return f"{head}\nContext: unknown (this session never completed a turn)"
    if not max_tokens:
        line = f"Context: {context_tokens:,} tokens (model context size unknown)"
        return f"{head}\n{line}{when}"

    ratio = context_tokens / max_tokens
    source = " (from config)" if max_from_config else ""
    line = (
        f"Context: {context_tokens:,} of {max_tokens:,} tokens{source} "
        f"({ratio:.0%})"
    )
    if context_tokens >= max_tokens:
        return f"{head}\n{line}. {OVERFLOW_ADVICE}"
    return f"{head}\n{line}{when}"


def _format_when(saved_at: str | None) -> str:
    if not saved_at:
        return ""
    try:
        stamp = datetime.fromisoformat(saved_at)
    except ValueError:
        return ""
    return f", as of {stamp.astimezone().strftime('%Y-%m-%d %H:%M')}"


async def detect_max_tokens(config: VibeConfig, model_alias: str) -> tuple[int, bool]:
    """(ceiling, from_config): the server's n_ctx when it can be asked,
    else the configured threshold.
    """
    model = next(m for m in config.models if m.alias == model_alias)
    configured = (model.auto_compact_threshold, True)
    if not config.auto_detect_context_size:
        return configured
    provider = config.get_provider_for_model(model)
    if provider.backend != "generic":
        return configured
    backend = GenericBackend(provider=provider, timeout=_PROBE_TIMEOUT_S)
    try:
        async with asyncio.timeout(_PROBE_TIMEOUT_S):
            info = await backend.fetch_model_endpoint_info(model.name)
    except Exception:
        return configured
    if info.context_size:
        return int(info.context_size), False
    return configured


def _saved_context_tokens(metadata: dict[str, Any]) -> int | None:
    stats = metadata.get("stats")
    if not isinstance(stats, dict):
        return None
    value = stats.get("context_tokens")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def build_context_report(config: VibeConfig, metadata: dict[str, Any]) -> str:
    saved_alias = metadata.get("active_model")
    alias = (
        saved_alias
        if isinstance(saved_alias, str) and config.unusable_model_reason(saved_alias) is None
        else config.active_model
    )
    max_tokens, from_config = asyncio.run(detect_max_tokens(config, alias))
    return format_context_report(
        session_id=str(metadata.get("session_id") or ""),
        name=metadata.get("name"),
        model_alias=alias,
        context_tokens=_saved_context_tokens(metadata),
        max_tokens=max_tokens,
        max_from_config=from_config,
        saved_at=metadata.get("end_time"),
    )


def run_context_query(
    config: VibeConfig, loaded_session: tuple[dict[str, Any], Path] | None
) -> int:
    """Print the report and return the exit code: 0 when the session fits,
    2 when it does not (scripts can branch on it), 1 when nothing was loaded.
    """
    if loaded_session is None:
        print(
            "Error: --context needs a session: privibe --context --resume SESSION_ID "
            "(or --continue for the latest one)",
            file=sys.stderr,
        )
        return 1
    metadata, _ = loaded_session
    report = build_context_report(config, metadata)
    print(report)
    return 2 if OVERFLOW_ADVICE in report else 0
