from __future__ import annotations

from typing import TYPE_CHECKING

from rich import print as rprint

from privibe.core.session.resume_sessions import short_session_id
from privibe.core.session.session_loader import SessionLoader
from privibe.core.types import AgentStats

if TYPE_CHECKING:
    from privibe.core.session.session_logger import SessionLogger


def resumable_session_id(session_logger: SessionLogger) -> str | None:
    """Short id to offer for --resume, or None when there is nothing on
    disk to resume (logging off, or no message was ever saved).
    """
    if not session_logger.enabled or not session_logger.session_id:
        return None
    session_path = SessionLoader.does_session_exist(
        session_logger.session_id, session_logger.session_config
    )
    if session_path is None:
        return None
    return short_session_id(session_logger.session_id)


def format_session_usage(stats: AgentStats) -> str:
    return (
        "Total tokens used this session: "
        f"input={stats.session_prompt_tokens:,} "
        f"output={stats.session_completion_tokens:,} "
        f"(total={stats.session_total_llm_tokens:,})"
    )


def print_session_resume_message(
    session_id: str | None, stats: AgentStats, *, plain: bool = False
) -> None:
    """Print usage and the resume commands. plain=True drops the rich
    markup for console mode, which prints no colors.
    """
    if not session_id:
        return

    print()
    print(format_session_usage(stats))
    print()
    if plain:
        print("To continue this session, run: privibe --continue")
        print(f"Or: privibe --resume {session_id}")
        return
    rprint("To continue this session, run: [bold dark_orange]privibe --continue[/]")
    rprint(f"Or: [bold dark_orange]privibe --resume {session_id}[/]")
