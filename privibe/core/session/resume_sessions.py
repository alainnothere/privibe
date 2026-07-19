from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from privibe.core.config import VibeConfig
from privibe.core.session.session_loader import SessionInfo, SessionLoader

ResumeSessionSource = Literal["local"]

SHORT_SESSION_ID_LEN = 8


def short_session_id(session_id: str, source: ResumeSessionSource = "local") -> str:
    return session_id[:SHORT_SESSION_ID_LEN]


@dataclass(frozen=True)
class ResumeSessionInfo:
    session_id: str
    source: ResumeSessionSource
    cwd: str
    title: str | None
    end_time: str | None
    status: str | None = None
    session_path: str | None = None

    @property
    def option_id(self) -> str:
        return f"{self.source}:{self.session_id}"


@dataclass(frozen=True)
class ResumeSessionIndex:
    """Sessions plus the per-session data the picker renders and searches."""

    sessions: list[ResumeSessionInfo]
    previews: dict[str, list[tuple[str, str]]]
    search_text: dict[str, str]


def _resume_session_from_info(session: SessionInfo) -> ResumeSessionInfo:
    return ResumeSessionInfo(
        session_id=session["session_id"],
        source="local",
        cwd=session["cwd"],
        title=session.get("title"),
        end_time=session.get("end_time"),
        session_path=session.get("session_path"),
    )


def list_local_resume_sessions(config: VibeConfig) -> list[ResumeSessionInfo]:
    return [
        _resume_session_from_info(session)
        for session in SessionLoader.list_sessions(config.session_logging, cwd=None)
    ]


def build_local_resume_index(
    config: VibeConfig, preview_lines: int
) -> ResumeSessionIndex:
    """Read every local session once, collecting previews and search text."""
    sessions: list[ResumeSessionInfo] = []
    previews: dict[str, list[tuple[str, str]]] = {}
    search_text: dict[str, str] = {}

    for scan in SessionLoader._scan_sessions(
        config.session_logging,
        cwd=None,
        collect_text=True,
        preview_lines=preview_lines,
    ):
        session = _resume_session_from_info(scan.info)
        sessions.append(session)
        previews[session.option_id] = scan.preview or []
        search_text[session.option_id] = scan.search_text or ""

    return ResumeSessionIndex(
        sessions=sessions, previews=previews, search_text=search_text
    )
