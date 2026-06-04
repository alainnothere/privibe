from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from privibe.core.config import VibeConfig
from privibe.core.session.session_loader import SessionLoader

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


def list_local_resume_sessions(config: VibeConfig) -> list[ResumeSessionInfo]:
    return [
        ResumeSessionInfo(
            session_id=session["session_id"],
            source="local",
            cwd=session["cwd"],
            title=session.get("title"),
            end_time=session.get("end_time"),
            session_path=session.get("session_path"),
        )
        for session in SessionLoader.list_sessions(config.session_logging, cwd=None)
    ]
