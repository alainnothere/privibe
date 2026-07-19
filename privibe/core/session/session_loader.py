from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from privibe.core.types import LLMMessage, SessionMetadata
from privibe.core.utils.io import read_safe
from privibe.core.utils.tags import strip_known_tags

if TYPE_CHECKING:
    from privibe.core.config import SessionLoggingConfig


METADATA_FILENAME = "meta.json"
MESSAGES_FILENAME = "messages.jsonl"

_SEARCHABLE_ROLES = frozenset({"user", "assistant"})


class SessionInfo(TypedDict):
    session_id: str
    cwd: str
    title: str | None
    end_time: str | None
    session_path: str


@dataclass(frozen=True)
class SessionScan:
    """One session read in a single pass over its metadata and messages."""

    info: SessionInfo
    preview: list[tuple[str, str]] | None
    search_text: str | None


class SessionLoader:
    @staticmethod
    def _is_valid_session(session_dir: Path) -> bool:
        """Check if a session directory contains valid metadata and messages."""
        metadata_path = session_dir / METADATA_FILENAME
        messages_path = session_dir / MESSAGES_FILENAME

        if not metadata_path.is_file() or not messages_path.is_file():
            return False

        try:
            with metadata_path.open("r", encoding="utf-8", errors="ignore") as f:
                metadata = json.load(f)
            if not isinstance(metadata, dict):
                return False

            with messages_path.open("r", encoding="utf-8", errors="ignore") as f:
                has_messages = False
                for line in f:
                    has_messages = True
                    message = json.loads(line)
                    if not isinstance(message, dict):
                        return False
            if not has_messages:
                return False
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False

        return True

    @staticmethod
    def latest_session(session_dirs: list[Path]) -> Path | None:
        sessions_with_mtime: list[tuple[Path, float]] = []
        for session in session_dirs:
            messages_path = session / MESSAGES_FILENAME
            if not messages_path.is_file():
                continue
            try:
                mtime = messages_path.stat().st_mtime
                sessions_with_mtime.append((session, mtime))
            except OSError:
                continue

        if not sessions_with_mtime:
            return None

        sessions_with_mtime.sort(key=lambda x: x[1], reverse=True)

        for session, _mtime in sessions_with_mtime:
            if SessionLoader._is_valid_session(session):
                return session

        return None

    @staticmethod
    def find_latest_session(config: SessionLoggingConfig) -> Path | None:
        save_dir = Path(config.save_dir)
        if not save_dir.exists():
            return None

        pattern = f"{config.session_prefix}_*"
        session_dirs = list(save_dir.glob(pattern))

        return SessionLoader.latest_session(session_dirs)

    @staticmethod
    def find_session_by_id(
        session_id: str, config: SessionLoggingConfig
    ) -> Path | None:
        matches = SessionLoader._find_session_dirs_by_short_id(session_id, config)

        return SessionLoader.latest_session(matches)

    @staticmethod
    def does_session_exist(
        session_id: str, config: SessionLoggingConfig
    ) -> Path | None:
        for session_dir in SessionLoader._find_session_dirs_by_short_id(
            session_id, config
        ):
            if (session_dir / MESSAGES_FILENAME).is_file():
                return session_dir
        return None

    @staticmethod
    def _find_session_dirs_by_short_id(
        session_id: str, config: SessionLoggingConfig
    ) -> list[Path]:
        save_dir = Path(config.save_dir)
        if not save_dir.exists():
            return []

        short_id = session_id[:8]
        return list(save_dir.glob(f"{config.session_prefix}_*_{short_id}"))

    @staticmethod
    def _convert_to_utc_iso(date_str: str) -> str:
        dt = datetime.fromisoformat(date_str)
        if dt.tzinfo is None:
            dt = dt.astimezone()
        utc_dt = dt.astimezone(UTC)
        return utc_dt.isoformat()

    @staticmethod
    def _raw_content_text(value: Any) -> str | None:
        """Flatten a raw stored ``content`` value the way LLMMessage would."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for part in value:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                else:
                    parts.append(str(part))
            return "\n".join(parts)
        return str(value)

    @staticmethod
    def _scan_messages(
        messages_path: Path, collect_text: bool, preview_lines: int
    ) -> tuple[list[tuple[str, str]] | None, list[str] | None] | None:
        """Read messages.jsonl once, returning None if it is not a valid session."""
        preview: list[tuple[str, str]] | None = [] if preview_lines > 0 else None
        search_parts: list[str] | None = [] if collect_text else None
        has_messages = False

        with messages_path.open("r", encoding="utf-8") as f:
            for line in f:
                has_messages = True
                message = json.loads(line)
                if not isinstance(message, dict):
                    return None

                if preview is None and search_parts is None:
                    continue

                role = message.get("role")
                if role not in _SEARCHABLE_ROLES:
                    continue

                content = SessionLoader._raw_content_text(message.get("content"))
                if not content:
                    continue

                if search_parts is not None and not message.get("injected"):
                    # Harness-injected text (context refreshes, warnings) would
                    # otherwise match every session.
                    searchable = strip_known_tags(content).strip()
                    if searchable:
                        search_parts.append(searchable)
                if preview is not None:
                    preview.append((str(role), SessionLoader._clean_text(content)))
                    del preview[:-preview_lines]

        if not has_messages:
            return None

        return preview, search_parts

    @staticmethod
    def _scan_session_dir(
        session_dir: Path,
        cwd: str | None,
        collect_text: bool,
        preview_lines: int,
    ) -> SessionScan | None:
        """Read one session's metadata and messages once, or None if unusable."""
        metadata_path = session_dir / METADATA_FILENAME
        messages_path = session_dir / MESSAGES_FILENAME

        if not metadata_path.is_file() or not messages_path.is_file():
            return None

        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
            if not isinstance(metadata, dict):
                return None

            session_id = metadata.get("session_id")
            if not session_id:
                return None

            session_cwd = metadata.get("environment", {}).get("working_directory", "")
            if cwd is not None and session_cwd != cwd:
                return None

            scanned = SessionLoader._scan_messages(
                messages_path, collect_text, preview_lines
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None

        if scanned is None:
            return None
        preview, search_parts = scanned

        end_time = metadata.get("end_time")
        if end_time:
            try:
                end_time = SessionLoader._convert_to_utc_iso(end_time)
            except (ValueError, OSError):
                end_time = None

        title = metadata.get("title")
        search_text: str | None = None
        if search_parts is not None:
            if title:
                search_parts.append(str(title))
            search_text = "\n".join(search_parts).lower()

        info: SessionInfo = {
            "session_id": session_id,
            "cwd": session_cwd,
            "title": title,
            "end_time": end_time,
            "session_path": str(messages_path),
        }
        return SessionScan(info=info, preview=preview, search_text=search_text)

    @staticmethod
    def _scan_sessions(
        config: SessionLoggingConfig,
        cwd: str | None = None,
        collect_text: bool = False,
        preview_lines: int = 0,
    ) -> list[SessionScan]:
        save_dir = Path(config.save_dir)
        if not save_dir.exists():
            return []

        # Non-recursive: nested agents/ subdirs have their own messages.jsonl.
        pattern = f"{config.session_prefix}_*"

        scans: list[SessionScan] = []
        for session_dir in save_dir.glob(pattern):
            scan = SessionLoader._scan_session_dir(
                session_dir, cwd, collect_text, preview_lines
            )
            if scan is not None:
                scans.append(scan)

        return scans

    @staticmethod
    def list_sessions(
        config: SessionLoggingConfig, cwd: str | None = None
    ) -> list[SessionInfo]:
        return [scan.info for scan in SessionLoader._scan_sessions(config, cwd)]

    @staticmethod
    def load_metadata(session_dir: Path) -> SessionMetadata:
        metadata_path = session_dir / METADATA_FILENAME
        if not metadata_path.exists():
            raise ValueError(f"Session metadata not found at {session_dir}")

        try:
            metadata_content = read_safe(metadata_path)
            return SessionMetadata.model_validate_json(metadata_content)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(
                f"Failed to load session metadata at {session_dir}: {e}"
            ) from e

    @staticmethod
    def load_session(filepath: Path) -> tuple[list[LLMMessage], dict[str, Any]]:
        # Load session messages from MESSAGES_FILENAME
        messages_filepath = filepath / MESSAGES_FILENAME

        try:
            content = read_safe(messages_filepath).split("\n")
            if content and content[-1] == "":
                content.pop()
        except Exception as e:
            raise ValueError(
                f"Error reading session messages at {filepath}: {e}"
            ) from e

        if not content:
            raise ValueError(
                f"Session messages file is empty (may have been corrupted by interruption): "
                f"{filepath}"
            )

        try:
            data = [json.loads(line) for line in content]
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Session messages contain invalid JSON (may have been corrupted): "
                f"{filepath}\nDetails: {e}"
            ) from e

        messages = [
            LLMMessage.model_validate(msg)
            for msg in data
            if msg.get("role") != "system"
        ]

        # Load session metadata from METADATA_FILENAME
        metadata_filepath = filepath / METADATA_FILENAME

        if metadata_filepath.exists():
            try:
                with metadata_filepath.open(
                    "r", encoding="utf-8", errors="ignore"
                ) as f:
                    metadata = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Session metadata contains invalid JSON (may have been corrupted): "
                    f"{filepath}\nDetails: {e}"
                ) from e
        else:
            metadata = {}

        return messages, metadata

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.strip().replace("\n", " ")
        return text or "(empty message)"
