from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
import getpass
import json
import logging
import os
from pathlib import Path
import subprocess
from threading import Lock
from typing import TYPE_CHECKING, Any

from anyio import NamedTemporaryFile, Path as AsyncPath

from privibe.core.session.session_loader import (
    MESSAGES_FILENAME,
    METADATA_FILENAME,
    SessionLoader,
)
from privibe.core.types import AgentStats, LLMMessage, Role, SessionMetadata
from privibe.core.utils import is_windows, utc_now
from privibe.core.utils.io import read_safe_async

if TYPE_CHECKING:
    from privibe.core.agents.models import AgentProfile
    from privibe.core.config import SessionLoggingConfig, VibeConfig
    from privibe.core.tools.manager import ToolManager


TMP_CLEANUP_INTERVAL = timedelta(seconds=5)

# Bounded retry for os.replace on Windows: antivirus / indexers briefly hold
# freshly written files open, which turns the rename into a transient
# PermissionError (WinError 5). Exhausting the retries still raises.
_REPLACE_ATTEMPTS = 5
_REPLACE_BACKOFF_S = 0.1

logger = logging.getLogger(__name__)


class SessionLogger:
    def __init__(self, session_config: SessionLoggingConfig, session_id: str) -> None:
        self.session_config = session_config
        self.enabled = session_config.enabled
        self._last_tmp_cleanup_at: datetime | None = None
        self._tmp_cleanup_lock = Lock()
        # Serializes save_interaction bodies. Two overlapping saves (e.g. an
        # end-of-turn save racing a mode-switch save) would interleave their
        # read-modify-replace cycles on meta.json; on Windows a reader holding
        # the file open makes the other writer's os.replace fail (WinError 5).
        self._save_lock = asyncio.Lock()

        if not self.enabled:
            self.save_dir: Path | None = None
            self.session_prefix: str | None = None
            self.session_id: str = "disabled"
            self.session_start_time: str = "N/A"
            self.session_dir: Path | None = None
            self.session_metadata: SessionMetadata | None = None
            return

        self.save_dir = Path(session_config.save_dir)
        self.session_prefix = session_config.session_prefix
        self.session_id = session_id
        self.session_start_time = utc_now().isoformat()

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir = self.save_folder
        self.session_metadata = self._initialize_session_metadata()

    @property
    def save_folder(self) -> Path:
        if self.save_dir is None or self.session_prefix is None:
            raise RuntimeError(
                "Cannot get session save folder when logging is disabled"
            )

        timestamp = utc_now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{self.session_prefix}_{timestamp}_{self.session_id[:8]}"
        return self.save_dir / folder_name

    @property
    def metadata_filepath(self) -> Path:
        if self.session_dir is None:
            raise RuntimeError(
                "Cannot get session metadata filepath when logging is disabled"
            )
        return self.session_dir / METADATA_FILENAME

    @property
    def messages_filepath(self) -> Path:
        if self.session_dir is None:
            raise RuntimeError(
                "Cannot get session messages filepath when logging is disabled"
            )
        return self.session_dir / MESSAGES_FILENAME

    @property
    def git_commit(self) -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                stdin=subprocess.DEVNULL if is_windows() else None,
                text=True,
                timeout=5.0,
            )
            if result.returncode == 0 and result.stdout:
                return result.stdout.strip()
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            pass
        return None

    @property
    def git_branch(self) -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                stdin=subprocess.DEVNULL if is_windows() else None,
                text=True,
                timeout=5.0,
            )
            if result.returncode == 0 and result.stdout:
                return result.stdout.strip()
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            pass
        return None

    @property
    def username(self) -> str:
        try:
            return getpass.getuser()
        except Exception:
            return "unknown"

    def _initialize_session_metadata(self) -> SessionMetadata:
        git_commit = self.git_commit
        git_branch = self.git_branch
        user_name = self.username

        return SessionMetadata(
            session_id=self.session_id,
            start_time=self.session_start_time,
            end_time=None,
            git_commit=git_commit,
            git_branch=git_branch,
            username=user_name,
            environment={"working_directory": str(Path.cwd())},
        )

    def _get_title(self, messages: Sequence[LLMMessage]) -> str:
        first_user_message = None
        for message in messages:
            if message.role == Role.user:
                first_user_message = message
                break

        if first_user_message is None:
            title = "Untitled session"
        else:
            MAX_TITLE_LENGTH = 50
            text = str(first_user_message.content)
            title = text[:MAX_TITLE_LENGTH]
            if len(text) > MAX_TITLE_LENGTH:
                title += "…"

        return title

    @staticmethod
    async def _replace_with_retry(src: Path, dest: str) -> None:
        """os.replace with a bounded backoff retry on PermissionError.

        On Windows, AV scanners and indexers briefly hold freshly written
        files open, failing the rename with WinError 5. A short retry rides
        that out; exhausting the attempts re-raises the last error.
        """
        for attempt in range(1, _REPLACE_ATTEMPTS + 1):
            try:
                os.replace(src, dest)
                return
            except PermissionError:
                if attempt == _REPLACE_ATTEMPTS:
                    raise
                logger.warning(
                    "os.replace(%s -> %s) hit PermissionError (attempt %d/%d); "
                    "retrying in %.1fs",
                    src,
                    dest,
                    attempt,
                    _REPLACE_ATTEMPTS,
                    _REPLACE_BACKOFF_S,
                )
                await asyncio.sleep(_REPLACE_BACKOFF_S)

    @staticmethod
    async def persist_metadata(metadata: Any, session_dir: Path) -> None:
        temp_metadata_filepath = None
        metadata_filepath = session_dir / METADATA_FILENAME
        try:
            async with NamedTemporaryFile(
                mode="w",
                suffix=".json.tmp",
                dir=str(session_dir),
                delete=False,
                encoding="utf-8",
            ) as f:
                temp_metadata_filepath = Path(str(f.name))
                await f.write(json.dumps(metadata, indent=2, ensure_ascii=False))
                await f.flush()
                os.fsync(f.wrapped.fileno())

            await SessionLogger._replace_with_retry(
                temp_metadata_filepath, str(metadata_filepath)
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to persist session metadata to {metadata_filepath}: {e}"
            ) from e
        finally:
            if (
                temp_metadata_filepath
                and temp_metadata_filepath.exists()
                and temp_metadata_filepath.is_file()
            ):
                temp_metadata_filepath.unlink()

    @staticmethod
    async def persist_messages(messages: list[dict], session_dir: Path) -> None:
        messages_filepath = session_dir / "messages.jsonl"
        try:
            if not messages_filepath.exists():
                messages_filepath.touch()

            async with await AsyncPath(messages_filepath).open(
                "a", encoding="utf-8"
            ) as f:
                for message in messages:
                    await f.write(json.dumps(message, ensure_ascii=False) + "\n")
                    await f.flush()
                    os.fsync(f.wrapped.fileno())
        except Exception as e:
            raise RuntimeError(
                f"Failed to persist session messages to {messages_filepath}: {e}"
            ) from e

    def _count_persisted_messages(self) -> int:
        """Number of messages already durably written to messages.jsonl.

        The jsonl file is the source of truth for what is persisted. Trusting
        meta.json's total_messages instead re-appends (duplicates) messages
        after a crash that landed between the two writes: persist_messages
        succeeds first, then persist_metadata fails, leaving meta.json one or
        more messages behind the jsonl.
        """
        if not self.messages_filepath.exists():
            return 0
        count = 0
        with self.messages_filepath.open("rb") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    async def save_interaction(
        self,
        messages: Sequence[LLMMessage],
        stats: AgentStats,
        base_config: VibeConfig,
        tool_manager: ToolManager,
        agent_profile: AgentProfile,
    ) -> None:
        if not self.enabled or self.session_dir is None:
            return

        if self.session_metadata is None:
            return

        if not any(msg.role != Role.system for msg in messages):
            return

        async with self._save_lock:
            await self._save_interaction_locked(
                messages, stats, base_config, tool_manager, agent_profile
            )

    async def _save_interaction_locked(
        self,
        messages: Sequence[LLMMessage],
        stats: AgentStats,
        base_config: VibeConfig,
        tool_manager: ToolManager,
        agent_profile: AgentProfile,
    ) -> None:
        # If the session directory does not exist, create it
        try:
            self.session_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise RuntimeError(
                f"Failed to create session directory at {self.session_dir}: {type(e).__name__}: {e}"
            ) from e

        # The jsonl line count decides what still needs appending; meta.json's
        # total_messages is only compared against it to surface divergence
        # left behind by a crash between the two writes.
        try:
            old_total_messages = self._count_persisted_messages()
        except OSError as e:
            raise RuntimeError(
                f"Failed to read session messages at {self.messages_filepath}: {e}"
            ) from e

        needs_meta_reconcile = False
        try:
            if self.metadata_filepath.exists():
                raw = await read_safe_async(self.metadata_filepath)
                old_metadata = json.loads(raw)
                meta_total = old_metadata.get("total_messages")
                if meta_total is not None and meta_total != old_total_messages:
                    needs_meta_reconcile = True
                    logger.warning(
                        "Session %s: meta.json says %s persisted messages but "
                        "messages.jsonl holds %d (a previous save was likely "
                        "interrupted). Using the jsonl count; meta.json will "
                        "be reconciled by this save.",
                        self.session_id,
                        meta_total,
                        old_total_messages,
                    )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Failed to read session metadata at {self.metadata_filepath}: {e}"
            ) from e

        try:
            non_system_messages = [m for m in messages if m.role != Role.system]
            # Append new messages
            new_messages = non_system_messages[old_total_messages:]

            if len(new_messages) == 0 and not needs_meta_reconcile:
                return

            messages_data = [m.model_dump(exclude_none=True) for m in new_messages]
            await SessionLogger.persist_messages(messages_data, self.session_dir)

            # If message update succeeded, write metadata
            tools_available = [
                {
                    "type": "function",
                    "function": {
                        "name": tool_class.get_name(),
                        "description": tool_class.description,
                        "parameters": tool_class.get_parameters(),
                    },
                }
                for tool_class in tool_manager.available_tools.values()
            ]

            title = self._get_title(messages)
            system_prompt = (
                messages[0].model_dump()
                if len(messages) > 0 and messages[0].role == Role.system
                else None
            )
            total_messages = len(non_system_messages)

            metadata_dump = {
                **self.session_metadata.model_dump(),
                "end_time": utc_now().isoformat(),
                "stats": stats.model_dump(),
                "title": title,
                "total_messages": total_messages,
                "tools_available": tools_available,
                "config": base_config.model_dump(mode="json"),
                "agent_profile": {
                    "name": agent_profile.name,
                    "overrides": agent_profile.overrides,
                },
                "system_prompt": system_prompt,
            }

            await SessionLogger.persist_metadata(metadata_dump, self.session_dir)
        except Exception as e:
            raise RuntimeError(
                f"Failed to save session to {self.session_dir}: {e}"
            ) from e
        finally:
            self.maybe_cleanup_tmp_files()

    def reset_session(self, session_id: str) -> None:
        """Clear existing session info and setup a new session"""
        if not self.enabled:
            return

        self.session_id = session_id
        self.session_start_time = utc_now().isoformat()
        self.session_dir = self.save_folder
        self.session_metadata = self._initialize_session_metadata()

    def resume_existing_session(self, session_id: str, session_dir: Path) -> None:
        if not self.enabled:
            return

        self.session_id = session_id
        self.session_dir = session_dir
        self.session_metadata = SessionLoader.load_metadata(session_dir)

        if self.session_metadata.start_time:
            self.session_start_time = self.session_metadata.start_time

    def cleanup_tmp_files(self) -> None:
        """Delete temporary files created more than 5 minutes ago"""
        if not self.enabled or not self.save_dir:
            return

        now = utc_now()
        ago = now - timedelta(minutes=5)

        tmp_files = self.save_dir.glob("**/*.json.tmp")  # Recursive search

        for file_path in tmp_files:
            if file_path.is_file():
                try:
                    file_mtime = datetime.fromtimestamp(
                        file_path.stat().st_mtime, tz=UTC
                    )
                    if file_mtime < ago:
                        file_path.unlink()
                except Exception:
                    continue

    def maybe_cleanup_tmp_files(self) -> None:
        if not self.enabled or not self.save_dir:
            return

        if not self._tmp_cleanup_lock.acquire(blocking=False):
            return
        try:
            now = utc_now()
            if (
                self._last_tmp_cleanup_at is not None
                and now - self._last_tmp_cleanup_at < TMP_CLEANUP_INTERVAL
            ):
                return

            self.cleanup_tmp_files()
            self._last_tmp_cleanup_at = now
        finally:
            self._tmp_cleanup_lock.release()
