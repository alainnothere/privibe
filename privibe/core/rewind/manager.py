from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
import os
from pathlib import Path

from privibe.core.conversation import ConversationList
from privibe.core.logger import logger
from privibe.core.types import LLMMessage, Role


class RewindError(Exception):
    """Raised when a rewind operation fails."""


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    """Snapshot of a single file's content at a point in time.

    content is None if the file did not exist (was created after the snapshot).
    """

    path: str
    content: bytes | None


@dataclass
class Checkpoint:
    """Snapshot of tracked files taken before a user message."""

    message_index: int
    files: list[FileSnapshot] = field(default_factory=list)


class RewindManager:
    """Manages conversation rewind: file snapshots, message truncation, and session forking."""

    def __init__(
        self,
        messages: ConversationList,
        save_messages: Callable[[], Awaitable[None]],
        reset_session: Callable[[], None],
    ) -> None:
        self._checkpoints: list[Checkpoint] = []
        self._messages = messages
        self._save_messages = save_messages
        self._reset_session = reset_session
        self._messages.on_reset(self._on_messages_reset)

    # -- Checkpoint management -------------------------------------------------

    @property
    def checkpoints(self) -> list[Checkpoint]:
        return list(self._checkpoints)

    def create_checkpoint(self) -> None:
        """Snapshot known files and start a new checkpoint at the current message position.

        Files known from the previous checkpoint are re-read from disk so
        that each checkpoint captures the actual state at that point in time.
        """
        index = len(self._messages)
        if self._checkpoints and self._checkpoints[-1].message_index == index:
            # Already taken at this slot (a batch boundary followed by a
            # steering message lands here twice): one checkpoint per slot.
            return
        files: list[FileSnapshot] = []
        if self._checkpoints:
            for snap in self._checkpoints[-1].files:
                files.append(self._read_snapshot(snap.path))
        self._checkpoints.append(Checkpoint(message_index=index, files=files))

    def add_snapshot(self, snapshot: FileSnapshot) -> None:
        """Record a file snapshot into every checkpoint that doesn't have it yet."""
        for cp in self._checkpoints:
            if all(s.path != snapshot.path for s in cp.files):
                cp.files.append(snapshot)

    def has_file_changes_at(self, message_index: int) -> bool:
        """Check if files have changed since the checkpoint at *message_index*."""
        checkpoint = self._get_checkpoint(message_index)
        if checkpoint is None:
            return False
        return self._has_changes_since(checkpoint)

    # -- Rewind operations -----------------------------------------------------

    def get_rewindable_messages(self) -> list[tuple[int, str]]:
        """Return (message_index, content) for each user message."""
        return [
            (i, msg.content or "")
            for i, msg in enumerate(self._messages)
            if msg.role == Role.user and msg.content and not msg.injected
        ]

    def batch_boundaries(self) -> list[int]:
        """Indices where a tool batch has just closed: every call of the
        assistant message before it has its result, and the message at the
        index (if any) is not another result. A conversation cut there ends on
        tool results, which is the one slot the loop itself inserts a user
        message mid-turn (steering), so a message appended there is a steering
        message, not a lie about history. Includes len(messages) when the
        conversation ends on a closed batch.
        """
        messages: list[LLMMessage] = list(self._messages)
        boundaries: list[int] = []
        for index in range(1, len(messages) + 1):
            if messages[index - 1].role != Role.tool:
                continue
            if index < len(messages) and messages[index].role == Role.tool:
                continue
            if self._batch_closed_at(index):
                boundaries.append(index)
        return boundaries

    def _batch_closed_at(self, index: int) -> bool:
        """messages[:index] ends with results, and the assistant message that
        made those calls has every call answered among them.
        """
        messages: list[LLMMessage] = list(self._messages)
        answered: set[str] = set()
        j = index - 1
        while j >= 0 and messages[j].role == Role.tool:
            if messages[j].tool_call_id:
                answered.add(messages[j].tool_call_id or "")
            j -= 1
        if j < 0 or messages[j].role != Role.assistant or not messages[j].tool_calls:
            return False
        wanted = {tc.id for tc in (messages[j].tool_calls or ()) if tc.id}
        return wanted <= answered

    async def rewind_to_message(
        self, message_index: int, *, restore_files: bool
    ) -> tuple[str, list[str]]:
        """Rewind the session to the given user message index.

        Saves the current session, truncates messages, optionally restores
        files, and forks to a new session.

        Returns a tuple of (message_content, restore_errors).

        Raises:
            RewindError: If the message index is invalid or not a user message.
        """
        messages: Sequence[LLMMessage] = self._messages
        if message_index < 0 or message_index >= len(messages):
            raise RewindError(f"Invalid message index: {message_index}")

        user_msg = messages[message_index]
        if user_msg.role != Role.user:
            raise RewindError(f"Message at index {message_index} is not a user message")

        message_content = user_msg.content or ""
        restore_errors = await self._truncate_to(
            message_index, restore_files=restore_files
        )
        return message_content, restore_errors

    async def rewind_to_index(
        self, message_index: int, *, restore_files: bool
    ) -> list[str]:
        """Cut the conversation at a closed tool batch: messages[:index] stay,
        the tail is the batch's results, and the next user message is a
        steering message into the turn. Same save, checkpoint drop and
        session fork as a user-message rewind. Returns the restore errors.

        Raises:
            RewindError: when the index is not a batch boundary.
        """
        if message_index not in self.batch_boundaries():
            raise RewindError(
                f"Message index {message_index} does not close a tool batch"
            )
        return await self._truncate_to(message_index, restore_files=restore_files)

    async def _truncate_to(
        self, message_index: int, *, restore_files: bool
    ) -> list[str]:
        messages: list[LLMMessage] = list(self._messages)
        restore_errors: list[str] = []
        if restore_files:
            checkpoint = self._get_checkpoint(message_index)
            if checkpoint:
                restore_errors = self._restore_checkpoint(checkpoint)

        await self._save_messages()
        self._checkpoints = [
            cp for cp in self._checkpoints if cp.message_index < message_index
        ]
        n_to_remove = len(messages) - message_index
        # Suppress reset hooks: we already updated _checkpoints manually above.
        with self._messages.no_reset_hooks():
            self._messages.rewind(n_to_remove)
        self._reset_session()
        return restore_errors

    # -- Private helpers -------------------------------------------------------

    def _get_checkpoint(self, message_index: int) -> Checkpoint | None:
        for cp in self._checkpoints:
            if cp.message_index == message_index:
                return cp
        return None

    def _restore_checkpoint(self, checkpoint: Checkpoint) -> list[str]:
        """Restore files on disk to match the checkpoint state.

        Returns a list of human-readable error messages for files that
        could not be restored (empty when everything succeeded).
        """
        errors: list[str] = []
        for snap in checkpoint.files:
            path = Path(snap.path)
            if snap.content is None:
                if path.exists():
                    try:
                        os.remove(path)
                    except Exception:
                        errors.append(f"Failed to delete file: {snap.path}")
            else:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(snap.content)
                except Exception:
                    errors.append(f"Failed to restore file: {snap.path}")
        return errors

    @staticmethod
    def _has_changes_since(checkpoint: Checkpoint) -> bool:
        for snap in checkpoint.files:
            try:
                current: bytes | None = Path(snap.path).read_bytes()
            except FileNotFoundError:
                current = None
            if current != snap.content:
                return True
        return False

    @staticmethod
    def _read_snapshot(path: str) -> FileSnapshot:
        try:
            content: bytes | None = Path(path).read_bytes()
        except FileNotFoundError:
            content = None
        except Exception:
            logger.warning("Failed to read file for checkpoint: %s", path)
            content = None
        return FileSnapshot(path=path, content=content)

    def _on_messages_reset(self) -> None:
        """Called when the message list is rewound (session switch, clear, compact, etc.)."""
        self._checkpoints.clear()
