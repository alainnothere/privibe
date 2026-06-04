from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from privibe.core.rewind.manager import FileSnapshot
from privibe.core.tools.utils import normalize_tool_path

# A file is edited at most a handful of times in a self-correction loop; keep a
# small backward history per file and evict the oldest beyond this.
_DEFAULT_MAX_VERSIONS = 10
# Don't hoard very large files in RAM. write_file caps content at 64 KB; the
# hashed tools can edit larger files, but a multi-megabyte snapshot per edit is
# not worth keeping for an undo convenience.
_DEFAULT_MAX_ENTRY_BYTES = 5_000_000


class NothingToRestoreError(Exception):
    """Raised when restore is requested for a path with no captured versions."""


@dataclass(frozen=True, slots=True)
class RestoreOutcome:
    path: str
    action: str  # "restored" (wrote previous content) or "deleted" (undid a create)
    remaining: int  # versions still available to walk further back


def canonical_key(path_str: str) -> str:
    """Canonical absolute key for a path.

    Must match the path form produced by ``BaseTool.get_file_snapshot_for_path``
    so that a version captured during an edit is found again at restore time.
    """
    return str(normalize_tool_path(path_str).resolve())


class FileUndoStack:
    """Per-agent, in-memory history of pre-edit file versions.

    Every file-mutating tool's pre-edit snapshot is pushed here via the same
    hook that feeds the rewind checkpoint. ``restore`` pops one version and
    writes it back to disk, so repeated calls walk backward one edit at a time.

    State lives only in memory and is owned by a single agent: it dies when that
    agent is torn down and is cleared on session reset/clear/compact/rewind.
    Cross-agent and crash recovery are deliberately out of scope here — the
    user-facing rewind covers those.
    """

    def __init__(
        self,
        max_versions: int = _DEFAULT_MAX_VERSIONS,
        max_entry_bytes: int = _DEFAULT_MAX_ENTRY_BYTES,
    ) -> None:
        self._stacks: dict[str, list[FileSnapshot]] = {}
        self._max_versions = max_versions
        self._max_entry_bytes = max_entry_bytes

    def capture(self, snapshot: FileSnapshot) -> None:
        """Record a file's pre-edit state. Snapshot path is already canonical."""
        key = snapshot.path
        content = snapshot.content

        # The shared snapshot helper collapses "file did not exist" and "file
        # could not be read" both to None. Recording a transient read failure as
        # a None version would let a later restore *delete* a file that actually
        # existed. Only treat None as a delete-target when the path genuinely
        # does not exist on disk.
        if content is None and Path(key).exists():
            return

        # Bound per-entry memory; skip rather than hoard oversized files.
        if content is not None and len(content) > self._max_entry_bytes:
            return

        stack = self._stacks.setdefault(key, [])

        # Skip-if-same: the snapshot is taken before the tool runs, so a tool
        # that errors out or rewrites identical bytes would otherwise push a
        # redundant version and waste a slot.
        if stack and stack[-1].content == content:
            return

        stack.append(snapshot)
        if len(stack) > self._max_versions:
            del stack[0 : len(stack) - self._max_versions]

    def has_versions(self, path_str: str) -> bool:
        return bool(self._stacks.get(canonical_key(path_str)))

    def restore(self, path_str: str) -> RestoreOutcome:
        """Revert the file to its state before the most recent recorded edit.

        Pops one version (consume, not toggle), so calling again walks further
        back. Raises NothingToRestoreError if no version is recorded.
        """
        key = canonical_key(path_str)
        stack = self._stacks.get(key)
        if not stack:
            raise NothingToRestoreError(
                f"No restore point recorded for '{path_str}'. restore_file can only "
                "undo edits made by the file tools earlier in this session."
            )

        snapshot = stack.pop()
        if not stack:
            self._stacks.pop(key, None)

        target = Path(key)
        if snapshot.content is None:
            # Pre-edit state was "did not exist": a prior edit created the file,
            # so reverting means removing it again.
            if target.exists():
                target.unlink()
            return RestoreOutcome(path=key, action="deleted", remaining=len(stack))

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(snapshot.content)
        return RestoreOutcome(path=key, action="restored", remaining=len(stack))

    def clear(self) -> None:
        self._stacks.clear()
