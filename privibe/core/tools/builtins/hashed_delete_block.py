from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, ClassVar, final

from pydantic import BaseModel, Field

from privibe.core.rewind.manager import FileSnapshot
from privibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
)
from privibe.core.tools.builtins._hashed_core import (
    LineReplacement,
    apply_replacements_to_file,
)
from privibe.core.tools.permissions import PermissionContext
from privibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from privibe.core.tools.utils import display_path, resolve_file_tool_permission
from privibe.core.types import ToolStreamEvent

if TYPE_CHECKING:
    from privibe.core.types import ToolResultEvent


class DeleteBlockItem(BaseModel):
    line: int = Field(description="1-based line number of the first line to delete (from hashed_read).")
    hash: str = Field(description="4-char hash of line from hashed_read.")
    end_line: int = Field(description="1-based line number of the last line to delete (inclusive, from hashed_read).")
    end_hash: str = Field(description="4-char hash of end_line from hashed_read.")


class HashedDeleteBlockArgs(BaseModel):
    path: str
    deletions: list[DeleteBlockItem] = Field(
        min_length=1,
        description=(
            "One or more block deletions to apply atomically. Each item addresses a range "
            "of lines (line through end_line, inclusive). Both line and end_line are required. "
            "To delete a single line use hashed_delete_line."
        ),
    )


class HashedDeleteBlockResult(BaseModel):
    path: str
    total_deletions: int
    total_lines_deleted: int
    context: str
    path_note: str | None = Field(
        default=None,
        description="Set when the input path was rewritten across path dialects.",
    )


class HashedDeleteBlock(
    BaseTool[HashedDeleteBlockArgs, HashedDeleteBlockResult, BaseToolConfig, BaseToolState],
    ToolUIData[HashedDeleteBlockArgs, HashedDeleteBlockResult],
):
    description: ClassVar[str] = (
        "Delete a range of lines from a file using (line, hash, end_line, end_hash) addresses "
        "from hashed_read. Both line and end_line are required — this tool always deletes the "
        "full range between them. To delete a single line use hashed_delete_line."
    )

    mutates_files: ClassVar[bool] = True

    def get_file_snapshot(self, args: HashedDeleteBlockArgs) -> FileSnapshot | None:
        return self.get_file_snapshot_for_path(args.path)

    def resolve_permission(self, args: HashedDeleteBlockArgs) -> PermissionContext | None:
        return resolve_file_tool_permission(
            args.path,
            tool_name=self.get_name(),
            allowlist=self.config.allowlist,
            denylist=self.config.denylist,
            config_permission=self.config.permission,
            sensitive_patterns=self.config.sensitive_patterns,
        )

    @final
    async def run(
        self, args: HashedDeleteBlockArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | HashedDeleteBlockResult, None]:
        replacements = [
            LineReplacement(
                line=d.line,
                hash=d.hash,
                new_content="",
                end_line=d.end_line,
                end_hash=d.end_hash,
            )
            for d in args.deletions
        ]
        result = await apply_replacements_to_file(args.path, replacements)
        yield HashedDeleteBlockResult(
            path=result.path,
            total_deletions=result.total_ops,
            total_lines_deleted=result.total_lines_changed,
            context=result.context,
            path_note=result.path_note,
        )

    @classmethod
    def format_call_display(cls, args: HashedDeleteBlockArgs) -> ToolCallDisplay:
        n = len(args.deletions)
        summary = f"Deleting {n} block{'s' if n != 1 else ''} from {args.path}"
        content = ", ".join(f"lines {d.line}–{d.end_line}" for d in args.deletions)
        return ToolCallDisplay(summary=summary, content=content)

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, HashedDeleteBlockResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        r = event.result
        return ToolResultDisplay(
            success=True,
            message=(
                f"Deleted {r.total_deletions} block{'s' if r.total_deletions != 1 else ''} "
                f"({r.total_lines_deleted} line{'s' if r.total_lines_deleted != 1 else ''} deleted) "
                f"from {display_path(r.path)}"
            ),
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Deleting blocks (hashed)"
