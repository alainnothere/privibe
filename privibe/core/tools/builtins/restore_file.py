from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import ClassVar, final

from pydantic import BaseModel, Field

from privibe.core.rewind.undo_stack import NothingToRestoreError
from privibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from privibe.core.tools.permissions import PermissionContext
from privibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from privibe.core.tools.utils import (
    display_path,
    normalization_note,
    normalize_tool_path,
    resolve_file_tool_permission,
)
from privibe.core.types import ToolResultEvent, ToolStreamEvent


class RestoreFileArgs(BaseModel):
    path: str = Field(description="Path of the file to revert to its previous state.")


class RestoreFileResult(BaseModel):
    path: str
    action: str = Field(
        description="'restored' if previous content was rewritten, 'deleted' if the "
        "reverted edit had created the file."
    )
    remaining: int = Field(
        description="How many further restore points remain for this file."
    )
    path_note: str | None = Field(
        default=None,
        description="Set when the input path was rewritten across path dialects.",
    )


class RestoreFileConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    sensitive_patterns: list[str] = Field(
        default=["**/.env", "**/.env.*"],
        description="File patterns that trigger ASK even when permission is ALWAYS.",
    )


class RestoreFile(
    BaseTool[RestoreFileArgs, RestoreFileResult, RestoreFileConfig, BaseToolState],
    ToolUIData[RestoreFileArgs, RestoreFileResult],
):
    description: ClassVar[str] = (
        "Undo the most recent edit a file tool made to a file, reverting it to the "
        "content it had just before that edit (or deleting it if the edit had created "
        "it). Call again to walk further back, one edit at a time. Prefer this over "
        "rewriting a file from scratch when an edit went wrong."
    )

    # Reverting writes to disk, so it must run serially with other writes.
    mutates_files: ClassVar[bool] = True

    # Intentionally no get_file_snapshot override: restore must NOT push a new
    # version (it consumes the stack), so it is excluded from edit capture.

    def resolve_permission(self, args: RestoreFileArgs) -> PermissionContext | None:
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
        self, args: RestoreFileArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | RestoreFileResult, None]:
        if not args.path.strip():
            raise ToolError("Path cannot be empty")
        if ctx is None or ctx.undo_stack is None:
            raise ToolError("Restore is not available in this context.")

        try:
            outcome = ctx.undo_stack.restore(args.path)
        except NothingToRestoreError as exc:
            raise ToolError(str(exc)) from exc

        yield RestoreFileResult(
            path=outcome.path,
            action=outcome.action,
            remaining=outcome.remaining,
            path_note=normalization_note(args.path, normalize_tool_path(args.path)),
        )

    @classmethod
    def format_call_display(cls, args: RestoreFileArgs) -> ToolCallDisplay:
        return ToolCallDisplay(
            summary=f"Restoring previous version of {args.path}",
            content="",
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, RestoreFileResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        r = event.result
        name = display_path(r.path)
        if r.action == "deleted":
            message = f"Reverted {name}: the edit had created it, so it was removed"
        else:
            message = f"Restored {name} to its previous content"
        if r.remaining:
            message += f" ({r.remaining} more restore point{'s' if r.remaining != 1 else ''})"
        return ToolResultDisplay(success=True, message=message)

    @classmethod
    def get_status_text(cls) -> str:
        return "Restoring file"
