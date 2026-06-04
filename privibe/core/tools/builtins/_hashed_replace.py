from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
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
    ApplyResult,
    LineReplacement,
    apply_replacements_to_file,
)
from privibe.core.tools.permissions import PermissionContext
from privibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from privibe.core.tools.utils import resolve_file_tool_permission
from privibe.core.types import ToolStreamEvent

if TYPE_CHECKING:
    from privibe.core.types import ToolResultEvent

# Re-export LineReplacement so existing imports from this module keep working.
__all__ = ["HashedReplace", "HashedReplaceArgs", "HashedReplaceResult", "LineReplacement"]


class HashedReplaceArgs(BaseModel):
    path: str
    replacements: list[LineReplacement] = Field(
        min_length=1,
        description=(
            "One or more replacements to apply atomically. All hashes are validated "
            "against the original file before any change is written. Replacements are "
            "applied bottom-to-top so line numbers stay valid across the batch."
        ),
    )


class HashedReplaceResult(BaseModel):
    path: str
    total_replacements: int
    total_lines_replaced: int
    context: str
    path_note: str | None = Field(
        default=None,
        description="Set when the input path was rewritten across path dialects.",
    )


class HashedReplace(
    BaseTool[HashedReplaceArgs, HashedReplaceResult, BaseToolConfig, BaseToolState],
    ToolUIData[HashedReplaceArgs, HashedReplaceResult],
):
    description: ClassVar[str] = (
        "Replace one or more lines in a file using (line, hash) addresses from hashed_read. "
        "Pass all replacements at once — they are validated atomically and applied bottom-to-top "
        "so shifts from one replacement never invalidate another."
    )

    def get_file_snapshot(self, args: HashedReplaceArgs) -> FileSnapshot | None:
        return self.get_file_snapshot_for_path(args.path)

    def resolve_permission(self, args: HashedReplaceArgs) -> PermissionContext | None:
        return resolve_file_tool_permission(
            args.path,
            tool_name=self.get_name(),
            allowlist=self.config.allowlist,
            denylist=self.config.denylist,
            config_permission=self.config.permission,
        )

    @final
    async def run(
        self, args: HashedReplaceArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | HashedReplaceResult, None]:
        result = await apply_replacements_to_file(args.path, args.replacements)
        yield HashedReplaceResult(
            path=result.path,
            total_replacements=result.total_ops,
            total_lines_replaced=result.total_lines_changed,
            context=result.context,
            path_note=result.path_note,
        )

    @classmethod
    def format_call_display(cls, args: HashedReplaceArgs) -> ToolCallDisplay:
        n = len(args.replacements)
        summary = f"Replacing {n} region{'s' if n != 1 else ''} in {args.path}"
        content = "\n---\n".join(
            (f"line {r.line}–{r.end_line}" if r.end_line else f"line {r.line}")
            + f":\n{r.new_content}"
            for r in args.replacements
        )
        return ToolCallDisplay(summary=summary, content=content)

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, HashedReplaceResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        r = event.result
        return ToolResultDisplay(
            success=True,
            message=(
                f"Applied {r.total_replacements} replacement{'s' if r.total_replacements != 1 else ''} "
                f"({r.total_lines_replaced} line{'s' if r.total_lines_replaced != 1 else ''} replaced) "
                f"in {Path(r.path).name}"
            ),
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Editing file (hashed)"
