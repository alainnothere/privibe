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


class ReplaceBlockItem(BaseModel):
    line: int = Field(description="1-based line number of the first line to replace (from hashed_read).")
    hash: str = Field(description="4-char hash of line from hashed_read.")
    end_line: int = Field(description="1-based line number of the last line to replace (inclusive, from hashed_read).")
    end_hash: str = Field(description="4-char hash of end_line from hashed_read.")
    new_content: str = Field(
        description=(
            "Replacement text for the entire block. May be multiline. "
            "Do not include a trailing newline."
        )
    )


class HashedReplaceBlockArgs(BaseModel):
    path: str
    replacements: list[ReplaceBlockItem] = Field(
        min_length=1,
        description=(
            "One or more block replacements to apply atomically. Each item addresses a range "
            "of lines (line through end_line, inclusive). Both line and end_line are required. "
            "To replace a single line use hashed_replace_line."
            "Avoid doing more than 4 replacements at a time, small group of changes allows you to see the feedback of the change, and correct if needed, instead of waiting 10 minutes, 5 for creating a huge set of changes, noticing the error and another 5 for applying the correction."
        ),
    )
    allow_literal: bool = Field(
        default=False,
        description=(
            "By default a leaked hashed_read prefix (e.g. '   11 b1c4  ') is stripped "
            "from your new_content. Set true to write new_content exactly as given."
        ),
    )
    keep_duplicate: bool = Field(
        default=False,
        description=(
            "By default a first/last new line that exactly duplicates the untouched line "
            "just outside the edited region is dropped. Set true to keep the duplicate."
        ),
    )


class HashedReplaceBlockResult(BaseModel):
    path: str
    total_replacements: int
    total_lines_replaced: int
    context: str
    path_note: str | None = Field(
        default=None,
        description="Set when the input path was rewritten across path dialects.",
    )
    content_note: str | None = Field(
        default=None,
        description=(
            "Set when the tool corrected your new_content (stripped a leaked "
            "hashed_read prefix or removed a boundary-duplicate line). Names the "
            "affected lines and the flag to override."
        ),
    )


class HashedReplaceBlock(
    BaseTool[HashedReplaceBlockArgs, HashedReplaceBlockResult, BaseToolConfig, BaseToolState],
    ToolUIData[HashedReplaceBlockArgs, HashedReplaceBlockResult],
):
    description: ClassVar[str] = (
        "Replace a range of lines in a file using (line, hash, end_line, end_hash) addresses "
        "from hashed_read. Both line and end_line are required — this tool always replaces the "
        "full range between them. To replace a single line use hashed_replace_line."
    )

    mutates_files: ClassVar[bool] = True

    def get_file_snapshot(self, args: HashedReplaceBlockArgs) -> FileSnapshot | None:
        return self.get_file_snapshot_for_path(args.path)

    def resolve_permission(self, args: HashedReplaceBlockArgs) -> PermissionContext | None:
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
        self, args: HashedReplaceBlockArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | HashedReplaceBlockResult, None]:
        replacements = [
            LineReplacement(
                line=r.line,
                hash=r.hash,
                new_content=r.new_content,
                end_line=r.end_line,
                end_hash=r.end_hash,
            )
            for r in args.replacements
        ]
        result = await apply_replacements_to_file(
            args.path,
            replacements,
            allow_literal=args.allow_literal,
            keep_duplicate=args.keep_duplicate,
        )
        yield HashedReplaceBlockResult(
            path=result.path,
            total_replacements=result.total_ops,
            total_lines_replaced=result.total_lines_changed,
            context=result.context,
            path_note=result.path_note,
            content_note=result.content_note,
        )

    @classmethod
    def format_call_display(cls, args: HashedReplaceBlockArgs) -> ToolCallDisplay:
        n = len(args.replacements)
        summary = f"Replacing {n} block{'s' if n != 1 else ''} in {args.path}"
        content = "\n---\n".join(
            f"lines {r.line}–{r.end_line}:\n{r.new_content}" for r in args.replacements
        )
        return ToolCallDisplay(summary=summary, content=content)

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, HashedReplaceBlockResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        r = event.result
        return ToolResultDisplay(
            success=True,
            message=(
                f"Replaced {r.total_replacements} block{'s' if r.total_replacements != 1 else ''} "
                f"({r.total_lines_replaced} line{'s' if r.total_lines_replaced != 1 else ''} replaced) "
                f"in {display_path(r.path)}"
            ),
            warnings=[r.content_note] if r.content_note else [],
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Replacing blocks (hashed)"
