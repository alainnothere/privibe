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


class ReplaceLineItem(BaseModel):
    line: int = Field(description="1-based line number from hashed_read.")
    hash: str = Field(description="4-char hash from hashed_read for that line.")
    new_content: str = Field(
        description=(
            "Replacement text for this single line. May be multiline — the line expands "
            "into however many lines new_content contains. Do not include a trailing newline."
            "Avoid doing more than 4 replacements at a time, small group of changes allows you to see the feedback of the change, and correct if needed, instead of waiting 10 minutes, 5 for creating a huge set of changes, noticing the error and another 5 for applying the correction."
        )
    )


class HashedReplaceLineArgs(BaseModel):
    path: str
    replacements: list[ReplaceLineItem] = Field(
        min_length=1,
        description=(
            "One or more single-line replacements to apply atomically. Each item addresses "
            "exactly one line; use hashed_replace_block to replace a range of lines."
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


class HashedReplaceLineResult(BaseModel):
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


class HashedReplaceLine(
    BaseTool[HashedReplaceLineArgs, HashedReplaceLineResult, BaseToolConfig, BaseToolState],
    ToolUIData[HashedReplaceLineArgs, HashedReplaceLineResult],
):
    description: ClassVar[str] = (
        "Replace individual lines in a file using (line, hash) addresses from hashed_read. "
        "Each replacement targets exactly one line — new_content may be multiline to expand "
        "that line into a block. To replace a range of lines use hashed_replace_block."
    )

    mutates_files: ClassVar[bool] = True

    def get_file_snapshot(self, args: HashedReplaceLineArgs) -> FileSnapshot | None:
        return self.get_file_snapshot_for_path(args.path)

    def resolve_permission(self, args: HashedReplaceLineArgs) -> PermissionContext | None:
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
        self, args: HashedReplaceLineArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | HashedReplaceLineResult, None]:
        replacements = [
            LineReplacement(line=r.line, hash=r.hash, new_content=r.new_content)
            for r in args.replacements
        ]
        result = await apply_replacements_to_file(
            args.path,
            replacements,
            allow_literal=args.allow_literal,
            keep_duplicate=args.keep_duplicate,
        )
        yield HashedReplaceLineResult(
            path=result.path,
            total_replacements=result.total_ops,
            total_lines_replaced=result.total_lines_changed,
            context=result.context,
            path_note=result.path_note,
            content_note=result.content_note,
        )

    @classmethod
    def format_call_display(cls, args: HashedReplaceLineArgs) -> ToolCallDisplay:
        n = len(args.replacements)
        summary = f"Replacing {n} line{'s' if n != 1 else ''} in {args.path}"
        content = "\n---\n".join(
            f"line {r.line}:\n{r.new_content}" for r in args.replacements
        )
        return ToolCallDisplay(summary=summary, content=content)

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, HashedReplaceLineResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        r = event.result
        return ToolResultDisplay(
            success=True,
            message=(
                f"Replaced {r.total_replacements} line{'s' if r.total_replacements != 1 else ''} "
                f"in {display_path(r.path)}"
            ),
            warnings=[r.content_note] if r.content_note else [],
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Replacing lines (hashed)"
