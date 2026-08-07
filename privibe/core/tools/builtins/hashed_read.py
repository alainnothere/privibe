from __future__ import annotations

import zlib
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, final

import anyio
from pydantic import BaseModel, Field

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
    large_file_advisory,
    normalization_note,
    normalize_tool_path,
    resolve_file_tool_permission,
)
from privibe.core.types import ToolStreamEvent

if TYPE_CHECKING:
    from privibe.core.types import ToolResultEvent


def _line_hash(content: str) -> str:
    """4-char hex hash of line content (newline stripped, trailing spaces included)."""
    return f"{zlib.adler32(content.encode('utf-8')) & 0xFFFF:04x}"


def format_hashed_lines(lines: list[str], start_num: int) -> str:
    parts = []
    for i, line in enumerate(lines):
        content = line.rstrip("\r\n")
        h = _line_hash(content)
        parts.append(f"{start_num + i}|{h}|{content}")
    return "\n".join(parts)


class HashedReadArgs(BaseModel):
    path: str
    offset: int = Field(
        default=0,
        description="Line number to start reading from (0-indexed, inclusive).",
    )
    limit: int | None = Field(
        default=None, description="Maximum number of lines to read."
    )


class HashedReadResult(BaseModel):
    path: str
    content: str
    start_line: int
    lines_read: int
    was_truncated: bool = Field(
        description="True if reading was stopped due to the byte limit."
    )
    path_note: str | None = Field(
        default=None,
        description="Set when the input path was rewritten across path dialects.",
    )
    advisory: str | None = Field(
        default=None,
        description=(
            "Set when a whole-file read hit an over-threshold file and only a "
            "head preview was returned. Explains the recommended workflow: "
            "search first, then ranged reads."
        ),
    )


class HashedReadConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    sensitive_patterns: list[str] = Field(
        default=["**/.env", "**/.env.*"],
        description="File patterns that trigger ASK even when permission is ALWAYS.",
    )
    max_read_bytes: int = Field(default=64_000)
    large_file_threshold_kb: int = Field(
        default=128,
        description=(
            "A whole-file read (no offset/limit) of a file larger than this "
            "returns only a head preview plus an advisory instead of "
            "max_read_bytes of content. Targeted reads are unaffected."
        ),
    )
    large_file_preview_kb: int = Field(
        default=10,
        description="Size of the head preview returned for over-threshold files.",
    )


class HashedRead(
    BaseTool[HashedReadArgs, HashedReadResult, HashedReadConfig, BaseToolState],
    ToolUIData[HashedReadArgs, HashedReadResult],
):
    description: ClassVar[str] = (
        "Read a file returning each line prefixed with its 0-based line number and a "
        "4-char hash. Use these (line, hash) pairs as addresses for hashed_replace_line,"
        "hashed_replace_block, hashed_delete_line and hashed_delete_block"
    )

    @final
    async def run(
        self, args: HashedReadArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | HashedReadResult, None]:
        file_path = self._resolve_path(args.path)

        # Same policy as read_file: a naive whole-file read of an
        # over-threshold file gets a head preview plus an advisory; targeted
        # (offset/limit) reads pass through untouched.
        advisory: str | None = None
        max_bytes = self.config.max_read_bytes
        naive = args.offset == 0 and args.limit is None
        threshold_bytes = self.config.large_file_threshold_kb * 1024
        preview_bytes = self.config.large_file_preview_kb * 1024
        size = file_path.stat().st_size
        if naive and size > threshold_bytes:
            max_bytes = min(max_bytes, preview_bytes)

        lines, was_truncated = await self._read_lines(
            args, file_path, max_bytes=max_bytes
        )

        if naive and size > threshold_bytes:
            advisory = large_file_advisory(
                size_bytes=size,
                preview_bytes=sum(len(line.encode("utf-8")) for line in lines),
                preview_lines=len(lines),
                threshold_kb=self.config.large_file_threshold_kb,
            )

        start_num = args.offset + 1  # convert 0-indexed offset to 1-based line number
        yield HashedReadResult(
            path=str(file_path),
            content=format_hashed_lines(lines, start_num),
            start_line=start_num,
            lines_read=len(lines),
            was_truncated=was_truncated,
            path_note=normalization_note(args.path, file_path),
            advisory=advisory,
        )

    def resolve_permission(self, args: HashedReadArgs) -> PermissionContext | None:
        return resolve_file_tool_permission(
            args.path,
            tool_name=self.get_name(),
            allowlist=self.config.allowlist,
            denylist=self.config.denylist,
            config_permission=self.config.permission,
            sensitive_patterns=self.config.sensitive_patterns,
        )

    def _resolve_path(self, path_str: str) -> Path:
        if not path_str.strip():
            raise ToolError("Path cannot be empty")
        resolved = normalize_tool_path(path_str).resolve()
        if not resolved.exists():
            raise ToolError(f"File not found: {path_str}")
        if resolved.is_dir():
            raise ToolError(f"Path is a directory: {path_str}")
        return resolved

    async def _read_lines(
        self, args: HashedReadArgs, file_path: Path, *, max_bytes: int | None = None
    ) -> tuple[list[str], bool]:
        byte_cap = max_bytes if max_bytes is not None else self.config.max_read_bytes
        try:
            lines: list[str] = []
            bytes_read = 0
            was_truncated = False
            async with await anyio.Path(file_path).open(encoding="utf-8", errors="replace") as f:
                line_index = 0
                async for line in f:
                    if line_index < args.offset:
                        line_index += 1
                        continue
                    if args.limit is not None and len(lines) >= args.limit:
                        break
                    line_bytes = len(line.encode("utf-8"))
                    if bytes_read + line_bytes > byte_cap:
                        was_truncated = True
                        break
                    lines.append(line)
                    bytes_read += line_bytes
                    line_index += 1
            return lines, was_truncated
        except OSError as exc:
            raise ToolError(f"Error reading {file_path}: {exc}") from exc

    @classmethod
    def format_call_display(cls, args: HashedReadArgs) -> ToolCallDisplay:
        summary = f"Reading {args.path} (hashed)"
        if args.offset > 0 or args.limit is not None:
            parts = []
            if args.offset > 0:
                parts.append(f"from line {args.offset}")
            if args.limit is not None:
                parts.append(f"limit {args.limit}")
            summary += f" ({', '.join(parts)})"
        return ToolCallDisplay(summary=summary)

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, HashedReadResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        r = event.result
        msg = f"Read {r.lines_read} line{'s' if r.lines_read != 1 else ''} from {Path(r.path).name} (hashed)"
        warnings = []
        if r.advisory:
            warnings.append("Large file: head preview only (see advisory)")
        elif r.was_truncated:
            warnings.append("File was truncated due to size limit")
        return ToolResultDisplay(
            success=True,
            message=msg,
            warnings=warnings,
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Reading file (hashed)"
