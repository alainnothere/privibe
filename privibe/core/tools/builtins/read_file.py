from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, NamedTuple, final

import anyio
from pydantic import BaseModel, Field

from privibe.core.config.harness_files import get_harness_files_manager
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
    large_file_advisory,
    normalization_note,
    normalize_tool_path,
    resolve_file_tool_permission,
)
from privibe.core.types import ToolStreamEvent
from privibe.core.utils import VIBE_WARNING_TAG

if TYPE_CHECKING:
    from privibe.core.types import ToolResultEvent


class _ReadResult(NamedTuple):
    lines: list[str]
    bytes_read: int
    was_truncated: bool


class ReadFileArgs(BaseModel):
    path: str
    offset: int = Field(
        default=0,
        description="Line number to start reading from (0-indexed, inclusive).",
    )
    limit: int | None = Field(
        default=None, description="Maximum number of lines to read."
    )


class ReadFileResult(BaseModel):
    path: str
    content: str
    lines_read: int
    was_truncated: bool = Field(
        description="True if the read stopped before end of file, whether due "
        "to the max_read_bytes cap or the line limit."
    )
    path_note: str | None = Field(
        default=None,
        description="Set when the input path was rewritten across path dialects "
        "(e.g. /c/foo → C:\\foo). Lets the model learn the canonical form.",
    )
    advisory: str | None = Field(
        default=None,
        description=(
            "Set when a whole-file read hit an over-threshold file and only a "
            "head preview was returned. Explains the recommended workflow: "
            "search first, then ranged reads."
        ),
    )


class ReadFileToolConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    sensitive_patterns: list[str] = Field(
        default=["**/.env", "**/.env.*"],
        description="File patterns that trigger ASK even when permission is ALWAYS.",
    )

    max_read_bytes: int = Field(
        default=64_000, description="Maximum total bytes to read from a file in one go."
    )
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


class ReadFileState(BaseToolState):
    injected_agents_md: set[str] = Field(default_factory=set)


class ReadFile(
    BaseTool[ReadFileArgs, ReadFileResult, ReadFileToolConfig, ReadFileState],
    ToolUIData[ReadFileArgs, ReadFileResult],
):
    description: ClassVar[str] = (
        "Read a UTF-8 file, returning content from a specific line range. "
        "Reading is capped by a byte limit for safety."
    )

    @final
    async def run(
        self, args: ReadFileArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | ReadFileResult, None]:
        file_path = self._prepare_and_validate_path(args)

        # A naive whole-file read (no offset/limit) of an over-threshold file
        # gets a small head preview plus an advisory instead of a 64KB flood.
        # Targeted reads pass through untouched: punishing them would teach
        # the model that ranged reads don't work either.
        advisory: str | None = None
        max_bytes = self.config.max_read_bytes
        naive = args.offset == 0 and args.limit is None
        threshold_bytes = self.config.large_file_threshold_kb * 1024
        preview_bytes = self.config.large_file_preview_kb * 1024
        size = file_path.resolve().stat().st_size
        if naive and size > threshold_bytes:
            max_bytes = min(max_bytes, preview_bytes)

        read_result = await self._read_file(args, file_path, max_bytes=max_bytes)

        if naive and size > threshold_bytes:
            advisory = large_file_advisory(
                size_bytes=size,
                preview_bytes=read_result.bytes_read,
                preview_lines=len(read_result.lines),
                threshold_kb=self.config.large_file_threshold_kb,
            )

        yield ReadFileResult(
            path=str(file_path),
            content="".join(read_result.lines),
            lines_read=len(read_result.lines),
            was_truncated=read_result.was_truncated,
            path_note=normalization_note(args.path, file_path),
            advisory=advisory,
        )

    def resolve_permission(self, args: ReadFileArgs) -> PermissionContext | None:
        return resolve_file_tool_permission(
            args.path,
            tool_name=self.get_name(),
            allowlist=self.config.allowlist,
            denylist=self.config.denylist,
            config_permission=self.config.permission,
            sensitive_patterns=self.config.sensitive_patterns,
        )

    def get_result_extra(self, result: ReadFileResult) -> str | None:
        try:
            mgr = get_harness_files_manager()
        except RuntimeError:
            return None
        docs = mgr.find_subdirectory_agents_md(Path(result.path))
        new_docs = [
            (d, c)
            for d, c in docs
            if str(d.resolve()) not in self.state.injected_agents_md
        ]
        if not new_docs:
            return None
        for d, _ in new_docs:
            self.state.injected_agents_md.add(str(d.resolve()))
        sections = [
            f"Contents of {d}/AGENTS.md (project instructions for this directory):\n\n{c.strip()}"
            for d, c in new_docs
        ]
        return f"<{VIBE_WARNING_TAG}>\n{'\n\n'.join(sections)}\n</{VIBE_WARNING_TAG}>"

    def _prepare_and_validate_path(self, args: ReadFileArgs) -> Path:
        self._validate_inputs(args)

        file_path = normalize_tool_path(args.path)

        self._validate_path(file_path)
        return file_path

    async def _read_file(
        self, args: ReadFileArgs, file_path: Path, *, max_bytes: int | None = None
    ) -> _ReadResult:
        try:
            return await self._do_read_file(
                args, file_path, encoding="utf-8", max_bytes=max_bytes
            )
        except (UnicodeDecodeError, ValueError):
            return await self._do_read_file(
                args, file_path, errors="replace", max_bytes=max_bytes
            )

    async def _do_read_file(
        self,
        args: ReadFileArgs,
        file_path: Path,
        *,
        encoding: str | None = None,
        errors: str | None = None,
        max_bytes: int | None = None,
    ) -> _ReadResult:
        byte_cap = max_bytes if max_bytes is not None else self.config.max_read_bytes
        try:
            lines_to_return: list[str] = []
            bytes_read = 0
            was_truncated = False

            async with await anyio.Path(file_path).open(
                encoding=encoding, errors=errors
            ) as f:
                line_index = 0
                async for line in f:
                    if line_index < args.offset:
                        line_index += 1
                        continue

                    if args.limit is not None and len(lines_to_return) >= args.limit:
                        # Reaching this check means the iterator produced a
                        # line beyond the limit, so more content provably
                        # exists; a file with exactly `limit` lines exits the
                        # loop naturally and stays untruncated.
                        was_truncated = True
                        break

                    line_bytes = len(line.encode("utf-8"))
                    if bytes_read + line_bytes > byte_cap:
                        was_truncated = True
                        break

                    lines_to_return.append(line)
                    bytes_read += line_bytes
                    line_index += 1

            return _ReadResult(
                lines=lines_to_return,
                bytes_read=bytes_read,
                was_truncated=was_truncated,
            )

        except OSError as exc:
            raise ToolError(f"Error reading {file_path}: {exc}") from exc

    def _validate_inputs(self, args: ReadFileArgs) -> None:
        if not args.path.strip():
            raise ToolError("Path cannot be empty")
        if args.offset < 0:
            raise ToolError("Offset cannot be negative")
        if args.limit is not None and args.limit <= 0:
            raise ToolError("Limit, if provided, must be a positive number")

    def _validate_path(self, file_path: Path) -> None:
        try:
            resolved_path = file_path.resolve()
        except ValueError:
            raise ToolError(
                f"Security error: Cannot read path '{file_path}' outside of the project directory '{Path.cwd()}'."
            )
        except FileNotFoundError:
            raise ToolError(f"File not found at: {file_path}")

        if not resolved_path.exists():
            raise ToolError(f"File not found at: {file_path}")
        if resolved_path.is_dir():
            raise ToolError(f"Path is a directory, not a file: {file_path}")

    @classmethod
    def format_call_display(cls, args: ReadFileArgs) -> ToolCallDisplay:
        summary = f"Reading {args.path}"
        if args.offset > 0 or args.limit is not None:
            parts = []
            if args.offset > 0:
                parts.append(f"from line {args.offset}")
            if args.limit is not None:
                parts.append(f"limit {args.limit} lines")
            summary += f" ({', '.join(parts)})"
        return ToolCallDisplay(summary=summary)

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ReadFileResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = f"Read {event.result.lines_read} line{'' if event.result.lines_read <= 1 else 's'} from {display_path(event.result.path)}"
        if event.result.was_truncated:
            message += " (truncated)"

        warnings = []
        if event.result.advisory:
            warnings.append("Large file: head preview only (see advisory)")
        elif event.result.was_truncated:
            warnings.append("File was truncated due to size limit")
        return ToolResultDisplay(
            success=True,
            message=message,
            warnings=warnings,
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Reading file"
