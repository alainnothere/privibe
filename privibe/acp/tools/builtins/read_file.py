from __future__ import annotations

from pathlib import Path

from privibe import VIBE_ROOT
from privibe.acp.tools.base import AcpToolState, BaseAcpTool
from privibe.core.tools.base import ToolError
from privibe.core.tools.builtins.read_file import (
    ReadFile as CoreReadFileTool,
    ReadFileArgs,
    ReadFileResult,
    ReadFileState,
    _ReadResult,
)

ReadFileResult = ReadFileResult


class AcpReadFileState(ReadFileState, AcpToolState):
    pass


class ReadFile(CoreReadFileTool, BaseAcpTool[AcpReadFileState]):
    state: AcpReadFileState
    prompt_path = VIBE_ROOT / "core" / "tools" / "builtins" / "prompts" / "read_file.md"

    @classmethod
    def _get_tool_state_class(cls) -> type[AcpReadFileState]:
        return AcpReadFileState

    async def _read_file(
        self, args: ReadFileArgs, file_path: Path, *, max_bytes: int | None = None
    ) -> _ReadResult:
        client, session_id, _ = self._load_state()

        line = args.offset + 1 if args.offset > 0 else None
        limit = args.limit

        await self._send_in_progress_session_update()

        try:
            response = await client.read_text_file(
                session_id=session_id, path=str(file_path), line=line, limit=limit
            )
        except Exception as e:
            raise ToolError(f"Error reading {file_path}: {e}") from e

        content_lines = response.content.splitlines(keepends=True)

        # The ACP client has no byte cap of its own, so the large-file
        # preview cap is enforced here after the fact.
        was_truncated = False
        if max_bytes is not None:
            capped: list[str] = []
            bytes_so_far = 0
            for content_line in content_lines:
                line_bytes = len(content_line.encode("utf-8"))
                if bytes_so_far + line_bytes > max_bytes:
                    was_truncated = True
                    break
                capped.append(content_line)
                bytes_so_far += line_bytes
            content_lines = capped

        lines_read = len(content_lines)
        bytes_read = sum(len(line.encode("utf-8")) for line in content_lines)

        was_truncated = was_truncated or (
            args.limit is not None and lines_read >= args.limit
        )

        return _ReadResult(
            lines=content_lines, bytes_read=bytes_read, was_truncated=was_truncated
        )
