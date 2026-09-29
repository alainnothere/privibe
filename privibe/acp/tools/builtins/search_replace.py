from __future__ import annotations

from pathlib import Path

from acp.helpers import SessionUpdate
from acp.schema import (
    FileEditToolCallContent,
    ToolCallLocation,
    ToolCallProgress,
    ToolCallStart,
)

from privibe import VIBE_ROOT
from privibe.acp.tools.base import AcpToolState, BaseAcpTool
from privibe.core.tools.base import BaseToolState, ToolError
from privibe.core.tools.builtins.search_replace import (
    SearchReplace as CoreSearchReplaceTool,
    SearchReplaceArgs,
    SearchReplaceResult,
)
from privibe.core.types import ToolCallEvent, ToolResultEvent
from privibe.core.utils.textfile import TextFile, render_text


class AcpSearchReplaceState(BaseToolState, AcpToolState):
    file_backup_content: str | None = None


class SearchReplace(CoreSearchReplaceTool, BaseAcpTool[AcpSearchReplaceState]):
    state: AcpSearchReplaceState
    prompt_path = (
        VIBE_ROOT / "core" / "tools" / "builtins" / "prompts" / "search_replace.md"
    )

    @classmethod
    def _get_tool_state_class(cls) -> type[AcpSearchReplaceState]:
        return AcpSearchReplaceState

    async def _read_file(self, file_path: Path) -> TextFile:
        client, session_id, _ = self._load_state()

        await self._send_in_progress_session_update()

        try:
            response = await client.read_text_file(
                session_id=session_id, path=str(file_path)
            )
        except Exception as e:
            raise ToolError(f"Unexpected error reading {file_path}: {e}") from e

        self.state.file_backup_content = response.content
        # The client hands us a decoded string; its endings survive in it, so
        # the same per-line re-attachment applies. A BOM, if the client kept
        # one, arrives as U+FEFF and is remembered the same way.
        content = response.content
        bom = content.startswith("\ufeff")
        tf = TextFile.from_text(content[1:] if bom else content)
        tf.bom = bom
        return tf

    async def _backup_file(self, file_path: Path) -> None:
        if self.state.file_backup_content is None:
            return

        await self._send_text(
            file_path.with_suffix(file_path.suffix + ".bak"),
            self.state.file_backup_content,
        )

    async def _write_file(self, file_path: Path, tf: TextFile, lines: list[str]) -> None:
        await self._send_text(file_path, render_text(tf, lines))

    async def _send_text(self, file_path: Path, content: str) -> None:
        client, session_id, _ = self._load_state()

        try:
            await client.write_text_file(
                session_id=session_id, path=str(file_path), content=content
            )
        except Exception as e:
            raise ToolError(f"Error writing {file_path}: {e}") from e

    @classmethod
    def tool_call_session_update(cls, event: ToolCallEvent) -> SessionUpdate | None:
        args = event.args
        if args is None:
            return ToolCallStart(
                session_update="tool_call",
                title="search_replace",
                tool_call_id=event.tool_call_id,
                kind="edit",
                content=None,
                raw_input=None,
            )
        if not isinstance(args, SearchReplaceArgs):
            return None

        blocks = cls._parse_search_replace_blocks(args.content)

        return ToolCallStart(
            session_update="tool_call",
            title=cls.get_call_display(event).summary,
            tool_call_id=event.tool_call_id,
            kind="edit",
            content=[
                FileEditToolCallContent(
                    type="diff",
                    path=args.file_path,
                    old_text=block.search,
                    new_text=block.replace,
                )
                for block in blocks
            ],
            locations=[ToolCallLocation(path=args.file_path)],
            raw_input=args.model_dump_json(),
        )

    @classmethod
    def tool_result_session_update(cls, event: ToolResultEvent) -> SessionUpdate | None:
        if event.error:
            return ToolCallProgress(
                session_update="tool_call_update",
                tool_call_id=event.tool_call_id,
                status="failed",
            )

        result = event.result
        if not isinstance(result, SearchReplaceResult):
            return None

        blocks = cls._parse_search_replace_blocks(result.content)

        return ToolCallProgress(
            session_update="tool_call_update",
            tool_call_id=event.tool_call_id,
            status="completed",
            content=[
                FileEditToolCallContent(
                    type="diff",
                    path=result.file,
                    old_text=block.search,
                    new_text=block.replace,
                )
                for block in blocks
            ],
            locations=[ToolCallLocation(path=result.file)],
            raw_output=result.model_dump_json(),
        )
