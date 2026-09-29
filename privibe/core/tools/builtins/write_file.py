from __future__ import annotations

import codecs
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import ClassVar, Literal, final

from anyio import to_thread
from pydantic import BaseModel, Field, PrivateAttr

from privibe.core.rewind.manager import FileSnapshot
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
    normalization_note,
    normalize_tool_path,
    resolve_file_tool_permission,
)
from privibe.core.types import ToolResultEvent, ToolStreamEvent
from privibe.core.utils.asciify import asciify, format_ascii_notes, resolve_ascii
from privibe.core.utils.newfile_conventions import conventions_for, editorconfig_for
from privibe.core.utils.textfile import (
    BinaryFileError,
    TextFile,
    UnencodableError,
    content_to_lines,
    ending_name,
    load_text_file_async,
    trim_trailing_whitespace,
    write_text_file,
)


class WriteFileArgs(BaseModel):
    path: str
    content: str
    overwrite: bool = Field(
        default=False, description="Must be set to true to overwrite an existing file."
    )
    allow_unicode: bool | None = Field(
        default=None,
        description=(
            "Omit to use the session default. true: write non-ASCII characters "
            "verbatim as UTF-8. false: transliterate to ASCII best effort and "
            "report every replacement."
        ),
    )
    newline: Literal["lf", "crlf"] | None = Field(
        default=None,
        description=(
            "Line ending for the file. Omit to match the repo (.gitattributes, "
            ".editorconfig, sibling files) or, on overwrite, the file's own. "
            "Set only when you know better, e.g. a Windows-only script."
        ),
    )
    encoding: str | None = Field(
        default=None,
        description=(
            "Python codec name for the bytes on disk. Omit for UTF-8 without BOM "
            "(or the existing file's encoding on overwrite). 'utf-8-sig' writes "
            "a BOM, e.g. for a PowerShell 5.1 script or a CSV Excel must open."
        ),
    )
    # Notes produced by prepare_args (transliterations), carried to the result.
    _prepare_notes: list[str] = PrivateAttr(default_factory=list)


class WriteFileResult(BaseModel):
    path: str
    bytes_written: int
    file_existed: bool
    content: str = Field(
        description="The content as written (after any ASCII transliteration)."
    )
    path_note: str | None = Field(
        default=None,
        description="Set when the input path was rewritten across path dialects.",
    )
    content_note: str | None = Field(
        default=None,
        description=(
            "Set when the tool changed or decided something about your content: "
            "non-ASCII characters transliterated (names the line and the flag to "
            "keep them), a non-UTF-8 file re-encoded as itself, or a new file's "
            "conventions taken from the repo."
        ),
    )


class WriteFileConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    sensitive_patterns: list[str] = Field(
        default=["**/.env", "**/.env.*"],
        description="File patterns that trigger ASK even when permission is ALWAYS.",
    )
    max_write_bytes: int = 64_000
    create_parent_dirs: bool = True


class WriteFile(
    BaseTool[WriteFileArgs, WriteFileResult, WriteFileConfig, BaseToolState],
    ToolUIData[WriteFileArgs, WriteFileResult],
):
    description: ClassVar[str] = (
        "Create a file, or overwrite one with 'overwrite=True'. Line endings, "
        "encoding and BOM follow the repo for new files and the file itself on "
        "overwrite; content is written as ASCII by default (see allow_unicode)."
    )

    mutates_files: ClassVar[bool] = True

    @classmethod
    def format_call_display(cls, args: WriteFileArgs) -> ToolCallDisplay:
        return ToolCallDisplay(
            summary=f"write_file {args.path}{' (overwrite)' if args.overwrite else ''}",
            content=args.content,
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if isinstance(event.result, WriteFileResult):
            action = "overwrote" if event.result.file_existed else "created"
            return ToolResultDisplay(
                success=True,
                message=f"{action} {event.result.path}",
                warnings=[event.result.content_note] if event.result.content_note else [],
            )

        return ToolResultDisplay(success=True, message="File written")

    @classmethod
    def get_status_text(cls) -> str:
        return "Writing file"

    def get_file_snapshot(self, args: WriteFileArgs) -> FileSnapshot | None:
        return self.get_file_snapshot_for_path(args.path)

    permission_group: ClassVar[str] = "file"

    def prepare_args(self, args: WriteFileArgs) -> None:
        if resolve_ascii(args.allow_unicode, self.config.ascii_default):
            result = asciify(args.content)
            if result.changed:
                args.content = result.text
                args._prepare_notes = format_ascii_notes(result.replacements)

    def resolve_permission(self, args: WriteFileArgs) -> PermissionContext | None:
        return resolve_file_tool_permission(
            args.path,
            tool_name=self.get_name(),
            allowlist=self.config.allowlist,
            denylist=self.config.denylist,
            config_permission=self.config.permission,
            sensitive_patterns=self.config.sensitive_patterns,
            protected_paths=self.config.protected_paths,
            protect_outside_workdir=self.config.protect_outside_workdir,
            outside_workdir_exempt=self.config.outside_workdir_exempt,
        )

    @final
    async def run(
        self, args: WriteFileArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | WriteFileResult, None]:
        file_path, file_existed, _ = self._prepare_and_validate_path(args)

        tf, trim, notes = await self._shape_for(args, file_path, file_existed)
        lines = self._content_lines(args.content, trim, notes)
        bytes_written = await self._write_file(file_path, tf, lines)

        all_notes = [*args._prepare_notes, *notes]
        yield WriteFileResult(
            path=str(file_path),
            bytes_written=bytes_written,
            file_existed=file_existed,
            content=args.content,
            path_note=normalization_note(args.path, file_path),
            content_note="\n".join(all_notes) if all_notes else None,
        )

    def _prepare_and_validate_path(self, args: WriteFileArgs) -> tuple[Path, bool, int]:
        if not args.path.strip():
            raise ToolError("Path cannot be empty")

        content_bytes = len(args.content.encode("utf-8"))
        if content_bytes > self.config.max_write_bytes:
            raise ToolError(
                f"Content exceeds {self.config.max_write_bytes} bytes limit"
            )

        file_path = normalize_tool_path(args.path).resolve()

        file_existed = file_path.exists()

        if file_existed and not args.overwrite:
            raise ToolError(
                f"File '{file_path}' exists. Set overwrite=True to replace."
            )

        if self.config.create_parent_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        elif not file_path.parent.exists():
            raise ToolError(f"Parent directory does not exist: {file_path.parent}")

        return file_path, file_existed, content_bytes

    async def _shape_for(
        self, args: WriteFileArgs, file_path: Path, file_existed: bool
    ) -> tuple[TextFile, bool, list[str]]:
        """The TextFile whose endings/BOM/encoding the new content adopts,
        whether .editorconfig wants trailing whitespace trimmed, and notes.

        Overwrite blends with the file as it is. A new file asks the repo. An
        explicit ``newline`` or ``encoding`` on the call is a deliberate
        conversion and wins over both.
        """
        notes: list[str] = []
        tf: TextFile | None = None
        if file_existed:
            try:
                tf = await load_text_file_async(
                    file_path, charset_hint=editorconfig_for(file_path).charset
                )
            except BinaryFileError:
                tf = None
                notes.append("existing file was binary; written as a new text file")
            except OSError as exc:
                raise ToolError(f"Error reading {file_path}: {exc}") from exc
            if tf is not None and tf.encoding_note:
                notes.append(tf.encoding_note)
        if tf is None:
            conv = await to_thread.run_sync(conventions_for, file_path)
            tf = TextFile.empty(
                newline=conv.newline,
                encoding=conv.encoding,
                bom=conv.bom,
                final_newline=conv.final_newline,
            )
            trim = conv.trim_trailing_whitespace
        else:
            trim = bool(editorconfig_for(file_path).trim_trailing_whitespace)

        if args.newline is not None:
            wanted = "\n" if args.newline == "lf" else "\r\n"
            if tf.existed and wanted != tf.newline:
                notes.append(
                    f"line endings converted to {ending_name(wanted)} as requested"
                )
            # A deliberate conversion: every line gets the requested ending.
            tf = TextFile.empty(
                newline=wanted,
                encoding=tf.encoding,
                bom=tf.bom,
                final_newline=tf.had_final_newline,
            )
        if args.encoding is not None:
            name = args.encoding.strip().lower()
            bom = name in {"utf-8-sig", "utf-8-bom", "utf8-sig"}
            codec_name = "utf-8" if bom else name
            try:
                codecs.lookup(codec_name)
            except LookupError as exc:
                raise ToolError(f"Unknown encoding {args.encoding!r}") from exc
            if tf.existed and (codec_name != tf.encoding or bom != tf.bom):
                notes.append(
                    f"file re-encoded as {codec_name}{' with BOM' if bom else ''} as requested"
                )
            tf = TextFile.empty(
                newline=tf.newline,
                encoding=codec_name,
                bom=bom,
                final_newline=tf.had_final_newline,
            )
        return tf, trim, notes

    @staticmethod
    def _content_lines(content: str, trim: bool, notes: list[str]) -> list[str]:
        # A trailing newline in the content is the final newline, not an
        # extra blank line; whether the file ends with one is the file's (or
        # the repo's) decision, remembered in tf.had_final_newline.
        body = content[:-1] if content.endswith("\n") else content
        body = body[:-1] if body.endswith("\r") else body
        lines = content_to_lines(body) if body else ([] if not content else ["\n"])
        if trim and lines:
            lines, trimmed = trim_trailing_whitespace(lines)
            if trimmed:
                notes.append(
                    f"trimmed trailing whitespace from {trimmed} "
                    f"line{'s' if trimmed != 1 else ''} (.editorconfig)"
                )
        return lines

    async def _write_file(self, file_path: Path, tf: TextFile, lines: list[str]) -> int:
        try:
            return await write_text_file(file_path, tf, lines)
        except UnencodableError as exc:
            raise ToolError(f"Error writing {file_path}: {exc}") from exc
        except Exception as e:
            raise ToolError(f"Error writing {file_path}: {e}") from e
