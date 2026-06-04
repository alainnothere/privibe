from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from enum import StrEnum, auto
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, ClassVar

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
    display_path,
    normalization_note,
    normalize_tool_path,
    resolve_file_tool_permission,
)
from privibe.core.types import ToolStreamEvent
from privibe.core.utils.io import read_safe

if TYPE_CHECKING:
    from privibe.core.types import ToolResultEvent


class GrepBackend(StrEnum):
    RIPGREP = auto()
    GNU_GREP = auto()


class GrepToolConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    sensitive_patterns: list[str] = Field(
        default=["**/.env", "**/.env.*"],
        description="File patterns that trigger ASK even when permission is ALWAYS.",
    )

    max_output_bytes: int = Field(
        default=64_000, description="Hard cap for the total size of matched lines."
    )
    default_max_matches: int = Field(
        default=100, description="Default maximum number of matches to return."
    )
    default_timeout: int = Field(
        default=60, description="Default timeout for the search command in seconds."
    )
    exclude_patterns: list[str] = Field(
        default=[
            ".venv/",
            "venv/",
            ".env/",
            "env/",
            "node_modules/",
            ".git/",
            "__pycache__/",
            ".pytest_cache/",
            ".mypy_cache/",
            ".tox/",
            ".nox/",
            ".coverage/",
            "htmlcov/",
            "dist/",
            "build/",
            ".idea/",
            ".vscode/",
            "*.egg-info",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".DS_Store",
            "Thumbs.db",
        ],
        description="List of glob patterns to exclude from search (dirs should end with /).",
    )
    codeignore_file: str = Field(
        default=".vibeignore",
        description="Name of the file to read for additional exclusion patterns.",
    )


class GrepArgs(BaseModel):
    pattern: str
    path: str = "."
    max_matches: int | None = Field(
        default=None, description="Override the default maximum number of matches."
    )
    use_default_ignore: bool = Field(
        default=True, description="Whether to respect .gitignore and .ignore files."
    )


class GrepResult(BaseModel):
    matches: str
    match_count: int
    was_truncated: bool = Field(
        description="True if output was cut short by max_matches or max_output_bytes."
    )
    path_note: str | None = Field(
        default=None,
        description="Set when the input path was rewritten across path dialects.",
    )
    # Echoed back for the user-facing result line; excluded from the model result.
    pattern: str = Field(default="", exclude=True)
    path: str = Field(default="", exclude=True)


class Grep(
    BaseTool[GrepArgs, GrepResult, GrepToolConfig, BaseToolState],
    ToolUIData[GrepArgs, GrepResult],
):
    description: ClassVar[str] = (
        "Recursively search files for a regex pattern using ripgrep (rg) or grep. "
        "Respects .gitignore and .codeignore files by default when using ripgrep."
    )

    def resolve_permission(self, args: GrepArgs) -> PermissionContext | None:
        return resolve_file_tool_permission(
            args.path,
            tool_name=self.get_name(),
            allowlist=self.config.allowlist,
            denylist=self.config.denylist,
            config_permission=self.config.permission,
            sensitive_patterns=self.config.sensitive_patterns,
        )

    def _detect_backend(self) -> GrepBackend:
        if shutil.which("rg"):
            return GrepBackend.RIPGREP
        if shutil.which("grep"):
            return GrepBackend.GNU_GREP
        raise ToolError(
            "Neither ripgrep (rg) nor grep is installed. "
            "Please install ripgrep: https://github.com/BurntSushi/ripgrep#installation"
        )

    async def run(
        self, args: GrepArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | GrepResult, None]:
        backend = self._detect_backend()
        resolved_path = self._validate_args(args)

        exclude_patterns = self._collect_exclude_patterns()
        cmd = self._build_command(args, exclude_patterns, backend, resolved_path)
        stdout = await self._execute_search(cmd)

        result = self._parse_output(
            stdout, args.max_matches or self.config.default_max_matches
        )
        result.path_note = normalization_note(args.path, resolved_path)
        result.pattern = args.pattern
        result.path = args.path
        yield result

    def _validate_args(self, args: GrepArgs) -> Path:
        if not args.pattern.strip():
            raise ToolError("Empty search pattern provided.")

        path_obj = normalize_tool_path(args.path)

        if not path_obj.exists():
            raise ToolError(f"Path does not exist: {args.path}")
        return path_obj

    def _collect_exclude_patterns(self) -> list[str]:
        patterns = list(self.config.exclude_patterns)

        codeignore_path = Path.cwd() / self.config.codeignore_file
        if codeignore_path.is_file():
            patterns.extend(self._load_codeignore_patterns(codeignore_path))

        return patterns

    def _load_codeignore_patterns(self, codeignore_path: Path) -> list[str]:
        patterns = []
        try:
            content = read_safe(codeignore_path)
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)
        except OSError:
            pass

        return patterns

    def _build_command(
        self,
        args: GrepArgs,
        exclude_patterns: list[str],
        backend: GrepBackend,
        resolved_path: Path,
    ) -> list[str]:
        if backend == GrepBackend.RIPGREP:
            return self._build_ripgrep_command(args, exclude_patterns, resolved_path)
        return self._build_gnu_grep_command(args, exclude_patterns, resolved_path)

    def _build_ripgrep_command(
        self, args: GrepArgs, exclude_patterns: list[str], resolved_path: Path
    ) -> list[str]:
        max_matches = args.max_matches or self.config.default_max_matches

        cmd = [
            "rg",
            "--line-number",
            "--no-heading",
            "--smart-case",
            "--no-binary",
            # Request one extra to detect truncation
            "--max-count",
            str(max_matches + 1),
        ]

        if not args.use_default_ignore:
            cmd.append("--no-ignore")

        for pattern in exclude_patterns:
            cmd.extend(["--glob", f"!{pattern}"])

        cmd.extend(["-e", args.pattern, str(resolved_path)])

        return cmd

    def _build_gnu_grep_command(
        self, args: GrepArgs, exclude_patterns: list[str], resolved_path: Path
    ) -> list[str]:
        max_matches = args.max_matches or self.config.default_max_matches

        cmd = ["grep", "-r", "-n", "-I", "-E", f"--max-count={max_matches + 1}"]

        if args.pattern.islower():
            cmd.append("-i")

        for pattern in exclude_patterns:
            if pattern.endswith("/"):
                dir_pattern = pattern.rstrip("/")
                cmd.append(f"--exclude-dir={dir_pattern}")
            else:
                cmd.append(f"--exclude={pattern}")

        cmd.extend(["-e", args.pattern, str(resolved_path)])

        return cmd

    async def _execute_search(self, cmd: list[str]) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=self.config.default_timeout
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                raise ToolError(
                    f"Search timed out after {self.config.default_timeout}s"
                )

            stdout = (
                stdout_bytes.decode("utf-8", errors="ignore") if stdout_bytes else ""
            )
            stderr = (
                stderr_bytes.decode("utf-8", errors="ignore") if stderr_bytes else ""
            )

            if proc.returncode not in {0, 1}:
                error_msg = stderr or f"Process exited with code {proc.returncode}"
                raise ToolError(f"grep error: {error_msg}")

            return stdout

        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"Error running grep: {exc}") from exc

    def _parse_output(self, stdout: str, max_matches: int) -> GrepResult:
        output_lines = stdout.splitlines() if stdout else []

        truncated_lines = output_lines[:max_matches]
        truncated_output = "\n".join(truncated_lines)

        was_truncated = (
            len(output_lines) > max_matches
            or len(truncated_output) > self.config.max_output_bytes
        )

        final_output = truncated_output[: self.config.max_output_bytes]

        return GrepResult(
            matches=final_output,
            match_count=len(truncated_lines),
            was_truncated=was_truncated,
        )

    @classmethod
    def format_call_display(cls, args: GrepArgs) -> ToolCallDisplay:
        summary = f"Grepping '{args.pattern}'"
        if args.path != ".":
            summary += f" in {args.path}"
        if args.max_matches:
            summary += f" (max {args.max_matches} matches)"
        if not args.use_default_ignore:
            summary += " [no-ignore]"
        return ToolCallDisplay(summary=summary)

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, GrepResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        r = event.result
        where = f" in {display_path(r.path)}" if r.path and r.path != "." else ""
        message = f'Found {r.match_count} matches for "{r.pattern}"{where}'
        if r.was_truncated:
            message += " (truncated)"

        warnings = []
        if r.was_truncated:
            warnings.append("Output was truncated due to size/match limits")

        return ToolResultDisplay(success=True, message=message, warnings=warnings)

    @classmethod
    def get_status_text(cls) -> str:
        return "Searching files"
