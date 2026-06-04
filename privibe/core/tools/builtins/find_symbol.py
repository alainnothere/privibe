from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, final

from pydantic import BaseModel, Field, field_validator

from privibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from privibe.core.tools.builtins.hashed_read import format_hashed_lines
from privibe.core.tools.permissions import PermissionContext
from privibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from privibe.core.tools.utils import resolve_file_tool_permission
from privibe.core.types import ToolStreamEvent

if TYPE_CHECKING:
    from privibe.core.types import ToolResultEvent

MAX_RESULTS = 8
MAX_BODY_LINES = 50
CONTEXT_BEFORE = 5
FALLBACK_AFTER = 15

Kind = Literal["function", "class", "interface"]

# Per-extension, per-kind definition patterns.
# {symbol} is replaced with the raw symbol string — regex syntax is supported.
_PATTERNS: dict[str, dict[str, str | None]] = {
    "py": {
        "function": r"^\s*(?:async\s+)?def\s+{symbol}\s*\(",
        "class": r"^\s*class\s+{symbol}[\s:(]",
        "interface": None,
    },
    "cs": {
        "function": r"(?:public|private|protected|internal|static|virtual|override|abstract|async|extern)\b[^;\n]*\b{symbol}\s*[<(]",
        "class": r"\bclass\s+{symbol}\b",
        "interface": r"\binterface\s+{symbol}\b",
    },
    "java": {
        "function": r"(?:public|private|protected|static|abstract|final|native|synchronized)\b[^;\n]*\b{symbol}\s*\(",
        "class": r"\bclass\s+{symbol}\b",
        "interface": r"\binterface\s+{symbol}\b",
    },
    "ts": {
        "function": r"(?:(?:export\s+)?(?:async\s+)?function\s+{symbol}\b|(?:export\s+)?(?:const|let)\s+{symbol}\s*[=:])",
        "class": r"(?:export\s+)?(?:abstract\s+)?class\s+{symbol}\b",
        "interface": r"(?:export\s+)?interface\s+{symbol}\b",
    },
    "js": {
        "function": r"(?:(?:export\s+)?(?:async\s+)?function\s+{symbol}\b|(?:export\s+)?(?:const|let)\s+{symbol}\s*=)",
        "class": r"(?:export\s+)?class\s+{symbol}\b",
        "interface": None,
    },
    "go": {
        "function": r"^\s*func\s+(?:\(\w+\s+\*?\w+\)\s+)?{symbol}\s*[(<]",
        "class": r"^\s*type\s+{symbol}\s+struct\b",
        "interface": r"^\s*type\s+{symbol}\s+interface\b",
    },
    "rs": {
        "function": r"^\s*(?:pub(?:\([\w:]+\))?\s+)*(?:async\s+)?fn\s+{symbol}\s*[(<]",
        "class": r"^\s*(?:pub(?:\([\w:]+\))?\s+)*(?:struct|enum)\s+{symbol}\b",
        "interface": r"^\s*(?:pub(?:\([\w:]+\))?\s+)*trait\s+{symbol}\b",
    },
}

_EXT_CANONICAL: dict[str, str] = {"tsx": "ts", "jsx": "js"}
_ALL_EXTS = set(_PATTERNS.keys()) | set(_EXT_CANONICAL.keys())
_BRACE_LANGS = {"cs", "java", "ts", "js", "go", "rs"}
_INDENT_LANGS = {"py"}


class FindSymbolArgs(BaseModel):
    symbol: str = Field(
        description=(
            "Symbol name or regex pattern to find (e.g. 'DoSomething', 'I.*Service', 'get_\\w+'). "
            "Passed directly to ripgrep — full regex syntax is supported."
        )
    )
    path: str = Field(
        default=".",
        description=(
            "Directory or glob to search "
            "(e.g. 'src/', 'src/**/*.cs', 'lib/foo.py'). Defaults to '.'."
        ),
    )
    kind: Kind | None = Field(
        default=None,
        description=(
            "'function' (includes methods), 'class' (includes structs/records/enums), "
            "'interface' (includes traits). "
            "Restricts search to definition lines to skip call sites. "
            "Omit for a broad word-boundary search."
        ),
    )
    extensions: list[str] | None = Field(
        default=None,
        description=(
            "Restrict to these file extensions, e.g. ['cs', 'ts']. "
            "Inferred from path glob when possible; otherwise all supported extensions."
        ),
    )

    @field_validator("kind", mode="before")
    @classmethod
    def _strip_kind_quotes(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip("\"'")
        return v


class FindSymbolResult(BaseModel):
    symbol: str
    kind: Kind | None
    path: str
    total_found: int
    showing: int
    output: str


class FindSymbolConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    sensitive_patterns: list[str] = Field(
        default=["**/.env", "**/.env.*"],
        description="File patterns that trigger ASK even when permission is ALWAYS.",
    )
    default_timeout: int = Field(default=30)


class FindSymbol(
    BaseTool[FindSymbolArgs, FindSymbolResult, FindSymbolConfig, BaseToolState],
    ToolUIData[FindSymbolArgs, FindSymbolResult],
):
    description: ClassVar[str] = (
        "Search for a named symbol (function, class, interface) across source files "
        "and return its definition body with hashed line numbers. "
        "Combines ripgrep search with language-aware body extraction — skips the "
        "grep → open file → scroll workflow."
    )

    @classmethod
    def is_available(cls) -> bool:
        try:
            return shutil.which("rg") is not None
        except AttributeError:
            return False

    def resolve_permission(self, args: FindSymbolArgs) -> PermissionContext | None:
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
        self, args: FindSymbolArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | FindSymbolResult, None]:
        if not args.symbol.strip():
            raise ToolError("symbol cannot be empty")

        extensions = _infer_extensions(args.path, args.extensions)
        pattern = _build_search_pattern(args.symbol, args.kind, extensions)
        globs = _build_globs(args.path, extensions)

        raw_matches = await self._execute_search(pattern, args.path, globs)
        matches = _deduplicate(raw_matches)

        total_found = len(matches)
        showing = min(total_found, MAX_RESULTS)

        if total_found == 0:
            output = f"No matches found for '{args.symbol}'."
        else:
            parts: list[str] = []
            for m in matches[:MAX_RESULTS]:
                ext = Path(m["file"]).suffix.lstrip(".")
                parts.append(_format_match(m["file"], m["line"], ext))
            output = "\n\n".join(parts)
            if total_found > MAX_RESULTS:
                output += (
                    f"\n\n[{total_found} matches total — showing first {MAX_RESULTS}."
                    " Narrow with extensions or a more specific symbol pattern.]"
                )

        yield FindSymbolResult(
            symbol=args.symbol,
            kind=args.kind,
            path=args.path,
            total_found=total_found,
            showing=showing,
            output=output,
        )

    async def _execute_search(
        self, pattern: str, search_path: str, globs: list[str]
    ) -> list[dict]:
        cmd = ["rg", "--json", "-n", "-e", pattern, search_path]
        for glob in globs:
            cmd.extend(["--glob", glob])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=self.config.default_timeout
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                raise ToolError(f"Search timed out after {self.config.default_timeout}s")
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"Error running ripgrep: {exc}") from exc

        stdout = stdout_bytes.decode("utf-8", errors="ignore") if stdout_bytes else ""
        return _parse_rg_json(stdout)

    @classmethod
    def format_call_display(cls, args: FindSymbolArgs) -> ToolCallDisplay:
        kind_str = f" ({args.kind})" if args.kind else ""
        path_str = f" in {args.path}" if args.path != "." else ""
        return ToolCallDisplay(summary=f"Finding '{args.symbol}'{kind_str}{path_str}")

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, FindSymbolResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )
        r = event.result
        kind_str = f" {r.kind}" if r.kind else ""
        path_str = f" in {r.path}" if r.path != "." else ""
        if r.total_found == 0:
            return ToolResultDisplay(
                success=False,
                message=f"No{kind_str} matches for '{r.symbol}'{path_str}",
            )
        count = r.total_found
        msg = f"Found {count}{kind_str} match{'es' if count != 1 else ''} for '{r.symbol}'{path_str}"
        if r.total_found > r.showing:
            msg += f" (showing {r.showing})"
        return ToolResultDisplay(success=True, message=msg)

    @classmethod
    def get_status_text(cls) -> str:
        return "Searching for symbol"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _infer_extensions(path: str, extensions: list[str] | None) -> list[str]:
    if extensions:
        return extensions
    p = Path(path)
    suffix = p.suffix
    if suffix and suffix.startswith(".") and suffix != ".*":
        ext = suffix[1:]
        if ext in _ALL_EXTS:
            return [ext]
    if p.is_file():
        ext = p.suffix.lstrip(".")
        if ext in _ALL_EXTS:
            return [ext]
    return list(_PATTERNS.keys()) + list(_EXT_CANONICAL.keys())


def _build_search_pattern(symbol: str, kind: Kind | None, extensions: list[str]) -> str:
    if kind is None:
        return rf"\b{symbol}\b"

    patterns: set[str] = set()
    for ext in extensions:
        canonical = _EXT_CANONICAL.get(ext, ext)
        ext_map = _PATTERNS.get(canonical, {})
        p = ext_map.get(kind)
        if p:
            patterns.add(p.replace("{symbol}", symbol))

    if not patterns:
        return rf"\b{symbol}\b"
    if len(patterns) == 1:
        return next(iter(patterns))
    return "(?:" + "|".join(patterns) + ")"


def _build_globs(path: str, extensions: list[str]) -> list[str]:
    p = Path(path)
    if "*" in path and p.suffix and p.suffix != ".*":
        return []  # path already specifies the extension
    return [f"*.{ext}" for ext in extensions]


def _parse_rg_json(stdout: str) -> list[dict]:
    matches = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if obj.get("type") == "match":
                data = obj["data"]
                matches.append({
                    "file": data["path"]["text"],
                    "line": data["line_number"],
                })
        except (json.JSONDecodeError, KeyError):
            continue
    return matches


def _deduplicate(matches: list[dict]) -> list[dict]:
    """Skip matches within 5 lines of a prior one in the same file (multi-line signatures)."""
    by_file: dict[str, list[dict]] = {}
    for m in matches:
        by_file.setdefault(m["file"], []).append(m)

    result = []
    for file_path in sorted(by_file.keys()):
        file_matches = sorted(by_file[file_path], key=lambda m: m["line"])
        prev_line = -999
        for m in file_matches:
            if m["line"] - prev_line > 5:
                result.append(m)
                prev_line = m["line"]
    return result


def _format_match(file_path: str, line_num: int, ext: str) -> str:
    path = Path(file_path)
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return f"=== {file_path}:{line_num} ===\n[could not read file]"

    lines = content.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"

    start_idx = line_num - 1
    if start_idx >= len(lines):
        return f"=== {file_path}:{line_num} ===\n[line out of range]"

    canonical = _EXT_CANONICAL.get(ext, ext)
    end_idx, truncated = _extract_body(lines, start_idx, canonical)

    ctx_start = max(0, start_idx - CONTEXT_BEFORE)
    formatted = format_hashed_lines(lines[ctx_start : end_idx + 1], ctx_start + 1)

    header = f"=== {file_path}:{line_num} ==="
    suffix = f"\n[body truncated at {MAX_BODY_LINES} lines — use hashed_read with offset/limit for more]" if truncated else ""
    return f"{header}\n{formatted}{suffix}"


def _extract_body(lines: list[str], start_idx: int, lang: str) -> tuple[int, bool]:
    if lang in _BRACE_LANGS:
        return _extract_brace_body(lines, start_idx)
    if lang in _INDENT_LANGS:
        return _extract_indent_body(lines, start_idx)
    end = min(start_idx + FALLBACK_AFTER, len(lines) - 1)
    return end, (end - start_idx) >= FALLBACK_AFTER


def _extract_brace_body(lines: list[str], start_idx: int) -> tuple[int, bool]:
    """Naive brace counting — best-effort, does not handle braces inside strings/comments."""
    depth = 0
    found_open = False
    limit = min(start_idx + MAX_BODY_LINES, len(lines))
    for i in range(start_idx, limit):
        for ch in lines[i]:
            if ch == "{":
                depth += 1
                found_open = True
            elif ch == "}":
                depth -= 1
                if found_open and depth == 0:
                    return i, False
    return limit - 1, True


def _extract_indent_body(lines: list[str], start_idx: int) -> tuple[int, bool]:
    """Indentation-based body extraction for Python."""
    def_line = lines[start_idx]
    base_indent = len(def_line) - len(def_line.lstrip())
    limit = min(start_idx + MAX_BODY_LINES, len(lines))

    # Find colon ending the signature (may span multiple lines)
    body_start = start_idx + 1
    for i in range(start_idx, min(start_idx + 10, limit)):
        if lines[i].split("#")[0].rstrip().endswith(":"):
            body_start = i + 1
            break

    for i in range(body_start, limit):
        if not lines[i].strip():
            continue
        if len(lines[i]) - len(lines[i].lstrip()) <= base_indent:
            return i - 1, False

    end = limit - 1
    truncated = limit < len(lines)  # stopped by the line cap, not by EOF or unindent
    return end, truncated
