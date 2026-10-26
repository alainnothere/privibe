"""How one tool call is shown to the nanny.

bash: the whole command. write_file: path facts plus the content, whole when
short, otherwise first/middle/last ten lines with the rest available through
a `more` verdict. search_replace: the block whole, capped. Anything else: the
tool's own call display. The nanny never sees tool output.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from privibe.core.tools.permissions import RequiredPermission

PREVIEW_LINES = 10
SEARCH_REPLACE_CAP = 6000
GENERIC_CAP = 3000


@dataclass(frozen=True)
class NannyCallPresentation:
    text: str
    # The content that was shortened out of `text`, when any; answers a
    # `more` verdict. None means `text` already shows everything.
    full_text: str | None = None


def _summarise_lines(content: str, full_lines: int) -> tuple[str, bool]:
    lines = content.splitlines()
    total = len(lines)
    if total <= max(full_lines, 3 * PREVIEW_LINES):
        return content, False
    head = lines[:PREVIEW_LINES]
    mid_start = max(PREVIEW_LINES, total // 2 - PREVIEW_LINES // 2)
    mid = lines[mid_start : mid_start + PREVIEW_LINES]
    tail = lines[-PREVIEW_LINES:]
    gap1 = mid_start - PREVIEW_LINES
    gap2 = total - PREVIEW_LINES - (mid_start + PREVIEW_LINES)
    parts = [
        *head,
        f"... ({gap1} lines omitted)",
        *mid,
        f"... ({gap2} lines omitted)",
        *tail,
    ]
    return "\n".join(parts), True


def _file_facts(path_str: str) -> str:
    path = Path(path_str).expanduser()
    try:
        if path.exists():
            size = path.stat().st_size
            return f"exists, {size} bytes"
        return "does not exist yet"
    except OSError:
        return "unreadable"


def _cap(text: str, cap: int) -> tuple[str, bool]:
    if len(text) <= cap:
        return text, False
    return text[:cap] + f"\n... (truncated, {len(text)} chars total)", True


def _generic(tool_name: str, tool_class: Any, args: BaseModel) -> str:
    format_display = getattr(tool_class, "format_call_display", None)
    if format_display is not None:
        try:
            display = format_display(args)
            parts = [f"{tool_name}: {display.summary}"]
            if display.content:
                parts.append(_cap(display.content, GENERIC_CAP)[0])
            return "\n".join(parts)
        except Exception:
            pass
    try:
        dumped = args.model_dump_json()
    except Exception:
        dumped = json.dumps(str(args))
    return f"{tool_name} args: {_cap(dumped, GENERIC_CAP)[0]}"


def present_call(
    tool_name: str, tool_class: Any, args: BaseModel, *, write_full_lines: int
) -> NannyCallPresentation:
    data = args.model_dump() if isinstance(args, BaseModel) else {}
    match tool_name:
        case "bash":
            return NannyCallPresentation(
                text=f"bash command:\n{data.get('command', '')}"
            )
        case "write_file":
            path = str(data.get("path", ""))
            content = str(data.get("content", ""))
            shown, shortened = _summarise_lines(content, write_full_lines)
            lines = (
                f"write_file: {path} ({_file_facts(path)}, "
                f"overwrite={'yes' if data.get('overwrite') else 'no'}, "
                f"{len(content.splitlines())} lines)\n"
                f"content:\n{shown}"
            )
            return NannyCallPresentation(
                text=lines, full_text=content if shortened else None
            )
        case "search_replace":
            path = str(data.get("file_path", ""))
            content = str(data.get("content", ""))
            shown, capped = _cap(content, SEARCH_REPLACE_CAP)
            return NannyCallPresentation(
                text=f"search_replace in {path} ({_file_facts(path)}):\n{shown}",
                full_text=content if capped else None,
            )
        case _:
            return NannyCallPresentation(text=_generic(tool_name, tool_class, args))


def present_permissions(required_permissions: list[RequiredPermission] | None) -> str:
    if not required_permissions:
        return ""
    return (
        "Permissions this call needs: "
        + "; ".join(rp.label for rp in required_permissions)
        + "\n"
    )
