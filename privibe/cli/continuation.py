"""/set-context-file and /resume-from-context-file, the parts both UIs share.

A conversation can carry a context file beside its name (meta.json). A
continuation is a fresh history that keeps the context file and takes the
name "<base> -- continuation <local time>", where <base> is the current name
with any earlier "-- continuation ..." tail removed, so chained continuations
do not stack.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

CONTINUATION_SEP = " -- continuation "
_CONTINUATION_TAIL = re.compile(r"\s+--\s+continuation\s+.*$")


def resolve_context_file(raw: str, cwd: Path | None = None) -> Path:
    """`~` and relative paths resolved against cwd; must be an existing file."""
    path = Path(raw.strip()).expanduser()
    if not path.is_absolute():
        path = (cwd or Path.cwd()) / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Context file not found: {path}")
    return path


def continuation_name(current: str, now: datetime | None = None) -> str:
    base = _CONTINUATION_TAIL.sub("", current.strip()) or current.strip()
    stamp = (now or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    return f"{base}{CONTINUATION_SEP}{stamp}"


def read_context_file_prompt(path: str) -> str:
    return f"Read {path}"
