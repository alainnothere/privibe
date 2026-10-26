from __future__ import annotations

from dataclasses import dataclass, field
import difflib
from pathlib import Path

from privibe.core.tools.builtins.hashed_read import format_hashed_lines
from privibe.core.utils.io import read_safe


class BundleTooLargeError(ValueError):
    def __init__(self, measured: int, limit: int) -> None:
        super().__init__(
            f"Pre-fed files render to {measured} bytes, over the "
            f"orchestrate.bundle_bytes limit of {limit}. Drop files from the "
            f"list or raise the limit; nothing was truncated."
        )
        self.measured = measured
        self.limit = limit


# Lines of untouched context shown around each changed region of a re-rendered
# file. Line numbers stay absolute, so the hashed edit tools address the rows
# exactly as they would in a full render.
CHANGED_CONTEXT_LINES = 12


def render_changed_file(path: str, old: str, new: str) -> str:
    """A changed file as the tail shows it: only the regions that differ from
    the snapshot, each with some context, in the same `line|hash|content` rows.
    A whole-file render would put a 50 KB file back in the prompt for a
    one-line edit, so this is what keeps the tail small.
    """
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    groups = list(matcher.get_grouped_opcodes(CHANGED_CONTEXT_LINES))
    if not groups:
        return render_file(path, new)
    header = (
        f"### {path} ({len(new_lines)} lines; changed regions only, "
        f"line numbers are absolute)\n"
    )
    chunks: list[str] = []
    for group in groups:
        start = group[0][3]
        end = group[-1][4]
        if end <= start:
            continue
        chunks.append(format_hashed_lines(new_lines[start:end], start + 1))
    return header + "\n...\n".join(chunks) + "\n"


def render_file(path: str, content: str) -> str:
    """One file as the prompt shows it: a header with the measured line count,
    then `line|hash|content` rows, the format the hashed edit tools address.
    """
    lines = content.splitlines(keepends=True)
    if not lines:
        return f"### {path} (0 lines, empty)\n"
    body = format_hashed_lines(lines, 1)
    return f"### {path} ({len(lines)} lines)\n{body}\n"


@dataclass
class Bundle:
    """The files pre-fed to every call, frozen as they were when the job
    started. `render_frozen()` is byte-identical for the life of the job; the
    files that have since changed are re-rendered separately for the tail.
    """

    paths: list[str]
    contents: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, paths: list[str], *, max_bytes: int) -> Bundle:
        contents: dict[str, str] = {}
        for p in paths:
            contents[p] = read_safe(Path(p), raise_on_error=True)
        bundle = cls(paths=list(paths), contents=contents)
        measured = len(bundle.render_frozen().encode("utf-8"))
        if measured > max_bytes:
            raise BundleTooLargeError(measured, max_bytes)
        return bundle

    def render_frozen(self) -> str:
        return "".join(render_file(p, self.contents[p]) for p in self.paths)

    def changed_files(self) -> list[str]:
        """Paths whose current content differs from the snapshot, in bundle
        order. A deleted file counts as changed.
        """
        changed: list[str] = []
        for p in self.paths:
            current = self._read_current(p)
            if current != self.contents[p]:
                changed.append(p)
        return changed

    def render_changed(self) -> str:
        parts: list[str] = []
        for p in self.changed_files():
            current = self._read_current(p)
            if current is None:
                parts.append(f"### {p} (deleted)\n")
            else:
                parts.append(render_changed_file(p, self.contents[p], current))
        return "".join(parts)

    @staticmethod
    def _read_current(path: str) -> str | None:
        file = Path(path)
        if not file.is_file():
            return None
        try:
            return read_safe(file, raise_on_error=True)
        except (UnicodeDecodeError, ValueError, OSError):
            return None
