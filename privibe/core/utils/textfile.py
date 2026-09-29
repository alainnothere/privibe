"""Byte-faithful text file codec shared by every file tool.

The law this module enforces: a line the model did not name never changes a
byte. Line endings, the byte order mark, the encoding and the presence of a
final newline are all read from the file, remembered, and written back exactly
as found. New lines blend in with their neighbours. Nothing here normalizes an
existing file, on purpose; a whole-file rewrite is a commit somebody signs, not
a side effect of editing line 40.

Readers get ``load_text_file`` (decode without ever showing a U+FEFF) and
writers get ``render`` / ``write_text_file`` (re-attach endings, BOM, encoding).
Internally every line is normalized to a bare ``"\\n"`` terminator so the edit
logic (hashes, shift maps, duplicate detection) stays ending-agnostic.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import difflib
from pathlib import Path
import re

import anyio

BOM = b"\xef\xbb\xbf"
_BINARY_SNIFF_BYTES = 8192

# Splits on the three real line terminators only. ``str.splitlines`` also
# splits on \v, \f, \x1c-\x1e, \x85,   and  , which a text editor
# treats as content, so it is never used on file bytes here.
_LINE_RE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+$")

# Fallback order after strict UTF-8. cp1252 is tried before latin-1 because
# every Windows-authored legacy file is cp1252; latin-1 decodes anything and
# so is the last resort, never a guess.
_FALLBACK_ENCODINGS = ("cp1252", "latin-1")


class BinaryFileError(ValueError):
    """The file has a NUL byte in its head; it is not text."""


class UnencodableError(ValueError):
    """New content has a character the file's encoding cannot represent."""

    def __init__(self, line_no: int, char: str, encoding: str) -> None:
        self.line_no = line_no
        self.char = char
        self.encoding = encoding
        super().__init__(
            f"line {line_no}: {char!r} (U+{ord(char):04X}) cannot be written in "
            f"this file's encoding ({encoding}); transliterate it, or pass "
            "encoding='utf-8' on a whole-file overwrite to convert the file"
        )


def split_keepends(text: str) -> list[str]:
    """Split on \\r\\n, \\r or \\n only, keeping each line's terminator."""
    if not text:
        return []
    return _LINE_RE.findall(text)


def _ending_of(line: str) -> str:
    if line.endswith("\r\n"):
        return "\r\n"
    if line.endswith("\n") or line.endswith("\r"):
        return line[-1]
    return ""


def _strip_ending(line: str) -> str:
    return line[: len(line) - len(_ending_of(line))]


def dominant_ending(endings: list[str], default: str = "\n") -> str:
    """Most common non-empty ending; ties go to the default."""
    counts = Counter(e for e in endings if e)
    if not counts:
        return default
    best = max(counts.values())
    winners = [e for e, c in counts.items() if c == best]
    if default in winners:
        return default
    return winners[0]


@dataclass
class TextFile:
    """A decoded text file plus everything needed to write it back faithfully.

    ``lines`` are normalized: every element ends with exactly one ``"\\n"``,
    including the last one even when the file had no final newline
    (``had_final_newline`` remembers that). ``endings`` holds each original
    line's real terminator, ``""`` for an unterminated last line.
    """

    lines: list[str]
    endings: list[str]
    encoding: str = "utf-8"
    bom: bool = False
    had_final_newline: bool = True
    newline: str = "\n"
    existed: bool = True
    # Set when a non-UTF-8 fallback decoded the file; surfaced as a note.
    encoding_note: str | None = None
    # Free-form facts a writer may want to report (mixed endings, etc.).
    notes: list[str] = field(default_factory=list)

    @property
    def mixed(self) -> bool:
        return len({e for e in self.endings if e}) > 1

    @property
    def text(self) -> str:
        """The normalized text (bare \\n endings) the edit logic works on."""
        return "".join(self.lines)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        default_newline: str = "\n",
        charset_hint: str | None = None,
    ) -> TextFile:
        if b"\x00" in data[:_BINARY_SNIFF_BYTES]:
            raise BinaryFileError("file contains NUL bytes; refusing to treat as text")
        bom = data.startswith(BOM)
        if bom:
            data = data[len(BOM) :]
        text, encoding, note = _decode(data, charset_hint)
        tf = cls.from_text(text, default_newline=default_newline)
        tf.bom = bom
        tf.encoding = encoding
        tf.encoding_note = note
        return tf

    @classmethod
    def from_text(cls, text: str, *, default_newline: str = "\n") -> TextFile:
        """Build from an already-decoded string whose endings are intact."""
        raw = split_keepends(text)
        endings = [_ending_of(line) for line in raw]
        lines = [_strip_ending(line) + "\n" for line in raw]
        had_final_newline = not raw or endings[-1] != ""
        tf = cls(
            lines=lines,
            endings=endings,
            had_final_newline=had_final_newline,
            newline=dominant_ending(endings, default_newline),
        )
        if tf.mixed:
            tf.notes.append(
                "file has mixed line endings; untouched lines kept as they were, "
                f"new lines follow their neighbour ({_name(tf.newline)} where "
                "there is none)"
            )
        return tf

    @classmethod
    def empty(
        cls,
        *,
        newline: str = "\n",
        encoding: str = "utf-8",
        bom: bool = False,
        final_newline: bool = True,
    ) -> TextFile:
        """The shape of a file that does not exist yet."""
        return cls(
            lines=[],
            endings=[],
            encoding=encoding,
            bom=bom,
            had_final_newline=final_newline,
            newline=newline,
            existed=False,
        )


def _decode(data: bytes, charset_hint: str | None) -> tuple[str, str, str | None]:
    try:
        return data.decode("utf-8"), "utf-8", None
    except UnicodeDecodeError:
        pass
    candidates: list[str] = []
    if charset_hint and charset_hint not in {"utf-8", "utf-8-bom", "utf-8-sig"}:
        candidates.append(charset_hint)
    candidates.extend(e for e in _FALLBACK_ENCODINGS if e not in candidates)
    for enc in candidates:
        try:
            text = data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        return (
            text,
            enc,
            f"file is not UTF-8; decoded as {enc} and will be written back as {enc}",
        )
    # latin-1 never fails, so this is unreachable in practice.
    return data.decode("latin-1"), "latin-1", "decoded as latin-1"


def _name(ending: str) -> str:
    return {"\r\n": "CRLF", "\n": "LF", "\r": "CR"}.get(ending, repr(ending))


def normalize_newlines(text: str) -> str:
    """Bare-\\n form of any string, for matching model text against a file."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def reattach_endings(tf: TextFile, new_lines: list[str]) -> list[str]:
    """Give each normalized new line its real terminator.

    Lines that survive from the original (by content, via difflib) keep their
    own ending. Inserted or replaced lines take the ending of the line just
    above them, else the ending of the next surviving original line, else the
    file's dominant ending. The original's final-newline state is preserved:
    a file that ended without one still ends without one.
    """
    old = tf.lines
    out: list[str] = []
    matcher = difflib.SequenceMatcher(None, old, new_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                out.append(_strip_ending(old[i1 + k]) + _with_default(tf.endings[i1 + k], tf.newline))
            continue
        if tag == "delete":
            continue
        if tag == "replace":
            # A replaced line keeps its own ending, positionally. Lines past
            # the replaced range (the block grew) continue with the ending of
            # the last replaced line.
            last = tf.newline
            for k in range(j1, j2):
                pos = i1 + (k - j1)
                if pos < i2:
                    last = _with_default(tf.endings[pos], tf.newline)
                out.append(_strip_ending(new_lines[k]) + last)
            continue
        # insert: the neighbour above, else the neighbour below, else dominant
        if out:
            neighbour = _ending_of(out[-1]) or tf.newline
        elif i2 < len(old):
            neighbour = _with_default(tf.endings[i2], tf.newline)
        else:
            neighbour = tf.newline
        for k in range(j1, j2):
            out.append(_strip_ending(new_lines[k]) + neighbour)
    if out and not tf.had_final_newline:
        out[-1] = _strip_ending(out[-1])
    return out


def _with_default(ending: str, default: str) -> str:
    return ending if ending else default


def render_text(tf: TextFile, new_lines: list[str]) -> str:
    """The exact string (endings re-attached, BOM as U+FEFF) to put on disk."""
    body = "".join(reattach_endings(tf, new_lines))
    return ("﻿" + body) if tf.bom else body


def render(tf: TextFile, new_lines: list[str]) -> bytes:
    """Encode for disk; raises UnencodableError naming the offending line."""
    body = "".join(reattach_endings(tf, new_lines))
    try:
        encoded = body.encode(tf.encoding)
    except UnicodeEncodeError as exc:
        line_no = body.count("\n", 0, exc.start) + 1
        raise UnencodableError(line_no, body[exc.start], tf.encoding) from exc
    return (BOM + encoded) if tf.bom else encoded


def load_text_file(
    path: Path,
    *,
    default_newline: str = "\n",
    charset_hint: str | None = None,
) -> TextFile:
    return TextFile.from_bytes(
        path.read_bytes(), default_newline=default_newline, charset_hint=charset_hint
    )


async def load_text_file_async(
    path: Path,
    *,
    default_newline: str = "\n",
    charset_hint: str | None = None,
) -> TextFile:
    data = await anyio.Path(path).read_bytes()
    return TextFile.from_bytes(
        data, default_newline=default_newline, charset_hint=charset_hint
    )


async def write_text_file(path: Path, tf: TextFile, new_lines: list[str]) -> int:
    """Render and write; returns the number of bytes written."""
    data = render(tf, new_lines)
    await anyio.Path(path).write_bytes(data)
    return len(data)


def content_to_lines(content: str) -> list[str]:
    """Model-supplied content to normalized lines (each ending in \\n)."""
    if not content:
        return []
    return [line + "\n" for line in normalize_newlines(content).split("\n")]


def trim_trailing_whitespace(lines: list[str]) -> tuple[list[str], int]:
    """Strip trailing spaces/tabs from normalized lines; returns (lines, count)."""
    out: list[str] = []
    trimmed = 0
    for line in lines:
        body = _strip_ending(line)
        stripped = body.rstrip(" \t")
        if stripped != body:
            trimmed += 1
        out.append(stripped + "\n")
    return out, trimmed


def ending_name(ending: str) -> str:
    return _name(ending)
