"""Best-effort ASCII transliteration of model-written content.

Characters obey config, not the file: by default new content is written as
ASCII. Accents come off through NFKD, a small explicit table covers the
typographic and symbol characters a model reaches for, and everything else
(emoji, scripts with no ASCII twin) is dropped. Every replacement is reported
per line so a forgotten ``allow_unicode`` is a one-glance fix, never a mystery.

This deliberately is not a broad-brush transliterator. The table fits on a
screen; if a character is not in it and NFKD cannot reduce it, it is gone and
the note says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import unicodedata

# Explicit replacements. Kept small and readable on purpose.
_TABLE: dict[str, str] = {
    # dashes and hyphens
    "\u2010": "-",  # hyphen
    "\u2011": "-",  # non-breaking hyphen
    "\u2012": "-",  # figure dash
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2015": "-",  # horizontal bar
    "\u2212": "-",  # minus sign
    # quotes
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
    "\u2039": "<",
    "\u203a": ">",
    "\u00ab": "<<",
    "\u00bb": ">>",
    "\u2032": "'",  # prime
    "\u2033": '"',  # double prime
    # spaces and dots
    "\u00a0": " ",  # nbsp
    "\u2007": " ",
    "\u202f": " ",
    "\u2026": "...",
    "\u2022": "*",  # bullet
    "\u00b7": "*",  # middle dot
    # arrows
    "\u2192": "->",
    "\u2190": "<-",
    "\u2194": "<->",
    "\u21d2": "=>",
    "\u21d0": "<=",
    "\u21d4": "<=>",
    "\u2191": "^",
    "\u2193": "v",
    # comparison and math
    "\u2264": "<=",
    "\u2265": ">=",
    "\u2260": "!=",
    "\u2248": "~=",
    "\u00d7": "x",
    "\u00f7": "/",
    "\u00b1": "+/-",
    "\u221e": "inf",
    "\u00b0": " deg",
    "\u00b5": "u",  # micro sign
    "\u03bc": "u",  # greek mu
    "\u2030": "%o",
    # currency and legal
    "\u20ac": "EUR",
    "\u00a3": "GBP",
    "\u00a5": "JPY",
    "\u00a2": "c",
    "\u00a9": "(c)",
    "\u00ae": "(R)",
    "\u2122": "(TM)",
    # check marks that read as status in docs
    "\u2713": "[x]",
    "\u2714": "[x]",
    "\u2705": "[x]",
    "\u274c": "[ ]",
    "\u274e": "[ ]",
    "\u2717": "[ ]",
    "\u2718": "[ ]",
    # letters NFKD does not split
    "\u00df": "ss",
    "\u00e6": "ae",
    "\u00c6": "AE",
    "\u0153": "oe",
    "\u0152": "OE",
    "\u00f8": "o",
    "\u00d8": "O",
    "\u0142": "l",
    "\u0141": "L",
    "\u0111": "d",
    "\u0110": "D",
    "\u00fe": "th",
    "\u00de": "Th",
    "\u00f0": "d",
    "\u00d0": "D",
}

# Emoji presentation selector and zero-width joiners: pure noise once the
# emoji itself is gone.
_SILENT_DROP = {"\ufe0f", "\ufe0e", "\u200d", "\u200b", "\u200c", "\ufeff"}

_QUOTES = {"'", '"'}
_ASCII_LIMIT = 128


@dataclass(frozen=True)
class Replacement:
    line_no: int
    char: str
    replacement: str  # "" means dropped

    def describe(self) -> str:
        name = unicodedata.name(self.char, f"U+{ord(self.char):04X}").lower()
        if self.replacement == "":
            return f"{name} dropped"
        return f"{name} -> {self.replacement!r}"


@dataclass
class AsciifyResult:
    text: str
    replacements: list[Replacement] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.replacements)


def _transliterate_char(ch: str) -> str | None:
    """ASCII stand-in for one character, or None when it must be dropped."""
    if ch in _TABLE:
        return _TABLE[ch]
    if ch in _SILENT_DROP:
        return ""
    decomposed = unicodedata.normalize("NFKD", ch)
    ascii_part = "".join(c for c in decomposed if ord(c) < _ASCII_LIMIT)
    if ascii_part:
        return ascii_part
    return None


def asciify(text: str) -> AsciifyResult:
    """Transliterate ``text`` to ASCII, recording every change by line."""
    if text.isascii():
        return AsciifyResult(text=text)
    out: list[str] = []
    replacements: list[Replacement] = []
    line_no = 1
    for ch in text:
        if ch == "\n":
            line_no += 1
            out.append(ch)
            continue
        if ord(ch) < _ASCII_LIMIT:
            out.append(ch)
            continue
        stand_in = _transliterate_char(ch)
        if stand_in is None:
            replacements.append(Replacement(line_no, ch, ""))
            continue
        out.append(stand_in)
        if ch not in _SILENT_DROP:
            replacements.append(Replacement(line_no, ch, stand_in))
    return AsciifyResult(text="".join(out), replacements=replacements)


_NOTE_CAP = 12


def format_ascii_notes(
    replacements: list[Replacement], *, flag: str = "allow_unicode", cap: int = _NOTE_CAP
) -> list[str]:
    """One note per affected line, capped, plus the flag that keeps the bytes."""
    if not replacements:
        return []
    by_line: dict[int, list[Replacement]] = {}
    for r in replacements:
        by_line.setdefault(r.line_no, []).append(r)
    notes: list[str] = []
    shown = 0
    for line_no in sorted(by_line):
        if shown == cap:
            break
        items = by_line[line_no]
        seen: dict[tuple[str, str], int] = {}
        for r in items:
            seen[(r.char, r.replacement)] = seen.get((r.char, r.replacement), 0) + 1
        parts = []
        for (char, rep), count in seen.items():
            desc = Replacement(line_no, char, rep).describe()
            parts.append(f"{desc} x{count}" if count > 1 else desc)
        quote_risk = any(r.replacement in _QUOTES for r in items)
        suffix = " (check quoting)" if quote_risk else ""
        notes.append(f"line {line_no}: non-ASCII written as ASCII: {', '.join(parts)}{suffix}")
        shown += 1
    remaining = len(by_line) - shown
    if remaining > 0:
        notes.append(f"...and {remaining} more line{'s' if remaining != 1 else ''} with non-ASCII replacements")
    notes.append(f"(pass {flag}=true to write these characters verbatim as UTF-8)")
    return notes


# ---------------------------------------------------------------------------
# Session default. The config knob is the birth-time default; a slash command
# may flip it for the running session without touching tool schemas or the
# frozen prompt. Tri-state per-call flags win over both.
# ---------------------------------------------------------------------------

_session_override: bool | None = None


def set_session_ascii_override(value: bool | None) -> None:
    """Set (True/False) or clear (None) the session-wide ASCII default."""
    global _session_override
    _session_override = value


def session_ascii_override() -> bool | None:
    return _session_override


def resolve_ascii(explicit_allow_unicode: bool | None, config_ascii_default: bool) -> bool:
    """True when content must be written as ASCII.

    Precedence: an explicit per-call ``allow_unicode`` value, then the session
    override, then the configured default.
    """
    if explicit_allow_unicode is not None:
        return not explicit_allow_unicode
    if _session_override is not None:
        return _session_override
    return config_ascii_default
