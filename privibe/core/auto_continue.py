"""Auto-continue: notice a turn that ended without finishing, and nudge.

A turn ends on one rule: the model replied with no tool call. Two things
make that rule lie. The server cut the reply at its token cap (finish
reason "length"); the model announced work and stopped ("I'll now update
the parser.") the way some models do when they expect a "continue". This
module decides, from the final assistant message alone, whether a nudge is
due and which one. The loop owns the counting, the cap and the appending.

Everything here is a heuristic except the length check, and a false
positive costs one extra model turn that says "nothing more to do". A
false negative costs the user typing "continue", which is where we were.
"""

from __future__ import annotations

from enum import StrEnum
import re

from privibe.core.types import LLMMessage


class AutoContinueMode(StrEnum):
    OFF = "off"
    # Only when the server reports it cut the reply off.
    TRUNCATED = "truncated"
    # Truncation plus the announced-but-did-not heuristics.
    HEURISTIC = "heuristic"


class NudgeKind(StrEnum):
    TRUNCATED = "truncated"
    UNFINISHED = "unfinished"


FINISH_LENGTH = "length"

# The last non-empty line of a reply that opens like a plan of action. Case
# insensitive, anchored at the line start after optional list markers.
_INTENT_OPENERS = (
    r"let me\b",
    r"let's\b",
    r"i'?ll\b",
    r"i will\b",
    r"i'?m going to\b",
    r"i am going to\b",
    r"i'?m now going\b",
    r"now,? (?:i|let)\b",
    r"next,? (?:i|let)\b",
    r"first,? (?:i|let)\b",
    r"then,? (?:i|let)\b",
    r"moving on\b",
    r"proceeding\b",
    r"starting (?:with|by|on)\b",
    r"going to\b",
    r"time to\b",
)
_INTENT_LINE = re.compile(
    r"^[\s*_`#>\-\d.)]*(?:" + "|".join(_INTENT_OPENERS) + r")",
    re.IGNORECASE,
)
# A question to the user is not unfinished work: the model is waiting on
# them, and a nudge would answer in their place.
_ASKS_USER = re.compile(r"\?\s*$|\b(?:should i|do you want|would you like|shall i)\b", re.IGNORECASE)


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _last_sentence(line: str) -> str:
    parts = [p for p in _SENTENCE_SPLIT.split(line) if p.strip()]
    return parts[-1].strip() if parts else ""


def _open_fence(text: str) -> bool:
    return text.count("```") % 2 == 1


def looks_unfinished(content: str | None) -> bool:
    """True when the visible reply reads as announced-but-not-done.

    Signals, any one of which is enough: an unclosed code fence; a reply
    ending in a colon or an ellipsis; a last sentence that opens like a
    plan ("Let me ...", "I'll now ...", "Found it. Next, I will ..."). A
    reply that asks the user something is never unfinished, whatever else
    it looks like.
    """
    text = (content or "").rstrip()
    if not text:
        return False
    if _ASKS_USER.search(_last_line(text)):
        return False
    if _open_fence(text):
        return True
    if text.endswith((":", "...", "…")):
        return True
    return bool(_INTENT_LINE.match(_last_sentence(_last_line(text))))


def nudge_kind(message: LLMMessage, mode: AutoContinueMode | str) -> NudgeKind | None:
    """Which nudge the final assistant message earns under `mode`, or None.

    The message must be an assistant reply with no tool calls (the loop
    only asks for those); a reply the server truncated earns TRUNCATED in
    either active mode, an unfinished-looking one earns UNFINISHED only in
    HEURISTIC mode.
    """
    mode = AutoContinueMode(mode)
    if mode is AutoContinueMode.OFF:
        return None
    if message.tool_calls:
        return None
    finish = message.meta.finish_reason if message.meta is not None else None
    if finish == FINISH_LENGTH:
        return NudgeKind.TRUNCATED
    if mode is AutoContinueMode.HEURISTIC and looks_unfinished(message.content):
        return NudgeKind.UNFINISHED
    return None


__all__ = [
    "AutoContinueMode",
    "FINISH_LENGTH",
    "NudgeKind",
    "looks_unfinished",
    "nudge_kind",
]
