"""The nanny's answer: a flat VERDICT / REASON pair, parsed leniently.

Flat text rather than JSON on purpose: small local models get two labelled
lines right far more often than a JSON object, and an unreadable answer costs
the user a prompt, never a silent approval.

The model is allowed to think. A thinking model often writes the whole
answer in its reasoning channel and then a stub ("VERDICT: approve\nREASON:",
or just "VERDICT") in the visible one, so the parser reads both: the visible
answer first, the reasoning as the fallback for the verdict and for a
missing reason. In every channel the LAST verdict wins, because that is
where a deliberating model lands. Nothing it writes is a rule it must obey;
only an answer with no verdict anywhere is unreadable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re


class NannyVerdict(StrEnum):
    APPROVE = "approve"
    REFUSE = "refuse"
    MORE = "more"


UNREADABLE_REASON = "nanny answer could not be read"
NO_REASON = "no reason given"

_VERDICT_WORDS: dict[str, NannyVerdict] = {
    "approve": NannyVerdict.APPROVE,
    "approved": NannyVerdict.APPROVE,
    "allow": NannyVerdict.APPROVE,
    "allowed": NannyVerdict.APPROVE,
    "yes": NannyVerdict.APPROVE,
    "refuse": NannyVerdict.REFUSE,
    "refused": NannyVerdict.REFUSE,
    "deny": NannyVerdict.REFUSE,
    "denied": NannyVerdict.REFUSE,
    "reject": NannyVerdict.REFUSE,
    "rejected": NannyVerdict.REFUSE,
    "no": NannyVerdict.REFUSE,
    "more": NannyVerdict.MORE,
}

# "VERDICT: approve", "**Verdict:** refuse", "verdict - more". The word after
# the label decides; anything else on the line is ignored.
_VERDICT_LINE = re.compile(
    r"^[\s*_`#>-]*verdict[\s*_`]*[:\-][\s*_`]*(?P<word>[a-z]+)",
    re.IGNORECASE | re.MULTILINE,
)
_REASON_LINE = re.compile(
    r"^[\s*_`#>-]*reason[\s*_`]*[:\-][\s*_`]*(?P<text>.*)$",
    re.IGNORECASE | re.MULTILINE,
)

# A sentence that opens with a decisive verdict word, at a line start or
# after sentence punctuation: "Approve.", "Refuse: formats the disk",
# "Need judge. Refuse: pipes .env". The weak words (yes, no, more) are not
# accepted bare: "No. Wait." mid-deliberation must not count. The rest of
# the line is the reason.
_BARE_WORDS = (
    "approve",
    "approved",
    "allow",
    "allowed",
    "refuse",
    "refused",
    "deny",
    "denied",
    "reject",
    "rejected",
)
_BARE_LINE = re.compile(
    r"(?:^[\s*_`#>-]*|[.!?;]\s+)(?P<word>"
    + "|".join(_BARE_WORDS)
    + r")\b[\s*_`:.,;\-]*(?P<text>.*)$",
    re.IGNORECASE | re.MULTILINE,
)

# A reason lifted from free-form reasoning is capped: it is a log line and a
# prompt line, not an essay.
REASON_FROM_REASONING_CAP = 200


@dataclass(frozen=True)
class NannyAnswer:
    verdict: NannyVerdict
    reason: str
    raw: str
    # False when the text had no readable verdict and REFUSE was substituted.
    readable: bool = True

    @classmethod
    def unreadable(cls, raw: str) -> NannyAnswer:
        return cls(
            verdict=NannyVerdict.REFUSE,
            reason=UNREADABLE_REASON,
            raw=raw,
            readable=False,
        )


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        inner = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        return inner.rsplit("```", 1)[0]
    return text


def _clean_reason(text: str) -> str:
    text = " ".join(text.split())
    return text.strip(" *_`").strip() or NO_REASON


def _labelled(body: str) -> tuple[NannyVerdict, str] | None:
    """The last VERDICT: line and the reason text after it, or None."""
    matches = list(_VERDICT_LINE.finditer(body))
    if not matches:
        return None
    last = matches[-1]
    verdict = _VERDICT_WORDS.get(last.group("word").lower())
    if verdict is None:
        return None
    tail = body[last.end() :]
    reasons = list(_REASON_LINE.finditer(tail))
    if reasons:
        reason_text = tail[reasons[0].start("text") :]
    else:
        # No REASON label: whatever follows the verdict line is the reason.
        reason_text = tail.split("\n", 1)[1] if "\n" in tail else ""
    return verdict, reason_text


def _bare(body: str) -> tuple[NannyVerdict, str] | None:
    """The last line that opens with a decisive verdict word, or None."""
    matches = list(_BARE_LINE.finditer(body))
    if not matches:
        return None
    last = matches[-1]
    return _VERDICT_WORDS[last.group("word").lower()], last.group("text")


def _reason_from_reasoning(reasoning: str) -> str:
    """A REASON: line in the reasoning, else its last non-empty line."""
    reasons = list(_REASON_LINE.finditer(reasoning))
    if reasons:
        text = reasons[-1].group("text")
    else:
        lines = [line.strip() for line in reasoning.splitlines() if line.strip()]
        text = lines[-1] if lines else ""
    cleaned = _clean_reason(text)
    if len(cleaned) > REASON_FROM_REASONING_CAP:
        cleaned = cleaned[: REASON_FROM_REASONING_CAP - 1] + "…"
    return cleaned


def parse_verdict(text: str, reasoning: str | None = None) -> NannyAnswer:
    """Parse the nanny's reply: the visible answer, then its reasoning.

    Order: a labelled verdict in the answer; a bare verdict word opening a
    line of the answer; a labelled verdict in the reasoning; a bare one in
    the reasoning. The last verdict in whichever channel decides wins. A
    reason missing from the answer is taken from the reasoning. No verdict
    anywhere is a refusal with UNREADABLE_REASON, never an approval.
    """
    raw = text or ""
    body = _strip_fences(raw)
    thought = _strip_fences(reasoning or "")

    found = _labelled(body) or _bare(body)
    if found is None and thought:
        found = _labelled(thought) or _bare(thought)
    if found is None:
        return NannyAnswer.unreadable(raw)

    verdict, reason_text = found
    reason = _clean_reason(reason_text)
    if reason == NO_REASON and thought:
        reason = _reason_from_reasoning(thought)
    return NannyAnswer(verdict=verdict, reason=reason, raw=raw)
