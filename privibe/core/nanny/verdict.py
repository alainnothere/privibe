"""The nanny's answer: a flat VERDICT / REASON pair, parsed leniently.

Flat text rather than JSON on purpose: small local models get two labelled
lines right far more often than a JSON object, and an unreadable answer costs
the user a prompt, never a silent approval.
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


def parse_verdict(text: str) -> NannyAnswer:
    """Parse the nanny's reply. The LAST verdict line wins, so a model that
    deliberates in prose before deciding is still read; a missing or unknown
    verdict is a refusal with UNREADABLE_REASON (never an approval).
    """
    raw = text or ""
    body = _strip_fences(raw)
    matches = list(_VERDICT_LINE.finditer(body))
    if not matches:
        return NannyAnswer.unreadable(raw)
    last = matches[-1]
    verdict = _VERDICT_WORDS.get(last.group("word").lower())
    if verdict is None:
        return NannyAnswer.unreadable(raw)

    tail = body[last.end() :]
    reasons = list(_REASON_LINE.finditer(tail))
    if reasons:
        start = reasons[0].start("text")
        reason_text = tail[start:]
    else:
        # No REASON label: whatever follows the verdict line is the reason.
        reason_text = tail.split("\n", 1)[1] if "\n" in tail else ""
    return NannyAnswer(verdict=verdict, reason=_clean_reason(reason_text), raw=raw)
