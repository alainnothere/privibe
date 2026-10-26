"""decisions.jsonl: one line per call the nanny was asked about.

This is the join between the main conversation and the nanny's: both are
append-only and neither is ever edited, so the record of what the nanny said
about which call lives beside them, keyed by the main tool_call_id.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any

from privibe.core.logger import logger

DECISIONS_FILENAME = "decisions.jsonl"


@dataclass
class TalkExchange:
    human: str
    nanny: str
    ask_index: int
    reply_index: int


@dataclass
class NannyDecision:
    tool_call_id: str
    tool_name: str
    agent: str
    verdict: str
    reason: str
    # Indices into the nanny conversation (0 is its system message).
    ask_index: int
    verdict_index: int
    header_session_id: str
    main_session_id: str
    # True when the nanny did not answer (timeout, error, unreadable text)
    # and the refusal is a stand-in; `notice` says what happened.
    failed: bool = False
    notice: str | None = None
    # Set once the human decided a refused call: "allow", "allow_always",
    # "no", "deny_always", "refuse", "cancelled".
    human_decision: str | None = None
    human_feedback: str | None = None
    talk: list[TalkExchange] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NannyDecision:
        talk = [TalkExchange(**t) for t in data.pop("talk", [])]
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(talk=talk, **known)

    @property
    def approved_by_nanny(self) -> bool:
        return self.verdict == "approve" and not self.failed


def append_decision(path: Path, decision: NannyDecision) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(decision.to_json() + "\n")
    except OSError as exc:
        logger.warning("nanny: could not write %s: %s", path, exc)


def read_decisions(path: Path) -> list[NannyDecision]:
    if not path.is_file():
        return []
    decisions: list[NannyDecision] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                decisions.append(NannyDecision.from_dict(json.loads(line)))
            except (ValueError, TypeError) as exc:
                logger.warning("nanny: skipping unreadable decision line: %s", exc)
    except OSError as exc:
        logger.warning("nanny: could not read %s: %s", path, exc)
    return decisions
