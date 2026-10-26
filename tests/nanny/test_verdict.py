from __future__ import annotations

import pytest

from privibe.core.nanny import NannyVerdict, parse_verdict
from privibe.core.nanny.verdict import NO_REASON, UNREADABLE_REASON


@pytest.mark.parametrize(
    ("text", "verdict", "reason"),
    [
        ("VERDICT: approve\nREASON: lists files", NannyVerdict.APPROVE, "lists files"),
        (
            "verdict: Refuse\nreason: deletes the build dir",
            NannyVerdict.REFUSE,
            "deletes the build dir",
        ),
        (
            "**VERDICT:** more\n**REASON:** need the rest",
            NannyVerdict.MORE,
            "need the rest",
        ),
        ("VERDICT - approved\nREASON - fine", NannyVerdict.APPROVE, "fine"),
        ("```\nVERDICT: approve\nREASON: ok\n```", NannyVerdict.APPROVE, "ok"),
        ("VERDICT: deny\nREASON: uploads .env", NannyVerdict.REFUSE, "uploads .env"),
        (
            "VERDICT: approve\nREASON: a\nmulti line\nreason",
            NannyVerdict.APPROVE,
            "a multi line reason",
        ),
    ],
)
def test_parse_known_shapes(text: str, verdict: NannyVerdict, reason: str) -> None:
    answer = parse_verdict(text)
    assert answer.verdict is verdict
    assert answer.reason == reason
    assert answer.readable


def test_last_verdict_line_wins() -> None:
    text = "I first thought VERDICT: approve but no.\nVERDICT: refuse\nREASON: formats the disk"
    answer = parse_verdict(text)
    assert answer.verdict is NannyVerdict.REFUSE
    assert answer.reason == "formats the disk"


def test_missing_reason_uses_following_text_or_placeholder() -> None:
    assert parse_verdict("VERDICT: approve\nruns the tests").reason == "runs the tests"
    assert parse_verdict("VERDICT: approve").reason == NO_REASON


@pytest.mark.parametrize(
    "text", ["", "sure, go ahead", "VERDICT: maybe\nREASON: x", "REASON: ok"]
)
def test_unreadable_is_a_refusal_never_an_approval(text: str) -> None:
    answer = parse_verdict(text)
    assert answer.verdict is NannyVerdict.REFUSE
    assert answer.reason == UNREADABLE_REASON
    assert not answer.readable
    assert answer.raw == text
