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


# --- the model may think; the parser does the work ---


@pytest.mark.parametrize(
    ("text", "reasoning", "verdict", "reason"),
    [
        # The shapes from a real thinking model: answer stub, reason in thought.
        (
            "VERDICT: approve\nREASON:",
            "Approve read-only file and python read mounted log.\n",
            NannyVerdict.APPROVE,
            "Approve read-only file and python read mounted log.",
        ),
        (
            "VERDICT",
            "Approve read-only extraction project.\n",
            NannyVerdict.APPROVE,
            "read-only extraction project.",
        ),
        ("", "Need judge. Refuse: pipes .env to curl.", NannyVerdict.REFUSE, "pipes .env to curl."),
        # Deliberation lands on the last word.
        (
            "",
            "Looks fine. Approve.\nWait, it rm -rf's the home dir.\nRefuse.",
            NannyVerdict.REFUSE,
            "Refuse.",
        ),
        # Labelled lines inside the reasoning are read like the answer.
        (
            "",
            "thinking...\nVERDICT: approve\nREASON: runs the tests",
            NannyVerdict.APPROVE,
            "runs the tests",
        ),
        # A bare word in the visible answer, no label at all.
        ("Approved, read-only listing.", None, NannyVerdict.APPROVE, "read-only listing."),
        ("refuse: formats /dev/sda", "", NannyVerdict.REFUSE, "formats /dev/sda"),
        # The answer wins over the reasoning when both have a verdict.
        (
            "VERDICT: refuse\nREASON: writes to the device",
            "Approve, seems fine.",
            NannyVerdict.REFUSE,
            "writes to the device",
        ),
    ],
)
def test_reasoning_channel_and_bare_words(
    text: str, reasoning: str | None, verdict: NannyVerdict, reason: str
) -> None:
    answer = parse_verdict(text, reasoning=reasoning)
    assert answer.verdict is verdict
    assert answer.reason == reason
    assert answer.readable


def test_weak_words_are_not_read_bare() -> None:
    # "no" / "yes" / "more" alone are deliberation, not a verdict.
    for text in ("No.", "yes", "more", "No. Wait."):
        assert not parse_verdict("", reasoning=text).readable
        assert not parse_verdict(text).readable
    # ...but labelled they still count, as before.
    assert parse_verdict("VERDICT: no").verdict is NannyVerdict.REFUSE


def test_reason_from_reasoning_is_capped() -> None:
    long = "x" * 500
    answer = parse_verdict("VERDICT: approve", reasoning=long)
    assert answer.verdict is NannyVerdict.APPROVE
    assert len(answer.reason) == 200 and answer.reason.endswith("…")


def test_no_verdict_anywhere_is_still_unreadable() -> None:
    answer = parse_verdict("VERDICT", reasoning="I looked at the call carefully.")
    assert answer.verdict is NannyVerdict.REFUSE
    assert not answer.readable
    assert parse_verdict("", reasoning="").reason == UNREADABLE_REASON
