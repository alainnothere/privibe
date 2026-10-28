"""The nanny's standing instructions: the config brief sent right behind the
session header, and tell() for messages outside any review.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from privibe.core.nanny import NannyManager, NannyState
from privibe.core.types import Role
from tests.nanny.conftest import APPROVE, READY, build_nanny, nanny_backend
from tests.nanny.test_manager import ask, roles

BRIEF_OK = "Understood: refuse anything touching the Garmin, builds are fine."


def _sent(nanny: NannyManager) -> list[str]:
    """Every user message in the nanny conversation, in order."""
    return [m.content for m in nanny.loop.messages if m.role == Role.user]


@pytest.mark.asyncio
async def test_brief_rides_behind_the_header(tmp_path: Path) -> None:
    nanny = build_nanny(
        tmp_path, [READY, BRIEF_OK, APPROVE], brief="Refuse anything touching /media."
    )
    async with nanny.review(ask()) as review:
        assert review.approved
    # system, header, READY, brief, confirmation, ask, verdict.
    assert roles(nanny) == [Role.system] + [Role.user, Role.assistant] * 3
    header, brief, _ask = _sent(nanny)
    assert "main-session-1234" in header
    assert "Refuse anything touching /media." in brief
    assert "applies to every call" in brief
    assert nanny.loop.messages[4].content == BRIEF_OK


@pytest.mark.asyncio
async def test_blank_brief_is_no_brief(tmp_path: Path) -> None:
    nanny = build_nanny(tmp_path, [READY, APPROVE], brief="   ")
    assert nanny.brief is None
    async with nanny.review(ask()):
        pass
    assert len(_sent(nanny)) == 2


@pytest.mark.asyncio
async def test_tell_outside_a_review_sends_header_first(tmp_path: Path) -> None:
    nanny = build_nanny(tmp_path, [READY, BRIEF_OK, APPROVE])
    reply = await nanny.tell("Builds and tests are fine; the Garmin mount is not.")
    assert reply == BRIEF_OK
    header, told = _sent(nanny)
    assert "main-session-1234" in header
    assert "the Garmin mount is not" in told
    assert nanny.state is NannyState.IDLE
    # The next ask lands after it, in the same conversation.
    async with nanny.review(ask()) as review:
        assert review.approved
    assert len(_sent(nanny)) == 3


@pytest.mark.asyncio
async def test_tell_strips_and_refuses_empty(tmp_path: Path) -> None:
    nanny = build_nanny(tmp_path, [READY, BRIEF_OK])
    with pytest.raises(ValueError):
        await nanny.tell("   ")
    await nanny.tell("  spaced  ")
    assert _sent(nanny)[-1].count("spaced") == 1
    assert "  spaced" not in _sent(nanny)[-1]


@pytest.mark.asyncio
async def test_tell_waits_for_the_open_review(tmp_path: Path) -> None:
    nanny = build_nanny(tmp_path, [READY, APPROVE, BRIEF_OK])
    order: list[str] = []

    async def reviewer() -> None:
        async with nanny.review(ask()):
            order.append("review-open")
            await asyncio.sleep(0.1)
            order.append("review-close")

    async def teller() -> None:
        await asyncio.sleep(0.02)
        await nanny.tell("watch for uploads")
        order.append("told")

    await asyncio.gather(reviewer(), teller())
    assert order == ["review-open", "review-close", "told"]


@pytest.mark.asyncio
async def test_tell_when_the_nanny_is_down_returns_a_notice(tmp_path: Path) -> None:
    nanny = build_nanny(
        tmp_path, backend=nanny_backend([], exception=RuntimeError("down"))
    )
    reply = await nanny.tell("anything")
    assert reply.startswith("(nanny did not answer:")
    assert nanny.state is NannyState.IDLE
