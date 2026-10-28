"""The TUI approval prompt when a nanny is on but did not see the call: two
"ask the nanny" rows, keys g/G, and the NANNY / NANNY_ALWAYS answers.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel
import pytest

from privibe.cli.textual_ui.widgets.approval_app import ApprovalApp
from privibe.core.tools.permissions import PermissionScope, RequiredPermission
from privibe.core.types import ApprovalResponse
from tests.cli.test_approval_app_nanny import StubReview
from tests.conftest import build_test_vibe_app


class ToolArgs(BaseModel):
    command: str = "cp /media/GARMIN/x /tmp/x"


ESCALATED = [
    RequiredPermission(
        scope=PermissionScope.OUTSIDE_DIRECTORY,
        invocation_pattern="/media/GARMIN/x",
        session_pattern="/media/GARMIN/*",
        label="outside workdir (/media/GARMIN/*)",
        escalated=True,
    )
]


async def open_prompt(app, review=None, permissions=None):
    app._pending_approval = asyncio.Future()
    app._pending_review = review
    await app._switch_to_approval_app("bash", ToolArgs(), permissions, review)


@pytest.mark.asyncio
async def test_no_nanny_no_delegate_rows() -> None:
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await open_prompt(app, permissions=ESCALATED)
        await pilot.pause(0.2)
        widget = app.query_one(ApprovalApp)
        assert widget.option_count == 4
        await pilot.press("g")
        await pilot.press("G")
        await pilot.pause(0.1)
        assert not app._pending_approval.done()


@pytest.mark.asyncio
async def test_nanny_on_without_review_offers_delegation() -> None:
    app = build_test_vibe_app()
    app._nanny = object()  # attached, as far as the prompt is concerned
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await open_prompt(app, permissions=ESCALATED)
        await pilot.pause(0.2)
        widget = app.query_one(ApprovalApp)
        assert widget.has_delegate and not widget.has_review
        assert widget.option_count == 6
        texts = [t for t, _ in widget._option_texts()]
        assert texts[4] == "Ask the nanny about this call"
        assert texts[5] == (
            "Ask the nanny about calls like this for this session: "
            "outside workdir (/media/GARMIN/*)"
        )
        assert "g ask the nanny" in str(widget.help_widget.content)
        # Review-only letters are inert here.
        await pilot.press("r")
        await pilot.press("t")
        await pilot.pause(0.1)
        assert not app._pending_approval.done()

        await pilot.press("g")
        await pilot.pause(0.2)
        assert app._pending_approval.result() == (ApprovalResponse.NANNY, None)


@pytest.mark.asyncio
async def test_capital_g_and_row_six_delegate_for_the_session() -> None:
    app = build_test_vibe_app()
    app._nanny = object()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await open_prompt(app, permissions=ESCALATED)
        await pilot.pause(0.2)
        await pilot.press("G")
        await pilot.pause(0.2)
        assert app._pending_approval.result() == (ApprovalResponse.NANNY_ALWAYS, None)

    app = build_test_vibe_app()
    app._nanny = object()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await open_prompt(app)
        await pilot.pause(0.2)
        widget = app.query_one(ApprovalApp)
        assert widget._option_texts()[5][0] == "Ask the nanny about bash calls for this session"
        await pilot.press("6")
        await pilot.pause(0.2)
        assert app._pending_approval.result() == (ApprovalResponse.NANNY_ALWAYS, None)


@pytest.mark.asyncio
async def test_with_a_review_the_rows_are_refuse_and_talk_as_before() -> None:
    app = build_test_vibe_app()
    app._nanny = object()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await open_prompt(app, review=StubReview(), permissions=ESCALATED)
        await pilot.pause(0.2)
        widget = app.query_one(ApprovalApp)
        assert widget.has_review and not widget.has_delegate
        assert widget._option_texts()[4][0] == "Refuse, and tell the agent the nanny's reason"
        await pilot.press("g")
        await pilot.pause(0.1)
        assert not app._pending_approval.done()
        await pilot.press("r")
        await pilot.pause(0.2)
        response, feedback = app._pending_approval.result()
        assert response is ApprovalResponse.NO
        assert feedback.startswith("Refused by the reviewer")
