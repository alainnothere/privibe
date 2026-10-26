"""The TUI approval prompt with and without a nanny review attached."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel
import pytest

from privibe.cli.textual_ui.app import BottomApp
from privibe.cli.textual_ui.widgets.approval_app import ApprovalApp
from privibe.core.types import ApprovalResponse
from tests.conftest import build_test_vibe_app


class ToolArgs(BaseModel):
    command: str = "rm -rf build"


class StubReview:
    def __init__(
        self, reason: str = "deletes a directory", failed: bool = False
    ) -> None:
        self.reason = reason
        self.failed = failed
        self.notice = "nanny did not answer" if failed else None
        self.human_choice: str | None = None
        self.talked: list[str] = []

    @property
    def approved(self) -> bool:
        return False

    def refusal_feedback(self) -> str:
        return f"Refused by the reviewer: {self.reason}"

    async def talk(self, text: str) -> str:
        self.talked.append(text)
        return "Understood: only for the build directory."


async def open_prompt(app, review=None):
    app._pending_approval = asyncio.Future()
    app._pending_review = review
    await app._switch_to_approval_app("bash", ToolArgs(), None, review)


@pytest.mark.asyncio
async def test_without_review_the_prompt_is_unchanged() -> None:
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await open_prompt(app)
        await pilot.pause(0.2)
        widget = app.query_one(ApprovalApp)
        assert widget.option_count == 4
        assert len(widget.option_widgets) == 4
        assert widget.nanny_widget is None
        # Nanny keys are inert without a review.
        await pilot.press("r")
        await pilot.press("t")
        await pilot.pause(0.1)
        assert not app._pending_approval.done()
        await pilot.press("escape")
        await pilot.pause(0.2)
        response, feedback = app._pending_approval.result()
        assert response is ApprovalResponse.NO
        assert "cancelled" in feedback.lower()


@pytest.mark.asyncio
async def test_refuse_option_returns_the_nanny_reason() -> None:
    app = build_test_vibe_app()
    review = StubReview()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await open_prompt(app, review)
        await pilot.pause(0.2)
        widget = app.query_one(ApprovalApp)
        assert widget.option_count == 6
        assert "nanny refused: deletes a directory" in str(widget.nanny_widget.content)
        await pilot.press("r")
        await pilot.pause(0.2)
        response, feedback = app._pending_approval.result()
        assert response is ApprovalResponse.NO
        assert feedback == "Refused by the reviewer: deletes a directory"
        assert review.human_choice == "refuse"


@pytest.mark.asyncio
async def test_talk_then_allow() -> None:
    app = build_test_vibe_app()
    review = StubReview()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await open_prompt(app, review)
        await pilot.pause(0.2)
        widget = app.query_one(ApprovalApp)
        await pilot.press("t")
        await pilot.pause(0.2)
        assert widget.talk_mode
        assert widget.talk_input.has_focus
        for ch in "only this once":
            await pilot.press(ch if ch != " " else "space")
        await pilot.press("enter")
        await pilot.pause(0.4)
        assert review.talked == ["only this once"]
        assert "nanny: Understood: only for the build directory." in widget._nanny_log
        assert not app._pending_approval.done()
        # Escape leaves talk mode; a second Escape would be the reject.
        await pilot.press("escape")
        await pilot.pause(0.2)
        assert not widget.talk_mode
        assert not app._pending_approval.done()
        await pilot.press("y")
        await pilot.pause(0.2)
        response, _ = app._pending_approval.result()
        assert response is ApprovalResponse.YES
        assert review.human_choice == "allow"


@pytest.mark.asyncio
async def test_failed_review_shows_the_notice_and_existing_keys_record_the_choice() -> (
    None
):
    app = build_test_vibe_app()
    review = StubReview(failed=True)
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await open_prompt(app, review)
        await pilot.pause(0.2)
        widget = app.query_one(ApprovalApp)
        assert "nanny: nanny did not answer" in str(widget.nanny_widget.content)
        await pilot.press("2")
        await pilot.pause(0.2)
        response, _ = app._pending_approval.result()
        assert response is ApprovalResponse.YES
        assert review.human_choice == "allow_always"
        assert app._current_bottom_app == BottomApp.Approval or True
