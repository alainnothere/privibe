"""Console approval prompt: g / gs hand the call to the nanny, and
/ensure-commands-safety tell <text> reaches it outside a review.
"""

from __future__ import annotations

import pytest

from privibe.cli.console_ui.app import ConsoleUI
from privibe.core.types import ApprovalResponse
from tests.cli.test_console_ui import StubAgentLoop, StubReview, ToolArgs, queue_replies


class StubNanny:
    model_alias = "reviewer"
    decisions: list = []

    def __init__(self) -> None:
        self.told: list[str] = []

    async def tell(self, text: str) -> str:
        self.told.append(text)
        return "Understood.\nI will refuse anything touching the Garmin."


@pytest.fixture
def ui() -> ConsoleUI:
    return ConsoleUI(StubAgentLoop())


@pytest.mark.asyncio
async def test_g_is_rejected_without_a_nanny(ui: ConsoleUI, capsys) -> None:
    queue_replies(ui, ["g", "y"])
    response, _ = await ui._approval_callback("bash", ToolArgs(), "t1", None)
    assert response is ApprovalResponse.YES
    assert "Please answer y, a, n, or d." in capsys.readouterr().out


@pytest.mark.asyncio
async def test_g_hands_the_call_to_the_nanny(ui: ConsoleUI, capsys) -> None:
    ui._nanny = StubNanny()
    queue_replies(ui, ["g"])
    response, feedback = await ui._approval_callback("bash", ToolArgs(), "t1", None)
    assert (response, feedback) == (ApprovalResponse.NANNY, None)


@pytest.mark.asyncio
async def test_gs_delegates_for_the_session(ui: ConsoleUI) -> None:
    ui._nanny = StubNanny()
    queue_replies(ui, ["gs"])
    response, _ = await ui._approval_callback("bash", ToolArgs(), "t1", None)
    assert response is ApprovalResponse.NANNY_ALWAYS


@pytest.mark.asyncio
async def test_g_is_not_offered_with_a_review(ui: ConsoleUI, capsys) -> None:
    ui._nanny = StubNanny()
    queue_replies(ui, ["g", "r"])
    response, _ = await ui._approval_callback(
        "bash", ToolArgs(), "t1", None, review=StubReview()
    )
    assert response is ApprovalResponse.NO
    assert "Please answer y, a, n, d, r, or t." in capsys.readouterr().out


@pytest.mark.asyncio
async def test_tell_command(ui: ConsoleUI, capsys) -> None:
    await ui._command_nanny("tell builds are fine, the Garmin is not")
    assert "Nanny is off" in capsys.readouterr().out

    nanny = StubNanny()
    ui._nanny = nanny
    await ui._command_nanny("tell")
    assert "Usage:" in capsys.readouterr().out

    await ui._command_nanny("tell builds are fine, the Garmin is not")
    assert nanny.told == ["builds are fine, the Garmin is not"]
    out = capsys.readouterr().out
    assert "[nanny] Understood." in out
    assert "[nanny] I will refuse anything touching the Garmin." in out
