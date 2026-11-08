"""Console: a bare /set-conversation-name or /set-context-file with a
value set loads it into the next prompt instead of clearing it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from privibe.cli.commands import BareClearGuard
from privibe.cli.console_ui.app import ConsoleUI
from tests.cli.test_console_ui import StubAgentLoop


class NamedLoop(StubAgentLoop):
    def __init__(self) -> None:
        super().__init__()
        self.conversation_name: str | None = None
        self.context_file: str | None = None

    async def set_conversation_name(self, name: str | None) -> None:
        self.conversation_name = name

    async def set_context_file(self, path: str | None) -> None:
        self.context_file = path


@pytest.fixture
def ui() -> ConsoleUI:
    return ConsoleUI(NamedLoop())


def test_guard_arms_for_one_submission_only() -> None:
    guard = BareClearGuard()
    guard.submission()
    assert not guard.bare_clears("name")
    guard.arm("name")
    guard.submission()
    assert guard.bare_clears("name")
    assert not guard.bare_clears("context_file")
    guard.submission()
    assert not guard.bare_clears("name")
    guard.arm("name")
    guard.disarm()
    guard.submission()
    assert not guard.bare_clears("name")


@pytest.mark.asyncio
async def test_bare_name_loads_then_clears(ui: ConsoleUI, capsys) -> None:
    await ui._handle_command("/set-conversation-name parser work")
    assert ui.agent_loop.conversation_name == "parser work"

    await ui._handle_command("/set-conversation-name")
    assert ui.agent_loop.conversation_name == "parser work"
    assert ui._next_input_default == "/set-conversation-name parser work"
    assert "empty the line" in capsys.readouterr().out

    await ui._handle_command("/name")
    assert ui.agent_loop.conversation_name is None
    assert "cleared" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_bare_context_file_loads_then_clears(
    ui: ConsoleUI, tmp_path: Path, capsys
) -> None:
    ctx = tmp_path / "ctx.md"
    ctx.write_text("# ctx")
    await ui._handle_command(f"/set-context-file {ctx}")
    assert ui.agent_loop.context_file == str(ctx.resolve())

    await ui._handle_command("/set-context-file")
    assert ui.agent_loop.context_file == str(ctx.resolve())
    assert ui._next_input_default == f"/set-context-file {ctx.resolve()}"

    await ui._handle_command("/context-file")
    assert ui.agent_loop.context_file is None
    assert "cleared" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_a_message_between_bare_calls_disarms(ui: ConsoleUI) -> None:
    await ui._handle_command("/name keep")
    await ui._handle_command("/name")
    ui._bare_clear.disarm()  # what _dispatch does for a plain message
    await ui._handle_command("/name")
    assert ui.agent_loop.conversation_name == "keep"
