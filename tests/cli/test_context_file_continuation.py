"""/set-context-file and /resume-from-context-file (2026-10-30).

The context file is a per-conversation record beside the name; a
continuation is /clear plus the same file, a "-- continuation <time>" name,
and "Read <file>" preloaded in the input box, never sent by itself.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import pytest

from privibe.cli.continuation import continuation_name, resolve_context_file
from privibe.cli.textual_ui.widgets.chat_input import ChatInputContainer
from privibe.cli.textual_ui.widgets.messages import ErrorMessage, UserCommandMessage
from privibe.core.config import SessionLoggingConfig
from privibe.core.session.session_loader import SessionLoader
from privibe.core.session.session_logger import SessionLogger
from privibe.core.types import LLMMessage, Role
from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_app,
    build_test_vibe_config,
)


class TestHelpers:
    def test_continuation_name_appends_local_stamp(self) -> None:
        now = datetime(2026, 9, 30, 14, 24, 58)
        assert (
            continuation_name("Really long and tedious work", now)
            == "Really long and tedious work -- continuation 2026-09-30 14:24:58"
        )

    def test_continuation_of_a_continuation_does_not_stack(self) -> None:
        now = datetime(2026, 10, 2, 9, 11, 3)
        assert (
            continuation_name("work -- continuation 2026-09-30 14:24:58", now)
            == "work -- continuation 2026-10-02 09:11:03"
        )

    def test_resolve_relative_and_tilde(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / "ctx.md").write_text("x")
        assert resolve_context_file("ctx.md", cwd=tmp_path) == tmp_path / "ctx.md"
        monkeypatch.setenv("HOME", str(tmp_path))
        assert resolve_context_file("~/ctx.md") == tmp_path / "ctx.md"
        with pytest.raises(FileNotFoundError):
            resolve_context_file("missing.md", cwd=tmp_path)


class TestRecordOnDisk:
    @pytest.mark.asyncio
    async def test_persists_resumes_and_lists(self, tmp_path: Path) -> None:
        save_dir = tmp_path / "sessions"
        save_dir.mkdir()
        cfg = SessionLoggingConfig(
            save_dir=str(save_dir), session_prefix="test", enabled=True
        )
        config = build_test_vibe_config(session_logging=cfg)
        agent = build_test_agent_loop(config=config)
        agent.messages.add(LLMMessage(role=Role.user, content="hello"))
        await agent.set_context_file("/tmp/ctx.md")
        await agent.messages.save()
        await agent.session_logger.save_interaction(
            agent.messages, agent.stats, config, agent.tool_manager, agent.agent_profile
        )
        session_dir = agent.session_logger.session_dir
        meta = json.loads((session_dir / "meta.json").read_text())
        assert meta["context_file"] == "/tmp/ctx.md"

        other = SessionLogger(cfg, "fresh")
        other.resume_existing_session(agent.session_id, session_dir)
        assert other.context_file == "/tmp/ctx.md"

        [info] = SessionLoader.list_sessions(cfg)
        assert info["context_file"] == "/tmp/ctx.md"

        # /clear is a new conversation: the file goes with the name.
        await agent.clear_history()
        assert agent.context_file is None


def _input_text(app) -> str:
    return app.query_one(ChatInputContainer)._body.input_widget.get_full_text()


@pytest.mark.asyncio
async def test_set_context_file_resolves_and_records(tmp_path: Path) -> None:
    ctx = tmp_path / "really_long_context_file.md"
    ctx.write_text("# context")
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        assert await app._handle_command(f"/set-context-file {ctx}")
        await pilot.pause(0.1)
        assert app.agent_loop.context_file == str(ctx.resolve())

        assert await app._handle_command("/set-context-file nope.md")
        await pilot.pause(0.1)
        assert app.agent_loop.context_file == str(ctx.resolve())
        assert any("not found" in str(m._error) for m in app.query(ErrorMessage))

        assert await app._handle_command("/context-file")
        await pilot.pause(0.1)
        assert app.agent_loop.context_file is None


@pytest.mark.asyncio
async def test_resume_refuses_without_name_then_without_file(tmp_path) -> None:
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        assert await app._handle_command("/resume-from-context-file")
        await pilot.pause(0.1)
        assert any("no name" in str(m._error) for m in app.query(ErrorMessage))
        assert _input_text(app) == "/set-conversation-name "

        await app.agent_loop.set_conversation_name("work")
        assert await app._handle_command("/resume_from_context_file")
        await pilot.pause(0.1)
        assert any("no context file" in str(m._error) for m in app.query(ErrorMessage))
        assert _input_text(app) == "/set-context-file "
        # Nothing was cleared or renamed.
        assert app.agent_loop.conversation_name == "work"


@pytest.mark.asyncio
async def test_resume_starts_a_continuation_with_read_prompt_loaded(
    tmp_path: Path,
) -> None:
    ctx = tmp_path / "work_context_file.md"
    ctx.write_text("# context")
    app = build_test_vibe_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        await app.agent_loop.set_conversation_name("Really long and tedious work")
        await app.agent_loop.set_context_file(str(ctx))
        app.agent_loop.messages.add(LLMMessage(role=Role.user, content="old talk"))
        app.agent_loop.messages.add(LLMMessage(role=Role.assistant, content="ok"))

        assert await app._handle_command("/resume-from-context-file")
        await pilot.pause(0.3)

        name = app.agent_loop.conversation_name
        assert name is not None
        assert name.startswith("Really long and tedious work -- continuation 20")
        assert app.agent_loop.context_file == str(ctx)
        # History is gone, nothing sent: only the system message remains.
        assert len(app.agent_loop.messages) == 1
        assert _input_text(app) == f"Read {ctx}"
        notices = [str(m._content) for m in app.query(UserCommandMessage)]
        assert any("Continuation of `Really long and tedious work`" in n for n in notices)

        # Running it again replaces the tail instead of stacking.
        assert await app._handle_command("/resume-from-context-file")
        await pilot.pause(0.3)
        assert app.agent_loop.conversation_name.count("-- continuation") == 1
