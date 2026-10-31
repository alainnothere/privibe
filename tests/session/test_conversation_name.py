"""/set-conversation-name: the name's life on disk and in memory.

Covers the non-obvious bits found in the 2026-10-06 research: the name must
ride in memory until the first save creates meta.json, must be rewritten
immediately when meta.json exists (without losing the other keys the last
save wrote), must survive reset_session (rewind/compact), must come back on
resume, and must be dropped by clear_history. Old meta.json files without
the field must still load.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from privibe.core.agents.models import AgentProfile, AgentSafety
from privibe.core.config import SessionLoggingConfig, VibeConfig
from privibe.core.session.resume_sessions import ResumeSessionInfo
from privibe.core.session.session_loader import SessionLoader
from privibe.core.session.session_logger import SessionLogger
from privibe.core.tools.manager import ToolManager
from privibe.core.types import AgentStats, LLMMessage, Role, SessionMetadata
from tests.conftest import build_test_agent_loop, build_test_vibe_config


@pytest.fixture
def session_config(tmp_path: Path) -> SessionLoggingConfig:
    save_dir = tmp_path / "sessions"
    save_dir.mkdir()
    return SessionLoggingConfig(
        save_dir=str(save_dir), session_prefix="test", enabled=True
    )


@pytest.fixture
def profile() -> AgentProfile:
    return AgentProfile(
        name="test-agent",
        display_name="Test Agent",
        description="A test agent",
        safety=AgentSafety.NEUTRAL,
        overrides={},
    )


@pytest.fixture
def tool_manager() -> ToolManager:
    manager = MagicMock(spec=ToolManager)
    manager.available_tools = {}
    return manager


@pytest.fixture
def vibe_config() -> VibeConfig:
    return build_test_vibe_config(active_model="test-model", models=[], providers=[])


def _messages() -> list[LLMMessage]:
    return [
        LLMMessage(role=Role.system, content="system"),
        LLMMessage(role=Role.user, content="hello there"),
        LLMMessage(role=Role.assistant, content="hi"),
    ]


async def _save(logger: SessionLogger, vibe_config, tool_manager, profile, messages=None):
    await logger.save_interaction(
        messages or _messages(), AgentStats(), vibe_config, tool_manager, profile
    )


def _meta(logger: SessionLogger) -> dict:
    return json.loads(logger.metadata_filepath.read_text(encoding="utf-8"))


class TestSessionMetadataField:
    def test_defaults_to_none(self) -> None:
        meta = SessionMetadata(
            session_id="s",
            start_time="t",
            end_time=None,
            git_commit=None,
            git_branch=None,
            environment={},
            username="u",
        )
        assert meta.name is None

    def test_old_meta_json_without_name_validates(self, tmp_path: Path) -> None:
        legacy = {
            "session_id": "abc",
            "start_time": "2026-01-01T00:00:00+00:00",
            "end_time": None,
            "git_commit": None,
            "git_branch": None,
            "environment": {"working_directory": "/x"},
            "username": "u",
            "title": "old auto title",
        }
        (tmp_path / "meta.json").write_text(json.dumps(legacy), encoding="utf-8")
        loaded = SessionLoader.load_metadata(tmp_path)
        assert loaded.name is None


class TestSetConversationName:
    @pytest.mark.asyncio
    async def test_starts_unnamed(self, session_config: SessionLoggingConfig) -> None:
        logger = SessionLogger(session_config, "sess-1")
        assert logger.conversation_name is None
        assert logger.session_metadata is not None
        assert logger.session_metadata.name is None

    @pytest.mark.asyncio
    async def test_name_before_first_save_rides_in_memory_then_lands(
        self, session_config, vibe_config, tool_manager, profile
    ) -> None:
        logger = SessionLogger(session_config, "sess-1")
        await logger.set_conversation_name("rewrite the parser")

        # No session folder yet: nothing on disk, name kept in memory.
        assert not logger.metadata_filepath.exists()
        assert logger.conversation_name == "rewrite the parser"
        assert logger.session_metadata.name == "rewrite the parser"

        await _save(logger, vibe_config, tool_manager, profile)
        assert _meta(logger)["name"] == "rewrite the parser"

    @pytest.mark.asyncio
    async def test_name_after_save_rewrites_meta_json_immediately(
        self, session_config, vibe_config, tool_manager, profile
    ) -> None:
        logger = SessionLogger(session_config, "sess-1")
        await _save(logger, vibe_config, tool_manager, profile)
        before = _meta(logger)
        assert before.get("name") is None

        await logger.set_conversation_name("parser work")

        after = _meta(logger)
        assert after["name"] == "parser work"
        # Everything the last save wrote is still there, untouched.
        for key in ("title", "total_messages", "stats", "config", "tools_available"):
            assert after[key] == before[key], key

    @pytest.mark.asyncio
    async def test_later_save_keeps_the_name(
        self, session_config, vibe_config, tool_manager, profile
    ) -> None:
        logger = SessionLogger(session_config, "sess-1")
        await _save(logger, vibe_config, tool_manager, profile)
        await logger.set_conversation_name("keep me")

        msgs = _messages() + [LLMMessage(role=Role.user, content="more")]
        await _save(logger, vibe_config, tool_manager, profile, msgs)

        meta = _meta(logger)
        assert meta["name"] == "keep me"
        assert meta["total_messages"] == 3
        # The auto title is still the first user message, independent of the name.
        assert meta["title"] == "hello there"

    @pytest.mark.asyncio
    async def test_clearing_the_name(
        self, session_config, vibe_config, tool_manager, profile
    ) -> None:
        logger = SessionLogger(session_config, "sess-1")
        await _save(logger, vibe_config, tool_manager, profile)
        await logger.set_conversation_name("temp")
        await logger.set_conversation_name(None)
        assert logger.conversation_name is None
        assert _meta(logger)["name"] is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("raw", ["", "   ", "\t\n"])
    async def test_blank_name_means_clear(self, session_config, raw: str) -> None:
        logger = SessionLogger(session_config, "sess-1")
        await logger.set_conversation_name("x")
        await logger.set_conversation_name(raw)
        assert logger.conversation_name is None

    @pytest.mark.asyncio
    async def test_name_is_stripped(self, session_config) -> None:
        logger = SessionLogger(session_config, "sess-1")
        await logger.set_conversation_name("  padded  ")
        assert logger.conversation_name == "padded"

    @pytest.mark.asyncio
    async def test_disabled_logging_keeps_name_in_memory_only(
        self, tmp_path: Path
    ) -> None:
        cfg = SessionLoggingConfig(
            save_dir=str(tmp_path), session_prefix="test", enabled=False
        )
        logger = SessionLogger(cfg, "sess-1")
        await logger.set_conversation_name("ephemeral")
        assert logger.conversation_name == "ephemeral"
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.asyncio
    async def test_corrupt_meta_json_raises_runtime_error(
        self, session_config, vibe_config, tool_manager, profile
    ) -> None:
        logger = SessionLogger(session_config, "sess-1")
        await _save(logger, vibe_config, tool_manager, profile)
        logger.metadata_filepath.write_text("{not json", encoding="utf-8")
        with pytest.raises(RuntimeError):
            await logger.set_conversation_name("x")
        # In-memory state still updated: the UI shows what the user typed.
        assert logger.conversation_name == "x"


class TestNameAcrossSessionBoundaries:
    @pytest.mark.asyncio
    async def test_reset_session_keeps_name(
        self, session_config, vibe_config, tool_manager, profile
    ) -> None:
        """rewind/compact mint a new session id for the same conversation."""
        logger = SessionLogger(session_config, "sess-1")
        await logger.set_conversation_name("survivor")
        old_dir = logger.session_dir

        logger.reset_session("sess-2")

        assert logger.conversation_name == "survivor"
        assert logger.session_metadata.name == "survivor"
        assert logger.session_dir != old_dir
        await _save(logger, vibe_config, tool_manager, profile)
        assert _meta(logger)["name"] == "survivor"

    @pytest.mark.asyncio
    async def test_resume_loads_name(
        self, session_config, vibe_config, tool_manager, profile
    ) -> None:
        first = SessionLogger(session_config, "sess-1")
        await _save(first, vibe_config, tool_manager, profile)
        await first.set_conversation_name("from disk")
        session_dir = first.session_dir

        second = SessionLogger(session_config, "sess-2")
        assert second.conversation_name is None
        second.resume_existing_session("sess-1", session_dir)
        assert second.conversation_name == "from disk"

    @pytest.mark.asyncio
    async def test_resume_unnamed_session_clears_previous_name(
        self, session_config, vibe_config, tool_manager, profile
    ) -> None:
        first = SessionLogger(session_config, "sess-1")
        await _save(first, vibe_config, tool_manager, profile)

        second = SessionLogger(session_config, "sess-2")
        await second.set_conversation_name("stale")
        second.resume_existing_session("sess-1", first.session_dir)
        assert second.conversation_name is None

    @pytest.mark.asyncio
    async def test_resume_then_save_keeps_name(
        self, session_config, vibe_config, tool_manager, profile
    ) -> None:
        first = SessionLogger(session_config, "sess-1")
        await _save(first, vibe_config, tool_manager, profile)
        await first.set_conversation_name("sticky")

        second = SessionLogger(session_config, "sess-2")
        second.resume_existing_session("sess-1", first.session_dir)
        msgs = _messages() + [LLMMessage(role=Role.user, content="again")]
        await _save(second, vibe_config, tool_manager, profile, msgs)
        assert _meta(second)["name"] == "sticky"


class TestSessionListing:
    @pytest.mark.asyncio
    async def test_list_sessions_exposes_name_and_searches_it(
        self, session_config, vibe_config, tool_manager, profile
    ) -> None:
        logger = SessionLogger(session_config, "sess-1")
        await _save(logger, vibe_config, tool_manager, profile)
        await logger.set_conversation_name("Needle Project")

        [info] = SessionLoader.list_sessions(session_config)
        assert info["name"] == "Needle Project"
        assert info["title"] == "hello there"

        [scan] = SessionLoader._scan_sessions(
            session_config, collect_text=True, preview_lines=1
        )
        assert "needle project" in (scan.search_text or "")

    @pytest.mark.asyncio
    async def test_blank_name_on_disk_reads_as_none(
        self, session_config, vibe_config, tool_manager, profile
    ) -> None:
        logger = SessionLogger(session_config, "sess-1")
        await _save(logger, vibe_config, tool_manager, profile)
        meta = _meta(logger)
        meta["name"] = "   "
        logger.metadata_filepath.write_text(json.dumps(meta), encoding="utf-8")
        [info] = SessionLoader.list_sessions(session_config)
        assert info["name"] is None

    def test_resume_info_display_name_prefers_name(self) -> None:
        named = ResumeSessionInfo(
            session_id="a", source="local", cwd="/", title="auto", end_time=None,
            name="chosen",
        )
        unnamed = ResumeSessionInfo(
            session_id="b", source="local", cwd="/", title="auto", end_time=None
        )
        assert named.display_name == "chosen"
        assert unnamed.display_name == "auto"


class TestAgentLoopIntegration:
    @pytest.mark.asyncio
    async def test_clear_history_drops_name(self, tmp_path: Path) -> None:
        config = build_test_vibe_config(
            session_logging=SessionLoggingConfig(
                save_dir=str(tmp_path), session_prefix="test", enabled=True
            )
        )
        agent = build_test_agent_loop(config=config)
        agent.messages.add(LLMMessage(role=Role.user, content="hello"))
        await agent.set_conversation_name("doomed")
        assert agent.conversation_name == "doomed"
        old_dir = agent.session_logger.session_dir

        await agent.clear_history()

        assert agent.conversation_name is None
        # The old session kept its name on disk.
        assert json.loads((old_dir / "meta.json").read_text())["name"] == "doomed"


class TestNannyModelBesideTheName:
    """2026-10-30: the nanny is per conversation, recorded in meta.json like
    the name (never in config.toml), restored on resume.
    """

    def test_old_meta_json_without_nanny_validates(self, tmp_path: Path) -> None:
        legacy = {
            "session_id": "abc",
            "start_time": "2026-01-01T00:00:00+00:00",
            "end_time": None,
            "git_commit": None,
            "git_branch": None,
            "environment": {},
            "username": "u",
        }
        (tmp_path / "meta.json").write_text(json.dumps(legacy), encoding="utf-8")
        assert SessionLoader.load_metadata(tmp_path).nanny_model is None

    @pytest.mark.asyncio
    async def test_rides_in_memory_then_lands_and_is_rewritten_live(
        self, session_config, vibe_config, tool_manager, profile
    ) -> None:
        logger = SessionLogger(session_config, "sess-1")
        await logger.set_nanny_model("reviewer")
        assert not logger.metadata_filepath.exists()
        assert logger.nanny_model == "reviewer"

        await _save(logger, vibe_config, tool_manager, profile)
        before = _meta(logger)
        assert before["nanny_model"] == "reviewer"

        await logger.set_nanny_model(None)
        after = _meta(logger)
        assert after["nanny_model"] is None
        for key in ("title", "total_messages", "stats", "config"):
            assert after[key] == before[key], key

    @pytest.mark.asyncio
    async def test_survives_reset_and_comes_back_on_resume(
        self, session_config, vibe_config, tool_manager, profile
    ) -> None:
        first = SessionLogger(session_config, "sess-1")
        await first.set_nanny_model("reviewer")
        first.reset_session("sess-2")
        assert first.nanny_model == "reviewer"
        await _save(first, vibe_config, tool_manager, profile)
        assert _meta(first)["nanny_model"] == "reviewer"

        second = SessionLogger(session_config, "sess-3")
        await second.set_nanny_model("stale")
        second.resume_existing_session("sess-2", first.session_dir)
        assert second.nanny_model == "reviewer"

        third = SessionLogger(session_config, "sess-4")
        await _save(third, vibe_config, tool_manager, profile)
        fourth = SessionLogger(session_config, "sess-5")
        await fourth.set_nanny_model("stale")
        fourth.resume_existing_session("sess-4", third.session_dir)
        assert fourth.nanny_model is None
