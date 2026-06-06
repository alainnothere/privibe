from __future__ import annotations

import sys

import pytest

from privibe.core.agents import AgentManager
from privibe.core.skills.manager import SkillManager
from privibe.core.system_prompt import (
    build_context_refresh_content,
    get_universal_system_prompt,
)
from privibe.core.tools.manager import ToolManager
from privibe.core.types import LLMMessage, Role
from privibe.core.utils.tags import CONTEXT_REFRESH_TAG
from tests.conftest import build_test_agent_loop, build_test_vibe_config


def _is_initial_context(msg: LLMMessage) -> bool:
    return bool(msg.injected) and f"<{CONTEXT_REFRESH_TAG}>" in (msg.content or "")


def test_get_universal_system_prompt_includes_windows_prompt_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("COMSPEC", "C:\\Windows\\System32\\cmd.exe")

    config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=True,
        include_model_info=False,
        include_commit_signature=False,
    )
    tool_manager = ToolManager(lambda: config)
    skill_manager = SkillManager(lambda: config)
    agent_manager = AgentManager(lambda: config)

    prompt = get_universal_system_prompt(
        tool_manager, config, skill_manager, agent_manager
    )

    assert "You are Vibe, a super useful programming assistant." in prompt
    assert (
        "The operating system is Windows with shell `C:\\Windows\\System32\\cmd.exe`"
        in prompt
    )
    assert "DO NOT use Unix commands like `ls`, `grep`, `cat`" in prompt
    assert "Use: `dir` (Windows) for directory listings" in prompt
    assert "Use: backslashes (\\\\) for paths" in prompt
    assert "Check command availability with: `where command` (Windows)" in prompt
    assert "Script shebang: Not applicable on Windows" in prompt


# --- stable_system_prefix --------------------------------------------------
# When enabled, the volatile datetime + project context (git/tree) are kept out
# of the system prompt and delivered as a separate first injected message, so the
# static system prompt stays a stable, KV-cacheable prefix across sessions.


def _managers(config):
    return (
        ToolManager(lambda: config),
        SkillManager(lambda: config),
        AgentManager(lambda: config),
    )


def test_datetime_is_in_system_prompt_by_default() -> None:
    config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=True,
    )
    prompt = get_universal_system_prompt(*_reorder(config))
    assert "The current date and time is" in prompt


def test_stable_prefix_keeps_datetime_out_of_system_prompt() -> None:
    config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=True,
        stable_system_prefix=True,
    )
    prompt = get_universal_system_prompt(*_reorder(config))
    assert "The current date and time is" not in prompt
    # Static OS info still belongs in the (cacheable) system prompt.
    assert "The operating system is" in prompt


def test_stable_prefix_keeps_project_context_out_of_system_prompt() -> None:
    default_config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=True,
        include_prompt_detail=True,
    )
    default_prompt = get_universal_system_prompt(*_reorder(default_config))
    assert "gitStatus:" in default_prompt

    stable_config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=True,
        include_prompt_detail=True,
        stable_system_prefix=True,
    )
    stable_prompt = get_universal_system_prompt(*_reorder(stable_config))
    assert "gitStatus:" not in stable_prompt


def test_initial_context_content_omits_session_resumed_lead() -> None:
    config = build_test_vibe_config(
        system_prompt_id="tests", include_project_context=False
    )
    resumed = build_context_refresh_content(config, resumed=True)
    fresh = build_context_refresh_content(config, resumed=False)

    assert "Session resumed." in resumed
    assert "Session resumed." not in fresh
    assert "The current date and time is" in fresh


def test_agent_loop_emits_injected_context_only_with_stable_prefix() -> None:
    off = build_test_agent_loop(
        config=build_test_vibe_config(
            system_prompt_id="tests", include_project_context=False
        )
    )
    assert len(off.messages) == 1
    assert off.messages[0].role == Role.system

    on = build_test_agent_loop(
        config=build_test_vibe_config(
            system_prompt_id="tests",
            include_project_context=False,
            stable_system_prefix=True,
        )
    )
    assert on.messages[0].role == Role.system
    assert on.messages[1].role == Role.user
    assert on.messages[1].injected is True
    assert f"<{CONTEXT_REFRESH_TAG}>" in (on.messages[1].content or "")


def _reorder(config):
    tool_manager, skill_manager, agent_manager = _managers(config)
    return tool_manager, config, skill_manager, agent_manager


def _stable_config(stable: bool):
    return build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        stable_system_prefix=stable,
    )


# Toggling stable_system_prefix at runtime applies on the next rebuild (/reload).
# The reload path must stay consistent: add the injected context when newly on,
# drop it when newly off, and never duplicate it.


@pytest.mark.asyncio
async def test_reload_adds_injected_context_when_toggled_on() -> None:
    loop = build_test_agent_loop(config=_stable_config(False))
    loop.messages.add(LLMMessage(role=Role.user, content="hi"))
    assert not any(_is_initial_context(m) for m in loop.messages)

    await loop.reload_with_initial_messages(base_config=_stable_config(True))

    assert loop.messages[0].role == Role.system
    assert _is_initial_context(loop.messages[1])
    assert loop.messages[2].content == "hi"  # conversation preserved after context


@pytest.mark.asyncio
async def test_reload_drops_injected_context_when_toggled_off() -> None:
    loop = build_test_agent_loop(config=_stable_config(True))
    assert _is_initial_context(loop.messages[1])
    loop.messages.add(LLMMessage(role=Role.user, content="hi"))

    await loop.reload_with_initial_messages(base_config=_stable_config(False))

    assert not any(_is_initial_context(m) for m in loop.messages)
    assert loop.messages[0].role == Role.system
    assert loop.messages[1].content == "hi"


@pytest.mark.asyncio
async def test_reload_does_not_duplicate_injected_context() -> None:
    loop = build_test_agent_loop(config=_stable_config(True))
    loop.messages.add(LLMMessage(role=Role.user, content="hi"))

    await loop.reload_with_initial_messages(base_config=_stable_config(True))

    assert sum(_is_initial_context(m) for m in loop.messages) == 1
