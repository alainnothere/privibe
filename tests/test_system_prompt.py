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
    # No Git Bash found: the cmd.exe fallback rules must apply.
    monkeypatch.setattr(
        "privibe.core.system_prompt.resolve_windows_bash", lambda paths=None: None
    )

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

    assert "You are privibe, a CLI coding assistant." in prompt
    assert (
        "The operating system is Windows with shell `C:\\Windows\\System32\\cmd.exe`"
        in prompt
    )
    assert "executes commands via `cmd.exe`" in prompt
    assert "DO NOT rely on Unix commands like `ls`, `grep`, `cat`" in prompt
    assert "Use: `dir` (Windows) for directory listings" in prompt
    assert "Use: backslashes (\\\\) for paths" in prompt
    assert "Check command availability with: `where command` (Windows)" in prompt
    assert "Script shebang: Not applicable on Windows" in prompt


def test_get_universal_system_prompt_reports_git_bash_when_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bash_path = "C:\\Program Files\\Git\\bin\\bash.exe"
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("COMSPEC", "C:\\Windows\\System32\\cmd.exe")
    monkeypatch.setattr(
        "privibe.core.system_prompt.resolve_windows_bash",
        lambda paths=None: bash_path,
    )

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

    assert f"The operating system is Windows with shell `{bash_path}`" in prompt
    assert "SHELL NOTES (Git Bash)" in prompt
    assert "cmd //c <command>" in prompt
    assert "powershell.exe -Command <command>" in prompt
    # The cmd.exe fallback rules must NOT appear when Git Bash is active.
    assert "COMMAND COMPATIBILITY RULES" not in prompt


# --- frozen system prompt ---------------------------------------------------
# The system prompt is generated exactly once, at session creation, with the
# datetime and project context baked in as fixed historical facts. Nothing may
# regenerate it afterwards: it is the llama.cpp KV-cache prefix. Fresh
# datetime / model / project state reaches the model only via the
# context_refresh tail message on resume.


def _managers(config):
    return (
        ToolManager(lambda: config),
        SkillManager(lambda: config),
        AgentManager(lambda: config),
    )


def _reorder(config):
    tool_manager, skill_manager, agent_manager = _managers(config)
    return tool_manager, config, skill_manager, agent_manager


def test_datetime_is_in_system_prompt() -> None:
    config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=True,
    )
    prompt = get_universal_system_prompt(*_reorder(config))
    assert "The current date and time is" in prompt


def test_project_context_is_in_system_prompt() -> None:
    config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=True,
        include_prompt_detail=True,
    )
    prompt = get_universal_system_prompt(*_reorder(config))
    assert "gitStatus:" in prompt


def test_context_refresh_content_omits_session_resumed_lead() -> None:
    config = build_test_vibe_config(
        system_prompt_id="tests", include_project_context=False
    )
    resumed = build_context_refresh_content(config, resumed=True)
    fresh = build_context_refresh_content(config, resumed=False)

    assert "Session resumed." in resumed
    assert "Session resumed." not in fresh
    assert "The current date and time is" in fresh


def test_context_refresh_includes_active_model() -> None:
    config = build_test_vibe_config(
        system_prompt_id="tests",
        include_project_context=False,
        include_model_info=True,
    )
    content = build_context_refresh_content(config)
    assert "The active model is" in content


def test_new_session_starts_with_only_the_system_message() -> None:
    loop = build_test_agent_loop(
        config=build_test_vibe_config(
            system_prompt_id="tests", include_project_context=False
        )
    )
    assert len(loop.messages) == 1
    assert loop.messages[0].role == Role.system


@pytest.mark.asyncio
async def test_apply_runtime_config_never_touches_messages() -> None:
    loop = build_test_agent_loop(
        config=build_test_vibe_config(
            system_prompt_id="tests", include_project_context=False
        )
    )
    loop.messages.add(LLMMessage(role=Role.user, content="hi"))
    system_identity = loop.messages[0]
    before = [(m.role, m.content) for m in loop.messages]

    await loop.apply_runtime_config(
        base_config=build_test_vibe_config(
            system_prompt_id="tests", include_project_context=False
        )
    )

    assert [(m.role, m.content) for m in loop.messages] == before
    assert loop.messages[0] is system_identity
    assert not any(_is_initial_context(m) for m in loop.messages)
