from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path
import tomllib
from typing import TYPE_CHECKING, Any

from privibe.core.paths import PLANS_DIR

if TYPE_CHECKING:
    from privibe.core.config import VibeConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class AgentSafety(StrEnum):
    SAFE = auto()
    NEUTRAL = auto()
    DESTRUCTIVE = auto()
    YOLO = auto()


class AgentType(StrEnum):
    AGENT = auto()
    SUBAGENT = auto()


class BuiltinAgentName(StrEnum):
    DEFAULT = "default"
    CHAT = "chat"
    PLAN = "plan"
    ACCEPT_EDITS = "accept-edits"
    AUTO_APPROVE = "auto-approve"
    EXPLORE = "explore"
    ORCHESTRATE_WORKER = "orchestrate-worker"
    NANNY = "nanny"


@dataclass(frozen=True)
class AgentProfile:
    name: str
    display_name: str
    description: str
    safety: AgentSafety
    agent_type: AgentType = AgentType.AGENT
    overrides: dict[str, Any] = field(default_factory=dict)
    install_required: bool = False
    # Spawned only by harness code (the orchestrate tool), never by the model:
    # left out of the system prompt's subagent list and refused by `task`.
    internal: bool = False

    def apply_to_config(self, base: VibeConfig) -> VibeConfig:
        from privibe.core.config import VibeConfig as VC

        merged = _deep_merge(
            base.model_dump(),
            {k: v for k, v in self.overrides.items() if k != "base_disabled"},
        )
        base_disabled = self.overrides.get("base_disabled")
        if isinstance(base_disabled, list):
            merged["disabled_tools"] = list({
                *base_disabled,
                *merged.get("disabled_tools", []),
            })

        derived = VC.model_validate(merged)
        # The fallback record lives in PrivateAttrs, which the dump/validate
        # round-trip drops — and the dump already carries the rewritten
        # active_model, so the derived config can never re-detect the fallback
        # on its own. Carry the record over or the UI warning stays silent.
        derived._fallback_from_model = base._fallback_from_model
        derived._fallback_env_var = base._fallback_env_var
        return derived

    @classmethod
    def from_toml(cls, path: Path) -> AgentProfile:
        with path.open("rb") as f:
            data = tomllib.load(f)
        return cls(
            name=path.stem,
            display_name=data.pop("display_name", path.stem.replace("-", " ").title()),
            description=data.pop("description", ""),
            safety=AgentSafety(data.pop("safety", AgentSafety.NEUTRAL)),
            agent_type=AgentType(data.pop("agent_type", AgentType.AGENT)),
            overrides=data,
        )


CHAT_AGENT_TOOLS = ["grep", "read_file", "hashed_read", "find_symbol", "ask_user_question", "task"]

# Tools that can change files or run arbitrary commands. Profiles restrict
# these via per-tool permissions, never by shrinking the advertised tool set:
# the tools array is part of the llama.cpp KV-cache prefix and must stay
# byte-identical for the life of a session.
MUTATING_TOOLS = [
    "write_file",
    "search_replace",
    "hashed_replace_line",
    "hashed_replace_block",
    "hashed_delete_line",
    "hashed_delete_block",
    "restore_file",
    "bash",
]


def _chat_overrides() -> dict[str, Any]:
    return {
        "auto_approve": True,
        "tools": {name: {"permission": "never"} for name in MUTATING_TOOLS},
    }


def _plan_overrides() -> dict[str, Any]:
    plans_pattern = str(PLANS_DIR.path / "*")
    return {
        "tools": {
            "write_file": {"permission": "never", "allowlist": [plans_pattern]},
            "search_replace": {"permission": "never", "allowlist": [plans_pattern]},
            "hashed_replace_line": {"permission": "never", "allowlist": [plans_pattern]},
            "hashed_replace_block": {"permission": "never", "allowlist": [plans_pattern]},
            "hashed_delete_line": {"permission": "never", "allowlist": [plans_pattern]},
            "hashed_delete_block": {"permission": "never", "allowlist": [plans_pattern]},
            "restore_file": {"permission": "never", "allowlist": [plans_pattern]},
        }
    }


DEFAULT = AgentProfile(
    BuiltinAgentName.DEFAULT,
    "Default",
    "Requires approval for tool executions",
    AgentSafety.NEUTRAL,
)
PLAN = AgentProfile(
    BuiltinAgentName.PLAN,
    "Plan",
    "Read-only agent for exploration and planning",
    AgentSafety.SAFE,
    overrides=_plan_overrides(),
)
CHAT = AgentProfile(
    BuiltinAgentName.CHAT,
    "Chat",
    "Read-only conversational mode for questions and discussions",
    AgentSafety.SAFE,
    overrides=_chat_overrides(),
)
ACCEPT_EDITS = AgentProfile(
    BuiltinAgentName.ACCEPT_EDITS,
    "Accept Edits",
    "Auto-approves file edits only",
    AgentSafety.DESTRUCTIVE,
    overrides={
        "tools": {
            "write_file": {"permission": "always"},
            "search_replace": {"permission": "always"},
            "hashed_replace_line": {"permission": "always"},
            "hashed_replace_block": {"permission": "always"},
            "hashed_delete_line": {"permission": "always"},
            "hashed_delete_block": {"permission": "always"},
            "restore_file": {"permission": "always"},
        },
    },
)
AUTO_APPROVE = AgentProfile(
    BuiltinAgentName.AUTO_APPROVE,
    "Auto Approve",
    "Auto-approves all tool executions",
    AgentSafety.YOLO,
    overrides={"auto_approve": True},
)

EXPLORE = AgentProfile(
    name=BuiltinAgentName.EXPLORE,
    display_name="Explore",
    description="Read-only subagent for codebase exploration",
    safety=AgentSafety.SAFE,
    agent_type=AgentType.SUBAGENT,
    overrides={"enabled_tools": ["grep", "read_file", "hashed_read", "find_symbol"], "system_prompt_id": "explore"},
)

# One profile for every orchestrate role (manager, brainstorm, worker). The
# role lives in the tail of the user message, not in the system prompt, so
# all roles share a single KV-cache prefix. The prompt-size flags are off so
# message 0 depends only on the prompt file and the tools array. Scope
# (which paths the workers may edit) is per job, so the orchestrate tool
# merges it into the base config's `tools` table; this profile must not set
# `tools` or it would win that merge.
ORCHESTRATE_WORKER = AgentProfile(
    name=BuiltinAgentName.ORCHESTRATE_WORKER,
    display_name="Orchestrate Worker",
    description="Internal role agent for the orchestrate tool",
    safety=AgentSafety.DESTRUCTIVE,
    agent_type=AgentType.SUBAGENT,
    internal=True,
    overrides={
        "enabled_tools": [
            "grep",
            "read_file",
            "hashed_read",
            "find_symbol",
            "write_file",
            "search_replace",
            "hashed_replace_line",
            "hashed_replace_block",
            "hashed_delete_line",
            "hashed_delete_block",
            "restore_file",
            "bash",
        ],
        "system_prompt_id": "orchestrate",
        "include_project_context": False,
        "include_commit_signature": False,
        "include_model_info": False,
    },
)

# The nanny: a second model that reviews tool-approval prompts. Answer-only:
# every tool is disabled, no MCP server is started, and the system prompt is
# the nanny prompt file alone (no project context, no tool prompts), so its
# KV-cache prefix depends on nothing but that file. `active_model` is set by
# the manager on the base config it builds from, so the profile stays static.
# `compaction_model` is blanked because profile application re-runs every
# VibeConfig validator, and the user's compaction model may be on another
# provider than the nanny.
NANNY = AgentProfile(
    name=BuiltinAgentName.NANNY,
    display_name="Nanny",
    description="Internal reviewer of tool-approval prompts",
    safety=AgentSafety.SAFE,
    agent_type=AgentType.SUBAGENT,
    internal=True,
    overrides={
        "disabled_tools": ["*"],
        "mcp_servers": [],
        "system_prompt_id": "nanny",
        "include_project_context": False,
        "include_commit_signature": False,
        "include_model_info": False,
        "include_prompt_detail": False,
        "compaction_model": None,
        "auto_approve": False,
        "context_warnings": False,
    },
)

BUILTIN_AGENTS: dict[str, AgentProfile] = {
    BuiltinAgentName.DEFAULT: DEFAULT,
    BuiltinAgentName.PLAN: PLAN,
    BuiltinAgentName.ACCEPT_EDITS: ACCEPT_EDITS,
    BuiltinAgentName.AUTO_APPROVE: AUTO_APPROVE,
    BuiltinAgentName.EXPLORE: EXPLORE,
    BuiltinAgentName.ORCHESTRATE_WORKER: ORCHESTRATE_WORKER,
    BuiltinAgentName.NANNY: NANNY,
}
