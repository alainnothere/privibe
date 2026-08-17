from __future__ import annotations

from datetime import datetime
import html
import os
from pathlib import Path
from string import Template
import subprocess
import sys
from typing import TYPE_CHECKING

from privibe.core.config.harness_files import get_harness_files_manager
from privibe.core.paths import VIBE_HOME
from privibe.core.paths.dialect import dialect_hint
from privibe.core.prompts import UtilityPrompt
from privibe.core.utils import (
    CONTEXT_REFRESH_TAG,
    is_dangerous_directory,
    is_windows,
    resolve_windows_bash,
)

if TYPE_CHECKING:
    from privibe.core.agents import AgentManager
    from privibe.core.config import ProjectContextConfig, VibeConfig
    from privibe.core.skills.manager import SkillManager
    from privibe.core.tools.manager import ToolManager

_git_status_cache: dict[Path, str] = {}


class ProjectContextProvider:
    def __init__(
        self, config: ProjectContextConfig, root_path: str | Path = "."
    ) -> None:
        self.root_path = Path(root_path).resolve()
        self.config = config

    def get_git_status(self) -> str:
        if self.root_path in _git_status_cache:
            return _git_status_cache[self.root_path]

        result = self._fetch_git_status()
        _git_status_cache[self.root_path] = result
        return result

    def _fetch_git_status(self) -> str:
        try:
            timeout = min(self.config.timeout_seconds, 10.0)
            num_commits = self.config.default_commit_count

            current_branch = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                check=True,
                cwd=self.root_path,
                stdin=subprocess.DEVNULL if is_windows() else None,
                text=True,
                timeout=timeout,
            ).stdout.strip()

            main_branch = "main"
            try:
                branches_output = subprocess.run(
                    ["git", "branch", "-r"],
                    capture_output=True,
                    check=True,
                    cwd=self.root_path,
                    stdin=subprocess.DEVNULL if is_windows() else None,
                    text=True,
                    timeout=timeout,
                ).stdout
                if "origin/master" in branches_output:
                    main_branch = "master"
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass

            status_output = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                check=True,
                cwd=self.root_path,
                stdin=subprocess.DEVNULL if is_windows() else None,
                text=True,
                timeout=timeout,
            ).stdout.strip()

            if status_output:
                status_lines = status_output.splitlines()
                MAX_GIT_STATUS_SIZE = 50
                if len(status_lines) > MAX_GIT_STATUS_SIZE:
                    status = (
                        f"({len(status_lines)} changes - use 'git status' for details)"
                    )
                else:
                    status = f"({len(status_lines)} changes)"
            else:
                status = "(clean)"

            log_output = subprocess.run(
                ["git", "log", "--oneline", f"-{num_commits}", "--decorate"],
                capture_output=True,
                check=True,
                cwd=self.root_path,
                stdin=subprocess.DEVNULL if is_windows() else None,
                text=True,
                timeout=timeout,
            ).stdout.strip()

            recent_commits = []
            for line in log_output.split("\n"):
                if not (line := line.strip()):
                    continue

                if " " in line:
                    commit_hash, commit_msg = line.split(" ", 1)
                    if (
                        "(" in commit_msg
                        and ")" in commit_msg
                        and (paren_index := commit_msg.rfind("(")) > 0
                    ):
                        commit_msg = commit_msg[:paren_index].strip()
                    recent_commits.append(f"{commit_hash} {commit_msg}")
                else:
                    recent_commits.append(line)

            git_info_parts = [
                f"Current branch: {current_branch}",
                f"Main branch (you will usually use this for PRs): {main_branch}",
                f"Status: {status}",
            ]

            if recent_commits:
                git_info_parts.append("Recent commits:")
                git_info_parts.extend(recent_commits)

            return "\n".join(git_info_parts)

        except subprocess.TimeoutExpired:
            return "Git operations timed out (large repository)"
        except subprocess.CalledProcessError:
            return "Not a git repository or git not available"
        except Exception as e:
            return f"Error getting git status: {e}"

    def get_full_context(self) -> str:
        git_status = self.get_git_status()
        from privibe.core.project_tree import build_tree
        tree = build_tree(self.root_path)

        template = UtilityPrompt.PROJECT_CONTEXT.read()
        return Template(template).safe_substitute(
            abs_path=str(self.root_path), git_status=git_status, tree=tree
        )


def _get_platform_name() -> str:
    platform_names = {
        "win32": "Windows",
        "darwin": "macOS",
        "linux": "Linux",
        "freebsd": "FreeBSD",
        "openbsd": "OpenBSD",
        "netbsd": "NetBSD",
    }
    return platform_names.get(sys.platform, "Unix-like")


def _get_bash_search_paths_override(config: VibeConfig | None) -> list[str] | None:
    """User's [tools.bash] bash_search_paths, or None to use the defaults.

    Mirrors the absent-vs-empty semantics of the bash tool: a missing field
    means "use the built-in default paths", an explicit [] opts out of the
    static path search.
    """
    if config is None:
        return None
    return config.tools.get("bash", {}).get("bash_search_paths")


def _get_default_shell(config: VibeConfig | None = None) -> str:
    """Get the shell the bash tool actually executes commands with.

    On Unix, uses $SHELL env var and defaults to sh.
    On Windows, the resolved Git Bash when found, else COMSPEC or cmd.exe.
    """
    if is_windows():
        if bash := resolve_windows_bash(_get_bash_search_paths_override(config)):
            return bash
        return os.environ.get("COMSPEC", "cmd.exe")
    return os.environ.get("SHELL", "sh")


def _get_os_system_prompt(config: VibeConfig | None = None) -> str:
    shell = _get_default_shell(config)
    platform_name = _get_platform_name()
    prompt = f"The operating system is {platform_name} with shell `{shell}`"

    if is_windows():
        bash_path = resolve_windows_bash(_get_bash_search_paths_override(config))
        prompt += "\n" + _get_windows_system_prompt(bash_path)

    if (hint := dialect_hint()) is not None:
        prompt += "\n" + hint

    # The datetime is rendered ONCE, at session creation, and becomes a fixed
    # historical fact of the session: message 0 is never regenerated. Fresh
    # datetimes reach the model via the context_refresh tail message on resume.
    now = datetime.now().strftime("%B %d, %Y %I:%M %p")
    prompt += f"\nThe current date and time is {now} (local time)"

    return prompt


def _get_windows_system_prompt(bash_path: str | None = None) -> str:
    if bash_path:
        return (
            "### SHELL NOTES (Git Bash):\n"
            f"- The `bash` tool runs commands through Git Bash (`{bash_path}`): "
            "Unix commands (`ls`, `grep`, `cat`, ...) and bash syntax work\n"
            "- Paths may appear in Unix form like `/c/Users/...`; Windows forms are also understood\n"
            "- Commands resolve to their Unix versions, not cmd.exe builtins "
            "(e.g. `date` prints the date instead of prompting to set it)\n"
            "- For Windows-native tooling, invoke it explicitly: "
            "`cmd //c <command>` or `powershell.exe -Command <command>`"
        )
    return (
        "### COMMAND COMPATIBILITY RULES (MUST FOLLOW):\n"
        "- The `bash` tool executes commands via `cmd.exe` (Git Bash was not found) - use cmd syntax, not bash\n"
        "- DO NOT rely on Unix commands like `ls`, `grep`, `cat` - they may not exist\n"
        "- Use: `dir` (Windows) for directory listings\n"
        "- Use: backslashes (\\\\) for paths\n"
        "- Check command availability with: `where command` (Windows)\n"
        "- DO NOT use interactive cmd builtins like `date` or `time` without `/T` - they prompt for input\n"
        "- Script shebang: Not applicable on Windows\n"
        "### ALWAYS verify commands work on the detected platform before suggesting them"
    )


def _add_commit_signature() -> str:
    return ""


def _get_available_skills_section(skill_manager: SkillManager) -> str:
    skills = skill_manager.available_skills
    if not skills:
        return ""

    lines = [
        "# Available Skills",
        "",
        "You have access to the following skills. When a task matches a skill's description,",
        "use the `skill` tool if available to load the full skill instructions, if it is not available, read the files manually.",
        "",
        "<available_skills>",
    ]

    for name, info in sorted(skills.items()):
        lines.append("  <skill>")
        lines.append(f"    <name>{html.escape(str(name))}</name>")
        lines.append(
            f"    <description>{html.escape(str(info.description))}</description>"
        )
        lines.append(f"    <path>{html.escape(str(info.skill_path))}</path>")
        lines.append("  </skill>")

    lines.append("</available_skills>")

    return "\n".join(lines)


def _get_available_subagents_section(agent_manager: AgentManager) -> str:
    agents = agent_manager.get_subagents()
    if not agents:
        return ""

    lines = ["# Available Subagents", ""]
    lines.append("The following subagents can be spawned via the Task tool:")
    for agent in agents:
        lines.append(f"- **{agent.name}**: {agent.description}")

    return "\n".join(lines)


# The duplicate-call dedup needs no system prompt note: the runtime skip
# notice (agent_loop) names the executed call, points at its result, and
# explains how to intentionally re-run — in-band, at the moment it matters.


def get_universal_system_prompt(
    tool_manager: ToolManager,
    config: VibeConfig,
    skill_manager: SkillManager,
    agent_manager: AgentManager,
) -> str:
    """Build the system prompt. Called exactly once per session, at AgentLoop
    construction. The result is frozen for the life of the session: it is the
    llama.cpp KV-cache prefix, and nothing may regenerate or edit it.
    """
    sections = [config.system_prompt]

    if config.include_commit_signature:
        sections.append(_add_commit_signature())

    if config.include_model_info:
        sections.append(f"Your model name is: `{config.active_model}`")

    if config.include_prompt_detail:
        sections.append(_get_os_system_prompt(config=config))
        tool_prompts = []
        for tool_class in tool_manager.available_tools.values():
            if prompt := tool_class.get_tool_prompt():
                tool_prompts.append(prompt)
        if tool_prompts:
            sections.append("\n---\n".join(tool_prompts))

        skills_section = _get_available_skills_section(skill_manager)
        if skills_section:
            sections.append(skills_section)

        subagents_section = _get_available_subagents_section(agent_manager)
        if subagents_section:
            sections.append(subagents_section)

    if config.include_project_context:
        is_dangerous, reason = is_dangerous_directory()
        if is_dangerous:
            # Safety warning always stays in the system prompt.
            template = UtilityPrompt.DANGEROUS_DIRECTORY.read()
            context = Template(template).safe_substitute(
                reason=reason.lower(), abs_path=Path(".").resolve()
            )
            sections.append(context)
        else:
            context = ProjectContextProvider(
                config=config.project_context, root_path=Path.cwd()
            ).get_full_context()
            sections.append(context)

        if docs_section := _get_agents_docs_section(config):
            sections.append(docs_section)

    return "\n\n".join(sections)


def _get_agents_docs_section(config: VibeConfig) -> str | None:
    mgr = get_harness_files_manager()
    user_doc = mgr.load_user_doc()
    project_docs = mgr.load_project_docs()
    extra_docs = mgr.load_extra_instruction_files(config.extra_instruction_files)

    doc_sections: list[str] = []
    if user_doc.strip():
        doc_sections.append(
            f"## User instructions\n\nContents of {VIBE_HOME.path}/AGENTS.md (user-level instructions):\n\n{user_doc.strip()}"
        )
    if project_docs or extra_docs:
        doc_sections.append("## Project instructions (checked into the codebase)")
    for doc_dir, doc_content in project_docs:
        doc_sections.append(
            f"Contents of {doc_dir}/AGENTS.md:\n\n{doc_content.strip()}"
        )
    for extra_path, extra_content in extra_docs:
        doc_sections.append(
            f"Contents of {extra_path}:\n\n{extra_content}"
        )
    if not doc_sections:
        return None
    template = UtilityPrompt.AGENTS_DOC.read()
    return Template(template).safe_substitute(sections="\n\n".join(doc_sections))


def build_context_refresh_content(config: VibeConfig, resumed: bool = True) -> str:
    """Volatile facts delivered as a tail message on resume. This is the ONLY
    place fresh datetime / model / project state reaches an existing
    conversation; the prefix (message 0 and history) is never rewritten.
    """
    now = datetime.now().strftime("%B %d, %Y %I:%M %p")
    lead = "Session resumed. " if resumed else ""
    lines = [f"{lead}The current date and time is {now} (local time)."]
    if config.include_model_info:
        lines.append(f"The active model is: `{config.active_model}`")
    if config.include_project_context:
        is_dangerous, _ = is_dangerous_directory()
        if not is_dangerous:
            context = ProjectContextProvider(
                config=config.project_context, root_path=Path.cwd()
            ).get_full_context()
            lines.append(context)
    content = "\n\n".join(lines)
    return f"<{CONTEXT_REFRESH_TAG}>{content}</{CONTEXT_REFRESH_TAG}>"
