from __future__ import annotations

from dataclasses import dataclass
import sys

ALT_KEY = "⌥" if sys.platform == "darwin" else "Alt"


@dataclass
class Command:
    aliases: frozenset[str]
    description: str
    handler: str
    exits: bool = False


class CommandRegistry:
    def __init__(self, excluded_commands: list[str] | None = None) -> None:
        if excluded_commands is None:
            excluded_commands = []
        self.commands = {
            "help": Command(
                aliases=frozenset(["/help"]),
                description="Show help message",
                handler="_show_help",
            ),
            "config": Command(
                aliases=frozenset(["/config"]),
                description="Edit config settings",
                handler="_show_config",
            ),
            "model": Command(
                aliases=frozenset(["/model"]),
                description="Select active model",
                handler="_show_model",
            ),
            "reload": Command(
                aliases=frozenset(["/reload"]),
                description="Reload configuration from disk",
                handler="_reload_config",
            ),
            "clear": Command(
                aliases=frozenset(["/clear"]),
                description="Clear conversation history",
                handler="_clear_history",
            ),
            "log": Command(
                aliases=frozenset(["/log"]),
                description="Show path to current interaction log file",
                handler="_show_log_path",
            ),
            "compact": Command(
                aliases=frozenset(["/compact"]),
                description="Compact conversation history by summarizing",
                handler="_compact_history",
            ),
            "exit": Command(
                aliases=frozenset(["/exit"]),
                description="Exit the application",
                handler="_exit_app",
                exits=True,
            ),
            "terminal-setup": Command(
                aliases=frozenset(["/terminal-setup"]),
                description="Configure Shift+Enter for newlines",
                handler="_setup_terminal",
            ),
            "status": Command(
                aliases=frozenset(["/status"]),
                description="Display agent statistics",
                handler="_show_status",
            ),
            "proxy-setup": Command(
                aliases=frozenset(["/proxy-setup"]),
                description="Configure proxy and SSL certificate settings",
                handler="_show_proxy_setup",
            ),
            "resume": Command(
                aliases=frozenset(["/resume", "/continue"]),
                description="Browse and resume past sessions",
                handler="_show_session_picker",
            ),
            "voice": Command(
                aliases=frozenset(["/voice"]),
                description="Configure voice settings",
                handler="_show_voice_settings",
            ),
            "rewind": Command(
                aliases=frozenset(["/rewind"]),
                description="Rewind to a previous message",
                handler="_start_rewind_mode",
            ),
            "autocopy": Command(
                aliases=frozenset(["/autocopy"]),
                description="Toggle auto-copy to clipboard on mouse selection",
                handler="_toggle_autocopy",
            ),
            "detect_context_size": Command(
                aliases=frozenset(["/detect-context-size"]),
                description="Toggle context-size auto-detection on startup; re-enabling also retries detection now",
                handler="_toggle_auto_detect_context_size",
            ),
            "preview_lines": Command(
                aliases=frozenset(["/preview-lines"]),
                description="Cycle the tool-result preview length (3 → 5 → 10 lines)",
                handler="_cycle_preview_lines",
            ),
            "scrollback": Command(
                aliases=frozenset(["/scrollback"]),
                description="Cycle how many rows of message history to keep before pruning (50 → 100 → 250 → 500 → 1000)",
                handler="_cycle_scrollback",
            ),
            "list_tools": Command(
                aliases=frozenset(["/list-tools"]),
                description="Show active tools for the current session",
                handler="_show_active_tools",
            ),
            "list_agents": Command(
                aliases=frozenset(["/list-agents"]),
                description="Show available agents and the currently active one",
                handler="_show_available_agents",
            ),
            "list_subagents": Command(
                aliases=frozenset(["/list-subagents"]),
                description="Show available subagents",
                handler="_show_available_subagents",
            ),
            # DEBUG LLM COMMUNICATIONS
            "llm_debug": Command(
                aliases=frozenset(["/llm-debug"]),
                description="Toggle LLM debug dump (messages + payload to ./debug/)",
                handler="_toggle_llm_debug",
            ),
        }

        for command in excluded_commands:
            self.commands.pop(command, None)

        self._alias_map = {}
        for cmd_name, cmd in self.commands.items():
            for alias in cmd.aliases:
                self._alias_map[alias] = cmd_name

    def find_command(self, user_input: str) -> Command | None:
        cmd_name = self.get_command_name(user_input)
        return self.commands.get(cmd_name) if cmd_name else None

    def get_command_name(self, user_input: str) -> str | None:
        return self._alias_map.get(user_input.lower().strip())

    def get_help_text(self) -> str:
        lines: list[str] = [
            "### Keyboard Shortcuts",
            "",
            "- `Enter` Submit message",
            "- `Ctrl+J` / `Shift+Enter` / `Ctrl+Enter` Insert newline",
            "- `Escape` Interrupt agent or close dialogs",
            "- `Ctrl+C` Cancel operation or quit (press twice within 3s to exit)",
            "- `Ctrl+G` Edit input in external editor",
            "- `Ctrl+O` Toggle tool output view",
            "- `Shift+Tab` Toggle auto-approve mode",
            f"- `{ALT_KEY}+↑↓` / `Ctrl+P/N` Rewind to previous/next message",
            "",
            "### Special Features",
            "",
            "- `!<command>` Execute bash command directly",
            "- `@path/to/file/` Autocompletes file paths",
            "",
            "### Commands",
            "",
        ]

        for cmd in self.commands.values():
            aliases = ", ".join(f"`{alias}`" for alias in sorted(cmd.aliases))
            lines.append(f"- {aliases}: {cmd.description}")
        return "\n".join(lines)
