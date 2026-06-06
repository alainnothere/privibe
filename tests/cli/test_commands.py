from __future__ import annotations

from privibe.cli.commands import Command, CommandRegistry


class TestCommandRegistry:
    def test_get_command_name_returns_canonical_name_for_alias(self) -> None:
        registry = CommandRegistry()
        assert registry.get_command_name("/help") == "help"
        assert registry.get_command_name("/config") == "config"
        assert registry.get_command_name("/model") == "model"
        assert registry.get_command_name("/clear") == "clear"
        assert registry.get_command_name("/exit") == "exit"
    def test_get_command_name_normalizes_input(self) -> None:
        registry = CommandRegistry()
        assert registry.get_command_name("  /help  ") == "help"
        assert registry.get_command_name("/HELP") == "help"

    def test_get_command_name_returns_none_for_unknown(self) -> None:
        registry = CommandRegistry()
        assert registry.get_command_name("/unknown") is None
        assert registry.get_command_name("hello") is None
        assert registry.get_command_name("") is None

    def test_find_command_returns_command_when_alias_matches(self) -> None:
        registry = CommandRegistry()
        cmd = registry.find_command("/help")
        assert cmd is not None
        assert cmd.handler == "_show_help"
        assert isinstance(cmd, Command)

    def test_find_command_returns_none_when_no_match(self) -> None:
        registry = CommandRegistry()
        assert registry.find_command("/nonexistent") is None

    def test_find_command_uses_get_command_name(self) -> None:
        """find_command and get_command_name stay in sync for same input."""
        registry = CommandRegistry()
        for alias in ["/help", "/config", "/clear", "/exit"]:
            cmd_name = registry.get_command_name(alias)
            cmd = registry.find_command(alias)
            if cmd_name is None:
                assert cmd is None
            else:
                assert cmd is not None
                assert cmd_name in registry.commands
                assert registry.commands[cmd_name] is cmd

    def test_excluded_commands_not_in_registry(self) -> None:
        registry = CommandRegistry(excluded_commands=["exit"])
        assert registry.get_command_name("/exit") is None
        assert registry.find_command("/exit") is None
        assert registry.get_command_name("/help") == "help"

    def test_resume_command_registration(self) -> None:
        registry = CommandRegistry()
        assert registry.get_command_name("/resume") == "resume"
        assert registry.get_command_name("/continue") == "resume"
        cmd = registry.find_command("/resume")
        assert cmd is not None
        assert cmd.handler == "_show_session_picker"

    def test_autocopy_command_registration(self) -> None:
        registry = CommandRegistry()
        assert registry.get_command_name("/autocopy") == "autocopy"
        cmd = registry.find_command("/autocopy")
        assert cmd is not None
        assert cmd.handler == "_toggle_autocopy"

    def test_stable_prefix_command_registration(self) -> None:
        registry = CommandRegistry()
        assert registry.get_command_name("/stable-prefix") == "stable_prefix"
        cmd = registry.find_command("/stable-prefix")
        assert cmd is not None
        assert cmd.handler == "_toggle_stable_system_prefix"

    def test_every_command_handler_exists_on_the_app(self) -> None:
        # The app dispatches via getattr(self, command.handler); guard against a
        # typo'd or missing handler for any registered command.
        from privibe.cli.textual_ui.app import VibeApp

        registry = CommandRegistry()
        for name, cmd in registry.commands.items():
            assert hasattr(VibeApp, cmd.handler), (
                f"command '{name}' -> missing handler '{cmd.handler}' on VibeApp"
            )

