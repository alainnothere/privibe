"""Console mode entry point - plain text interactive interface.

This module implements a simple stdin/stdout based interface that replaces
the Textual TUI. It provides the same core functionality (chat, tool use,
commands) but without any terminal rendering overhead.

Key design decisions:
- Uses prompt_toolkit for input (better Windows support than raw readline)
- Outputs plain text with optional ANSI colors (disabled by default)
- Handles all AgentLoop events by printing them to stdout
- Implements console versions of approval prompts, question dialogs, and pickers
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from privibe.core.types import (
    AssistantEvent,
    BaseEvent,
    CompactEndEvent,
    CompactStartEvent,
    ReasoningEvent,
    ToolCallEvent,
    ToolResultEvent,
    WaitingForInputEvent,
)

if TYPE_CHECKING:
    from privibe.core.agent_loop import AgentLoop
    from privibe.core.types import ApprovalResponse


class ConsoleUI:
    """Main console mode interface.

    Replaces VibeApp with a simple event loop that:
    1. Reads user input from stdin
    2. Feeds it to AgentLoop
    3. Prints events as they arrive
    4. Handles approval/question callbacks with console prompts
    """

    def __init__(self, agent_loop: AgentLoop) -> None:
        self.agent_loop = agent_loop
        self._running = True

    async def run(self, initial_prompt: str | None = None) -> None:
        """Run the console interface loop.

        Args:
            initial_prompt: Optional first message to send to the agent
        """
        # Set up the callbacks that AgentLoop needs to interact with the user
        self.agent_loop.set_user_input_callback(self._user_input_callback)

        # If an initial prompt was provided, send it and wait for response
        if initial_prompt:
            await self._process_agent_response(initial_prompt)

        # Main loop: read input, process, repeat until user exits
        while self._running:
            try:
                user_input = await self._read_input()

                if not user_input:
                    continue

                # Handle slash commands
                if user_input.startswith("/"):
                    if not await self._handle_command(user_input):
                        continue  # Command handled, loop again

                # Send to agent and wait for response
                await self._process_agent_response(user_input)

            except KeyboardInterrupt:
                print("\nInterrupted. Type /exit to quit.")
                continue
            except EOFError:
                break

    async def _read_input(self) -> str:
        """Read user input from stdin with prompt_toolkit.

        Returns:
            User's input string, or empty string if nothing entered
        """
        # TODO: Use prompt_toolkit.prompt() for better line editing and history
        # For now, use simple input() - will need to add:
        # - Command history (up/down arrows)
        # - Ctrl+C handling
        # - Windows compatibility
        try:
            return input("\nyou: ").strip()
        except (KeyboardInterrupt, EOFError):
            return ""

    async def _user_input_callback(self, args: object) -> object:
        """Handle ask_user_question tool calls.

        This is called by AgentLoop when a tool needs to ask the user questions.
        We need to display the questions and collect answers via console prompts.

        TODO: Implement full AskUserQuestionArgs handling with:
        - Multiple questions
        - Choice selection (numbered list)
        - Multi-select support
        - "Other" free text option
        """
        # Placeholder - will raise error until fully implemented
        from privibe.core.tools.builtins.ask_user_question import (
            AskUserQuestionArgs,
            Answer,
            AskUserQuestionResult,
        )

        if not isinstance(args, AskUserQuestionArgs):
            raise RuntimeError(f"Unexpected question type: {type(args)}")

        print("\n[Questions from agent]")
        # TODO: Actually display questions and collect answers
        # For now, just return cancelled to unblock
        return AskUserQuestionResult(answers=[], cancelled=True)

    async def _process_agent_response(self, prompt: str) -> None:
        """Send prompt to agent and stream the response to stdout.

        Args:
            prompt: User's message to send to the agent
        """
        print()  # Blank line for readability

        # Stream events from AgentLoop and print them as they arrive
        async for event in self.agent_loop.act(prompt):
            await self._handle_event(event)

    async def _handle_event(self, event: BaseEvent) -> None:
        """Print an event to stdout.

        Each event type gets different formatting:
        - AssistantEvent: Print the assistant's response
        - ToolCallEvent: Show tool is running
        - ToolResultEvent: Show tool completed
        - ReasoningEvent: Optionally show reasoning (can be verbose)
        - Compact events: Show progress
        """
        match event:
            case AssistantEvent():
                # Stream assistant content as it arrives
                if hasattr(event, 'content') and event.content:
                    print(event.content, end="", flush=True)

            case ToolCallEvent():
                # Show tool starting
                tool_name = getattr(event, 'tool_name', 'unknown')
                print(f"\n[tool] {tool_name}...", flush=True)

            case ToolResultEvent():
                # Show tool completed
                tool_name = getattr(event, 'tool_name', 'unknown')
                if getattr(event, 'error', None):
                    print(f"[tool] {tool_name}: ERROR", flush=True)
                elif getattr(event, 'skipped', False):
                    print(f"[tool] {tool_name}: skipped", flush=True)
                else:
                    print(f"[tool] {tool_name}: done", flush=True)

            case ReasoningEvent():
                # Optionally show reasoning (many users want to see this)
                if hasattr(event, 'content') and event.content:
                    print(f"\n[thinking] {event.content}\n", flush=True)

            case CompactStartEvent():
                print("\n[compacting conversation history...]", flush=True)

            case CompactEndEvent():
                print("[done compacting]", flush=True)

            case WaitingForInputEvent():
                # Agent is waiting for user input
                pass

    async def _handle_command(self, command: str) -> bool:
        """Handle slash commands.

        Returns:
            True if command was handled, False if it should be passed to agent
        """
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        match cmd:
            case "/exit" | "/quit":
                print("Goodbye!")
                self._running = False
                return True

            case "/help":
                self._print_help()
                return True

            case "/model":
                await self._handle_model_command()
                return True

            case "/compact":
                # TODO: Implement compact command
                print("Compacting conversation history...")
                # TODO: Actually trigger compaction
                return True

            case _:
                # Unknown command, pass to agent
                return False

    def _print_help(self) -> None:
        """Print help message with available commands."""
        print("""
Available commands:
  /help      Show this help message
  /exit      Exit the application
  /model     Switch to a different model
  /compact   Summarize conversation history

Keyboard shortcuts:
  Ctrl+C     Interrupt agent (type again to exit)
""")

    async def _handle_model_command(self) -> None:
        """Handle /model command - show available models and let user pick.

        TODO: Implement full model picker with:
        - List all configured models
        - Show which one is active
        - Let user select by number
        - Switch agent_loop to new model
        """
        print("\nAvailable models:")
        print("  1. local (llama.cpp)")
        print("  2. Add more models in config.toml")
        print("\nModel switching not yet implemented in console mode.")
