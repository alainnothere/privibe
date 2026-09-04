"""Console mode entry point - plain text interactive interface.

This module implements a simple stdin/stdout based interface that replaces
the Textual TUI. It provides the same core functionality (chat, tool use,
approvals, questions, commands) but without any terminal rendering overhead:
every event is printed once, append-only, and never repainted.

Key design decisions:
- prompt_toolkit for input (prompt() only, never full-screen mode); it is
  the one Python line-input library with first-class Windows support
- Plain text output, no colors, no markup
- Same AgentLoop event stream the TUI consumes; nothing in core changes
- Console implementations of the approval and ask_user_question callbacks
"""

from __future__ import annotations

import asyncio
import difflib
from pathlib import Path
from typing import TYPE_CHECKING, cast

from privibe.core.config import (
    VibeConfig,
    cycle_llm_calls_per_turn,
    cycle_preview_lines,
    load_dotenv_values,
)
from privibe.core.session.session_loader import SessionLoader
from privibe.core.skills.parser import SkillParseError
from privibe.core.skills.render import load_skill_content
from privibe.core.tools.builtins.ask_user_question import (
    Answer,
    AskUserQuestionArgs,
    AskUserQuestionResult,
    Question,
)
from privibe.core.tools.builtins.search_replace import SEARCH_REPLACE_BLOCK_RE
from privibe.core.types import (
    AgentProfileChangedEvent,
    ApprovalResponse,
    AssistantEvent,
    BaseEvent,
    CompactEndEvent,
    CompactStartEvent,
    ReasoningEvent,
    Role,
    ToolCallEvent,
    ToolResultEvent,
    ToolStreamEvent,
    WaitingForInputEvent,
)
from privibe.core.utils.tags import (
    CancellationReason,
    get_user_cancellation_message,
    strip_known_tags,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

    from privibe.core.agent_loop import AgentLoop
    from privibe.core.tools.permissions import RequiredPermission

# Approval prompts show the tool args; cap the preview so a large file write
# does not flood the console.
_APPROVAL_PREVIEW_LIMIT = 4000

# How many transcript entries the automatic replay after a resume shows.
# /replay prints the whole conversation.
_REPLAY_TAIL_ENTRIES = 20


def _truncate_lines(content: str, max_lines: int) -> tuple[str, str | None]:
    """Truncate content to max_lines, returning (content, truncation_info).

    Same rule the TUI result widgets apply, re-expressed here because their
    version lives in a textual-importing module.
    """
    lines = content.strip("\n").split("\n")
    if len(lines) <= max_lines:
        return "\n".join(lines), None
    remaining = len(lines) - max_lines
    return "\n".join(lines[:max_lines]), f"... ({remaining} more lines)"


def _parse_search_replace_to_diff(content: str) -> list[str]:
    """Parse SEARCH/REPLACE blocks into unified diff lines (TUI parity)."""
    all_diff_lines: list[str] = []
    matches = SEARCH_REPLACE_BLOCK_RE.findall(content)
    if not matches:
        return [content[:500]] if content else []

    for i, (search_text, replace_text) in enumerate(matches):
        if i > 0:
            all_diff_lines.append("")
        search_lines = search_text.strip("\n").split("\n")
        replace_lines = replace_text.strip("\n").split("\n")
        diff = difflib.unified_diff(search_lines, replace_lines, lineterm="", n=2)
        all_diff_lines.extend(list(diff)[2:])

    return all_diff_lines


_TODO_STATUS_MARKS = {
    "in_progress": "[>]",
    "pending": "[ ]",
    "completed": "[x]",
    "cancelled": "[-]",
}

_HELP_TEXT = """
Available commands:
  /help      Show this help message
  /model     List models and switch the active one
  /compact   Summarize conversation history to free context
  /clear     Clear conversation history
  /resume    Pick a previous session to resume (replaces current conversation)
  /replay    Reprint the whole conversation (resume shows only the tail)
  /log       Show the current session log directory
  /preview-lines  Cycle how many tool output lines are shown (3 / 5 / 10)
  /llm-calls-per-turn  Cycle the LLM-call budget per message (30 / 60 / 90)
  /effort    Cycle reasoning effort stamped on new messages (low / medium / xhigh)
  /exit      Exit (also /quit, or Ctrl+D)

Anything else is sent to the agent. /name runs a skill if one matches.
Ctrl+C interrupts the agent while it is responding.
"""


class ConsoleUI:
    """Interactive console frontend on the AgentLoop event seam.

    Reads user input line by line, feeds it to AgentLoop.act(), prints
    events as they arrive, and answers approval/question callbacks with
    console prompts.
    """

    def __init__(self, agent_loop: AgentLoop) -> None:
        self.agent_loop = agent_loop
        self._running = True
        # Which streamed block is currently open on stdout: None,
        # "assistant" or "reasoning". Used to insert prefixes/newlines
        # exactly once around delta sequences.
        self._open_block: str | None = None
        self._prompt_session = None  # created lazily; needs a terminal

    async def run(
        self,
        initial_prompt: str | None = None,
        show_resume_picker: bool = False,
    ) -> None:
        self.agent_loop.set_approval_callback(self._approval_callback)
        self.agent_loop.set_user_input_callback(self._user_input_callback)

        if show_resume_picker:
            await self._pick_and_resume_session()
        elif len(self.agent_loop.messages) > 1:
            # History restored before we started (--resume <id> / --continue).
            self._print_transcript(_REPLAY_TAIL_ENTRIES)

        if initial_prompt:
            await self._dispatch(initial_prompt)

        while self._running:
            try:
                user_input = (await self._read_line("\nyou: ")).strip()
            except KeyboardInterrupt:
                print("(Ctrl+C at the prompt does nothing; /exit or Ctrl+D quits)")
                continue
            except EOFError:
                break

            if not user_input:
                continue

            await self._dispatch(user_input)

        print("Bye!")

    # ------------------------------------------------------------------
    # Input plumbing
    # ------------------------------------------------------------------

    async def _read_line(self, prompt: str) -> str:
        """Read one line with history and line editing.

        Raises KeyboardInterrupt on Ctrl+C and EOFError on Ctrl+D, exactly
        like prompt_toolkit's prompt_async.
        """
        if self._prompt_session is None:
            from prompt_toolkit import PromptSession

            self._prompt_session = PromptSession()
        return await self._prompt_session.prompt_async(prompt)

    async def _read_reply(self, prompt: str) -> str:
        """Read a short reply for approvals/questions, without polluting the
        main input history. Cancellation exceptions propagate to the caller.
        """
        from prompt_toolkit.shortcuts import PromptSession

        session: PromptSession[str] = PromptSession()
        return (await session.prompt_async(prompt)).strip()

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, user_input: str) -> None:
        if user_input.startswith("/"):
            if await self._handle_command(user_input):
                return
            skill_prompt = self._resolve_skill(user_input)
            if skill_prompt is None:
                cmd = user_input.split(maxsplit=1)[0]
                print(f"Unknown command: {cmd} (see /help)")
                return
            await self._run_turn(skill_prompt)
            return

        await self._run_turn(user_input)

    def _resolve_skill(self, user_input: str) -> str | None:
        """Map /name [args] to a skill prompt, mirroring the TUI behavior."""
        parts = user_input[1:].strip().split(None, 1)
        if not parts:
            return None
        skill_info = self.agent_loop.skill_manager.get_skill(parts[0].lower())
        if not skill_info:
            return None

        try:
            skill_content = load_skill_content(skill_info)
        except (OSError, SkillParseError) as e:
            print(f"Failed to read skill file: {e}")
            # Handled: do not fall through to "unknown command".
            return ""

        if len(parts) > 1:
            skill_content = f"{user_input}\n\n{skill_content}"
        return skill_content

    # ------------------------------------------------------------------
    # Agent turn
    # ------------------------------------------------------------------

    async def _run_turn(self, prompt: str) -> None:
        if not prompt:
            return
        print()
        try:
            async for event in self.agent_loop.act(prompt):
                self._handle_event(event)
        except asyncio.CancelledError:
            # First Ctrl+C during a turn: asyncio.Runner cancels the main
            # task; swallow it here so the REPL survives, and uncancel so
            # the runner does not tear the loop down afterwards.
            task = asyncio.current_task()
            if task is not None:
                task.uncancel()
            self._close_block()
            print("\n[interrupted]")
        except Exception as e:
            self._close_block()
            print(f"\n[error] {e}")
        finally:
            self._close_block()

    def _close_block(self) -> None:
        """Terminate an open assistant/reasoning stream with a newline."""
        if self._open_block is not None:
            print(flush=True)
            self._open_block = None

    def _handle_event(self, event: BaseEvent) -> None:
        match event:
            case AssistantEvent():
                if event.content:
                    if self._open_block != "assistant":
                        self._close_block()
                        self._open_block = "assistant"
                    print(event.content, end="", flush=True)

            case ReasoningEvent():
                if event.content:
                    if self._open_block != "reasoning":
                        self._close_block()
                        print("[thinking] ", end="", flush=True)
                        self._open_block = "reasoning"
                    print(event.content, end="", flush=True)

            case ToolCallEvent():
                self._close_block()
                summary = self._call_summary(event)
                print(f"[tool] {event.tool_name}: {summary}", flush=True)

            case ToolStreamEvent():
                self._close_block()
                print(f"  | {event.message}", flush=True)

            case ToolResultEvent():
                self._close_block()
                print(self._result_line(event), flush=True)
                self._print_result_body(event)

            case CompactStartEvent():
                self._close_block()
                print(
                    f"[compact] context at {event.current_context_tokens} tokens "
                    f"(threshold {event.threshold}), summarizing...",
                    flush=True,
                )

            case CompactEndEvent():
                self._close_block()
                print(
                    f"[compact] done: {event.old_context_tokens} -> "
                    f"{event.new_context_tokens} tokens",
                    flush=True,
                )

            case AgentProfileChangedEvent():
                self._close_block()
                print(f"[agent] switched to {event.agent_name}", flush=True)

            case WaitingForInputEvent():
                pass

    def _call_summary(self, event: ToolCallEvent) -> str:
        get_display = getattr(event.tool_class, "get_call_display", None)
        if get_display is not None:
            try:
                return get_display(event).summary
            except Exception:
                pass
        return "running"

    def _result_line(self, event: ToolResultEvent) -> str:
        if event.error:
            return f"[tool] {event.tool_name}: ERROR: {event.error}"
        if event.cancelled:
            return f"[tool] {event.tool_name}: cancelled"
        if event.skipped:
            reason = f" ({event.skip_reason})" if event.skip_reason else ""
            return f"[tool] {event.tool_name}: skipped{reason}"

        message = "done"
        if event.tool_class is not None:
            get_display = getattr(event.tool_class, "get_result_display", None)
            if get_display is not None:
                try:
                    message = get_display(event).message
                except Exception:
                    message = "done"
        return f"[tool] {event.tool_name}: {message}"

    def _print_result_body(self, event: ToolResultEvent) -> None:
        """Print the tool result content below the status line.

        Mirrors what the TUI shows in its default collapsed state: warnings,
        then a red/green file diff for file-mutating tools, otherwise the
        per-tool preview (bash stdout, grep matches, ...) truncated to the
        shared tool_result_preview_lines config value.
        """
        if event.error or event.skipped or event.cancelled:
            return

        preview_lines = max(
            1, getattr(self.agent_loop.config, "tool_result_preview_lines", 3)
        )

        for warning in self._result_warnings(event):
            print(f"  ! {warning}", flush=True)

        # File-mutating tools show a diff instead of a field dump; the rows
        # arrive display-ready from core, +/- prefixes included. Same
        # search_replace exception as the TUI, which keeps its own diff view.
        if event.file_diff is not None and event.tool_name != "search_replace":
            for line in self._file_diff_lines(event.file_diff, preview_lines):
                print(f"  {line}", flush=True)
            return

        for line in self._tool_body_lines(event, preview_lines):
            print(f"  {line}", flush=True)

    def _result_warnings(self, event: ToolResultEvent) -> list[str]:
        if event.tool_class is None:
            return []
        get_display = getattr(event.tool_class, "get_result_display", None)
        if get_display is None:
            return []
        try:
            return get_display(event).warnings
        except Exception:
            return []

    @staticmethod
    def _file_diff_lines(file_diff: object, max_hunks: int) -> list[str]:
        note = getattr(file_diff, "note", None)
        if note:
            return [note]
        hunks = getattr(file_diff, "hunks", [])
        kind = getattr(file_diff, "kind", "diff")
        shown = hunks[:max_hunks] if kind == "diff" else hunks
        lines = [text for hunk in shown for _css_class, text in hunk]
        remaining = len(hunks) - len(shown)
        if remaining > 0:
            label = "change" if remaining == 1 else "changes"
            lines.append(f"({remaining} more {label} not shown)")
        return lines

    def _tool_body_lines(
        self, event: ToolResultEvent, preview_lines: int
    ) -> list[str]:
        """Per-tool collapsed preview, matching the TUI result widgets."""
        result = event.result
        if result is None:
            return []

        match event.tool_name:
            case "bash":
                stdout = getattr(result, "stdout", "")
                lines = (
                    self._preview(stdout, preview_lines)
                    if stdout
                    else ["(no content)"]
                )
            case "grep":
                lines = self._preview(getattr(result, "matches", ""), preview_lines)
            case "write_file":
                # Only reached when core produced no file diff (e.g. no-op).
                lines = self._preview(getattr(result, "content", ""), preview_lines)
            case "search_replace":
                content_value = getattr(result, "content", "")
                lines = (
                    _parse_search_replace_to_diff(content_value)
                    if content_value
                    else []
                )
            case "todo":
                lines = self._todo_lines(getattr(result, "todos", None) or [])
            case _:
                # read_file, ask_user_question and unknown tools show nothing
                # in the TUI's collapsed state either.
                lines = []
        return lines

    @staticmethod
    def _preview(text: str, preview_lines: int) -> list[str]:
        if not text:
            return []
        content, info = _truncate_lines(text, preview_lines)
        return content.split("\n") + ([info] if info else [])

    @staticmethod
    def _todo_lines(todos: list) -> list[str]:
        lines: list[str] = []
        for status in ("in_progress", "pending", "completed", "cancelled"):
            for todo in todos:
                todo_status = getattr(todo.status, "value", str(todo.status))
                if todo_status == status:
                    lines.append(f"{_TODO_STATUS_MARKS[status]} {todo.content}")
        return lines

    # ------------------------------------------------------------------
    # Approval callback
    # ------------------------------------------------------------------

    async def _approval_callback(
        self,
        tool_name: str,
        args: BaseModel,
        tool_call_id: str,
        required_permissions: list[RequiredPermission] | None,
    ) -> tuple[ApprovalResponse, str | None]:
        # Mirror the TUI: auto-approve only when configured AND the tool is
        # enabled in the main agent, so subagents respect tool restrictions.
        # Escalated permissions always reach the human.
        if (
            self.agent_loop.config.auto_approve
            and tool_name in self.agent_loop.tool_manager.available_tools
            and not any(rp.escalated for rp in required_permissions or [])
        ):
            return (ApprovalResponse.YES, None)

        self._close_block()
        print(f"\n[approval needed] {tool_name}")
        self._print_approval_details(tool_name, args, required_permissions)

        while True:
            try:
                reply = (
                    await self._read_reply(
                        "Allow? [y]es once / [a]lways / [n]o / [d]eny for session: "
                    )
                ).lower()
            except (KeyboardInterrupt, EOFError):
                feedback = str(
                    get_user_cancellation_message(
                        CancellationReason.OPERATION_CANCELLED
                    )
                )
                return (ApprovalResponse.NO, feedback)

            match reply:
                case "y" | "yes":
                    return (ApprovalResponse.YES, None)
                case "a" | "always":
                    self.agent_loop.approve_always(tool_name, required_permissions)
                    return (ApprovalResponse.YES, None)
                case "n" | "no":
                    feedback = await self._deny_feedback()
                    return (ApprovalResponse.NO, feedback)
                case "d" | "deny":
                    self.agent_loop.deny_always(tool_name, required_permissions)
                    return (
                        ApprovalResponse.NO,
                        "Denied for this session. Do not attempt this again.",
                    )
                case _:
                    print("Please answer y, a, n, or d.")

    def _print_approval_details(
        self,
        tool_name: str,
        args: BaseModel,
        required_permissions: list[RequiredPermission] | None,
    ) -> None:
        tool_class = self.agent_loop.tool_manager.available_tools.get(tool_name)
        summary: str | None = None
        content: str | None = None
        format_display = getattr(tool_class, "format_call_display", None)
        if format_display is not None:
            try:
                display = format_display(args)
                summary = display.summary
                content = display.content
            except Exception:
                summary = None

        if summary:
            print(f"  {summary}")
        if content:
            if len(content) > _APPROVAL_PREVIEW_LIMIT:
                content = (
                    content[:_APPROVAL_PREVIEW_LIMIT]
                    + f"\n  ... (truncated, {len(content)} chars total)"
                )
            for line in content.splitlines():
                print(f"  {line}")
        if summary is None and content is None:
            args_text = args.model_dump_json()
            if len(args_text) > _APPROVAL_PREVIEW_LIMIT:
                args_text = args_text[:_APPROVAL_PREVIEW_LIMIT] + "..."
            print(f"  args: {args_text}")

        for rp in required_permissions or []:
            print(f"  permission: {rp.label}")

    async def _deny_feedback(self) -> str:
        try:
            feedback = await self._read_reply("Feedback for the agent (optional): ")
        except (KeyboardInterrupt, EOFError):
            feedback = ""
        if feedback:
            return feedback
        return str(
            get_user_cancellation_message(CancellationReason.OPERATION_CANCELLED)
        )

    # ------------------------------------------------------------------
    # ask_user_question callback
    # ------------------------------------------------------------------

    async def _user_input_callback(self, args: BaseModel) -> BaseModel:
        if not isinstance(args, AskUserQuestionArgs):
            raise RuntimeError(f"Unexpected question args type: {type(args)}")

        self._close_block()
        if args.content_preview:
            print(f"\n{args.content_preview}")

        answers: list[Answer] = []
        for question in args.questions:
            try:
                answer = await self._ask_one_question(question)
            except (KeyboardInterrupt, EOFError):
                answer = None
            if answer is None:
                print("(cancelled)")
                return AskUserQuestionResult(answers=[], cancelled=True)
            answers.append(answer)

        return AskUserQuestionResult(answers=answers, cancelled=False)

    async def _ask_one_question(self, question: Question) -> Answer | None:
        """Ask a single question; returns None when the user cancels."""
        header = f"[{question.header}] " if question.header else ""
        print(f"\n{header}{question.question}")

        labels = [choice.label for choice in question.options]
        for i, choice in enumerate(question.options, start=1):
            desc = f" - {choice.description}" if choice.description else ""
            print(f"  {i}. {choice.label}{desc}")
        other_index = None
        if not question.hide_other:
            other_index = len(labels) + 1
            print(f"  {other_index}. Other (type your own answer)")

        if question.multi_select:
            hint = "numbers, comma separated, or c to cancel"
        else:
            hint = "number, or c to cancel"

        while True:
            reply = await self._read_reply(f"Choose ({hint}): ")
            if reply.lower() == "c":
                return None
            picks = self._parse_picks(
                reply,
                max_index=other_index or len(labels),
                multi=question.multi_select,
            )
            if picks is None:
                print("Invalid selection, try again.")
                continue

            selected: list[str] = []
            is_other = False
            for pick in picks:
                if other_index is not None and pick == other_index:
                    free_text = await self._read_reply("Your answer: ")
                    if not free_text:
                        print("Empty answer, try again.")
                        break
                    selected.append(free_text)
                    is_other = True
                else:
                    selected.append(labels[pick - 1])
            else:
                return Answer(
                    question=question.question,
                    answer=", ".join(selected),
                    is_other=is_other,
                )

    @staticmethod
    def _parse_picks(reply: str, max_index: int, multi: bool) -> list[int] | None:
        parts = [p.strip() for p in reply.split(",")] if multi else [reply.strip()]
        if not multi and "," in reply:
            return None
        picks: list[int] = []
        for part in parts:
            if not part.isdigit():
                return None
            value = int(part)
            if not 1 <= value <= max_index:
                return None
            if value not in picks:
                picks.append(value)
        return picks or None

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    async def _handle_command(self, command: str) -> bool:
        """Run a builtin command. Returns True when the input was consumed."""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()

        match cmd:
            case "/exit" | "/quit":
                self._running = False
            case "/help":
                print(_HELP_TEXT)
            case "/model":
                await self._command_model()
            case "/compact":
                await self._command_compact()
            case "/clear":
                await self._command_clear()
            case "/resume":
                await self._pick_and_resume_session()
            case "/log":
                self._command_log()
            case "/preview-lines":
                self._command_preview_lines()
            case "/llm-calls-per-turn":
                self._command_llm_calls_per_turn()
            case "/effort":
                self._command_effort()
            case "/replay":
                self._command_replay()
            case _:
                return False
        return True

    def _command_replay(self) -> None:
        if not self._transcript_entries(None):
            print("No conversation history.")
            return
        self._print_transcript(None)

    def _command_effort(self) -> None:
        from privibe.cli.commands import effort_cycle_notice
        from privibe.core.config import cycle_reasoning_effort

        new_value = cycle_reasoning_effort(self.agent_loop.current_reasoning_effort())
        self.agent_loop.set_reasoning_effort(new_value)
        config = self.agent_loop.config
        provider = config.get_provider_for_model(config.get_active_model())
        notice = effort_cycle_notice(
            new_value,
            provider.name,
            getattr(provider, "per_message_reasoning_effort", False),
            getattr(self, "_effort_backend_noticed", False),
        )
        if new_value != "off":
            self._effort_backend_noticed = True
        print(notice)

    def _command_preview_lines(self) -> None:
        config = self.agent_loop.config
        new_value = cycle_preview_lines(
            config.tool_result_preview_lines,
            config.tool_result_preview_options,
        )
        VibeConfig.save_updates({"tool_result_preview_lines": new_value})
        self.agent_loop.refresh_config()
        print(f"Tool result preview set to {new_value} lines.")

    def _command_llm_calls_per_turn(self) -> None:
        config = self.agent_loop.config
        new_value = cycle_llm_calls_per_turn(
            config.max_llm_calls_per_turn,
            config.llm_calls_per_turn_options,
        )
        VibeConfig.save_updates({"max_llm_calls_per_turn": new_value})
        self.agent_loop.refresh_config()
        print(f"LLM-call budget set to {new_value} calls per turn.")

    async def _command_model(self) -> None:
        config = self.agent_loop.config
        models = config.models
        if not models:
            print("No models configured.")
            return

        active_alias = str(config.active_model)
        print("\nAvailable models:")
        for i, model in enumerate(models, start=1):
            marker = "*" if model.alias == active_alias else " "
            name = f" ({model.name})" if model.name != model.alias else ""
            print(f" {marker} {i}. {model.alias}{name}")

        try:
            reply = await self._read_reply("Switch to (number, empty to keep): ")
        except (KeyboardInterrupt, EOFError):
            return
        if not reply:
            return
        if not reply.isdigit() or not 1 <= int(reply) <= len(models):
            print("Invalid selection.")
            return

        alias = models[int(reply) - 1].alias
        if alias == active_alias:
            print(f"Already using {alias}.")
            return

        # Same sequence the TUI runs on model switch: persist, re-read .env,
        # reload the config into the running loop, then re-detect context size.
        VibeConfig.save_updates({"active_model": alias})
        load_dotenv_values()
        base_config = VibeConfig.load()
        notice = await self.agent_loop.apply_runtime_config(base_config=base_config)
        if notice:
            print(notice)
        fallback_notice = self.agent_loop.config.model_fallback_notice()
        if fallback_notice:
            print(fallback_notice)
        ctx_msg = await self.agent_loop.resolve_context_size()
        if ctx_msg:
            print(ctx_msg)
        print(f"Active model: {self.agent_loop.config.active_model}")

    async def _command_compact(self) -> None:
        if len(self.agent_loop.messages) <= 1:
            print("No conversation history to compact yet.")
            return
        print("Compacting conversation history...")
        try:
            summary = await self.agent_loop.compact()
        except Exception as e:
            print(f"Compaction failed: {e}")
            return
        print(f"Compacted. Summary:\n{summary}")

    async def _command_clear(self) -> None:
        await self.agent_loop.clear_history()
        print("Conversation history cleared.")

    def _command_log(self) -> None:
        logger = self.agent_loop.session_logger
        if not logger.enabled:
            print("Session logging is disabled in configuration.")
            return
        print(str(logger.session_dir))

    # ------------------------------------------------------------------
    # Transcript replay
    # ------------------------------------------------------------------

    def _transcript_entries(self, limit: int | None) -> list[str]:
        """Render restored history as replay entries, newest last.

        Same conversation definition core uses for resume-picker previews:
        user/assistant roles only, harness-injected messages dropped, known
        tags stripped. Tool calls become one-line markers for continuity;
        tool outputs and reasoning are not replayed.
        """
        entries: list[str] = []
        for message in self.agent_loop.messages:
            if message.injected:
                continue
            if message.role == Role.user:
                text = strip_known_tags(message.content or "").strip()
                if text:
                    entries.append(f"you: {text}")
            elif message.role == Role.assistant:
                text = strip_known_tags(message.content or "").strip()
                if text:
                    entries.append(text)
                for tool_call in message.tool_calls or []:
                    if tool_call.function.name:
                        entries.append(f"[tool] {tool_call.function.name}")

        if limit is not None and len(entries) > limit:
            hidden = len(entries) - limit
            entries = [
                f"(... {hidden} earlier entries not shown; /replay shows all)",
                *entries[-limit:],
            ]
        return entries

    def _print_transcript(self, limit: int | None) -> None:
        entries = self._transcript_entries(limit)
        if not entries:
            return
        print("\n--- resumed conversation ---")
        for i, entry in enumerate(entries):
            if i > 0 and entry.startswith("you: "):
                print()
            print(entry)
        print("--- end of history ---")

    # ------------------------------------------------------------------
    # Session resume
    # ------------------------------------------------------------------

    async def _pick_and_resume_session(self) -> None:
        config = self.agent_loop.config
        if not config.session_logging.enabled:
            print("Session logging is disabled; nothing to resume.")
            return

        sessions = SessionLoader.list_sessions(config.session_logging)
        if not sessions:
            print("No previous sessions found.")
            return

        sessions.sort(key=lambda s: s["end_time"] or "", reverse=True)
        sessions = sessions[:20]

        print("\nPrevious sessions (newest first):")
        for i, info in enumerate(sessions, start=1):
            title = info["title"] or info["session_id"]
            when = f" ({info['end_time']})" if info["end_time"] else ""
            print(f"  {i}. {title}{when}")
        print("Resuming replaces the current conversation.")

        try:
            reply = await self._read_reply("Resume (number, empty to skip): ")
        except (KeyboardInterrupt, EOFError):
            return
        if not reply:
            return
        if not reply.isdigit() or not 1 <= int(reply) <= len(sessions):
            print("Invalid selection.")
            return

        info = sessions[int(reply) - 1]
        # SessionInfo.session_path is the messages.jsonl file; the loader and
        # restore APIs all take the session directory.
        session_path = Path(info["session_path"]).parent
        try:
            _, metadata = SessionLoader.load_session(session_path)
        except Exception as e:
            print(f"Failed to load session: {e}")
            return

        # Same steps as cli._resume_previous_session.
        session_id = cast(
            str, metadata.get("session_id", self.agent_loop.session_id)
        )
        self.agent_loop.session_id = session_id
        self.agent_loop.session_logger.resume_existing_session(
            session_id, session_path
        )
        self.agent_loop.messages.restore(session_path)
        print(
            f"Resumed session {session_id} "
            f"with {len(self.agent_loop.messages)} messages."
        )
        self._print_transcript(_REPLAY_TAIL_ENTRIES)
