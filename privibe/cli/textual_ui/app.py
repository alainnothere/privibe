from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum, auto
import gc
import os
from pathlib import Path
import signal
import time
from typing import Any, ClassVar, assert_never, cast
from weakref import WeakKeyDictionary
import webbrowser

from pydantic import BaseModel
from rich import print as rprint
from textual.app import WINDOWS, App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, VerticalGroup, VerticalScroll
from textual.driver import Driver
from textual.events import AppBlur, AppFocus, MouseUp
from textual.widget import Widget
from textual.widgets import Static

from privibe import __version__ as CORE_VERSION
from privibe.cli.clipboard import copy_selection_to_clipboard, is_reliable_clipboard_available
from privibe.cli.commands import CommandRegistry
from privibe.cli.narrator_manager import (
    NarratorManager,
    NarratorManagerPort,
    NarratorState,
)
from privibe.cli.terminal_setup import setup_terminal
from privibe.cli.textual_ui.handlers.event_handler import EventHandler
from privibe.cli.textual_ui.notifications import (
    NotificationContext,
    NotificationPort,
    TextualNotificationAdapter,
)
from privibe.cli.textual_ui.session_exit import print_session_resume_message
from privibe.cli.textual_ui.widgets.approval_app import ApprovalApp
from privibe.cli.textual_ui.widgets.banner.banner import Banner
from privibe.cli.textual_ui.widgets.chat_input import ChatInputContainer
from privibe.cli.textual_ui.widgets.chat_input.text_area import ChatTextArea
from privibe.cli.textual_ui.widgets.compact import CompactMessage
from privibe.cli.textual_ui.widgets.config_app import ConfigApp
from privibe.cli.textual_ui.widgets.context_progress import ContextProgress, TokenState
from privibe.cli.textual_ui.widgets.feedback_bar import FeedbackBar
from privibe.cli.textual_ui.widgets.load_more import HistoryLoadMoreRequested
from privibe.cli.textual_ui.widgets.loading import LoadingWidget, paused_timer
from privibe.cli.textual_ui.widgets.messages import (
    BashOutputMessage,
    ErrorMessage,
    InterruptMessage,
    ReasoningMessage,
    StreamingMessageBase,
    UserCommandMessage,
    UserMessage,
    WarningMessage,
)
from privibe.cli.textual_ui.widgets.model_picker import ModelPickerApp
from privibe.cli.textual_ui.widgets.narrator_status import NarratorStatus
from privibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from privibe.cli.textual_ui.widgets.path_display import PathDisplay
from privibe.cli.textual_ui.widgets.proxy_setup_app import ProxySetupApp
from privibe.cli.textual_ui.widgets.question_app import QuestionApp
from privibe.cli.textual_ui.widgets.rewind_app import RewindApp
from privibe.cli.textual_ui.widgets.session_picker import SessionPickerApp
from privibe.cli.textual_ui.widgets.tools import ToolResultMessage
from privibe.cli.textual_ui.widgets.voice_app import VoiceApp
from privibe.cli.textual_ui.windowing import (
    HISTORY_RESUME_TAIL_MESSAGES,
    LOAD_MORE_BATCH_SIZE,
    HistoryLoadMoreManager,
    SessionWindowing,
    build_history_widgets,
    create_resume_plan,
    non_system_history_messages,
    should_resume_history,
    sync_backfill_state,
)
from privibe.cli.voice_manager import VoiceManager, VoiceManagerPort
from privibe.cli.voice_manager.voice_manager_port import TranscribeState
from privibe.core.agent_loop import AgentLoop
from privibe.core.agents import AgentProfile
from privibe.core.audio_player.audio_player import AudioPlayer
from privibe.core.audio_recorder import AudioRecorder
from privibe.core.autocompletion.path_prompt_adapter import render_path_prompt
from privibe.core.config import (
    VibeConfig,
    cycle_message_prune_rows,
    cycle_preview_lines,
)
from privibe.core.config.harness_files._harness_manager import get_harness_files_manager
from privibe.core.logger import logger
from privibe.core.paths import AGENTS_MD_FILENAME, HISTORY_FILE, VIBE_HOME
from privibe.core.rewind import RewindError
from privibe.core.session.resume_sessions import (
    ResumeSessionInfo,
    list_local_resume_sessions,
    short_session_id,
)
from privibe.core.session.session_loader import SessionLoader
from privibe.core.tools.builtins.ask_user_question import (
    AskUserQuestionArgs,
    AskUserQuestionResult,
    Choice,
    Question,
)
from privibe.core.tools.permissions import RequiredPermission
from privibe.core.transcribe import make_transcribe_client
from privibe.core.types import (
    AgentStats,
    ApprovalResponse,
    BaseEvent,
    LLMMessage,
    RateLimitError,
    Role,
    WaitingForInputEvent,
)
from privibe.core.utils import (
    CancellationReason,
    get_user_cancellation_message,
    is_dangerous_directory,
)
from privibe.core.utils.io import read_safe


class BottomApp(StrEnum):
    """Bottom panel app types.

    Convention: Each value must match the widget class name with "App" suffix removed.
    E.g., ApprovalApp -> Approval, ConfigApp -> Config, QuestionApp -> Question.
    This allows dynamic lookup via: BottomApp[type(widget).__name__.removesuffix("App")]
    """

    Approval = auto()
    Config = auto()
    Input = auto()
    ModelPicker = auto()
    ProxySetup = auto()
    Question = auto()
    Rewind = auto()
    SessionPicker = auto()
    Voice = auto()


class ChatScroll(VerticalScroll):
    """Optimized scroll container that skips cascading style recalculations."""

    @property
    def is_at_bottom(self) -> bool:
        return self.scroll_target_y >= (self.max_scroll_y - 3)

    _reanchor_pending: bool = False
    _scrolling_down: bool = False

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        self._scrolling_down = new_value >= old_value

    def release_anchor(self) -> None:
        super().release_anchor()
        # Textual's MRO dispatch calls Widget._on_mouse_scroll_down AFTER
        # our override, so any re-anchor we do gets immediately undone.
        # Defer the re-check until all handlers for this event have finished.
        if not self._reanchor_pending:
            self._reanchor_pending = True
            self.call_later(self._maybe_reanchor)

    def _maybe_reanchor(self) -> None:
        self._reanchor_pending = False
        if (
            self._anchored
            and self._anchor_released
            and self.is_at_bottom
            and self._scrolling_down
        ):
            self.anchor()

    def update_node_styles(self, animate: bool = True) -> None:
        pass


async def prune_oldest_children(
    messages_area: Widget, low_mark: int, high_mark: int
) -> bool:
    """Remove the oldest children so the virtual height stays within bounds.

    Walks children back-to-front to find how much to keep (up to *low_mark*
    of visible height), then removes everything before that point.
    """
    total_height = messages_area.virtual_size.height
    if total_height <= high_mark:
        return False

    children = messages_area.children
    if not children:
        return False

    accumulated = 0
    cut = len(children)

    for child in reversed(children):
        if not child.display:
            cut -= 1
            continue
        accumulated += child.outer_size.height
        cut -= 1
        if accumulated >= low_mark:
            break

    to_remove = list(children[:cut])
    if not to_remove:
        return False

    await messages_area.remove_children(to_remove)
    return True


@dataclass(frozen=True, slots=True)
class StartupOptions:
    initial_prompt: str | None = None
    show_resume_picker: bool = False


class VibeApp(App):  # noqa: PLR0904
    ENABLE_COMMAND_PALETTE = False
    CSS_PATH = "app.tcss"
    PAUSE_GC_ON_SCROLL: ClassVar[bool] = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+c", "clear_quit", "Quit", show=False),
        Binding("ctrl+d", "force_quit", "Quit", show=False, priority=True),
        Binding("ctrl+z", "suspend_with_message", "Suspend", show=False, priority=True),
        Binding("escape", "interrupt", "Interrupt", show=False, priority=True),
        Binding("ctrl+o", "toggle_tool", "Toggle Tool", show=False),
        Binding("ctrl+y", "copy_selection", "Copy", show=False, priority=True),
        Binding("ctrl+shift+c", "copy_selection", "Copy", show=False, priority=True),
        Binding("shift+tab", "cycle_mode", "Cycle Mode", show=False, priority=True),
        Binding("shift+up", "scroll_chat_up", "Scroll Up", show=False, priority=True),
        Binding(
            "shift+down", "scroll_chat_down", "Scroll Down", show=False, priority=True
        ),
        Binding(
            "shift+home", "scroll_chat_home", "Scroll To Top", show=False, priority=True
        ),
        Binding(
            "shift+end", "scroll_chat_end", "Scroll To Bottom", show=False, priority=True
        ),
        Binding("pageup", "scroll_chat_page_up", "Page Up", show=False, priority=True),
        Binding(
            "pagedown", "scroll_chat_page_down", "Page Down", show=False, priority=True
        ),
        Binding("alt+up", "rewind_prev", "Rewind Previous", show=False, priority=True),
        Binding("ctrl+p", "rewind_prev", "Rewind Previous", show=False, priority=True),
        Binding("alt+down", "rewind_next", "Rewind Next", show=False, priority=True),
        Binding("ctrl+n", "rewind_next", "Rewind Next", show=False, priority=True),
    ]

    def __init__(
        self,
        agent_loop: AgentLoop,
        startup: StartupOptions | None = None,
        terminal_notifier: NotificationPort | None = None,
        voice_manager: VoiceManagerPort | None = None,
        narrator_manager: NarratorManagerPort | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.scroll_sensitivity_y = 4.0
        self.agent_loop = agent_loop
        self._voice_manager: VoiceManagerPort = (
            voice_manager or self._make_default_voice_manager()
        )
        self._terminal_notifier = terminal_notifier or TextualNotificationAdapter(
            self,
            get_enabled=lambda: self.config.enable_notifications,
            default_title="Privibe",
        )
        self._agent_running = False
        self._interrupt_requested = False
        self._agent_task: asyncio.Task | None = None
        self._ctrl_c_exit_time: float | None = None

        self._loading_widget: LoadingWidget | None = None
        self._pending_approval: asyncio.Future | None = None
        self._pending_question: asyncio.Future | None = None
        self._user_interaction_lock = asyncio.Lock()

        self.event_handler: EventHandler | None = None

        self.commands = CommandRegistry()

        self._chat_input_container: ChatInputContainer | None = None
        self._current_bottom_app: BottomApp = BottomApp.Input

        self.history_file = HISTORY_FILE.path

        self._tools_collapsed = True
        self._windowing = SessionWindowing(load_more_batch_size=LOAD_MORE_BATCH_SIZE)
        self._load_more = HistoryLoadMoreManager()
        self._tool_call_map: dict[str, str] | None = None
        self._history_widget_indices: WeakKeyDictionary[Widget, int] = (
            WeakKeyDictionary()
        )
        opts = startup or StartupOptions()
        self._initial_prompt = opts.initial_prompt
        self._show_resume_picker = opts.show_resume_picker
        self._last_escape_time: float | None = None
        self._banner: Banner | None = None
        self._cached_messages_area: Widget | None = None
        self._cached_chat: ChatScroll | None = None
        self._cached_loading_area: Widget | None = None
        self._switch_agent_generation = 0
        self._narrator_manager: NarratorManagerPort = (
            narrator_manager or self._make_default_narrator_manager()
        )

        self._rewind_mode = False
        self._rewind_highlighted_widget: UserMessage | None = None

    @property
    def config(self) -> VibeConfig:
        return self.agent_loop.config

    def compose(self) -> ComposeResult:
        with ChatScroll(id="chat"):
            self._banner = Banner(
                self.config, self.agent_loop.skill_manager, self.agent_loop.mcp_registry
            )
            yield self._banner
            yield VerticalGroup(id="messages")

        with Horizontal(id="loading-area"):
            yield NarratorStatus(self._narrator_manager)
            yield Static(id="loading-area-content")
            yield FeedbackBar()

        with Static(id="bottom-app-container"):
            yield ChatInputContainer(
                history_file=self.history_file,
                command_registry=self.commands,
                id="input-container",
                safety=self.agent_loop.agent_profile.safety,
                agent_name=self.agent_loop.agent_profile.display_name.lower(),
                skill_entries_getter=self._get_skill_entries,
                file_watcher_for_autocomplete_getter=self._is_file_watcher_enabled,
                voice_manager=self._voice_manager,
            )

        with Horizontal(id="bottom-bar"):
            yield PathDisplay(self.config.displayed_workdir or Path.cwd())
            yield NoMarkupStatic(id="spacer")
            yield ContextProgress()

    async def on_mount(self) -> None:
        self.theme = "textual-ansi"
        self._terminal_notifier.restore()

        self._cached_messages_area = self.query_one("#messages")
        self._cached_chat = self.query_one("#chat", ChatScroll)
        self._cached_loading_area = self.query_one("#loading-area-content")
        self._feedback_bar = self.query_one(FeedbackBar)

        self.event_handler = EventHandler(
            mount_callback=self._mount_and_scroll,
            get_tools_collapsed=lambda: self._tools_collapsed,
            on_profile_changed=self._on_profile_changed,
        )

        self._chat_input_container = self.query_one(ChatInputContainer)
        context_progress = self.query_one(ContextProgress)

        def update_context_progress(stats: AgentStats) -> None:
            detected = self.agent_loop.detected_model_display_name()
            if detected and detected.lower().endswith(".gguf"):
                detected = detected[:-5]
            context_progress.tokens = TokenState(
                max_tokens=self.config.get_active_model().auto_compact_threshold,
                current_tokens=stats.context_tokens,
                model_name=detected,
                tokens_per_second=stats.tokens_per_second,
                prompt_tokens_per_second=stats.prompt_tokens_per_second,
            )

        self.agent_loop.stats.add_listener("context_tokens", update_context_progress)
        self.agent_loop.stats.trigger_listeners()

        self.agent_loop.set_approval_callback(self._approval_callback)
        self.agent_loop.set_user_input_callback(self._user_input_callback)
        self._refresh_profile_widgets()

        chat_input_container = self.query_one(ChatInputContainer)
        chat_input_container.focus_input()
        # History resume must run before any helper that mounts into #messages —
        # _resume_history_from_messages early-exits if #messages already has
        # children, so a warning/info widget mounted first would silently drop
        # the resumed conversation when --continue / --resume <id> is used.
        await self._resume_history_from_messages()
        await self._show_dangerous_directory_warning()
        await self._show_clipboard_warning()
        await self._show_instruction_files_info()
        ctx_msg = await self.agent_loop.resolve_context_size()
        if ctx_msg:
            await self._mount_and_scroll(WarningMessage(ctx_msg, show_border=False))
        # Detection above may have set the cosmetic model name; refresh the
        # context bar so it shows immediately rather than only after turn one.
        self.agent_loop.stats.trigger_listeners()
        # If bootstrap appended new commented-out config stubs, tell the user
        # about it once. The mount happens on every launch, but
        # pop_pending_message returns None except on the launch where the
        # appending actually happened.
        from privibe.core.config.migration import pop_pending_message
        migration_msg = pop_pending_message()
        if migration_msg:
            await self._mount_and_scroll(
                WarningMessage(migration_msg, show_border=False)
            )
        self.agent_loop.start_preflight_warmup_if_enabled()

        self.call_after_refresh(self._refresh_banner)
        self._show_skill_load_errors()
        self._show_model_fallback_warning()

        if self._show_resume_picker:
            self.run_worker(self._show_session_picker(), exclusive=False)
        elif self._initial_prompt:
            self.call_after_refresh(self._process_initial_prompt)

        gc.collect()
        gc.freeze()

    def _process_initial_prompt(self) -> None:
        if self._initial_prompt:
            self.run_worker(
                self._handle_user_message(self._initial_prompt), exclusive=False
            )

    def _is_file_watcher_enabled(self) -> bool:
        return self.config.file_watcher_for_autocomplete

    async def on_chat_input_container_submitted(
        self, event: ChatInputContainer.Submitted
    ) -> None:
        if self._banner:
            self._banner.freeze_animation()

        value = event.value.strip()
        if not value:
            return

        input_widget = self.query_one(ChatInputContainer)
        input_widget.value = ""

        # Handle bash commands (!) and UI slash commands (/autocopy, /rewind, etc.)
        # immediately, regardless of whether the agent is running. These are local
        # operations that have nothing to do with the model conversation.
        if value.startswith("!"):
            await self._handle_bash_command(value[1:])
            return

        if await self._handle_command(value):
            return

        # agent steering — When the agent is running, instead of cancelling
        # (which breaks the LLM/harness cycle and forces full reprocess),
        # queue the user's message as a steering instruction. It will be
        # injected into the next tool result. Show the user an info message
        # explaining that the message is queued and that Esc/Ctrl+C still
        # cancels immediately.
        if self._agent_running:
            self.agent_loop.queue_steering(value)
            await self._mount_and_scroll(
                UserCommandMessage(
                    f"Message queued — will steer the conversation at the next tool call. "
                    f"Press Esc or Ctrl+C to cancel immediately."
                )
            )
            return

        if await self._handle_skill(value):
            return

        await self._handle_user_message(value)

    async def on_approval_app_approval_granted(
        self, message: ApprovalApp.ApprovalGranted
    ) -> None:
        if self._pending_approval and not self._pending_approval.done():
            self._pending_approval.set_result((ApprovalResponse.YES, None))

    async def on_approval_app_approval_granted_always_tool(
        self, message: ApprovalApp.ApprovalGrantedAlwaysTool
    ) -> None:
        self.agent_loop.approve_always(message.tool_name, message.required_permissions)

        if self._pending_approval and not self._pending_approval.done():
            self._pending_approval.set_result((ApprovalResponse.YES, None))

    async def on_approval_app_approval_rejected(
        self, message: ApprovalApp.ApprovalRejected
    ) -> None:
        if self._pending_approval and not self._pending_approval.done():
            feedback = str(
                get_user_cancellation_message(CancellationReason.OPERATION_CANCELLED)
            )
            self._pending_approval.set_result((ApprovalResponse.NO, feedback))

        if self._loading_widget and self._loading_widget.parent:
            await self._remove_loading_widget()

    async def on_question_app_answered(self, message: QuestionApp.Answered) -> None:
        if self._pending_question and not self._pending_question.done():
            result = AskUserQuestionResult(answers=message.answers, cancelled=False)
            self._pending_question.set_result(result)

    async def on_question_app_cancelled(self, message: QuestionApp.Cancelled) -> None:

        if self._pending_question and not self._pending_question.done():
            result = AskUserQuestionResult(answers=[], cancelled=True)
            self._pending_question.set_result(result)

    def on_chat_text_area_feedback_key_pressed(
        self, message: ChatTextArea.FeedbackKeyPressed
    ) -> None:
        self._feedback_bar.handle_feedback_key(message.rating)

    def on_chat_text_area_non_feedback_key_pressed(
        self, message: ChatTextArea.NonFeedbackKeyPressed
    ) -> None:
        self._feedback_bar.hide()

    def on_feedback_bar_feedback_given(
        self, message: FeedbackBar.FeedbackGiven
    ) -> None:
        pass

    async def _remove_loading_widget(self) -> None:
        if self._loading_widget and self._loading_widget.parent:
            await self._loading_widget.remove()
            self._loading_widget = None

    async def on_config_app_open_model_picker(
        self, _message: ConfigApp.OpenModelPicker
    ) -> None:
        config_app = self.query_one(ConfigApp)
        changes = config_app._convert_changes_for_save()
        if changes:
            VibeConfig.save_updates(changes)
            await self._reload_config()
        await self._switch_to_input_app()
        await self._switch_to_model_picker_app()

    async def _ensure_loading_widget(self, status: str = "Generating") -> None:
        if self._loading_widget and self._loading_widget.parent:
            self._loading_widget.set_status(status)
            return

        loading_area = self._cached_loading_area
        if loading_area is None:
            try:
                loading_area = self.query_one("#loading-area-content")
            except Exception:
                return
        loading = LoadingWidget(status=status)
        self._loading_widget = loading
        await loading_area.mount(loading)

    async def on_config_app_config_closed(
        self, message: ConfigApp.ConfigClosed
    ) -> None:
        await self._handle_config_settings_closed(message.changes)
        await self._switch_to_input_app()

    async def on_voice_app_config_closed(self, message: VoiceApp.ConfigClosed) -> None:
        await self._handle_voice_settings_closed(message.changes)
        await self._switch_to_input_app()

    async def _handle_config_settings_closed(
        self, changes: dict[str, str | bool]
    ) -> None:
        if changes:
            VibeConfig.save_updates(changes)
            await self._reload_config()
        else:
            await self._mount_and_scroll(
                UserCommandMessage("Configuration closed (no changes saved).")
            )

    async def _handle_voice_settings_closed(
        self, changes: dict[str, str | bool]
    ) -> None:
        if not changes:
            await self._mount_and_scroll(
                UserCommandMessage("Voice settings closed (no changes saved).")
            )
            return

        if "voice_mode_enabled" in changes:
            current = self._voice_manager.is_enabled
            desired = changes["voice_mode_enabled"]
            if current != desired:
                self._voice_manager.toggle_voice_mode()
                self.agent_loop.refresh_config()
                if desired:
                    await self._mount_and_scroll(
                        UserCommandMessage(
                            "Voice mode enabled. Press ctrl+r to start recording."
                        )
                    )
                else:
                    await self._mount_and_scroll(
                        UserCommandMessage("Voice mode disabled.")
                    )

        non_voice_changes = {
            k: v for k, v in changes.items() if k != "voice_mode_enabled"
        }
        if non_voice_changes:
            VibeConfig.save_updates(non_voice_changes)
            self.agent_loop.refresh_config()
            self._narrator_manager.sync()

    async def on_model_picker_app_model_selected(
        self, message: ModelPickerApp.ModelSelected
    ) -> None:
        VibeConfig.save_updates({"active_model": message.alias})
        await self._reload_config()
        # Switching models changes the active alias, so resolve_context_size
        # re-detects (per-alias latches); refresh the banner afterwards so the
        # detected name for the new model is reflected.
        ctx_msg = await self.agent_loop.resolve_context_size()
        if ctx_msg:
            await self._mount_and_scroll(WarningMessage(ctx_msg, show_border=False))
        self._refresh_banner()
        await self._switch_to_input_app()

    async def on_model_picker_app_cancelled(
        self, _event: ModelPickerApp.Cancelled
    ) -> None:
        await self._switch_to_input_app()

    async def on_proxy_setup_app_proxy_setup_closed(
        self, message: ProxySetupApp.ProxySetupClosed
    ) -> None:
        if message.error:
            await self._mount_and_scroll(
                ErrorMessage(f"Failed to save proxy settings: {message.error}")
            )
        elif message.saved:
            await self._mount_and_scroll(
                UserCommandMessage(
                    "Proxy settings saved. Restart the CLI for changes to take effect."
                )
            )
        else:
            await self._mount_and_scroll(UserCommandMessage("Proxy setup cancelled."))

        await self._switch_to_input_app()

    async def on_compact_message_completed(
        self, message: CompactMessage.Completed
    ) -> None:
        messages_area = self._cached_messages_area or self.query_one("#messages")
        children = list(messages_area.children)

        try:
            compact_index = children.index(message.compact_widget)
        except ValueError:
            return

        if compact_index == 0:
            return

        with self.batch_update():
            for widget in children[:compact_index]:
                await widget.remove()

    async def _handle_command(self, user_input: str) -> bool:
        if command := self.commands.find_command(user_input):
            await self._mount_and_scroll(UserMessage(user_input))
            handler = getattr(self, command.handler)
            if asyncio.iscoroutinefunction(handler):
                await handler()
            else:
                handler()
            return True
        return False

    def _get_skill_entries(self) -> list[tuple[str, str]]:
        if not self.agent_loop:
            return []
        return [
            (f"/{name}", info.description)
            for name, info in self.agent_loop.skill_manager.available_skills.items()
            if info.user_invocable
        ]

    async def _handle_skill(self, user_input: str) -> bool:
        if not user_input.startswith("/"):
            return False

        if not self.agent_loop:
            return False

        parts = user_input[1:].strip().split(None, 1)
        if not parts:
            return False
        skill_name = parts[0].lower()

        skill_info = self.agent_loop.skill_manager.get_skill(skill_name)
        if not skill_info:
            return False

        try:
            skill_content = read_safe(skill_info.skill_path)
        except OSError as e:
            await self._mount_and_scroll(
                ErrorMessage(
                    f"Failed to read skill file: {e}", collapsed=self._tools_collapsed
                )
            )
            return True

        if len(parts) > 1:
            skill_content = f"{user_input}\n\n{skill_content}"

        await self._handle_user_message(skill_content)
        return True

    async def _handle_bash_command(self, command: str) -> None:
        if not command:
            await self._mount_and_scroll(
                ErrorMessage(
                    "No command provided after '!'", collapsed=self._tools_collapsed
                )
            )
            return

        proc = None
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=30
                )
            except TimeoutError:
                if proc.returncode is None:
                    proc.kill()
                    await proc.wait()
                await self._mount_and_scroll(
                    ErrorMessage(
                        "Command timed out after 30 seconds",
                        collapsed=self._tools_collapsed,
                    )
                )
                return

            stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
            stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
            output = stdout or stderr or "(no output)"
            exit_code = proc.returncode or 0
            await self._mount_and_scroll(
                BashOutputMessage(command, str(Path.cwd()), output, exit_code)
            )
        except asyncio.CancelledError:
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()
            raise
        except Exception as e:
            await self._mount_and_scroll(
                ErrorMessage(f"Command failed: {e}", collapsed=self._tools_collapsed)
            )

    async def _handle_user_message(self, message: str) -> None:
        # message_index is where the user message will land in agent_loop.messages
        # (checkpoint is created in agent_loop.act())
        message_index = len(self.agent_loop.messages)
        user_message = UserMessage(message, message_index=message_index)

        await self._mount_and_scroll(user_message)

        if not self._agent_running:
            await self._remove_loading_widget()
            self._agent_task = asyncio.create_task(
                self._handle_agent_loop_turn(message)
            )

    def _reset_ui_state(self) -> None:
        self._windowing.reset()
        self._tool_call_map = None
        self._history_widget_indices = WeakKeyDictionary()

    async def _resume_history_from_messages(self) -> None:
        messages_area = self._cached_messages_area or self.query_one("#messages")
        if not should_resume_history(list(messages_area.children)):
            return

        history_messages = non_system_history_messages(self.agent_loop.messages)
        if (
            plan := create_resume_plan(history_messages, HISTORY_RESUME_TAIL_MESSAGES)
        ) is None:
            return
        await self._mount_history_batch(
            plan.tail_messages,
            messages_area,
            plan.tool_call_map,
            start_index=plan.tail_start_index,
        )
        chat = self._cached_chat or self.query_one("#chat", ChatScroll)
        self.call_after_refresh(chat.anchor)
        self._tool_call_map = plan.tool_call_map
        self._windowing.set_backfill(plan.backfill_messages)
        await self._load_more.set_visible(
            messages_area,
            visible=self._windowing.has_backfill,
            remaining=self._windowing.remaining,
        )

    async def _mount_history_batch(
        self,
        batch: list[LLMMessage],
        messages_area: Widget,
        tool_call_map: dict[str, str],
        *,
        start_index: int,
        before: Widget | int | None = None,
        after: Widget | None = None,
    ) -> None:
        widgets = build_history_widgets(
            batch=batch,
            tool_call_map=tool_call_map,
            start_index=start_index,
            tools_collapsed=self._tools_collapsed,
            history_widget_indices=self._history_widget_indices,
        )

        with self.batch_update():
            if not widgets:
                return
            if before is not None:
                await messages_area.mount_all(widgets, before=before)
                return
            if after is not None:
                await messages_area.mount_all(widgets, after=after)
                return
            await messages_area.mount_all(widgets)

    def _is_tool_enabled_in_main_agent(self, tool: str) -> bool:
        return tool in self.agent_loop.tool_manager.available_tools

    async def _approval_callback(
        self,
        tool: str,
        args: BaseModel,
        tool_call_id: str,
        required_permissions: list[RequiredPermission] | None,
    ) -> tuple[ApprovalResponse, str | None]:
        # Auto-approve only if parent is in auto-approve mode AND tool is enabled
        # This ensures subagents respect the main agent's tool restrictions
        if self.agent_loop and self.agent_loop.config.auto_approve:
            if self._is_tool_enabled_in_main_agent(tool):
                return (ApprovalResponse.YES, None)

        async with self._user_interaction_lock:
            self._pending_approval = asyncio.Future()
            self._terminal_notifier.notify(NotificationContext.ACTION_REQUIRED)
            try:
                with paused_timer(self._loading_widget):
                    await self._switch_to_approval_app(tool, args, required_permissions)
                    result = await self._pending_approval
                return result
            finally:
                self._pending_approval = None
                await self._switch_to_input_app()

    async def _user_input_callback(self, args: BaseModel) -> BaseModel:
        question_args = cast(AskUserQuestionArgs, args)

        async with self._user_interaction_lock:
            self._pending_question = asyncio.Future()
            self._terminal_notifier.notify(NotificationContext.ACTION_REQUIRED)
            try:
                with paused_timer(self._loading_widget):
                    await self._switch_to_question_app(question_args)
                    result = await self._pending_question
                return result
            finally:
                self._pending_question = None
                await self._switch_to_input_app()

    async def _handle_turn_error(self) -> None:
        if self._loading_widget and self._loading_widget.parent:
            await self._loading_widget.remove()
        if self.event_handler:
            self.event_handler.stop_current_tool_call(success=False)

    async def _handle_agent_loop_turn(self, prompt: str) -> None:
        self._agent_running = True

        await self._remove_loading_widget()
        await self._ensure_loading_widget()

        try:
            rendered_prompt = render_path_prompt(prompt, base_dir=Path.cwd())
            self._narrator_manager.cancel()
            self._narrator_manager.on_turn_start(rendered_prompt)
            async for event in self.agent_loop.act(rendered_prompt):
                self._narrator_manager.on_turn_event(event)
                if isinstance(event, WaitingForInputEvent):
                    await self._remove_loading_widget()
                elif self._loading_widget is None and not isinstance(event, WaitingForInputEvent):
                    await self._ensure_loading_widget()
                if self.event_handler:
                    await self.event_handler.handle_event(
                        event,
                        loading_active=self._loading_widget is not None,
                        loading_widget=self._loading_widget,
                    )


        except asyncio.CancelledError:
            await self._handle_turn_error()
            self._narrator_manager.on_turn_cancel()
            raise
        except Exception as e:
            await self._handle_turn_error()

            message = str(e)
            if isinstance(e, RateLimitError):
                message = self._rate_limit_message()
            self._narrator_manager.on_turn_error(message)

            await self._mount_and_scroll(
                ErrorMessage(message, collapsed=self._tools_collapsed)
            )
        finally:
            self._narrator_manager.on_turn_end()
            self._agent_running = False
            self._interrupt_requested = False
            self._agent_task = None
            if self._loading_widget:
                await self._loading_widget.remove()
            self._loading_widget = None
            if self.event_handler:
                await self.event_handler.finalize_streaming()
            await self._refresh_windowing_from_history()
            self._terminal_notifier.notify(NotificationContext.COMPLETE)

        # agent steering — After the agent loop finishes (normal exit or error),
        # check for queued steering messages via agent_loop.drain_steering_queue().
        # If non-empty, submit each as a new user message through
        # _handle_user_message() — the exact same path as a normal user message.
        # This handles the case where the model finished with no tool calls but
        # the user had typed a steering message that couldn't be injected.
        # Must run AFTER the finally block so _agent_running is False.
        for msg in self.agent_loop.drain_steering_queue():
            await self._handle_user_message(msg)

    def _rate_limit_message(self) -> str:
        return "Rate limits exceeded. Please wait a moment before trying again."

    async def _interrupt_agent_loop(self) -> None:
        if not self._agent_running or self._interrupt_requested:
            return

        self._interrupt_requested = True

        if self._pending_approval and not self._pending_approval.done():
            feedback = str(
                get_user_cancellation_message(CancellationReason.TOOL_INTERRUPTED)
            )
            self._pending_approval.set_result((ApprovalResponse.NO, feedback))
        if self._pending_question and not self._pending_question.done():
            self._pending_question.set_result(
                AskUserQuestionResult(answers=[], cancelled=True)
            )

        if self._agent_task and not self._agent_task.done():
            self._agent_task.cancel()
            try:
                await self._agent_task
            except asyncio.CancelledError:
                pass

        if self.event_handler:
            self.event_handler.stop_current_tool_call(success=False)
            self.event_handler.stop_current_compact()
            await self.event_handler.finalize_streaming()

        self._agent_running = False
        loading_area = self._cached_loading_area or self.query_one(
            "#loading-area-content"
        )
        await loading_area.remove_children()
        self._loading_widget = None

        await self._mount_and_scroll(InterruptMessage())

        self._interrupt_requested = False

    async def _show_help(self) -> None:
        help_text = self.commands.get_help_text()
        await self._mount_and_scroll(UserCommandMessage(help_text))

    async def _show_status(self) -> None:
        stats = self.agent_loop.stats
        status_text = f"""## Agent Statistics

- **Steps**: {stats.steps:,}
- **Session Prompt Tokens**: {stats.session_prompt_tokens:,}
- **Session Completion Tokens**: {stats.session_completion_tokens:,}
- **Session Total LLM Tokens**: {stats.session_total_llm_tokens:,}
- **Last Turn Tokens**: {stats.last_turn_total_tokens:,}
- **Cost**: ${stats.session_cost:.4f}
"""
        await self._mount_and_scroll(UserCommandMessage(status_text))

    async def _show_config(self) -> None:
        """Switch to the configuration app in the bottom panel."""
        if self._current_bottom_app == BottomApp.Config:
            return
        await self._switch_to_config_app()

    async def _show_model(self) -> None:
        """Switch to the model picker in the bottom panel."""
        if self._current_bottom_app == BottomApp.ModelPicker:
            return
        await self._switch_to_model_picker_app()

    async def _show_proxy_setup(self) -> None:
        if self._current_bottom_app == BottomApp.ProxySetup:
            return
        await self._switch_to_proxy_setup_app()

    async def _show_session_picker(self) -> None:
        cwd = str(Path.cwd())
        sessions = (
            list_local_resume_sessions(self.config)
            if self.config.session_logging.enabled
            else []
        )

        if not sessions:
            await self._mount_and_scroll(
                UserCommandMessage("No sessions found.")
            )
            return

        sessions = sorted(
            sessions,
            key=lambda s: (s.cwd == cwd, s.end_time or ""),
            reverse=True,
        )

        latest_messages: dict[str, list[tuple[str, str]]] = {
            s.option_id: SessionLoader.get_last_messages(
                s.session_id, self.config.session_logging, n=self.config.session_logging.resume_preview_lines
            )
            for s in sessions
        }

        picker = SessionPickerApp(sessions=sessions, latest_messages=latest_messages)
        await self._switch_from_input(picker)

    async def on_session_picker_app_session_selected(
        self, event: SessionPickerApp.SessionSelected
    ) -> None:
        await self._switch_to_input_app()
        session = ResumeSessionInfo(
            session_id=event.session_id,
            source="local",
            cwd="",
            title=None,
            end_time=None,
        )
        try:
            await self._resume_local_session(session)
        except Exception as e:
            await self._mount_and_scroll(
                ErrorMessage(
                    f"Failed to load session: {e}", collapsed=self._tools_collapsed
                )
            )

    async def on_session_picker_app_cancelled(
        self, event: SessionPickerApp.Cancelled
    ) -> None:
        await self._switch_to_input_app()

        await self._mount_and_scroll(UserCommandMessage("Resume cancelled."))

    async def _resume_local_session(self, session: ResumeSessionInfo) -> None:
        session_config = self.config.session_logging
        session_path = SessionLoader.find_session_by_id(
            session.session_id, session_config
        )

        if not session_path:
            raise ValueError(
                f"Session `{short_session_id(session.session_id)}` not found."
            )

        if self._chat_input_container:
            self._chat_input_container.set_custom_border(None)

        self.agent_loop.session_id = session.session_id
        self.agent_loop.session_logger.resume_existing_session(
            session.session_id, session_path
        )
        self.agent_loop.messages.restore(session_path)
        self._refresh_profile_widgets()

        self._reset_ui_state()
        await self._load_more.hide()

        messages_area = self._cached_messages_area or self.query_one("#messages")
        await messages_area.remove_children()

        await self._resume_history_from_messages()
        await self._mount_and_scroll(
            UserCommandMessage(
                f"Resumed session `{short_session_id(session.session_id)}`"
            )
        )


    async def remove_loading(self) -> None:
        await self._remove_loading_widget()

    async def ensure_loading(self, status: str = "Generating") -> None:
        await self._ensure_loading_widget(status)

    @property
    def loading_widget(self) -> LoadingWidget | None:
        return self._loading_widget

    async def _reload_config(self) -> None:
        try:
            self._reset_ui_state()
            await self._load_more.hide()
            base_config = VibeConfig.load()

            await self.agent_loop.reload_with_initial_messages(base_config=base_config)
            self._narrator_manager.sync()

            if self._banner:
                self._banner.set_state(
                    base_config,
                    self.agent_loop.skill_manager,
                    self.agent_loop.mcp_registry,
                    None,
                    active_model_label=self._active_model_label(),
                )
            await self._mount_and_scroll(UserCommandMessage("Configuration reloaded."))
        except Exception as e:
            await self._mount_and_scroll(
                ErrorMessage(
                    f"Failed to reload config: {e}", collapsed=self._tools_collapsed
                )
            )

    async def _clear_history(self) -> None:
        try:
            self._reset_ui_state()
            if self._chat_input_container:
                self._chat_input_container.set_custom_border(None)
            await self.agent_loop.clear_history()
            if self.event_handler:
                await self.event_handler.finalize_streaming()
            messages_area = self._cached_messages_area or self.query_one("#messages")
            await messages_area.remove_children()

            await messages_area.mount(UserMessage("/clear"))
            await self._mount_and_scroll(
                UserCommandMessage("Conversation history cleared!")
            )
            chat = self._cached_chat or self.query_one("#chat", ChatScroll)
            chat.scroll_home(animate=False)

        except Exception as e:
            await self._mount_and_scroll(
                ErrorMessage(
                    f"Failed to clear history: {e}", collapsed=self._tools_collapsed
                )
            )

    async def _show_log_path(self) -> None:
        if not self.agent_loop.session_logger.enabled:
            await self._mount_and_scroll(
                ErrorMessage(
                    "Session logging is disabled in configuration.",
                    collapsed=self._tools_collapsed,
                )
            )
            return

        try:
            log_path = str(self.agent_loop.session_logger.session_dir)
            await self._mount_and_scroll(
                UserCommandMessage(
                    f"## Current Log Directory\n\n`{log_path}`\n\nYou can send this directory to share your interaction."
                )
            )
        except Exception as e:
            await self._mount_and_scroll(
                ErrorMessage(
                    f"Failed to get log path: {e}", collapsed=self._tools_collapsed
                )
            )

    async def _compact_history(self) -> None:
        if self._agent_running:
            await self._mount_and_scroll(
                ErrorMessage(
                    "Cannot compact while agent loop is processing. Please wait.",
                    collapsed=self._tools_collapsed,
                )
            )
            return

        if len(self.agent_loop.messages) <= 1:
            await self._mount_and_scroll(
                ErrorMessage(
                    "No conversation history to compact yet.",
                    collapsed=self._tools_collapsed,
                )
            )
            return

        if not self.event_handler:
            return

        old_tokens = self.agent_loop.stats.context_tokens
        compact_msg = CompactMessage()
        self.event_handler.current_compact = compact_msg
        await self._mount_and_scroll(compact_msg)

        self._agent_task = asyncio.create_task(
            self._run_compact(compact_msg, old_tokens)
        )

    async def _run_compact(self, compact_msg: CompactMessage, old_tokens: int) -> None:
        self._agent_running = True
        try:
            await self.agent_loop.compact()
            new_tokens = self.agent_loop.stats.context_tokens
            compact_msg.set_complete(old_tokens=old_tokens, new_tokens=new_tokens)

        except asyncio.CancelledError:
            compact_msg.set_error("Compaction interrupted")
            raise
        except Exception as e:
            compact_msg.set_error(str(e))
        finally:
            self._agent_running = False
            self._agent_task = None
            if self.event_handler:
                self.event_handler.current_compact = None

    def _get_session_resume_info(self) -> str | None:
        if not self.agent_loop.session_logger.enabled:
            return None
        if not self.agent_loop.session_logger.session_id:
            return None
        session_config = self.agent_loop.session_logger.session_config
        session_path = SessionLoader.does_session_exist(
            self.agent_loop.session_logger.session_id, session_config
        )
        if session_path is None:
            return None
        return short_session_id(self.agent_loop.session_logger.session_id)

    async def _exit_app(self) -> None:
        await self._narrator_manager.close()
        self.exit(result=self._get_session_resume_info())

    async def _setup_terminal(self) -> None:
        result = setup_terminal()

        if result.success:
            if result.requires_restart:
                message = f"{result.message or 'Set up Shift+Enter keybind'} (You may need to restart your terminal.)"
                await self._mount_and_scroll(
                    UserCommandMessage(f"{result.terminal.value}: {message}")
                )
            else:
                message = result.message or "Shift+Enter keybind already set up"
                await self._mount_and_scroll(
                    WarningMessage(f"{result.terminal.value}: {message}")
                )
        else:
            await self._mount_and_scroll(
                ErrorMessage(result.message, collapsed=self._tools_collapsed)
            )

    def _make_default_voice_manager(self) -> VoiceManager:
        try:
            model = self.config.get_active_transcribe_model()
            provider = self.config.get_transcribe_provider_for_model(model)
            transcribe_client = make_transcribe_client(provider, model)
        except (ValueError, KeyError) as exc:
            logger.error(
                "Failed to initialize transcription, check transcribe model configuration",
                exc_info=exc,
            )
            transcribe_client = None

        return VoiceManager(
            lambda: self.config,
            audio_recorder=AudioRecorder(),
            transcribe_client=transcribe_client,
        )

    async def _show_voice_settings(self) -> None:
        if self._current_bottom_app == BottomApp.Voice:
            return
        await self._switch_to_voice_app()

    async def _switch_from_input(self, widget: Widget, scroll: bool = False) -> None:
        bottom_container = self.query_one("#bottom-app-container")
        chat = self._cached_chat or self.query_one("#chat", ChatScroll)
        should_scroll = scroll and chat.is_at_bottom

        if self._chat_input_container:
            self._chat_input_container.display = False
            self._chat_input_container.disabled = True

        self._feedback_bar.hide()

        self._current_bottom_app = BottomApp[type(widget).__name__.removesuffix("App")]
        await bottom_container.mount(widget)

        self.call_after_refresh(widget.focus)
        if should_scroll:
            self.call_after_refresh(chat.anchor)

    async def _switch_to_config_app(self) -> None:
        if self._current_bottom_app == BottomApp.Config:
            return

        await self._mount_and_scroll(UserCommandMessage("Configuration opened..."))
        await self._switch_from_input(ConfigApp(self.config))

    async def _switch_to_voice_app(self) -> None:
        if self._current_bottom_app == BottomApp.Voice:
            return

        await self._mount_and_scroll(UserCommandMessage("Voice settings opened..."))
        await self._switch_from_input(VoiceApp(self.config))

    async def _switch_to_model_picker_app(self) -> None:
        if self._current_bottom_app == BottomApp.ModelPicker:
            return

        model_aliases = [m.alias for m in self.config.models]
        current_model = str(self.config.active_model)
        await self._switch_from_input(
            ModelPickerApp(model_aliases=model_aliases, current_model=current_model)
        )

    async def _switch_to_proxy_setup_app(self) -> None:
        if self._current_bottom_app == BottomApp.ProxySetup:
            return

        await self._mount_and_scroll(UserCommandMessage("Proxy setup opened..."))
        await self._switch_from_input(ProxySetupApp())

    async def _switch_to_approval_app(
        self,
        tool_name: str,
        tool_args: BaseModel,
        required_permissions: list[RequiredPermission] | None = None,
    ) -> None:
        approval_app = ApprovalApp(
            tool_name=tool_name,
            tool_args=tool_args,
            config=self.config,
            required_permissions=required_permissions,
        )
        await self._switch_from_input(approval_app, scroll=True)

    async def _switch_to_question_app(self, args: AskUserQuestionArgs) -> None:
        await self._switch_from_input(QuestionApp(args=args), scroll=True)

    async def _switch_to_input_app(self) -> None:
        if self._chat_input_container:
            self._chat_input_container.disabled = False
            self._chat_input_container.display = True
            self._current_bottom_app = BottomApp.Input
            self._refresh_profile_widgets()

        for app in BottomApp:
            if app != BottomApp.Input:
                try:
                    await self.query_one(f"#{app.value}-app").remove()
                except Exception:
                    pass

        if self._chat_input_container:
            self.call_after_refresh(self._chat_input_container.focus_input)
            chat = self._cached_chat or self.query_one("#chat", ChatScroll)
            if chat.is_at_bottom:
                self.call_after_refresh(chat.anchor)

    def _focus_current_bottom_app(self) -> None:
        try:
            match self._current_bottom_app:
                case BottomApp.Input:
                    self.query_one(ChatInputContainer).focus_input()
                case BottomApp.Config:
                    self.query_one(ConfigApp).focus()
                case BottomApp.ModelPicker:
                    self.query_one(ModelPickerApp).focus()
                case BottomApp.ProxySetup:
                    self.query_one(ProxySetupApp).focus()
                case BottomApp.Approval:
                    self.query_one(ApprovalApp).focus()
                case BottomApp.Question:
                    self.query_one(QuestionApp).focus()
                case BottomApp.SessionPicker:
                    self.query_one(SessionPickerApp).focus()
                case BottomApp.Rewind:
                    self.query_one(RewindApp).focus()
                case BottomApp.Voice:
                    self.query_one(VoiceApp).focus()
                case app:
                    assert_never(app)
        except Exception:
            pass

    def _handle_config_app_escape(self) -> None:
        try:
            config_app = self.query_one(ConfigApp)
            config_app.action_close()
        except Exception:
            pass
        self._last_escape_time = None

    def _handle_voice_app_escape(self) -> None:
        try:
            voice_app = self.query_one(VoiceApp)
            voice_app.action_close()
        except Exception:
            pass
        self._last_escape_time = None

    def _handle_approval_app_escape(self) -> None:
        try:
            approval_app = self.query_one(ApprovalApp)
            approval_app.action_reject()
        except Exception:
            pass
        self._last_escape_time = None

    def _handle_question_app_escape(self) -> None:
        try:
            question_app = self.query_one(QuestionApp)
            question_app.action_cancel()
        except Exception:
            pass
        self._last_escape_time = None

    def _handle_model_picker_app_escape(self) -> None:
        try:
            model_picker = self.query_one(ModelPickerApp)
            model_picker.post_message(ModelPickerApp.Cancelled())
        except Exception:
            pass
        self._last_escape_time = None

    def _handle_session_picker_app_escape(self) -> None:
        try:
            session_picker = self.query_one(SessionPickerApp)
            session_picker.post_message(SessionPickerApp.Cancelled())
        except Exception:
            pass
        self._last_escape_time = None

    # --- Rewind mode ---

    def _get_user_message_widgets(self) -> list[UserMessage]:
        """Return all UserMessage widgets currently visible in #messages.

        Only includes messages with a valid message_index (i.e. real user
        messages, not slash-command echo messages).
        """
        messages_area = self._cached_messages_area or self.query_one("#messages")
        return [
            child
            for child in messages_area.children
            if isinstance(child, UserMessage) and child.message_index is not None
        ]

    def _start_rewind_mode(self) -> None:
        self.action_rewind_prev()

    def action_rewind_prev(self) -> None:
        if self._agent_running:
            return

        user_widgets = self._get_user_message_widgets()
        if not user_widgets:
            return

        if not self._rewind_mode:
            self._rewind_mode = True
            target = user_widgets[-1]
        elif self._rewind_highlighted_widget is not None:
            try:
                idx = user_widgets.index(self._rewind_highlighted_widget)
            except ValueError:
                idx = len(user_widgets)
            if idx <= 0:
                self.run_worker(self._rewind_prev_at_top(), exclusive=False)
                return
            target = user_widgets[idx - 1]
        else:
            target = user_widgets[-1]

        self.run_worker(self._select_rewind_widget(target), exclusive=False)

    async def _rewind_prev_at_top(self) -> None:
        """Handle alt+up at the topmost mounted user message.

        The previous user message may be windowed out into the backfill, and a
        single load-more batch can contain no user message at all (e.g. a
        tool-heavy turn). Keep loading more until a user message is mounted
        above the current one, or the backfill is exhausted.
        """
        while self._load_more.widget is not None and self._windowing.has_backfill:
            remaining_before = self._windowing.remaining
            await self.on_history_load_more_requested(HistoryLoadMoreRequested())
            if self._rewind_highlighted_widget is None:
                break
            user_widgets = self._get_user_message_widgets()
            try:
                idx = user_widgets.index(self._rewind_highlighted_widget)
            except ValueError:
                idx = 0
            if idx > 0:
                await self._select_rewind_widget(user_widgets[idx - 1])
                return
            # Defensive: if a load did not actually consume backfill, stop
            # rather than spin forever.
            if self._windowing.remaining >= remaining_before:
                break
        # Backfill exhausted (or none): already at the first message, scroll up.
        chat = self._cached_chat or self.query_one("#chat", ChatScroll)
        self.call_after_refresh(chat.scroll_home, animate=False)

    def action_rewind_next(self) -> None:
        if not self._rewind_mode:
            return

        if self._rewind_highlighted_widget is None:
            return

        user_widgets = self._get_user_message_widgets()
        try:
            idx = user_widgets.index(self._rewind_highlighted_widget)
        except ValueError:
            return
        if idx >= len(user_widgets) - 1:
            return

        self.run_worker(
            self._select_rewind_widget(user_widgets[idx + 1]), exclusive=False
        )

    async def _select_rewind_widget(self, widget: UserMessage) -> None:
        """Highlight the given user message widget and show the rewind panel."""
        if self._rewind_highlighted_widget is not None:
            self._rewind_highlighted_widget.remove_class("rewind-selected")

        widget.add_class("rewind-selected")
        self._rewind_highlighted_widget = widget

        msg_index = widget.message_index
        has_file_changes = (
            msg_index is not None
            and self.agent_loop.rewind_manager.has_file_changes_at(msg_index)
        )

        await self._switch_to_rewind_app(
            widget.get_content(), has_file_changes=has_file_changes
        )

        chat = self._cached_chat or self.query_one("#chat", ChatScroll)
        self.call_after_refresh(chat.scroll_to_widget, widget, animate=False, top=True)

    async def _switch_to_rewind_app(
        self, message_preview: str, *, has_file_changes: bool
    ) -> None:
        """Show the rewind action panel at the bottom."""
        if self._current_bottom_app == BottomApp.Rewind:
            # Reuse existing widget if the option set hasn't changed
            try:
                existing = self.query_one(RewindApp)
                if existing.has_file_changes == has_file_changes:
                    existing.update_preview(message_preview)
                    return
                await existing.remove()
            except Exception:
                pass

            rewind_app = RewindApp(
                message_preview=message_preview, has_file_changes=has_file_changes
            )
            bottom_container = self.query_one("#bottom-app-container")
            self._current_bottom_app = BottomApp.Rewind
            await bottom_container.mount(rewind_app)
            self.call_after_refresh(rewind_app.focus)
        else:
            rewind_app = RewindApp(
                message_preview=message_preview, has_file_changes=has_file_changes
            )
            await self._switch_from_input(rewind_app)

    def _clear_rewind_state(self) -> None:
        if self._rewind_highlighted_widget is not None:
            self._rewind_highlighted_widget.remove_class("rewind-selected")
            self._rewind_highlighted_widget = None
        self._rewind_mode = False

    async def _exit_rewind_mode(self) -> None:
        """Exit rewind mode and restore the input panel."""
        self._clear_rewind_state()
        await self._switch_to_input_app()

    async def on_rewind_app_rewind_with_restore(
        self, message: RewindApp.RewindWithRestore
    ) -> None:
        await self._execute_rewind(restore_files=True)

    async def on_rewind_app_rewind_without_restore(
        self, message: RewindApp.RewindWithoutRestore
    ) -> None:
        await self._execute_rewind(restore_files=False)

    async def _execute_rewind(self, *, restore_files: bool) -> None:
        """Fork the session at the selected user message."""
        if not self._rewind_mode or self._rewind_highlighted_widget is None:
            return

        target_widget = self._rewind_highlighted_widget
        msg_index = target_widget.message_index

        if msg_index is None:
            return

        try:
            (
                message_content,
                restore_errors,
            ) = await self.agent_loop.rewind_manager.rewind_to_message(
                msg_index, restore_files=restore_files
            )
        except RewindError as exc:
            self.notify(str(exc), severity="error")
            return

        for error in restore_errors:
            self.notify(error, severity="warning")

        # Remove UI widgets from the selected message onward
        messages_area = self._cached_messages_area or self.query_one("#messages")
        children = list(messages_area.children)
        try:
            target_idx = children.index(target_widget)
        except ValueError:
            target_idx = len(children)
        to_remove = children[target_idx:]
        if to_remove:
            await messages_area.remove_children(to_remove)

        self._clear_rewind_state()

        # Switch back to input and pre-fill with the original message
        await self._switch_to_input_app()
        if self._chat_input_container:
            self._chat_input_container.value = message_content

    # --- End rewind mode ---

    def _handle_input_app_escape(self) -> None:
        try:
            input_widget = self.query_one(ChatInputContainer)
            input_widget.value = ""
        except Exception:
            pass
        self._last_escape_time = None

    def _handle_agent_running_escape(self) -> None:
        self.run_worker(self._interrupt_agent_loop(), exclusive=False)

    def action_interrupt(self) -> None:  # noqa: PLR0911
        if self._voice_manager.transcribe_state != TranscribeState.IDLE:
            self._voice_manager.cancel_recording()
            return

        current_time = time.monotonic()

        if self._current_bottom_app == BottomApp.Config:
            self._handle_config_app_escape()
            return

        if self._current_bottom_app == BottomApp.Voice:
            self._handle_voice_app_escape()
            return

        if self._current_bottom_app == BottomApp.ProxySetup:
            try:
                proxy_setup_app = self.query_one(ProxySetupApp)
                proxy_setup_app.action_close()
            except Exception:
                pass
            self._last_escape_time = None
            return

        if self._current_bottom_app == BottomApp.Approval:
            self._handle_approval_app_escape()
            return

        if self._current_bottom_app == BottomApp.Question:
            self._handle_question_app_escape()
            return

        if self._current_bottom_app == BottomApp.ModelPicker:
            self._handle_model_picker_app_escape()
            return

        if self._current_bottom_app == BottomApp.SessionPicker:
            self._handle_session_picker_app_escape()
            return

        if self._current_bottom_app == BottomApp.Rewind:
            self.run_worker(self._exit_rewind_mode(), exclusive=False)
            self._last_escape_time = None
            return

        if (
            self._current_bottom_app == BottomApp.Input
            and self._last_escape_time is not None
            and (current_time - self._last_escape_time) < 0.2  # noqa: PLR2004
        ):
            self._handle_input_app_escape()
            return

        if (
            self._narrator_manager.is_playing
            or self._narrator_manager.state != NarratorState.IDLE
        ):
            self._narrator_manager.cancel()
            return

        if self._agent_running:
            self._handle_agent_running_escape()

        self._last_escape_time = current_time
        chat = self._cached_chat or self.query_one("#chat", ChatScroll)
        if chat.is_at_bottom:
            self.call_after_refresh(chat.anchor)
        self._focus_current_bottom_app()

    async def on_history_load_more_requested(self, _: HistoryLoadMoreRequested) -> None:
        self._load_more.set_enabled(False)
        try:
            if not self._windowing.has_backfill:
                await self._load_more.hide()
                return
            if (batch := self._windowing.next_load_more_batch()) is None:
                await self._load_more.hide()
                return
            messages_area = self._cached_messages_area or self.query_one("#messages")
            if self._tool_call_map is None:
                self._tool_call_map = {}
            if self._load_more.widget:
                before: Widget | int | None = None
                after: Widget | None = self._load_more.widget
            else:
                before = 0
                after = None
            await self._mount_history_batch(
                batch.messages,
                messages_area,
                self._tool_call_map,
                start_index=batch.start_index,
                before=before,
                after=after,
            )
            if not self._windowing.has_backfill:
                await self._load_more.hide()
            else:
                await self._load_more.show(messages_area, self._windowing.remaining)
        finally:
            self._load_more.set_enabled(True)

    async def action_toggle_tool(self) -> None:
        self._tools_collapsed = not self._tools_collapsed

        for result in self.query(ToolResultMessage):
            await result.set_collapsed(self._tools_collapsed)

        for reasoning in self.query(ReasoningMessage):
            await reasoning.set_collapsed(self._tools_collapsed)

        try:
            for error_msg in self.query(ErrorMessage):
                error_msg.set_collapsed(self._tools_collapsed)
        except Exception:
            pass

    def action_cycle_mode(self) -> None:
        if self._current_bottom_app != BottomApp.Input:
            return
        self._refresh_profile_widgets()
        self._focus_current_bottom_app()
        self.run_worker(self._cycle_agent(), group="mode_switch", exclusive=True)

    def _refresh_profile_widgets(self) -> None:
        self._update_profile_widgets(self.agent_loop.agent_profile)

    def _on_profile_changed(self) -> None:
        self._refresh_profile_widgets()
        self._refresh_banner()

    def _active_model_label(self) -> str:
        """Display label for the active model: the alias, plus the server-reported
        name when auto-detection found one that differs (e.g. "Max (local) -
        Qwen3.6-..."). Purely cosmetic — never changes config or matching."""
        label = str(self.config.active_model)
        detected = self.agent_loop.detected_model_display_name()
        if not detected:
            return label
        # Strip a trailing .gguf for display; keep the rest as the server reports it.
        if detected.lower().endswith(".gguf"):
            detected = detected[:-5]
        active = self.config.get_active_model()
        # Skip the suffix when it would just repeat the alias or configured name.
        if detected and detected not in (label, active.alias, active.name):
            label = f"{label} - {detected}"
        return label

    def _refresh_banner(self) -> None:
        if self._banner:
            self._banner.set_state(
                self.config,
                self.agent_loop.skill_manager,
                self.agent_loop.mcp_registry,
                None,
                active_model_label=self._active_model_label(),
            )

    def _update_profile_widgets(self, profile: AgentProfile) -> None:
        if self._chat_input_container:
            self._chat_input_container.set_safety(profile.safety)
            self._chat_input_container.set_agent_name(profile.display_name.lower())
            self._chat_input_container.set_custom_border(None)

    async def _cycle_agent(self) -> None:
        new_profile = self.agent_loop.agent_manager.next_agent(
            self.agent_loop.agent_profile
        )
        self._update_profile_widgets(new_profile)
        if self._chat_input_container:
            self._chat_input_container.switching_mode = True

        def schedule_switch() -> None:
            self._switch_agent_generation += 1
            my_gen = self._switch_agent_generation

            def switch_agent_sync() -> None:
                try:
                    asyncio.run(self.agent_loop.switch_agent(new_profile.name))
                    self.agent_loop.set_approval_callback(self._approval_callback)
                    self.agent_loop.set_user_input_callback(self._user_input_callback)
                finally:
                    if (
                        self._chat_input_container
                        and self._switch_agent_generation == my_gen
                    ):
                        self.call_from_thread(self._refresh_banner)
                        self.call_from_thread(
                            setattr, self._chat_input_container, "switching_mode", False
                        )

            self.run_worker(
                switch_agent_sync, group="switch_agent", exclusive=True, thread=True
            )

        self.call_after_refresh(schedule_switch)

    def action_clear_quit(self) -> None:
        # If input has text, clear it and reset exit countdown.
        input_widgets = self.query(ChatInputContainer)
        if input_widgets:
            input_widget = input_widgets.first()
            if input_widget.value:
                input_widget.value = ""
                self._ctrl_c_exit_time = None
                return

        # If something is running, interrupt it (same as Esc) and reset countdown.
        if self._agent_running or self._voice_manager.transcribe_state != TranscribeState.IDLE:
            self.action_interrupt()
            self._ctrl_c_exit_time = None
            return

        # Nothing running — double-press to exit.
        now = time.monotonic()
        if self._ctrl_c_exit_time is not None and (now - self._ctrl_c_exit_time) < 3.0:
            self.action_force_quit()
        else:
            self._ctrl_c_exit_time = now
            self.notify("Press Ctrl+C again to exit", severity="warning", timeout=3)

    def action_force_quit(self) -> None:
        if self._agent_task and not self._agent_task.done():
            self._agent_task.cancel()
        self._narrator_manager.cancel()
        self.exit(result=self._get_session_resume_info())

    def action_scroll_chat_up(self) -> None:
        try:
            chat = self._cached_chat or self.query_one("#chat", ChatScroll)
            chat.scroll_relative(y=-5, animate=False)
        except Exception:
            pass

    def action_scroll_chat_down(self) -> None:
        try:
            chat = self._cached_chat or self.query_one("#chat", ChatScroll)
            chat.scroll_relative(y=5, animate=False)
        except Exception:
            pass

    def action_scroll_chat_home(self) -> None:
        try:
            chat = self._cached_chat or self.query_one("#chat", ChatScroll)
            chat.scroll_home(animate=False)
        except Exception:
            pass

    def action_scroll_chat_end(self) -> None:
        try:
            chat = self._cached_chat or self.query_one("#chat", ChatScroll)
            chat.scroll_end(animate=False)
        except Exception:
            pass

    def action_scroll_chat_page_up(self) -> None:
        try:
            chat = self._cached_chat or self.query_one("#chat", ChatScroll)
            chat.scroll_page_up(animate=False)
        except Exception:
            pass

    def action_scroll_chat_page_down(self) -> None:
        try:
            chat = self._cached_chat or self.query_one("#chat", ChatScroll)
            chat.scroll_page_down(animate=False)
        except Exception:
            pass

    def _show_skill_load_errors(self) -> None:
        errors = self.agent_loop.skill_manager.load_errors
        for err in errors:
            self.notify(
                f"Failed to load skill at {err.path}: {err.error}",
                severity="warning",
                timeout=10,
            )

    def _show_model_fallback_warning(self) -> None:
        config = self.config
        try:
            active = config.get_active_model()
        except ValueError:
            self.notify(
                f"Active model '{config.active_model}' not found and no models are configured.",
                severity="error",
                timeout=10,
            )
            return
        if active.alias != config.active_model:
            self.notify(
                f"Active model '{config.active_model}' not found, using '{active.alias}' instead.",
                severity="warning",
                timeout=10,
            )

    async def _show_clipboard_warning(self) -> None:
        if self.config.autocopy_to_clipboard and not is_reliable_clipboard_available():
            await self._mount_and_scroll(
                WarningMessage(
                    "No clipboard tool found (xclip, wl-copy). "
                    "Auto-copy may not work reliably. "
                    "Install xclip or disable auto-copy via /autocopy.",
                    show_border=False,
                )
            )

    async def _show_instruction_files_info(self) -> None:
        mgr = get_harness_files_manager()
        loaded: list[str] = []

        if mgr.load_user_doc().strip():
            loaded.append(f"`{VIBE_HOME.path / AGENTS_MD_FILENAME}` (user)")

        for doc_dir, _ in mgr.load_project_docs():
            loaded.append(f"`{doc_dir / AGENTS_MD_FILENAME}` (project)")

        for extra_path, _ in mgr.load_extra_instruction_files(self.config.extra_instruction_files):
            loaded.append(f"`{extra_path}` (extra)")

        if loaded:
            items = "\n".join(f"- {p}" for p in loaded)
            msg = f"**Instruction files loaded:**\n\n{items}"
        else:
            msg = "No instruction files loaded (no AGENTS.md or extra_instruction_files found)."

        await self._mount_and_scroll(UserCommandMessage(msg))

    async def _toggle_autocopy(self) -> None:
        new_value = not self.config.autocopy_to_clipboard
        VibeConfig.save_updates({"autocopy_to_clipboard": new_value})
        self.agent_loop.refresh_config()
        state = "enabled" if new_value else "disabled"
        await self._mount_and_scroll(UserCommandMessage(f"Auto-copy to clipboard {state}."))

    async def _cycle_preview_lines(self) -> None:
        new_value = cycle_preview_lines(self.config.tool_result_preview_lines)
        VibeConfig.save_updates({"tool_result_preview_lines": new_value})
        self.agent_loop.refresh_config()
        await self._mount_and_scroll(
            UserCommandMessage(f"Tool result preview set to {new_value} lines.")
        )

    async def _cycle_scrollback(self) -> None:
        new_value = cycle_message_prune_rows(self.config.message_prune_keep_rows)
        VibeConfig.save_updates({"message_prune_keep_rows": new_value})
        self.agent_loop.refresh_config()
        # Apply the new (often lower) threshold to the already-mounted history now.
        await self._try_prune()
        await self._mount_and_scroll(
            UserCommandMessage(f"Message scrollback set to {new_value} rows.")
        )

    async def _toggle_auto_detect_context_size(self) -> None:
        new_value = not self.config.auto_detect_context_size
        VibeConfig.save_updates({"auto_detect_context_size": new_value})
        self.agent_loop.refresh_config()

        if not new_value:
            await self._mount_and_scroll(
                UserCommandMessage("Context-size auto-detection disabled.")
            )
            return

        # Re-enabling: clear the per-model "already settled this run" latches so
        # this retry actually re-pulls context size and the model name, then run
        # detection now and refresh the banner with whatever name was detected.
        from privibe.core import agent_loop as _agent_loop_module
        _agent_loop_module.reset_context_size_detection_state()
        ctx_msg = await self.agent_loop.resolve_context_size()
        if ctx_msg:
            await self._mount_and_scroll(WarningMessage(ctx_msg, show_border=False))
        else:
            await self._mount_and_scroll(
                UserCommandMessage("Context-size auto-detection enabled.")
            )
        self._refresh_banner()
        # DEBUG LLM COMMUNICATIONS
    async def _toggle_llm_debug(self) -> None:
        new_value = not self.config.llm_debug_dump
        VibeConfig.save_updates({"llm_debug_dump": new_value})
        self.agent_loop.refresh_config()
        state = "enabled" if new_value else "disabled"
        await self._mount_and_scroll(
            UserCommandMessage(f"LLM debug dump {state}. Files written to ./debug/ on each LLM call.")
        )

    async def _toggle_stable_system_prefix(self) -> None:
        new_value = not self.config.stable_system_prefix
        VibeConfig.save_updates({"stable_system_prefix": new_value})
        self.agent_loop.refresh_config()
        state = "enabled" if new_value else "disabled"
        await self._mount_and_scroll(
            UserCommandMessage(
                f"Stable system prefix {state}. Keeps the datetime + project context "
                "out of the system prompt for better KV-cache reuse. Applies to new "
                "sessions; run /reload to apply it to the current one."
            )
        )

    async def _show_active_tools(self) -> None:
        tools = self.agent_loop.tool_manager.available_tools
        names = sorted(tools.keys())
        items = "\n".join(
            f"- **{name}** — {tools[name].description.split('.')[0].strip()}"
            for name in names
        )
        await self._mount_and_scroll(
            UserCommandMessage(f"**Active tools ({len(names)}):**\n\n{items}")
        )

    async def _show_available_agents(self) -> None:
        mgr = self.agent_loop.agent_manager
        active = mgr.active_profile.name
        agents = mgr.available_agents
        order = mgr.get_agent_order()
        remaining = [n for n in sorted(agents) if n not in order]
        items = "\n".join(
            f"- **{name}** _(active)_ — {agents[name].description}"
            if name == active
            else f"- **{name}** — {agents[name].description}"
            for name in order + remaining
            if name in agents
        )
        await self._mount_and_scroll(
            UserCommandMessage(f"**Available agents ({len(agents)}):**\n\n{items}")
        )

    async def _show_available_subagents(self) -> None:
        subagents = self.agent_loop.agent_manager.get_subagents()
        items = "\n".join(
            f"- **{p.name}** — {p.description}"
            for p in sorted(subagents, key=lambda p: p.name)
        )
        await self._mount_and_scroll(
            UserCommandMessage(f"**Available subagents ({len(subagents)}):**\n\n{items}")
        )

    async def _show_dangerous_directory_warning(self) -> None:
        is_dangerous, reason = is_dangerous_directory()
        if is_dangerous:
            warning = (
                f"⚠ WARNING: {reason}\n\nRunning in this location is not recommended."
            )
            await self._mount_and_scroll(WarningMessage(warning, show_border=False))

    async def _mount_and_scroll(
        self, widget: Widget, after: Widget | None = None
    ) -> None:
        messages_area = self._cached_messages_area or self.query_one("#messages")
        chat = self._cached_chat or self.query_one("#chat", ChatScroll)

        is_user_initiated = isinstance(widget, (UserMessage, UserCommandMessage))
        should_anchor = is_user_initiated or chat.is_at_bottom

        if after is not None and after.parent is messages_area:
            await messages_area.mount(widget, after=after)
        else:
            await messages_area.mount(widget)
        if isinstance(widget, StreamingMessageBase):
            await widget.write_initial_content()

        self.call_after_refresh(self._try_prune)
        if should_anchor:
            chat.anchor()

    async def _try_prune(self) -> None:
        messages_area = self._cached_messages_area or self.query_one("#messages")
        low = self.config.message_prune_keep_rows
        high = low + low // 2
        pruned = await prune_oldest_children(messages_area, low, high)
        if self._load_more.widget and not self._load_more.widget.parent:
            self._load_more.widget = None
        if pruned:
            chat = self._cached_chat or self.query_one("#chat", ChatScroll)
            if chat.is_at_bottom:
                self.call_later(chat.anchor)

    async def _refresh_windowing_from_history(self) -> None:
        if self._load_more.widget is None:
            return
        messages_area = self._cached_messages_area or self.query_one("#messages")
        has_backfill, tool_call_map = sync_backfill_state(
            history_messages=non_system_history_messages(self.agent_loop.messages),
            messages_children=list(messages_area.children),
            history_widget_indices=self._history_widget_indices,
            windowing=self._windowing,
        )
        self._tool_call_map = tool_call_map
        await self._load_more.set_visible(
            messages_area, visible=has_backfill, remaining=self._windowing.remaining
        )

    def action_copy_selection(self) -> None:
        copy_selection_to_clipboard(self, show_toast=False)

    def on_mouse_up(self, event: MouseUp) -> None:
        if self.config.autocopy_to_clipboard:
            copy_selection_to_clipboard(self, show_toast=True)

    def on_app_blur(self, event: AppBlur) -> None:
        self._terminal_notifier.on_blur()
        if self._chat_input_container and self._chat_input_container.input_widget:
            self._chat_input_container.input_widget.set_app_focus(False)

    def on_app_focus(self, event: AppFocus) -> None:
        self._terminal_notifier.on_focus()
        if self._chat_input_container and self._chat_input_container.input_widget:
            self._chat_input_container.input_widget.set_app_focus(True)

    def action_suspend_with_message(self) -> None:
        if WINDOWS or self._driver is None or not self._driver.can_suspend:
            return
        with self.suspend():
            rprint(
                "Privibe has been suspended. Run [bold cyan]fg[/bold cyan] to bring Privibe back."
            )
            os.kill(os.getpid(), signal.SIGTSTP)

    def _on_driver_signal_resume(self, event: Driver.SignalResume) -> None:
        # Textual doesn't repaint after resuming from Ctrl+Z (SIGTSTP);
        # force a full layout refresh so the UI isn't garbled.
        self.refresh(layout=True)

    def _make_default_narrator_manager(self) -> NarratorManager:
        return NarratorManager(
            config_getter=lambda: self.config, audio_player=AudioPlayer()
        )


def run_textual_ui(
    agent_loop: AgentLoop, startup: StartupOptions | None = None
) -> None:
    app = VibeApp(agent_loop=agent_loop, startup=startup)
    session_id = app.run()
    print_session_resume_message(session_id, agent_loop.stats)
