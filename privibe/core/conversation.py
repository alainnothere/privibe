from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, overload

from privibe.core.session.session_loader import SessionLoader
from privibe.core.types import LLMMessage, Role
from privibe.core.utils.tags import CONTEXT_REFRESH_TAG

if TYPE_CHECKING:
    from privibe.core.config import VibeConfig


def _fix_dangling_tool_calls(messages: list[LLMMessage]) -> list[LLMMessage]:
    """Append shim tool-result responses for any unanswered tool calls at the tail.

    Interrupted sessions end with an assistant message whose tool_calls were never
    responded to. The gap is always at the tail (never mid-history), so we append
    rather than insert.
    """
    from privibe.core.utils import CancellationReason, get_user_cancellation_message

    result = list(messages)
    if not result:
        return result

    last = result[-1]
    if last.role != Role.assistant or not last.tool_calls:
        return result

    responded_ids = {
        m.tool_call_id
        for m in result
        if m.role == Role.tool and m.tool_call_id
    }

    for tc in last.tool_calls:
        if (tc.id or "") not in responded_ids:
            result.append(
                LLMMessage(
                    role=Role.tool,
                    tool_call_id=tc.id or "",
                    name=(tc.function.name or "") if tc.function else "",
                    content=str(
                        get_user_cancellation_message(CancellationReason.TOOL_NO_RESPONSE)
                    ),
                )
            )

    return result


def _apply_context_refresh(
    messages: list[LLMMessage], config: VibeConfig
) -> list[LLMMessage]:
    """Drop a stale context_refresh tail message, then append a fresh one."""
    from privibe.core.system_prompt import build_context_refresh_content

    result = list(messages)
    if result and f"<{CONTEXT_REFRESH_TAG}>" in (result[-1].content or ""):
        result.pop()
    content = build_context_refresh_content(config)
    result.append(LLMMessage(role=Role.user, content=content, injected=True))
    return result


class ConversationList:
    """Conversation message list with a strict append-only / top-removal interface.

    The only ways to mutate the list:
      add(msg)        — append one message to the top (end)
      rewind(n)       — remove the n top messages
      save()          — persist to disk via the registered save function
      restore(path)   — full rebuild from a saved session on disk

    Nothing outside this class may modify the stored messages. There is no
    insert, no reset, no update_system_prompt.
    """

    def __init__(
        self,
        observer: Callable[[LLMMessage], None] | None = None,
        config_getter: Callable[[], VibeConfig] | None = None,
    ) -> None:
        self._data: list[LLMMessage] = []
        self._observer = observer
        self._config_getter = config_getter
        self._save_fn: Callable[[], Awaitable[None]] | None = None
        self._reset_hooks: list[Callable[[], None]] = []
        self._silent: bool = False

    # ------------------------------------------------------------------
    # Write operations (the full public mutation API)
    # ------------------------------------------------------------------

    def add(self, msg: LLMMessage) -> None:
        self._data.append(msg)
        if not self._silent and self._observer is not None:
            self._observer(msg)

    def rewind(self, n: int) -> None:
        if n <= 0:
            return
        keep = max(0, len(self._data) - n)
        self._data = self._data[:keep]
        self._fire_reset_hooks()

    async def save(self) -> None:
        if self._save_fn is not None:
            await self._save_fn()

    def restore(self, session_path: Path) -> None:
        """Rebuild the full conversation from a saved session on disk.

        Loads non-system messages from messages.jsonl and the system prompt
        from meta.json, cleans up any dangling tool calls at the tail, and
        appends a fresh context_refresh message.
        """
        non_system_messages, metadata = SessionLoader.load_session(session_path)

        system_prompt_data = metadata.get("system_prompt")
        if system_prompt_data:
            system_msg = LLMMessage.model_validate(system_prompt_data)
        else:
            system_msg = LLMMessage(role=Role.system, content="")

        messages: list[LLMMessage] = [system_msg, *non_system_messages]
        messages = _fix_dangling_tool_calls(messages)

        config = self._config_getter() if self._config_getter is not None else None
        if config is not None:
            messages = _apply_context_refresh(messages, config)

        self._data = messages
        self._fire_reset_hooks()
        if self._observer is not None:
            for msg in self._data:
                self._observer(msg)

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def set_save_fn(self, fn: Callable[[], Awaitable[None]]) -> None:
        self._save_fn = fn

    def on_reset(self, hook: Callable[[], None]) -> None:
        self._reset_hooks.append(hook)

    @contextmanager
    def silent(self) -> Iterator[None]:
        prev = self._silent
        self._silent = True
        try:
            yield
        finally:
            self._silent = prev

    @contextmanager
    def no_reset_hooks(self) -> Iterator[None]:
        """Suppress reset hook notifications for this block.

        Use when you manage checkpoint state yourself (e.g. rewind_to_message
        filters its own checkpoints before calling rewind()) or when an internal
        rebuild should not discard rewind history (e.g. reload_with_initial_messages).
        """
        saved = list(self._reset_hooks)
        self._reset_hooks = []
        try:
            yield
        finally:
            self._reset_hooks = saved

    # ------------------------------------------------------------------
    # Read-only Sequence interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._data)

    @overload
    def __getitem__(self, index: int) -> LLMMessage: ...
    @overload
    def __getitem__(self, index: slice) -> list[LLMMessage]: ...
    def __getitem__(self, index: int | slice) -> LLMMessage | list[LLMMessage]:
        return self._data[index]

    def __iter__(self) -> Iterator[LLMMessage]:
        return iter(self._data)

    def __contains__(self, item: object) -> bool:
        return item in self._data

    def __bool__(self) -> bool:
        return bool(self._data)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fire_reset_hooks(self) -> None:
        for hook in self._reset_hooks:
            hook()
