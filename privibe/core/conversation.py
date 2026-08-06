from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, overload

from privibe.core.session.session_loader import SessionLoader
from privibe.core.types import AvailableTool, LLMMessage, Role
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
    """The full LLM-visible context with a strict append-only / top-removal
    interface: messages AND the advertised tools array. Everything the model
    sees on the wire is owned here; the payload is this object, verbatim.

    The only ways to mutate the state:
      add(msg)        — append one message to the top (end)
      rewind(n)       — remove the n top messages
      freeze_tools(t) — set the advertised tools, exactly once per session
      save()          — persist to disk via the registered save function
      restore(path)   — full rebuild from a saved session on disk

    Nothing outside this class may modify the stored messages or the frozen
    tools. There is no insert, no reset, no update_system_prompt, no
    tools setter after the freeze.
    """

    def __init__(
        self,
        observer: Callable[[LLMMessage], None] | None = None,
        config_getter: Callable[[], VibeConfig] | None = None,
        tools_provider: Callable[[], list[AvailableTool]] | None = None,
    ) -> None:
        self._data: list[LLMMessage] = []
        self._observer = observer
        self._config_getter = config_getter
        self._tools_provider = tools_provider
        # The advertised tools array. Part of the llama.cpp KV-cache prefix
        # (chat templates render it before the first user message), so it is
        # session state exactly like message 0: frozen at first use, stored
        # with the session, restored verbatim. None = not frozen yet.
        self._tools: list[AvailableTool] | None = None
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
        """Remove the n top (most recent) messages.

        A leading system message can never be removed: message 0 is generated
        once per session and is the llama.cpp KV-cache prefix. The only way to
        replace it is restore(), which is a session-boundary operation loading
        persisted bytes. Raising here (rather than clamping) makes any future
        rewind(len)-then-rebuild pattern fail loudly instead of silently
        reintroducing prompt mutation.
        """
        if n <= 0:
            return
        if self._data and self._data[0].role == Role.system and n >= len(self._data):
            raise ValueError(
                "rewind() may not remove the system message: message 0 is "
                "immutable for the life of the session (KV-cache prefix)."
            )
        keep = max(0, len(self._data) - n)
        self._data = self._data[:keep]
        self._fire_reset_hooks()

    async def save(self) -> None:
        if self._save_fn is not None:
            await self._save_fn()

    def freeze_tools(self, tools: list[AvailableTool]) -> None:
        """Set the advertised tools array, exactly once per session.

        After this call the tools are immutable for the life of the session;
        a second call is a broken invariant and fails loudly.
        """
        if self._tools is not None:
            raise ValueError(
                "tools are already frozen for this session: the advertised "
                "tools array is part of the KV-cache prefix and may never "
                "change after first use."
            )
        self._tools = list(tools)

    @property
    def frozen_tools(self) -> list[AvailableTool] | None:
        """The frozen tools array, or None if not frozen yet."""
        return list(self._tools) if self._tools is not None else None

    def tools_for_request(self) -> list[AvailableTool]:
        """The tools array to advertise on an LLM request.

        Freezes on first use: MCP servers connect asynchronously after
        startup, so the live set is only complete by the time the first
        request goes out. From then on the frozen array is returned no
        matter what the live tool set does.
        """
        if self._tools is None:
            if self._tools_provider is None:
                raise RuntimeError(
                    "tools_for_request() called with no frozen tools and no "
                    "tools provider wired."
                )
            self._tools = list(self._tools_provider())
        return list(self._tools)

    def restore(self, session_path: Path) -> None:
        """Rebuild the full conversation from a saved session on disk.

        Loads exactly what was stored: messages from messages.jsonl, and the
        frozen base (system prompt + tools) from base.json. Then applies the
        only two legal operations: pops shim-closing any dangling tool calls
        at the tail, and pushes a fresh context_refresh message.
        """
        non_system_messages, metadata = SessionLoader.load_session(session_path)

        base = SessionLoader.load_base(session_path)
        if base is not None:
            system_msg = LLMMessage.model_validate(base["system_prompt"])
            self._tools = [
                AvailableTool.model_validate(t) for t in base.get("tools", [])
            ]
        else:
            # MIGRATION KLUDGE - READ BEFORE TOUCHING, DO NOT COPY THIS PATTERN.
            #
            # The rule everywhere else: a session's stored base is written once
            # at creation and NEVER modified after; resume reads and sends it
            # verbatim (push/pop only). This branch is the single sanctioned
            # exception, for sessions created before base.json existed.
            #
            # Why: those sessions never stored their tools, and the original
            # bytes are unrecoverable (the binary that generated them is
            # gone). Setting _tools = None here means the next request adopts
            # the current live tools (tools_for_request), and the next save
            # writes them into base.json exactly once - after which this
            # branch never runs for that session again. The system prompt
            # falls back to the legacy meta.json copy.
            #
            # Why this must not be copied: any other code path that derives
            # part of the sent payload from anything but the stored session
            # silently changes what the model sees, breaks the llama.cpp
            # cache prefix, and reintroduces the exact bug family this design
            # exists to kill. If you think you need this pattern, you don't -
            # push a message instead.
            self._tools = None
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
        filters its own checkpoints before calling rewind()).
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
