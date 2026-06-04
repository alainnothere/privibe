from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual.reactive import reactive

from privibe.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic


@dataclass
class TokenState:
    max_tokens: int = 0
    current_tokens: int = 0
    model_name: str | None = None
    # tokens_per_second: generation/decode rate. prompt_tokens_per_second:
    # prompt-processing/prefill rate (server-reported only).
    tokens_per_second: float = 0.0
    prompt_tokens_per_second: float = 0.0


class ContextProgress(NoMarkupStatic):
    tokens = reactive(TokenState())

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def watch_tokens(self, new_state: TokenState) -> None:
        if new_state.max_tokens == 0:
            self.update("")
            return

        ratio = min(1, new_state.current_tokens / new_state.max_tokens)
        parts: list[str] = []
        if new_state.model_name:
            parts.append(new_state.model_name)
        parts.append(f"{ratio:.0%} of {new_state.max_tokens // 1000}k tokens")
        # Speeds: when the server reports prompt-processing too (llama.cpp), label
        # both as pp (prefill) / tg (generation). When only generation is known
        # (server-reported or local estimate), show the bare number. Providers
        # that expose nothing simply omit this segment.
        if new_state.prompt_tokens_per_second > 0 and new_state.tokens_per_second > 0:
            parts.append(
                f"pp {new_state.prompt_tokens_per_second:.0f} · "
                f"tg {new_state.tokens_per_second:.0f} tok/s"
            )
        elif new_state.tokens_per_second > 0:
            parts.append(f"{new_state.tokens_per_second:.0f} tok/s")
        self.update(" · ".join(parts))
