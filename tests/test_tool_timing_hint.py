"""Where a tool line's time went, live and after resume.

The finished hint is derived from two MessageMeta: the assistant message
that made the call (request_meta) and the tool response (tool_meta). These
tests pin that both reach the ToolResultEvent on every emission path, that
llama.cpp's clocks travel from the response to the line, that the formatter
does the subtraction right, and that resume feeds the same formatter.
"""

from __future__ import annotations

from pydantic import BaseModel
import pytest

from privibe.cli.textual_ui.widgets.tools import finished_hint_text
from privibe.cli.textual_ui.windowing.history import build_history_widgets
from privibe.core.config import ProviderConfig
from privibe.core.llm.backend.reasoning_adapter import ReasoningAdapter
from privibe.core.tools.base import ToolPermission
from privibe.core.tools.builtins.bash import Bash
from privibe.core.types import (
    ApprovalResponse,
    FunctionCall,
    LLMChunk,
    LLMMessage,
    MessageMeta,
    Role,
    ServerTimings,
    ToolCall,
    ToolOutcome,
    ToolResultEvent,
)
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from tests.test_agent_tool_call import (
    act_and_collect_events,
    make_agent_loop,
    make_todo_tool_call,
)

TIMINGS = ServerTimings(
    cache_n=180_000,
    prompt_n=1_200,
    prompt_ms=12_300.0,
    predicted_n=310,
    predicted_ms=4_100.0,
)


def _with_timings(chunk: LLMChunk) -> LLMChunk:
    assert chunk.usage is not None
    return chunk.model_copy(
        update={"usage": chunk.usage.model_copy(update={"server_timings": TIMINGS})}
    )


def _results(events: list) -> list[ToolResultEvent]:
    return [e for e in events if isinstance(e, ToolResultEvent)]


# ---------------------------------------------------------------------------
# Live: every result event carries both metas
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_results_carry_request_and_tool_meta() -> None:
    # Two identical calls: the second is deduplicated, which is one of the
    # paths that used to yield its event before adding its message.
    backend = FakeBackend([
        [
            _with_timings(
                mock_llm_chunk(
                    content="",
                    tool_calls=[
                        make_todo_tool_call("c1", index=0),
                        make_todo_tool_call("c2", index=1),
                    ],
                )
            )
        ],
        [mock_llm_chunk(content="done")],
    ])
    loop = make_agent_loop(auto_approve=True, backend=backend)

    results = _results(await act_and_collect_events(loop, "todos"))

    # The duplicate's skip is emitted before the real call runs.
    by_id = {r.tool_call_id: r for r in results}
    assert set(by_id) == {"c1", "c2"}
    first, second = by_id["c1"], by_id["c2"]
    for r in results:
        assert r.request_meta is not None
        assert r.request_meta.request_sent_at is not None
        assert r.request_meta.response_done_at is not None
        assert r.request_meta.server_timings == TIMINGS
        assert r.tool_meta is not None
        assert r.tool_meta.tool_finished_at is not None
    assert first.first_in_message and not second.first_in_message
    assert first.tool_meta is not None and second.tool_meta is not None
    assert first.tool_meta.tool_outcome == ToolOutcome.success
    assert first.tool_meta.tool_started_at is not None
    assert second.tool_meta.tool_outcome == ToolOutcome.skipped


@pytest.mark.asyncio
async def test_live_skip_without_approval_callback_carries_tool_meta() -> None:
    backend = FakeBackend([
        [mock_llm_chunk(content="", tool_calls=[make_todo_tool_call("c1")])],
        [mock_llm_chunk(content="ok")],
    ])
    loop = make_agent_loop(
        auto_approve=False, todo_permission=ToolPermission.ASK, backend=backend
    )

    (result,) = _results(await act_and_collect_events(loop, "todos"))

    assert result.skipped
    assert result.tool_meta is not None
    assert result.tool_meta.tool_outcome == ToolOutcome.skipped


@pytest.mark.asyncio
async def test_live_approval_wait_is_recorded() -> None:
    async def approve(
        _n: str, _a: BaseModel, _i: str, _rp: list | None = None
    ) -> tuple[ApprovalResponse, str | None]:
        return (ApprovalResponse.YES, None)

    backend = FakeBackend([
        [mock_llm_chunk(content="", tool_calls=[make_todo_tool_call("c1")])],
        [mock_llm_chunk(content="ok")],
    ])
    loop = make_agent_loop(
        auto_approve=False,
        todo_permission=ToolPermission.ASK,
        approval_callback=approve,
        backend=backend,
    )

    (result,) = _results(await act_and_collect_events(loop, "todos"))

    meta = result.tool_meta
    assert meta is not None
    assert meta.approval_asked_at is not None
    assert meta.approval_answered_at is not None
    assert meta.approval_asked_at <= meta.approval_answered_at
    assert meta.tool_started_at is not None
    assert meta.approval_answered_at <= meta.tool_started_at
    assert "approval " in finished_hint_text(result.duration, None, tool_meta=meta)


def test_reasoning_adapter_keeps_the_server_clocks() -> None:
    provider = ProviderConfig(
        name="l", api_base="http://x", api_key_env_var="K", api_style="openai"
    )
    chunk = ReasoningAdapter().parse_response(
        {
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "timings": {
                "cache_n": 180000,
                "prompt_n": 1200,
                "prompt_ms": 12300.0,
                "prompt_per_second": 97.5,
                "predicted_n": 310,
                "predicted_ms": 4100.0,
                "predicted_per_second": 75.6,
            },
        },
        provider,
    )
    assert chunk.usage is not None
    assert chunk.usage.server_timings == TIMINGS


# ---------------------------------------------------------------------------
# The formatter
# ---------------------------------------------------------------------------


def _request(**kw: object) -> MessageMeta:
    return MessageMeta(**kw)  # type: ignore[arg-type]


class TestFinishedHint:
    def test_server_clocks_with_overhead(self) -> None:
        # wall 20s, server 12.3 + 4.1 = 16.4s -> 3.6s nobody's clock saw
        meta = _request(
            request_sent_at=1000.0, response_done_at=1020.0, server_timings=TIMINGS
        )
        text = finished_hint_text(0.03, None, request_meta=meta, include_request=True)
        assert text.startswith(
            "(prefill 12.3s 1.2k new 180.0k cached · decode 4.1s 310 tok · "
            "overhead 3.6s · run 0.0s · "
        )

    def test_overhead_hidden_when_negligible(self) -> None:
        meta = _request(
            request_sent_at=1000.0, response_done_at=1016.45, server_timings=TIMINGS
        )
        text = finished_hint_text(None, None, request_meta=meta, include_request=True)
        assert "overhead" not in text

    def test_no_cache_hit_says_nothing_about_cache(self) -> None:
        meta = _request(server_timings=ServerTimings(prompt_n=50, prompt_ms=200.0))
        text = finished_hint_text(None, None, request_meta=meta, include_request=True)
        assert text == "(prefill 0.2s 50 new)"

    def test_fallback_to_our_own_instants(self) -> None:
        meta = _request(
            request_sent_at=1000.0, first_chunk_at=1002.0, response_done_at=1005.5
        )
        text = finished_hint_text(1.0, None, request_meta=meta, include_request=True)
        assert text.startswith("(first token 2.0s · gen 3.5s · run 1.0s · ")

    def test_wall_only(self) -> None:
        meta = _request(request_sent_at=1000.0, response_done_at=1090.0)
        text = finished_hint_text(None, None, request_meta=meta, include_request=True)
        assert text.startswith("(llm 1m30s · ")

    def test_later_calls_of_a_batch_do_not_claim_the_request(self) -> None:
        meta = _request(request_sent_at=1000.0, server_timings=TIMINGS)
        text = finished_hint_text(0.5, None, request_meta=meta, include_request=False)
        assert "prefill" not in text
        assert text.startswith("(run 0.5s · ")

    def test_approval_and_run(self) -> None:
        tool = MessageMeta(approval_asked_at=10.0, approval_answered_at=14.2)
        assert finished_hint_text(2.5, None, tool_meta=tool) == (
            "(approval 4.2s · run 2.5s)"
        )

    def test_nothing_known_is_empty(self) -> None:
        assert finished_hint_text(None, None) == ""

    def test_clock_prefers_the_stored_request_instant(self) -> None:
        a = finished_hint_text(None, 5.0, request_meta=_request(request_sent_at=5.0))
        b = finished_hint_text(None, 5.0)
        assert a == b and a.startswith("(") and ":" in a


# ---------------------------------------------------------------------------
# Resume: the same inputs reach the same formatter
# ---------------------------------------------------------------------------


def test_history_links_request_meta_and_first_call() -> None:
    from weakref import WeakKeyDictionary

    from privibe.cli.textual_ui.widgets.tools import ToolResultMessage

    request = MessageMeta(request_sent_at=1.0, server_timings=TIMINGS)

    def call(i: str, idx: int) -> ToolCall:
        return ToolCall(
            id=i,
            index=idx,
            function=FunctionCall(name="bash", arguments='{"command":"ls"}'),
        )

    def result(i: str) -> LLMMessage:
        return LLMMessage(
            role=Role.tool,
            name="bash",
            tool_call_id=i,
            content="x",
            meta=MessageMeta(
                tool_started_at=2.0,
                tool_finished_at=3.0,
                tool_outcome=ToolOutcome.success,
                tool_result={
                    "command": "ls",
                    "stdout": "",
                    "stderr": "",
                    "returncode": 0,
                },
            ),
        )

    batch = [
        LLMMessage(
            role=Role.assistant,
            content="",
            tool_calls=[call("a", 0), call("b", 1)],
            meta=request,
        ),
        result("a"),
        result("b"),
    ]
    widgets = build_history_widgets(
        batch,
        {},
        start_index=0,
        tools_collapsed=True,
        history_widget_indices=WeakKeyDictionary(),
        tool_classes={"bash": Bash},
    )
    events = [w._event for w in widgets if isinstance(w, ToolResultMessage)]
    assert [e.tool_call_id for e in events if e] == ["a", "b"]
    a, b = events
    assert a is not None and b is not None
    assert a.request_meta == request and a.first_in_message
    assert b.request_meta == request and not b.first_in_message
    assert a.duration == 1.0
