"""The Tracer: what a sub-agent's run hands the parent.

Its tool calls and results pass through stamped with the parent call's id and
a depth; headers, verifier lines and the closing rule are rows of their own
(TraceLineEvent). Section headers are one row wide and the run ends on a rule
that names its outcome.
"""

from __future__ import annotations

from privibe.core.tools.builtins.bash import Bash, BashArgs, BashResult
from privibe.core.tools.subagent_trace import (
    LABEL_CHARS,
    SECTION_WIDTH,
    Tracer,
    fit_label,
    section_line,
)
from privibe.core.types import (
    AssistantEvent,
    ReasoningEvent,
    ToolCallEvent,
    ToolResultEvent,
    TraceLineEvent,
)

LONG_TASK = (
    "worker · round 1/3 · task 1: Replace line 1 of /tmp/orch-demo/src/answer.txt "
    "so that it reads ANSWER=2 and leave every other file in the directory alone"
)


def _tracer(depth: int = 1) -> Tracer:
    return Tracer("orchestrate", "call-1", depth)


def _bash_events(command: str, stdout: str) -> tuple[ToolCallEvent, ToolResultEvent]:
    call = ToolCallEvent(
        tool_call_id="b1",
        tool_name="bash",
        tool_class=Bash,
        args=BashArgs(command=command),
    )
    result = ToolResultEvent(
        tool_call_id="b1",
        tool_name="bash",
        tool_class=Bash,
        result=BashResult(command=command, stdout=stdout, stderr="", returncode=0),
    )
    return call, result


class TestSectionLine:
    def test_short_label_is_padded_to_the_width(self) -> None:
        line = section_line("manager · plan")
        assert line.startswith("── manager · plan ──")
        assert len(line) == SECTION_WIDTH

    def test_long_label_is_cut_so_the_rule_stays_one_row(self) -> None:
        line = section_line(LONG_TASK)
        assert len(line) == SECTION_WIDTH
        assert line.startswith("── worker · round 1/3 · task 1: Replace line 1")
        assert "…" in line
        assert "leave every other file" not in line

    def test_stamp_fits_after_a_cut_label(self) -> None:
        line = section_line(LONG_TASK, "7s · 3 calls")
        assert line.endswith("── 7s · 3 calls")
        assert len(line) < 80

    def test_fit_label_keeps_first_line_only(self) -> None:
        assert fit_label("title\n\nsecond paragraph") == "title"
        assert fit_label("x" * LABEL_CHARS) == "x" * LABEL_CHARS
        cut = fit_label("x" * (LABEL_CHARS + 1))
        assert len(cut) == LABEL_CHARS and cut.endswith("…")


class TestPassThrough:
    def test_calls_and_results_carry_the_parent_and_depth(self) -> None:
        tracer = _tracer()
        call, result = _bash_events("ls", "a\nb")
        [out_call] = tracer.on_event(call)
        [out_result] = tracer.on_event(result)
        assert isinstance(out_call, ToolCallEvent)
        assert isinstance(out_result, ToolResultEvent)
        assert out_call.parent_call_id == "call-1" and out_call.depth == 1
        assert out_result.parent_call_id == "call-1" and out_result.depth == 1
        # Nothing else about the event changed: the UI draws it as itself.
        assert out_call.tool_call_id == "b1" and out_call.args is call.args
        assert out_result.result is result.result

    def test_calls_and_results_carry_the_sections_tag(self) -> None:
        tracer = _tracer()
        tracer.start_section(
            "worker · round 2/10 · task 3: x", tag="worker 2/10 · task 3"
        )
        call, result = _bash_events("ls", "")
        [out_call] = tracer.on_event(call)
        [out_result] = tracer.on_event(result)
        assert out_call.section == "worker 2/10 · task 3"
        assert out_result.section == "worker 2/10 · task 3"
        # A grandchild already tagged by the section closest to it keeps that.
        [inner] = tracer.on_event(call.model_copy(update={"section": "explore"}))
        assert inner.section == "explore"
        # No tag given: nothing is stamped.
        plain = Tracer("task", "t-1")
        plain.start_section("explore · look")
        [untagged] = plain.on_event(call)
        assert untagged.section is None

    def test_a_grandchild_keeps_its_own_depth_on_top(self) -> None:
        tracer = _tracer(depth=1)
        call, _ = _bash_events("ls", "")
        nested = call.model_copy(update={"parent_call_id": "inner", "depth": 1})
        [out] = tracer.on_event(nested)
        assert isinstance(out, ToolCallEvent)
        assert out.depth == 2 and out.parent_call_id == "call-1"
        row = TraceLineEvent(tool_call_id="inner", message="── x ──", depth=1)
        [out_row] = tracer.on_event(row)
        assert isinstance(out_row, TraceLineEvent) and out_row.depth == 2

    def test_text_and_reasoning_are_dropped_a_stop_gets_a_row(self) -> None:
        tracer = _tracer()
        assert tracer.on_event(AssistantEvent(content="thinking out loud")) == []
        assert tracer.on_event(ReasoningEvent(content="hmm")) == []
        [row] = tracer.on_event(AssistantEvent(content="", stopped_by_middleware=True))
        assert isinstance(row, TraceLineEvent)
        assert row.message == "✕ stopped by middleware" and row.depth == 1


class TestSections:
    def test_header_is_a_row_that_is_stamped_in_place_at_the_end(self) -> None:
        tracer = _tracer()
        [header] = tracer.start_section("worker · round 1/3 · task 1")
        assert isinstance(header, TraceLineEvent)
        assert header.message.startswith("── worker · round 1/3 · task 1 ──")
        assert header.entry_id is not None and header.tool_call_id == "call-1"
        call, result = _bash_events("true", "")
        tracer.on_event(call)
        tracer.on_event(result)
        stamped, answer = tracer.end_section("done\n## Notes\nok")
        assert isinstance(stamped, TraceLineEvent)
        assert stamped.entry_id == header.entry_id
        assert stamped.message.rstrip().endswith("s · 1 call")
        assert isinstance(answer, TraceLineEvent) and answer.entry_id is None
        assert answer.message == "  done\n  ## Notes\n  ok"
        assert tracer.section_label == "worker · round 1/3 · task 1"

    def test_empty_answer_adds_no_row(self) -> None:
        tracer = _tracer()
        tracer.start_section("explore · look around")
        events = tracer.end_section("")
        assert len(events) == 1

    def test_close_is_a_rule_with_the_detail_under_it(self) -> None:
        [event] = _tracer().close(
            "end · done", "1 round · 4 calls", "verifier passes after a worker"
        )
        assert isinstance(event, TraceLineEvent)
        first, second = event.message.splitlines()
        assert first.startswith("── end · done ──")
        assert first.endswith(" 1 round · 4 calls")
        assert second == "  verifier passes after a worker"
        assert event.entry_id is None

    def test_close_without_detail_is_one_line(self) -> None:
        [event] = _tracer().close("end · cancelled")
        assert isinstance(event, TraceLineEvent)
        assert "\n" not in event.message
        assert event.message.startswith("── end · cancelled ──")
