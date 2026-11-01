"""The Tracer: what a sub-agent's run leaves in the parent's tool call body.

Section headers are one row wide, a result's preview never repeats the head
line that introduced it, and a run ends on a rule that names its outcome.
"""

from __future__ import annotations

from privibe.core.tools.builtins.bash import Bash, BashArgs, BashResult
from privibe.core.tools.builtins.hashed_replace_line import (
    HashedReplaceLine,
    HashedReplaceLineResult,
)
from privibe.core.tools.subagent_trace import (
    LABEL_CHARS,
    SECTION_WIDTH,
    Tracer,
    fit_label,
    section_line,
)
from privibe.core.types import ToolCallEvent, ToolResultEvent

LONG_TASK = (
    "worker · round 1/3 · task 1: Replace line 1 of /tmp/orch-demo/src/answer.txt "
    "so that it reads ANSWER=2 and leave every other file in the directory alone"
)


def _tracer(preview_lines: int = 5) -> Tracer:
    return Tracer("orchestrate", "call-1", preview_lines)


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
        # Head, two dashes of fill and the stamp; never wider than a terminal row.
        assert len(line) < 80

    def test_fit_label_keeps_first_line_only(self) -> None:
        assert fit_label("title\n\nsecond paragraph") == "title"
        assert fit_label("x" * LABEL_CHARS) == "x" * LABEL_CHARS
        cut = fit_label("x" * (LABEL_CHARS + 1))
        assert len(cut) == LABEL_CHARS and cut.endswith("…")


class TestResultPreview:
    def test_bash_with_output_shows_the_output_once(self) -> None:
        tracer = _tracer()
        call, result = _bash_events("ls src", "answer.txt\nnotes.md")
        tracer.on_event(call)
        [event] = tracer.on_event(result)
        assert event.message == "✓ bash $ ls src\n  answer.txt\n  notes.md"

    def test_bash_with_no_output_does_not_repeat_its_command(self) -> None:
        tracer = _tracer()
        call, result = _bash_events("sed -i 's/1/2/' src/answer.txt", "")
        tracer.on_event(call)
        [event] = tracer.on_event(result)
        assert event.message == "✓ bash $ sed -i 's/1/2/' src/answer.txt"
        assert event.message.count("sed -i") == 1

    def test_edit_result_does_not_repeat_its_path(self) -> None:
        tracer = _tracer()
        path = "/tmp/orch-demo/src/answer.txt"
        result = ToolResultEvent(
            tool_call_id="h1",
            tool_name="hashed_replace_line",
            tool_class=HashedReplaceLine,
            result=HashedReplaceLineResult(
                path=path, total_replacements=1, total_lines_replaced=1, context="a"
            ),
        )
        [event] = tracer.on_event(result)
        head = event.message.splitlines()[0]
        assert path in head
        # The path was the biggest string field; it is already on the head.
        assert event.message.count(path) == 1

    def test_running_line_is_replaced_by_its_result(self) -> None:
        tracer = _tracer()
        call, result = _bash_events("true", "")
        [running] = tracer.on_event(call)
        [done] = tracer.on_event(result)
        assert running.message == "⋯ bash $ true"
        assert running.entry_id == done.entry_id


class TestClose:
    def test_close_is_a_rule_with_the_detail_under_it(self) -> None:
        tracer = _tracer()
        [event] = tracer.close(
            "end · done", "1 round · 4 calls", "verifier passes after a worker"
        )
        first, second = event.message.splitlines()
        assert first.startswith("── end · done ──")
        assert first.endswith(" 1 round · 4 calls")
        assert second == "  verifier passes after a worker"
        assert event.entry_id is None

    def test_close_without_detail_is_one_line(self) -> None:
        [event] = _tracer().close("end · cancelled")
        assert "\n" not in event.message
        assert event.message.startswith("── end · cancelled ──")
