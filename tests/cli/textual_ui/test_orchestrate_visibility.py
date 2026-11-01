"""What orchestrate shows when it starts and when it is done.

The call line used to be overwritten by a one-line result the moment the run
finished, taking the goal, scope, files and done-command off the screen. The
tool now keeps them under the outcome (ToolUIData.format_finished_text), and
the goal in full is the call's fold-out content from the start.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult

from privibe.cli.textual_ui.widgets.tool_widgets import get_approval_widget
from privibe.cli.textual_ui.widgets.tools import ToolCallMessage, ToolResultMessage
from privibe.core.tools.builtins.bash import Bash, BashArgs, BashResult
from privibe.core.tools.builtins.orchestrate import (
    Orchestrate,
    OrchestrateArgs,
    OrchestrateResult,
)
from privibe.core.tools.ui import ToolUIData, ToolUIDataAdapter
from privibe.core.types import ToolCallEvent, ToolResultEvent

GOAL = (
    "Fix the plan slice in parse.py.\n\n"
    "Requirements:\n1. add a regression test\n2. keep the tests green"
)


def _args(**overrides: object) -> OrchestrateArgs:
    base: dict[str, object] = {
        "goal": GOAL,
        "files": ["/r/parse.py", "/r/tests/test_parse.py"],
        "scope": ["/r/privibe/core/orchestrate"],
        "done_command": "uv run pytest tests/orchestrate -q",
        "max_rounds": 6,
    }
    base.update(overrides)
    return OrchestrateArgs.model_validate(base)


def _result(**overrides: object) -> OrchestrateResult:
    base: dict[str, object] = {
        "status": "done",
        "detail": "verifier passes after a worker",
        "rounds": 2,
        "calls": 5,
        "verifier_exit_code": 0,
        "changed_files": ["/r/parse.py"],
        "ledger_dir": "/ledger",
    }
    base.update(overrides)
    return OrchestrateResult.model_validate(base)


def _events(
    args: OrchestrateArgs, result: OrchestrateResult
) -> tuple[ToolCallEvent, ToolResultEvent]:
    call = ToolCallEvent(
        tool_call_id="c1", tool_name="orchestrate", tool_class=Orchestrate, args=args
    )
    done = ToolResultEvent(
        tool_call_id="c1",
        tool_name="orchestrate",
        tool_class=Orchestrate,
        result=result,
    )
    return call, done


class _Host(App[None]):
    def __init__(self, *widgets) -> None:
        super().__init__()
        self._widgets = widgets

    def compose(self) -> ComposeResult:
        yield from self._widgets


async def _finish(
    call: ToolCallMessage, done: ToolResultEvent, host: _Host, pilot
) -> str:
    """Mount the result under the call, as the live path does, and return the
    call line's text afterwards.
    """
    await host.mount(ToolResultMessage(done, call, collapsed=False))
    await pilot.pause()
    return str(call._text_widget.render())


# --- core: what the finished line says -------------------------------------


def test_finished_text_is_the_outcome_then_the_job() -> None:
    message = "orchestrate done after 2 rounds, 5 calls: verifier passes after a worker"
    text = Orchestrate.format_finished_text(_args(), message)
    assert text.splitlines() == [
        message,
        '  goal: "Fix the plan slice in parse.py."',
        "  scope: /r/privibe/core/orchestrate",
        "  files: /r/parse.py, /r/tests/test_parse.py",
        "  done when: `uv run pytest tests/orchestrate -q` exits 0",
        "  max_rounds: 6",
    ]


def test_finished_text_says_so_when_nothing_verified_the_job() -> None:
    text = Orchestrate.format_finished_text(
        _args(done_command="", scope=[], files=[], max_rounds=None), "outcome"
    )
    assert "  done when: the manager says so (unverified)" in text
    assert "  scope: (none)" in text
    assert "  files: (none)" in text
    assert "max_rounds" not in text


def test_finished_text_with_an_empty_goal_has_no_goal_line() -> None:
    text = Orchestrate.format_finished_text(_args(goal=""), "outcome")
    assert "goal:" not in text
    assert text.splitlines()[0] == "outcome"


def test_finished_goal_is_cut_like_the_call_line() -> None:
    text = Orchestrate.format_finished_text(_args(goal="y" * 400), "outcome")
    goal_line = text.splitlines()[1]
    assert goal_line == '  goal: "' + "y" * 299 + '…"'


def test_a_goal_up_to_300_characters_is_shown_whole() -> None:
    """The line used to cut at 80, which clipped an ordinary one-sentence goal.
    A first line of exactly 300 characters is not cut; 301 is.
    """
    sentence = (
        "Answer this question: how much is 1 + 1? Do not edit any file and do "
        "not run any command. Just state the answer in your final text reply."
    )
    assert 80 < len(sentence) < 300
    shown = Orchestrate.format_finished_text(_args(goal=sentence), "outcome")
    assert f'  goal: "{sentence}"' in shown.splitlines()
    assert "…" not in shown

    exact = Orchestrate.format_finished_text(_args(goal="z" * 300), "outcome")
    assert exact.splitlines()[1] == '  goal: "' + "z" * 300 + '"'
    over = Orchestrate.format_finished_text(_args(goal="z" * 301), "outcome")
    assert over.splitlines()[1] == '  goal: "' + "z" * 299 + '…"'


@pytest.mark.parametrize(
    ("rounds", "calls", "expected"),
    [
        (1, 1, "after 1 round, 1 call"),
        (1, 5, "after 1 round, 5 calls"),
        (3, 1, "after 3 rounds, 1 call"),
        (0, 0, "after 0 rounds, 0 calls"),
    ],
)
def test_result_line_counts_in_the_singular(
    rounds: int, calls: int, expected: str
) -> None:
    display = Orchestrate.format_result_display(_result(rounds=rounds, calls=calls))
    assert display.message.startswith(f"orchestrate done {expected}")


def test_result_line_success_follows_the_status() -> None:
    ok = {"done", "done_unverified"}
    for status in (
        "done",
        "done_unverified",
        "rounds_exhausted",
        "stalled",
        "cancelled",
    ):
        display = Orchestrate.format_result_display(_result(status=status))
        assert display.success is (status in ok), status


def test_default_finished_text_is_the_result_line_alone() -> None:
    """Every other tool keeps today's behaviour: the result line, unchanged."""
    assert Bash.format_finished_text(BashArgs(command="ls"), "bash $ ls") == "bash $ ls"
    assert ToolUIData.format_finished_text(BashArgs(command="ls"), "x") == "x"


def test_adapter_finished_text_never_raises() -> None:
    adapter = ToolUIDataAdapter(Orchestrate)
    no_args = ToolCallEvent(
        tool_call_id="c", tool_name="orchestrate", tool_class=Orchestrate, args=None
    )
    assert adapter.get_finished_text(no_args, "outcome") == "outcome"

    wrong_args = ToolCallEvent(
        tool_call_id="c",
        tool_name="orchestrate",
        tool_class=Orchestrate,
        args=BashArgs(command="ls"),
    )
    assert adapter.get_finished_text(wrong_args, "outcome") == "outcome"

    plain = ToolUIDataAdapter(SimpleNamespace)
    assert plain.get_finished_text(no_args, "outcome") == "outcome"


# --- the approval prompt ----------------------------------------------------


@pytest.mark.asyncio
async def test_approval_shows_the_whole_job() -> None:
    widget = get_approval_widget("orchestrate", _args())
    async with _Host(widget).run_test():
        shown = " ".join(str(w.render()) for w in widget.query("Static, Markdown"))
        assert "Fix the plan slice in parse.py." in shown
        assert "scope: /r/privibe/core/orchestrate" in shown
        assert "files: /r/parse.py, /r/tests/test_parse.py" in shown
        assert "`uv run pytest tests/orchestrate -q` exits 0" in shown
        assert "max_rounds: 6" in shown


# --- the call widget: while running, then done ------------------------------


@pytest.mark.asyncio
async def test_running_line_has_the_job_and_the_goal_is_one_click_away() -> None:
    call_event, _ = _events(_args(), _result())
    call = ToolCallMessage(call_event)
    async with _Host(call).run_test():
        line = str(call._text_widget.render())
        assert line.startswith('orchestrate "Fix the plan slice in parse.py."\n')
        assert "done when: `uv run pytest tests/orchestrate -q` exits 0" in line
        # The goal is longer than its first line: there is a fold-out for it.
        assert call._triangle_widget.display
        assert str(call._params_widget.render()) == GOAL
        assert not call._body.display
        await call.set_collapsed(False)
        assert call._body.display
        assert "Requirements:" in str(call._params_widget.render())


@pytest.mark.asyncio
async def test_job_stays_on_screen_after_the_result_lands() -> None:
    call_event, done_event = _events(_args(), _result())
    call = ToolCallMessage(call_event)
    host = _Host(call)
    async with host.run_test() as pilot:
        line = await _finish(call, done_event, host, pilot)
        lines = line.splitlines()
        assert lines[0] == (
            "orchestrate done after 2 rounds, 5 calls: verifier passes after a worker"
        )
        assert '  goal: "Fix the plan slice in parse.py."' in lines
        assert "  scope: /r/privibe/core/orchestrate" in lines
        assert "  files: /r/parse.py, /r/tests/test_parse.py" in lines
        assert "  done when: `uv run pytest tests/orchestrate -q` exits 0" in lines
        assert "  max_rounds: 6" in lines
        assert call.success
        # The full goal is still in the fold-out after the run.
        assert str(call._params_widget.render()) == GOAL


@pytest.mark.asyncio
async def test_failed_job_keeps_the_job_and_shows_the_failure() -> None:
    result = _result(
        status="rounds_exhausted",
        detail="6 rounds used; the finalizer ran and the verifier still fails",
        rounds=6,
        calls=17,
        verifier_exit_code=1,
    )
    call_event, done_event = _events(_args(), result)
    call = ToolCallMessage(call_event)
    host = _Host(call)
    async with host.run_test() as pilot:
        line = await _finish(call, done_event, host, pilot)
        assert line.splitlines()[0].startswith(
            "orchestrate rounds_exhausted after 6 rounds"
        )
        assert "the verifier still fails" in line.splitlines()[0]
        assert "  done when: `uv run pytest tests/orchestrate -q` exits 0" in line
        assert not call.success


@pytest.mark.asyncio
async def test_unverified_job_says_so_after_the_run_too() -> None:
    args = _args(done_command="")
    result = _result(
        status="done_unverified", detail="manager declared done; no verifier"
    )
    call_event, done_event = _events(args, result)
    call = ToolCallMessage(call_event)
    host = _Host(call)
    async with host.run_test() as pilot:
        line = await _finish(call, done_event, host, pilot)
        assert line.splitlines()[0].startswith("orchestrate done_unverified")
        assert "  done when: the manager says so (unverified)" in line


@pytest.mark.asyncio
async def test_one_line_goal_needs_no_fold_out_and_shows_once() -> None:
    args = _args(goal="Make f return 2.")
    call_event, done_event = _events(args, _result())
    call = ToolCallMessage(call_event)
    host = _Host(call)
    async with host.run_test() as pilot:
        assert not call._triangle_widget.display
        line = await _finish(call, done_event, host, pilot)
        assert line.count("Make f return 2.") == 1
        assert '  goal: "Make f return 2."' in line


@pytest.mark.asyncio
async def test_errored_call_keeps_the_job_and_the_error_is_in_the_body() -> None:
    """An orchestrate call that errored (a file that is not there, a protected
    path) never got a result. The line still says which job it was, so a
    refusal is not an anonymous "orchestrate error"; the error text itself is
    the result body, as for every tool.
    """
    call_event, _ = _events(_args(), _result())
    error = ToolResultEvent(
        tool_call_id="c1",
        tool_name="orchestrate",
        tool_class=Orchestrate,
        error="not a file: /r/parse.py",
    )
    call = ToolCallMessage(call_event)
    result = ToolResultMessage(error, call, collapsed=False)
    host = _Host(call)
    async with host.run_test() as pilot:
        await host.mount(result)
        await pilot.pause()
        lines = str(call._text_widget.render()).splitlines()
        assert lines[0] == "orchestrate error"
        assert "  done when: `uv run pytest tests/orchestrate -q` exits 0" in lines
        assert not call.success
        body = " ".join(str(w.render()) for w in result.query("Static"))
        assert "not a file: /r/parse.py" in body


@pytest.mark.asyncio
async def test_denied_call_keeps_the_job_and_says_skipped() -> None:
    """Saying no at the approval prompt: the line reads skipped, with the job it
    was about to run.
    """
    call_event, _ = _events(_args(), _result())
    skipped = ToolResultEvent(
        tool_call_id="c1",
        tool_name="orchestrate",
        tool_class=Orchestrate,
        skipped=True,
        skip_reason="User denied",
    )
    call = ToolCallMessage(call_event)
    host = _Host(call)
    async with host.run_test() as pilot:
        line = await _finish(call, skipped, host, pilot)
        lines = line.splitlines()
        assert lines[0] == "orchestrate skipped"
        assert '  goal: "Fix the plan slice in parse.py."' in lines
        assert not call.success


@pytest.mark.asyncio
async def test_other_tools_keep_a_bare_error_line() -> None:
    """The default is untouched for everything but orchestrate."""
    call_event = ToolCallEvent(
        tool_call_id="b2",
        tool_name="bash",
        tool_class=Bash,
        args=BashArgs(command="false"),
    )
    error = ToolResultEvent(
        tool_call_id="b2", tool_name="bash", tool_class=Bash, error="exit 1"
    )
    call = ToolCallMessage(call_event)
    host = _Host(call)
    async with host.run_test() as pilot:
        assert await _finish(call, error, host, pilot) == "bash error"


@pytest.mark.asyncio
async def test_other_tools_finish_exactly_as_before() -> None:
    call_event = ToolCallEvent(
        tool_call_id="b1",
        tool_name="bash",
        tool_class=Bash,
        args=BashArgs(command="ls -la"),
    )
    done = ToolResultEvent(
        tool_call_id="b1",
        tool_name="bash",
        tool_class=Bash,
        result=BashResult(command="ls -la", stdout="a\n", stderr="", returncode=0),
    )
    call = ToolCallMessage(call_event)
    host = _Host(call)
    async with host.run_test() as pilot:
        before = str(call._text_widget.render())
        after = await _finish(call, done, host, pilot)
        assert before == "bash $ ls -la"
        assert after == "bash $ ls -la"


# --- resume: the same line is drawn from the stored call -------------------


@pytest.mark.asyncio
async def test_history_line_keeps_the_job_too() -> None:
    """On resume the call widget is rebuilt from the stored arguments and the
    result is mounted under it, the same path as live, so it must read the
    same.
    """
    call_event, done_event = _events(_args(), _result())
    call = ToolCallMessage(call_event, tool_name="orchestrate", history=True)
    host = _Host(call)
    async with host.run_test() as pilot:
        line = await _finish(call, done_event, host, pilot)
        assert line.splitlines()[0].startswith("orchestrate done after 2 rounds")
        assert "  scope: /r/privibe/core/orchestrate" in line
        assert "  done when: `uv run pytest tests/orchestrate -q` exits 0" in line


@pytest.mark.asyncio
async def test_history_line_without_a_known_tool_is_the_bare_result() -> None:
    call = ToolCallMessage(None, tool_name="orchestrate", history=True)
    host = _Host(call)
    async with host.run_test():
        assert call.finished_text("orchestrate done") == "orchestrate done"
