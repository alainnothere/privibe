from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel
import pytest

from privibe.cli.console_ui.app import ConsoleUI
from privibe.core.tools.builtins.ask_user_question import (
    AskUserQuestion,
    AskUserQuestionArgs,
    AskUserQuestionResult,
    Choice,
    Question,
)
from privibe.core.types import (
    ApprovalResponse,
    AssistantEvent,
    FunctionCall,
    LLMMessage,
    ReasoningEvent,
    Role,
    ToolCall,
    ToolCallEvent,
    ToolResultEvent,
)


class ToolArgs(BaseModel):
    command: str = "echo hi"


class StubAgentLoop:
    """Just enough of AgentLoop for ConsoleUI unit tests."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(auto_approve=False)
        self.tool_manager = SimpleNamespace(available_tools={})
        self.skill_manager = SimpleNamespace(get_skill=lambda name: None)
        self.approval_callback = None
        self.user_input_callback = None
        self.approved_always: list[tuple[str, Any]] = []
        self.acted_prompts: list[str] = []
        self.act_events: list[Any] = []

    def set_approval_callback(self, callback: Any) -> None:
        self.approval_callback = callback

    def set_user_input_callback(self, callback: Any) -> None:
        self.user_input_callback = callback

    def approve_always(self, tool_name: str, required_permissions: Any) -> None:
        self.approved_always.append((tool_name, required_permissions))

    async def act(self, prompt: str):
        self.acted_prompts.append(prompt)
        for event in self.act_events:
            yield event


@pytest.fixture
def ui() -> ConsoleUI:
    return ConsoleUI(StubAgentLoop())


def queue_replies(ui: ConsoleUI, replies: list[str]) -> None:
    pending = list(replies)

    async def fake_read_reply(prompt: str) -> str:
        return pending.pop(0)

    ui._read_reply = fake_read_reply  # type: ignore[method-assign]


# ----------------------------------------------------------------------
# Event formatting
# ----------------------------------------------------------------------


def test_assistant_deltas_concatenate_then_close(ui: ConsoleUI, capsys) -> None:
    ui._handle_event(AssistantEvent(content="Hel"))
    ui._handle_event(AssistantEvent(content="lo"))
    ui._handle_event(
        ToolCallEvent(
            tool_call_id="t1",
            tool_name="ask_user_question",
            tool_class=AskUserQuestion,
        )
    )
    out = capsys.readouterr().out
    assert "Hello\n" in out
    assert "[tool] ask_user_question:" in out


def test_reasoning_prefix_printed_once(ui: ConsoleUI, capsys) -> None:
    ui._handle_event(ReasoningEvent(content="thinking "))
    ui._handle_event(ReasoningEvent(content="hard"))
    ui._handle_event(AssistantEvent(content="answer"))
    ui._close_block()
    out = capsys.readouterr().out
    assert out.count("[thinking]") == 1
    assert "thinking hard" in out
    assert "answer" in out


def test_tool_result_lines(ui: ConsoleUI) -> None:
    base = {"tool_name": "bash", "tool_class": None, "tool_call_id": "t1"}
    assert "ERROR: boom" in ui._result_line(ToolResultEvent(**base, error="boom"))
    assert "skipped (nope)" in ui._result_line(
        ToolResultEvent(**base, skipped=True, skip_reason="nope")
    )
    assert "cancelled" in ui._result_line(ToolResultEvent(**base, cancelled=True))
    assert ui._result_line(ToolResultEvent(**base)) == "[tool] bash: done"


# ----------------------------------------------------------------------
# Dispatch and commands
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exit_command_stops_loop_without_agent_turn(ui: ConsoleUI) -> None:
    await ui._dispatch("/exit")
    assert ui._running is False
    assert ui.agent_loop.acted_prompts == []


@pytest.mark.asyncio
async def test_unknown_command_not_sent_to_agent(ui: ConsoleUI, capsys) -> None:
    await ui._dispatch("/definitely-not-a-command")
    assert ui.agent_loop.acted_prompts == []
    assert "Unknown command" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_help_command_prints_help(ui: ConsoleUI, capsys) -> None:
    await ui._dispatch("/help")
    out = capsys.readouterr().out
    assert "/model" in out
    assert ui.agent_loop.acted_prompts == []


@pytest.mark.asyncio
async def test_plain_message_goes_to_agent(ui: ConsoleUI) -> None:
    ui.agent_loop.act_events = [AssistantEvent(content="hi")]
    await ui._dispatch("hello there")
    assert ui.agent_loop.acted_prompts == ["hello there"]


@pytest.mark.asyncio
async def test_slash_skill_sends_skill_content(ui: ConsoleUI, tmp_path) -> None:
    skill_file = tmp_path / "greet.md"
    skill_file.write_text("Greet the user warmly.")
    skill_info = SimpleNamespace(skill_path=skill_file)
    ui.agent_loop.skill_manager = SimpleNamespace(
        get_skill=lambda name: skill_info if name == "greet" else None
    )

    await ui._dispatch("/greet loudly")

    assert len(ui.agent_loop.acted_prompts) == 1
    prompt = ui.agent_loop.acted_prompts[0]
    assert "Greet the user warmly." in prompt
    assert "/greet loudly" in prompt


# ----------------------------------------------------------------------
# Approval callback
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approval_yes(ui: ConsoleUI) -> None:
    queue_replies(ui, ["y"])
    response, feedback = await ui._approval_callback("bash", ToolArgs(), "t1", None)
    assert response is ApprovalResponse.YES
    assert feedback is None


@pytest.mark.asyncio
async def test_approval_always_registers_session_rule(ui: ConsoleUI) -> None:
    queue_replies(ui, ["a"])
    response, _ = await ui._approval_callback("bash", ToolArgs(), "t1", None)
    assert response is ApprovalResponse.YES
    assert ui.agent_loop.approved_always == [("bash", None)]


@pytest.mark.asyncio
async def test_approval_no_with_feedback(ui: ConsoleUI) -> None:
    queue_replies(ui, ["n", "use ls instead"])
    response, feedback = await ui._approval_callback("bash", ToolArgs(), "t1", None)
    assert response is ApprovalResponse.NO
    assert feedback == "use ls instead"


@pytest.mark.asyncio
async def test_approval_no_without_feedback_uses_cancellation_text(
    ui: ConsoleUI,
) -> None:
    queue_replies(ui, ["n", ""])
    response, feedback = await ui._approval_callback("bash", ToolArgs(), "t1", None)
    assert response is ApprovalResponse.NO
    assert feedback
    assert "cancelled" in feedback.lower()


@pytest.mark.asyncio
async def test_approval_reprompts_on_garbage(ui: ConsoleUI) -> None:
    queue_replies(ui, ["what", "y"])
    response, _ = await ui._approval_callback("bash", ToolArgs(), "t1", None)
    assert response is ApprovalResponse.YES


@pytest.mark.asyncio
async def test_approval_auto_approve_skips_prompt(ui: ConsoleUI) -> None:
    ui.agent_loop.config.auto_approve = True
    ui.agent_loop.tool_manager.available_tools = {"bash": object}
    # No replies queued: a prompt would raise IndexError.
    response, feedback = await ui._approval_callback("bash", ToolArgs(), "t1", None)
    assert response is ApprovalResponse.YES
    assert feedback is None


# ----------------------------------------------------------------------
# ask_user_question callback
# ----------------------------------------------------------------------


def one_question(**overrides: Any) -> AskUserQuestionArgs:
    question = Question(
        question="Pick a database",
        options=[
            Choice(label="Postgres", description="relational"),
            Choice(label="Redis"),
        ],
        **overrides,
    )
    return AskUserQuestionArgs(questions=[question])


@pytest.mark.asyncio
async def test_question_single_choice(ui: ConsoleUI) -> None:
    queue_replies(ui, ["1"])
    result = await ui._user_input_callback(one_question())
    assert isinstance(result, AskUserQuestionResult)
    assert not result.cancelled
    assert result.answers[0].answer == "Postgres"
    assert result.answers[0].is_other is False


@pytest.mark.asyncio
async def test_question_other_free_text(ui: ConsoleUI) -> None:
    queue_replies(ui, ["3", "SQLite"])
    result = await ui._user_input_callback(one_question())
    assert result.answers[0].answer == "SQLite"
    assert result.answers[0].is_other is True


@pytest.mark.asyncio
async def test_question_multi_select(ui: ConsoleUI) -> None:
    queue_replies(ui, ["1,2"])
    result = await ui._user_input_callback(one_question(multi_select=True))
    assert result.answers[0].answer == "Postgres, Redis"


@pytest.mark.asyncio
async def test_question_cancel(ui: ConsoleUI) -> None:
    queue_replies(ui, ["c"])
    result = await ui._user_input_callback(one_question())
    assert result.cancelled is True
    assert result.answers == []


@pytest.mark.asyncio
async def test_question_invalid_then_valid(ui: ConsoleUI) -> None:
    queue_replies(ui, ["0", "zebra", "2"])
    result = await ui._user_input_callback(one_question())
    assert result.answers[0].answer == "Redis"


@pytest.mark.asyncio
async def test_question_hide_other_rejects_extra_index(ui: ConsoleUI) -> None:
    queue_replies(ui, ["3", "1"])
    result = await ui._user_input_callback(one_question(hide_other=True))
    assert result.answers[0].answer == "Postgres"


def test_parse_picks() -> None:
    parse = ConsoleUI._parse_picks
    assert parse("2", max_index=3, multi=False) == [2]
    assert parse("1,3", max_index=3, multi=True) == [1, 3]
    assert parse("1,1", max_index=3, multi=True) == [1]
    assert parse("1,2", max_index=3, multi=False) is None
    assert parse("4", max_index=3, multi=True) is None
    assert parse("x", max_index=3, multi=False) is None
    assert parse("", max_index=3, multi=False) is None


# ----------------------------------------------------------------------
# Tool result bodies (TUI-parity previews)
# ----------------------------------------------------------------------


class FakeBashResult(BaseModel):
    command: str = "x"
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0


class FakeGrepResult(BaseModel):
    matches: str = ""


def result_event(tool_name: str, result: BaseModel | None, **kwargs: Any):
    return ToolResultEvent(
        tool_name=tool_name,
        tool_class=None,
        tool_call_id="t1",
        result=result,
        **kwargs,
    )


def test_bash_stdout_preview_truncates_to_config(ui: ConsoleUI, capsys) -> None:
    ui.agent_loop.config.tool_result_preview_lines = 2
    stdout = "line1\nline2\nline3\nline4"
    ui._handle_event(result_event("bash", FakeBashResult(stdout=stdout)))
    out = capsys.readouterr().out
    assert "  line1" in out
    assert "  line2" in out
    assert "line3" not in out
    assert "(2 more lines)" in out


def test_bash_empty_stdout_shows_no_content(ui: ConsoleUI, capsys) -> None:
    ui._handle_event(result_event("bash", FakeBashResult(stdout="")))
    assert "(no content)" in capsys.readouterr().out


def test_grep_matches_preview(ui: ConsoleUI, capsys) -> None:
    ui.agent_loop.config.tool_result_preview_lines = 3
    ui._handle_event(result_event("grep", FakeGrepResult(matches="a\nb")))
    out = capsys.readouterr().out
    assert "  a" in out
    assert "  b" in out


def test_error_and_skip_print_no_body(ui: ConsoleUI, capsys) -> None:
    ui._handle_event(
        result_event("bash", FakeBashResult(stdout="hidden"), error="boom")
    )
    ui._handle_event(
        result_event("bash", FakeBashResult(stdout="hidden"), skipped=True)
    )
    assert "hidden" not in capsys.readouterr().out


def test_read_file_shows_no_body(ui: ConsoleUI, capsys) -> None:
    class FakeReadResult(BaseModel):
        path: str = "a.py"
        content: str = "secret"

    ui._handle_event(result_event("read_file", FakeReadResult()))
    assert "secret" not in capsys.readouterr().out


def test_file_diff_rows_with_hunk_budget(ui: ConsoleUI, capsys) -> None:
    from privibe.core.types import FileDiff

    ui.agent_loop.config.tool_result_preview_lines = 2
    diff = FileDiff(
        path="a.py",
        kind="diff",
        hunks=[
            [("diff-removed", "-old1"), ("diff-added", "+new1")],
            [("diff-added", "+new2")],
            [("diff-added", "+new3")],
        ],
    )
    ui._handle_event(
        result_event("write_file", FakeBashResult(), file_diff=diff)
    )
    out = capsys.readouterr().out
    assert "  -old1" in out
    assert "  +new1" in out
    assert "  +new2" in out
    assert "+new3" not in out
    assert "(1 more change not shown)" in out


def test_binary_file_diff_prints_note(ui: ConsoleUI, capsys) -> None:
    from privibe.core.types import FileDiff

    diff = FileDiff(path="a.bin", kind="binary", note="Binary file - no diff shown")
    ui._handle_event(result_event("write_file", None, file_diff=diff))
    assert "Binary file - no diff shown" in capsys.readouterr().out


def test_search_replace_body_renders_diff_lines(ui: ConsoleUI, capsys) -> None:
    class FakeSearchReplaceResult(BaseModel):
        content: str

    content = (
        "<<<<<<< SEARCH\nold line\n=======\nnew line\n>>>>>>> REPLACE\n"
    )
    ui._handle_event(
        result_event("search_replace", FakeSearchReplaceResult(content=content))
    )
    out = capsys.readouterr().out
    assert "-old line" in out
    assert "+new line" in out


def test_todo_lines_ordering_and_marks() -> None:
    todos = [
        SimpleNamespace(status="completed", content="done thing"),
        SimpleNamespace(status="in_progress", content="current thing"),
        SimpleNamespace(status="pending", content="next thing"),
    ]
    lines = ConsoleUI._todo_lines(todos)
    assert lines == [
        "[>] current thing",
        "[ ] next thing",
        "[x] done thing",
    ]


# ----------------------------------------------------------------------
# Transcript replay
# ----------------------------------------------------------------------


def make_history() -> list[LLMMessage]:
    return [
        LLMMessage(role=Role.system, content="system prompt"),
        LLMMessage(role=Role.user, content="first question"),
        LLMMessage(
            role=Role.assistant,
            content="let me check",
            tool_calls=[
                ToolCall(id="t1", function=FunctionCall(name="bash", arguments="{}"))
            ],
        ),
        LLMMessage(role=Role.tool, content="huge tool output", name="bash"),
        LLMMessage(role=Role.assistant, content="the answer"),
        LLMMessage(role=Role.user, content="injected note", injected=True),
        LLMMessage(
            role=Role.user,
            content="<context_refresh>refresh</context_refresh>",
        ),
    ]


def test_transcript_filters_and_markers(ui: ConsoleUI) -> None:
    ui.agent_loop.messages = make_history()
    entries = ui._transcript_entries(None)
    assert entries == [
        "you: first question",
        "let me check",
        "[tool] bash",
        "the answer",
    ]


def test_transcript_cap_keeps_tail_with_header(ui: ConsoleUI) -> None:
    ui.agent_loop.messages = make_history()
    entries = ui._transcript_entries(2)
    assert entries == [
        "(... 2 earlier entries not shown; /replay shows all)",
        "[tool] bash",
        "the answer",
    ]


def test_print_transcript_output(ui: ConsoleUI, capsys) -> None:
    ui.agent_loop.messages = make_history()
    ui._print_transcript(None)
    out = capsys.readouterr().out
    assert "--- resumed conversation ---" in out
    assert "you: first question" in out
    assert "huge tool output" not in out
    assert "--- end of history ---" in out


def test_print_transcript_silent_when_empty(ui: ConsoleUI, capsys) -> None:
    ui.agent_loop.messages = [LLMMessage(role=Role.system, content="system prompt")]
    ui._print_transcript(None)
    assert capsys.readouterr().out == ""


@pytest.mark.asyncio
async def test_replay_command_prints_everything(ui: ConsoleUI, capsys) -> None:
    ui.agent_loop.messages = make_history()
    handled = await ui._handle_command("/replay")
    out = capsys.readouterr().out
    assert handled is True
    assert "you: first question" in out
    assert "not shown" not in out


@pytest.mark.asyncio
async def test_replay_command_without_history(ui: ConsoleUI, capsys) -> None:
    ui.agent_loop.messages = []
    handled = await ui._handle_command("/replay")
    assert handled is True
    assert "No conversation history." in capsys.readouterr().out


@pytest.mark.asyncio
async def test_preview_lines_command_cycles_and_persists(
    ui: ConsoleUI, capsys, monkeypatch
) -> None:
    import privibe.cli.console_ui.app as console_app

    saved: dict = {}
    monkeypatch.setattr(
        console_app.VibeConfig,
        "save_updates",
        classmethod(lambda cls, updates: saved.update(updates)),
    )
    ui.agent_loop.config.tool_result_preview_lines = 3
    ui.agent_loop.config.tool_result_preview_options = [3, 5, 10]
    ui.agent_loop.refresh_config = lambda: None

    handled = await ui._handle_command("/preview-lines")

    assert handled is True
    assert saved == {"tool_result_preview_lines": 5}
    assert "5 lines" in capsys.readouterr().out
