"""-p and its session: the id on stderr, how the run ended, the resume line,
and a resume that continues the session where it lives.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import sys
import threading
from types import SimpleNamespace

import pytest

from privibe.core import programmatic
from privibe.core.agents.models import BuiltinAgentName
from privibe.core.config import SessionLoggingConfig, VibeConfig
from privibe.core.programmatic import (
    ProgrammaticRunTerminated,
    _run_and_announce,
    resume_command,
    run_programmatic,
)
from privibe.core.session.session_loader import SessionLoader
from privibe.core.utils import ConversationLimitException
from tests.conftest import build_test_vibe_config
from tests.mock.mock_backend_factory import mock_backend_factory
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


@contextmanager
def every_backend_is(make_backend):
    def factory(provider, **kwargs):
        return make_backend()

    with mock_backend_factory("mistral", factory), mock_backend_factory(
        "generic", factory
    ):
        yield


def answering(text: str = "ok"):
    return lambda: FakeBackend(mock_llm_chunk(content=text))


def logged_config(save_dir: Path, **overrides) -> VibeConfig:
    return build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(save_dir)),
        system_prompt_id="tests",
        include_project_context=False,
        include_prompt_detail=False,
        include_model_info=False,
        include_commit_signature=False,
        **overrides,
    )


def session_dirs(save_dir: Path) -> list[Path]:
    return sorted(p for p in save_dir.iterdir() if p.is_dir())


def stored_messages(session_dir: Path) -> list[dict]:
    lines = (session_dir / "messages.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def short_id(session_dir: Path) -> str:
    return SessionLoader.load_metadata(session_dir).session_id[:8]


# ---------------------------------------------------------------------------
# The id, at the start and at the end
# ---------------------------------------------------------------------------


def test_a_finished_run_names_its_session_on_stderr_and_keeps_stdout_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    with every_backend_is(answering("the answer")):
        result = run_programmatic(config=logged_config(tmp_path), prompt="hi")

    (session_dir,) = session_dirs(tmp_path)
    ref = short_id(session_dir)
    captured = capsys.readouterr()
    assert result == "the answer"
    assert captured.err.splitlines() == [
        f"[privibe] session {ref}",
        f"[privibe] session {ref} finished",
    ]
    assert "[privibe]" not in captured.out
    assert "--resume" not in captured.err


def test_the_id_is_on_stderr_before_the_first_request_goes_out(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    seen_at_request_time: list[str] = []

    class PeekingBackend(FakeBackend):
        async def complete(self, **kwargs):
            seen_at_request_time.append(capsys.readouterr().err)
            return await super().complete(**kwargs)

    with every_backend_is(lambda: PeekingBackend(mock_llm_chunk(content="ok"))):
        run_programmatic(config=logged_config(tmp_path), prompt="hi")

    assert seen_at_request_time[0].startswith("[privibe] session ")


def test_nothing_is_announced_when_session_logging_is_off(
    capsys: pytest.CaptureFixture,
) -> None:
    with every_backend_is(answering()):
        run_programmatic(config=build_test_vibe_config(), prompt="hi")

    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# A run that did not finish
# ---------------------------------------------------------------------------


def test_a_run_whose_model_dies_prints_the_resume_command(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    dead = lambda: FakeBackend(exception_to_raise=RuntimeError("connection refused"))

    with every_backend_is(dead), pytest.raises(Exception, match="connection refused"):
        run_programmatic(
            config=logged_config(tmp_path),
            prompt="hi",
            resume_flags=["--nanny", "sitter"],
        )

    (session_dir,) = session_dirs(tmp_path)
    ref = short_id(session_dir)
    last_line = capsys.readouterr().err.splitlines()[-1]
    assert last_line.startswith(f"[privibe] session {ref} did not finish (failed: ")
    assert last_line.endswith(
        f"To continue it: privibe --resume {ref} --nanny sitter -p"
    )
    # What it had before dying is on disk, so the command has something to load.
    assert stored_messages(session_dir)[0]["content"] == "hi"


def test_sigterm_saves_the_session_and_prints_the_resume_command(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    if sys.platform == "win32" or threading.current_thread() is not threading.main_thread():
        pytest.skip("needs a signal handler on the event loop")

    class HangingBackend(FakeBackend):
        async def complete(self, **kwargs):
            # The run is now inside the request: the handler is installed.
            threading.Timer(0.05, os.kill, (os.getpid(), signal.SIGTERM)).start()
            await asyncio.sleep(8)
            raise AssertionError("SIGTERM never arrived")

    with every_backend_is(HangingBackend), pytest.raises(ProgrammaticRunTerminated) as excinfo:
        run_programmatic(config=logged_config(tmp_path), prompt="hi")

    assert excinfo.value.exit_code == 143
    (session_dir,) = session_dirs(tmp_path)
    ref = short_id(session_dir)
    assert capsys.readouterr().err.splitlines()[-1] == (
        f"[privibe] session {ref} did not finish (terminated by SIGTERM). "
        f"To continue it: privibe --resume {ref} -p"
    )
    assert stored_messages(session_dir)[0]["content"] == "hi"


def stub_loop(tmp_path: Path, *, enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        session_logger=SimpleNamespace(
            enabled=enabled,
            session_id="abcdef12-0000-0000-0000-000000000000",
            session_config=SessionLoggingConfig(enabled=enabled, save_dir=str(tmp_path)),
        )
    )


def raising(exc: BaseException):
    async def run() -> str | None:
        raise exc

    return run


@pytest.fixture
def session_on_disk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        programmatic.SessionLoader,
        "does_session_exist",
        staticmethod(lambda session_id, config: tmp_path),
    )


@pytest.mark.parametrize(
    ("exc", "raised", "problem"),
    [
        (KeyboardInterrupt(), KeyboardInterrupt, "interrupted"),
        (
            ConversationLimitException("turn limit"),
            ConversationLimitException,
            "stopped at a limit",
        ),
        (RuntimeError("boom"), RuntimeError, "failed: RuntimeError"),
    ],
)
def test_every_unfinished_ending_says_why_and_how_to_go_on(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
    session_on_disk: None,
    exc: BaseException,
    raised: type[BaseException],
    problem: str,
) -> None:
    with pytest.raises(raised):
        _run_and_announce(raising(exc), stub_loop(tmp_path), [], ["--agent", "plan"])

    assert capsys.readouterr().err == (
        f"[privibe] session abcdef12 did not finish ({problem}). "
        "To continue it: privibe --resume abcdef12 --agent plan -p\n"
    )


def test_a_cancellation_that_was_sigterm_becomes_a_termination(
    tmp_path: Path, capsys: pytest.CaptureFixture, session_on_disk: None
) -> None:
    with pytest.raises(ProgrammaticRunTerminated):
        _run_and_announce(
            raising(asyncio.CancelledError()), stub_loop(tmp_path), [True], []
        )

    assert "terminated by SIGTERM" in capsys.readouterr().err


def test_a_cancellation_that_was_not_sigterm_is_left_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture, session_on_disk: None
) -> None:
    with pytest.raises(asyncio.CancelledError):
        _run_and_announce(raising(asyncio.CancelledError()), stub_loop(tmp_path), [], [])

    assert capsys.readouterr().err == ""


def test_a_run_that_died_before_anything_was_saved_offers_no_resume(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    with pytest.raises(RuntimeError):
        _run_and_announce(raising(RuntimeError("boom")), stub_loop(tmp_path), [], [])

    err = capsys.readouterr().err
    assert "nothing to resume" in err
    assert "--resume" not in err


def test_resume_command_quotes_what_a_shell_would_split() -> None:
    command = resume_command(
        "abcdef12", ["--nanny", "sitter", "--nanny-brief", "rm in build/ is fine"]
    )

    assert command == (
        "privibe --resume abcdef12 --nanny sitter "
        "--nanny-brief 'rm in build/ is fine' -p"
    )


# ---------------------------------------------------------------------------
# Resume continues the session where it lives
# ---------------------------------------------------------------------------


def test_resume_appends_to_the_same_session_instead_of_copying_it(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    with every_backend_is(answering("first answer")):
        run_programmatic(config=logged_config(tmp_path), prompt="first")
    (session_dir,) = session_dirs(tmp_path)
    ref = short_id(session_dir)
    before = stored_messages(session_dir)
    capsys.readouterr()

    with every_backend_is(answering("second answer")):
        run_programmatic(
            config=logged_config(tmp_path), prompt="continue", session_path=session_dir
        )

    assert session_dirs(tmp_path) == [session_dir]
    after = stored_messages(session_dir)
    assert after[: len(before)] == before
    assert [m["content"] for m in after if m["role"] == "assistant"] == [
        "first answer",
        "second answer",
    ]
    assert after[-2]["content"] == "continue"
    assert capsys.readouterr().err.splitlines() == [
        f"[privibe] session {ref} (resumed)",
        f"[privibe] session {ref} finished",
    ]


def two_model_config(save_dir: Path, active: str) -> VibeConfig:
    base = build_test_vibe_config().get_active_model()
    models = [
        base.model_copy(update={"alias": "first-model"}),
        base.model_copy(update={"alias": "second-model"}),
    ]
    return logged_config(save_dir, models=models, active_model=active)


def saved_model(session_dir: Path) -> str | None:
    return SessionLoader.load_metadata(session_dir).active_model


def test_resume_goes_on_with_the_model_the_session_was_running_on(
    tmp_path: Path,
) -> None:
    with every_backend_is(answering()):
        run_programmatic(config=two_model_config(tmp_path, "first-model"), prompt="a")
        (session_dir,) = session_dirs(tmp_path)
        assert saved_model(session_dir) == "first-model"

        # config.toml moved on to another model in the meantime.
        run_programmatic(
            config=two_model_config(tmp_path, "second-model"),
            prompt="continue",
            session_path=session_dir,
        )

    assert saved_model(session_dir) == "first-model"


def test_resume_with_an_explicit_model_keeps_the_explicit_model(
    tmp_path: Path,
) -> None:
    with every_backend_is(answering()):
        run_programmatic(config=two_model_config(tmp_path, "first-model"), prompt="a")
        (session_dir,) = session_dirs(tmp_path)

        run_programmatic(
            config=two_model_config(tmp_path, "second-model"),
            prompt="continue",
            session_path=session_dir,
            pinned=frozenset({"model"}),
        )

    assert saved_model(session_dir) == "second-model"


# ---------------------------------------------------------------------------
# The nanny of a resumed session
# ---------------------------------------------------------------------------


class RecordingNanny:
    model_alias = "sitter"

    def __init__(self) -> None:
        self.resumed_with: list[tuple[Path, str]] = []

    def resume(self, session_dir: Path, session_id: str) -> bool:
        self.resumed_with.append((session_dir, session_id))
        return True

    def note_user_prompt(self, *args, **kwargs) -> None: ...

    def note_outcome(self, *args, **kwargs) -> None: ...

    async def close(self) -> None: ...


@pytest.fixture
def recording_nanny(monkeypatch: pytest.MonkeyPatch) -> RecordingNanny:
    from privibe.core.nanny import NannyManager

    nanny = RecordingNanny()
    monkeypatch.setattr(
        NannyManager, "for_loop", classmethod(lambda cls, loop, alias=None: nanny)
    )
    return nanny


def test_a_resumed_run_reloads_the_nannys_own_conversation(
    tmp_path: Path, recording_nanny: RecordingNanny
) -> None:
    with every_backend_is(answering()):
        run_programmatic(config=logged_config(tmp_path), prompt="first")
        (session_dir,) = session_dirs(tmp_path)

        run_programmatic(
            config=logged_config(tmp_path),
            prompt="continue",
            session_path=session_dir,
            nanny_alias="sitter",
        )

    session_id = SessionLoader.load_metadata(session_dir).session_id
    assert recording_nanny.resumed_with == [(session_dir, session_id)]
    # And the choice is on file for the next resume to find.
    assert SessionLoader.load_metadata(session_dir).nanny_model == "sitter"


def test_a_fresh_run_has_no_nanny_conversation_to_reload(
    tmp_path: Path, recording_nanny: RecordingNanny
) -> None:
    with every_backend_is(answering()):
        run_programmatic(
            config=logged_config(tmp_path), prompt="first", nanny_alias="sitter"
        )

    assert recording_nanny.resumed_with == []


# ---------------------------------------------------------------------------
# The agent profile of a resumed session
# ---------------------------------------------------------------------------


def saved_agent(session_dir: Path) -> str | None:
    return SessionLoader.load_metadata(session_dir).agent_name


def test_resume_goes_on_with_the_agent_profile_the_session_was_running_on(
    tmp_path: Path,
) -> None:
    with every_backend_is(answering()):
        run_programmatic(
            config=logged_config(tmp_path),
            prompt="a",
            agent_name=BuiltinAgentName.DEFAULT,
        )
        (session_dir,) = session_dirs(tmp_path)
        assert saved_agent(session_dir) == BuiltinAgentName.DEFAULT

        # A bare `-p --resume` starts on auto-approve; the session must not.
        run_programmatic(
            config=logged_config(tmp_path),
            prompt="continue",
            session_path=session_dir,
            agent_name=BuiltinAgentName.AUTO_APPROVE,
        )

    assert saved_agent(session_dir) == BuiltinAgentName.DEFAULT


def test_resume_with_an_explicit_agent_keeps_the_explicit_agent(
    tmp_path: Path,
) -> None:
    with every_backend_is(answering()):
        run_programmatic(
            config=logged_config(tmp_path),
            prompt="a",
            agent_name=BuiltinAgentName.DEFAULT,
        )
        (session_dir,) = session_dirs(tmp_path)

        run_programmatic(
            config=logged_config(tmp_path),
            prompt="continue",
            session_path=session_dir,
            agent_name=BuiltinAgentName.AUTO_APPROVE,
            pinned=frozenset({"agent"}),
        )

    assert saved_agent(session_dir) == BuiltinAgentName.AUTO_APPROVE


# ---------------------------------------------------------------------------
# End to end: a nanny-reviewed run, stopped, resumed with a bare -p --resume
# ---------------------------------------------------------------------------


def test_a_resumed_nanny_reviewed_run_goes_on_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from privibe.core.nanny import NannyManager
    from privibe.core.tools.base import ToolPermission
    from privibe.core.types import FunctionCall, ToolCall
    from tests.conftest import build_test_agent_loop
    from tests.nanny.conftest import APPROVE, READY, REFUSE, nanny_backend

    def worker(call_id: str):
        call = ToolCall(
            id=call_id,
            index=0,
            function=FunctionCall(name="todo", arguments='{"action": "read"}'),
        )
        return lambda: FakeBackend([
            [mock_llm_chunk(content="Checking the list.", tool_calls=[call])],
            [mock_llm_chunk(content="Done.")],
        ])

    def config() -> VibeConfig:
        return logged_config(
            tmp_path / "sessions",
            enabled_tools=["todo"],
            tools={"todo": {"permission": ToolPermission.ASK.value}},
        )

    # The nanny model is scripted too; everything else about it is real:
    # its manager, its transcript and decisions stored beside the session.
    nanny_backends: list[FakeBackend] = []
    real_for_loop = NannyManager.for_loop.__func__

    def for_loop(cls, loop, alias=None, **kwargs):
        def config_loader(**overrides) -> VibeConfig:
            return build_test_vibe_config(
                nanny_model=alias, include_project_context=False, **overrides
            )

        def loop_factory(nanny_config: VibeConfig):
            return build_test_agent_loop(
                config=nanny_config,
                agent_name=BuiltinAgentName.NANNY,
                backend=nanny_backends[-1],
            )

        return real_for_loop(
            cls, loop, alias, config_loader=config_loader, loop_factory=loop_factory
        )

    monkeypatch.setattr(NannyManager, "for_loop", classmethod(for_loop))
    alias = build_test_vibe_config().active_model

    # Run 1: started with --nanny; the nanny gets its header, approves a call.
    nanny_backends.append(nanny_backend([READY, APPROVE]))
    with every_backend_is(worker("call_1")):
        run_programmatic(
            config=config(),
            prompt="what is on my list?",
            agent_name=BuiltinAgentName.DEFAULT,
            nanny_alias=alias,
        )
    (session_dir,) = session_dirs(tmp_path / "sessions")
    metadata = SessionLoader.load_metadata(session_dir)
    assert metadata.nanny_model == alias
    first_run_nanny_turns = len(nanny_backends[0].requests_messages)
    assert first_run_nanny_turns == 2  # header, then the one ask

    # Run 2: what `privibe --resume <id> -p` does. No --nanny, no --agent:
    # the CLI reads the nanny from meta.json, the agent comes back with the
    # session. This time the nanny refuses.
    nanny_backends.append(nanny_backend([REFUSE]))
    with every_backend_is(worker("call_2")):
        run_programmatic(
            config=config(),
            prompt="continue",
            session_path=session_dir,
            agent_name=BuiltinAgentName.AUTO_APPROVE,
            nanny_alias=metadata.nanny_model,
        )

    # Same session, still asking, still reviewed by the same nanny.
    assert session_dirs(tmp_path / "sessions") == [session_dir]
    after = SessionLoader.load_metadata(session_dir)
    assert after.agent_name == BuiltinAgentName.DEFAULT
    assert after.nanny_model == alias

    # The nanny was asked once more, with no second header, and that ask
    # arrived on top of everything it had seen before the stop.
    (resumed_ask,) = nanny_backends[1].requests_messages
    earlier = nanny_backends[0].requests_messages[-1]
    assert [m.content for m in resumed_ask[: len(earlier)]] == [
        m.content for m in earlier
    ]
    assert len(resumed_ask) > len(earlier)

    # And its refusal reached the worker: the second call did not run.
    tool_results = [m for m in stored_messages(session_dir) if m["role"] == "tool"]
    assert len(tool_results) == 2
    assert "deletes a directory outside the project" in tool_results[1]["content"]
    assert "deletes a directory outside the project" not in tool_results[0]["content"]
