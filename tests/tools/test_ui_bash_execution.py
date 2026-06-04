from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest
from textual.widgets import Static

from privibe.cli.textual_ui.app import VibeApp
from privibe.cli.textual_ui.widgets.chat_input.container import ChatInputContainer
from privibe.cli.textual_ui.widgets.messages import BashOutputMessage, ErrorMessage


async def _wait_for_bash_output_message(
    vibe_app: VibeApp, pilot, timeout: float = 1.0
) -> BashOutputMessage:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if message := next(iter(vibe_app.query(BashOutputMessage)), None):
            return message
        await pilot.pause(0.05)
    raise TimeoutError(f"BashOutputMessage did not appear within {timeout}s")


def assert_no_command_error(vibe_app: VibeApp) -> None:
    errors = list(vibe_app.query(ErrorMessage))
    if not errors:
        return

    disallowed = {
        "Command failed",
        "Command timed out",
        "No command provided after '!'",
    }
    offending = [
        getattr(err, "_error", "")
        for err in errors
        if getattr(err, "_error", "")
        and any(phrase in getattr(err, "_error", "") for phrase in disallowed)
    ]
    assert not offending, f"Unexpected command errors: {offending}"


@pytest.mark.asyncio
async def test_ui_reports_no_output(vibe_app: VibeApp) -> None:
    async with vibe_app.run_test() as pilot:
        chat_input = vibe_app.query_one(ChatInputContainer)
        chat_input.value = "!true"

        await pilot.press("enter")
        message = await _wait_for_bash_output_message(vibe_app, pilot)
        output_widget = message.query_one(".bash-output", Static)
        assert str(output_widget.render()) == "(no output)"
        assert_no_command_error(vibe_app)


@pytest.mark.asyncio
async def test_ui_shows_success_in_case_of_zero_code(vibe_app: VibeApp) -> None:
    async with vibe_app.run_test() as pilot:
        chat_input = vibe_app.query_one(ChatInputContainer)
        chat_input.value = "!true"

        await pilot.press("enter")
        message = await _wait_for_bash_output_message(vibe_app, pilot)
        assert message.has_class("bash-success")
        assert not message.has_class("bash-error")


@pytest.mark.asyncio
async def test_ui_shows_failure_in_case_of_non_zero_code(vibe_app: VibeApp) -> None:
    async with vibe_app.run_test() as pilot:
        chat_input = vibe_app.query_one(ChatInputContainer)
        chat_input.value = "!bash -lc 'exit 7'"

        await pilot.press("enter")
        message = await _wait_for_bash_output_message(vibe_app, pilot)
        assert message.has_class("bash-error")
        assert not message.has_class("bash-success")


@pytest.mark.asyncio
async def test_ui_handles_non_utf8_output(vibe_app: VibeApp) -> None:
    """Assert the UI accepts decoding a non-UTF8 sequence like `printf '\xf0\x9f\x98'`.
    Whereas `printf '\xf0\x9f\x98\x8b'` prints a smiley face (😋) and would work even without those changes.
    """
    async with vibe_app.run_test() as pilot:
        chat_input = vibe_app.query_one(ChatInputContainer)
        chat_input.value = "!printf '\\xff\\xfe'"

        await pilot.press("enter")
        message = await _wait_for_bash_output_message(vibe_app, pilot)
        output_widget = message.query_one(".bash-output", Static)
        # accept both possible encodings, as some shells emit escaped bytes as literal strings
        assert str(output_widget.render()) in {"��", "\xff\xfe", r"\xff\xfe"}
        assert_no_command_error(vibe_app)


@pytest.mark.asyncio
async def test_ui_handles_utf8_output(vibe_app: VibeApp) -> None:
    async with vibe_app.run_test() as pilot:
        chat_input = vibe_app.query_one(ChatInputContainer)
        chat_input.value = "!echo hello"

        await pilot.press("enter")
        message = await _wait_for_bash_output_message(vibe_app, pilot)
        output_widget = message.query_one(".bash-output", Static)
        assert str(output_widget.render()) == "hello"
        assert_no_command_error(vibe_app)


@pytest.mark.asyncio
async def test_ui_handles_non_utf8_stderr(vibe_app: VibeApp) -> None:
    async with vibe_app.run_test() as pilot:
        chat_input = vibe_app.query_one(ChatInputContainer)
        chat_input.value = "!bash -lc \"printf '\\\\xff\\\\xfe' 1>&2\""

        await pilot.press("enter")
        message = await _wait_for_bash_output_message(vibe_app, pilot)
        output_widget = message.query_one(".bash-output", Static)
        assert str(output_widget.render()) == "��"
        assert_no_command_error(vibe_app)


async def _wait_for_error_message(
    vibe_app: VibeApp, pilot, timeout: float = 2.0
) -> ErrorMessage:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if message := next(iter(vibe_app.query(ErrorMessage)), None):
            return message
        await pilot.pause(0.05)
    raise TimeoutError(f"ErrorMessage did not appear within {timeout}s")


@pytest.mark.asyncio
async def test_ui_reports_timeout_error(vibe_app: VibeApp) -> None:
    """Verify the timeout path shows an error and kills the process."""
    import asyncio
    from unittest.mock import MagicMock

    real_wait_for = asyncio.wait_for

    async def _hanging_communicate():
        await asyncio.sleep(9999)
        return b"", b""

    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.communicate = _hanging_communicate
    mock_proc.kill = MagicMock()
    mock_proc.wait = AsyncMock(return_value=None)

    async def _patched_wait_for(coro, timeout=None, **kw):
        # Intercept only the wait_for wrapping our fake communicate coroutine.
        coro_name = getattr(getattr(coro, "cr_code", None), "co_name", "")
        if coro_name == "_hanging_communicate":
            coro.close()
            raise TimeoutError
        return await real_wait_for(coro, timeout=timeout, **kw)

    with patch("asyncio.create_subprocess_shell", new=AsyncMock(return_value=mock_proc)):
        with patch("asyncio.wait_for", side_effect=_patched_wait_for):
            async with vibe_app.run_test() as pilot:
                chat_input = vibe_app.query_one(ChatInputContainer)
                chat_input.value = "!echo test"

                await pilot.press("enter")
                error = await _wait_for_error_message(vibe_app, pilot)
                assert "timed out" in getattr(error, "_error", "").lower()
                mock_proc.kill.assert_called_once()


@pytest.mark.asyncio
async def test_ui_uses_async_subprocess(vibe_app: VibeApp) -> None:
    """Verify _handle_bash_command uses asyncio.create_subprocess_shell, not subprocess.run."""
    import asyncio

    calls: list[str] = []
    original = asyncio.create_subprocess_shell

    async def _spy(*args, **kwargs):
        calls.append(args[0])
        return await original(*args, **kwargs)

    with patch("asyncio.create_subprocess_shell", side_effect=_spy):
        async with vibe_app.run_test() as pilot:
            chat_input = vibe_app.query_one(ChatInputContainer)
            chat_input.value = "!echo privibe"

            await pilot.press("enter")
            await _wait_for_bash_output_message(vibe_app, pilot)

    assert any("echo privibe" in c for c in calls)
