from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e.common import (
    SpawnedVibeProcessFixture,
    ansi_tolerant_pattern,
    send_exit_sequence,
    wait_for_main_screen,
    wait_for_request_count,
)
from tests.e2e.mock_server import StreamingMockServer


@pytest.mark.timeout(15)
@pytest.mark.xdist_group("e2e")
def test_spawn_cli_to_send_and_receive_message(
    streaming_mock_server: StreamingMockServer,
    setup_e2e_env: None,
    e2e_workdir: Path,
    spawned_vibe_process: SpawnedVibeProcessFixture,
) -> None:
    """Send a message to the CLI and verify the mock LLM response is rendered.

    Quirks and known failure modes:

    1. EXIT IS A CTRL+C HANDSHAKE
       The TUI (privibe/cli/textual_ui/app.py:action_clear_quit) consumes a
       Ctrl+C press to interrupt a running agent or clear input text; only an
       idle app starts the double-press exit countdown.  send_exit_sequence
       presses Ctrl+C until the "Press Ctrl+C again to exit" toast renders,
       then confirms and waits for EOF.

    2. XDIST WORKERS EXPORT COLUMNS/LINES
       pytest-xdist workers run with COLUMNS=80 and LINES=24 in os.environ,
       and Textual prefers those env vars over the real pty size, so a
       spawned TUI inheriting them lays out at 80x24 and never renders the
       banner ("Privibe v").  The spawn fixture (tests/e2e/conftest.py)
       strips them (plus proxy vars, which would misroute the request to the
       localhost mock server), so these tests also pass under "-n auto".
       The xdist_group("e2e") marker plus "--dist loadgroup" in addopts keeps
       the spawn-heavy e2e tests serialized on a single worker.

    3. TIMEOUT IS TIGHT (15 s)
       The test must complete within 15 s: ~5 s for CLI startup + ~10 s for
       the LLM round-trip.  If the mock server or uv spawn is slow the test
       will hit the pytest-timeout fence.
    """
    with spawned_vibe_process(e2e_workdir) as (child, captured):
        wait_for_main_screen(child, timeout=15)
        child.send("Greet")
        child.send("\r")

        wait_for_request_count(
            lambda: len(streaming_mock_server.requests), expected_count=1, timeout=10
        )
        child.expect(ansi_tolerant_pattern("Hello from mock server"), timeout=10)

        send_exit_sequence(child)

    output = captured.getvalue()
    assert "Welcome to Privibe" not in output

    request_payload = streaming_mock_server.requests[-1]
    assert request_payload.get("stream") is True
    assert request_payload.get("model") == "mock-model"
    stream_options = request_payload.get("stream_options")
    assert stream_options is not None
    assert stream_options.get("include_usage") is True
    messages = request_payload.get("messages")
    assert messages is not None
    assert any(
        message.get("role") == "user" and message.get("content") == "Greet"
        for message in messages
    )
