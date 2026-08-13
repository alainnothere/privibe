from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e.common import (
    SpawnedVibeProcessFixture,
    wait_for_main_screen,
    wait_for_rendered_text,
    wait_for_request_count,
)
from tests.e2e.mock_server import ChatCompletionsRequestPayload, StreamingMockServer

PREDICTABLE_OUTPUT = "__E2E_BASH_OK__"
TOOL_ARGUMENTS = f'{{"command":"printf \\"{PREDICTABLE_OUTPUT}\\\\n\\""}}'


def _tool_call_factory(
    request_index: int, _payload: ChatCompletionsRequestPayload
) -> list[dict[str, object]]:
    if request_index == 0:
        return [
            StreamingMockServer.build_chunk(
                created=1,
                delta=StreamingMockServer.build_tool_call_delta(
                    call_id="call_bash_1", tool_name="bash", arguments=TOOL_ARGUMENTS
                ),
                finish_reason=None,
            ),
            StreamingMockServer.build_chunk(
                created=2,
                delta={},
                finish_reason="tool_calls",
                usage={"prompt_tokens": 3, "completion_tokens": 4},
            ),
        ]

    return [
        StreamingMockServer.build_chunk(
            created=3,
            delta={
                "role": "assistant",
                "content": f"The string {PREDICTABLE_OUTPUT} has been printed successfully.",
            },
            finish_reason=None,
        ),
        StreamingMockServer.build_chunk(
            created=4, delta={"content": PREDICTABLE_OUTPUT}, finish_reason=None
        ),
        StreamingMockServer.build_chunk(
            created=5,
            delta={},
            finish_reason="stop",
            usage={"prompt_tokens": 3, "completion_tokens": 4},
        ),
    ]


@pytest.mark.timeout(60)
@pytest.mark.xdist_group("e2e")
@pytest.mark.parametrize(
    "streaming_mock_server",
    [pytest.param(_tool_call_factory, id="tool-call-stream")],
    indirect=True,
)
def test_spawn_cli_asks_bash_permission_and_shows_tool_output_after_approval(
    streaming_mock_server: StreamingMockServer,
    setup_e2e_env: None,
    e2e_workdir: Path,
    spawned_vibe_process: SpawnedVibeProcessFixture,
) -> None:
    """Test that the CLI prompts for bash permission and shows tool output after approval.

    Quirks and known failure modes:

    1. XDIST WORKERS EXPORT COLUMNS/LINES
       See test_cli_tui_streaming.py -- the spawn fixture strips COLUMNS/LINES
       and proxy vars from the child env, and xdist_group("e2e") plus
       "--dist loadgroup" keeps the e2e tests serialized on one worker.

    2. NO EOF WAIT AFTER TOOL OUTPUT
       After the tool runs, the agent makes a second LLM request to summarize
       the result.  Waiting for EOF here is unreliable -- the agent may still
       be processing.  The test asserts only that the tool output appears, then
       exits via the context manager without waiting for EOF.

    3. TIMEOUT IS 60 s
       Longer than the streaming test because it covers two LLM round-trips
       (tool call + follow-up) plus user interaction for permission.
    """
    with spawned_vibe_process(e2e_workdir) as (child, captured):
        wait_for_main_screen(child, timeout=15)
        child.send("Run a shell command")
        child.send("\r")

        wait_for_request_count(
            lambda: len(streaming_mock_server.requests), expected_count=1, timeout=10
        )
        wait_for_rendered_text(child, captured, needle="bash command", timeout=10)
        child.send("y")
        child.send("\r")
        wait_for_rendered_text(child, captured, needle=PREDICTABLE_OUTPUT, timeout=10)
