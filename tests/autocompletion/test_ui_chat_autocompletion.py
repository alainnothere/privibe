from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
import time

import pytest
from textual.content import Content
from textual.pilot import Pilot
from textual.style import Style
from textual.widgets import Markdown

from privibe.cli.textual_ui.app import VibeApp
from privibe.cli.textual_ui.widgets.chat_input.completion_popup import CompletionPopup
from privibe.cli.textual_ui.widgets.chat_input.container import ChatInputContainer

# Under a fully parallel run (-n auto) the xdist workers oversubscribe the CPU
# and a worker's event loop can stall for ~10 s at test start, which delays
# pilot.press key delivery by the same amount. The waits below and this module
# timeout are sized to survive that; the default 10 s pytest timeout is not.
pytestmark = pytest.mark.timeout(60)


@pytest.mark.asyncio
async def test_popup_appears_with_matching_suggestions(vibe_app: VibeApp) -> None:
    async with start_app(vibe_app) as pilot:
        chat_input = vibe_app.query_one(ChatInputContainer)
        popup = vibe_app.query_one(CompletionPopup)

        await pilot.press(*"/com")

        popup_content = str(popup.render())
        assert popup.styles.display == "block"
        assert "/compact" in popup_content
        assert "Compact conversation history by summarizing" in popup_content
        assert chat_input.value == "/com"


@pytest.mark.asyncio
async def test_popup_hides_when_input_cleared(vibe_app: VibeApp) -> None:
    async with start_app(vibe_app) as pilot:
        popup = vibe_app.query_one(CompletionPopup)

        await pilot.press(*"/c")
        await pilot.press("backspace", "backspace")

        assert popup.styles.display == "none"


@pytest.mark.asyncio
async def test_pressing_tab_writes_selected_command_and_keeps_popup_visible(
    vibe_app: VibeApp,
) -> None:
    async with start_app(vibe_app) as pilot:
        chat_input = vibe_app.query_one(ChatInputContainer)
        popup = vibe_app.query_one(CompletionPopup)

        await pilot.press(*"/co")
        await pilot.press("tab")

        assert chat_input.value == "/compact"
        assert popup.styles.display == "block"


def ensure_selected_command(popup: CompletionPopup, expected_alias: str) -> None:
    renderable = popup.render()
    assert isinstance(renderable, Content)
    content = str(renderable)

    selected_aliases: list[str] = []
    for span in renderable.spans:
        style = span.style
        if isinstance(style, Style) and style.reverse:
            alias_text = content[span.start : span.end].strip()
            alias = alias_text.split()[0] if alias_text else ""
            selected_aliases.append(alias)

    assert len(selected_aliases) == 1
    assert selected_aliases[0] == expected_alias


@pytest.mark.asyncio
async def test_arrow_navigation_updates_selected_suggestion(vibe_app: VibeApp) -> None:
    async with start_app(vibe_app) as pilot:
        popup = vibe_app.query_one(CompletionPopup)

        await pilot.press(*"/c")

        ensure_selected_command(popup, "/clear")
        await pilot.press("down")
        ensure_selected_command(popup, "/compact")
        await pilot.press("up")
        ensure_selected_command(popup, "/clear")


@pytest.mark.asyncio
async def test_arrow_navigation_cycles_through_suggestions(vibe_app: VibeApp) -> None:
    async with start_app(vibe_app) as pilot:
        popup = vibe_app.query_one(CompletionPopup)

        await pilot.press(*"/co")

        ensure_selected_command(popup, "/compact")
        await pilot.press("down")
        ensure_selected_command(popup, "/config")
        await pilot.press("up")
        ensure_selected_command(popup, "/compact")


@pytest.mark.asyncio
async def test_pressing_enter_submits_selected_command_and_hides_popup(
    vibe_app: VibeApp,
) -> None:
    async with start_app(vibe_app) as pilot:
        chat_input = vibe_app.query_one(ChatInputContainer)
        popup = vibe_app.query_one(CompletionPopup)

        await pilot.press(*"/hel")  # typos:disable-line
        await pilot.press("enter")

        assert chat_input.value == ""
        assert popup.styles.display == "none"
        # The instruction-files preamble (mounted in on_mount) is also a
        # .user-command-message, so query_one would grab the wrong widget.
        # Pick the most recently mounted one — that's what /hel just produced.
        message = list(vibe_app.query(".user-command-message"))[-1]
        message_content = message.query_one(Markdown)
        assert "Show help message" in message_content.source



@pytest.fixture()
def file_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "src" / "utils").mkdir(parents=True)
    (tmp_path / "src" / "utils" / "config.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "utils" / "database.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "utils" / "error_handling.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "utils" / "logger.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "utils" / "sanitize.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "utils" / "validate.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "main.py").write_text("", encoding="utf-8")
    (tmp_path / "privibe" / "acp").mkdir(parents=True)
    (tmp_path / "privibe" / "acp" / "entrypoint.py").write_text("", encoding="utf-8")
    (tmp_path / "privibe" / "acp" / "agent.py").write_text("", encoding="utf-8")
    (tmp_path / "README.md").write_text("", encoding="utf-8")
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@asynccontextmanager
async def start_app(vibe_app: VibeApp, timeout: float = 20.0) -> AsyncIterator[Pilot]:
    """run_test, but only yield once the chat input has received focus.

    The input is focused during app startup (focus_input in on_mount); under
    heavy load run_test can yield before that lands, and keys pressed while
    nothing is focused are silently dropped, leaving the input empty.
    """
    async with vibe_app.run_test() as pilot:
        deadline = time.monotonic() + timeout
        while pilot.app.focused is None:
            if time.monotonic() > deadline:
                raise AssertionError("Chat input never received focus.")
            await pilot.pause(0.05)
        yield pilot


async def wait_for_popup_content(
    pilot: Pilot, popup: CompletionPopup, *needles: str, timeout: float = 20.0
) -> str:
    """Wait until every needle is rendered in the completion popup.

    Path completions are computed on PathCompletionController's worker thread
    and land via call_after_refresh, so the popup fills asynchronously after
    the keystrokes. Asserting on popup.render() immediately is a race that
    loses under CPU load (e.g. a full parallel test run).
    """
    deadline = time.monotonic() + timeout
    while True:
        content = str(popup.render())
        if all(needle in content for needle in needles):
            return content
        if time.monotonic() > deadline:
            input_value = popup.app.query_one(ChatInputContainer).value
            raise AssertionError(
                f"Completion popup never rendered {needles!r}."
                f" Last content: {content!r}; input text: {input_value!r};"
                f" focused: {popup.app.focused!r}"
            )
        await pilot.pause(0.05)


@pytest.mark.asyncio
async def test_path_completion_popup_lists_files_and_directories(
    vibe_app: VibeApp, file_tree: Path
) -> None:
    async with start_app(vibe_app) as pilot:
        popup = vibe_app.query_one(CompletionPopup)

        await pilot.press(*"@s")

        await wait_for_popup_content(pilot, popup, "src/")
        assert popup.styles.display == "block"


@pytest.mark.asyncio
async def test_path_completion_popup_shows_up_to_ten_results(
    vibe_app: VibeApp, file_tree: Path
) -> None:
    async with start_app(vibe_app) as pilot:
        (file_tree / "src" / "core" / "extra").mkdir(parents=True)
        [
            (file_tree / "src" / "core" / "extra" / f"extra_file_{i}.py").write_text(
                "", encoding="utf-8"
            )
            for i in range(1, 13)
        ]
        popup = vibe_app.query_one(CompletionPopup)

        await pilot.press(*"@src/core/extra/")

        await wait_for_popup_content(
            pilot,
            popup,
            "src/core/extra/extra_file_1.py",
            "src/core/extra/extra_file_10.py",
            "src/core/extra/extra_file_11.py",
            "src/core/extra/extra_file_12.py",
            "src/core/extra/extra_file_2.py",
            "src/core/extra/extra_file_3.py",
            "src/core/extra/extra_file_4.py",
            "src/core/extra/extra_file_5.py",
            "src/core/extra/extra_file_6.py",
            "src/core/extra/extra_file_7.py",
        )
        assert popup.styles.display == "block"


@pytest.mark.asyncio
async def test_pressing_tab_writes_selected_path_name_and_hides_popup(
    vibe_app: VibeApp, file_tree: Path
) -> None:
    async with start_app(vibe_app) as pilot:
        chat_input = vibe_app.query_one(ChatInputContainer)
        popup = vibe_app.query_one(CompletionPopup)

        await pilot.press(*"Print @REA")
        await wait_for_popup_content(pilot, popup, "README.md")
        await pilot.press("tab")

        assert chat_input.value == "Print @README.md "
        assert popup.styles.display == "none"


@pytest.mark.asyncio
async def test_pressing_enter_writes_selected_path_name_and_hides_popup(
    vibe_app: VibeApp, file_tree: Path
) -> None:
    async with start_app(vibe_app) as pilot:
        chat_input = vibe_app.query_one(ChatInputContainer)
        popup = vibe_app.query_one(CompletionPopup)

        await pilot.press(*"Print @src/m")
        await wait_for_popup_content(pilot, popup, "src/main.py")
        await pilot.press("enter")

        assert chat_input.value == "Print @src/main.py "
        assert popup.styles.display == "none"


@pytest.mark.asyncio
async def test_fuzzy_matches_subsequence_characters(
    file_tree: Path, vibe_app: VibeApp
) -> None:
    async with start_app(vibe_app) as pilot:
        popup = vibe_app.query_one(CompletionPopup)

        await pilot.press(*"@src/utils/handling")

        await wait_for_popup_content(pilot, popup, "src/utils/error_handling.py")
        assert popup.styles.display == "block"


@pytest.mark.asyncio
async def test_fuzzy_matches_word_boundaries(
    file_tree: Path, vibe_app: VibeApp
) -> None:
    async with start_app(vibe_app) as pilot:
        popup = vibe_app.query_one(CompletionPopup)

        await pilot.press(*"@src/utils/eh")

        await wait_for_popup_content(pilot, popup, "src/utils/error_handling.py")
        assert popup.styles.display == "block"


@pytest.mark.asyncio
async def test_finds_files_recursively_by_filename(
    file_tree: Path, vibe_app: VibeApp
) -> None:
    async with start_app(vibe_app) as pilot:
        popup = vibe_app.query_one(CompletionPopup)

        await pilot.press(*"@entryp")

        await wait_for_popup_content(pilot, popup, "privibe/acp/entrypoint.py")
        assert popup.styles.display == "block"


@pytest.mark.asyncio
async def test_finds_files_recursively_with_partial_path(
    file_tree: Path, vibe_app: VibeApp
) -> None:
    async with start_app(vibe_app) as pilot:
        popup = vibe_app.query_one(CompletionPopup)

        await pilot.press(*"@acp/entry")

        await wait_for_popup_content(pilot, popup, "privibe/acp/entrypoint.py")
        assert popup.styles.display == "block"


@pytest.mark.asyncio
async def test_does_not_trigger_completion_when_navigating_history(
    file_tree: Path, vibe_app: VibeApp
) -> None:
    async with start_app(vibe_app) as pilot:
        chat_input = vibe_app.query_one(ChatInputContainer)
        popup = vibe_app.query_one(CompletionPopup)
        message_with_path = "Check @src/m"
        message_to_fill_history = "Yet another message to fill history"

        await pilot.press(*message_with_path)
        await wait_for_popup_content(pilot, popup, "src/main.py")
        await pilot.press("tab", "enter")
        await pilot.press(*message_to_fill_history)
        await pilot.press("enter")
        await pilot.press("up", "up")
        assert chat_input.value == "Check @src/main.py"
        await pilot.pause(0.2)
        # ensure popup is hidden - user was navigating history: we don't want to interrupt
        assert popup.styles.display == "none"
        await pilot.press("down")
        await pilot.pause(0.1)
        assert popup.styles.display == "none"
        # get back to the message with path completion; ensure again
        await pilot.press("up")
        await pilot.pause(0.1)
        assert chat_input.value == "Check @src/main.py"
        await pilot.pause(0.2)
        assert popup.styles.display == "none"
