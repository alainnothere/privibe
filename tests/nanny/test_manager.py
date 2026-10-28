from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from privibe.core.nanny import (
    REFUSAL_PREFIX,
    NannyAskRequest,
    NannyCallPresentation,
    NannyConfigError,
    NannyListener,
    NannyManager,
    NannyState,
    NannyVerdict,
    decisions_for,
    read_decisions,
)
from privibe.core.types import ApprovalResponse, Role
from tests.conftest import build_test_vibe_config
from tests.nanny.conftest import APPROVE, READY, REFUSE, build_nanny, nanny_backend


def ask(
    call_id: str = "call_1", text: str = "bash command:\nls", full: str | None = None
) -> NannyAskRequest:
    return NannyAskRequest(
        tool_name="bash",
        tool_call_id=call_id,
        agent="default",
        presentation=NannyCallPresentation(text=text, full_text=full),
        worker_text="Let me list the files.",
    )


def roles(nanny: NannyManager) -> list[Role]:
    return [m.role for m in nanny.loop.messages]


@pytest.mark.asyncio
async def test_header_then_ask_then_approve(tmp_path: Path) -> None:
    nanny = build_nanny(tmp_path, [READY, APPROVE])
    async with nanny.review(ask()) as review:
        assert review.approved
        assert review.reason == "runs the project's own tooling"
    # system, header, READY, ask, verdict: strictly alternating after system.
    assert roles(nanny) == [
        Role.system,
        Role.user,
        Role.assistant,
        Role.user,
        Role.assistant,
    ]
    backend = nanny.loop.backend
    header = backend.requests_messages[0][1].content
    assert "main-session-1234" in header
    ask_text = backend.requests_messages[1][3].content
    assert "bash command:\nls" in ask_text
    assert "Let me list the files." in ask_text
    assert 'from agent "default"' in ask_text
    assert nanny.loop.messages[0].content.startswith("You are the nanny")


@pytest.mark.asyncio
async def test_preamble_is_drained_into_the_next_ask(tmp_path: Path) -> None:
    nanny = build_nanny(tmp_path, [READY, APPROVE, APPROVE])
    nanny.note_user_prompt("please clean the build dir", "default")
    async with nanny.review(ask("c1")):
        pass
    nanny.note_outcome("c1", "bash", "success", "default")
    nanny.note_user_prompt("now run tests", "default", steering=True)
    async with nanny.review(ask("c2")):
        pass
    requests = nanny.loop.backend.requests_messages
    first_ask = requests[1][-1].content
    second_ask = requests[2][-1].content
    assert "please clean the build dir" in first_ask
    assert "Outcome: bash call c1" in second_ask
    assert "Steering message" in second_ask and "now run tests" in second_ask
    assert "please clean the build dir" not in second_ask


@pytest.mark.asyncio
async def test_refusal_reaches_the_block_and_is_recorded(tmp_path: Path) -> None:
    nanny = build_nanny(tmp_path, [READY, REFUSE])
    seen: list = []

    class Listener(NannyListener):
        def on_nanny_decision(self, decision):
            seen.append(decision)

    nanny.add_listener(Listener())
    async with nanny.review(ask()) as review:
        assert not review.approved
        assert (
            review.refusal_feedback()
            == f"{REFUSAL_PREFIX}deletes a directory outside the project"
        )
        review.human_choice = "refuse"
        review.human_decided(ApprovalResponse.NO, review.refusal_feedback())
    assert len(seen) == 1
    decision = seen[0]
    assert decision.verdict == "refuse"
    assert decision.human_decision == "refuse"
    assert decision.ask_index == 3 and decision.verdict_index == 4
    stored = read_decisions(nanny.decisions_path)
    assert [d.tool_call_id for d in stored] == ["call_1"]
    assert decisions_for(tmp_path / "session")["call_1"].reason == decision.reason
    # The next ask carries the human's decision.
    assert any("Human decision" in line for line in nanny._preamble)


@pytest.mark.asyncio
async def test_talk_rounds_alternate_and_are_recorded(tmp_path: Path) -> None:
    nanny = build_nanny(
        tmp_path,
        [
            READY,
            REFUSE,
            "Understood: only this once.",
            "VERDICT: refuse\nREASON: still",
        ],
    )
    states: list[NannyState] = []

    class Listener(NannyListener):
        def on_nanny_state_change(self, state):
            states.append(state)

    nanny.add_listener(Listener())
    async with nanny.review(ask()) as review:
        reply = await review.talk(
            "this is allowed just this once because it is the temp dir"
        )
        assert reply == "Understood: only this once."
        review.human_choice = "allow"
        review.human_decided(ApprovalResponse.YES, None)
    assert roles(nanny)[1:] == [Role.user, Role.assistant] * 3
    talk = nanny.decisions[-1].talk
    assert len(talk) == 1 and talk[0].human.startswith("this is allowed")
    assert NannyState.TALKING in states and states[-1] is NannyState.IDLE
    talk_prompt = nanny.loop.backend.requests_messages[2][-1].content
    assert "Message from the human" in talk_prompt


@pytest.mark.asyncio
async def test_talk_outside_a_review_is_refused(tmp_path: Path) -> None:
    nanny = build_nanny(tmp_path, [READY, REFUSE])
    async with nanny.review(ask()) as review:
        pass
    with pytest.raises(RuntimeError):
        await review.talk("hello?")


@pytest.mark.asyncio
async def test_more_fetches_full_content_once(tmp_path: Path) -> None:
    nanny = build_nanny(tmp_path, [READY, "VERDICT: more\nREASON: show me", APPROVE])
    async with nanny.review(
        ask(text="write_file: x (shortened)", full="FULL CONTENT HERE")
    ) as review:
        assert review.approved
    more_prompt = nanny.loop.backend.requests_messages[2][-1].content
    assert "FULL CONTENT HERE" in more_prompt


@pytest.mark.asyncio
async def test_more_with_nothing_more_is_a_refusal(tmp_path: Path) -> None:
    nanny = build_nanny(tmp_path, [READY, "VERDICT: more\nREASON: show me"])
    async with nanny.review(ask()) as review:
        assert not review.approved
        assert review.answer.verdict is NannyVerdict.REFUSE


@pytest.mark.asyncio
async def test_unreadable_answer_is_a_refusal_with_notice(tmp_path: Path) -> None:
    # Two unreadable replies: the first earns one retry, the second is final.
    nanny = build_nanny(tmp_path, [READY, "sure go ahead", "still sure"])
    async with nanny.review(ask()) as review:
        assert not review.approved
        assert not review.failed
        assert review.notice and "could not be read" in review.notice


@pytest.mark.asyncio
async def test_backend_error_falls_back_to_the_human(tmp_path: Path) -> None:
    nanny = build_nanny(
        tmp_path, backend=nanny_backend([], exception=RuntimeError("boom"))
    )
    async with nanny.review(ask()) as review:
        assert review.failed
        assert not review.approved
        assert "boom" in review.notice
    # The dangling header ask was closed with a stand-in assistant turn so the
    # next ask does not follow a user message.
    assert roles(nanny)[-1] is Role.assistant
    assert nanny.decisions[-1].failed


@pytest.mark.asyncio
async def test_timeout_falls_back_to_the_human(tmp_path: Path) -> None:
    import tests.stubs.fake_backend as fb

    class SlowBackend(fb.FakeBackend):
        async def complete(self, **kwargs):
            await asyncio.sleep(5)
            return await super().complete(**kwargs)

    nanny = build_nanny(tmp_path, backend=SlowBackend([]), timeout=0.05)
    async with nanny.review(ask()) as review:
        assert review.failed
        assert "no answer within" in review.notice
    assert roles(nanny)[-1] is Role.assistant


@pytest.mark.asyncio
async def test_reviews_are_serialised_including_the_human_part(tmp_path: Path) -> None:
    nanny = build_nanny(tmp_path, [READY, REFUSE, APPROVE])
    order: list[str] = []
    human_done = asyncio.Event()

    async def first() -> None:
        async with nanny.review(ask("a")) as review:
            order.append("a-asked")
            assert not review.approved
            await human_done.wait()
            order.append("a-decided")

    async def second() -> None:
        await asyncio.sleep(0)
        async with nanny.review(ask("b")):
            order.append("b-asked")

    t1 = asyncio.create_task(first())
    t2 = asyncio.create_task(second())
    await asyncio.sleep(0.05)
    assert order == ["a-asked"]
    human_done.set()
    await asyncio.gather(t1, t2)
    assert order == ["a-asked", "a-decided", "b-asked"]


@pytest.mark.asyncio
async def test_wait_idle_waits_for_the_review(tmp_path: Path) -> None:
    nanny = build_nanny(tmp_path, [READY, REFUSE])
    release = asyncio.Event()
    idle_returned = asyncio.Event()

    async def review() -> None:
        async with nanny.review(ask()):
            await release.wait()

    async def waiter() -> None:
        await asyncio.sleep(0)
        await nanny.wait_idle()
        idle_returned.set()

    tasks = [asyncio.create_task(review()), asyncio.create_task(waiter())]
    await asyncio.sleep(0.05)
    assert nanny.busy and not idle_returned.is_set()
    release.set()
    await asyncio.gather(*tasks)
    assert idle_returned.is_set() and not nanny.busy


def test_unknown_alias_is_a_config_error(tmp_path: Path) -> None:
    def loader(**overrides):
        overrides.pop("session_logging", None)
        return build_test_vibe_config(nanny_model="ghost", **overrides)

    with pytest.raises(NannyConfigError):
        NannyManager(
            config=build_test_vibe_config(nanny_model="ghost"),
            session_dir=None,
            main_session_id="x",
            profile_name="default",
            config_loader=loader,
        )


def test_no_alias_is_a_config_error() -> None:
    with pytest.raises(NannyConfigError):
        NannyManager(
            config=build_test_vibe_config(),
            session_dir=None,
            main_session_id="x",
            profile_name="default",
        )


@pytest.mark.asyncio
async def test_session_reset_moves_the_sidecar(tmp_path: Path) -> None:
    nanny = build_nanny(tmp_path, [READY, APPROVE])
    async with nanny.review(ask()):
        pass
    new_dir = tmp_path / "session-2"
    nanny.on_main_session_reset(new_dir, "main-session-5678")
    assert nanny.decisions_path == new_dir / "nanny" / "decisions.jsonl"
    assert [d.tool_call_id for d in read_decisions(nanny.decisions_path)] == ["call_1"]
    assert nanny.main_session_id == "main-session-5678"


# --- thinking models: reasoning is read, one retry on an unreadable answer ---


@pytest.mark.asyncio
async def test_reason_in_the_reasoning_channel_is_read(tmp_path: Path) -> None:
    from tests.mock.utils import mock_llm_chunk
    from privibe.core.nanny.verdict import NO_REASON
    from tests.stubs.fake_backend import FakeBackend

    backend = FakeBackend([
        [mock_llm_chunk(content=READY)],
        [
            mock_llm_chunk(
                content="VERDICT: approve\nREASON:",
                reasoning_content="Approve read-only strings on the mounted log.\n",
            )
        ],
    ])
    nanny = build_nanny(tmp_path, backend=backend)
    async with nanny.review(ask()) as review:
        assert review.approved
        assert review.reason == "Approve read-only strings on the mounted log."
        assert review.reason != NO_REASON
    assert nanny.decisions[-1].reason == "Approve read-only strings on the mounted log."


@pytest.mark.asyncio
async def test_verdict_only_in_the_reasoning_is_read(tmp_path: Path) -> None:
    from tests.mock.utils import mock_llm_chunk
    from tests.stubs.fake_backend import FakeBackend

    backend = FakeBackend([
        [mock_llm_chunk(content=READY)],
        [mock_llm_chunk(content="VERDICT", reasoning_content="Refuse: deletes /etc.\n")],
    ])
    nanny = build_nanny(tmp_path, backend=backend)
    async with nanny.review(ask()) as review:
        assert not review.approved
        assert review.answer.readable
        assert review.reason == "deletes /etc."
    # No retry was needed: header, READY, ask, verdict.
    assert len(nanny.loop.messages) == 5


@pytest.mark.asyncio
async def test_reasoning_only_reply_is_not_an_empty_answer(tmp_path: Path) -> None:
    from tests.mock.utils import mock_llm_chunk
    from tests.stubs.fake_backend import FakeBackend

    backend = FakeBackend([
        [mock_llm_chunk(content=READY)],
        [mock_llm_chunk(content="", reasoning_content="Approve. Lists files.")],
    ])
    nanny = build_nanny(tmp_path, backend=backend)
    async with nanny.review(ask()) as review:
        assert review.approved and not review.failed
        assert review.reason == "Lists files."


@pytest.mark.asyncio
async def test_unreadable_answer_gets_one_retry(tmp_path: Path) -> None:
    nanny = build_nanny(tmp_path, [READY, "hmm, let me see", APPROVE])
    async with nanny.review(ask()) as review:
        assert review.approved
        assert review.answer.readable
    # header, READY, ask, junk, retry nudge, verdict.
    assert len(nanny.loop.messages) == 7
    nudge = nanny.loop.messages[5].content
    assert "no verdict I could read" in nudge
    assert nanny.decisions[-1].verdict_index == 6


@pytest.mark.asyncio
async def test_retry_still_unreadable_falls_back_to_the_human(tmp_path: Path) -> None:
    nanny = build_nanny(tmp_path, [READY, "hmm", "still hmm", APPROVE])
    async with nanny.review(ask()) as review:
        assert not review.approved
        assert not review.answer.readable
        assert review.notice == "nanny answer could not be read; asking you instead"
    assert len(nanny.loop.messages) == 7
