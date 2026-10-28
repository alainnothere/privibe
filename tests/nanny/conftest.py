from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from privibe.core.agents.models import BuiltinAgentName
from privibe.core.config import SessionLoggingConfig, VibeConfig
from privibe.core.nanny import NannyManager
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

# A scripted nanny: the first reply answers the session header, the rest
# answer asks (and talk rounds) in order.
READY = "READY"
APPROVE = "VERDICT: approve\nREASON: runs the project's own tooling"
REFUSE = "VERDICT: refuse\nREASON: deletes a directory outside the project"


def nanny_backend(
    replies: Sequence[str], *, exception: Exception | None = None
) -> FakeBackend:
    return FakeBackend(
        [[mock_llm_chunk(content=text)] for text in replies],
        exception_to_raise=exception,
    )


def build_nanny(
    tmp_path: Path,
    replies: Sequence[str] = (READY, APPROVE),
    *,
    backend: FakeBackend | None = None,
    session_dir: Path | None | str = "default",
    timeout: float = 5.0,
    main_session_id: str = "main-session-1234",
    brief: str | None = None,
) -> NannyManager:
    resolved_backend = backend or nanny_backend(replies)
    resolved_dir = tmp_path / "session" if session_dir == "default" else session_dir

    def config_loader(**overrides) -> VibeConfig:
        overrides.setdefault("session_logging", SessionLoggingConfig(enabled=False))
        return build_test_vibe_config(
            nanny_model="devstral-latest",
            nanny_timeout_seconds=timeout,
            nanny_brief=brief,
            include_project_context=False,
            **overrides,
        )

    def loop_factory(config: VibeConfig):
        return build_test_agent_loop(
            config=config, agent_name=BuiltinAgentName.NANNY, backend=resolved_backend
        )

    return NannyManager(
        config=config_loader(),
        session_dir=resolved_dir,
        main_session_id=main_session_id,
        profile_name="default",
        cwd=tmp_path,
        loop_factory=loop_factory,
        config_loader=config_loader,
    )
