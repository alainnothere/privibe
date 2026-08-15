from __future__ import annotations

from pathlib import Path

from tests.conftest import build_test_vibe_config


class TestExtraInstructionFilesExpansion:
    def test_tilde_expands_to_home(self) -> None:
        config = build_test_vibe_config(
            extra_instruction_files=["~/privibeInstructions.md"]
        )
        assert config.extra_instruction_files == [
            Path.home() / "privibeInstructions.md"
        ]

    def test_absolute_path_unchanged(self) -> None:
        config = build_test_vibe_config(
            extra_instruction_files=["/home/someone/notes.md"]
        )
        assert config.extra_instruction_files == [Path("/home/someone/notes.md")]

    def test_relative_path_stays_relative(self) -> None:
        """Relative paths resolve against the effective cwd at use time
        (load_extra_instruction_files), so the validator must not pin them
        to the config-load cwd.
        """
        config = build_test_vibe_config(extra_instruction_files=["./notes.md"])
        assert not config.extra_instruction_files[0].is_absolute()

    def test_empty_list(self) -> None:
        config = build_test_vibe_config(extra_instruction_files=[])
        assert config.extra_instruction_files == []


class TestAgentPathsExpansion:
    def test_tilde_expands_to_home(self) -> None:
        config = build_test_vibe_config(agent_paths=["~/myAgents"])
        assert config.agent_paths == [Path.home() / "myAgents"]

    def test_relative_path_stays_relative(self) -> None:
        config = build_test_vibe_config(agent_paths=["./agents"])
        assert not config.agent_paths[0].is_absolute()
