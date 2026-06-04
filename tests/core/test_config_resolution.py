from __future__ import annotations

from pathlib import Path
import tomllib

import pytest
import tomli_w

from tests.conftest import build_test_vibe_config
from privibe.core.config import ModelConfig, ProviderConfig, VibeConfig
from privibe.core.config.harness_files import (
    HarnessFilesManager,
    init_harness_files_manager,
    reset_harness_files_manager,
)
from privibe.core.paths import VIBE_HOME
from privibe.core.trusted_folders import trusted_folders_manager


class TestResolveConfigFile:
    def test_resolves_local_config_when_exists_and_folder_is_trusted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        local_config_dir = tmp_path / ".privibe"
        local_config_dir.mkdir()
        local_config = local_config_dir / "config.toml"
        local_config.write_text('active_model = "test"', encoding="utf-8")

        monkeypatch.setattr(trusted_folders_manager, "is_trusted", lambda _: True)

        reset_harness_files_manager()
        init_harness_files_manager("user", "project")
        from privibe.core.config.harness_files import get_harness_files_manager

        mgr = get_harness_files_manager()
        resolved = mgr.config_file
        assert resolved is not None
        assert resolved == local_config
        assert resolved.is_file()
        assert resolved.read_text(encoding="utf-8") == 'active_model = "test"'

    def test_resolves_global_config_when_folder_is_not_trusted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        local_config_dir = tmp_path / ".privibe"
        local_config_dir.mkdir()
        local_config = local_config_dir / "config.toml"
        local_config.write_text('active_model = "test"', encoding="utf-8")

        reset_harness_files_manager()
        init_harness_files_manager("user", "project")
        from privibe.core.config.harness_files import get_harness_files_manager

        mgr = get_harness_files_manager()
        assert mgr.config_file == VIBE_HOME.path / "config.toml"

    def test_falls_back_to_global_config_when_local_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        # Ensure no local config exists
        assert not (tmp_path / ".privibe" / "config.toml").exists()

        reset_harness_files_manager()
        init_harness_files_manager("user", "project")
        from privibe.core.config.harness_files import get_harness_files_manager

        mgr = get_harness_files_manager()
        assert mgr.config_file == VIBE_HOME.path / "config.toml"

    def test_respects_vibe_home_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert VIBE_HOME.path != tmp_path
        monkeypatch.setenv("VIBE_HOME", str(tmp_path))
        assert VIBE_HOME.path == tmp_path

    def test_returns_none_when_no_sources(self) -> None:
        mgr = HarnessFilesManager(sources=())
        assert mgr.config_file is None

    def test_user_only_returns_global_config(self) -> None:
        mgr = HarnessFilesManager(sources=("user",))
        assert mgr.config_file == VIBE_HOME.path / "config.toml"


class TestMigrateRemovesFindFromBashAllowlist:
    def test_removes_find_from_config_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VIBE_HOME", str(tmp_path))
        config_file = tmp_path / "config.toml"
        data = {"tools": {"bash": {"allowlist": ["echo", "find", "ls"]}}}
        with config_file.open("wb") as f:
            tomli_w.dump(data, f)

        reset_harness_files_manager()
        init_harness_files_manager("user")
        VibeConfig._migrate()

        with config_file.open("rb") as f:
            result = tomllib.load(f)
        assert "find" not in result["tools"]["bash"]["allowlist"]
        assert result["tools"]["bash"]["allowlist"] == ["echo", "ls"]

    def test_noop_when_find_not_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VIBE_HOME", str(tmp_path))
        config_file = tmp_path / "config.toml"
        data = {"tools": {"bash": {"allowlist": ["echo", "ls"]}}}
        with config_file.open("wb") as f:
            tomli_w.dump(data, f)

        reset_harness_files_manager()
        init_harness_files_manager("user")
        VibeConfig._migrate()

        with config_file.open("rb") as f:
            result = tomllib.load(f)
        assert result["tools"]["bash"]["allowlist"] == ["echo", "ls"]

    def test_noop_when_no_bash_tools_section(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VIBE_HOME", str(tmp_path))
        config_file = tmp_path / "config.toml"
        data = {"active_model": "test"}
        with config_file.open("wb") as f:
            tomli_w.dump(data, f)

        reset_harness_files_manager()
        init_harness_files_manager("user")
        VibeConfig._migrate()

        with config_file.open("rb") as f:
            result = tomllib.load(f)
        assert "tools" not in result


class TestAutoCompactThresholdFallback:
    def test_model_without_explicit_threshold_inherits_global(self) -> None:
        model = ModelConfig(name="m", provider="mistral", alias="m")
        cfg = build_test_vibe_config(
            auto_compact_threshold=42_000, models=[model], active_model="m"
        )
        assert cfg.get_active_model().auto_compact_threshold == 42_000

    def test_model_with_explicit_threshold_keeps_own_value(self) -> None:
        model = ModelConfig(
            name="m", provider="mistral", alias="m", auto_compact_threshold=99_000
        )
        cfg = build_test_vibe_config(
            auto_compact_threshold=42_000, models=[model], active_model="m"
        )
        assert cfg.get_active_model().auto_compact_threshold == 99_000

    def test_default_global_threshold_used_when_nothing_set(self) -> None:
        model = ModelConfig(name="m", provider="mistral", alias="m")
        cfg = build_test_vibe_config(models=[model], active_model="m")
        assert cfg.get_active_model().auto_compact_threshold == 200_000

    def test_changed_global_threshold_propagates_on_reload(self) -> None:
        model = ModelConfig(name="m", provider="mistral", alias="m")

        cfg1 = build_test_vibe_config(
            auto_compact_threshold=50_000, models=[model], active_model="m"
        )
        assert cfg1.get_active_model().auto_compact_threshold == 50_000

        # Simulate config reload with a different global threshold
        cfg2 = build_test_vibe_config(
            auto_compact_threshold=75_000, models=[model], active_model="m"
        )
        assert cfg2.get_active_model().auto_compact_threshold == 75_000


class TestCompactionModel:
    def test_get_compaction_model_returns_active_when_unset(self) -> None:
        cfg = build_test_vibe_config()
        assert cfg.get_compaction_model() == cfg.get_active_model()

    def test_get_compaction_model_returns_configured_model(self) -> None:
        compaction = ModelConfig(
            name="compact-model", provider="mistral", alias="compact"
        )
        cfg = build_test_vibe_config(compaction_model=compaction)
        assert cfg.get_compaction_model().name == "compact-model"

    def test_compaction_model_provider_must_match_active(self) -> None:
        from privibe.core.config import ProviderConfig

        compaction = ModelConfig(
            name="compact-model", provider="other", alias="compact"
        )
        providers = [
            ProviderConfig(
                name="mistral",
                api_base="https://api.mistral.ai/v1",
                api_key_env_var="MISTRAL_API_KEY",
            ),
            ProviderConfig(
                name="other",
                api_base="https://other.ai/v1",
                api_key_env_var="MISTRAL_API_KEY",
            ),
        ]
        with pytest.raises(ValueError, match="must share the same provider"):
            build_test_vibe_config(compaction_model=compaction, providers=providers)

    def test_compaction_model_provider_must_exist(self) -> None:
        compaction = ModelConfig(
            name="compact-model", provider="missing-provider", alias="compact"
        )
        with pytest.raises(
            ValueError,
            match="Provider 'missing-provider' for model 'compact-model' not found in configuration",
        ):
            build_test_vibe_config(compaction_model=compaction)

    def test_compaction_model_excluded_from_model_dump_when_none(self) -> None:
        cfg = build_test_vibe_config()
        dumped = cfg.model_dump()
        assert "compaction_model" not in dumped


class TestActiveModelFallback:
    @staticmethod
    def _providers() -> list[ProviderConfig]:
        return [
            ProviderConfig(
                name="with_key",
                api_base="https://a.example/v1",
                api_key_env_var="TEST_KEY_A",
            ),
            ProviderConfig(
                name="no_key",
                api_base="https://b.example/v1",
                api_key_env_var="TEST_KEY_B",
            ),
            ProviderConfig(
                name="no_env",
                api_base="http://localhost/v1",
                api_key_env_var="",
            ),
        ]

    def test_returns_active_model_when_alias_matches_and_key_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEST_KEY_A", "value")
        monkeypatch.delenv("TEST_KEY_B", raising=False)
        cfg = build_test_vibe_config(
            providers=self._providers(),
            models=[
                ModelConfig(name="m1", provider="with_key", alias="m1"),
                ModelConfig(name="m2", provider="no_key", alias="m2"),
            ],
            active_model="m1",
        )
        assert cfg.get_active_model().alias == "m1"
        assert cfg.active_model == "m1"

    def test_falls_back_when_active_alias_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEST_KEY_A", "value")
        cfg = build_test_vibe_config(
            providers=self._providers(),
            models=[ModelConfig(name="m1", provider="with_key", alias="m1")],
            active_model="nonexistent",
        )
        assert cfg.get_active_model().alias == "m1"
        assert cfg.active_model == "m1"

    def test_falls_back_when_active_model_api_key_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TEST_KEY_A", raising=False)
        monkeypatch.setenv("TEST_KEY_B", "value")
        cfg = build_test_vibe_config(
            providers=self._providers(),
            models=[
                ModelConfig(name="m1", provider="with_key", alias="m1"),
                ModelConfig(name="m2", provider="no_key", alias="m2"),
            ],
            active_model="m1",
        )
        assert cfg.get_active_model().alias == "m2"
        assert cfg.active_model == "m2"

    def test_skips_models_with_missing_key_in_fallback_loop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TEST_KEY_A", raising=False)
        monkeypatch.setenv("TEST_KEY_B", "value")
        cfg = build_test_vibe_config(
            providers=self._providers(),
            models=[
                ModelConfig(name="m1", provider="with_key", alias="m1"),
                ModelConfig(name="m2", provider="no_key", alias="m2"),
            ],
            active_model="nonexistent",
        )
        assert cfg.get_active_model().alias == "m2"
        assert cfg.active_model == "m2"

    def test_provider_without_env_var_is_always_usable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TEST_KEY_A", raising=False)
        monkeypatch.delenv("TEST_KEY_B", raising=False)
        cfg = build_test_vibe_config(
            providers=self._providers(),
            models=[ModelConfig(name="local", provider="no_env", alias="local")],
            active_model="local",
        )
        assert cfg.get_active_model().alias == "local"

    def test_raises_when_no_model_has_valid_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TEST_KEY_A", raising=False)
        monkeypatch.delenv("TEST_KEY_B", raising=False)
        cfg = build_test_vibe_config(
            providers=self._providers(),
            models=[
                ModelConfig(name="m1", provider="with_key", alias="m1"),
                ModelConfig(name="m2", provider="no_key", alias="m2"),
            ],
            active_model="m1",
        )
        with pytest.raises(
            ValueError,
            match="no other configured model has its API key",
        ):
            cfg.get_active_model()
