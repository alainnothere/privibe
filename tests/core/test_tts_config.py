from __future__ import annotations

import pytest

from tests.conftest import build_test_vibe_config
from privibe.core.config import (
    DEFAULT_TTS_MODELS,
    DEFAULT_TTS_PROVIDERS,
    TTSClient,
    TTSModelConfig,
    TTSProviderConfig,
)


class TestTTSConfigDefaults:
    def test_default_tts_providers_loaded(self) -> None:
        config = build_test_vibe_config()
        assert len(config.tts_providers) == len(DEFAULT_TTS_PROVIDERS)
        assert config.tts_providers[0].name == "tts"
        assert config.tts_providers[0].api_base == ""

    def test_default_tts_models_loaded(self) -> None:
        config = build_test_vibe_config()
        assert len(config.tts_models) == len(DEFAULT_TTS_MODELS)
        assert config.tts_models[0].alias == "tts"
        assert config.tts_models[0].name == "tts-1"

    def test_default_active_tts_model(self) -> None:
        config = build_test_vibe_config()
        assert config.active_tts_model == "tts"


class TestGetActiveTTSModel:
    def test_resolves_by_alias(self) -> None:
        config = build_test_vibe_config()
        model = config.get_active_tts_model()
        assert model.alias == "tts"
        assert model.name == "tts-1"

    def test_raises_for_unknown_alias(self) -> None:
        config = build_test_vibe_config(active_tts_model="nonexistent")
        with pytest.raises(ValueError, match="not found in configuration"):
            config.get_active_tts_model()


class TestGetTTSProviderForModel:
    def test_resolves_by_name(self) -> None:
        config = build_test_vibe_config()
        model = config.get_active_tts_model()
        provider = config.get_tts_provider_for_model(model)
        assert provider.name == "tts"
        assert provider.api_base == ""

    def test_raises_for_unknown_provider(self) -> None:
        config = build_test_vibe_config(
            tts_models=[
                TTSModelConfig(name="test-model", provider="nonexistent", alias="test")
            ],
            active_tts_model="test",
        )
        model = config.get_active_tts_model()
        with pytest.raises(ValueError, match="not found in configuration"):
            config.get_tts_provider_for_model(model)


class TestTTSModelUniqueness:
    def test_duplicate_aliases_raise(self) -> None:
        with pytest.raises(ValueError, match="Duplicate TTS model alias"):
            build_test_vibe_config(
                tts_models=[
                    TTSModelConfig(name="model-a", provider="tts", alias="same-alias"),
                    TTSModelConfig(name="model-b", provider="tts", alias="same-alias"),
                ],
                active_tts_model="same-alias",
            )


class TestTTSModelConfig:
    def test_alias_defaults_to_name(self) -> None:
        model = TTSModelConfig.model_validate({"name": "my-model", "provider": "tts"})
        assert model.alias == "my-model"

    def test_explicit_alias(self) -> None:
        model = TTSModelConfig(
            name="my-model", provider="tts", alias="custom-alias"
        )
        assert model.alias == "custom-alias"

    def test_default_values(self) -> None:
        model = TTSModelConfig(name="my-model", provider="tts", alias="my-model")
        assert model.voice == "alloy"
        assert model.response_format == "wav"


class TestTTSProviderConfig:
    def test_default_values(self) -> None:
        provider = TTSProviderConfig(name="test")
        assert provider.api_base == ""
        assert provider.api_key_env_var == ""
        assert provider.client == TTSClient.OPENAI
