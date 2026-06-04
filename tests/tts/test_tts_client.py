from __future__ import annotations

import httpx
import pytest

from privibe.core.config import TTSModelConfig, TTSProviderConfig
from privibe.core.tts import OpenAITTSClient, TTSResult


def _make_provider() -> TTSProviderConfig:
    return TTSProviderConfig(
        name="openai",
        api_base="https://api.openai.com",
        api_key_env_var="OPENAI_API_KEY",
    )


def _make_model() -> TTSModelConfig:
    return TTSModelConfig(
        name="tts-1", alias="tts", provider="openai", voice="alloy"
    )


class TestOpenAITTSClientInit:
    def test_client_configured_with_base_url_and_auth(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        client = OpenAITTSClient(_make_provider(), _make_model())
        assert str(client._client.base_url) == "https://api.openai.com/v1/"
        assert client._client.headers["authorization"] == "Bearer test-key"


class TestOpenAITTSClient:
    @pytest.mark.asyncio
    async def test_speak_returns_audio_bytes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        raw_audio = b"fake-audio-data-for-testing"

        async def mock_post(self_client, url, **kwargs):
            assert url == "/audio/speech"
            body = kwargs["json"]
            assert body["model"] == "tts-1"
            assert body["input"] == "Hello"
            assert body["voice"] == "alloy"
            assert body["response_format"] == "wav"
            return httpx.Response(
                status_code=200,
                content=raw_audio,
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        client = OpenAITTSClient(_make_provider(), _make_model())
        result = await client.speak("Hello")

        assert isinstance(result, TTSResult)
        assert result.audio_data == raw_audio
        await client.close()

    @pytest.mark.asyncio
    async def test_speak_raises_on_http_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        async def mock_post(self_client, url, **kwargs):
            return httpx.Response(
                status_code=500,
                json={"error": "Internal Server Error"},
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        client = OpenAITTSClient(_make_provider(), _make_model())
        with pytest.raises(httpx.HTTPStatusError):
            await client.speak("Hello")
        await client.close()

    @pytest.mark.asyncio
    async def test_close_closes_underlying_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        client = OpenAITTSClient(_make_provider(), _make_model())
        await client.close()
        assert client._client.is_closed
