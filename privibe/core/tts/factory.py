from __future__ import annotations

from privibe.core.config import TTSClient, TTSModelConfig, TTSProviderConfig
from privibe.core.tts.openai_tts_client import OpenAITTSClient
from privibe.core.tts.tts_client_port import TTSClientPort

TTS_CLIENT_MAP: dict[TTSClient, type[TTSClientPort]] = {
    TTSClient.OPENAI: OpenAITTSClient
}


def make_tts_client(
    provider: TTSProviderConfig, model: TTSModelConfig
) -> TTSClientPort:
    return TTS_CLIENT_MAP[provider.client](provider=provider, model=model)
