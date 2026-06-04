from __future__ import annotations

from privibe.core.config import (
    TranscribeClient,
    TranscribeModelConfig,
    TranscribeProviderConfig,
)
from privibe.core.transcribe.transcribe_client_port import TranscribeClientPort
from privibe.core.transcribe.whisperlive_transcribe_client import (
    WhisperLiveTranscribeClient,
)

TRANSCRIBE_CLIENT_MAP: dict[TranscribeClient, type[TranscribeClientPort]] = {
    TranscribeClient.WHISPERLIVE: WhisperLiveTranscribeClient
}


def make_transcribe_client(
    provider: TranscribeProviderConfig, model: TranscribeModelConfig
) -> TranscribeClientPort:
    return TRANSCRIBE_CLIENT_MAP[provider.client](provider=provider, model=model)
