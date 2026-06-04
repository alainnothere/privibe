from __future__ import annotations

from privibe.core.tts.factory import make_tts_client
from privibe.core.tts.openai_tts_client import OpenAITTSClient
from privibe.core.tts.tts_client_port import TTSClientPort, TTSResult

__all__ = ["OpenAITTSClient", "TTSClientPort", "TTSResult", "make_tts_client"]
