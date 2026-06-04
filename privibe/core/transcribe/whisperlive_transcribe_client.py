from __future__ import annotations

from array import array
from collections.abc import AsyncIterator
import asyncio
import contextlib
import json
import sys
from typing import Any
import uuid

import websockets

from privibe.core.config import TranscribeModelConfig, TranscribeProviderConfig
from privibe.core.logger import logger
from privibe.core.transcribe.transcribe_client_port import (
    TranscribeDone,
    TranscribeError,
    TranscribeEvent,
    TranscribeSessionCreated,
    TranscribeTextDelta,
)


class WhisperLiveTranscribeClient:
    def __init__(
        self, provider: TranscribeProviderConfig, model: TranscribeModelConfig
    ) -> None:
        self._url = provider.api_base
        self._model_name = model.name
        self._language = model.language

    async def transcribe(
        self, audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscribeEvent]:
        uid = str(uuid.uuid4())
        config = {
            "uid": uid,
            "language": self._language,
            "task": "transcribe",
            "model": self._model_name,
            "use_vad": True,
        }

        async with websockets.connect(self._url) as ws:
            await ws.send(json.dumps(config))
            sender = asyncio.create_task(self._send_audio(ws, audio_stream))
            emitted_completed = 0
            try:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    data = self._safe_load(raw)
                    if data is None:
                        continue
                    if data.get("uid") not in (uid, None):
                        continue
                    control = self._parse_control(data, uid)
                    if control is not None:
                        yield control
                        if isinstance(control, TranscribeDone):
                            return
                        continue
                    new_texts, emitted_completed = self._extract_new_completed(
                        data, emitted_completed
                    )
                    for text in new_texts:
                        yield TranscribeTextDelta(text=text)
            finally:
                sender.cancel()
                with contextlib.suppress(BaseException):
                    await sender

    @staticmethod
    def _safe_load(raw: str) -> dict[str, Any] | None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _parse_control(data: dict[str, Any], uid: str) -> TranscribeEvent | None:
        msg = data.get("message")
        status = data.get("status")
        if msg == "SERVER_READY":
            return TranscribeSessionCreated(request_id=uid)
        if msg == "DISCONNECT":
            return TranscribeDone()
        if status == "ERROR" or msg == "ERROR":
            text = str(
                data.get("error") or data.get("message") or "transcription error"
            )
            return TranscribeError(message=text)
        return None

    @staticmethod
    def _extract_new_completed(
        data: dict[str, Any], emitted_completed: int
    ) -> tuple[list[str], int]:
        segments = data.get("segments")
        if not isinstance(segments, list):
            return [], emitted_completed
        completed_texts = [
            str(s.get("text", "")).strip()
            for s in segments
            if isinstance(s, dict) and s.get("completed")
        ]
        new_texts = [t for t in completed_texts[emitted_completed:] if t]
        return new_texts, len(completed_texts)

    async def _send_audio(
        self, ws: Any, audio_stream: AsyncIterator[bytes]
    ) -> None:
        try:
            async for chunk in audio_stream:
                if not chunk:
                    continue
                await ws.send(_s16le_to_float32_bytes(chunk))
            with contextlib.suppress(Exception):
                await ws.send(b"END_OF_AUDIO")
        except Exception:
            logger.debug("audio sender exited", exc_info=True)


def _s16le_to_float32_bytes(buf: bytes) -> bytes:
    samples = array("h")
    samples.frombytes(buf)
    if sys.byteorder != "little":
        samples.byteswap()
    floats = array("f", (s / 32768.0 for s in samples))
    return floats.tobytes()
