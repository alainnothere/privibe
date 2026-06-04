from __future__ import annotations

from collections.abc import AsyncIterator
import contextlib
import json

import pytest

from privibe.core.config import TranscribeModelConfig, TranscribeProviderConfig
from privibe.core.transcribe import (
    TranscribeDone,
    TranscribeError,
    TranscribeSessionCreated,
    TranscribeTextDelta,
    WhisperLiveTranscribeClient,
)


def _make_provider() -> TranscribeProviderConfig:
    return TranscribeProviderConfig(
        name="whisperlive", api_base="ws://localhost:9090", api_key_env_var=""
    )


def _make_model() -> TranscribeModelConfig:
    return TranscribeModelConfig(
        name="small",
        alias="whisper",
        provider="whisperlive",
        encoding="pcm_s16le",
        sample_rate=16_000,
    )


async def _empty_audio_stream() -> AsyncIterator[bytes]:
    return
    yield


class FakeWebSocket:
    def __init__(self, server_messages: list[str | bytes]) -> None:
        self._server_messages = list(server_messages)
        self.sent: list[bytes | str] = []

    async def send(self, payload: bytes | str) -> None:
        self.sent.append(payload)

    def __aiter__(self) -> FakeWebSocket:
        return self

    async def __anext__(self) -> str | bytes:
        if not self._server_messages:
            raise StopAsyncIteration
        return self._server_messages.pop(0)

    async def __aenter__(self) -> FakeWebSocket:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None


def _patch_connect(monkeypatch: pytest.MonkeyPatch, fake_ws: FakeWebSocket) -> None:
    @contextlib.asynccontextmanager
    async def _connect(_url: str, **_kwargs: object) -> AsyncIterator[FakeWebSocket]:
        yield fake_ws

    import privibe.core.transcribe.whisperlive_transcribe_client as mod

    monkeypatch.setattr(mod.websockets, "connect", _connect)


async def _collect(
    client: WhisperLiveTranscribeClient,
) -> list[object]:
    events: list[object] = []
    async for event in client.transcribe(_empty_audio_stream()):
        events.append(event)
    return events


class TestEventMapping:
    @pytest.mark.asyncio
    async def test_session_created(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = FakeWebSocket([json.dumps({"message": "SERVER_READY"})])
        _patch_connect(monkeypatch, ws)
        client = WhisperLiveTranscribeClient(_make_provider(), _make_model())

        events = await _collect(client)

        assert len(events) == 1
        assert isinstance(events[0], TranscribeSessionCreated)
        assert events[0].request_id

    @pytest.mark.asyncio
    async def test_completed_segment_emits_text_delta(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws = FakeWebSocket([
            json.dumps({"segments": [{"text": "hello", "completed": True}]})
        ])
        _patch_connect(monkeypatch, ws)
        client = WhisperLiveTranscribeClient(_make_provider(), _make_model())

        events = await _collect(client)

        assert len(events) == 1
        assert isinstance(events[0], TranscribeTextDelta)
        assert events[0].text == "hello"

    @pytest.mark.asyncio
    async def test_in_progress_segment_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws = FakeWebSocket([
            json.dumps({"segments": [{"text": "in progress", "completed": False}]})
        ])
        _patch_connect(monkeypatch, ws)
        client = WhisperLiveTranscribeClient(_make_provider(), _make_model())

        events = await _collect(client)

        assert events == []

    @pytest.mark.asyncio
    async def test_only_new_completed_segments_emit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws = FakeWebSocket([
            json.dumps({"segments": [{"text": "first", "completed": True}]}),
            json.dumps({
                "segments": [
                    {"text": "first", "completed": True},
                    {"text": "second", "completed": True},
                ]
            }),
        ])
        _patch_connect(monkeypatch, ws)
        client = WhisperLiveTranscribeClient(_make_provider(), _make_model())

        events = await _collect(client)

        assert [e.text for e in events if isinstance(e, TranscribeTextDelta)] == [
            "first",
            "second",
        ]

    @pytest.mark.asyncio
    async def test_disconnect_emits_done_and_stops(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws = FakeWebSocket([
            json.dumps({"message": "SERVER_READY"}),
            json.dumps({"message": "DISCONNECT"}),
            json.dumps({"segments": [{"text": "post-done", "completed": True}]}),
        ])
        _patch_connect(monkeypatch, ws)
        client = WhisperLiveTranscribeClient(_make_provider(), _make_model())

        events = await _collect(client)

        assert len(events) == 2
        assert isinstance(events[0], TranscribeSessionCreated)
        assert isinstance(events[1], TranscribeDone)

    @pytest.mark.asyncio
    async def test_error_status_emits_transcribe_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws = FakeWebSocket([
            json.dumps({"status": "ERROR", "error": "model failed to load"})
        ])
        _patch_connect(monkeypatch, ws)
        client = WhisperLiveTranscribeClient(_make_provider(), _make_model())

        events = await _collect(client)

        assert len(events) == 1
        assert isinstance(events[0], TranscribeError)
        assert events[0].message == "model failed to load"

    @pytest.mark.asyncio
    async def test_malformed_json_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws = FakeWebSocket([
            "{not valid json",
            json.dumps({"message": "SERVER_READY"}),
        ])
        _patch_connect(monkeypatch, ws)
        client = WhisperLiveTranscribeClient(_make_provider(), _make_model())

        events = await _collect(client)

        assert len(events) == 1
        assert isinstance(events[0], TranscribeSessionCreated)

    @pytest.mark.asyncio
    async def test_handshake_sent_on_connect(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws = FakeWebSocket([json.dumps({"message": "SERVER_READY"})])
        _patch_connect(monkeypatch, ws)
        client = WhisperLiveTranscribeClient(_make_provider(), _make_model())

        await _collect(client)

        config_payload = ws.sent[0]
        assert isinstance(config_payload, str)
        config = json.loads(config_payload)
        assert config["model"] == "small"
        assert config["language"] == "en"
        assert config["task"] == "transcribe"
        assert config["use_vad"] is True
        assert "uid" in config
