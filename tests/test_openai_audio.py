"""Deterministic tests for OpenAI-compatible audio requests."""

import io
import wave

import numpy as np
import pytest

from voiced.config import ClientConfig
from voiced.openai_audio import OpenAIAudio


@pytest.fixture
def audio_client(monkeypatch):
    monkeypatch.setenv("TEST_OPENAI_KEY", "test-secret")
    return OpenAIAudio(
        ClientConfig(
            audio_base_url="https://audio.example/v1/",
            stt_model="stt-model",
            tts_model="tts-model",
            api_key_env="TEST_OPENAI_KEY",
            timeout=12.5,
        )
    )


def test_transcribe_posts_json_response_format_and_bearer_auth(audio_client, monkeypatch):
    import requests

    calls = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"text": "recognized words"}

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return Response()

    monkeypatch.setattr(requests, "post", post)
    assert (
        audio_client.transcribe(np.array([0.0, 0.5, -0.5], dtype=np.float32), 16000)
        == "recognized words"
    )

    args, kwargs = calls[0]
    assert args == ("https://audio.example/v1/audio/transcriptions",)
    assert kwargs["headers"] == {"Authorization": "Bearer test-secret"}
    assert kwargs["data"] == {"model": "stt-model", "response_format": "json"}
    assert kwargs["timeout"] == 12.5
    filename, payload, content_type = kwargs["files"]["file"]
    assert (filename, content_type) == ("audio.wav", "audio/wav")
    with wave.open(io.BytesIO(payload), "rb") as source:
        assert (source.getnchannels(), source.getframerate(), source.getsampwidth()) == (
            1,
            16000,
            2,
        )
        assert np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").tolist() == [
            0,
            16383,
            -16383,
        ]


def test_synthesize_posts_voice_and_decodes_wav(audio_client, monkeypatch):
    import requests

    wav = io.BytesIO()
    with wave.open(wav, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(24000)
        target.writeframes(np.array([0, 16384, -16384], dtype="<i2").tobytes())
    calls = []

    class Response:
        content = wav.getvalue()

        def raise_for_status(self):
            pass

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return Response()

    monkeypatch.setattr(requests, "post", post)
    result = audio_client.synthesize("hello")
    args, kwargs = calls[0]
    assert args == ("https://audio.example/v1/audio/speech",)
    assert kwargs["headers"] == {"Authorization": "Bearer test-secret"}
    assert kwargs["json"] == {
        "model": "tts-model",
        "input": "hello",
        "voice": "af_heart",
        "response_format": "wav",
    }
    assert kwargs["timeout"] == 12.5
    np.testing.assert_allclose(result, np.array([0.0, 0.5, -0.5], dtype=np.float32), atol=1 / 32768)


def test_requires_configured_api_key(monkeypatch):
    monkeypatch.delenv("TEST_OPENAI_KEY", raising=False)
    with pytest.raises(ValueError, match="TEST_OPENAI_KEY"):
        OpenAIAudio(
            ClientConfig(
                audio_base_url="https://audio.example",
                stt_model="stt",
                tts_model="tts",
                api_key_env="TEST_OPENAI_KEY",
            )
        )
