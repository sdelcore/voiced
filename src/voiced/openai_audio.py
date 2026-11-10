"""OpenAI-compatible HTTP audio transcription and speech synthesis."""

from __future__ import annotations

import io
import os
import wave

import numpy as np

from voiced.config import ClientConfig


class OpenAIAudio:
    """Audio provider backed by OpenAI-compatible audio endpoints."""

    def __init__(self, config: ClientConfig):
        if not config.audio_base_url:
            raise ValueError("client.audio_base_url must be configured")
        if not config.stt_model or not config.tts_model:
            raise ValueError("client.stt_model and client.tts_model must be configured")
        self.base_url = config.audio_base_url.rstrip("/")
        self.stt_model = config.stt_model
        self.tts_model = config.tts_model
        self.timeout = config.timeout
        api_key = os.environ.get(config.api_key_env, "")
        if not api_key:
            raise ValueError(f"API key environment variable {config.api_key_env!r} is not set")
        self.api_key = api_key

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        import requests

        samples = np.asarray(audio, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("audio must be mono float samples")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        wav = io.BytesIO()
        with wave.open(wav, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            pcm = np.clip(samples, -1.0, 1.0)
            output.writeframes((pcm * 32767).astype("<i2").tobytes())
        response = requests.post(
            f"{self.base_url}/audio/transcriptions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            data={"model": self.stt_model, "response_format": "json"},
            files={"file": ("audio.wav", wav.getvalue(), "audio/wav")},
            timeout=self.timeout,
        )
        response.raise_for_status()
        try:
            return response.json()["text"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError("audio transcription response did not contain text") from exc

    def synthesize(self, text: str) -> np.ndarray:
        import requests

        response = requests.post(
            f"{self.base_url}/audio/speech",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.tts_model,
                "input": text,
                "voice": "af_heart",
                "response_format": "wav",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        try:
            with wave.open(io.BytesIO(response.content), "rb") as source:
                if source.getnchannels() != 1 or source.getsampwidth() != 2:
                    raise ValueError("speech WAV must be mono 16-bit PCM")
                rate = source.getframerate()
                audio = (
                    np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(
                        np.float32
                    )
                    / 32768.0
                )
        except (wave.Error, EOFError) as exc:
            raise ValueError("speech response was not a valid WAV file") from exc
        if rate != 24000:
            raise ValueError(f"speech WAV sample rate was {rate}, expected 24000 Hz")
        return audio
