"""Tests for transcriber module.

These tests stub out NeMo so they can run without GPU or model downloads.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from voiced.config import TranscriptionConfig
from voiced.transcriber import STT_SAMPLE_RATE, Transcriber


def _fake_result(
    text: str = "hello world",
    *,
    segments: list[dict] | None = None,
    words: list[dict] | None = None,
) -> SimpleNamespace:
    """Build a fake NeMo transcription result."""
    timestamp: dict = {}
    if segments is not None:
        timestamp["segment"] = segments
    if words is not None:
        timestamp["word"] = words
    return SimpleNamespace(text=text, timestamp=timestamp)


class TestTranscriberConfig:
    def test_default_config(self):
        config = TranscriptionConfig()
        assert config.model == "nvidia/parakeet-tdt-0.6b-v3"
        assert config.device == "auto"
        assert config.language == "en"

    def test_custom_config(self):
        config = TranscriptionConfig(
            model="nvidia/parakeet-tdt-0.6b-v2",
            device="cpu",
            language="en",
        )
        assert config.model == "nvidia/parakeet-tdt-0.6b-v2"
        assert config.device == "cpu"


class TestTranscriberInit:
    def test_init(self):
        transcriber = Transcriber()
        assert transcriber.config.model == "nvidia/parakeet-tdt-0.6b-v3"
        assert not transcriber._host.is_loaded  # Lazy load

    def test_init_with_config(self):
        config = TranscriptionConfig(model="nvidia/parakeet-tdt-0.6b-v2")
        transcriber = Transcriber(config)
        assert transcriber.config.model == "nvidia/parakeet-tdt-0.6b-v2"

    def test_idle_timeout_from_minutes(self):
        transcriber = Transcriber(unload_timeout_minutes=15)
        assert transcriber._host._idle_timeout == 15 * 60

    def test_idle_timeout_disabled(self):
        transcriber = Transcriber(unload_timeout_minutes=0)
        assert transcriber._host._idle_timeout is None

    def test_idle_timeout_default(self):
        assert Transcriber()._host._idle_timeout == 15 * 60

    def test_device_explicit(self):
        config = TranscriptionConfig(device="cpu")
        transcriber = Transcriber(config)
        assert transcriber.device == "cpu"

    def test_device_explicit_cuda(self):
        config = TranscriptionConfig(device="cuda")
        transcriber = Transcriber(config)
        assert transcriber.device == "cuda"


class TestTranscriberFileNotFound:
    def test_transcribe_file_not_found(self):
        transcriber = Transcriber()
        with pytest.raises(FileNotFoundError):
            transcriber.transcribe_file("/nonexistent/audio.wav")

    def test_transcribe_file_with_segments_not_found(self):
        transcriber = Transcriber()
        with pytest.raises(FileNotFoundError):
            transcriber.transcribe_file_with_segments("/nonexistent/audio.wav")


class TestTranscribeMocked:
    """Drive the wrapper with a mocked NeMo model."""

    def _stub_model(self, transcriber: Transcriber, result: SimpleNamespace) -> MagicMock:
        model = MagicMock()
        model.transcribe.return_value = [result]
        transcriber._host._model = model
        return model

    def test_transcribe_audio_returns_text(self):
        transcriber = Transcriber()
        self._stub_model(transcriber, _fake_result(text="  hello world  "))
        audio = np.zeros(STT_SAMPLE_RATE, dtype=np.float32)
        assert transcriber.transcribe_audio(audio) == "hello world"

    def test_transcribe_audio_with_segments(self):
        transcriber = Transcriber()
        self._stub_model(
            transcriber,
            _fake_result(
                text="hello world",
                segments=[
                    {"start": 0.0, "end": 0.5, "segment": "hello"},
                    {"start": 0.5, "end": 1.0, "segment": "world"},
                ],
            ),
        )
        audio = np.zeros(STT_SAMPLE_RATE, dtype=np.float32)
        segments = transcriber.transcribe_audio_with_segments(audio)
        assert segments == [(0.0, 0.5, "hello"), (0.5, 1.0, "world")]

    def test_segments_fallback_when_no_timestamps(self):
        """If NeMo returns text but no segment stamps, emit a single span."""
        transcriber = Transcriber()
        self._stub_model(transcriber, _fake_result(text="bare text"))
        audio = np.zeros(STT_SAMPLE_RATE, dtype=np.float32)
        segments = transcriber.transcribe_audio_with_segments(audio)
        assert segments == [(0.0, 0.0, "bare text")]

    def test_transcribe_audio_with_words(self):
        transcriber = Transcriber()
        self._stub_model(
            transcriber,
            _fake_result(
                text="hello world",
                words=[
                    {"start": 0.0, "end": 0.4, "word": "hello"},
                    {"start": 0.5, "end": 0.9, "word": "world"},
                ],
            ),
        )
        audio = np.zeros(STT_SAMPLE_RATE, dtype=np.float32)
        text, words = transcriber.transcribe_audio_with_words(audio)
        assert text == "hello world"
        assert words == [
            ("hello", 0.0, 0.4, 1.0),
            ("world", 0.5, 0.9, 1.0),
        ]

    def test_transcribe_partial(self):
        transcriber = Transcriber()
        self._stub_model(transcriber, _fake_result(text="partial"))
        audio = np.zeros(STT_SAMPLE_RATE, dtype=np.float32)
        assert transcriber.transcribe_partial(audio) == "partial"

    def test_transcribe_stream_yields_segment_text(self):
        transcriber = Transcriber()
        self._stub_model(
            transcriber,
            _fake_result(
                text="a b",
                segments=[
                    {"start": 0.0, "end": 0.1, "segment": "a"},
                    {"start": 0.1, "end": 0.2, "segment": "b"},
                ],
            ),
        )
        audio = np.zeros(STT_SAMPLE_RATE, dtype=np.float32)
        assert list(transcriber.transcribe_stream(audio)) == ["a", "b"]

    def test_empty_results_raises(self):
        transcriber = Transcriber()
        model = MagicMock()
        model.transcribe.return_value = []
        transcriber._host._model = model
        audio = np.zeros(STT_SAMPLE_RATE, dtype=np.float32)
        with pytest.raises(RuntimeError, match="no results"):
            transcriber.transcribe_audio(audio)

    def test_unload_clears_model(self):
        transcriber = Transcriber()
        self._stub_model(transcriber, _fake_result())
        transcriber.unload()
        assert not transcriber._host.is_loaded


class TestSampleRateConstant:
    def test_stt_sample_rate_is_16khz(self):
        assert STT_SAMPLE_RATE == 16000
