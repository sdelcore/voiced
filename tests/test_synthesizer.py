"""Tests for the Kokoro synthesizer.

These tests stub out the Kokoro pipeline so they run without model downloads.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voiced.synthesizer import (
    DEFAULT_VOICE,
    SAMPLE_RATE,
    TTS_MODEL,
    Synthesizer,
    TTSConfig,
)


def _fake_pipeline_results(*chunks: np.ndarray):
    """Build fake KPipeline results: (graphemes, phonemes, audio) tuples."""
    return [("text", "phonemes", chunk) for chunk in chunks]


def _stub(synth: Synthesizer, *chunks: np.ndarray) -> MagicMock:
    """Wire a fake model + pipeline into the synthesizer, bypassing load."""
    model = MagicMock()
    synth._host._model = model
    pipeline = MagicMock(return_value=_fake_pipeline_results(*chunks))
    synth._pipelines["a"] = pipeline
    synth._pipelines["b"] = pipeline
    synth.voice_manager = MagicMock()
    synth.voice_manager.load_voice_tensor.return_value = MagicMock()
    return pipeline


class TestConfig:
    def test_model_is_fixed(self):
        assert TTS_MODEL == "hexgrad/Kokoro-82M"
        assert "model" not in TTSConfig.__dataclass_fields__
        assert "model_path" not in TTSConfig.__dataclass_fields__

    def test_defaults(self):
        config = TTSConfig()
        assert config.default_voice == DEFAULT_VOICE
        assert config.speed == 1.0

    def test_sample_rate(self):
        assert SAMPLE_RATE == 24000
        assert Synthesizer().sample_rate == 24000


class TestLazyLoad:
    def test_not_loaded_on_init(self):
        synth = Synthesizer()
        assert not synth.is_loaded

    def test_idle_timeout_zero_disables_unload(self):
        synth = Synthesizer(TTSConfig(unload_timeout_seconds=0))
        assert synth._host._idle_timeout is None

    def test_load_fails_without_kokoro(self):
        synth = Synthesizer()
        with patch("voiced.synthesizer.check_kokoro_installed", return_value=False):
            with pytest.raises(RuntimeError, match="not installed"):
                synth._load_model()


class TestSynthesize:
    def test_batch_concatenates_chunks(self):
        synth = Synthesizer()
        a = np.ones(100, dtype=np.float32)
        b = np.zeros(50, dtype=np.float32)
        _stub(synth, a, b)

        audio = synth.synthesize("Hello. World.")
        assert audio.dtype == np.float32
        assert len(audio) == 150

    def test_batch_raises_on_no_output(self):
        synth = Synthesizer()
        _stub(synth)
        with pytest.raises(RuntimeError, match="No audio output"):
            synth.synthesize("Hello")

    def test_streaming_yields_per_chunk(self):
        synth = Synthesizer()
        a = np.ones(100, dtype=np.float32)
        b = np.zeros(50, dtype=np.float32)
        _stub(synth, a, b)

        chunks = list(synth.synthesize_streaming("Hello. World."))
        assert len(chunks) == 2
        assert all(c.dtype == np.float32 for c in chunks)

    def test_default_voice_used(self):
        synth = Synthesizer()
        _stub(synth, np.ones(10, dtype=np.float32))
        synth.synthesize("Hi")
        synth.voice_manager.load_voice_tensor.assert_called_once_with(DEFAULT_VOICE)

    def test_british_voice_selects_british_pipeline(self):
        synth = Synthesizer()
        model = MagicMock()
        synth._host._model = model
        synth.voice_manager = MagicMock()
        synth.voice_manager.load_voice_tensor.return_value = MagicMock()

        with patch("kokoro.KPipeline") as kpipeline:
            kpipeline.return_value = MagicMock(
                return_value=_fake_pipeline_results(np.ones(10, dtype=np.float32))
            )
            synth.synthesize("Hi", voice="bf_emma")
            assert kpipeline.call_args.kwargs["lang_code"] == "b"

    def test_speed_passed_through(self):
        synth = Synthesizer(TTSConfig(speed=1.3))
        pipeline = _stub(synth, np.ones(10, dtype=np.float32))
        synth.synthesize("Hi")
        assert pipeline.call_args.kwargs["speed"] == 1.3


class TestStatus:
    def test_status_reports_fixed_model(self):
        synth = Synthesizer()
        synth.voice_manager = MagicMock()
        synth.voice_manager.list_downloaded.return_value = []
        status = synth.get_status()
        assert status["model"] == TTS_MODEL
        assert status["model_loaded"] is False
        assert "cfg_scale" not in status
