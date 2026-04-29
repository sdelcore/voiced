"""Tests for audio_codec module."""

import numpy as np

from voiced.audio_codec import audio_to_wav, normalize_audio, wav_to_audio


class TestNormalizeAudio:
    def test_int16_audio_is_scaled(self):
        audio = np.array([16384, -16384, 32767, -32768], dtype=np.int16)
        out = normalize_audio(audio)
        assert out.dtype == np.float32
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_float32_in_range_passthrough(self):
        audio = np.array([0.5, -0.5, 0.0], dtype=np.float32)
        out = normalize_audio(audio)
        assert out is audio
        assert out.dtype == np.float32

    def test_empty_array(self):
        audio = np.array([], dtype=np.float32)
        out = normalize_audio(audio)
        assert out.dtype == np.float32
        assert out.size == 0


class TestRoundTrip:
    def test_audio_to_wav_to_audio(self):
        audio = np.array([0.0, 0.5, -0.5, 0.25, -0.25], dtype=np.float32)
        wav_bytes = audio_to_wav(audio, sample_rate=16000)
        decoded, sample_rate = wav_to_audio(wav_bytes)
        assert sample_rate == 16000
        # int16 round-trip introduces a small quantization error
        assert decoded.shape == audio.shape
        np.testing.assert_allclose(decoded, audio, atol=1e-4)

    def test_wav_preserves_sample_rate(self):
        audio = np.zeros(100, dtype=np.float32)
        for sr in (8000, 16000, 24000, 44100, 48000):
            wav_bytes = audio_to_wav(audio, sample_rate=sr)
            _, decoded_sr = wav_to_audio(wav_bytes)
            assert decoded_sr == sr


class TestWavToAudio:
    def test_unsupported_sample_width_rejected(self):
        import io
        import wave

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(3)  # 24-bit, not supported
            wav.setframerate(16000)
            wav.writeframes(b"\x00" * 30)

        import pytest

        with pytest.raises(ValueError, match="Unsupported sample width"):
            wav_to_audio(buffer.getvalue())

    def test_stereo_is_downmixed_to_mono(self):
        import io
        import wave

        # Stereo: left=0.5, right=-0.5 → mono mean = 0.0
        left = (np.full(100, 0.5, dtype=np.float32) * 32767).astype(np.int16)
        right = (np.full(100, -0.5, dtype=np.float32) * 32767).astype(np.int16)
        interleaved = np.empty(200, dtype=np.int16)
        interleaved[0::2] = left
        interleaved[1::2] = right

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(interleaved.tobytes())

        audio, sr = wav_to_audio(buffer.getvalue())
        assert sr == 16000
        assert audio.shape == (100,)
        np.testing.assert_allclose(audio, 0.0, atol=1e-4)
