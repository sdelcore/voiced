"""End-to-end TTS → STT round-trip test.

Synthesizes a known sentence with VibeVoice, then transcribes it with
Parakeet-TDT, and asserts the result is recognizable as the original.

Skipped by default — opt in with ``pytest -m integration``.  Requires:
  * VibeVoice installed (TTS).
  * NeMo + Parakeet weights downloadable (STT).
  * A working voice preset (downloaded automatically by VoiceManager).
  * Enough VRAM for both models, or patience on CPU.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from voiced.config import TranscriptionConfig
from voiced.synthesizer import SAMPLE_RATE as TTS_SAMPLE_RATE
from voiced.synthesizer import Synthesizer, TTSConfig, check_vibevoice_installed
from voiced.transcriber import STT_SAMPLE_RATE, Transcriber

pytestmark = pytest.mark.integration


SENTENCE = "The quick brown fox jumps over the lazy dog."


def _normalize(text: str) -> list[str]:
    """Lowercase + strip punctuation → list of words."""
    return re.findall(r"[a-z']+", text.lower())


def _word_overlap(expected: list[str], actual: list[str]) -> float:
    """Fraction of expected words present in actual (order-independent)."""
    if not expected:
        return 1.0
    actual_set = set(actual)
    hits = sum(1 for w in expected if w in actual_set)
    return hits / len(expected)


def _resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Polyphase resample. Lazy-imports scipy to keep base test imports light."""
    from scipy import signal

    if src == dst:
        return audio.astype(np.float32)
    n = int(round(len(audio) * dst / src))
    return signal.resample(audio, n).astype(np.float32)


@pytest.fixture(scope="module")
def synthesizer() -> Synthesizer:
    if not check_vibevoice_installed():
        pytest.skip("VibeVoice not installed")
    synth = Synthesizer(TTSConfig(unload_timeout_seconds=0))
    yield synth
    synth.shutdown()


@pytest.fixture(scope="module")
def transcriber() -> Transcriber:
    pytest.importorskip("nemo")
    t = Transcriber(TranscriptionConfig())
    yield t
    t.unload()


def test_tts_to_stt_recovers_sentence(synthesizer: Synthesizer, transcriber: Transcriber):
    """Synthesize a sentence, transcribe it, expect most of the words back."""
    audio_24k = synthesizer.synthesize(SENTENCE)
    assert audio_24k.ndim == 1
    assert audio_24k.size > TTS_SAMPLE_RATE // 2, "TTS produced suspiciously short audio"

    audio_16k = _resample(audio_24k, TTS_SAMPLE_RATE, STT_SAMPLE_RATE)
    transcribed = transcriber.transcribe_audio(audio_16k)
    assert transcribed, "Transcriber returned empty text"

    expected = _normalize(SENTENCE)
    actual = _normalize(transcribed)
    overlap = _word_overlap(expected, actual)

    # 80% threshold — TTS+STT loses an article or substitutes a homophone occasionally.
    assert overlap >= 0.8, (
        f"Round-trip word overlap {overlap:.0%} below threshold.\n"
        f"  Expected: {SENTENCE!r}\n"
        f"  Got:      {transcribed!r}"
    )
