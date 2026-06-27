"""Transcription engine using NVIDIA Parakeet-TDT (NeMo)."""

import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np

from voiced.audio_codec import normalize_audio
from voiced.config import TranscriptionConfig
from voiced.device import resolve_device_config
from voiced.model_host import ModelHost

logger = logging.getLogger(__name__)

STT_SAMPLE_RATE = 16000


def _empty_cuda_cache(_model: Any) -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Transcriber:
    """Wrapper around NeMo Parakeet-TDT for speech-to-text transcription."""

    def __init__(self, config: TranscriptionConfig | None = None):
        """Initialize the transcriber.

        Args:
            config: Transcription configuration. Uses defaults if not provided.
        """
        self.config = config or TranscriptionConfig()
        self._device: str | None = None
        self._host: ModelHost[Any] = ModelHost(
            loader=self._load_model,
            idle_timeout=(
                self.config.unload_timeout_minutes * 60
                if self.config.unload_timeout_minutes > 0
                else None
            ),
            on_unload=_empty_cuda_cache,
            name=f"parakeet({self.config.model})",
        )

    @property
    def device(self) -> str:
        """The device the model runs on. Resolved without loading if not yet loaded."""
        return self._device or resolve_device_config(self.config.device).device

    def warmup(self) -> None:
        """Force model load. Useful for triggering load outside a transcribe call."""
        with self._host.use():
            pass

    def _load_model(self) -> Any:
        """Load the Parakeet ASR model from HuggingFace via NeMo."""
        import nemo.collections.asr as nemo_asr

        device = resolve_device_config(self.config.device).device
        self._device = device

        logger.info(f"Loading model '{self.config.model}' on {device}")

        model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.config.model)
        model = model.to(device)
        model.eval()
        return model

    def _run(self, source: Any, *, timestamps: bool = False) -> Any:
        """Run the model and return the first result."""
        with self._host.use() as model:
            results = model.transcribe([source], timestamps=timestamps)
        if not results:
            raise RuntimeError("Parakeet returned no results")
        return results[0]

    def transcribe_file(self, audio_path: str | Path) -> str:
        """Transcribe an audio file."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Transcribing file: {audio_path}")
        result = self._run(str(audio_path))
        return (result.text or "").strip()

    def transcribe_file_with_segments(
        self, audio_path: str | Path
    ) -> list[tuple[float, float, str]]:
        """Transcribe an audio file and return segments with timestamps."""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Transcribing file with segments: {audio_path}")
        result = self._run(str(audio_path), timestamps=True)
        return _segments_from_result(result)

    def transcribe_audio_with_segments(
        self,
        audio: np.ndarray,
        sample_rate: int = STT_SAMPLE_RATE,
    ) -> list[tuple[float, float, str]]:
        """Transcribe audio array and return segments with timestamps."""
        audio = normalize_audio(audio)
        logger.info(f"Transcribing audio with segments: {len(audio)} samples at {sample_rate}Hz")
        result = self._run(audio, timestamps=True)
        return _segments_from_result(result)

    def transcribe_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = STT_SAMPLE_RATE,
    ) -> str:
        """Transcribe audio from a numpy array."""
        audio = normalize_audio(audio)
        logger.info(f"Transcribing audio buffer: {len(audio)} samples at {sample_rate}Hz")
        result = self._run(audio)
        return (result.text or "").strip()

    def transcribe_audio_with_words(
        self,
        audio: np.ndarray,
        sample_rate: int = STT_SAMPLE_RATE,
    ) -> tuple[str, list[tuple[str, float, float, float]]]:
        """Transcribe audio with word-level timestamps for streaming.

        Returns (text, [(word, start, end, probability)]).  Parakeet does not emit
        per-word probabilities; 1.0 is returned in that slot.
        """
        audio = normalize_audio(audio)
        logger.debug(f"Transcribing with words: {len(audio)} samples at {sample_rate}Hz")
        result = self._run(audio, timestamps=True)

        text = (result.text or "").strip()
        words: list[tuple[str, float, float, float]] = []
        for stamp in _word_stamps(result):
            words.append((stamp["word"], float(stamp["start"]), float(stamp["end"]), 1.0))
        return text, words

    def transcribe_partial(self, audio: np.ndarray) -> str:
        """Fast partial transcription for streaming use cases."""
        audio = normalize_audio(audio)
        result = self._run(audio)
        return (result.text or "").strip()

    def transcribe_stream(
        self, audio: np.ndarray, sample_rate: int = STT_SAMPLE_RATE
    ) -> Generator[str, None, None]:
        """Yield text per detected segment.

        Parakeet's inference is non-streaming; we yield each timestamped segment
        once the full transcription completes.
        """
        for _start, _end, text in self.transcribe_audio_with_segments(audio, sample_rate):
            yield text

    def unload(self) -> None:
        """Unload the model to free memory. No-op if not loaded."""
        self._host.unload()
        self._device = None


def _segments_from_result(result: Any) -> list[tuple[float, float, str]]:
    """Extract (start, end, text) tuples from a NeMo transcription result."""
    timestamp = getattr(result, "timestamp", None) or {}
    segments = timestamp.get("segment") or []
    out: list[tuple[float, float, str]] = []
    for seg in segments:
        text = (seg.get("segment") or seg.get("text") or "").strip()
        if not text:
            continue
        out.append((float(seg["start"]), float(seg["end"]), text))

    if not out and getattr(result, "text", None):
        # Model returned text without segment timestamps — emit a single span.
        out.append((0.0, 0.0, result.text.strip()))
    return out


def _word_stamps(result: Any) -> list[dict]:
    """Extract word-level timestamps from a NeMo transcription result."""
    timestamp = getattr(result, "timestamp", None) or {}
    return timestamp.get("word") or []
