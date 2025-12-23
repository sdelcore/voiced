"""Transcription engine using faster-whisper."""

import logging
from collections.abc import Generator
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

from sttd.config import TranscriptionConfig

logger = logging.getLogger(__name__)


class Transcriber:
    """Wrapper around faster-whisper for speech-to-text transcription."""

    def __init__(self, config: TranscriptionConfig | None = None):
        """Initialize the transcriber.

        Args:
            config: Transcription configuration. Uses defaults if not provided.
        """
        self.config = config or TranscriptionConfig()
        self._model: WhisperModel | None = None

    @property
    def model(self) -> WhisperModel:
        """Lazy-load the whisper model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _get_device(self) -> str:
        """Determine the device to use for transcription."""
        if self.config.device != "auto":
            return self.config.device

        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            logger.info("CUDA available, using GPU")
            return "cuda"

        logger.info("Using CPU for transcription")
        return "cpu"

    def _get_compute_type(self, device: str) -> str:
        """Determine the compute type based on device."""
        if self.config.compute_type != "auto":
            return self.config.compute_type

        # Auto-select based on device
        if device == "cuda":
            return "float16"
        return "int8"

    def _load_model(self) -> WhisperModel:
        """Load the faster-whisper model."""
        device = self._get_device()
        compute_type = self._get_compute_type(device)

        logger.info(
            f"Loading model '{self.config.model}' on {device} with compute_type={compute_type}"
        )

        return WhisperModel(
            self.config.model,
            device=device,
            compute_type=compute_type,
        )

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to float32 in [-1, 1] range."""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / 32768.0
        return audio

    def transcribe_file(self, audio_path: str | Path) -> str:
        """Transcribe an audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            The transcribed text.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Transcribing file: {audio_path}")

        segments, info = self.model.transcribe(
            str(audio_path),
            language=self.config.language,
            beam_size=5,
            vad_filter=True,
        )

        logger.info(
            f"Detected language: {info.language} (probability: {info.language_probability:.2f})"
        )

        # Collect all segments into a single string
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        return " ".join(text_parts)

    def transcribe_file_with_segments(
        self, audio_path: str | Path
    ) -> list[tuple[float, float, str]]:
        """Transcribe an audio file and return segments with timestamps.

        Args:
            audio_path: Path to the audio file.

        Returns:
            List of (start, end, text) tuples.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Transcribing file with segments: {audio_path}")

        segments, info = self.model.transcribe(
            str(audio_path),
            language=self.config.language,
            beam_size=5,
            vad_filter=True,
        )

        logger.info(
            f"Detected language: {info.language} (probability: {info.language_probability:.2f})"
        )

        result = []
        for segment in segments:
            result.append((segment.start, segment.end, segment.text.strip()))

        return result

    def transcribe_audio_with_segments(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> list[tuple[float, float, str]]:
        """Transcribe audio array and return segments with timestamps.

        Args:
            audio: Audio data as numpy array (float32, mono).
            sample_rate: Sample rate of the audio (default: 16000).

        Returns:
            List of (start, end, text) tuples.
        """
        audio = self._normalize_audio(audio)
        logger.info(f"Transcribing audio with segments: {len(audio)} samples at {sample_rate}Hz")

        segments, info = self.model.transcribe(
            audio,
            language=self.config.language,
            beam_size=5,
            vad_filter=True,
        )

        logger.info(
            f"Detected language: {info.language} (probability: {info.language_probability:.2f})"
        )

        result = []
        for segment in segments:
            result.append((segment.start, segment.end, segment.text.strip()))

        return result

    def transcribe_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        initial_prompt: str | None = None,
    ) -> str:
        """Transcribe audio from a numpy array.

        Args:
            audio: Audio data as a numpy array (float32, mono).
            sample_rate: Sample rate of the audio (default: 16000).
            initial_prompt: Optional context from previous transcription.

        Returns:
            The transcribed text.
        """
        audio = self._normalize_audio(audio)
        logger.info(f"Transcribing audio buffer: {len(audio)} samples at {sample_rate}Hz")

        segments, info = self.model.transcribe(
            audio,
            language=self.config.language,
            beam_size=5,
            vad_filter=True,
            initial_prompt=initial_prompt,
        )

        logger.info(
            f"Detected language: {info.language} (probability: {info.language_probability:.2f})"
        )

        # Collect all segments into a single string
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        return " ".join(text_parts)

    def transcribe_audio_with_words(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        initial_prompt: str | None = None,
        beam_size: int = 1,
    ) -> tuple[str, list[tuple[str, float, float, float]]]:
        """Transcribe audio with word-level timestamps for streaming.

        Args:
            audio: Audio data as a numpy array (float32, mono).
            sample_rate: Sample rate of the audio (default: 16000).
            initial_prompt: Optional context from previous transcription.
            beam_size: Beam size for decoding (1 = greedy, faster).

        Returns:
            Tuple of (full_text, list of (word, start, end, probability) tuples).
        """
        audio = self._normalize_audio(audio)
        logger.debug(f"Transcribing with words: {len(audio)} samples at {sample_rate}Hz")

        segments, info = self.model.transcribe(
            audio,
            language=self.config.language,
            beam_size=beam_size,
            word_timestamps=True,
            vad_filter=False,  # We handle buffering externally for streaming
            initial_prompt=initial_prompt,
        )

        logger.debug(
            f"Detected language: {info.language} (probability: {info.language_probability:.2f})"
        )

        text_parts = []
        all_words = []

        for segment in segments:
            text_parts.append(segment.text.strip())
            if segment.words:
                for word in segment.words:
                    all_words.append((word.word, word.start, word.end, word.probability))

        return " ".join(text_parts), all_words

    def transcribe_stream(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> Generator[str, None, None]:
        """Transcribe audio and yield segments as they are generated.

        Args:
            audio: Audio data as a numpy array (float32, mono).
            sample_rate: Sample rate of the audio (default: 16000).

        Yields:
            Transcribed text segments.
        """
        audio = self._normalize_audio(audio)
        segments, _ = self.model.transcribe(
            audio,
            language=self.config.language,
            beam_size=5,
            vad_filter=True,
        )

        for segment in segments:
            yield segment.text.strip()

    def unload(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("Model unloaded")
