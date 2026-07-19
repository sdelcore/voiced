"""TTS synthesizer using Kokoro-82M."""

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from voiced.device import resolve_device_config
from voiced.model_host import ModelHost
from voiced.voice_manager import VoiceManager

logger = logging.getLogger(__name__)

TTS_MODEL = "hexgrad/Kokoro-82M"
DEFAULT_VOICE = "af_heart"
DEFAULT_UNLOAD_TIMEOUT = 900
SAMPLE_RATE = 24000

# Split on sentence boundaries so streaming yields audio per sentence.
SPLIT_PATTERN = r"(?<=[.!?…])\s+|\n+"


@dataclass
class TTSConfig:
    """Configuration for TTS synthesizer."""

    device: str = "auto"
    default_voice: str = DEFAULT_VOICE
    speed: float = 1.0
    unload_timeout_seconds: int = DEFAULT_UNLOAD_TIMEOUT


def check_kokoro_installed() -> bool:
    """Check if Kokoro is installed."""
    try:
        import importlib.util

        return importlib.util.find_spec("kokoro") is not None
    except ImportError:
        return False


class Synthesizer:
    """Kokoro TTS engine with lazy loading and auto-unload."""

    def __init__(self, config: TTSConfig | None = None):
        """Initialize synthesizer.

        Args:
            config: TTS configuration. If None, uses defaults.
        """
        self.config = config or TTSConfig()
        self.voice_manager = VoiceManager()
        self._last_used: float | None = None
        self._device: str | None = None
        self._pipelines: dict[str, Any] = {}
        self._host: ModelHost[Any] = ModelHost(
            loader=self._load_model,
            idle_timeout=(
                self.config.unload_timeout_seconds
                if self.config.unload_timeout_seconds > 0
                else None
            ),
            on_unload=self._on_unload,
            name="kokoro",
        )

    def _load_model(self) -> Any:
        """Load the Kokoro model from HuggingFace."""
        if not check_kokoro_installed():
            raise RuntimeError(
                "Kokoro is not installed. Please install it with:\n  pip install kokoro"
            )

        from kokoro import KModel

        device = resolve_device_config(self.config.device).device
        if device not in ("cuda", "cpu"):
            device = "cpu"
        self._device = device

        logger.info(f"Loading TTS model '{TTS_MODEL}' on {device}...")
        model = KModel(repo_id=TTS_MODEL).to(device).eval()
        logger.info("TTS model loaded successfully")
        return model

    def _on_unload(self, _model: Any) -> None:
        """Cleanup hook fired by ModelHost when the model is dropped."""
        self._pipelines.clear()
        self.voice_manager.clear_memory_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _pipeline_for(self, model: Any, voice: str) -> Any:
        """Get or create the phonemization pipeline for the voice's accent."""
        from kokoro import KPipeline

        lang_code = "b" if voice.startswith("b") else "a"
        pipeline = self._pipelines.get(lang_code)
        if pipeline is None:
            pipeline = KPipeline(lang_code=lang_code, model=model, repo_id=TTS_MODEL)
            self._pipelines[lang_code] = pipeline
        return pipeline

    def _generate(
        self,
        model: Any,
        text: str,
        voice: str | None,
        speed: float | None,
    ) -> Iterator[np.ndarray]:
        voice = (voice or self.config.default_voice).lower()
        speed = speed if speed is not None else self.config.speed

        pipeline = self._pipeline_for(model, voice)
        voice_tensor = self.voice_manager.load_voice_tensor(voice)

        for result in pipeline(text, voice=voice_tensor, speed=speed, split_pattern=SPLIT_PATTERN):
            _graphemes, _phonemes, audio = result
            if audio is None:
                continue
            if torch.is_tensor(audio):
                audio = audio.float().cpu().numpy()
            yield np.squeeze(audio).astype(np.float32)

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
    ) -> np.ndarray:
        """Generate audio from text (batch mode).

        Args:
            text: Text to synthesize
            voice: Voice name (default from config)
            speed: Speech rate multiplier (default from config)

        Returns:
            Audio samples as numpy array (float32, 24kHz)
        """
        with self._host.use() as model:
            self._last_used = time.time()
            logger.info(f"Synthesizing: {text[:50]}...")
            start_time = time.time()
            chunks = list(self._generate(model, text, voice, speed))
            gen_time = time.time() - start_time

        if not chunks:
            raise RuntimeError("No audio output generated")

        audio = np.concatenate(chunks)
        duration = len(audio) / SAMPLE_RATE
        rtf = gen_time / duration if duration > 0 else float("inf")
        logger.info(f"Synthesized {duration:.2f}s audio in {gen_time:.2f}s (RTF: {rtf:.2f}x)")
        return audio

    def synthesize_streaming(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
    ) -> Iterator[np.ndarray]:
        """Generate audio chunks for streaming, one per sentence.

        Args:
            text: Text to synthesize
            voice: Voice name (default from config)
            speed: Speech rate multiplier (default from config)

        Yields:
            Audio chunks as numpy arrays (float32, 24kHz)
        """
        # Hold a lease for the entire streaming call so the host cannot unload
        # the model while we're producing chunks.
        with self._host.use() as model:
            self._last_used = time.time()
            yield from self._generate(model, text, voice, speed)

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._host.is_loaded

    @property
    def device(self) -> str | None:
        """Get the current device."""
        return self._device

    @property
    def sample_rate(self) -> int:
        """Get the audio sample rate."""
        return SAMPLE_RATE

    def get_status(self) -> dict:
        """Get synthesizer status."""
        return {
            "model_loaded": self.is_loaded,
            "model": TTS_MODEL,
            "device": self._device,
            "default_voice": self.config.default_voice,
            "speed": self.config.speed,
            "unload_timeout_seconds": self.config.unload_timeout_seconds,
            "last_used": self._last_used,
            "voices_downloaded": self.voice_manager.list_downloaded(),
        }

    def shutdown(self):
        """Shutdown synthesizer and cleanup resources."""
        self._host.shutdown()
