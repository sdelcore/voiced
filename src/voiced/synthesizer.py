"""TTS synthesizer using VibeVoice."""

import copy
import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch

from voiced.voice_manager import VoiceManager

logger = logging.getLogger(__name__)

# Default TTS settings
DEFAULT_MODEL_PATH = "microsoft/VibeVoice-Realtime-0.5B"
DEFAULT_VOICE = "emma"
DEFAULT_CFG_SCALE = 1.5
DEFAULT_UNLOAD_TIMEOUT = 3600  # 1 hour in seconds
SAMPLE_RATE = 24000


@dataclass
class TTSConfig:
    """Configuration for TTS synthesizer."""

    model_path: str = DEFAULT_MODEL_PATH
    device: str = "auto"
    default_voice: str = DEFAULT_VOICE
    cfg_scale: float = DEFAULT_CFG_SCALE
    unload_timeout_seconds: int = DEFAULT_UNLOAD_TIMEOUT


def check_vibevoice_installed() -> bool:
    """Check if VibeVoice is installed."""
    try:
        import importlib.util

        return (
            importlib.util.find_spec("vibevoice.modular.modeling_vibevoice_streaming_inference")
            is not None
            and importlib.util.find_spec("vibevoice.processor.vibevoice_streaming_processor")
            is not None
        )
    except ImportError:
        return False


def get_device_and_dtype() -> tuple[str, torch.dtype, str]:
    """Determine the best device and dtype for the current system."""
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16, "flash_attention_2"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float32, "sdpa"
    else:
        return "cpu", torch.float32, "sdpa"


class Synthesizer:
    """VibeVoice TTS engine with lazy loading and auto-unload."""

    def __init__(self, config: TTSConfig | None = None):
        """Initialize synthesizer.

        Args:
            config: TTS configuration. If None, uses defaults.
        """
        self.config = config or TTSConfig()
        self.model = None
        self.processor = None
        self.voice_manager = VoiceManager()
        self._lock = threading.Lock()
        self._last_used: float | None = None
        self._unload_timer: threading.Timer | None = None
        self._device: str | None = None
        self._dtype: torch.dtype | None = None
        self._attn_impl: str | None = None

    def _ensure_loaded(self):
        """Lazy load model on first use."""
        with self._lock:
            if self.model is None:
                self._load_model()
            self._reset_unload_timer()

    def _load_model(self):
        """Load VibeVoice model from HuggingFace."""
        if not check_vibevoice_installed():
            raise RuntimeError(
                "VibeVoice is not installed. Please install it with:\n"
                "  pip install git+https://github.com/microsoft/VibeVoice.git"
            )

        from vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_streaming_processor import (
            VibeVoiceStreamingProcessor,
        )

        # Determine device
        if self.config.device == "auto":
            self._device, self._dtype, self._attn_impl = get_device_and_dtype()
        elif self.config.device == "cuda":
            self._device = "cuda"
            self._dtype = torch.bfloat16
            self._attn_impl = "flash_attention_2"
        elif self.config.device == "mps":
            self._device = "mps"
            self._dtype = torch.float32
            self._attn_impl = "sdpa"
        else:
            self._device = "cpu"
            self._dtype = torch.float32
            self._attn_impl = "sdpa"

        logger.info(f"Loading TTS processor from {self.config.model_path}...")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.config.model_path)

        logger.info(
            f"Loading TTS model from {self.config.model_path} "
            f"(device={self._device}, dtype={self._dtype})..."
        )

        try:
            if self._device == "mps":
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.config.model_path,
                    torch_dtype=self._dtype,
                    attn_implementation=self._attn_impl,
                    device_map=None,
                )
                self.model.to("mps")
            else:
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.config.model_path,
                    torch_dtype=self._dtype,
                    device_map=self._device,
                    attn_implementation=self._attn_impl,
                )
        except Exception as e:
            if self._attn_impl == "flash_attention_2":
                logger.warning(f"Flash attention failed: {e}")
                logger.info("Falling back to SDPA attention...")
                self._attn_impl = "sdpa"
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.config.model_path,
                    torch_dtype=self._dtype,
                    device_map=self._device if self._device != "mps" else None,
                    attn_implementation="sdpa",
                )
                if self._device == "mps":
                    self.model.to("mps")
            else:
                raise

        self.model.eval()
        self.model.set_ddpm_inference_steps(num_steps=5)
        logger.info("TTS model loaded successfully")

    def _unload_model(self):
        """Unload model to free GPU memory."""
        with self._lock:
            if self.model is not None:
                logger.info("Unloading TTS model due to inactivity...")
                self.model = None
                self.processor = None
                self.voice_manager.clear_memory_cache()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("TTS model unloaded")

    def _reset_unload_timer(self):
        """Reset the auto-unload timer."""
        self._last_used = time.time()

        if self._unload_timer is not None:
            self._unload_timer.cancel()

        if self.config.unload_timeout_seconds > 0:
            self._unload_timer = threading.Timer(
                self.config.unload_timeout_seconds,
                self._unload_model,
            )
            self._unload_timer.daemon = True
            self._unload_timer.start()

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        cfg_scale: float | None = None,
    ) -> np.ndarray:
        """Generate audio from text (batch mode).

        Args:
            text: Text to synthesize
            voice: Voice preset name (default from config)
            cfg_scale: Classifier-free guidance scale (default from config)

        Returns:
            Audio samples as numpy array (float32, 24kHz)
        """
        self._ensure_loaded()

        voice = voice or self.config.default_voice
        cfg_scale = cfg_scale if cfg_scale is not None else self.config.cfg_scale

        # Load voice cache
        voice_cache = self.voice_manager.load_voice_cache(voice, self._device)

        # Prepare inputs
        inputs = self.processor.process_input_with_cached_prompt(
            text=text,
            cached_prompt=voice_cache,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Move tensors to device
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self._device)

        # Generate
        logger.info(f"Synthesizing: {text[:50]}...")
        start_time = time.time()

        outputs = self.model.generate(
            **inputs,
            cfg_scale=cfg_scale,
            tokenizer=self.processor.tokenizer,
            generation_config={"do_sample": False},
            verbose=False,
            show_progress_bar=False,
            all_prefilled_outputs=copy.deepcopy(voice_cache),
        )

        gen_time = time.time() - start_time

        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            audio = outputs.speech_outputs[0]
            if torch.is_tensor(audio):
                audio = audio.float().cpu().numpy()
            audio = np.squeeze(audio).astype(np.float32)

            duration = len(audio) / SAMPLE_RATE
            rtf = gen_time / duration if duration > 0 else float("inf")
            logger.info(f"Synthesized {duration:.2f}s audio in {gen_time:.2f}s (RTF: {rtf:.2f}x)")
            return audio
        else:
            raise RuntimeError("No audio output generated")

    def synthesize_streaming(
        self,
        text: str,
        voice: str | None = None,
        cfg_scale: float | None = None,
    ) -> Iterator[np.ndarray]:
        """Generate audio chunks for streaming.

        Args:
            text: Text to synthesize
            voice: Voice preset name (default from config)
            cfg_scale: Classifier-free guidance scale (default from config)

        Yields:
            Audio chunks as numpy arrays (float32, 24kHz)
        """
        from vibevoice.modular.streamer import AudioStreamer

        self._ensure_loaded()

        voice = voice or self.config.default_voice
        cfg_scale = cfg_scale if cfg_scale is not None else self.config.cfg_scale

        # Load voice cache
        voice_cache = self.voice_manager.load_voice_cache(voice, self._device)

        # Prepare inputs
        inputs = self.processor.process_input_with_cached_prompt(
            text=text,
            cached_prompt=voice_cache,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self._device)

        # Create audio streamer
        audio_streamer = AudioStreamer(batch_size=1, stop_signal=None)
        generation_error = []

        def generate_thread():
            try:
                self.model.generate(
                    **inputs,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={"do_sample": False},
                    verbose=False,
                    show_progress_bar=False,
                    audio_streamer=audio_streamer,
                    all_prefilled_outputs=copy.deepcopy(voice_cache),
                )
            except Exception as e:
                generation_error.append(e)
                audio_streamer.end()

        # Start generation in background
        thread = threading.Thread(target=generate_thread, daemon=True)
        thread.start()

        # Yield chunks as they arrive
        try:
            for chunk in audio_streamer.get_stream(0):
                if torch.is_tensor(chunk):
                    chunk = chunk.float().cpu().numpy()
                chunk = np.squeeze(chunk).astype(np.float32)
                yield chunk
        finally:
            thread.join(timeout=5)
            if generation_error:
                raise generation_error[0]

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self.model is not None

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
            "model_path": self.config.model_path,
            "device": self._device,
            "default_voice": self.config.default_voice,
            "cfg_scale": self.config.cfg_scale,
            "unload_timeout_seconds": self.config.unload_timeout_seconds,
            "last_used": self._last_used,
            "voices_downloaded": self.voice_manager.list_downloaded(),
        }

    def shutdown(self):
        """Shutdown synthesizer and cleanup resources."""
        if self._unload_timer is not None:
            self._unload_timer.cancel()
        self._unload_model()
