"""Audio recording using sounddevice."""

import logging
import queue
import threading
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from voiced.config import AudioConfig

logger = logging.getLogger(__name__)


class Recorder:
    """Audio recorder using sounddevice for microphone capture."""

    def __init__(
        self,
        config: AudioConfig | None = None,
        on_chunk: Callable[[np.ndarray], None] | None = None,
        chunk_duration: float = 2.0,
    ):
        """Initialize the recorder.

        Args:
            config: Audio configuration. Uses defaults if not provided.
            on_chunk: Callback invoked with audio data when a chunk is ready.
                     Used for streaming transcription.
            chunk_duration: Duration of each chunk in seconds (default 2.0s).
        """
        self.config = config or AudioConfig()
        self._on_chunk = on_chunk
        self._chunk_duration = chunk_duration
        self._chunk_samples = int(self._chunk_duration * self.config.sample_rate)
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._is_recording = False
        self._stream: sd.InputStream | None = None
        self._recording_thread: threading.Thread | None = None
        self._audio_buffer: list[np.ndarray] = []
        self._chunk_buffer: list[np.ndarray] = []

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Callback for audio stream."""
        if status:
            logger.warning(f"Audio callback status: {status}")
        if self._is_recording:
            self._audio_queue.put(indata.copy())

    def start(self) -> None:
        """Start recording audio."""
        if self._is_recording:
            logger.warning("Already recording")
            return

        self._audio_buffer = []
        self._chunk_buffer = []
        self._is_recording = True

        # Clear any pending audio in queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        # Get the device
        device = None if self.config.device == "default" else self.config.device

        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype=np.float32,
            callback=self._audio_callback,
            device=device,
        )
        self._stream.start()

        # Start buffer collection thread
        self._recording_thread = threading.Thread(target=self._collect_audio, daemon=True)
        self._recording_thread.start()

        logger.info(
            f"Recording started: {self.config.sample_rate}Hz, {self.config.channels} channel(s)"
        )

    def _collect_audio(self) -> None:
        """Collect audio from the queue into the buffer."""
        while self._is_recording:
            try:
                data = self._audio_queue.get(timeout=0.1)
                self._audio_buffer.append(data)

                # Also collect for chunk callback (streaming transcription)
                if self._on_chunk is not None:
                    self._chunk_buffer.append(data)

                    # Calculate total samples in chunk buffer
                    total_samples = sum(len(c) for c in self._chunk_buffer)
                    if total_samples >= self._chunk_samples:
                        # Emit the chunk
                        chunk_audio = np.concatenate(self._chunk_buffer, axis=0)
                        # Convert to mono if needed
                        if len(chunk_audio.shape) > 1 and chunk_audio.shape[1] > 1:
                            chunk_audio = np.mean(chunk_audio, axis=1)
                        elif len(chunk_audio.shape) > 1:
                            chunk_audio = chunk_audio.flatten()

                        try:
                            self._on_chunk(chunk_audio)
                        except Exception as e:
                            logger.warning(f"Chunk callback error: {e}")

                        self._chunk_buffer = []

            except queue.Empty:
                continue

    def stop(self) -> np.ndarray:
        """Stop recording and return the captured audio.

        Returns:
            Numpy array of recorded audio (float32, mono).
        """
        if not self._is_recording:
            logger.warning("Not recording")
            return np.array([], dtype=np.float32)

        self._is_recording = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Wait for collection thread to finish
        if self._recording_thread is not None:
            self._recording_thread.join(timeout=1.0)
            self._recording_thread = None

        # Drain any remaining audio from the queue
        while not self._audio_queue.empty():
            try:
                data = self._audio_queue.get_nowait()
                self._audio_buffer.append(data)
                if self._on_chunk is not None:
                    self._chunk_buffer.append(data)
            except queue.Empty:
                break

        # Flush remaining chunk buffer for streaming
        if self._on_chunk is not None and self._chunk_buffer:
            chunk_audio = np.concatenate(self._chunk_buffer, axis=0)
            if len(chunk_audio.shape) > 1 and chunk_audio.shape[1] > 1:
                chunk_audio = np.mean(chunk_audio, axis=1)
            elif len(chunk_audio.shape) > 1:
                chunk_audio = chunk_audio.flatten()
            try:
                self._on_chunk(chunk_audio)
            except Exception as e:
                logger.warning(f"Final chunk callback error: {e}")
            self._chunk_buffer = []

        # Concatenate all audio chunks
        if not self._audio_buffer:
            logger.warning("No audio recorded")
            return np.array([], dtype=np.float32)

        audio = np.concatenate(self._audio_buffer, axis=0)

        # Convert to mono if stereo
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
        elif len(audio.shape) > 1:
            audio = audio.flatten()

        duration = len(audio) / self.config.sample_rate
        logger.info(f"Recording stopped: {duration:.2f}s of audio captured")

        return audio

    def get_devices(self) -> list[dict]:
        """Get a list of available audio input devices.

        Returns:
            List of device information dictionaries.
        """
        devices = sd.query_devices()
        input_devices = []

        for i, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                input_devices.append(
                    {
                        "index": i,
                        "name": device["name"],
                        "channels": device["max_input_channels"],
                        "sample_rate": device["default_samplerate"],
                    }
                )

        return input_devices
