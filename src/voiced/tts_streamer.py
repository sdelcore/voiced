"""Audio streaming playback for TTS output."""

import logging
import queue
import threading
import time
from collections.abc import Iterator

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

# VibeVoice outputs 24kHz audio
SAMPLE_RATE = 24000


class AudioPlayer:
    """Play audio using sounddevice."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        """Initialize audio player.

        Args:
            sample_rate: Audio sample rate (default 24000 for VibeVoice)
        """
        self.sample_rate = sample_rate

    def play(self, audio: np.ndarray, block: bool = True) -> None:
        """Play audio samples.

        Args:
            audio: Audio samples as numpy array (float32)
            block: Wait for playback to complete (default True)
        """
        if len(audio) == 0:
            return

        # Ensure float32
        audio = audio.astype(np.float32)

        sd.play(audio, self.sample_rate)
        if block:
            sd.wait()

    def stop(self) -> None:
        """Stop any current playback."""
        sd.stop()


class StreamingAudioPlayer:
    """Low-latency streaming audio player using callback-based output."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        buffer_size: int = 1024,
        silence_padding_ms: int = 250,
    ):
        """Initialize streaming audio player.

        Args:
            sample_rate: Audio sample rate (default 24000 for VibeVoice)
            buffer_size: Audio buffer size in samples
            silence_padding_ms: Silence to append at end to prevent cutoff
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.silence_padding_samples = int(sample_rate * silence_padding_ms / 1000)

        self._audio_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._stream: sd.OutputStream | None = None
        self._buffer: np.ndarray = np.array([], dtype=np.float32)
        self._lock = threading.Lock()
        self._finished = threading.Event()
        self._started = threading.Event()
        self._first_chunk_time: float | None = None

    def _audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        """Callback for sounddevice OutputStream."""
        if status:
            logger.warning(f"Audio output status: {status}")

        with self._lock:
            # Fill buffer from queue if needed
            while len(self._buffer) < frames:
                try:
                    chunk = self._audio_queue.get_nowait()
                    if chunk is None:
                        # End of stream - fill remaining with silence
                        if len(self._buffer) > 0:
                            outdata[: len(self._buffer), 0] = self._buffer
                            outdata[len(self._buffer) :, 0] = 0
                        else:
                            outdata[:, 0] = 0
                        self._buffer = np.array([], dtype=np.float32)
                        self._finished.set()
                        return
                    self._buffer = np.concatenate([self._buffer, chunk])
                except queue.Empty:
                    break

            # Output audio from buffer
            if len(self._buffer) >= frames:
                outdata[:, 0] = self._buffer[:frames]
                self._buffer = self._buffer[frames:]
            elif len(self._buffer) > 0:
                # Partial buffer - pad with silence
                outdata[: len(self._buffer), 0] = self._buffer
                outdata[len(self._buffer) :, 0] = 0
                self._buffer = np.array([], dtype=np.float32)
            else:
                # No data - output silence
                outdata[:, 0] = 0

    def start(self) -> None:
        """Start the audio output stream."""
        self._finished.clear()
        self._started.clear()
        self._first_chunk_time = None
        self._buffer = np.array([], dtype=np.float32)

        # Clear any leftover data in queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=self.buffer_size,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._started.set()

    def write(self, audio: np.ndarray) -> None:
        """Write audio chunk to playback queue.

        Args:
            audio: Audio samples as numpy array (float32)
        """
        if self._first_chunk_time is None:
            self._first_chunk_time = time.time()

        audio = audio.astype(np.float32)
        self._audio_queue.put(audio)

    def finish(self) -> None:
        """Signal end of audio stream and add silence padding."""
        # Add silence padding to prevent audio cutoff
        silence = np.zeros(self.silence_padding_samples, dtype=np.float32)
        self._audio_queue.put(silence)
        self._audio_queue.put(None)  # End signal

    def wait(self) -> None:
        """Wait for playback to complete."""
        self._finished.wait()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def stop(self) -> None:
        """Stop playback immediately."""
        self._finished.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @property
    def first_chunk_latency_ms(self) -> float | None:
        """Get latency from start to first chunk (if available)."""
        return self._first_chunk_time

    def play_stream(self, audio_chunks: Iterator[np.ndarray]) -> None:
        """Play audio from an iterator of chunks.

        This is a convenience method that handles start/write/finish/wait.

        Args:
            audio_chunks: Iterator yielding audio chunks
        """
        self.start()
        try:
            for chunk in audio_chunks:
                self.write(chunk)
            self.finish()
            self.wait()
        except Exception:
            self.stop()
            raise


def play_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Simple function to play audio.

    Args:
        audio: Audio samples as numpy array
        sample_rate: Sample rate (default 24000)
    """
    player = AudioPlayer(sample_rate)
    player.play(audio, block=True)


def play_streaming(
    audio_chunks: Iterator[np.ndarray],
    sample_rate: int = SAMPLE_RATE,
) -> None:
    """Play audio from streaming chunks with low latency.

    Args:
        audio_chunks: Iterator yielding audio chunks
        sample_rate: Sample rate (default 24000)
    """
    player = StreamingAudioPlayer(sample_rate)
    player.play_stream(audio_chunks)
