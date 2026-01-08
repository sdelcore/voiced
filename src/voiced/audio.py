"""Audio feedback (beep sounds) for voiced."""

import logging

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100


def generate_tone(
    frequency: float,
    duration: float,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = 0.3,
) -> np.ndarray:
    """Generate a sine wave tone.

    Args:
        frequency: Frequency in Hz.
        duration: Duration in seconds.
        sample_rate: Sample rate in Hz.
        amplitude: Volume (0.0 to 1.0).

    Returns:
        Numpy array of audio samples (float32).
    """
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    tone = amplitude * np.sin(2 * np.pi * frequency * t).astype(np.float32)

    # Apply fade in/out to avoid clicks
    fade_samples = int(sample_rate * 0.01)  # 10ms fade
    if fade_samples > 0 and len(tone) > 2 * fade_samples:
        fade_in = np.linspace(0, 1, fade_samples, dtype=np.float32)
        fade_out = np.linspace(1, 0, fade_samples, dtype=np.float32)
        tone[:fade_samples] *= fade_in
        tone[-fade_samples:] *= fade_out

    return tone


def play_tone(
    frequency: float,
    duration: float,
    amplitude: float = 0.3,
    blocking: bool = True,
) -> None:
    """Play a tone.

    Args:
        frequency: Frequency in Hz.
        duration: Duration in seconds.
        amplitude: Volume (0.0 to 1.0).
        blocking: If True, wait for playback to complete.
    """
    tone = generate_tone(frequency, duration, amplitude=amplitude)
    sd.play(tone, SAMPLE_RATE)
    if blocking:
        sd.wait()


def beep_start() -> None:
    """Play the 'recording started' beep (ascending tone)."""
    tone1 = generate_tone(800, 0.08, amplitude=0.25)
    tone2 = generate_tone(1200, 0.08, amplitude=0.25)
    silence = np.zeros(int(SAMPLE_RATE * 0.02), dtype=np.float32)

    audio = np.concatenate([tone1, silence, tone2])
    sd.play(audio, SAMPLE_RATE)
    sd.wait()


def beep_stop() -> None:
    """Play the 'recording stopped' beep (pleasant chime)."""
    tone1 = generate_tone(523, 0.1, amplitude=0.2)  # C5
    tone2 = generate_tone(659, 0.1, amplitude=0.2)  # E5
    tone3 = generate_tone(784, 0.15, amplitude=0.2)  # G5
    silence = np.zeros(int(SAMPLE_RATE * 0.03), dtype=np.float32)

    audio = np.concatenate([tone1, silence, tone2, silence, tone3])
    sd.play(audio, SAMPLE_RATE)
    sd.wait()


def beep_error() -> None:
    """Play an error beep (low buzz)."""
    tone = generate_tone(200, 0.3, amplitude=0.3)
    sd.play(tone, SAMPLE_RATE)
    sd.wait()


def beep_success() -> None:
    """Play a success beep (descending tone)."""
    tone1 = generate_tone(1200, 0.08, amplitude=0.25)
    tone2 = generate_tone(800, 0.08, amplitude=0.25)
    silence = np.zeros(int(SAMPLE_RATE * 0.02), dtype=np.float32)

    audio = np.concatenate([tone1, silence, tone2])
    sd.play(audio, SAMPLE_RATE)
    sd.wait()
