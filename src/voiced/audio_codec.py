"""WAV ↔ numpy audio conversion."""

import io
import wave

import numpy as np


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to float32 in [-1, 1].

    Detects int16-scaled input (values outside [-1, 1]) and rescales by 1/32768.
    """
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if audio.size and (audio.max() > 1.0 or audio.min() < -1.0):
        audio = audio / 32768.0
    return audio


def audio_to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert a float32 mono audio array to WAV bytes (16-bit PCM)."""
    audio = normalize_audio(audio)
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_int16.tobytes())
    return buffer.getvalue()


def wav_to_audio(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV bytes to a float32 mono audio array and sample rate.

    Supports 16-bit PCM and 32-bit float input. Multi-channel input is
    downmixed to mono by averaging channels.
    """
    buffer = io.BytesIO(wav_bytes)
    with wave.open(buffer, "rb") as wav:
        sample_rate = wav.getframerate()
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        n_frames = wav.getnframes()
        audio_bytes = wav.readframes(n_frames)

    if sample_width == 2:
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(audio_bytes, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    return audio, sample_rate
