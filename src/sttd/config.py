"""Configuration management for sttd."""

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib


@dataclass
class TranscriptionConfig:
    """Transcription settings."""

    model: str = "base"
    device: str = "auto"
    compute_type: str = "auto"
    language: str = "en"
    streaming: bool = True
    chunk_duration: float = 1.0  # Seconds per chunk (reduced from 2.0 for faster feedback)
    max_window: float = 30.0  # Max seconds of audio in sliding window
    beam_size: int = 1  # Beam size for streaming (1 = greedy decoding for speed)
    context_words: int = 200  # Words to keep as initial_prompt context after buffer trim
    min_confirmed_trim: float = 5.0  # Min confirmed audio seconds before trimming buffer
    trim_overlap: float = 3.0  # Seconds of confirmed audio to keep as overlap after trim


@dataclass
class AudioConfig:
    """Audio capture settings."""

    sample_rate: int = 16000
    channels: int = 1
    device: str = "default"
    beep_enabled: bool = True


@dataclass
class OutputConfig:
    """Output settings."""

    method: str = "wtype"  # wtype, clipboard, both


@dataclass
class DiarizationConfig:
    """Speaker identification settings."""

    device: str = "auto"  # auto, cuda, cpu
    similarity_threshold: float = 0.5  # Speaker matching threshold
    min_segment_duration: float = 0.5  # Minimum segment length for embedding (seconds)
    model: str = "speechbrain/spkrec-ecapa-voxceleb"  # SpeechBrain embedding model


@dataclass
class Config:
    """Main configuration container."""

    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)


def get_config_path() -> Path:
    """Get the path to the configuration file."""
    xdg_config = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    return Path(xdg_config) / "sttd" / "config.toml"


def get_cache_dir() -> Path:
    """Get the cache directory for sttd."""
    xdg_cache = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    cache_dir = Path(xdg_cache) / "sttd"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_socket_path() -> Path:
    """Get the path to the Unix domain socket."""
    return get_cache_dir() / "control.sock"


def get_pid_path() -> Path:
    """Get the path to the PID file."""
    return get_cache_dir() / "daemon.pid"


def get_profiles_dir() -> Path:
    """Get the directory for voice profiles."""
    config_dir = get_config_path().parent
    profiles_dir = config_dir / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    return profiles_dir


def load_config() -> Config:
    """Load configuration from file, falling back to defaults."""
    config_path = get_config_path()

    if not config_path.exists():
        return Config()

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    config = Config()

    if "transcription" in data:
        for key, value in data["transcription"].items():
            if hasattr(config.transcription, key):
                setattr(config.transcription, key, value)

    if "audio" in data:
        for key, value in data["audio"].items():
            if hasattr(config.audio, key):
                setattr(config.audio, key, value)

    if "output" in data:
        for key, value in data["output"].items():
            if hasattr(config.output, key):
                setattr(config.output, key, value)

    if "diarization" in data:
        for key, value in data["diarization"].items():
            if hasattr(config.diarization, key):
                setattr(config.diarization, key, value)

    return config


def save_default_config() -> None:
    """Save a default configuration file."""
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    default_config = """\
[transcription]
model = "base"           # tiny, base, small, medium, large-v3
device = "auto"          # auto, cuda, cpu
compute_type = "auto"    # auto, float16, int8, float32
language = "en"
streaming = true
chunk_duration = 1.0     # Seconds per chunk
max_window = 30.0        # Max seconds in sliding window
beam_size = 1            # Beam size for streaming (1 = greedy for speed)
context_words = 200      # Words to keep as context after buffer trim

[audio]
sample_rate = 16000
channels = 1
device = "default"       # or specific device name
beep_enabled = true      # audio feedback on start/stop

[output]
method = "wtype"         # wtype, clipboard, both

[diarization]
device = "auto"          # auto, cuda, cpu
similarity_threshold = 0.5  # Speaker matching threshold (0-1)
min_segment_duration = 0.5  # Minimum segment length for embedding (seconds)
# model = "speechbrain/spkrec-ecapa-voxceleb"  # SpeechBrain embedding model
"""
    with open(config_path, "w") as f:
        f.write(default_config)
