"""Configuration management for voiced."""

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


@dataclass
class AudioConfig:
    """Audio capture settings."""

    sample_rate: int = 16000
    channels: int = 1
    device: str = "default"
    beep_enabled: bool = True


@dataclass
class VadConfig:
    """Voice Activity Detection settings for faster-whisper."""

    enabled: bool = True  # Enable/disable VAD filtering
    threshold: float = 0.5  # Speech probability (0.0-1.0, lower = more sensitive)
    min_silence_duration_ms: int = 2000  # Silence duration to end speech segment
    speech_pad_ms: int = 400  # Padding around detected speech
    min_speech_duration_ms: int = 250  # Minimum speech segment duration


@dataclass
class DiarizationConfig:
    """Speaker diarization settings."""

    device: str = "auto"  # auto, cuda, cpu
    similarity_threshold: float = 0.5  # Profile matching threshold
    min_segment_duration: float = 0.5  # Minimum segment length for embedding (seconds)
    model: str = "speechbrain/spkrec-ecapa-voxceleb"  # SpeechBrain embedding model
    num_speakers: int | None = None  # Number of speakers (None = auto-detect)
    clustering_threshold: float = 0.7  # Clustering threshold when num_speakers is None
    profiles_path: str | None = None  # Custom profiles directory path


@dataclass
class TTSConfig:
    """Text-to-Speech settings."""

    enabled: bool = True
    model: str = "microsoft/VibeVoice-Realtime-0.5B"
    device: str = "auto"  # auto, cuda, mps, cpu
    default_voice: str = "emma"
    cfg_scale: float = 1.5  # Classifier-free guidance scale
    unload_timeout_minutes: int = 60  # Auto-unload model after inactivity (0 = never)


@dataclass
class DaemonConfig:
    """Daemon settings."""

    http_enabled: bool = False  # Start HTTP server alongside Unix socket
    http_host: str | None = None  # Override server.host for daemon (None = use server.host)
    http_port: int | None = None  # Override server.port for daemon (None = use server.port)


@dataclass
class ServerConfig:
    """HTTP server settings."""

    host: str = "127.0.0.1"
    port: int = 8765


@dataclass
class ClientConfig:
    """HTTP client settings."""

    server_url: str = "http://127.0.0.1:8765"
    timeout: float = 60.0


@dataclass
class Config:
    """Main configuration container."""

    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    client: ClientConfig = field(default_factory=ClientConfig)


def get_config_path() -> Path:
    """Get the path to the configuration file."""
    xdg_config = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    return Path(xdg_config) / "voiced" / "config.toml"


def get_cache_dir() -> Path:
    """Get the cache directory for voiced."""
    xdg_cache = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    cache_dir = Path(xdg_cache) / "voiced"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_socket_path() -> Path:
    """Get the path to the Unix domain socket."""
    return get_cache_dir() / "control.sock"


def get_pid_path() -> Path:
    """Get the path to the PID file."""
    return get_cache_dir() / "daemon.pid"


def get_profiles_dir(override_path: str | None = None) -> Path:
    """Get the directory for voice profiles.

    Args:
        override_path: Optional custom path to use instead of the default.
                      If provided, this path will be used directly.

    Returns:
        Path to the profiles directory (created if it doesn't exist).
    """
    if override_path is not None:
        profiles_dir = Path(override_path)
    else:
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

    if "vad" in data:
        for key, value in data["vad"].items():
            if hasattr(config.vad, key):
                setattr(config.vad, key, value)

    if "diarization" in data:
        for key, value in data["diarization"].items():
            if hasattr(config.diarization, key):
                setattr(config.diarization, key, value)

    if "daemon" in data:
        for key, value in data["daemon"].items():
            if hasattr(config.daemon, key):
                setattr(config.daemon, key, value)

    if "server" in data:
        for key, value in data["server"].items():
            if hasattr(config.server, key):
                setattr(config.server, key, value)

    if "tts" in data:
        for key, value in data["tts"].items():
            if hasattr(config.tts, key):
                setattr(config.tts, key, value)

    if "client" in data:
        for key, value in data["client"].items():
            if hasattr(config.client, key):
                setattr(config.client, key, value)

    return config


def get_server_url(cli_url: str | None = None) -> str:
    """Get the server URL from CLI, env, or config (in priority order)."""
    if cli_url:
        return cli_url

    env_url = os.environ.get("STTD_SERVER_URL")
    if env_url:
        return env_url

    config = load_config()
    return config.client.server_url


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

[audio]
sample_rate = 16000
channels = 1
device = "default"       # or specific device name
beep_enabled = true      # audio feedback on start/stop

[vad]
enabled = true           # Enable voice activity detection
threshold = 0.5          # Speech probability threshold (0.0-1.0, lower = more sensitive)
min_silence_duration_ms = 2000  # Silence duration to end speech segment
speech_pad_ms = 400      # Padding around detected speech
min_speech_duration_ms = 250    # Minimum speech segment duration

[diarization]
device = "auto"          # auto, cuda, cpu
similarity_threshold = 0.5  # Profile matching threshold (0-1)
min_segment_duration = 0.5  # Minimum segment length for embedding (seconds)
# model = "speechbrain/spkrec-ecapa-voxceleb"  # SpeechBrain embedding model
# num_speakers = 2       # Set if known, leave unset for auto-detect
# clustering_threshold = 0.7  # Clustering threshold when num_speakers is None
# profiles_path = "/path/to/profiles"  # Custom profiles directory

[tts]
enabled = true           # Enable TTS (requires VibeVoice)
model = "microsoft/VibeVoice-Realtime-0.5B"
device = "auto"          # auto, cuda, mps, cpu
default_voice = "emma"   # carter, davis, emma, frank, grace, mike
cfg_scale = 1.5          # Classifier-free guidance scale
unload_timeout_minutes = 60  # Auto-unload after inactivity (0 = never)

[daemon]
http_enabled = false     # Start HTTP server alongside Unix socket
# http_host = "0.0.0.0"  # Override server.host for daemon HTTP
# http_port = 8765       # Override server.port for daemon HTTP

[server]
host = "127.0.0.1"       # 0.0.0.0 to accept remote connections
port = 8765

[client]
server_url = "http://127.0.0.1:8765"
timeout = 60.0           # Request timeout in seconds
"""
    with open(config_path, "w") as f:
        f.write(default_config)
