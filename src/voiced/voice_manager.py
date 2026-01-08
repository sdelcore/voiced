"""Voice preset management for TTS."""

import logging
from pathlib import Path

import requests
import torch

logger = logging.getLogger(__name__)

# Available voice presets from VibeVoice repository
AVAILABLE_VOICES = {
    "carter": "en-Carter_man.pt",
    "davis": "en-Davis_man.pt",
    "emma": "en-Emma_woman.pt",
    "frank": "en-Frank_man.pt",
    "grace": "en-Grace_woman.pt",
    "mike": "en-Mike_man.pt",
}

# Base URL for downloading voice presets
VOICE_PRESET_BASE_URL = (
    "https://raw.githubusercontent.com/microsoft/VibeVoice/main/demo/voices/streaming_model"
)


class VoiceManager:
    """Manage voice preset downloads and caching."""

    def __init__(self, cache_dir: Path | None = None):
        """Initialize voice manager.

        Args:
            cache_dir: Directory for caching voice presets.
                       Defaults to ~/.cache/voiced/voices/
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "voiced" / "voices"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._voice_cache: dict[str, dict] = {}

    def list_available(self) -> list[str]:
        """List all available voices."""
        return sorted(AVAILABLE_VOICES.keys())

    def list_downloaded(self) -> list[str]:
        """List locally cached voices."""
        downloaded = []
        for name, filename in AVAILABLE_VOICES.items():
            if (self.cache_dir / filename).exists():
                downloaded.append(name)
        return sorted(downloaded)

    def is_downloaded(self, voice: str) -> bool:
        """Check if voice is cached locally."""
        voice = voice.lower()
        if voice not in AVAILABLE_VOICES:
            return False
        return (self.cache_dir / AVAILABLE_VOICES[voice]).exists()

    def get_voice_info(self, voice: str) -> dict:
        """Get information about a voice.

        Args:
            voice: Voice name

        Returns:
            Dictionary with voice information
        """
        voice = voice.lower()
        if voice not in AVAILABLE_VOICES:
            raise ValueError(
                f"Unknown voice '{voice}'. Available: {', '.join(self.list_available())}"
            )

        filename = AVAILABLE_VOICES[voice]
        path = self.cache_dir / filename
        downloaded = path.exists()

        info = {
            "name": voice,
            "filename": filename,
            "downloaded": downloaded,
        }

        if downloaded:
            info["size_bytes"] = path.stat().st_size
            info["path"] = str(path)

        return info

    def download(self, voice: str, force: bool = False) -> Path:
        """Download voice preset from GitHub.

        Args:
            voice: Voice name
            force: Re-download even if already cached

        Returns:
            Path to downloaded voice preset
        """
        voice = voice.lower()
        if voice not in AVAILABLE_VOICES:
            raise ValueError(
                f"Unknown voice '{voice}'. Available: {', '.join(self.list_available())}"
            )

        filename = AVAILABLE_VOICES[voice]
        cache_path = self.cache_dir / filename

        if cache_path.exists() and not force:
            logger.info(f"Voice preset already cached: {cache_path}")
            return cache_path

        url = f"{VOICE_PRESET_BASE_URL}/{filename}"
        logger.info(f"Downloading voice preset from {url}...")

        response = requests.get(url, timeout=120)
        response.raise_for_status()

        with open(cache_path, "wb") as f:
            f.write(response.content)

        size_mb = len(response.content) / (1024 * 1024)
        logger.info(f"Downloaded voice preset to {cache_path} ({size_mb:.1f} MB)")

        # Clear from memory cache if it was cached
        if voice in self._voice_cache:
            del self._voice_cache[voice]

        return cache_path

    def remove(self, voice: str) -> bool:
        """Remove cached voice preset.

        Args:
            voice: Voice name

        Returns:
            True if removed, False if not found
        """
        voice = voice.lower()
        if voice not in AVAILABLE_VOICES:
            raise ValueError(
                f"Unknown voice '{voice}'. Available: {', '.join(self.list_available())}"
            )

        filename = AVAILABLE_VOICES[voice]
        cache_path = self.cache_dir / filename

        if not cache_path.exists():
            return False

        cache_path.unlink()
        logger.info(f"Removed voice preset: {cache_path}")

        # Clear from memory cache
        if voice in self._voice_cache:
            del self._voice_cache[voice]

        return True

    def get_path(self, voice: str) -> Path:
        """Get path to voice preset, downloading if needed.

        Args:
            voice: Voice name

        Returns:
            Path to voice preset file
        """
        voice = voice.lower()
        if not self.is_downloaded(voice):
            return self.download(voice)
        return self.cache_dir / AVAILABLE_VOICES[voice]

    def load_voice_cache(self, voice: str, device: str = "cpu") -> dict:
        """Load voice preset tensor cache.

        Args:
            voice: Voice name
            device: Device to load tensors to

        Returns:
            Dictionary containing cached voice prompt tensors
        """
        voice = voice.lower()

        # Check memory cache
        cache_key = f"{voice}_{device}"
        if cache_key in self._voice_cache:
            return self._voice_cache[cache_key]

        # Load from disk
        path = self.get_path(voice)
        logger.info(f"Loading voice preset: {path}")

        voice_cache = torch.load(path, map_location=device, weights_only=False)
        self._voice_cache[cache_key] = voice_cache

        return voice_cache

    def clear_memory_cache(self):
        """Clear all voice presets from memory."""
        self._voice_cache.clear()
        logger.info("Cleared voice preset memory cache")
