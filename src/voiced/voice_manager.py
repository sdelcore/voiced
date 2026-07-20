"""Voice preset management for TTS (Kokoro voice packs).

torch is imported lazily in ``load_voice_tensor`` so listing/downloading
voices from the parent process does not pull torch in.
"""

import logging
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# English voice packs from the Kokoro-82M repository.
# Prefix encodes accent + gender: a=American, b=British; f=female, m=male.
AVAILABLE_VOICES = [
    "af_alloy",
    "af_aoede",
    "af_bella",
    "af_heart",
    "af_jessica",
    "af_kore",
    "af_nicole",
    "af_nova",
    "af_river",
    "af_sarah",
    "af_sky",
    "am_adam",
    "am_echo",
    "am_eric",
    "am_fenrir",
    "am_liam",
    "am_michael",
    "am_onyx",
    "am_puck",
    "am_santa",
    "bf_alice",
    "bf_emma",
    "bf_isabella",
    "bf_lily",
    "bm_daniel",
    "bm_fable",
    "bm_george",
    "bm_lewis",
]

VOICE_BASE_URL = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices"


class VoiceManager:
    """Manage voice pack downloads and caching."""

    def __init__(self, cache_dir: Path | None = None):
        """Initialize voice manager.

        Args:
            cache_dir: Directory for caching voice packs.
                       Defaults to ~/.cache/voiced/voices/
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "voiced" / "voices"
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._voice_cache: dict[str, Any] = {}

    def _filename(self, voice: str) -> str:
        return f"{voice}.pt"

    def list_available(self) -> list[str]:
        """List all available voices."""
        return sorted(AVAILABLE_VOICES)

    def list_downloaded(self) -> list[str]:
        """List locally cached voices."""
        return sorted(
            name for name in AVAILABLE_VOICES if (self.cache_dir / self._filename(name)).exists()
        )

    def is_downloaded(self, voice: str) -> bool:
        """Check if voice is cached locally."""
        voice = voice.lower()
        if voice not in AVAILABLE_VOICES:
            return False
        return (self.cache_dir / self._filename(voice)).exists()

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

        filename = self._filename(voice)
        path = self.cache_dir / filename
        downloaded = path.exists()

        info = {
            "name": voice,
            "filename": filename,
            "accent": "British" if voice.startswith("b") else "American",
            "gender": "female" if voice[1] == "f" else "male",
            "downloaded": downloaded,
        }

        if downloaded:
            info["size_bytes"] = path.stat().st_size
            info["path"] = str(path)

        return info

    def download(self, voice: str, force: bool = False) -> Path:
        """Download a voice pack from the Kokoro-82M repository.

        Args:
            voice: Voice name
            force: Re-download even if already cached

        Returns:
            Path to downloaded voice pack
        """
        voice = voice.lower()
        if voice not in AVAILABLE_VOICES:
            raise ValueError(
                f"Unknown voice '{voice}'. Available: {', '.join(self.list_available())}"
            )

        filename = self._filename(voice)
        cache_path = self.cache_dir / filename

        if cache_path.exists() and not force:
            logger.info(f"Voice pack already cached: {cache_path}")
            return cache_path

        url = f"{VOICE_BASE_URL}/{filename}"
        logger.info(f"Downloading voice pack from {url}...")

        response = requests.get(url, timeout=120)
        response.raise_for_status()

        with open(cache_path, "wb") as f:
            f.write(response.content)

        size_kb = len(response.content) / 1024
        logger.info(f"Downloaded voice pack to {cache_path} ({size_kb:.0f} KB)")

        self._voice_cache.pop(voice, None)
        return cache_path

    def remove(self, voice: str) -> bool:
        """Remove cached voice pack.

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

        cache_path = self.cache_dir / self._filename(voice)

        if not cache_path.exists():
            return False

        cache_path.unlink()
        logger.info(f"Removed voice pack: {cache_path}")

        self._voice_cache.pop(voice, None)
        return True

    def get_path(self, voice: str) -> Path:
        """Get path to voice pack, downloading if needed.

        Args:
            voice: Voice name

        Returns:
            Path to voice pack file
        """
        voice = voice.lower()
        if not self.is_downloaded(voice):
            return self.download(voice)
        return self.cache_dir / self._filename(voice)

    def load_voice_tensor(self, voice: str) -> Any:
        """Load a voice pack as a tensor, downloading if needed.

        Args:
            voice: Voice name

        Returns:
            Voice style tensor for the Kokoro pipeline
        """
        import torch

        voice = voice.lower()
        if voice in self._voice_cache:
            return self._voice_cache[voice]

        path = self.get_path(voice)
        logger.info(f"Loading voice pack: {path}")

        tensor = torch.load(path, map_location="cpu", weights_only=True)
        self._voice_cache[voice] = tensor
        return tensor

    def clear_memory_cache(self):
        """Clear all voice packs from memory."""
        self._voice_cache.clear()
        logger.info("Cleared voice pack memory cache")
