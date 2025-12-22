"""Voice profile management for speaker diarization."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from sttd.config import get_profiles_dir

logger = logging.getLogger(__name__)


@dataclass
class VoiceProfile:
    """A registered speaker's voice profile."""

    name: str
    embedding: list[float]
    created_at: str
    audio_duration: float
    model_version: str

    def embedding_array(self) -> np.ndarray:
        """Return embedding as numpy array."""
        return np.array(self.embedding, dtype=np.float32)


class ProfileManager:
    """Manages voice profiles for speaker diarization."""

    def __init__(self, profiles_dir: Path | str | None = None):
        if profiles_dir is None:
            self.profiles_dir = get_profiles_dir()
        elif isinstance(profiles_dir, str):
            self.profiles_dir = Path(profiles_dir)
        else:
            self.profiles_dir = profiles_dir
        # Ensure directory exists
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def _profile_path(self, name: str) -> Path:
        """Get path for a profile file."""
        safe_name = "".join(c for c in name if c.isalnum() or c in "._-").lower()
        return self.profiles_dir / f"{safe_name}.json"

    def save(self, profile: VoiceProfile) -> Path:
        """Save a voice profile to disk."""
        try:
            path = self._profile_path(profile.name)
            with open(path, "w") as f:
                json.dump(asdict(profile), f, indent=2)
            logger.info(f"Saved profile: {profile.name} -> {path}")
            return path
        except Exception as e:
            logger.error(f"Failed to save profile '{profile.name}': {e}")
            raise

    def load(self, name: str) -> VoiceProfile | None:
        """Load a voice profile by name."""
        path = self._profile_path(name)
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        profile = VoiceProfile(**data)
        self._check_compatibility(profile)
        return profile

    def _check_compatibility(self, profile: VoiceProfile) -> None:
        """Warn if profile was created with incompatible model."""
        if "pyannote" in profile.model_version.lower():
            logger.warning(
                f"Profile '{profile.name}' was created with pyannote (embedding dim: "
                f"{len(profile.embedding)}). Re-enrollment with SpeechBrain is recommended."
            )

    def load_all(self) -> list[VoiceProfile]:
        """Load all available profiles."""
        profiles = []
        for path in self.profiles_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                profile = VoiceProfile(**data)
                self._check_compatibility(profile)
                profiles.append(profile)
            except Exception as e:
                logger.warning(f"Failed to load profile {path}: {e}")
        return profiles

    def list_names(self) -> list[str]:
        """List all available profile names."""
        return [p.name for p in self.load_all()]

    def delete(self, name: str) -> bool:
        """Delete a voice profile."""
        path = self._profile_path(name)
        if path.exists():
            path.unlink()
            logger.info(f"Deleted profile: {name}")
            return True
        return False

    def exists(self, name: str) -> bool:
        """Check if a profile exists."""
        return self._profile_path(name).exists()
