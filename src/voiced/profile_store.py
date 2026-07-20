"""ProfileStore — single interface over local-fs and remote-HTTP voice-profile storage."""

import logging
from datetime import datetime
from typing import Protocol

import numpy as np

from voiced.config import DiarizationConfig
from voiced.profiles import ProfileManager, VoiceProfile

logger = logging.getLogger(__name__)


class ProfileStore(Protocol):
    """Interface for voice-profile storage. Local and remote adapters live behind this."""

    def list(self) -> list[VoiceProfile]: ...

    def get(self, name: str) -> VoiceProfile | None: ...

    def delete(self, name: str) -> bool:
        """Return True if a profile was deleted, False if it didn't exist."""
        ...

    def exists(self, name: str) -> bool: ...

    def register_from_audio(self, name: str, audio: np.ndarray, sample_rate: int) -> VoiceProfile:
        """Compute an embedding from audio and persist a new profile."""
        ...


class LocalProfileStore:
    """ProfileStore backed by on-disk JSON files. Lazily constructs the embedder."""

    def __init__(
        self,
        manager: ProfileManager | None = None,
        diarization_config: DiarizationConfig | None = None,
        device: str | None = None,
        embedder=None,
    ):
        self._manager = manager or ProfileManager()
        self._diar_cfg = diarization_config or DiarizationConfig()
        self._device = device
        # Injected embedder (e.g. worker-backed) or a SpeakerEmbedder
        # lazily constructed in this process when none is provided.
        self._embedder = embedder

    def _get_embedder(self):
        if self._embedder is None:
            from voiced.diarizer import SpeakerEmbedder

            self._embedder = SpeakerEmbedder(
                device=self._device or self._diar_cfg.device,
            )
        return self._embedder

    def list(self) -> list[VoiceProfile]:
        return self._manager.load_all()

    def get(self, name: str) -> VoiceProfile | None:
        return self._manager.load(name)

    def delete(self, name: str) -> bool:
        return self._manager.delete(name)

    def exists(self, name: str) -> bool:
        return self._manager.exists(name)

    def register_from_audio(self, name: str, audio: np.ndarray, sample_rate: int) -> VoiceProfile:
        embedder = self._get_embedder()
        embedding = embedder.extract_embedding_from_array(audio, sample_rate)
        profile = VoiceProfile(
            name=name,
            embedding=embedding.tolist(),
            created_at=datetime.now().isoformat(),
            audio_duration=len(audio) / sample_rate if sample_rate else 0.0,
            model_version=embedder.model_source,
        )
        self._manager.save(profile)
        return profile


class RemoteProfileStore:
    """ProfileStore backed by an HTTP server's /profiles endpoints."""

    def __init__(self, client):
        # client is a TranscriptionClient; typed as Any to avoid a hard import here
        self._client = client

    def list(self) -> list[VoiceProfile]:
        return [_dict_to_profile(d) for d in self._client.list_profiles()]

    def get(self, name: str) -> VoiceProfile | None:
        data = self._client.get_profile(name)
        return _dict_to_profile(data) if data else None

    def delete(self, name: str) -> bool:
        return self._client.delete_profile(name)

    def exists(self, name: str) -> bool:
        return self.get(name) is not None

    def register_from_audio(self, name: str, audio: np.ndarray, sample_rate: int) -> VoiceProfile:
        result = self._client.create_profile(name, audio, sample_rate)
        return _dict_to_profile(result)


def _dict_to_profile(data: dict) -> VoiceProfile:
    """Convert an HTTP API profile dict to a VoiceProfile.

    The server may omit some fields (e.g. ``embedding`` is hidden from list
    responses). Missing fields fall back to safe defaults so the client
    behaves the same shape as the local store.
    """
    return VoiceProfile(
        name=data.get("name", ""),
        embedding=data.get("embedding", []),
        created_at=data.get("created_at", ""),
        audio_duration=float(data.get("audio_duration", 0.0)),
        model_version=data.get("model_version", ""),
    )
