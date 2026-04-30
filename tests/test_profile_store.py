"""Tests for ProfileStore — local-fs and remote-HTTP adapters."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from voiced.profile_store import LocalProfileStore, RemoteProfileStore, _dict_to_profile
from voiced.profiles import ProfileManager, VoiceProfile


@pytest.fixture
def manager(tmp_path: Path) -> ProfileManager:
    return ProfileManager(tmp_path)


@pytest.fixture
def sample_profile() -> VoiceProfile:
    return VoiceProfile(
        name="alice",
        embedding=[0.1, 0.2, 0.3],
        created_at="2026-04-29T00:00:00",
        audio_duration=2.5,
        model_version="speechbrain/spkrec-ecapa-voxceleb",
    )


class TestLocalProfileStoreCRUD:
    def test_list_empty(self, manager: ProfileManager):
        store = LocalProfileStore(manager=manager)
        assert store.list() == []

    def test_list_returns_saved_profiles(
        self, manager: ProfileManager, sample_profile: VoiceProfile
    ):
        manager.save(sample_profile)
        store = LocalProfileStore(manager=manager)
        result = store.list()
        assert len(result) == 1
        assert result[0].name == "alice"

    def test_get_returns_profile(self, manager: ProfileManager, sample_profile: VoiceProfile):
        manager.save(sample_profile)
        store = LocalProfileStore(manager=manager)
        loaded = store.get("alice")
        assert loaded is not None
        assert loaded.name == "alice"
        assert loaded.embedding == [0.1, 0.2, 0.3]

    def test_get_missing_returns_none(self, manager: ProfileManager):
        store = LocalProfileStore(manager=manager)
        assert store.get("nobody") is None

    def test_exists(self, manager: ProfileManager, sample_profile: VoiceProfile):
        store = LocalProfileStore(manager=manager)
        assert not store.exists("alice")
        manager.save(sample_profile)
        assert store.exists("alice")

    def test_delete_existing(self, manager: ProfileManager, sample_profile: VoiceProfile):
        manager.save(sample_profile)
        store = LocalProfileStore(manager=manager)
        assert store.delete("alice") is True
        assert not store.exists("alice")

    def test_delete_missing(self, manager: ProfileManager):
        store = LocalProfileStore(manager=manager)
        assert store.delete("nobody") is False


class TestLocalProfileStoreRegister:
    def test_register_from_audio_uses_embedder(self, manager: ProfileManager):
        store = LocalProfileStore(manager=manager)

        # Inject a fake embedder so we don't load SpeechBrain
        fake_embedder = MagicMock()
        fake_embedder.extract_embedding_from_array.return_value = np.array(
            [0.5, 0.6, 0.7], dtype=np.float32
        )
        fake_embedder.model_source = "fake/model"
        store._embedder = fake_embedder

        audio = np.zeros(16000, dtype=np.float32)  # 1 second at 16kHz
        profile = store.register_from_audio("bob", audio, sample_rate=16000)

        assert profile.name == "bob"
        assert profile.audio_duration == 1.0
        assert profile.model_version == "fake/model"
        assert profile.embedding == pytest.approx([0.5, 0.6, 0.7])
        assert manager.exists("bob")

    def test_register_persists_through_manager(self, manager: ProfileManager):
        store = LocalProfileStore(manager=manager)
        store._embedder = MagicMock(
            extract_embedding_from_array=MagicMock(
                return_value=np.array([1.0, 2.0], dtype=np.float32)
            ),
            model_source="x",
        )
        store.register_from_audio("carol", np.zeros(8000, dtype=np.float32), 8000)

        # Reload via a fresh store/manager — the on-disk file should be there
        fresh = LocalProfileStore(manager=ProfileManager(manager.profiles_dir))
        loaded = fresh.get("carol")
        assert loaded is not None
        assert loaded.embedding == pytest.approx([1.0, 2.0])


class TestRemoteProfileStore:
    def _client(self) -> MagicMock:
        return MagicMock()

    def test_list_maps_dicts_to_profiles(self):
        client = self._client()
        client.list_profiles.return_value = [
            {
                "name": "alice",
                "embedding": [0.1, 0.2],
                "created_at": "2026-04-29",
                "audio_duration": 2.5,
                "model_version": "v1",
            }
        ]
        store = RemoteProfileStore(client)
        result = store.list()
        assert len(result) == 1
        assert isinstance(result[0], VoiceProfile)
        assert result[0].name == "alice"

    def test_get_returns_none_for_missing(self):
        client = self._client()
        client.get_profile.return_value = None
        store = RemoteProfileStore(client)
        assert store.get("nobody") is None

    def test_get_maps_dict_to_profile(self):
        client = self._client()
        client.get_profile.return_value = {
            "name": "alice",
            "embedding": [0.1, 0.2],
            "created_at": "2026",
            "audio_duration": 1.5,
            "model_version": "v1",
        }
        store = RemoteProfileStore(client)
        profile = store.get("alice")
        assert profile is not None
        assert profile.audio_duration == 1.5

    def test_exists_via_get(self):
        client = self._client()
        client.get_profile.return_value = None
        store = RemoteProfileStore(client)
        assert store.exists("nobody") is False

        client.get_profile.return_value = {"name": "alice"}
        assert store.exists("alice") is True

    def test_delete_proxies_to_client(self):
        client = self._client()
        client.delete_profile.return_value = True
        store = RemoteProfileStore(client)
        assert store.delete("alice") is True
        client.delete_profile.assert_called_once_with("alice")

    def test_register_from_audio_proxies_to_client(self):
        client = self._client()
        client.create_profile.return_value = {
            "name": "bob",
            "embedding": [],
            "created_at": "2026",
            "audio_duration": 1.5,
            "model_version": "remote-v1",
        }
        store = RemoteProfileStore(client)
        audio = np.zeros(8000, dtype=np.float32)

        profile = store.register_from_audio("bob", audio, 8000)

        assert profile.name == "bob"
        assert profile.audio_duration == 1.5
        # Client received the original audio + sample_rate
        client.create_profile.assert_called_once()
        args = client.create_profile.call_args
        assert args[0][0] == "bob"
        np.testing.assert_array_equal(args[0][1], audio)
        assert args[0][2] == 8000


class TestDictToProfile:
    def test_full_dict(self):
        d = {
            "name": "x",
            "embedding": [1.0, 2.0],
            "created_at": "2026",
            "audio_duration": 3.0,
            "model_version": "v1",
        }
        p = _dict_to_profile(d)
        assert p.name == "x"
        assert p.audio_duration == 3.0

    def test_partial_dict_uses_defaults(self):
        # Server's list response may omit embedding
        d = {"name": "x", "created_at": "2026", "audio_duration": 1.0}
        p = _dict_to_profile(d)
        assert p.embedding == []
        assert p.model_version == ""
