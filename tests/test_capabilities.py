"""Tests for Voiced composition root."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from voiced.capabilities import TranscribedSegment, TranscribeOutput, Voiced
from voiced.config import Config, DiarizationConfig, TranscriptionConfig, TTSConfig
from voiced.profiles import VoiceProfile


@pytest.fixture
def cfg() -> Config:
    c = Config()
    c.transcription = TranscriptionConfig()
    c.tts = TTSConfig()
    c.tts.enabled = False  # avoid lazy-loading vibevoice in tests
    c.diarization = DiarizationConfig()
    return c


def _voiced_with_mocks(cfg: Config) -> Voiced:
    """Voiced with everything mocked."""
    transcriber = MagicMock()
    transcriber.transcribe_audio_with_segments.return_value = [
        (0.0, 1.0, "hello"),
        (1.0, 2.0, "world"),
    ]
    profile_store = MagicMock()
    profile_store.list.return_value = []
    return Voiced(config=cfg, transcriber=transcriber, profile_store=profile_store)


class TestComposition:
    def test_holds_required_dependencies(self, cfg):
        v = _voiced_with_mocks(cfg)
        assert v.transcriber is not None
        assert v.profile_store is not None
        assert v.config is cfg

    def test_synthesizer_is_none_when_tts_disabled(self, cfg):
        v = _voiced_with_mocks(cfg)
        assert cfg.tts.enabled is False
        assert v.synthesizer is None

    def test_lazy_caches_dont_construct_until_accessed(self, cfg):
        v = _voiced_with_mocks(cfg)
        # speaker_identifier, speaker_diarizer, and voice_manager are lazy
        assert v._speaker_identifier is None
        assert v._speaker_diarizer is None
        assert v._voice_manager is None


class TestTranscribeWithoutSpeakers:
    def test_returns_segments_unlabeled(self, cfg):
        v = _voiced_with_mocks(cfg)
        audio = np.zeros(16000, dtype=np.float32)
        out = v.transcribe(audio, 16000, identify_speakers=False)
        assert isinstance(out, TranscribeOutput)
        assert out.text == "hello world"
        assert out.duration == 1.0
        assert len(out.segments) == 2
        assert all(s.speaker == "Unknown" for s in out.segments)

    def test_empty_segments_yields_empty_output(self, cfg):
        v = _voiced_with_mocks(cfg)
        v.transcriber.transcribe_audio_with_segments.return_value = []
        audio = np.zeros(16000, dtype=np.float32)
        out = v.transcribe(audio, 16000, identify_speakers=False)
        assert out.text == ""
        assert out.segments == []


class TestTranscribeWithSpeakers:
    def _profile(self, name: str) -> VoiceProfile:
        return VoiceProfile(
            name=name,
            embedding=[0.1, 0.2],
            created_at="2026",
            audio_duration=1.0,
            model_version="v1",
        )

    def _diarized(self, *, start, end, speaker, confidence=0.0):
        # Stand-in for IdentifiedSegment from the diarizer
        m = MagicMock()
        m.start = start
        m.end = end
        m.text = ""
        m.speaker = speaker
        m.confidence = confidence
        return m

    def test_with_profiles_runs_diarizer(self, cfg):
        v = _voiced_with_mocks(cfg)
        v.profile_store.list.return_value = [self._profile("alice")]

        # Stub the diarizer so we don't load SpeechBrain
        diar_segments = [
            self._diarized(start=0.0, end=2.0, speaker="alice", confidence=0.9),
        ]
        v._speaker_diarizer = MagicMock()
        v._speaker_diarizer.diarize_and_match_profiles_from_array.return_value = diar_segments

        out = v.transcribe(
            np.zeros(16000, dtype=np.float32),
            16000,
            identify_speakers=True,
        )
        # Both transcription segments overlap the single diarized segment → both are alice
        assert [s.speaker for s in out.segments] == ["alice", "alice"]
        assert [s.speaker_confidence for s in out.segments] == [0.9, 0.9]

    def test_no_profiles_runs_diarizer_with_cluster_labels(self, cfg):
        v = _voiced_with_mocks(cfg)
        v.profile_store.list.return_value = []
        v._speaker_diarizer = MagicMock()
        v._speaker_diarizer.diarize_and_match_profiles_from_array.return_value = [
            self._diarized(start=0.0, end=1.0, speaker="SPEAKER_00"),
            self._diarized(start=1.0, end=2.0, speaker="SPEAKER_01"),
        ]
        out = v.transcribe(
            np.zeros(16000, dtype=np.float32),
            16000,
            identify_speakers=True,
        )
        # Each transcription segment aligns with its diarized window
        assert out.segments[0].speaker == "SPEAKER_00"
        assert out.segments[1].speaker == "SPEAKER_01"

    def test_num_speakers_passed_to_diarizer(self, cfg):
        v = _voiced_with_mocks(cfg)
        v.profile_store.list.return_value = []
        v._speaker_diarizer = MagicMock()
        v._speaker_diarizer.diarize_and_match_profiles_from_array.return_value = []

        v.transcribe(
            np.zeros(16000, dtype=np.float32),
            16000,
            identify_speakers=True,
            num_speakers=3,
        )
        kwargs = v._speaker_diarizer.diarize_and_match_profiles_from_array.call_args.kwargs
        assert kwargs["num_speakers"] == 3

    def test_diarizer_failure_falls_back_unlabeled(self, cfg):
        v = _voiced_with_mocks(cfg)
        v.profile_store.list.return_value = [self._profile("alice")]
        v._speaker_diarizer = MagicMock()
        v._speaker_diarizer.diarize_and_match_profiles_from_array.side_effect = RuntimeError(
            "boom"
        )

        out = v.transcribe(
            np.zeros(16000, dtype=np.float32),
            16000,
            identify_speakers=True,
        )
        assert all(s.speaker == "Unknown" for s in out.segments)

    def test_per_call_profile_store_override(self, cfg):
        v = _voiced_with_mocks(cfg)
        v.profile_store.list.return_value = []
        override = MagicMock()
        override.list.return_value = [self._profile("bob")]
        v._speaker_diarizer = MagicMock()
        v._speaker_diarizer.diarize_and_match_profiles_from_array.return_value = [
            self._diarized(start=0.0, end=2.0, speaker="bob", confidence=0.7),
        ]

        out = v.transcribe(
            np.zeros(16000, dtype=np.float32),
            16000,
            identify_speakers=True,
            profile_store=override,
        )
        v.profile_store.list.assert_not_called()
        override.list.assert_called_once()
        assert out.segments[0].speaker == "bob"


class TestUnlabeledSegmentRounding:
    def test_segment_timestamps_are_rounded(self, cfg):
        v = _voiced_with_mocks(cfg)
        v.transcriber.transcribe_audio_with_segments.return_value = [
            (0.123456, 0.987654, "first"),
        ]
        out = v.transcribe(np.zeros(16000, dtype=np.float32), 16000)
        seg = out.segments[0]
        assert seg.start == 0.12
        assert seg.end == 0.99


class TestSegmentDataclass:
    def test_default_speaker_is_unknown(self):
        s = TranscribedSegment(start=0.0, end=1.0, text="x")
        assert s.speaker == "Unknown"
        assert s.speaker_confidence == 0.0
