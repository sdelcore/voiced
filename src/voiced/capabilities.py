"""Voiced — composition root that bundles the voice-AI capabilities.

Owns the long-lived models (transcriber, synthesizer) and stateful
sub-modules (profile store, speaker identifier, voice manager). Used as
a single dependency by the daemon and the HTTP server, so they don't
each reassemble the model wiring.

Per-request overrides (e.g. a different profiles directory passed in a
query string) are NOT part of the Voiced facade; callers that need them
construct an ad-hoc adapter for that request.
"""

import logging
from dataclasses import dataclass

import numpy as np

from voiced.config import Config, load_config
from voiced.profile_store import LocalProfileStore, ProfileStore
from voiced.transcriber import Transcriber

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscribedSegment:
    """One segment of transcribed text with optional speaker attribution."""

    start: float
    end: float
    text: str
    speaker: str = "Unknown"
    speaker_confidence: float = 0.0


@dataclass(frozen=True)
class TranscribeOutput:
    """Result of Voiced.transcribe — full text plus per-segment detail."""

    text: str
    segments: list[TranscribedSegment]
    duration: float


class Voiced:
    """Composition root for the voice-AI capabilities.

    Exposes long-lived sub-modules as attributes (``transcriber``,
    ``synthesizer``, ``profile_store``, ``speaker_identifier``,
    ``voice_manager``) and provides a small set of cross-cutting
    orchestration methods (``transcribe`` with optional speaker ID).
    """

    def __init__(
        self,
        config: Config,
        transcriber: Transcriber,
        profile_store: ProfileStore,
    ):
        self.config = config
        self.transcriber = transcriber
        self.profile_store = profile_store

        # Lazily constructed — touched only when first accessed
        self._synthesizer = None
        self._speaker_identifier = None  # used by webrtc_server for real-time
        self._speaker_diarizer = None  # used here for batch transcribe
        self._voice_manager = None

    @classmethod
    def from_config(cls, config: Config | None = None) -> "Voiced":
        """Build a Voiced from a Config, constructing default sub-modules."""
        cfg = config or load_config()
        return cls(
            config=cfg,
            transcriber=Transcriber(
                cfg.transcription,
                unload_timeout_minutes=cfg.unload_timeout_minutes,
            ),
            profile_store=LocalProfileStore(diarization_config=cfg.diarization),
        )

    # ----- lazy capabilities -----

    @property
    def synthesizer(self):
        """The TTS synthesizer. Lazily constructed; returns ``None`` if TTS is unavailable."""
        if self._synthesizer is not None:
            return self._synthesizer

        from voiced.synthesizer import Synthesizer, TTSConfig, check_kokoro_installed

        if not self.config.tts.enabled:
            return None
        if not check_kokoro_installed():
            logger.warning("Kokoro is not installed; TTS unavailable")
            return None

        tts_config = TTSConfig(
            device=self.config.tts.device,
            default_voice=self.config.tts.default_voice,
            speed=self.config.tts.speed,
            unload_timeout_seconds=self.config.unload_timeout_minutes * 60,
        )
        self._synthesizer = Synthesizer(tts_config)
        return self._synthesizer

    @property
    def speaker_identifier(self):
        """The per-segment speaker identifier. Used by WebRTC for real-time matching."""
        if self._speaker_identifier is None:
            from voiced.diarizer import SpeakerIdentifier

            self._speaker_identifier = SpeakerIdentifier(config=self.config.diarization)
        return self._speaker_identifier

    @property
    def speaker_diarizer(self):
        """The clustering diarizer. Used by batch transcribe (CLI + HTTP /transcribe)."""
        if self._speaker_diarizer is None:
            from voiced.diarizer import SpeakerDiarizer

            self._speaker_diarizer = SpeakerDiarizer(config=self.config.diarization)
        return self._speaker_diarizer

    @property
    def voice_manager(self):
        """The TTS voice-preset manager. Lazily constructed."""
        if self._voice_manager is None:
            from voiced.voice_manager import VoiceManager

            self._voice_manager = VoiceManager()
        return self._voice_manager

    # ----- orchestration -----

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        identify_speakers: bool = False,
        num_speakers: int | None = None,
        profile_store: ProfileStore | None = None,
    ) -> TranscribeOutput:
        """Transcribe audio with optional speaker diarization.

        Args:
            audio: float32 mono audio.
            sample_rate: sample rate of ``audio``.
            identify_speakers: if True, run clustering diarization on the
                full audio, match clusters to profiles, and align with
                transcription segments. Unmatched clusters surface as
                ``SPEAKER_00``, ``SPEAKER_01`` etc.
            num_speakers: hint for the diarizer; auto-detected when None.
            profile_store: override the default profile store for this call —
                used by the HTTP handler when a request specifies a custom
                profiles path.
        """
        from voiced.diarizer import align_transcription_with_diarization

        segments_raw = self.transcriber.transcribe_audio_with_segments(audio, sample_rate)
        full_text = " ".join(t for _, _, t in segments_raw).strip()
        duration = len(audio) / sample_rate if sample_rate else 0.0

        if not identify_speakers or not segments_raw:
            return TranscribeOutput(
                text=full_text,
                segments=_segments_unlabeled(segments_raw),
                duration=duration,
            )

        store = profile_store or self.profile_store
        try:
            profiles = store.list()
        except Exception:
            logger.exception("Failed to load profiles for speaker ID")
            profiles = []

        try:
            diar_segments = self.speaker_diarizer.diarize_and_match_profiles_from_array(
                audio, sample_rate, profiles=profiles, num_speakers=num_speakers
            )
        except Exception:
            logger.exception("Diarization failed; returning unlabeled segments")
            return TranscribeOutput(
                text=full_text,
                segments=_segments_unlabeled(segments_raw),
                duration=duration,
            )

        aligned = align_transcription_with_diarization(segments_raw, diar_segments)
        segments = [
            TranscribedSegment(
                start=round(seg.start, 2),
                end=round(seg.end, 2),
                text=seg.text,
                speaker=seg.speaker,
                speaker_confidence=round(seg.confidence, 2),
            )
            for seg in aligned
        ]
        return TranscribeOutput(text=full_text, segments=segments, duration=duration)

    # ----- lifecycle -----

    def shutdown(self) -> None:
        """Release model resources held by sub-modules."""
        if self._synthesizer is not None:
            try:
                self._synthesizer.shutdown()
            except Exception:
                logger.exception("Synthesizer shutdown failed")
        if self.transcriber is not None:
            try:
                self.transcriber.unload()
            except Exception:
                logger.exception("Transcriber unload failed")


def _segments_unlabeled(
    segments_raw: list[tuple[float, float, str]],
) -> list[TranscribedSegment]:
    return [
        TranscribedSegment(
            start=round(start, 2),
            end=round(end, 2),
            text=text,
        )
        for start, end, text in segments_raw
    ]
