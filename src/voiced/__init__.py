"""voiced - Voice Daemon.

A Python library for speech-to-text (STT) and text-to-speech (TTS) for Linux/Wayland.
Uses faster-whisper for STT and VibeVoice for TTS, with optional speaker identification.

Basic STT usage:
    from voiced import Transcriber, TranscriptionConfig

    transcriber = Transcriber()
    text = transcriber.transcribe_file("audio.wav")

With speaker identification:
    from voiced import Transcriber, SpeakerIdentifier, ProfileManager

    transcriber = Transcriber()
    segments = transcriber.transcribe_file_with_segments("meeting.wav")

    identifier = SpeakerIdentifier()
    profiles = ProfileManager().load_all()
    identified = identifier.identify_segments("meeting.wav", segments, profiles)

    for seg in identified:
        print(f"[{seg.start:.1f}-{seg.end:.1f}] {seg.speaker}: {seg.text}")
"""

__version__ = "0.2.0"

# Core transcription
# Configuration classes
from voiced.config import (
    AudioConfig,
    Config,
    DiarizationConfig,
    TranscriptionConfig,
)

# Speaker identification
from voiced.diarizer import (
    ENROLLMENT_PROMPT,
    DiarizedSegment,
    IdentifiedSegment,
    SpeakerDiarizer,
    SpeakerEmbedder,
    SpeakerIdentifier,
    align_transcription_with_diarization,
)

# Voice profiles
from voiced.profiles import ProfileManager, VoiceProfile

# Audio recording
from voiced.recorder import Recorder
from voiced.transcriber import Transcriber

__all__ = [
    # Version
    "__version__",
    # Transcription
    "Transcriber",
    "TranscriptionConfig",
    # Recording
    "Recorder",
    "AudioConfig",
    # Speaker identification
    "SpeakerEmbedder",
    "SpeakerIdentifier",
    "SpeakerDiarizer",
    "IdentifiedSegment",
    "DiarizedSegment",
    "DiarizationConfig",
    "ENROLLMENT_PROMPT",
    "align_transcription_with_diarization",
    # Profiles
    "VoiceProfile",
    "ProfileManager",
    # Configuration
    "Config",
]
