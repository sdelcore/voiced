"""sttd - Speech-to-Text Daemon.

A Python library for speech-to-text transcription using faster-whisper,
with optional speaker identification using SpeechBrain.

Basic usage:
    from sttd import Transcriber, TranscriptionConfig

    transcriber = Transcriber()
    text = transcriber.transcribe_file("audio.wav")

With speaker identification:
    from sttd import Transcriber, SpeakerIdentifier, ProfileManager

    transcriber = Transcriber()
    segments = transcriber.transcribe_file_with_segments("meeting.wav")

    identifier = SpeakerIdentifier()
    profiles = ProfileManager().load_all()
    identified = identifier.identify_segments("meeting.wav", segments, profiles)

    for seg in identified:
        print(f"[{seg.start:.1f}-{seg.end:.1f}] {seg.speaker}: {seg.text}")
"""

__version__ = "0.1.0"

# Core transcription
# Configuration classes
from sttd.config import (
    AudioConfig,
    Config,
    DiarizationConfig,
    TranscriptionConfig,
)

# Speaker identification
from sttd.diarizer import (
    ENROLLMENT_PROMPT,
    DiarizedSegment,
    IdentifiedSegment,
    SpeakerDiarizer,
    SpeakerEmbedder,
    SpeakerIdentifier,
    align_transcription_with_diarization,
)

# Voice profiles
from sttd.profiles import ProfileManager, VoiceProfile

# Audio recording
from sttd.recorder import Recorder
from sttd.transcriber import Transcriber

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
