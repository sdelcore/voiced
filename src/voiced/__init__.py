"""voiced - Voice Daemon.

A Python library for speech-to-text (STT) and text-to-speech (TTS) for Linux/Wayland.
Uses NVIDIA Parakeet-TDT (NeMo) for STT and Kokoro for TTS, with optional speaker identification.

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

__version__ = "0.5.0"

# Public names are resolved lazily (PEP 562) so importing the package does not
# pull in torch, sounddevice, or the diarization stack. The daemon parent
# process depends on this: GPU-touching modules must only be imported inside
# the inference worker process.
_EXPORTS = {
    # Transcription
    "Transcriber": "voiced.transcriber",
    "TranscriptionConfig": "voiced.config",
    # Recording
    "Recorder": "voiced.recorder",
    "AudioConfig": "voiced.config",
    # Speaker identification (segment types/alignment are torch-free)
    "SpeakerEmbedder": "voiced.diarizer",
    "SpeakerIdentifier": "voiced.diarizer",
    "SpeakerDiarizer": "voiced.diarizer",
    "IdentifiedSegment": "voiced.speaker_segments",
    "DiarizedSegment": "voiced.speaker_segments",
    "DiarizationConfig": "voiced.config",
    "ENROLLMENT_PROMPT": "voiced.diarizer",
    "align_transcription_with_diarization": "voiced.speaker_segments",
    # Profiles
    "VoiceProfile": "voiced.profiles",
    "ProfileManager": "voiced.profiles",
    # Configuration
    "Config": "voiced.config",
}

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'voiced' has no attribute '{name}'")
    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
