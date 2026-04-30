"""Transcribe — single interface over local and remote-HTTP transcription.

Local: ``Voiced`` itself satisfies the protocol structurally.
Remote: ``RemoteTranscribe`` wraps an HTTP ``TranscriptionClient``.

WebRTC has its own command (``voiced webrtc-client``) for real-time
streaming. For one-shot file transcription, HTTP is the simpler
transport — the WebRTC handshake+teardown adds no value.
"""

import logging
from typing import Protocol

import numpy as np

from voiced.capabilities import TranscribedSegment, TranscribeOutput

logger = logging.getLogger(__name__)


class Transcribe(Protocol):
    """Interface for one-shot audio → text transcription with optional speaker IDs."""

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        identify_speakers: bool = False,
    ) -> TranscribeOutput: ...


class RemoteTranscribe:
    """Transcribe adapter that posts audio to an HTTP server."""

    def __init__(self, client):
        # ``client`` is a TranscriptionClient; typed as Any to avoid the import here.
        self._client = client

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        identify_speakers: bool = False,
    ) -> TranscribeOutput:
        result = self._client.transcribe(
            audio,
            sample_rate=sample_rate,
            identify_speakers=identify_speakers,
        )
        segments = [
            TranscribedSegment(
                start=float(s.get("start", 0.0)),
                end=float(s.get("end", 0.0)),
                text=str(s.get("text", "")),
                speaker=str(s.get("speaker", "Unknown")),
                speaker_confidence=float(s.get("speaker_confidence", 0.0)),
            )
            for s in result.get("segments", [])
        ]
        return TranscribeOutput(
            text=result.get("text", ""),
            segments=segments,
            duration=float(result.get("duration", 0.0)),
        )
