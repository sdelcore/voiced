"""Tests for the Transcribe adapters in src/voiced/transcribe.py."""

from unittest.mock import MagicMock

import numpy as np

from voiced.capabilities import TranscribeOutput
from voiced.transcribe import RemoteTranscribe


class TestRemoteTranscribe:
    def _client(self) -> MagicMock:
        return MagicMock()

    def test_returns_transcribe_output(self):
        client = self._client()
        client.transcribe.return_value = {
            "text": "hello world",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello",
                    "speaker": "Unknown",
                    "speaker_confidence": 0.0,
                },
            ],
            "duration": 2.5,
        }
        adapter = RemoteTranscribe(client)
        out = adapter.transcribe(np.zeros(16000, dtype=np.float32), sample_rate=16000)
        assert isinstance(out, TranscribeOutput)
        assert out.text == "hello world"
        assert out.duration == 2.5
        assert len(out.segments) == 1
        assert out.segments[0].text == "hello"

    def test_passes_identify_speakers(self):
        client = self._client()
        client.transcribe.return_value = {"text": "", "segments": [], "duration": 0.0}
        adapter = RemoteTranscribe(client)
        adapter.transcribe(np.zeros(8000, dtype=np.float32), 8000, identify_speakers=True)

        client.transcribe.assert_called_once()
        kwargs = client.transcribe.call_args.kwargs
        assert kwargs["sample_rate"] == 8000
        assert kwargs["identify_speakers"] is True

    def test_passes_num_speakers(self):
        client = self._client()
        client.transcribe.return_value = {"text": "", "segments": [], "duration": 0.0}
        adapter = RemoteTranscribe(client)
        adapter.transcribe(
            np.zeros(8000, dtype=np.float32),
            8000,
            identify_speakers=True,
            num_speakers=3,
        )
        kwargs = client.transcribe.call_args.kwargs
        assert kwargs["num_speakers"] == 3

    def test_speaker_fields_preserved_when_present(self):
        client = self._client()
        client.transcribe.return_value = {
            "text": "alice speaks",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.5,
                    "text": "alice speaks",
                    "speaker": "alice",
                    "speaker_confidence": 0.92,
                },
            ],
            "duration": 1.5,
        }
        adapter = RemoteTranscribe(client)
        out = adapter.transcribe(np.zeros(24000, dtype=np.float32), 16000, identify_speakers=True)
        assert out.segments[0].speaker == "alice"
        assert out.segments[0].speaker_confidence == 0.92

    def test_partial_segment_fields_use_defaults(self):
        # The server may return segments without speaker info when ID is off
        client = self._client()
        client.transcribe.return_value = {
            "text": "x",
            "segments": [{"start": 0.0, "end": 0.5, "text": "x"}],
            "duration": 0.5,
        }
        adapter = RemoteTranscribe(client)
        out = adapter.transcribe(np.zeros(8000, dtype=np.float32), 16000)
        assert out.segments[0].speaker == "Unknown"
        assert out.segments[0].speaker_confidence == 0.0

    def test_empty_response_yields_empty_output(self):
        client = self._client()
        client.transcribe.return_value = {"text": "", "segments": [], "duration": 0.0}
        adapter = RemoteTranscribe(client)
        out = adapter.transcribe(np.zeros(8000, dtype=np.float32), 16000)
        assert out.text == ""
        assert out.segments == []
        assert out.duration == 0.0

    def test_missing_response_keys_use_safe_defaults(self):
        # Defensive: if the server returns a malformed dict, the adapter
        # shouldn't crash — it produces an empty output.
        client = self._client()
        client.transcribe.return_value = {}
        adapter = RemoteTranscribe(client)
        out = adapter.transcribe(np.zeros(8000, dtype=np.float32), 16000)
        assert out.text == ""
        assert out.segments == []
        assert out.duration == 0.0
