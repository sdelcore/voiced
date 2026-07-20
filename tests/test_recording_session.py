"""Tests for RecordingSession state machine and dispatch."""

import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from voiced.config import AudioConfig
from voiced.recording_session import (
    RecordingSession,
    SessionState,
    ToggleOutcomeKind,
    TranscriptionResult,
)


def _wait_until(predicate, timeout: float = 2.0, interval: float = 0.01):
    """Spin until ``predicate()`` is true or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class FakeRecorder:
    """Stand-in for voiced.recorder.Recorder. No real audio device."""

    def __init__(self, audio: np.ndarray | None = None):
        self._audio = (
            audio if audio is not None else np.zeros(16000, dtype=np.float32)
        )  # 1s @ 16kHz
        self.is_recording = False
        self._started = False
        self._stopped = False

    def start(self):
        self.is_recording = True
        self._started = True

    def stop(self):
        self.is_recording = False
        self._stopped = True
        return self._audio


@pytest.fixture(autouse=True)
def patch_recorder(monkeypatch):
    """Replace voiced.recording_session.Recorder with FakeRecorder."""
    instances: list[FakeRecorder] = []

    def factory(*args, **kwargs):
        rec = FakeRecorder()
        instances.append(rec)
        return rec

    monkeypatch.setattr("voiced.recording_session.Recorder", factory)
    return instances


@pytest.fixture
def fake_transcriber():
    t = MagicMock()
    t.transcribe_audio.return_value = "hello world"
    return t


@pytest.fixture
def session(fake_transcriber) -> RecordingSession:
    s = RecordingSession(
        transcriber=fake_transcriber,
        audio_config=AudioConfig(),
        sample_rate=16000,
    )
    yield s
    s.shutdown(transcribe_timeout=2.0)


class TestStateMachine:
    def test_starts_idle(self, session: RecordingSession):
        assert session.state == SessionState.IDLE

    def test_first_toggle_starts_recording(self, session: RecordingSession):
        outcome = session.toggle()
        assert outcome.kind == ToggleOutcomeKind.STARTED
        assert outcome.state == SessionState.RECORDING
        assert session.state == SessionState.RECORDING

    def test_second_toggle_starts_transcribing(self, session: RecordingSession):
        session.toggle()  # IDLE → RECORDING
        outcome = session.toggle()  # RECORDING → TRANSCRIBING
        assert outcome.kind == ToggleOutcomeKind.STOPPED
        assert outcome.state == SessionState.TRANSCRIBING

    def test_toggle_during_transcribing_is_rejected(
        self, session: RecordingSession, fake_transcriber
    ):
        # Make transcribe block until we say
        block = threading.Event()
        fake_transcriber.transcribe_audio.side_effect = lambda *a, **k: (block.wait(), "text")[1]

        session.toggle()
        session.toggle()
        # Now in TRANSCRIBING; toggle again
        outcome = session.toggle()
        assert outcome.kind == ToggleOutcomeKind.REJECTED
        assert outcome.reason == "transcription in progress"

        block.set()  # let the worker finish

    def test_returns_to_idle_after_transcription(self, session: RecordingSession, fake_transcriber):
        session.toggle()
        session.toggle()
        assert _wait_until(lambda: session.state == SessionState.IDLE, timeout=2.0)


class TestWarmup:
    def test_record_start_warms_transcriber(self, session: RecordingSession, fake_transcriber):
        session.toggle()
        assert _wait_until(lambda: fake_transcriber.warmup.called, timeout=2.0)

    def test_warmup_failure_does_not_break_session(
        self, session: RecordingSession, fake_transcriber
    ):
        fake_transcriber.warmup.side_effect = RuntimeError("no GPU for you")
        results: list[TranscriptionResult] = []
        session.on_result(results.append)

        session.toggle()
        assert _wait_until(lambda: fake_transcriber.warmup.called, timeout=2.0)
        session.toggle()
        assert _wait_until(lambda: len(results) == 1, timeout=2.0)
        assert results[0].text == "hello world"


class TestCallbacks:
    def test_state_change_callback_fires(self, session: RecordingSession):
        events: list[tuple[SessionState, SessionState]] = []
        session.on_state_change(lambda old, new: events.append((old, new)))

        session.toggle()
        assert _wait_until(lambda: len(events) >= 1)
        assert events[0] == (SessionState.IDLE, SessionState.RECORDING)

    def test_full_cycle_state_changes(self, session: RecordingSession):
        events: list[tuple[SessionState, SessionState]] = []
        session.on_state_change(lambda old, new: events.append((old, new)))

        session.toggle()
        session.toggle()
        assert _wait_until(lambda: len(events) >= 3, timeout=2.0)
        assert events == [
            (SessionState.IDLE, SessionState.RECORDING),
            (SessionState.RECORDING, SessionState.TRANSCRIBING),
            (SessionState.TRANSCRIBING, SessionState.IDLE),
        ]

    def test_result_callback_fires_with_text(self, session: RecordingSession):
        results: list[TranscriptionResult] = []
        session.on_result(lambda r: results.append(r))

        session.toggle()
        session.toggle()
        assert _wait_until(lambda: len(results) == 1, timeout=2.0)
        assert results[0].text == "hello world"
        assert results[0].duration == 1.0  # 16000 samples / 16000 hz

    def test_error_callback_fires_when_transcribe_raises(
        self, session: RecordingSession, fake_transcriber
    ):
        fake_transcriber.transcribe_audio.side_effect = RuntimeError("model OOM")
        errors: list[BaseException] = []
        session.on_error(lambda e: errors.append(e))

        session.toggle()
        session.toggle()
        assert _wait_until(lambda: len(errors) == 1, timeout=2.0)
        assert isinstance(errors[0], RuntimeError)
        assert "model OOM" in str(errors[0])

    def test_callback_exception_does_not_break_session(self, session: RecordingSession):
        """A misbehaving callback must not stop the session from making progress."""
        session.on_state_change(lambda old, new: (_ for _ in ()).throw(RuntimeError("boom")))
        results: list[TranscriptionResult] = []
        session.on_result(lambda r: results.append(r))

        session.toggle()
        session.toggle()
        # Despite the state-change callback raising, the result still arrives
        assert _wait_until(lambda: len(results) == 1, timeout=2.0)


class TestEmptyRecording:
    def test_empty_audio_rejects_and_returns_to_idle(self, monkeypatch):
        # Recorder returns zero-length audio
        empty_recorder = FakeRecorder(audio=np.array([], dtype=np.float32))
        monkeypatch.setattr("voiced.recording_session.Recorder", lambda *a, **k: empty_recorder)

        transcriber = MagicMock()
        s = RecordingSession(
            transcriber=transcriber,
            audio_config=AudioConfig(),
            sample_rate=16000,
        )
        try:
            s.toggle()
            outcome = s.toggle()
            assert outcome.kind == ToggleOutcomeKind.REJECTED
            assert "no audio" in outcome.reason.lower()
            transcriber.transcribe_audio.assert_not_called()
        finally:
            s.shutdown(transcribe_timeout=2.0)


class TestShutdown:
    def test_toggle_after_shutdown_rejected(self, session: RecordingSession):
        session.shutdown(transcribe_timeout=1.0)
        outcome = session.toggle()
        assert outcome.kind == ToggleOutcomeKind.REJECTED
        assert "shut down" in outcome.reason

    def test_shutdown_during_recording_cancels(self, fake_transcriber):
        s = RecordingSession(
            transcriber=fake_transcriber,
            audio_config=AudioConfig(),
            sample_rate=16000,
        )
        s.toggle()
        assert s.state == SessionState.RECORDING
        s.shutdown(transcribe_timeout=1.0)
        # After shutdown the state should be IDLE
        assert s.state == SessionState.IDLE


class TestThreadSafety:
    def test_concurrent_toggles_are_serialized(self, session: RecordingSession):
        """Multiple threads calling toggle() simultaneously must not corrupt state."""
        outcomes = []
        lock = threading.Lock()

        def hammer():
            o = session.toggle()
            with lock:
                outcomes.append(o)

        threads = [threading.Thread(target=hammer) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one started; the others either stopped a recording or were rejected.
        # Crucially, no exception leaked, and the state is consistent.
        assert len(outcomes) == 5
        kinds = [o.kind for o in outcomes]
        # At least one must be STARTED (the first toggle from IDLE)
        assert ToggleOutcomeKind.STARTED in kinds
        # State is now one of the legal states
        legal = {SessionState.IDLE, SessionState.RECORDING, SessionState.TRANSCRIBING}
        assert session.state in legal
