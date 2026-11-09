"""Behavioral tests for assistant IPC handling in the daemon."""

import threading

import pytest

from voiced.config import Config
from voiced.daemon import Daemon


class FakeRecorder:
    instances = []
    start_error = None
    stop_error = None
    audio_data = b"recorded audio"

    def __init__(self, config):
        self.config = config
        self.started = False
        self.stopped = False
        type(self).instances.append(self)

    def start(self):
        if type(self).start_error:
            raise type(self).start_error
        self.started = True

    def stop(self):
        if type(self).stop_error:
            raise type(self).stop_error
        self.stopped = True
        return type(self).audio_data


class FakeAssistant:
    def __init__(self):
        self.interruptions = 0
        self.submissions = []

    @property
    def state(self):
        return "idle"

    def interrupt(self):
        self.interruptions += 1

    def submit(self, text):
        self.submissions.append(text)


class FakeTranscriber:
    def __init__(self):
        self.calls = []
        self.result = "recognized speech"
        self.started = None

    def transcribe_audio(self, audio, sample_rate):
        self.calls.append((audio, sample_rate))
        if self.started:
            self.started.set()
        return self.result


@pytest.fixture
def daemon(monkeypatch):
    from voiced import recorder

    FakeRecorder.instances = []
    FakeRecorder.start_error = None
    FakeRecorder.stop_error = None
    FakeRecorder.audio_data = b"recorded audio"
    monkeypatch.setattr(recorder, "Recorder", FakeRecorder)
    instance = Daemon(config=Config())
    instance._assistant = FakeAssistant()
    instance._voiced = type("FakeVoiced", (), {"transcriber": FakeTranscriber()})()
    from voiced.recording_session import SessionState

    class IdleSession:
        state = SessionState.IDLE

    instance._session = IdleSession()
    yield instance


def test_assistant_start_requires_enabled_assistant(daemon):
    daemon._assistant = None
    assert daemon._handle_assistant_start({}) == {
        "status": "error",
        "message": "assistant is disabled",
    }


def test_assistant_start_interrupts_existing_turn_and_prevents_overlap(daemon):
    assert daemon._handle_assistant_start({})["status"] == "ok"
    recorder = daemon._assistant_recorder
    assert recorder.started
    assert daemon._assistant.interruptions == 1
    assert daemon._handle_assistant_start({}) == {
        "status": "error",
        "message": "assistant recording already active",
    }
    assert daemon._assistant_recorder is recorder
    assert daemon._assistant.interruptions == 1


def test_assistant_start_failure_is_reported(daemon):
    FakeRecorder.start_error = RuntimeError("device unavailable")
    response = daemon._handle_assistant_start({})
    assert response == {"status": "error", "message": "device unavailable"}
    assert daemon._assistant_recorder is None
    assert daemon._assistant_error == "device unavailable"


def test_assistant_ipc_recording_is_separate_from_dictation(daemon):
    from voiced.recording_session import SessionState

    class Session:
        def __init__(self):
            self.state = SessionState.IDLE
            self.starts = 0

        def start(self):
            self.starts += 1
            return type(
                "Outcome",
                (),
                {
                    "kind": type("Kind", (), {"value": "started"})(),
                    "state": type("State", (), {"value": "recording"})(),
                    "reason": None,
                },
            )()

    session = Session()
    daemon._session = session
    assert daemon._handle_assistant_start({})["status"] == "ok"
    assert daemon._handle_start_recording({})["message"] == "assistant recording is active"
    assert daemon._handle_stop_recording({})["message"] == "assistant recording is active"
    assert daemon._handle_assistant_stop({})["status"] == "ok"
    assert session.starts == 0
    response = daemon._handle_start_recording({})
    assert response["status"] == "ok"
    assert session.starts == 1


def test_assistant_stop_dispatches_transcription_and_submits_text(daemon, monkeypatch):
    started = threading.Event()
    daemon._voiced.transcriber.started = started
    dispatched = []

    class Thread:
        def __init__(self, *, target, args, daemon, name):
            self.target = target
            self.args = args
            dispatched.append((target, args, daemon, name))

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr("voiced.daemon.threading.Thread", Thread)
    assert daemon._handle_assistant_start({})["status"] == "ok"
    assert daemon._handle_assistant_stop({}) == {
        "status": "ok",
        "message": "Assistant recording stopped",
    }
    assert started.is_set()
    assert dispatched[0][2:] == (True, "AssistantTranscription")
    assert daemon._voiced.transcriber.calls == [
        (b"recorded audio", daemon.config.audio.sample_rate)
    ]
    assert daemon._assistant.submissions == ["recognized speech"]
    assert daemon._assistant_recorder is None


def test_assistant_stop_without_recording_and_empty_audio(daemon):
    assert daemon._handle_assistant_stop({}) == {
        "status": "error",
        "message": "assistant is not recording",
    }
    FakeRecorder.audio_data = b""
    assert daemon._handle_assistant_start({})["status"] == "ok"
    assert daemon._handle_assistant_stop({}) == {
        "status": "error",
        "message": "no audio recorded",
    }
    assert daemon._assistant_error == "no audio recorded"


def test_stale_assistant_transcription_is_discarded_after_new_start(daemon):
    old_entered = threading.Event()
    release_old = threading.Event()
    new_entered = threading.Event()
    release_new = threading.Event()

    def transcribe(audio, sample_rate):
        if audio == FakeRecorder.audio_data:
            old_entered.set()
            assert release_old.wait(timeout=2)
            return "stale response"
        new_entered.set()
        assert release_new.wait(timeout=2)
        return "new response"

    daemon._voiced.transcriber.transcribe_audio = transcribe
    assert daemon._handle_assistant_start({})["status"] == "ok"
    assert daemon._handle_assistant_stop({})["status"] == "ok"
    assert old_entered.wait(timeout=2)
    assert daemon._handle_assistant_start({})["status"] == "ok"

    new_generation = daemon._assistant_generation
    new_completion = threading.Thread(
        target=daemon._transcribe_assistant, args=(b"later", new_generation)
    )
    new_completion.start()
    assert new_entered.wait(timeout=2)
    release_old.set()
    assert daemon._assistant.submissions == []
    release_new.set()
    new_completion.join(timeout=2)
    assert not new_completion.is_alive()
    assert daemon._assistant.submissions == ["new response"]


def test_assistant_transcription_error_is_recorded_only_for_current_generation(daemon):
    def fail(audio, sample_rate):
        raise RuntimeError("transcription failed")

    daemon._voiced.transcriber.transcribe_audio = fail
    daemon._transcribe_assistant(b"audio", daemon._assistant_generation)
    assert daemon._assistant_error == "transcription failed"
    assert daemon._last_error == "transcription failed"
    daemon._assistant_error = None
    daemon._last_error = None
    generation = daemon._assistant_generation
    daemon._assistant_generation += 1
    daemon._transcribe_assistant(b"stale", generation)
    assert daemon._assistant_error is None
    assert daemon._last_error is None
