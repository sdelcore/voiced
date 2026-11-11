"""Isolated remote-client tests for OpenAI audio workflows."""

import sys
import threading
from types import ModuleType

import numpy as np
import pytest

from voiced import remote_daemon
from voiced.config import Config


class FakeTrayState:
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    ERROR = "error"


@pytest.fixture
def remote_modules(monkeypatch):
    tray = ModuleType("voiced.tray")
    tray.TrayIcon = object
    tray.TrayState = FakeTrayState
    monkeypatch.setattr(remote_daemon, "TrayState", FakeTrayState)
    monkeypatch.setitem(sys.modules, "voiced.tray", tray)
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    return remote_daemon.ClientState, remote_daemon.RemoteDaemon


@pytest.fixture
def daemon(monkeypatch, remote_modules):
    _, remote_daemon = remote_modules
    config = Config()
    config.client.audio_base_url = "https://audio.example/v1"
    config.client.stt_model = "stt"
    config.client.tts_model = "tts"
    config.assistant.base_url = "https://assistant.example/v1"
    config.assistant.model = "assistant-model"
    instance = remote_daemon("https://server.example", config=config)
    monkeypatch.setattr("voiced.remote_daemon.Recorder", FakeRecorder)
    monkeypatch.setattr("voiced.remote_daemon.inject_to_clipboard", lambda text: True)
    instance._audio_provider = FakeProvider()
    return instance


class FakeRecorder:
    def __init__(self, config=None):
        self.started = False
        self.input_level = 0.0

    def start(self):
        self.started = True
        self.input_level = 0.37

    def stop(self):
        return np.array([0.25], dtype=np.float32)


class FakeProvider:
    def __init__(self, text="dictated text"):
        self.text = text

    def transcribe(self, audio, sample_rate):
        return self.text


def test_gateway_status_reports_dictation_input_level(daemon):
    assert daemon._handle_dictation_start({})["status"] == "ok"

    status = daemon._handle_status({})

    assert status["input_level"] == 0.37
    assert status["state"] == "recording"
    assert status["assistant_state"] == "idle"


def test_gateway_status_reports_assistant_recording_and_input_level(daemon):
    assert daemon._handle_assistant_start({})["status"] == "ok"

    status = daemon._handle_status({})

    assert status["input_level"] == 0.37
    assert status["state"] == "idle"
    assert status["assistant_state"] == "recording"


def test_gateway_status_returns_idle_levels_after_recording_stops(daemon):
    class FakeAssistant:
        state = "thinking"

        def interrupt(self):
            pass

        def submit(self, text):
            pass

    daemon._assistant = FakeAssistant()
    assert daemon._handle_assistant_start({})["status"] == "ok"
    assert daemon._handle_assistant_stop({})["status"] == "ok"

    status = daemon._handle_status({})

    assert status["input_level"] == 0.0
    assert status["assistant_state"] != "recording"


def test_dictation_start_reports_recording_tray_state(daemon, remote_modules):
    client_state, _ = remote_modules

    class FakeTray:
        def __init__(self):
            self.states = []

        def set_state(self, state):
            self.states.append(state)

    tray = FakeTray()
    daemon._tray = tray
    daemon._update_tray_state()

    assert daemon._handle_dictation_start({})["status"] == "ok"
    assert daemon.state == client_state.RECORDING
    assert tray.states == [FakeTrayState.IDLE, FakeTrayState.RECORDING]


def test_gateway_transcription_error_recovers_to_idle(daemon, remote_modules):
    client_state, _ = remote_modules

    class BrokenProvider:
        def transcribe(self, audio, sample_rate):
            raise RuntimeError("gateway transcription failed")

    daemon._audio_provider = BrokenProvider()
    assert daemon._handle_dictation_start({})["status"] == "ok"
    assert daemon._handle_dictation_stop({})["status"] == "ok"
    worker = daemon._worker
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert daemon.state == client_state.IDLE
    assert daemon._last_error == "gateway transcription failed"
    assert not daemon._history


def test_dictation_records_transcribes_and_injects_text(daemon, monkeypatch, remote_modules):
    injected = []
    client_state, _ = remote_modules
    monkeypatch.setattr("voiced.remote_daemon.inject_to_clipboard", injected.append)
    entered = threading.Event()
    release = threading.Event()

    class BlockingProvider:
        def transcribe(self, audio, sample_rate):
            entered.set()
            assert release.wait(timeout=2)
            return "dictated text"

    daemon._audio_provider = BlockingProvider()
    assert daemon._handle_dictation_start({})["status"] == "ok"
    assert daemon._handle_dictation_stop({})["status"] == "ok"
    worker = daemon._worker
    assert entered.wait(timeout=2)
    release.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert injected == ["dictated text"]
    assert daemon._history[0]["text"] == "dictated text"
    assert daemon.state == client_state.IDLE


def test_assistant_transcribes_and_submits_without_clipboard(daemon, monkeypatch, remote_modules):
    submitted = threading.Event()

    class FakeAssistant:
        def __init__(self):
            self.submissions = []
            self.interruptions = 0

        def interrupt(self):
            self.interruptions += 1

        def submit(self, text):
            self.submissions.append(text)
            submitted.set()

    assistant = FakeAssistant()
    daemon._assistant = assistant
    injected = []
    monkeypatch.setattr("voiced.remote_daemon.inject_to_clipboard", injected.append)
    assert daemon._handle_assistant_start({})["status"] == "ok"
    assert daemon._handle_assistant_stop({})["status"] == "ok"
    assert submitted.wait(timeout=2)
    assert assistant.submissions == ["dictated text"]
    assert injected == []


def test_cancel_during_transcription_prevents_clipboard_injection(
    daemon, monkeypatch, remote_modules
):
    client_state, _ = remote_modules
    entered = threading.Event()
    release = threading.Event()
    injected = []

    class BlockingProvider:
        def transcribe(self, audio, sample_rate):
            entered.set()
            assert release.wait(timeout=2)
            return "must not be injected"

    daemon._audio_provider = BlockingProvider()
    monkeypatch.setattr("voiced.remote_daemon.inject_to_clipboard", injected.append)
    assert daemon._handle_dictation_start({})["status"] == "ok"
    assert daemon._handle_dictation_stop({})["status"] == "ok"
    worker = daemon._worker
    assert entered.wait(timeout=2)
    assert daemon._handle_cancel({})["status"] == "ok"
    release.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert injected == []
    assert not daemon._history
    assert daemon.state == client_state.IDLE
