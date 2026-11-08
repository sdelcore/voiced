"""Behavioral tests for desktop companion features without GTK or devices."""

import os
import threading
from types import SimpleNamespace

from voiced.config import Config
from voiced.desktop import DesktopWindow
from voiced.desktop_history import TranscriptHistory
from voiced.remote_daemon import ClientState, RemoteDaemon


class _WindowStub:
    """Provide the UI state used by DesktopWindow handlers, without GTK."""

    _poll = DesktopWindow._poll
    _render_status = DesktopWindow._render_status
    _render_history = DesktopWindow._render_history
    _clear_history = DesktopWindow._clear_history
    _apply_context_replacements = DesktopWindow._apply_context_replacements
    _meeting_results = DesktopWindow._meeting_results
    _annotation_stop_clicked = DesktopWindow._annotation_stop_clicked
    _annotation_cancel_clicked = DesktopWindow._annotation_cancel_clicked


def test_remote_start_stop_records_and_delivers_transcript(monkeypatch):
    daemon = RemoteDaemon("http://unused", config=Config())
    connected = []
    delivered = []
    partial_received = threading.Event()

    class Stream:
        input_level = 0.25

        def connect(self):
            connected.append("connect")

        def start_stt_stream(self, *, on_partial, language):
            assert language == daemon.config.transcription.language

            def deliver_partial():
                on_partial("partial")
                partial_received.set()

            threading.Thread(target=deliver_partial).start()

        def stop_stt(self):
            return SimpleNamespace(text="final transcript")

        def disconnect(self):
            connected.append("disconnect")

    monkeypatch.setattr("voiced.remote_daemon.SyncWebRTCClient", lambda config: Stream())
    monkeypatch.setattr("voiced.remote_daemon.inject_to_clipboard", delivered.append)

    started = daemon._handle_start_recording()
    assert started == {"status": "ok", "state": "recording"}
    assert daemon._handle_start_recording()["status"] == "rejected"
    assert partial_received.wait(2)
    assert daemon._handle_status()["partial_text"] == "partial"

    stopped = daemon._handle_stop_recording()
    assert stopped == {"status": "ok", "state": "transcribing"}
    assert daemon._handle_status()["transcribing"]
    daemon._worker.join(2)

    history = daemon._handle_history()["history"]
    assert [entry["text"] for entry in history] == ["final transcript"]
    assert history[0]["id"] == 1
    assert daemon._handle_status()["state"] == "idle"
    assert delivered == ["final transcript"]
    assert connected == ["connect", "disconnect"]


def test_cancel_during_remote_transcription_suppresses_late_result(monkeypatch):
    daemon = RemoteDaemon("http://unused", config=Config())
    entered = threading.Event()
    release = threading.Event()
    delivered = []

    class Recorder:
        is_recording = True

        def stop(self):
            self.is_recording = False

    class Stream:
        def stop_stt(self):
            entered.set()
            assert release.wait(2)
            return SimpleNamespace(text="must not be delivered")

        def disconnect(self):
            pass

    daemon._recorder = Recorder()
    daemon._stream_client = Stream()
    daemon._state = ClientState.RECORDING
    monkeypatch.setattr("voiced.remote_daemon.inject_to_clipboard", delivered.append)
    assert daemon._handle_stop_recording()["status"] == "ok"
    assert entered.wait(2)
    assert daemon._handle_cancel()["status"] == "ok"
    release.set()
    daemon._worker.join(2)

    status = daemon._handle_status()
    assert status["state"] == ClientState.IDLE.value
    assert status["history"] == []
    assert delivered == []


def test_context_replacements_are_applied_to_transcript_and_replaced_live():
    config = Config()
    config.transcription.replacements = {"cloud code": "Claude Code"}
    config.transcription.context_replacements = {
        "work": {"issue tracker": "Linear"},
        "home": {"issue tracker": "Todoist"},
    }
    daemon = RemoteDaemon("http://unused", config=config)
    assert daemon._handle_set_context_mode({"mode": "work"})["status"] == "ok"
    assert daemon._fix_context("Cloud Code and issue tracker", "work") == "Claude Code and Linear"
    assert daemon._handle_set_context_mode({"mode": "home"})["status"] == "ok"
    assert daemon._fix_context("issue tracker", "home") == "Todoist"
    assert daemon._handle_set_context_mode({"mode": "missing"})["status"] == "error"


def test_desktop_context_replacements_follow_selected_mode(monkeypatch):
    config = Config()
    config.transcription.context_replacements = {
        "work": {"issue tracker": "Linear"},
        "home": {"issue tracker": "Todoist"},
    }
    monkeypatch.setattr("voiced.desktop.load_config", lambda: config)
    window = _WindowStub()
    assert window._apply_context_replacements("Open the ISSUE TRACKER", "work") == "Open the Linear"
    assert (
        window._apply_context_replacements("Open the issue tracker", "home") == "Open the Todoist"
    )


def test_history_is_opt_in_bounded_and_clear_removes_persisted_data(tmp_path):
    from voiced.desktop_history import TranscriptHistory

    path = tmp_path / "history.json"
    transient = TranscriptHistory(path, limit=2)
    transient.add_local("private one")
    transient.add_local("private two")
    assert not path.exists()
    assert len(transient.entries()) == 2

    persisted = TranscriptHistory(path, persist=True, limit=2)
    for text in ("first", "second", "third"):
        persisted.add_local(text)
    entries = persisted.entries()
    assert [entry["text"] for entry in entries] == ["third", "second"]
    assert path.exists()
    assert os.stat(path).st_mode & 0o777 == 0o600
    restored = TranscriptHistory(path, persist=True, limit=2)
    assert [entry["text"] for entry in restored.entries()] == ["third", "second"]
    persisted.clear()
    assert not path.exists()
    assert TranscriptHistory(path, persist=True, limit=2).entries() == []


def test_desktop_clear_hides_daemon_history_until_daemon_entries_change():
    window = _WindowStub()
    window.history = TranscriptHistory()
    window.history_store = SimpleNamespace(rows=[], clear=lambda: window.history_store.rows.clear())
    window.history_store.append = lambda row: window.history_store.rows.append(row)
    window._last_daemon_history = [{"id": 7, "timestamp": "same", "text": "secret"}]
    window._hidden_history_tokens = set()

    window._clear_history()
    window._render_history([])
    assert window.history_store.rows == []
    window._hidden_history_tokens.intersection_update({("7", "same")})
    window._render_history(window._last_daemon_history)
    assert window.history_store.rows == []
    window._hidden_history_tokens.intersection_update({("8", "new")})
    assert window._hidden_history_tokens == set()
    window._render_history([{"id": 8, "timestamp": "new", "text": "new transcript"}])
    assert window.history_store.rows[0][1] == "new transcript"


def test_overlay_shows_recording_and_errors_then_hides_when_idle():
    class Label:
        def __init__(self):
            self.text = None

        def set_text(self, text):
            self.text = text

    class Value:
        def set_value(self, value):
            self.value = value

    class Overlay:
        def __init__(self):
            self.shown = False

        def show_all(self):
            self.shown = True

        def hide(self):
            self.shown = False

    window = _WindowStub()
    window.state_label = Label()
    window._context_modes = []
    window._last_status = {}
    window._busy = False
    window._recording_started = None
    window.record_button = SimpleNamespace(
        set_sensitive=lambda value: None, set_label=lambda value: None
    )
    window.mode_combo = SimpleNamespace(
        get_active_id=lambda: "default",
        set_active_id=lambda value: None,
        set_tooltip_text=lambda value: None,
    )
    window.level = Value()
    window.overlay_level = Value()
    window.transcript_buffer = SimpleNamespace(
        set_text=lambda text: setattr(window, "transcript", text)
    )
    window.timer_label = Label()
    window.overlay_timer = Label()
    window.overlay_state = Label()
    window.overlay_error = Label()
    window.overlay = Overlay()

    window._render_status({"state": "recording", "input_level": 0.4, "partial_text": "hello"})
    assert window.overlay.shown
    assert window.overlay_state.text == "Recording"
    assert window.level.value == 0.4
    assert window.transcript == "hello"

    window._render_status({"state": "idle", "error": "microphone failed"})
    assert window.overlay.shown
    assert window.overlay_error.text == "microphone failed"
    window._render_status({"state": "idle", "error": None})
    assert not window.overlay.shown


def test_meeting_results_preserve_speaker_text_and_report_partial_errors(monkeypatch):
    config = Config()
    config.transcription.context_replacements = {"work": {"issue tracker": "Linear"}}
    monkeypatch.setattr("voiced.desktop.load_config", lambda: config)
    window = _WindowStub()
    window._meeting_mode = "work"
    window.history = TranscriptHistory()
    result = SimpleNamespace(
        microphone=SimpleNamespace(
            segments=[SimpleNamespace(speaker="Speaker 1", text="Open issue tracker")]
        ),
        system_audio=None,
        errors={"system_audio": "device disconnected"},
    )

    message = window._meeting_results(result)

    assert message == "Meeting transcription errors: System audio: device disconnected"
    assert [entry["text"] for entry in window.history.entries()] == ["Speaker 1: Open Linear"]


def test_annotation_stop_and_cancel_set_distinct_events_and_feedback():
    window = _WindowStub()
    window._annotation_stop = threading.Event()
    window._annotation_cancel = threading.Event()
    window.error_label = SimpleNamespace(set_text=lambda text: setattr(window, "feedback", text))

    window._annotation_stop_clicked()
    assert window._annotation_stop.is_set()
    assert not window._annotation_cancel.is_set()
    assert window.feedback == "Finishing screen capture…"

    window._annotation_stop.clear()
    window._annotation_cancel_clicked()
    assert window._annotation_cancel.is_set()
    assert not window._annotation_stop.is_set()
    assert window.feedback == "Cancelling screen capture…"
