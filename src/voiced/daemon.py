"""Daemon process for voiced."""

import logging
import os
import signal
import sys
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime

from voiced import audio
from voiced.capabilities import Voiced
from voiced.config import Config, get_pid_path, load_config
from voiced.injector import inject_to_clipboard
from voiced.recording_session import (
    RecordingSession,
    SessionState,
    ToggleOutcomeKind,
    TranscriptionResult,
)
from voiced.server import Server
from voiced.transcriber import STT_MODEL
from voiced.tray import TrayIcon, TrayState

logger = logging.getLogger(__name__)

# Lazy import to avoid loading HTTP server dependencies unless needed
TranscriptionServer = None


# History — kept in-process so the tray menu can show recent transcriptions.


@dataclass
class HistoryEntry:
    """A single transcription history entry."""

    id: int
    text: str
    timestamp: datetime
    preview: str

    @staticmethod
    def create(entry_id: int, text: str) -> "HistoryEntry":
        """Create a new history entry with auto-generated preview."""
        preview = text[:50] + "..." if len(text) > 50 else text
        preview = preview.replace("\n", " ")
        return HistoryEntry(id=entry_id, text=text, timestamp=datetime.now(), preview=preview)


class TranscriptionHistory:
    """Thread-safe circular buffer for transcription history."""

    MAX_ENTRIES = 10

    def __init__(self):
        self._entries: deque[HistoryEntry] = deque(maxlen=self.MAX_ENTRIES)
        self._lock = threading.Lock()
        self._next_id = 1
        self._revision = 0

    def add(self, text: str) -> HistoryEntry:
        """Add a new transcription to history. Thread-safe."""
        with self._lock:
            entry = HistoryEntry.create(self._next_id, text)
            self._entries.appendleft(entry)
            self._next_id += 1
            self._revision += 1
            return entry

    def get_all(self) -> list[HistoryEntry]:
        """Get all history entries (most recent first). Thread-safe."""
        with self._lock:
            return list(self._entries)

    def get_by_id(self, entry_id: int) -> HistoryEntry | None:
        """Get a specific entry by ID. Thread-safe."""
        with self._lock:
            for entry in self._entries:
                if entry.id == entry_id:
                    return entry
            return None

    @property
    def revision(self) -> int:
        """Current layout revision for DBusMenu."""
        with self._lock:
            return self._revision


def _get_transcription_server():
    """Lazy load TranscriptionServer to avoid import overhead."""
    global TranscriptionServer
    if TranscriptionServer is None:
        from voiced.http_server import TranscriptionServer

    return TranscriptionServer


class Daemon:
    """Coordinator for the voice daemon.

    Wires a RecordingSession (which owns the state machine, recorder, and
    transcription) to: IPC server, tray icon, history, clipboard injector,
    and an optional embedded HTTP server.
    """

    def __init__(
        self,
        config: Config | None = None,
        http_enabled: bool | None = None,
        http_host: str | None = None,
        http_port: int | None = None,
    ):
        """Initialize the daemon.

        Args:
            config: Configuration. Uses defaults if not provided.
            http_enabled: Enable HTTP server. Overrides config if provided.
            http_host: HTTP server host. Overrides config if provided.
            http_port: HTTP server port. Overrides config if provided.
        """
        self.config = config or load_config()
        self._running = False
        self._shutdown_event = threading.Event()

        # HTTP server settings (CLI overrides config)
        if http_enabled is not None:
            self._http_enabled = http_enabled
        else:
            self._http_enabled = self.config.daemon.http_enabled

        # Host/port priority: CLI arg > daemon config > server config
        if http_host is not None:
            self._http_host = http_host
        elif self.config.daemon.http_host is not None:
            self._http_host = self.config.daemon.http_host
        else:
            self._http_host = self.config.server.host

        if http_port is not None:
            self._http_port = http_port
        elif self.config.daemon.http_port is not None:
            self._http_port = self.config.daemon.http_port
        else:
            self._http_port = self.config.server.port

        # Components
        self._server: Server | None = None
        self._voiced: Voiced | None = None
        self._tray: TrayIcon | None = None
        self._http_server = None  # TranscriptionServer instance
        self._session: RecordingSession | None = None

        # History
        self._history = TranscriptionHistory()

        # PID file
        self._pid_path = get_pid_path()

    @property
    def state(self) -> SessionState:
        """Get the current session state."""
        return self._session.state if self._session else SessionState.IDLE

    @property
    def is_running(self) -> bool:
        """Check if daemon is running."""
        return self._running

    def _write_pid(self) -> None:
        self._pid_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._pid_path, "w") as f:
            f.write(str(os.getpid()))
        logger.info(f"PID file written: {self._pid_path}")

    def _remove_pid(self) -> None:
        if self._pid_path.exists():
            try:
                self._pid_path.unlink()
                logger.info("PID file removed")
            except Exception as e:
                logger.warning(f"Failed to remove PID file: {e}")

    # ----- IPC handlers -----

    def _handle_toggle(self, args: dict) -> dict:
        if self._session is None:
            return {"status": "error", "message": "session not initialized"}
        outcome = self._session.toggle()
        return {
            "status": "ok" if outcome.kind != ToggleOutcomeKind.REJECTED else "rejected",
            "outcome": outcome.kind.value,
            "state": outcome.state.value,
            "reason": outcome.reason,
        }

    def _handle_status(self, args: dict) -> dict:
        state = self.state
        return {
            "status": "ok",
            "state": state.value,
            "transcribing": state == SessionState.TRANSCRIBING,
            "history_count": len(self._history.get_all()),
            "model": STT_MODEL,
            "device": self.config.transcription.device,
        }

    def _handle_stop(self, args: dict) -> dict:
        logger.info("Received stop command")
        self._shutdown_event.set()
        return {"status": "ok", "message": "Daemon stopping"}

    # ----- session callbacks -----

    def _on_session_state_change(self, old: SessionState, new: SessionState) -> None:
        self._update_tray_state(new)

        if new == SessionState.RECORDING and self.config.audio.beep_enabled:
            threading.Thread(target=audio.beep_start, daemon=True).start()
        elif old == SessionState.RECORDING and self.config.audio.beep_enabled:
            audio.beep_stop()

    def _on_session_result(self, result: TranscriptionResult) -> None:
        if not result.text:
            logger.warning("Empty transcription result")
            if self.config.audio.beep_enabled:
                audio.beep_error()
            return

        logger.info(f"Transcribed: {result.text[:100]}...")
        entry = self._history.add(result.text)
        logger.debug(f"Added to history: entry {entry.id}")

        if self._tray is not None:
            self._tray.notify_menu_updated(self._history.revision)

        success = inject_to_clipboard(result.text)
        if self.config.audio.beep_enabled:
            if success:
                audio.beep_success()
            else:
                audio.beep_error()

    def _on_session_error(self, exc: BaseException) -> None:
        logger.error(f"Recording session error: {exc}")
        if self.config.audio.beep_enabled:
            audio.beep_error()

    # ----- tray -----

    def _update_tray_state(self, state: SessionState) -> None:
        if self._tray is None:
            return
        if state == SessionState.RECORDING:
            self._tray.set_state(TrayState.RECORDING)
        elif state == SessionState.TRANSCRIBING:
            self._tray.set_state(TrayState.TRANSCRIBING)
        else:
            self._tray.set_state(TrayState.IDLE)

    def _on_tray_toggle(self) -> None:
        self._handle_toggle({})

    def _on_tray_quit(self) -> None:
        self._shutdown_event.set()

    def _on_copy_history(self, text: str) -> None:
        logger.info(f"Copying history item to clipboard: {text[:50]}...")
        success = inject_to_clipboard(text)
        if success and self.config.audio.beep_enabled:
            audio.beep_success()
        elif not success:
            logger.error("Failed to copy history item to clipboard")

    def _setup_signals(self) -> None:
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}")
            self._shutdown_event.set()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    def run(self) -> None:
        """Run the daemon."""
        logger.info("Starting daemon")

        self._setup_signals()
        self._write_pid()

        try:
            # IPC server for local clients
            self._server = Server()
            self._server.register_handler("toggle", self._handle_toggle)
            self._server.register_handler("status", self._handle_status)
            self._server.register_handler("stop", self._handle_stop)
            self._server.start()

            # Tray icon
            self._tray = TrayIcon(
                on_toggle=self._on_tray_toggle,
                on_quit=self._on_tray_quit,
                get_history=self._history.get_all,
                get_history_by_id=self._history.get_by_id,
                get_revision=lambda: self._history.revision,
                on_copy_history=self._on_copy_history,
            )
            self._tray.start()
            logger.info("Tray icon started")

            # Build the voice-capabilities composition root. No model is
            # loaded here: the inference worker process starts lazily on the
            # first STT/TTS operation and is terminated after the idle
            # timeout so its VRAM returns to the system.
            self._voiced = Voiced.from_config(self.config)

            # Recording session — owns state, recorder, transcription worker
            self._session = RecordingSession(
                transcriber=self._voiced.transcriber,
                audio_config=self.config.audio,
                sample_rate=self.config.audio.sample_rate,
            )
            self._session.on_state_change(self._on_session_state_change)
            self._session.on_result(self._on_session_result)
            self._session.on_error(self._on_session_error)

            # Optional HTTP server (shares the loaded transcriber)
            if self._http_enabled:
                self._start_http_server()

            self._running = True
            logger.info("Daemon ready")

            while not self._shutdown_event.is_set():
                self._shutdown_event.wait(timeout=1.0)

        except Exception:
            logger.exception("Daemon error")
            raise
        finally:
            self._cleanup()

    def _start_http_server(self) -> None:
        """Start the embedded HTTP server, sharing the daemon's Voiced instance."""
        server_class = _get_transcription_server()
        self._http_server = server_class(
            host=self._http_host,
            port=self._http_port,
            config=self.config,
            voiced=self._voiced,
        )
        self._http_server.start_background(preload=False)
        logger.info(f"HTTP server started on {self._http_host}:{self._http_port}")

    def _cleanup(self) -> None:
        logger.info("Cleaning up")
        self._running = False

        if self._session is not None:
            self._session.shutdown()
            self._session = None

        if self._http_server is not None:
            self._http_server.stop()
            self._http_server = None

        if self._server is not None:
            self._server.stop()

        if self._voiced is not None:
            self._voiced.shutdown()
            self._voiced = None

        if self._tray is not None:
            self._tray.stop()

        self._remove_pid()
        logger.info("Daemon stopped")


def daemonize() -> None:
    """Daemonize the current process (double fork)."""
    # First fork
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0)
    except OSError as e:
        logger.error(f"First fork failed: {e}")
        sys.exit(1)

    os.chdir("/")
    os.setsid()
    os.umask(0)

    # Second fork
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0)
    except OSError as e:
        logger.error(f"Second fork failed: {e}")
        sys.exit(1)

    sys.stdout.flush()
    sys.stderr.flush()

    with open("/dev/null", "rb", 0) as f:
        os.dup2(f.fileno(), sys.stdin.fileno())
    with open("/dev/null", "ab", 0) as f:
        os.dup2(f.fileno(), sys.stdout.fileno())
    with open("/dev/null", "ab", 0) as f:
        os.dup2(f.fileno(), sys.stderr.fileno())


def get_running_pid() -> int | None:
    """Get the PID of a running daemon."""
    pid_path = get_pid_path()
    if not pid_path.exists():
        return None

    try:
        with open(pid_path) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        return None


def is_daemon_running() -> bool:
    """Check if daemon is running."""
    return get_running_pid() is not None
