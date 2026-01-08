"""Daemon process for voiced."""

import logging
import os
import signal
import sys
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from queue import Queue

import numpy as np

from voiced import audio
from voiced.config import Config, get_pid_path, load_config
from voiced.injector import inject_to_clipboard
from voiced.recorder import Recorder
from voiced.server import Server
from voiced.transcriber import Transcriber
from voiced.tray import TrayIcon, TrayState

logger = logging.getLogger(__name__)

# Lazy import to avoid loading HTTP server dependencies unless needed
TranscriptionServer = None


# History and Queue Data Structures


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


@dataclass
class TranscriptionJob:
    """A queued transcription job."""

    audio_data: np.ndarray
    job_id: int


class TranscriptionQueue:
    """Thread-safe queue for pending transcriptions."""

    def __init__(self):
        self._queue: Queue[TranscriptionJob] = Queue()
        self._next_job_id = 1
        self._lock = threading.Lock()
        self._pending_count = 0

    def enqueue(self, audio_data: np.ndarray) -> int:
        """Add audio to the transcription queue. Returns job ID."""
        with self._lock:
            job_id = self._next_job_id
            self._next_job_id += 1
            self._pending_count += 1

        job = TranscriptionJob(audio_data=audio_data, job_id=job_id)
        self._queue.put(job)
        return job_id

    def dequeue(self, timeout: float = 0.5) -> TranscriptionJob | None:
        """Get next job from queue. Returns None if empty/timeout."""
        try:
            job = self._queue.get(timeout=timeout)
            with self._lock:
                self._pending_count -= 1
            return job
        except Exception:
            return None

    @property
    def pending_count(self) -> int:
        """Number of jobs waiting in queue."""
        with self._lock:
            return self._pending_count

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()


def _get_transcription_server():
    """Lazy load TranscriptionServer to avoid import overhead."""
    global TranscriptionServer
    if TranscriptionServer is None:
        from voiced.http_server import TranscriptionServer

    return TranscriptionServer


class DaemonState(Enum):
    """Daemon state."""

    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"


class Daemon:
    """Main daemon process for speech-to-text."""

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
        self._state = DaemonState.IDLE
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
        self._recorder: Recorder | None = None
        self._transcriber: Transcriber | None = None
        self._tray: TrayIcon | None = None
        self._http_server = None  # TranscriptionServer instance

        # History and transcription queue
        self._history = TranscriptionHistory()
        self._transcription_queue = TranscriptionQueue()

        # Background transcription worker
        self._transcription_worker: threading.Thread | None = None
        self._worker_stop_event = threading.Event()

        # Track if transcription is active (for tray icon)
        self._is_transcribing = False
        self._transcription_lock = threading.Lock()

        # PID file
        self._pid_path = get_pid_path()

    @property
    def state(self) -> DaemonState:
        """Get the current daemon state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if daemon is running."""
        return self._running

    def _write_pid(self) -> None:
        """Write PID file."""
        self._pid_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._pid_path, "w") as f:
            f.write(str(os.getpid()))
        logger.info(f"PID file written: {self._pid_path}")

    def _remove_pid(self) -> None:
        """Remove PID file."""
        if self._pid_path.exists():
            try:
                self._pid_path.unlink()
                logger.info("PID file removed")
            except Exception as e:
                logger.warning(f"Failed to remove PID file: {e}")

    def _handle_toggle(self, args: dict) -> dict:
        """Handle toggle command.

        Recording can start even while transcription is processing in the background.
        """
        if self._state == DaemonState.IDLE:
            return self._start_recording()
        elif self._state == DaemonState.RECORDING:
            return self._stop_recording()
        else:
            # Should not reach here with new state machine
            return {
                "status": "busy",
                "message": f"Daemon is busy ({self._state.value})",
            }

    def _handle_status(self, args: dict) -> dict:
        """Handle status command."""
        with self._transcription_lock:
            is_transcribing = self._is_transcribing
        return {
            "status": "ok",
            "state": self._state.value,
            "transcribing": is_transcribing,
            "queue_pending": self._transcription_queue.pending_count,
            "history_count": len(self._history.get_all()),
            "model": self.config.transcription.model,
            "device": self.config.transcription.device,
        }

    def _handle_stop(self, args: dict) -> dict:
        """Handle stop command."""
        logger.info("Received stop command")
        self._shutdown_event.set()
        return {"status": "ok", "message": "Daemon stopping"}

    def _start_recording(self) -> dict:
        """Start recording audio."""
        if self._recorder is not None and self._recorder.is_recording:
            return {"status": "error", "message": "Already recording"}

        self._state = DaemonState.RECORDING
        self._update_tray_state()

        self._recorder = Recorder(config=self.config.audio)
        self._recorder.start()

        # Play start beep
        if self.config.audio.beep_enabled:
            threading.Thread(target=audio.beep_start, daemon=True).start()

        logger.info("Recording started")
        return {"status": "ok", "state": "recording"}

    def _stop_recording(self) -> dict:
        """Stop recording and queue transcription.

        Returns immediately to IDLE state, allowing new recordings to start
        while previous transcriptions are processed in the background.
        """
        if self._recorder is None or not self._recorder.is_recording:
            return {"status": "error", "message": "Not recording"}

        # Play stop beep
        if self.config.audio.beep_enabled:
            audio.beep_stop()

        # Stop recording
        audio_data = self._recorder.stop()

        # Return to IDLE immediately (allows new recordings)
        self._state = DaemonState.IDLE
        self._update_tray_state()

        if len(audio_data) == 0:
            return {"status": "error", "message": "No audio recorded"}

        # Queue for background transcription
        job_id = self._transcription_queue.enqueue(audio_data)
        logger.info(f"Queued transcription job {job_id}")

        return {
            "status": "ok",
            "state": "queued",
            "job_id": job_id,
            "queue_pending": self._transcription_queue.pending_count,
        }

    def _start_transcription_worker(self) -> None:
        """Start the background transcription worker thread."""
        self._worker_stop_event.clear()
        self._transcription_worker = threading.Thread(
            target=self._transcription_worker_loop,
            daemon=True,
            name="TranscriptionWorker",
        )
        self._transcription_worker.start()
        logger.info("Transcription worker started")

    def _stop_transcription_worker(self) -> None:
        """Stop the background transcription worker."""
        self._worker_stop_event.set()
        if self._transcription_worker is not None:
            self._transcription_worker.join(timeout=2.0)
            self._transcription_worker = None
        logger.info("Transcription worker stopped")

    def _transcription_worker_loop(self) -> None:
        """Background loop that processes queued transcriptions one at a time."""
        while not self._worker_stop_event.is_set():
            job = self._transcription_queue.dequeue(timeout=0.5)
            if job is None:
                continue

            self._process_transcription_job(job)

    def _process_transcription_job(self, job: TranscriptionJob) -> None:
        """Process a single transcription job."""
        logger.info(f"Processing transcription job {job.job_id}")

        # Update tray to show transcribing
        with self._transcription_lock:
            self._is_transcribing = True
        self._update_tray_state()

        try:
            if self._transcriber is None:
                self._transcriber = Transcriber(
                    self.config.transcription, vad_config=self.config.vad
                )

            text = self._transcriber.transcribe_audio(
                job.audio_data,
                sample_rate=self.config.audio.sample_rate,
            )

            if text:
                logger.info(f"Job {job.job_id} transcribed: {text[:100]}...")

                # Add to history
                entry = self._history.add(text)
                logger.debug(f"Added to history: entry {entry.id}")

                # Notify tray menu of layout change
                if self._tray is not None:
                    self._tray.notify_menu_updated(self._history.revision)

                # Copy to clipboard
                success = inject_to_clipboard(text)

                if success and self.config.audio.beep_enabled:
                    audio.beep_success()
                elif not success and self.config.audio.beep_enabled:
                    audio.beep_error()
            else:
                logger.warning(f"Job {job.job_id}: No text transcribed")
                if self.config.audio.beep_enabled:
                    audio.beep_error()

        except Exception:
            logger.exception(f"Transcription error for job {job.job_id}")
            if self.config.audio.beep_enabled:
                audio.beep_error()
        finally:
            with self._transcription_lock:
                self._is_transcribing = False
            self._update_tray_state()

    def _update_tray_state(self) -> None:
        """Update tray icon to match daemon state.

        The tray shows TRANSCRIBING if background transcription is active,
        even when the daemon state is IDLE (allowing new recordings).
        """
        if self._tray is None:
            return

        if self._state == DaemonState.RECORDING:
            self._tray.set_state(TrayState.RECORDING)
        elif self._is_transcribing:
            self._tray.set_state(TrayState.TRANSCRIBING)
        else:
            self._tray.set_state(TrayState.IDLE)

    def _on_tray_toggle(self) -> None:
        """Handle tray icon toggle click."""
        self._handle_toggle({})

    def _on_tray_quit(self) -> None:
        """Handle tray icon quit click."""
        self._shutdown_event.set()

    def _on_copy_history(self, text: str) -> None:
        """Handle copying a history item to clipboard."""
        logger.info(f"Copying history item to clipboard: {text[:50]}...")
        success = inject_to_clipboard(text)
        if success and self.config.audio.beep_enabled:
            audio.beep_success()
        elif not success:
            logger.error("Failed to copy history item to clipboard")

    def _setup_signals(self) -> None:
        """Set up signal handlers."""

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
            # Initialize Unix socket server for local IPC
            self._server = Server()
            self._server.register_handler("toggle", self._handle_toggle)
            self._server.register_handler("status", self._handle_status)
            self._server.register_handler("stop", self._handle_stop)
            self._server.start()

            # Start tray icon with history callbacks
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

            # Pre-load model (optional, speeds up first transcription)
            logger.info("Pre-loading transcription model...")
            self._transcriber = Transcriber(self.config.transcription, vad_config=self.config.vad)
            _ = self._transcriber.model  # Trigger model load
            logger.info("Model loaded")

            # Start background transcription worker
            self._start_transcription_worker()

            # Start HTTP server if enabled (shares transcriber with daemon)
            if self._http_enabled:
                self._start_http_server()

            self._running = True
            logger.info("Daemon ready")

            # Wait for shutdown
            while not self._shutdown_event.is_set():
                self._shutdown_event.wait(timeout=1.0)

        except Exception:
            logger.exception("Daemon error")
            raise
        finally:
            self._cleanup()

    def _start_http_server(self) -> None:
        """Start the embedded HTTP server in a background thread."""
        server_class = _get_transcription_server()

        # Create HTTP server that shares the transcriber instance
        self._http_server = server_class(
            host=self._http_host,
            port=self._http_port,
            config=self.config,
        )

        # Share the already-loaded transcriber to avoid loading model twice
        self._http_server.transcriber = self._transcriber

        # Start in background thread (preload=False since model already loaded)
        self._http_server.start_background(preload=False)
        logger.info(f"HTTP server started on {self._http_host}:{self._http_port}")

    def _cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up")
        self._running = False

        # Stop transcription worker first
        self._stop_transcription_worker()

        # Stop HTTP server (it may be using the transcriber)
        if self._http_server is not None:
            self._http_server.stop()
            self._http_server = None

        if self._server is not None:
            self._server.stop()

        if self._recorder is not None and self._recorder.is_recording:
            self._recorder.stop()

        if self._transcriber is not None:
            self._transcriber.unload()

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
            # Parent exits
            sys.exit(0)
    except OSError as e:
        logger.error(f"First fork failed: {e}")
        sys.exit(1)

    # Decouple from parent environment
    os.chdir("/")
    os.setsid()
    os.umask(0)

    # Second fork
    try:
        pid = os.fork()
        if pid > 0:
            # Parent exits
            sys.exit(0)
    except OSError as e:
        logger.error(f"Second fork failed: {e}")
        sys.exit(1)

    # Redirect standard file descriptors
    sys.stdout.flush()
    sys.stderr.flush()

    with open("/dev/null", "rb", 0) as f:
        os.dup2(f.fileno(), sys.stdin.fileno())
    with open("/dev/null", "ab", 0) as f:
        os.dup2(f.fileno(), sys.stdout.fileno())
    with open("/dev/null", "ab", 0) as f:
        os.dup2(f.fileno(), sys.stderr.fileno())


def get_running_pid() -> int | None:
    """Get the PID of a running daemon.

    Returns:
        PID if daemon is running, None otherwise.
    """
    pid_path = get_pid_path()
    if not pid_path.exists():
        return None

    try:
        with open(pid_path) as f:
            pid = int(f.read().strip())

        # Check if process is running
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        return None


def is_daemon_running() -> bool:
    """Check if daemon is running.

    Returns:
        True if daemon is running, False otherwise.
    """
    return get_running_pid() is not None
