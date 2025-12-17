"""Daemon process for sttd."""

import logging
import os
import queue
import signal
import sys
import threading
from enum import Enum

import numpy as np

from sttd import audio
from sttd.config import Config, get_pid_path, load_config
from sttd.injector import inject_backspaces, inject_text
from sttd.recorder import Recorder
from sttd.server import Server
from sttd.transcriber import Transcriber
from sttd.tray import TrayIcon, TrayState

logger = logging.getLogger(__name__)


class DaemonState(Enum):
    """Daemon state."""

    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"


class Daemon:
    """Main daemon process for speech-to-text."""

    def __init__(self, config: Config | None = None):
        """Initialize the daemon.

        Args:
            config: Configuration. Uses defaults if not provided.
        """
        self.config = config or load_config()
        self._state = DaemonState.IDLE
        self._running = False
        self._shutdown_event = threading.Event()

        # Components
        self._server: Server | None = None
        self._recorder: Recorder | None = None
        self._transcriber: Transcriber | None = None
        self._tray: TrayIcon | None = None

        # Streaming transcription
        self._audio_chunk_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._streaming_thread: threading.Thread | None = None

        # Streaming transcription tracking
        self._accumulated_audio: list[np.ndarray] = []
        self._previous_text: str = ""
        self._previous_text_len: int = 0
        self._finalized_context: str = ""  # Context from chunks that fell off window

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
        """Handle toggle command."""
        if self._state == DaemonState.IDLE:
            return self._start_recording()
        elif self._state == DaemonState.RECORDING:
            return self._stop_recording()
        else:
            return {
                "status": "busy",
                "message": f"Daemon is busy ({self._state.value})",
            }

    def _handle_status(self, args: dict) -> dict:
        """Handle status command."""
        return {
            "status": "ok",
            "state": self._state.value,
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

        # Check if streaming is enabled
        if self.config.transcription.streaming:
            # Clear the chunk queue
            while not self._audio_chunk_queue.empty():
                try:
                    self._audio_chunk_queue.get_nowait()
                except queue.Empty:
                    break

            # Reset streaming transcription state
            self._accumulated_audio = []
            self._previous_text = ""
            self._previous_text_len = 0
            self._finalized_context = ""

            # Start streaming transcription thread
            self._streaming_thread = threading.Thread(
                target=self._streaming_transcribe,
                daemon=True,
            )
            self._streaming_thread.start()

            # Create recorder with chunk callback
            self._recorder = Recorder(
                config=self.config.audio,
                on_chunk=self._on_audio_chunk,
                chunk_duration=self.config.transcription.chunk_duration,
            )
        else:
            # Non-streaming mode
            self._recorder = Recorder(config=self.config.audio)

        self._recorder.start()

        # Play start beep
        if self.config.audio.beep_enabled:
            threading.Thread(target=audio.beep_start, daemon=True).start()

        logger.info(f"Recording started (streaming={self.config.transcription.streaming})")
        return {"status": "ok", "state": "recording"}

    def _stop_recording(self) -> dict:
        """Stop recording and transcribe."""
        if self._recorder is None or not self._recorder.is_recording:
            return {"status": "error", "message": "Not recording"}

        # Play stop beep
        if self.config.audio.beep_enabled:
            audio.beep_stop()

        # Stop recording (this will flush any remaining audio chunks)
        audio_data = self._recorder.stop()

        # Handle streaming mode
        if self.config.transcription.streaming:
            # Signal streaming thread to stop
            self._audio_chunk_queue.put(None)

            # Wait for streaming thread to finish
            if self._streaming_thread is not None:
                self._streaming_thread.join(timeout=5.0)
                self._streaming_thread = None

            # In streaming mode, we already transcribed as we went
            self._state = DaemonState.IDLE
            self._update_tray_state()
            logger.info("Streaming recording stopped")
            return {"status": "ok", "state": "idle"}

        # Non-streaming mode: batch transcription
        self._state = DaemonState.TRANSCRIBING
        self._update_tray_state()

        if len(audio_data) == 0:
            self._state = DaemonState.IDLE
            return {"status": "error", "message": "No audio recorded"}

        # Transcribe in a separate thread to not block
        threading.Thread(
            target=self._transcribe_and_inject,
            args=(audio_data,),
            daemon=True,
        ).start()

        return {"status": "ok", "state": "transcribing"}

    def _on_audio_chunk(self, audio_data: np.ndarray) -> None:
        """Callback when a chunk of audio is ready for streaming transcription."""
        self._audio_chunk_queue.put(audio_data)

    def _streaming_transcribe(self) -> None:
        """Background thread for streaming transcription with sliding window.

        Uses a sliding window to keep memory bounded. When chunks fall off the
        window, their text is preserved as context via initial_prompt.
        """
        logger.info("Streaming transcription thread started")
        max_chunks = int(
            self.config.transcription.max_window / self.config.transcription.chunk_duration
        )

        while self._state == DaemonState.RECORDING:
            try:
                chunk = self._audio_chunk_queue.get(timeout=0.5)

                if chunk is None:
                    break

                self._accumulated_audio.append(chunk)

                # Sliding window: trim oldest chunks, preserve text as context
                while len(self._accumulated_audio) > max_chunks:
                    self._accumulated_audio.pop(0)
                    if self._previous_text:
                        self._finalized_context = self._previous_text

                if self._transcriber is None:
                    self._transcriber = Transcriber(self.config.transcription)

                full_audio = np.concatenate(self._accumulated_audio)
                new_text = self._transcriber.transcribe_audio(
                    full_audio,
                    sample_rate=self.config.audio.sample_rate,
                    initial_prompt=self._finalized_context or None,
                )

                if new_text:
                    new_text = new_text.strip()

                if new_text and new_text != self._previous_text:
                    logger.info(
                        f"Streaming ({len(self._accumulated_audio)} chunks): {new_text[:50]}..."
                    )

                    if self._previous_text_len > 0:
                        inject_backspaces(self._previous_text_len)

                    inject_text(new_text, self.config.output.method)

                    self._previous_text = new_text
                    self._previous_text_len = len(new_text)

            except queue.Empty:
                continue
            except Exception as e:
                logger.warning(f"Streaming transcription error: {e}")

        logger.info("Streaming transcription thread stopped")

    def _transcribe_and_inject(self, audio_data: np.ndarray) -> None:
        """Transcribe audio and inject text (batch mode)."""
        try:
            if self._transcriber is None:
                self._transcriber = Transcriber(self.config.transcription)

            text = self._transcriber.transcribe_audio(
                audio_data,
                sample_rate=self.config.audio.sample_rate,
            )

            if text:
                logger.info(f"Transcribed: {text[:100]}...")
                success = inject_text(text, self.config.output.method)

                if success and self.config.audio.beep_enabled:
                    audio.beep_success()
                elif not success and self.config.audio.beep_enabled:
                    audio.beep_error()
            else:
                logger.warning("No text transcribed")
                if self.config.audio.beep_enabled:
                    audio.beep_error()

        except Exception:
            logger.exception("Transcription error")
            if self.config.audio.beep_enabled:
                audio.beep_error()
        finally:
            self._state = DaemonState.IDLE
            self._update_tray_state()

    def _update_tray_state(self) -> None:
        """Update tray icon to match daemon state."""
        if self._tray is None:
            return

        state_map = {
            DaemonState.IDLE: TrayState.IDLE,
            DaemonState.RECORDING: TrayState.RECORDING,
            DaemonState.TRANSCRIBING: TrayState.TRANSCRIBING,
        }
        self._tray.set_state(state_map.get(self._state, TrayState.IDLE))

    def _on_tray_toggle(self) -> None:
        """Handle tray icon toggle click."""
        self._handle_toggle({})

    def _on_tray_quit(self) -> None:
        """Handle tray icon quit click."""
        self._shutdown_event.set()

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
            # Initialize server
            self._server = Server()
            self._server.register_handler("toggle", self._handle_toggle)
            self._server.register_handler("status", self._handle_status)
            self._server.register_handler("stop", self._handle_stop)
            self._server.start()

            # Start tray icon
            self._tray = TrayIcon(
                on_toggle=self._on_tray_toggle,
                on_quit=self._on_tray_quit,
            )
            self._tray.start()
            logger.info("Tray icon started")

            # Pre-load model (optional, speeds up first transcription)
            logger.info("Pre-loading transcription model...")
            self._transcriber = Transcriber(self.config.transcription)
            _ = self._transcriber.model  # Trigger model load
            logger.info("Model loaded")

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

    def _cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up")
        self._running = False

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
