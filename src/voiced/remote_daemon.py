"""Remote client daemon for voiced.

Records audio locally and sends to a remote server for transcription.
"""

import logging
import os
import signal
import sys
import threading
from enum import Enum

import numpy as np

from voiced import audio
from voiced.config import Config, get_cache_dir, load_config
from voiced.http_client import HttpConnectionError, HttpTimeoutError, ServerError, TranscriptionClient
from voiced.injector import inject_to_clipboard
from voiced.recorder import Recorder
from voiced.tray import TrayIcon, TrayState

logger = logging.getLogger(__name__)


class ClientState(Enum):
    """Client daemon state."""

    IDLE = "idle"
    RECORDING = "recording"
    SENDING = "sending"


class RemoteDaemon:
    """Client daemon that records locally and transcribes remotely."""

    def __init__(
        self,
        server_url: str,
        config: Config | None = None,
        timeout: float | None = None,
    ):
        """Initialize the remote daemon.

        Args:
            server_url: URL of the transcription server.
            config: Configuration. Uses defaults if not provided.
            timeout: Request timeout in seconds.
        """
        self.config = config or load_config()
        self.server_url = server_url

        effective_timeout = timeout or self.config.client.timeout
        self.client = TranscriptionClient(server_url, timeout=effective_timeout)

        self._state = ClientState.IDLE
        self._running = False
        self._shutdown_event = threading.Event()

        self._recorder: Recorder | None = None
        self._tray: TrayIcon | None = None

        self._pid_path = get_cache_dir() / "client.pid"

    @property
    def state(self) -> ClientState:
        """Get the current client state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if client is running."""
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

    def _handle_toggle(self) -> dict:
        """Handle toggle command."""
        if self._state == ClientState.IDLE:
            return self._start_recording()
        elif self._state == ClientState.RECORDING:
            return self._stop_and_send()
        else:
            return {
                "status": "busy",
                "message": f"Client is busy ({self._state.value})",
            }

    def _start_recording(self) -> dict:
        """Start recording audio."""
        if self._recorder is not None and self._recorder.is_recording:
            return {"status": "error", "message": "Already recording"}

        self._state = ClientState.RECORDING
        self._update_tray_state()

        self._recorder = Recorder(config=self.config.audio)
        self._recorder.start()

        if self.config.audio.beep_enabled:
            threading.Thread(target=audio.beep_start, daemon=True).start()

        logger.info("Recording started")
        return {"status": "ok", "state": "recording"}

    def _stop_and_send(self) -> dict:
        """Stop recording and send to server."""
        if self._recorder is None or not self._recorder.is_recording:
            return {"status": "error", "message": "Not recording"}

        if self.config.audio.beep_enabled:
            audio.beep_stop()

        audio_data = self._recorder.stop()

        self._state = ClientState.SENDING
        self._update_tray_state()

        if len(audio_data) == 0:
            self._state = ClientState.IDLE
            self._update_tray_state()
            return {"status": "error", "message": "No audio recorded"}

        threading.Thread(
            target=self._send_and_inject,
            args=(audio_data,),
            daemon=True,
        ).start()

        return {"status": "ok", "state": "sending"}

    def _send_and_inject(self, audio_data: np.ndarray) -> None:
        """Send audio to server and inject result to clipboard."""
        try:
            text = self.client.transcribe(
                audio_data,
                sample_rate=self.config.audio.sample_rate,
                language=self.config.transcription.language,
            )

            if text:
                logger.info(f"Transcribed: {text[:100]}...")
                success = inject_to_clipboard(text)

                if success and self.config.audio.beep_enabled:
                    audio.beep_success()
                elif not success and self.config.audio.beep_enabled:
                    audio.beep_error()
            else:
                logger.warning("No text transcribed")
                if self.config.audio.beep_enabled:
                    audio.beep_error()

        except HttpConnectionError as e:
            logger.error(f"Connection error: {e}")
            if self.config.audio.beep_enabled:
                audio.beep_error()
        except ServerError as e:
            logger.error(f"Server error: {e}")
            if self.config.audio.beep_enabled:
                audio.beep_error()
        except HttpTimeoutError as e:
            logger.error(f"Timeout: {e}")
            if self.config.audio.beep_enabled:
                audio.beep_error()
        except Exception:
            logger.exception("Unexpected error during transcription")
            if self.config.audio.beep_enabled:
                audio.beep_error()
        finally:
            self._state = ClientState.IDLE
            self._update_tray_state()

    def _update_tray_state(self) -> None:
        """Update tray icon to match client state."""
        if self._tray is None:
            return

        state_map = {
            ClientState.IDLE: TrayState.IDLE,
            ClientState.RECORDING: TrayState.RECORDING,
            ClientState.SENDING: TrayState.TRANSCRIBING,
        }
        self._tray.set_state(state_map.get(self._state, TrayState.IDLE))

    def _on_tray_toggle(self) -> None:
        """Handle tray icon toggle click."""
        self._handle_toggle()

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

    def _check_server(self) -> bool:
        """Check if server is available."""
        try:
            health = self.client.health_check()
            logger.info(
                f"Server available: model={health.get('model')}, device={health.get('device')}"
            )
            return True
        except HttpConnectionError as e:
            logger.error(f"Server not available: {e}")
            return False

    def run(self) -> None:
        """Run the remote client daemon."""
        logger.info(f"Starting remote client (server: {self.server_url})")

        if not self._check_server():
            logger.error("Cannot connect to server. Is it running?")
            sys.exit(1)

        self._setup_signals()
        self._write_pid()

        try:
            self._tray = TrayIcon(
                on_toggle=self._on_tray_toggle,
                on_quit=self._on_tray_quit,
            )
            self._tray.start()
            logger.info("Tray icon started")

            self._running = True
            logger.info("Remote client ready")

            while not self._shutdown_event.is_set():
                self._shutdown_event.wait(timeout=1.0)

        except Exception:
            logger.exception("Remote client error")
            raise
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up")
        self._running = False

        if self._recorder is not None and self._recorder.is_recording:
            self._recorder.stop()

        if self._tray is not None:
            self._tray.stop()

        self._remove_pid()
        logger.info("Remote client stopped")


def get_client_pid() -> int | None:
    """Get the PID of a running client daemon.

    Returns:
        PID if client is running, None otherwise.
    """
    pid_path = get_cache_dir() / "client.pid"
    if not pid_path.exists():
        return None

    try:
        with open(pid_path) as f:
            pid = int(f.read().strip())

        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        return None


def is_client_running() -> bool:
    """Check if client daemon is running.

    Returns:
        True if client is running, False otherwise.
    """
    return get_client_pid() is not None
