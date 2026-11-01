"""Remote client daemon for voiced.

Records audio locally and sends to a remote server for transcription.
"""

import json
import logging
import os
import signal
import sys
import threading
from collections import deque
from datetime import datetime
from enum import Enum
from urllib.error import URLError
from urllib.request import Request, urlopen

from voiced import audio
from voiced.config import Config, get_cache_dir, load_config
from voiced.injector import inject_to_clipboard
from voiced.server import Server
from voiced.tray import TrayIcon, TrayState
from voiced.webrtc_client import SyncWebRTCClient, WebRTCClientConfig

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
        self._webrtc_config = WebRTCClientConfig(
            server_url=server_url,
            timeout=effective_timeout,
            input_device=None
            if self.config.audio.device == "default"
            else self.config.audio.device,
        )

        self._state = ClientState.IDLE
        self._running = False
        self._shutdown_event = threading.Event()
        self._lock = threading.RLock()
        self._tray: TrayIcon | None = None
        self._server: Server | None = None
        self._history: deque[dict] = deque(maxlen=10)
        self._next_history_id = 1
        self._last_error: str | None = None
        self._cancel_generation = 0
        self._partial_text = ""
        self._context_mode = "default"
        self._recording_mode = "default"
        self._input_level = 0.0
        self._pid_path = get_cache_dir() / "client.pid"
        self._worker: threading.Thread | None = None
        self._stream_client: SyncWebRTCClient | None = None

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

    def _handle_toggle(self, args: dict | None = None) -> dict:
        with self._lock:
            if self._state == ClientState.IDLE:
                if self._worker is not None and self._worker.is_alive():
                    return self._result("rejected", "transcription in progress")
                return self._start_recording()
            if self._state == ClientState.RECORDING:
                return self._stop_and_send()
            return {
                "status": "rejected",
                "state": "transcribing",
                "reason": "transcription in progress",
            }

    def _result(self, status: str = "ok", message: str | None = None) -> dict:
        result = {
            "status": status,
            "state": "transcribing" if self._state == ClientState.SENDING else self._state.value,
        }
        if message:
            result["message"] = message
        return result

    def _handle_start_recording(self, args: dict | None = None) -> dict:
        with self._lock:
            if self._state != ClientState.IDLE or (
                self._worker is not None and self._worker.is_alive()
            ):
                return self._result("rejected", "not idle or transcription in progress")
            return self._start_recording()

    def _handle_stop_recording(self, args: dict | None = None) -> dict:
        with self._lock:
            if self._state != ClientState.RECORDING:
                return self._result("rejected", "not recording")
            return self._stop_and_send()

    def _handle_cancel(self, args: dict | None = None) -> dict:
        with self._lock:
            if self._state not in (ClientState.RECORDING, ClientState.SENDING):
                return self._result("rejected", "not recording")
            self._cancel_generation += 1
            client = self._stream_client
            self._stream_client = None
            self._state = ClientState.IDLE
            self._partial_text = ""
        if client is not None:
            client.disconnect()
        self._update_tray_state()
        return self._result()

    def _serialize_history(self) -> list[dict]:
        return list(self._history)

    def _handle_history(self, args: dict | None = None) -> dict:
        with self._lock:
            return {"status": "ok", "history": self._serialize_history()}

    def _handle_copy_last(self, args: dict | None = None) -> dict:
        with self._lock:
            if not self._history:
                return {"status": "error", "message": "no transcript available"}
            text = self._history[0]["text"]
        success = inject_to_clipboard(text)
        return {"status": "ok" if success else "error", "text": text}

    def _handle_status(self, args: dict | None = None) -> dict:
        with self._lock:
            history = self._serialize_history()
            input_level = (
                self._stream_client.input_level if self._stream_client is not None else 0.0
            )
            return {
                "status": "ok",
                "state": "transcribing"
                if self._state == ClientState.SENDING
                else self._state.value,
                "transcribing": self._state == ClientState.SENDING,
                "history_count": len(history),
                "history": history,
                "last_transcript": history[0]["text"] if history else None,
                "error": self._last_error,
                "mode": "remote",
                "model": None,
                "device": self.config.transcription.device,
                "partial_text": self._partial_text,
                "input_level": input_level,
                "context_mode": self._context_mode,
            }

    def _handle_set_context_mode(self, args: dict | None = None) -> dict:
        mode = (args or {}).get("mode")
        if not isinstance(mode, str) or not mode:
            return {"status": "error", "message": "mode must be a non-empty string"}
        with self._lock:
            configured = self.config.transcription.context_replacements
            if mode != "default" and mode not in configured:
                return {"status": "error", "message": f"unknown context mode: {mode}"}
            self._context_mode = mode
            return {"status": "ok", "context_mode": mode}

    def _fix_context(self, text: str, mode: str) -> str:
        import re

        with self._lock:
            replacements = dict(self.config.transcription.replacements)
            replacements.update(self.config.transcription.context_replacements.get(mode, {}))
        for wrong, right in replacements.items():
            text = re.sub(rf"\b{re.escape(wrong)}\b", lambda _: right, text, flags=re.IGNORECASE)
        return text

    def _handle_update_replacements(self, args: dict | None = None) -> dict:
        args = args or {}
        replacements = args.get("replacements")
        context_replacements = args.get(
            "context_replacements", self.config.transcription.context_replacements
        )
        if not self._valid_replacement_map(replacements) or not self._valid_context_replacements(
            context_replacements
        ):
            return {"status": "error", "message": "invalid replacement mappings"}
        with self._lock:
            if self._context_mode != "default" and self._context_mode not in context_replacements:
                return {"status": "error", "message": f"unknown context mode: {self._context_mode}"}
            self.config.transcription.replacements = dict(replacements)
            self.config.transcription.context_replacements = {
                mode: dict(mapping) for mode, mapping in context_replacements.items()
            }
            return {"status": "ok", "context_mode": self._context_mode}

    @staticmethod
    def _valid_replacement_map(mapping) -> bool:
        return isinstance(mapping, dict) and all(
            isinstance(key, str) and isinstance(value, str) for key, value in mapping.items()
        )

    @classmethod
    def _valid_context_replacements(cls, mapping) -> bool:
        return isinstance(mapping, dict) and all(
            isinstance(mode, str) and cls._valid_replacement_map(replacements)
            for mode, replacements in mapping.items()
        )

    def _handle_shutdown(self, args: dict | None = None) -> dict:
        self._shutdown_event.set()
        return {"status": "ok", "message": "Daemon stopping"}

    def _start_recording(self) -> dict:
        if self._state != ClientState.IDLE:
            return self._result("rejected", "not idle")
        client = SyncWebRTCClient(self._webrtc_config)
        try:
            client.connect()
            with self._lock:
                mode = self._context_mode

            def on_partial(text: str) -> None:
                with self._lock:
                    if self._state == ClientState.RECORDING:
                        self._partial_text = self._fix_context(text, mode)

            client.start_stt_stream(
                on_partial=on_partial,
                language=self.config.transcription.language,
            )
        except Exception as exc:
            client.disconnect()
            self._last_error = str(exc)
            return self._result("error", self._last_error)
        self._stream_client = client
        self._recording_mode = mode
        self._state = ClientState.RECORDING
        if self.config.audio.beep_enabled:
            threading.Thread(target=audio.beep_start, daemon=True).start()
        return self._result()

    def _stop_and_send(self) -> dict:
        if self._stream_client is None:
            return self._result("error", "Not recording")
        try:
            if self.config.audio.beep_enabled:
                audio.beep_stop()
        except Exception as exc:
            self._last_error = str(exc)
            self._state = ClientState.IDLE
            client, self._stream_client = self._stream_client, None
            if client is not None:
                client.disconnect()
            return self._result("error", self._last_error)
        self._state = ClientState.SENDING
        generation = self._cancel_generation
        self._partial_text = ""
        self._worker = threading.Thread(
            target=self._send_and_inject,
            args=(generation,),
            daemon=True,
            name="RemoteTranscriptionWorker",
        )
        self._worker.start()
        self._update_tray_state()
        return self._result()

    def _send_and_inject(self, generation: int) -> None:
        """Wait for the stable final transcript and deliver it."""
        with self._lock:
            client = self._stream_client
            mode = self._recording_mode
        if client is None:
            return
        try:
            result = client.stop_stt()
            text = self._fix_context(result.text if result else "", mode)
            with self._lock:
                if generation != self._cancel_generation:
                    return
                if text:
                    entry = {
                        "id": self._next_history_id,
                        "text": text,
                        "timestamp": datetime.now().isoformat(),
                        "preview": (text[:50] + ("..." if len(text) > 50 else "")).replace(
                            "\\n", " "
                        ),
                    }
                    self._history.appendleft(entry)
                    self._next_history_id += 1
                    self._last_error = None
                    self._partial_text = ""
                    inject_to_clipboard(text)
                else:
                    self._last_error = "No text transcribed"
        except Exception as exc:
            with self._lock:
                if generation == self._cancel_generation:
                    self._last_error = str(exc)
            logger.exception("Remote transcription failed")
        finally:
            client.disconnect()
            with self._lock:
                if self._stream_client is client:
                    self._stream_client = None
                self._worker = None
                self._state = ClientState.IDLE
                self._partial_text = ""
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
        """Check if server is available via HTTP health endpoint."""
        try:
            url = f"{self.server_url.rstrip('/')}/health"
            request = Request(url, method="GET")
            response = urlopen(request, timeout=5.0)
            health = json.loads(response.read().decode("utf-8"))
            logger.info(
                f"Server available: model={health.get('model')}, device={health.get('device')}"
            )
            return True
        except URLError as e:
            logger.error(f"Server not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Error checking server: {e}")
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
            self._server = Server()
            self._server.register_handler("toggle", self._handle_toggle)
            self._server.register_handler("start", self._handle_start_recording)
            self._server.register_handler("stop_recording", self._handle_stop_recording)
            self._server.register_handler("cancel", self._handle_cancel)
            self._server.register_handler("status", self._handle_status)
            self._server.register_handler("history", self._handle_history)
            self._server.register_handler("copy_last", self._handle_copy_last)
            self._server.register_handler("set_context_mode", self._handle_set_context_mode)
            self._server.register_handler("update_replacements", self._handle_update_replacements)
            self._server.register_handler("stop", self._handle_shutdown)
            self._server.start()
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
        worker = self._worker
        if worker is not None and worker.is_alive():
            with self._lock:
                self._cancel_generation += 1
            worker.join()
        client, self._stream_client = self._stream_client, None
        if client is not None:
            client.disconnect()
        if self._tray is not None:
            self._tray.stop()
            self._tray = None
        if self._server is not None:
            self._server.stop()
            self._server = None
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
