"""Unix domain socket server for IPC."""

import json
import logging
import os
import selectors
import socket
import threading
from collections.abc import Callable
from pathlib import Path

from sttd.config import get_socket_path

logger = logging.getLogger(__name__)

# Maximum message size (64KB)
MAX_MESSAGE_SIZE = 65536


class Server:
    """Unix domain socket server for daemon IPC."""

    def __init__(self, socket_path: Path | None = None):
        """Initialize the server.

        Args:
            socket_path: Path to the Unix socket. Uses default if not provided.
        """
        self.socket_path = socket_path or get_socket_path()
        self._socket: socket.socket | None = None
        self._selector = selectors.DefaultSelector()
        self._running = False
        self._handlers: dict[str, Callable] = {}
        self._server_thread: threading.Thread | None = None

    def register_handler(self, command: str, handler: Callable) -> None:
        """Register a handler for a command.

        Args:
            command: Command name.
            handler: Handler function that takes command args and returns a response dict.
        """
        self._handlers[command] = handler

    def _handle_client(self, conn: socket.socket) -> None:
        """Handle a client connection."""
        try:
            data = conn.recv(MAX_MESSAGE_SIZE)
            if not data:
                return

            try:
                request = json.loads(data.decode("utf-8"))
            except json.JSONDecodeError as e:
                response = {"status": "error", "message": f"Invalid JSON: {e}"}
                conn.sendall(json.dumps(response).encode("utf-8"))
                return

            command = request.get("command")
            args = request.get("args", {})

            if not command:
                response = {"status": "error", "message": "No command specified"}
            elif command not in self._handlers:
                response = {"status": "error", "message": f"Unknown command: {command}"}
            else:
                try:
                    response = self._handlers[command](args)
                except Exception as e:
                    logger.exception(f"Handler error for {command}")
                    response = {"status": "error", "message": str(e)}

            conn.sendall(json.dumps(response).encode("utf-8"))

        except Exception:
            logger.exception("Error handling client")
        finally:
            conn.close()

    def _accept_connection(self, sock: socket.socket) -> None:
        """Accept a new connection."""
        conn, _ = sock.accept()
        conn.setblocking(False)
        self._selector.register(conn, selectors.EVENT_READ, self._handle_client)

    def _run(self) -> None:
        """Main server loop."""
        while self._running:
            try:
                events = self._selector.select(timeout=0.5)
                for key, mask in events:
                    if key.data is None:
                        # This is the server socket
                        self._accept_connection(key.fileobj)
                    else:
                        # This is a client connection
                        key.data(key.fileobj)
                        self._selector.unregister(key.fileobj)
            except Exception:
                if self._running:
                    logger.exception("Server loop error")

    def start(self) -> None:
        """Start the server."""
        if self._running:
            logger.warning("Server already running")
            return

        # Remove existing socket file
        if self.socket_path.exists():
            self.socket_path.unlink()

        # Create socket directory
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)

        # Create and bind socket
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.setblocking(False)
        self._socket.bind(str(self.socket_path))
        self._socket.listen(5)

        # Set socket permissions (owner only)
        os.chmod(self.socket_path, 0o600)

        self._selector.register(self._socket, selectors.EVENT_READ, None)
        self._running = True

        self._server_thread = threading.Thread(target=self._run, daemon=True)
        self._server_thread.start()

        logger.info(f"Server started on {self.socket_path}")

    def stop(self) -> None:
        """Stop the server."""
        if not self._running:
            return

        self._running = False

        if self._server_thread is not None:
            self._server_thread.join(timeout=2.0)
            self._server_thread = None

        if self._socket is not None:
            self._selector.unregister(self._socket)
            self._socket.close()
            self._socket = None

        self._selector.close()

        if self.socket_path.exists():
            self.socket_path.unlink()

        logger.info("Server stopped")

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running
