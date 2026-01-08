"""Unix domain socket client for IPC."""

import json
import logging
import socket
from pathlib import Path

from voiced.config import get_socket_path

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT = 5.0
RECV_TIMEOUT = 30.0
MAX_RESPONSE_SIZE = 65536


class ClientError(Exception):
    """Client communication error."""

    pass


class DaemonNotRunning(ClientError):
    """Daemon is not running."""

    pass


def send_command(
    command: str,
    args: dict | None = None,
    socket_path: Path | None = None,
) -> dict:
    """Send a command to the daemon and return the response.

    Args:
        command: Command name.
        args: Optional command arguments.
        socket_path: Path to the Unix socket. Uses default if not provided.

    Returns:
        Response dictionary from the daemon.

    Raises:
        DaemonNotRunning: If the daemon is not running.
        ClientError: If there's a communication error.
    """
    socket_path = socket_path or get_socket_path()

    if not socket_path.exists():
        raise DaemonNotRunning("Daemon is not running (socket does not exist)")

    request = {"command": command}
    if args:
        request["args"] = args

    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(CONNECT_TIMEOUT)

        try:
            sock.connect(str(socket_path))
        except (ConnectionRefusedError, FileNotFoundError):
            raise DaemonNotRunning("Daemon is not running (connection refused)")

        # Send request
        sock.sendall(json.dumps(request).encode("utf-8"))

        # Receive response
        sock.settimeout(RECV_TIMEOUT)
        data = sock.recv(MAX_RESPONSE_SIZE)

        if not data:
            raise ClientError("Empty response from daemon")

        return json.loads(data.decode("utf-8"))

    except json.JSONDecodeError as e:
        raise ClientError(f"Invalid response from daemon: {e}")
    except TimeoutError:
        raise ClientError("Timeout waiting for daemon response")
    except Exception as e:
        raise ClientError(f"Communication error: {e}")
    finally:
        sock.close()


def is_daemon_running(socket_path: Path | None = None) -> bool:
    """Check if the daemon is running.

    Args:
        socket_path: Path to the Unix socket. Uses default if not provided.

    Returns:
        True if daemon is running, False otherwise.
    """
    try:
        response = send_command("status", socket_path=socket_path)
        return response.get("status") == "ok"
    except DaemonNotRunning:
        return False
    except ClientError:
        return False


def toggle_recording(socket_path: Path | None = None) -> dict:
    """Toggle recording on the daemon.

    Args:
        socket_path: Path to the Unix socket. Uses default if not provided.

    Returns:
        Response from the daemon.
    """
    return send_command("toggle", socket_path=socket_path)


def get_status(socket_path: Path | None = None) -> dict:
    """Get daemon status.

    Args:
        socket_path: Path to the Unix socket. Uses default if not provided.

    Returns:
        Status response from the daemon.
    """
    return send_command("status", socket_path=socket_path)


def stop_daemon(socket_path: Path | None = None) -> dict:
    """Stop the daemon.

    Args:
        socket_path: Path to the Unix socket. Uses default if not provided.

    Returns:
        Response from the daemon.
    """
    return send_command("stop", socket_path=socket_path)
