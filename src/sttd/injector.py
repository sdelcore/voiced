"""Text injection using wtype and clipboard."""

import logging
import shutil
import subprocess
from enum import Enum

logger = logging.getLogger(__name__)


class InjectionMethod(Enum):
    """Text injection method."""

    WTYPE = "wtype"
    CLIPBOARD = "clipboard"
    BOTH = "both"


def is_wtype_available() -> bool:
    """Check if wtype is available."""
    return shutil.which("wtype") is not None


def is_clipboard_available() -> bool:
    """Check if wl-copy and wl-paste are available."""
    return shutil.which("wl-copy") is not None and shutil.which("wl-paste") is not None


def inject_with_wtype(text: str) -> bool:
    """Inject text using wtype.

    Args:
        text: Text to inject.

    Returns:
        True if successful, False otherwise.
    """
    if not text:
        return True

    if not is_wtype_available():
        logger.error("wtype is not available")
        return False

    try:
        result = subprocess.run(
            ["wtype", "--", text],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            logger.error(f"wtype failed: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("wtype timed out")
        return False
    except Exception as e:
        logger.error(f"wtype error: {e}")
        return False


def inject_with_clipboard(text: str) -> bool:
    """Inject text using clipboard (wl-copy + wl-paste).

    Args:
        text: Text to inject.

    Returns:
        True if successful, False otherwise.
    """
    if not text:
        return True

    if not is_clipboard_available():
        return False

    try:
        result = subprocess.run(
            ["wl-copy", "--", text],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            logger.error(f"wl-copy failed: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("Clipboard operation timed out")
        return False
    except Exception as e:
        logger.error(f"Clipboard error: {e}")
        return False


def inject_backspaces(count: int) -> bool:
    """Inject backspace keys to delete previous text.

    Args:
        count: Number of backspaces to inject.

    Returns:
        True if successful, False otherwise.
    """
    if count <= 0:
        return True

    if not is_wtype_available():
        logger.error("wtype is not available for backspace injection")
        return False

    try:
        # Build command with repeated -k BackSpace arguments
        cmd = ["wtype"]
        for _ in range(count):
            cmd.extend(["-k", "BackSpace"])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,  # Longer timeout for many backspaces
        )
        if result.returncode != 0:
            logger.error(f"wtype backspace failed: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("wtype backspace timed out")
        return False
    except Exception as e:
        logger.error(f"wtype backspace error: {e}")
        return False


def inject_text(text: str, method: str = "wtype") -> bool:
    """Inject text using the specified method.

    Args:
        text: Text to inject.
        method: Injection method ("wtype", "clipboard", or "both").

    Returns:
        True if successful, False otherwise.
    """
    if not text:
        logger.warning("No text to inject")
        return True

    method = method.lower()

    if method == "wtype":
        success = inject_with_wtype(text)
        if not success:
            logger.info("Falling back to clipboard")
            return inject_with_clipboard(text)
        return success

    elif method == "clipboard":
        return inject_with_clipboard(text)

    elif method == "both":
        # Try wtype first, then also copy to clipboard
        wtype_success = inject_with_wtype(text)
        clipboard_success = inject_with_clipboard(text)
        return wtype_success or clipboard_success

    else:
        logger.error(f"Unknown injection method: {method}")
        return False


def get_available_methods() -> list[str]:
    """Get a list of available injection methods.

    Returns:
        List of available method names.
    """
    methods = []
    if is_wtype_available():
        methods.append("wtype")
    if is_clipboard_available():
        methods.append("clipboard")
    return methods
