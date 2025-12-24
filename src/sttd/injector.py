"""Text injection using clipboard."""

import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)


def is_clipboard_available() -> bool:
    """Check if wl-copy is available."""
    return shutil.which("wl-copy") is not None


def inject_to_clipboard(text: str) -> bool:
    """Copy text to clipboard using wl-copy.

    Args:
        text: Text to copy to clipboard.

    Returns:
        True if successful, False otherwise.
    """
    if not text:
        return True

    if not is_clipboard_available():
        logger.error("wl-copy is not available")
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
