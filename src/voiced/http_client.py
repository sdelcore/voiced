"""HTTP client for transcription requests."""

import io
import json
import logging
import wave
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

logger = logging.getLogger(__name__)


class ServerError(Exception):
    """Server returned an error."""

    def __init__(self, message: str, code: str = "UNKNOWN"):
        super().__init__(message)
        self.code = code


class HttpConnectionError(Exception):
    """Could not connect to server."""

    pass


class HttpTimeoutError(Exception):
    """Request timed out."""

    pass


def audio_to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert numpy audio array to WAV bytes.

    Args:
        audio: Audio data as numpy array (float32, mono).
        sample_rate: Sample rate of the audio.

    Returns:
        WAV file bytes.
    """
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(audio_int16.tobytes())

    return buffer.getvalue()


class TranscriptionClient:
    """HTTP client for remote transcription."""

    def __init__(self, server_url: str, timeout: float = 60.0):
        """Initialize the transcription client.

        Args:
            server_url: Base URL of the transcription server.
            timeout: Request timeout in seconds.
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str | None = None,
        identify_speakers: bool = True,
    ) -> dict:
        """Send audio to server for transcription.

        Args:
            audio: Audio data as numpy array (float32, mono).
            sample_rate: Sample rate of the audio.
            language: Optional language code to use.
            identify_speakers: Whether to identify speakers (default True).

        Returns:
            Full response dict with text, segments, speaker info.

        Raises:
            ServerError: If server returned an error.
            HttpConnectionError: If could not connect to server.
            HttpTimeoutError: If request timed out.
        """
        wav_bytes = audio_to_wav(audio, sample_rate)

        # Build query parameters
        params = {"identify_speakers": str(identify_speakers).lower()}
        if language:
            params["language"] = language

        url = f"{self.server_url}/transcribe?{urlencode(params)}"

        logger.info(f"Sending {len(wav_bytes)} bytes to {url}")

        req = Request(
            url,
            data=wav_bytes,
            headers={"Content-Type": "audio/wav"},
            method="POST",
        )

        try:
            response = urlopen(req, timeout=self.timeout)
            result = json.loads(response.read().decode("utf-8"))
            logger.info(
                f"Transcription completed: {result.get('processing_time', 0):.1f}s processing time"
            )
            return result

        except HTTPError as e:
            try:
                error_body = json.loads(e.read().decode("utf-8"))
                raise ServerError(
                    error_body.get("error", str(e)), error_body.get("code", "UNKNOWN")
                )
            except json.JSONDecodeError:
                raise ServerError(str(e), "HTTP_ERROR")

        except URLError as e:
            if "timed out" in str(e.reason).lower():
                raise HttpTimeoutError(f"Request timed out after {self.timeout}s")
            raise HttpConnectionError(f"Could not connect to server: {e.reason}")

        except HttpTimeoutError:
            raise HttpTimeoutError(f"Request timed out after {self.timeout}s")

    def transcribe_text_only(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> str:
        """Legacy method that returns just the text.

        Args:
            audio: Audio data as numpy array (float32, mono).
            sample_rate: Sample rate of the audio.
            language: Optional language code to use.

        Returns:
            Transcribed text string.
        """
        result = self.transcribe(audio, sample_rate, language, identify_speakers=False)
        return result.get("text", "")

    def list_profiles(self) -> list[dict]:
        """List all voice profiles.

        Returns:
            List of profile dicts with name, created_at, audio_duration.

        Raises:
            ServerError: If server returned an error.
            HttpConnectionError: If could not connect to server.
        """
        url = f"{self.server_url}/profiles"

        try:
            response = urlopen(url, timeout=self.timeout)
            return json.loads(response.read().decode("utf-8"))

        except HTTPError as e:
            try:
                error_body = json.loads(e.read().decode("utf-8"))
                raise ServerError(
                    error_body.get("error", str(e)), error_body.get("code", "UNKNOWN")
                )
            except json.JSONDecodeError:
                raise ServerError(str(e), "HTTP_ERROR")

        except URLError as e:
            raise HttpConnectionError(f"Could not connect to server: {e.reason}")

    def get_profile(self, name: str) -> dict | None:
        """Get a specific voice profile.

        Args:
            name: Name of the profile to retrieve.

        Returns:
            Profile dict or None if not found.

        Raises:
            ServerError: If server returned an error (other than 404).
            HttpConnectionError: If could not connect to server.
        """
        url = f"{self.server_url}/profiles/{name}"

        try:
            response = urlopen(url, timeout=self.timeout)
            return json.loads(response.read().decode("utf-8"))

        except HTTPError as e:
            if e.code == 404:
                return None
            try:
                error_body = json.loads(e.read().decode("utf-8"))
                raise ServerError(
                    error_body.get("error", str(e)), error_body.get("code", "UNKNOWN")
                )
            except json.JSONDecodeError:
                raise ServerError(str(e), "HTTP_ERROR")

        except URLError as e:
            raise HttpConnectionError(f"Could not connect to server: {e.reason}")

    def create_profile(self, name: str, audio: np.ndarray, sample_rate: int = 16000) -> dict:
        """Create a voice profile from audio.

        Args:
            name: Name for the new profile.
            audio: Audio data as numpy array (float32, mono).
            sample_rate: Sample rate of the audio.

        Returns:
            Created profile info.

        Raises:
            ServerError: If server returned an error.
            HttpConnectionError: If could not connect to server.
        """
        wav_bytes = audio_to_wav(audio, sample_rate)
        url = f"{self.server_url}/profiles/{name}"

        logger.info(f"Creating profile '{name}' with {len(wav_bytes)} bytes")

        req = Request(
            url,
            data=wav_bytes,
            headers={"Content-Type": "audio/wav"},
            method="POST",
        )

        try:
            response = urlopen(req, timeout=self.timeout)
            result = json.loads(response.read().decode("utf-8"))
            logger.info(f"Profile '{name}' created successfully")
            return result

        except HTTPError as e:
            try:
                error_body = json.loads(e.read().decode("utf-8"))
                raise ServerError(
                    error_body.get("error", str(e)), error_body.get("code", "UNKNOWN")
                )
            except json.JSONDecodeError:
                raise ServerError(str(e), "HTTP_ERROR")

        except URLError as e:
            raise HttpConnectionError(f"Could not connect to server: {e.reason}")

    def delete_profile(self, name: str) -> bool:
        """Delete a voice profile.

        Args:
            name: Name of the profile to delete.

        Returns:
            True if deleted, False if not found.

        Raises:
            ServerError: If server returned an error (other than 404).
            HttpConnectionError: If could not connect to server.
        """
        url = f"{self.server_url}/profiles/{name}"

        req = Request(url, method="DELETE")

        try:
            urlopen(req, timeout=self.timeout)
            logger.info(f"Profile '{name}' deleted successfully")
            return True

        except HTTPError as e:
            if e.code == 404:
                return False
            try:
                error_body = json.loads(e.read().decode("utf-8"))
                raise ServerError(
                    error_body.get("error", str(e)), error_body.get("code", "UNKNOWN")
                )
            except json.JSONDecodeError:
                raise ServerError(str(e), "HTTP_ERROR")

        except URLError as e:
            raise HttpConnectionError(f"Could not connect to server: {e.reason}")

    def health_check(self) -> dict:
        """Check server health.

        Returns:
            Health status dictionary with model, device, etc.

        Raises:
            HttpConnectionError: If could not connect to server.
        """
        url = f"{self.server_url}/health"

        try:
            response = urlopen(url, timeout=5.0)
            return json.loads(response.read().decode("utf-8"))
        except URLError as e:
            raise HttpConnectionError(f"Could not connect to server: {e.reason}")
        except Exception as e:
            raise HttpConnectionError(f"Health check failed: {e}")

    def get_status(self) -> dict:
        """Get detailed server status.

        Returns:
            Status dictionary with model, device, uptime, request count, etc.

        Raises:
            HttpConnectionError: If could not connect to server.
        """
        url = f"{self.server_url}/status"

        try:
            response = urlopen(url, timeout=5.0)
            return json.loads(response.read().decode("utf-8"))
        except URLError as e:
            raise HttpConnectionError(f"Could not connect to server: {e.reason}")
        except Exception as e:
            raise HttpConnectionError(f"Status check failed: {e}")

    def is_available(self) -> bool:
        """Check if server is available.

        Returns:
            True if server is reachable and healthy.
        """
        try:
            health = self.health_check()
            return health.get("status") == "healthy"
        except Exception:
            return False
