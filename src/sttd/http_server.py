"""HTTP server for transcription requests."""

import io
import json
import logging
import re
import tempfile
import threading
import time
import wave
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from sttd.config import Config, load_config
from sttd.diarizer import SpeakerIdentifier
from sttd.profiles import ProfileManager, VoiceProfile
from sttd.transcriber import Transcriber

logger = logging.getLogger(__name__)


def wav_to_audio(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Convert WAV bytes to numpy array and sample rate."""
    buffer = io.BytesIO(wav_bytes)
    with wave.open(buffer, "rb") as wav:
        sample_rate = wav.getframerate()
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        n_frames = wav.getnframes()
        audio_bytes = wav.readframes(n_frames)

        if sample_width == 2:
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio = audio_int16.astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio_float32 = np.frombuffer(audio_bytes, dtype=np.float32)
            audio = audio_float32
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)
            audio = np.mean(audio, axis=1)

    return audio, sample_rate


class TranscriptionHandler(BaseHTTPRequestHandler):
    """Handle transcription HTTP requests."""

    transcriber: Transcriber
    config: Config
    start_time: float
    request_count: int = 0
    protocol_version = "HTTP/1.1"

    # Lazy-loaded speaker identifier (class attribute for sharing across requests)
    _speaker_identifier: SpeakerIdentifier | None = None
    _profile_manager: ProfileManager | None = None

    def log_message(self, format: str, *args) -> None:
        logger.info("%s - %s", self.address_string(), format % args)

    def _send_json(self, status: int, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str, code: str) -> None:
        self._send_json(status, {"error": message, "code": code})

    def _get_profile_manager(self, profiles_path: str | None = None) -> ProfileManager:
        """Get or create profile manager."""
        if profiles_path:
            return ProfileManager(profiles_path)
        if TranscriptionHandler._profile_manager is None:
            TranscriptionHandler._profile_manager = ProfileManager()
        return TranscriptionHandler._profile_manager

    def _get_speaker_identifier(self) -> SpeakerIdentifier:
        """Get or create speaker identifier (lazy-loaded)."""
        if TranscriptionHandler._speaker_identifier is None:
            TranscriptionHandler._speaker_identifier = SpeakerIdentifier(
                config=self.config.diarization
            )
        return TranscriptionHandler._speaker_identifier

    def _parse_profile_name(self, path: str) -> str | None:
        """Parse profile name from /profiles/{name} path."""
        match = re.match(r"^/profiles/([^/]+)$", path)
        if match:
            return match.group(1)
        return None

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/health":
            self._handle_health()
        elif path == "/status":
            self._handle_status()
        elif path == "/profiles":
            self._handle_list_profiles()
        elif path.startswith("/profiles/"):
            name = self._parse_profile_name(path)
            if name:
                self._handle_get_profile(name)
            else:
                self._send_error_json(404, "Not found", "NOT_FOUND")
        else:
            self._send_error_json(404, "Not found", "NOT_FOUND")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/transcribe":
            self._handle_transcribe(parsed.query)
        elif path.startswith("/profiles/"):
            name = self._parse_profile_name(path)
            if name:
                self._handle_create_profile(name)
            else:
                self._send_error_json(404, "Not found", "NOT_FOUND")
        else:
            self._send_error_json(404, "Not found", "NOT_FOUND")

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/profiles/"):
            name = self._parse_profile_name(path)
            if name:
                self._handle_delete_profile(name)
            else:
                self._send_error_json(404, "Not found", "NOT_FOUND")
        else:
            self._send_error_json(404, "Not found", "NOT_FOUND")

    def _handle_health(self) -> None:
        device = self.transcriber._get_device()
        self._send_json(
            200,
            {
                "status": "healthy",
                "model": self.transcriber.config.model,
                "device": device,
            },
        )

    def _handle_status(self) -> None:
        device = self.transcriber._get_device()
        uptime = time.time() - self.start_time
        self._send_json(
            200,
            {
                "status": "ok",
                "state": "idle",
                "model": self.transcriber.config.model,
                "device": device,
                "language": self.transcriber.config.language,
                "request_count": TranscriptionHandler.request_count,
                "uptime_seconds": round(uptime, 1),
            },
        )

    def _handle_transcribe(self, query_string: str) -> None:
        content_length = int(self.headers.get("Content-Length", 0))

        if content_length == 0:
            self._send_error_json(400, "No audio data provided", "NO_AUDIO")
            return

        if content_length > 100 * 1024 * 1024:
            self._send_error_json(413, "Audio file too large (max 100MB)", "AUDIO_TOO_LARGE")
            return

        query_params = parse_qs(query_string)
        language = query_params.get("language", [None])[0]
        identify_speakers_param = query_params.get("identify_speakers", ["true"])[0]
        identify_speakers = identify_speakers_param.lower() == "true"
        profiles_path = query_params.get("profiles_path", [None])[0]

        wav_bytes = self.rfile.read(content_length)

        try:
            audio, sample_rate = wav_to_audio(wav_bytes)
        except Exception as e:
            logger.error(f"Failed to parse WAV: {e}")
            self._send_error_json(400, f"Invalid WAV format: {e}", "INVALID_AUDIO")
            return

        duration = len(audio) / sample_rate
        logger.info(f"Transcribing {duration:.1f}s of audio at {sample_rate}Hz")

        try:
            original_language = self.transcriber.config.language
            if language:
                self.transcriber.config.language = language

            start_time = time.time()

            # Use transcribe method that returns segments with info
            segments_raw, info = self._transcribe_with_info(audio, sample_rate)

            # Build full text from segments
            full_text = " ".join(text for _, _, text in segments_raw)

            # Apply speaker identification if enabled
            segments_output = []
            if identify_speakers and segments_raw:
                try:
                    profile_manager = self._get_profile_manager(profiles_path)
                    profiles = profile_manager.load_all()

                    if profiles:
                        identifier = self._get_speaker_identifier()
                        identified = identifier.identify_segments_from_array(
                            audio,
                            sample_rate,
                            segments_raw,
                            profiles,
                        )
                        for seg in identified:
                            segments_output.append({
                                "start": round(seg.start, 2),
                                "end": round(seg.end, 2),
                                "text": seg.text,
                                "speaker": seg.speaker,
                                "speaker_confidence": round(seg.confidence, 2),
                            })
                    else:
                        # No profiles, return segments without speaker info
                        for start, end, text in segments_raw:
                            segments_output.append({
                                "start": round(start, 2),
                                "end": round(end, 2),
                                "text": text,
                                "speaker": "Unknown",
                                "speaker_confidence": 0.0,
                            })
                except Exception as e:
                    logger.warning(f"Speaker identification failed: {e}")
                    # Fall back to segments without speaker info
                    for start, end, text in segments_raw:
                        segments_output.append({
                            "start": round(start, 2),
                            "end": round(end, 2),
                            "text": text,
                            "speaker": "Unknown",
                            "speaker_confidence": 0.0,
                        })
            else:
                # No speaker identification, return segments without speaker info
                for start, end, text in segments_raw:
                    segments_output.append({
                        "start": round(start, 2),
                        "end": round(end, 2),
                        "text": text,
                        "speaker": "Unknown",
                        "speaker_confidence": 0.0,
                    })

            elapsed = time.time() - start_time

            if language:
                self.transcriber.config.language = original_language

            TranscriptionHandler.request_count += 1

            self._send_json(
                200,
                {
                    "text": full_text,
                    "duration": round(duration, 2),
                    "language": info.language,
                    "language_probability": round(info.language_probability, 2),
                    "processing_time": round(elapsed, 2),
                    "model": self.transcriber.config.model,
                    "segments": segments_output,
                },
            )
        except Exception as e:
            logger.exception(f"Transcription failed: {e}")
            self._send_error_json(500, f"Transcription failed: {e}", "TRANSCRIPTION_ERROR")

    def _transcribe_with_info(
        self, audio: np.ndarray, sample_rate: int
    ) -> tuple[list[tuple[float, float, str]], any]:
        """Transcribe audio and return segments with info object."""
        audio = self.transcriber._normalize_audio(audio)
        logger.info(f"Transcribing audio with segments: {len(audio)} samples at {sample_rate}Hz")

        segments, info = self.transcriber.model.transcribe(
            audio,
            language=self.transcriber.config.language,
            beam_size=5,
            vad_filter=True,
        )

        logger.info(
            f"Detected language: {info.language} (probability: {info.language_probability:.2f})"
        )

        result = []
        for segment in segments:
            result.append((segment.start, segment.end, segment.text.strip()))

        return result, info

    def _handle_list_profiles(self) -> None:
        """Handle GET /profiles - list all profiles."""
        try:
            profile_manager = self._get_profile_manager()
            profiles = profile_manager.load_all()

            profiles_data = []
            for p in profiles:
                profiles_data.append({
                    "name": p.name,
                    "created_at": p.created_at,
                    "audio_duration": p.audio_duration,
                })

            self._send_json(200, {"profiles": profiles_data})
        except Exception as e:
            logger.exception(f"Failed to list profiles: {e}")
            self._send_error_json(500, f"Failed to list profiles: {e}", "PROFILE_ERROR")

    def _handle_get_profile(self, name: str) -> None:
        """Handle GET /profiles/{name} - get single profile."""
        try:
            profile_manager = self._get_profile_manager()
            profile = profile_manager.load(name)

            if profile is None:
                self._send_error_json(404, f"Profile '{name}' not found", "PROFILE_NOT_FOUND")
                return

            self._send_json(200, {
                "name": profile.name,
                "created_at": profile.created_at,
                "audio_duration": profile.audio_duration,
                "model_version": profile.model_version,
            })
        except Exception as e:
            logger.exception(f"Failed to get profile: {e}")
            self._send_error_json(500, f"Failed to get profile: {e}", "PROFILE_ERROR")

    def _handle_create_profile(self, name: str) -> None:
        """Handle POST /profiles/{name} - create profile from audio."""
        content_length = int(self.headers.get("Content-Length", 0))

        if content_length == 0:
            self._send_error_json(400, "No audio data provided", "NO_AUDIO")
            return

        if content_length > 50 * 1024 * 1024:
            self._send_error_json(413, "Audio file too large (max 50MB)", "AUDIO_TOO_LARGE")
            return

        wav_bytes = self.rfile.read(content_length)

        try:
            audio, sample_rate = wav_to_audio(wav_bytes)
        except Exception as e:
            logger.error(f"Failed to parse WAV: {e}")
            self._send_error_json(400, f"Invalid WAV format: {e}", "INVALID_AUDIO")
            return

        duration = len(audio) / sample_rate
        logger.info(f"Creating profile '{name}' from {duration:.1f}s of audio")

        try:
            from sttd.diarizer import SpeakerEmbedder

            # Get or create embedder
            identifier = self._get_speaker_identifier()
            embedder = identifier.embedder

            # Extract embedding from audio
            embedding = embedder.extract_embedding_from_array(audio, sample_rate)

            # Create profile
            profile = VoiceProfile(
                name=name,
                embedding=embedding.tolist(),
                created_at=datetime.utcnow().isoformat(),
                audio_duration=round(duration, 1),
                model_version=embedder.model_source,
            )

            # Save profile
            profile_manager = self._get_profile_manager()
            profile_manager.save(profile)

            self._send_json(201, {
                "status": "created",
                "name": name,
                "audio_duration": round(duration, 1),
                "model_version": embedder.model_source,
            })
        except Exception as e:
            logger.exception(f"Failed to create profile: {e}")
            self._send_error_json(500, f"Failed to create profile: {e}", "PROFILE_ERROR")

    def _handle_delete_profile(self, name: str) -> None:
        """Handle DELETE /profiles/{name} - delete profile."""
        try:
            profile_manager = self._get_profile_manager()

            if not profile_manager.exists(name):
                self._send_error_json(404, f"Profile '{name}' not found", "PROFILE_NOT_FOUND")
                return

            deleted = profile_manager.delete(name)
            if deleted:
                self._send_json(200, {"status": "deleted", "name": name})
            else:
                self._send_error_json(500, f"Failed to delete profile '{name}'", "DELETE_ERROR")
        except Exception as e:
            logger.exception(f"Failed to delete profile: {e}")
            self._send_error_json(500, f"Failed to delete profile: {e}", "PROFILE_ERROR")


class TranscriptionServer:
    """HTTP server wrapper for transcription service."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        config: Config | None = None,
    ):
        self.config = config or load_config()
        self.host = host or self.config.server.host
        self.port = port or self.config.server.port
        self.transcriber = Transcriber(self.config.transcription)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def _preload_model(self) -> None:
        logger.info("Pre-loading transcription model...")
        _ = self.transcriber.model
        logger.info("Model loaded successfully")

    def start(self, preload: bool = True) -> None:
        """Start the HTTP server."""
        if self._running:
            return

        if preload:
            self._preload_model()

        TranscriptionHandler.transcriber = self.transcriber
        TranscriptionHandler.config = self.config
        TranscriptionHandler.start_time = time.time()
        TranscriptionHandler.request_count = 0

        self._server = ThreadingHTTPServer(
            (self.host, self.port),
            TranscriptionHandler,
        )
        self._running = True

        logger.info(f"Starting HTTP server on {self.host}:{self.port}")
        self._server.serve_forever()

    def start_background(self, preload: bool = True) -> None:
        """Start the HTTP server in a background thread."""
        if self._running:
            return

        if preload:
            self._preload_model()

        TranscriptionHandler.transcriber = self.transcriber
        TranscriptionHandler.config = self.config
        TranscriptionHandler.start_time = time.time()
        TranscriptionHandler.request_count = 0

        self._server = ThreadingHTTPServer(
            (self.host, self.port),
            TranscriptionHandler,
        )
        self._running = True

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info(f"HTTP server started on {self.host}:{self.port}")

    def stop(self) -> None:
        """Stop the HTTP server."""
        if not self._running:
            return

        self._running = False
        if self._server:
            self._server.shutdown()
            self._server = None

        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

        # Clean up speaker identifier if loaded
        if TranscriptionHandler._speaker_identifier is not None:
            TranscriptionHandler._speaker_identifier.unload()
            TranscriptionHandler._speaker_identifier = None

        self.transcriber.unload()
        logger.info("HTTP server stopped")
