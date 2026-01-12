"""HTTP server for transcription and TTS requests."""

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
from urllib.parse import parse_qs, urlparse

import numpy as np

from voiced.config import Config, load_config
from voiced.diarizer import SpeakerIdentifier
from voiced.profiles import ProfileManager, VoiceProfile
from voiced.transcriber import Transcriber
from voiced.webrtc_server import WebRTCConnectionManager

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
    """Handle transcription and TTS HTTP requests."""

    transcriber: Transcriber
    config: Config
    start_time: float
    request_count: int = 0
    tts_request_count: int = 0
    protocol_version = "HTTP/1.1"

    # Lazy-loaded speaker identifier (class attribute for sharing across requests)
    _speaker_identifier: SpeakerIdentifier | None = None
    _profile_manager: ProfileManager | None = None

    # TTS synthesizer (lazy-loaded, shared across requests)
    _synthesizer = None

    # WebRTC connection manager
    _webrtc_manager: WebRTCConnectionManager | None = None
    _asyncio_loop = None

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
        elif path == "/voices":
            self._handle_list_voices()
        elif path.startswith("/voices/"):
            name = self._parse_voice_name(path)
            if name:
                self._handle_get_voice(name)
            else:
                self._send_error_json(404, "Not found", "NOT_FOUND")
        else:
            self._send_error_json(404, "Not found", "NOT_FOUND")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/transcribe":
            self._handle_transcribe(parsed.query)
        elif path == "/synthesize":
            self._handle_synthesize(parsed.query)
        elif path.startswith("/profiles/"):
            name = self._parse_profile_name(path)
            if name:
                self._handle_create_profile(name)
            else:
                self._send_error_json(404, "Not found", "NOT_FOUND")
        elif path.startswith("/voices/") and path.endswith("/download"):
            name = self._parse_voice_download_name(path)
            if name:
                self._handle_download_voice(name)
            else:
                self._send_error_json(404, "Not found", "NOT_FOUND")
        elif path == "/webrtc/offer":
            self._handle_webrtc_offer()
        elif path == "/webrtc/ice":
            self._handle_webrtc_ice()
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

        status_data = {
            "status": "ok",
            "state": "idle",
            "model": self.transcriber.config.model,
            "device": device,
            "language": self.transcriber.config.language,
            "request_count": TranscriptionHandler.request_count,
            "uptime_seconds": round(uptime, 1),
            "tts": {
                "enabled": self.config.tts.enabled,
                "model_loaded": (
                    TranscriptionHandler._synthesizer is not None
                    and TranscriptionHandler._synthesizer.is_loaded
                ),
                "request_count": TranscriptionHandler.tts_request_count,
                "default_voice": self.config.tts.default_voice,
            },
        }

        self._send_json(200, status_data)

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

        audio_bytes = self.rfile.read(content_length)

        # Use FFmpeg to convert any audio format to WAV, then read with soundfile
        import os
        import subprocess

        import soundfile as sf

        # Determine file extension from Content-Type header
        content_type = self.headers.get("Content-Type", "audio/wav")
        ext_map = {
            "audio/wav": ".wav",
            "audio/webm": ".webm",
            "audio/mpeg": ".mp3",
            "audio/mp3": ".mp3",
            "audio/ogg": ".ogg",
            "audio/flac": ".flac",
            "audio/mp4": ".m4a",
            "audio/x-m4a": ".m4a",
        }
        suffix = ext_map.get(content_type, ".wav")

        temp_path = None
        wav_path = None
        try:
            # Write input audio to temp file
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(audio_bytes)
                temp_path = f.name

            # For non-WAV formats, convert to WAV using FFmpeg
            if suffix != ".wav":
                wav_path = temp_path.rsplit(".", 1)[0] + ".wav"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        temp_path,
                        "-ar",
                        "16000",  # Resample to 16kHz (optimal for Whisper)
                        "-ac",
                        "1",  # Convert to mono
                        "-f",
                        "wav",
                        wav_path,
                    ],
                    capture_output=True,
                    check=True,
                )
                audio_path = wav_path
            else:
                audio_path = temp_path

            audio, sample_rate = sf.read(audio_path, dtype="float32")

            # Convert stereo to mono if needed (for WAV input that wasn't converted)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
            self._send_error_json(
                400, f"Failed to convert audio: {e.stderr.decode()}", "INVALID_AUDIO"
            )
            return
        except Exception as e:
            logger.error(f"Failed to read audio: {e}")
            self._send_error_json(400, f"Failed to read audio file: {e}", "INVALID_AUDIO")
            return
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)

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
                            segments_output.append(
                                {
                                    "start": round(seg.start, 2),
                                    "end": round(seg.end, 2),
                                    "text": seg.text,
                                    "speaker": seg.speaker,
                                    "speaker_confidence": round(seg.confidence, 2),
                                }
                            )
                    else:
                        # No profiles, return segments without speaker info
                        for start, end, text in segments_raw:
                            segments_output.append(
                                {
                                    "start": round(start, 2),
                                    "end": round(end, 2),
                                    "text": text,
                                    "speaker": "Unknown",
                                    "speaker_confidence": 0.0,
                                }
                            )
                except Exception as e:
                    logger.warning(f"Speaker identification failed: {e}")
                    # Fall back to segments without speaker info
                    for start, end, text in segments_raw:
                        segments_output.append(
                            {
                                "start": round(start, 2),
                                "end": round(end, 2),
                                "text": text,
                                "speaker": "Unknown",
                                "speaker_confidence": 0.0,
                            }
                        )
            else:
                # No speaker identification, return segments without speaker info
                for start, end, text in segments_raw:
                    segments_output.append(
                        {
                            "start": round(start, 2),
                            "end": round(end, 2),
                            "text": text,
                            "speaker": "Unknown",
                            "speaker_confidence": 0.0,
                        }
                    )

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
            **self.transcriber._get_vad_kwargs(),
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
                profiles_data.append(
                    {
                        "name": p.name,
                        "created_at": p.created_at,
                        "audio_duration": p.audio_duration,
                    }
                )

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

            self._send_json(
                200,
                {
                    "name": profile.name,
                    "created_at": profile.created_at,
                    "audio_duration": profile.audio_duration,
                    "model_version": profile.model_version,
                },
            )
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

            self._send_json(
                201,
                {
                    "status": "created",
                    "name": name,
                    "audio_duration": round(duration, 1),
                    "model_version": embedder.model_source,
                },
            )
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

    # =========================================================================
    # TTS Endpoints
    # =========================================================================

    def _parse_voice_name(self, path: str) -> str | None:
        """Parse voice name from /voices/{name} path."""
        match = re.match(r"^/voices/([^/]+)$", path)
        if match:
            return match.group(1)
        return None

    def _parse_voice_download_name(self, path: str) -> str | None:
        """Parse voice name from /voices/{name}/download path."""
        match = re.match(r"^/voices/([^/]+)/download$", path)
        if match:
            return match.group(1)
        return None

    def _get_synthesizer(self):
        """Get or create TTS synthesizer (lazy-loaded)."""
        if TranscriptionHandler._synthesizer is None:
            if not self.config.tts.enabled:
                return None

            from voiced.synthesizer import Synthesizer, TTSConfig, check_vibevoice_installed

            if not check_vibevoice_installed():
                logger.warning("VibeVoice is not installed, TTS endpoints will be unavailable")
                return None

            tts_config = TTSConfig(
                model_path=self.config.tts.model,
                device=self.config.tts.device,
                default_voice=self.config.tts.default_voice,
                cfg_scale=self.config.tts.cfg_scale,
                unload_timeout_seconds=self.config.tts.unload_timeout_minutes * 60,
            )
            TranscriptionHandler._synthesizer = Synthesizer(tts_config)

        return TranscriptionHandler._synthesizer

    def _handle_list_voices(self) -> None:
        """Handle GET /voices - list available voice presets."""
        try:
            from voiced.voice_manager import VoiceManager

            vm = VoiceManager()
            available = vm.list_available()
            downloaded = set(vm.list_downloaded())

            voices_data = []
            for name in available:
                info = vm.get_voice_info(name)
                voices_data.append(
                    {
                        "name": name,
                        "downloaded": name in downloaded,
                        "filename": info.get("filename"),
                        "size_bytes": info.get("size_bytes") if info.get("downloaded") else None,
                    }
                )

            self._send_json(200, {"voices": voices_data})
        except Exception as e:
            logger.exception(f"Failed to list voices: {e}")
            self._send_error_json(500, f"Failed to list voices: {e}", "VOICE_ERROR")

    def _handle_get_voice(self, name: str) -> None:
        """Handle GET /voices/{name} - get voice info."""
        try:
            from voiced.voice_manager import VoiceManager

            vm = VoiceManager()

            try:
                info = vm.get_voice_info(name)
            except ValueError as e:
                self._send_error_json(404, str(e), "VOICE_NOT_FOUND")
                return

            self._send_json(200, info)
        except Exception as e:
            logger.exception(f"Failed to get voice info: {e}")
            self._send_error_json(500, f"Failed to get voice info: {e}", "VOICE_ERROR")

    def _handle_download_voice(self, name: str) -> None:
        """Handle POST /voices/{name}/download - download voice preset."""
        try:
            from voiced.voice_manager import VoiceManager

            vm = VoiceManager()

            logger.info(f"Downloading voice preset: {name}")
            path = vm.download(name, force=False)

            info = vm.get_voice_info(name)
            self._send_json(
                200,
                {
                    "status": "downloaded",
                    "name": name,
                    "path": str(path),
                    "size_bytes": info.get("size_bytes"),
                },
            )
        except ValueError as e:
            self._send_error_json(404, str(e), "VOICE_NOT_FOUND")
        except Exception as e:
            logger.exception(f"Failed to download voice: {e}")
            self._send_error_json(500, f"Failed to download voice: {e}", "VOICE_ERROR")

    def _handle_synthesize(self, query_string: str) -> None:
        """Handle POST /synthesize - synthesize speech from text."""
        # Check if TTS is enabled
        synthesizer = self._get_synthesizer()
        if synthesizer is None:
            self._send_error_json(
                503,
                "TTS is not available (VibeVoice not installed or TTS disabled)",
                "TTS_UNAVAILABLE",
            )
            return

        # Parse request body
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_error_json(400, "No request body", "NO_BODY")
            return

        if content_length > 1024 * 1024:  # 1MB max for text
            self._send_error_json(413, "Request body too large (max 1MB)", "BODY_TOO_LARGE")
            return

        try:
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as e:
            self._send_error_json(400, f"Invalid JSON: {e}", "INVALID_JSON")
            return

        text = data.get("text", "").strip()
        if not text:
            self._send_error_json(400, "No text provided", "NO_TEXT")
            return

        if len(text) > 10000:
            self._send_error_json(400, "Text too long (max 10000 chars)", "TEXT_TOO_LONG")
            return

        voice = data.get("voice") or self.config.tts.default_voice
        cfg_scale = data.get("cfg_scale") or self.config.tts.cfg_scale

        logger.info(f"Synthesizing {len(text)} chars with voice '{voice}'")

        try:
            start_time = time.time()
            audio = synthesizer.synthesize(text, voice=voice, cfg_scale=cfg_scale)
            elapsed = time.time() - start_time

            # Convert to WAV bytes
            wav_bytes = self._audio_to_wav(audio, synthesizer.sample_rate)

            duration = len(audio) / synthesizer.sample_rate
            TranscriptionHandler.tts_request_count += 1

            # Send WAV response
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav_bytes)))
            self.send_header("X-Audio-Duration", str(round(duration, 2)))
            self.send_header("X-Processing-Time", str(round(elapsed, 2)))
            self.send_header("X-Voice", voice)
            self.end_headers()
            self.wfile.write(wav_bytes)

        except Exception as e:
            logger.exception(f"TTS synthesis failed: {e}")
            self._send_error_json(500, f"Synthesis failed: {e}", "SYNTHESIS_ERROR")

    def _handle_webrtc_offer(self) -> None:
        """Handle WebRTC offer for connection establishment."""
        import asyncio

        if TranscriptionHandler._webrtc_manager is None:
            self._send_error_json(503, "WebRTC not initialized", "WEBRTC_NOT_INITIALIZED")
            return

        if TranscriptionHandler._asyncio_loop is None:
            self._send_error_json(503, "WebRTC event loop not running", "WEBRTC_NOT_READY")
            return

        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_error_json(400, "No offer provided", "NO_OFFER")
            return

        if content_length > 64 * 1024:  # 64KB limit for SDP
            self._send_error_json(413, "Offer too large", "OFFER_TOO_LARGE")
            return

        try:
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8"))
            offer_sdp = data.get("sdp")

            if not offer_sdp:
                self._send_error_json(400, "No SDP in offer", "NO_SDP")
                return

            # Run async operation in the event loop
            future = asyncio.run_coroutine_threadsafe(
                TranscriptionHandler._webrtc_manager.create_session(offer_sdp),
                TranscriptionHandler._asyncio_loop,
            )
            session_id, answer_sdp, ice_candidates = future.result(timeout=10.0)

            self._send_json(200, {
                "session_id": session_id,
                "sdp": answer_sdp,
                "type": "answer",
                "ice_candidates": ice_candidates,
            })

        except json.JSONDecodeError:
            self._send_error_json(400, "Invalid JSON", "INVALID_JSON")
        except TimeoutError:
            self._send_error_json(504, "Connection timeout", "CONNECTION_TIMEOUT")
        except Exception as e:
            logger.exception(f"WebRTC offer handling failed: {e}")
            self._send_error_json(500, f"Failed to create session: {e}", "SESSION_ERROR")

    def _handle_webrtc_ice(self) -> None:
        """Handle ICE candidate from client."""
        import asyncio

        if TranscriptionHandler._webrtc_manager is None:
            self._send_error_json(503, "WebRTC not initialized", "WEBRTC_NOT_INITIALIZED")
            return

        if TranscriptionHandler._asyncio_loop is None:
            self._send_error_json(503, "WebRTC event loop not running", "WEBRTC_NOT_READY")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_error_json(400, "No ICE candidate provided", "NO_ICE")
            return

        try:
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8"))

            session_id = data.get("session_id")
            candidate = data.get("candidate")

            if not session_id:
                self._send_error_json(400, "No session_id provided", "NO_SESSION_ID")
                return

            if not candidate:
                self._send_error_json(400, "No candidate provided", "NO_CANDIDATE")
                return

            # Run async operation in the event loop
            future = asyncio.run_coroutine_threadsafe(
                TranscriptionHandler._webrtc_manager.add_ice_candidate(session_id, candidate),
                TranscriptionHandler._asyncio_loop,
            )
            success = future.result(timeout=5.0)

            if success:
                self._send_json(200, {"status": "ok"})
            else:
                self._send_error_json(404, "Session not found", "SESSION_NOT_FOUND")

        except json.JSONDecodeError:
            self._send_error_json(400, "Invalid JSON", "INVALID_JSON")
        except TimeoutError:
            self._send_error_json(504, "ICE handling timeout", "ICE_TIMEOUT")
        except Exception as e:
            logger.exception(f"ICE candidate handling failed: {e}")
            self._send_error_json(500, f"Failed to add ICE candidate: {e}", "ICE_ERROR")

    def _audio_to_wav(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Convert numpy audio array to WAV bytes."""
        buffer = io.BytesIO()

        # Convert float32 to int16
        audio_int16 = (audio * 32767).astype(np.int16)

        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(sample_rate)
            wav.writeframes(audio_int16.tobytes())

        return buffer.getvalue()


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
        self.transcriber = Transcriber(self.config.transcription, vad_config=self.config.vad)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._asyncio_thread: threading.Thread | None = None
        self._asyncio_loop = None

    def _preload_model(self) -> None:
        logger.info("Pre-loading transcription model...")
        _ = self.transcriber.model
        logger.info("Model loaded successfully")

    def _start_asyncio_loop(self) -> None:
        """Start asyncio event loop in a separate thread."""
        import asyncio

        self._asyncio_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._asyncio_loop)
        self._asyncio_loop.run_forever()

    def _init_webrtc(self) -> None:
        """Initialize WebRTC connection manager."""
        # Start asyncio event loop in background thread
        self._asyncio_thread = threading.Thread(
            target=self._start_asyncio_loop, daemon=True, name="asyncio-webrtc"
        )
        self._asyncio_thread.start()

        # Wait for event loop to be ready
        while self._asyncio_loop is None:
            time.sleep(0.01)

        # Create WebRTC connection manager
        webrtc_manager = WebRTCConnectionManager(
            transcriber=self.transcriber,
            synthesizer=TranscriptionHandler._synthesizer,
            speaker_identifier=TranscriptionHandler._speaker_identifier,
        )
        webrtc_manager.set_event_loop(self._asyncio_loop)

        TranscriptionHandler._webrtc_manager = webrtc_manager
        TranscriptionHandler._asyncio_loop = self._asyncio_loop

        logger.info("WebRTC enabled")

    def _stop_webrtc(self) -> None:
        """Stop WebRTC and asyncio event loop."""
        import asyncio

        if TranscriptionHandler._webrtc_manager is not None:
            # Close all WebRTC sessions
            if self._asyncio_loop and self._asyncio_loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    TranscriptionHandler._webrtc_manager.close_all(),
                    self._asyncio_loop,
                )
                try:
                    future.result(timeout=5.0)
                except Exception as e:
                    logger.warning(f"Error closing WebRTC sessions: {e}")

            TranscriptionHandler._webrtc_manager = None

        # Stop asyncio event loop
        if self._asyncio_loop:
            self._asyncio_loop.call_soon_threadsafe(self._asyncio_loop.stop)

        if self._asyncio_thread:
            self._asyncio_thread.join(timeout=5)
            self._asyncio_thread = None

        TranscriptionHandler._asyncio_loop = None
        self._asyncio_loop = None

        logger.info("WebRTC stopped")

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

        # Initialize WebRTC
        self._init_webrtc()

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

        # Initialize WebRTC
        self._init_webrtc()

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

        # Stop WebRTC first
        self._stop_webrtc()

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

        # Clean up TTS synthesizer if loaded
        if TranscriptionHandler._synthesizer is not None:
            TranscriptionHandler._synthesizer.shutdown()
            TranscriptionHandler._synthesizer = None

        self.transcriber.unload()
        logger.info("HTTP server stopped")
