"""WebRTC server for real-time audio streaming."""

import asyncio
import base64
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from av import AudioFrame
from scipy import signal

logger = logging.getLogger(__name__)

# Sample rates
WEBRTC_SAMPLE_RATE = 48000
WHISPER_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000

# Audio frame settings
FRAME_SAMPLES = 960  # 20ms at 48kHz
CHANNELS = 1


def resample_audio(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample audio from one sample rate to another."""
    if from_rate == to_rate:
        return audio
    num_samples = int(len(audio) * to_rate / from_rate)
    return signal.resample(audio, num_samples).astype(np.float32)


@dataclass
class STTConfig:
    """Configuration for STT session."""

    language: str | None = None
    identify_speakers: bool = False
    profiles_path: str | None = None


@dataclass
class TTSConfig:
    """Configuration for TTS session."""

    voice: str | None = None
    cfg_scale: float | None = None


class AudioBufferTrack(MediaStreamTrack):
    """Audio track that buffers incoming audio for STT."""

    kind = "audio"

    def __init__(self):
        super().__init__()
        self._buffer: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._recording = False
        self._total_samples = 0

    def start_recording(self) -> None:
        """Start buffering audio."""
        with self._lock:
            self._buffer = []
            self._total_samples = 0
            self._recording = True
        logger.info("STT recording started")

    def stop_recording(self) -> np.ndarray:
        """Stop buffering and return accumulated audio at 16kHz."""
        with self._lock:
            self._recording = False
            if not self._buffer:
                return np.array([], dtype=np.float32)
            audio_48k = np.concatenate(self._buffer)
            self._buffer = []
        logger.info(f"STT recording stopped: {len(audio_48k)} samples at 48kHz")
        return resample_audio(audio_48k, WEBRTC_SAMPLE_RATE, WHISPER_SAMPLE_RATE)

    def get_buffer_duration(self) -> float:
        """Get current buffer duration in seconds."""
        with self._lock:
            return self._total_samples / WEBRTC_SAMPLE_RATE

    def get_partial_audio(self) -> np.ndarray:
        """Get current buffer for partial transcription (non-destructive)."""
        with self._lock:
            if not self._buffer:
                return np.array([], dtype=np.float32)
            audio_48k = np.concatenate(self._buffer)
        return resample_audio(audio_48k, WEBRTC_SAMPLE_RATE, WHISPER_SAMPLE_RATE)

    async def recv(self) -> AudioFrame:
        """Receive is not used for input track."""
        raise NotImplementedError("AudioBufferTrack is for receiving only")

    def on_frame(self, frame: AudioFrame) -> None:
        """Process incoming audio frame."""
        if not self._recording:
            return

        # Convert frame to numpy array
        audio = frame.to_ndarray()
        if audio.ndim > 1:
            audio = audio.mean(axis=0)  # Convert to mono
        audio = audio.astype(np.float32)

        # Normalize if needed
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / 32768.0

        with self._lock:
            self._buffer.append(audio)
            self._total_samples += len(audio)


class TTSOutputTrack(MediaStreamTrack):
    """Audio track that streams TTS output."""

    kind = "audio"

    def __init__(self):
        super().__init__()
        self._queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        self._buffer = np.array([], dtype=np.float32)
        self._pts = 0
        self._streaming = False

    async def start_streaming(self) -> None:
        """Start streaming TTS audio."""
        self._streaming = True
        self._buffer = np.array([], dtype=np.float32)
        self._pts = 0
        logger.info("TTS streaming started")

    async def stop_streaming(self) -> None:
        """Stop streaming TTS audio."""
        self._streaming = False
        await self._queue.put(None)
        logger.info("TTS streaming stopped")

    async def push_audio(self, audio_24k: np.ndarray) -> None:
        """Push TTS audio chunk (24kHz) to be streamed."""
        audio_48k = resample_audio(audio_24k, TTS_SAMPLE_RATE, WEBRTC_SAMPLE_RATE)
        await self._queue.put(audio_48k)

    async def recv(self) -> AudioFrame:
        """Generate audio frames for WebRTC."""
        # Fill buffer until we have enough for a frame
        while len(self._buffer) < FRAME_SAMPLES:
            if not self._streaming and self._queue.empty():
                # Generate silence when not streaming
                self._buffer = np.concatenate([
                    self._buffer,
                    np.zeros(FRAME_SAMPLES, dtype=np.float32)
                ])
                break

            try:
                chunk = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                if chunk is None:
                    # End of stream, pad with silence
                    self._buffer = np.concatenate([
                        self._buffer,
                        np.zeros(FRAME_SAMPLES - len(self._buffer), dtype=np.float32)
                    ])
                    break
                self._buffer = np.concatenate([self._buffer, chunk])
            except asyncio.TimeoutError:
                # No data, generate silence
                self._buffer = np.concatenate([
                    self._buffer,
                    np.zeros(FRAME_SAMPLES, dtype=np.float32)
                ])
                break

        # Extract frame from buffer
        frame_data = self._buffer[:FRAME_SAMPLES]
        self._buffer = self._buffer[FRAME_SAMPLES:]

        # Convert to int16 for WebRTC
        frame_int16 = (frame_data * 32767).astype(np.int16)

        # Create audio frame
        frame = AudioFrame(format="s16", layout="mono", samples=FRAME_SAMPLES)
        frame.sample_rate = WEBRTC_SAMPLE_RATE
        frame.pts = self._pts
        frame.planes[0].update(frame_int16.tobytes())
        self._pts += FRAME_SAMPLES

        return frame


@dataclass
class WebRTCSession:
    """Represents a single WebRTC client session."""

    session_id: str
    peer_connection: RTCPeerConnection
    data_channel: Any = None
    stt_track: AudioBufferTrack | None = None
    tts_track: TTSOutputTrack | None = None
    stt_config: STTConfig = field(default_factory=STTConfig)
    tts_config: TTSConfig = field(default_factory=TTSConfig)
    _stt_active: bool = False
    _tts_active: bool = False
    _partial_task: asyncio.Task | None = None

    async def send_message(self, message: dict) -> None:
        """Send JSON message via data channel."""
        if self.data_channel and self.data_channel.readyState == "open":
            self.data_channel.send(json.dumps(message))

    async def close(self) -> None:
        """Close the session."""
        if self._partial_task:
            self._partial_task.cancel()
        await self.peer_connection.close()
        logger.info(f"Session {self.session_id} closed")


class WebRTCConnectionManager:
    """Manages WebRTC peer connections and sessions."""

    def __init__(
        self,
        transcriber: Any = None,
        synthesizer: Any = None,
        speaker_identifier: Any = None,
    ):
        """Initialize connection manager.

        Args:
            transcriber: Transcriber instance for STT
            synthesizer: Synthesizer instance for TTS
            speaker_identifier: SpeakerIdentifier for speaker labels
        """
        self.transcriber = transcriber
        self.synthesizer = synthesizer
        self.speaker_identifier = speaker_identifier
        self.sessions: dict[str, WebRTCSession] = {}
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._relay = MediaRelay()

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the asyncio event loop for async operations."""
        self._loop = loop

    async def create_session(self, offer_sdp: str) -> tuple[str, str, list[dict]]:
        """Create a new WebRTC session from client offer.

        Args:
            offer_sdp: SDP offer from client

        Returns:
            Tuple of (session_id, answer_sdp, ice_candidates)
        """
        session_id = str(uuid.uuid4())

        pc = RTCPeerConnection()
        session = WebRTCSession(session_id=session_id, peer_connection=pc)

        # Create TTS output track
        tts_track = TTSOutputTrack()
        session.tts_track = tts_track
        pc.addTrack(tts_track)

        # Create STT input track handler
        stt_track = AudioBufferTrack()
        session.stt_track = stt_track

        # Handle incoming tracks (client audio)
        @pc.on("track")
        def on_track(track: MediaStreamTrack):
            logger.info(f"Session {session_id}: Received track {track.kind}")
            if track.kind == "audio":
                # Relay and process incoming audio
                @track.on("ended")
                async def on_ended():
                    logger.info(f"Session {session_id}: Track ended")

                # Start a task to receive frames
                asyncio.create_task(self._receive_audio_frames(session, track))

        # Handle data channel
        @pc.on("datachannel")
        def on_datachannel(channel):
            logger.info(f"Session {session_id}: Data channel '{channel.label}' opened")
            session.data_channel = channel

            @channel.on("message")
            def on_message(message):
                asyncio.create_task(self._handle_data_message(session, message))

        # Handle connection state changes
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"Session {session_id}: Connection state: {pc.connectionState}")
            if pc.connectionState == "failed" or pc.connectionState == "closed":
                await self._remove_session(session_id)

        # Collect ICE candidates
        ice_candidates: list[dict] = []

        @pc.on("icecandidate")
        def on_icecandidate(candidate):
            if candidate:
                ice_candidates.append({
                    "candidate": candidate.candidate,
                    "sdpMid": candidate.sdpMid,
                    "sdpMLineIndex": candidate.sdpMLineIndex,
                })

        # Set remote description (offer)
        offer = RTCSessionDescription(sdp=offer_sdp, type="offer")
        await pc.setRemoteDescription(offer)

        # Create and set local description (answer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # Wait for ICE gathering to complete
        await self._wait_for_ice_gathering(pc)

        # Store session
        with self._lock:
            self.sessions[session_id] = session

        logger.info(f"Created session {session_id}")
        return session_id, pc.localDescription.sdp, ice_candidates

    async def _wait_for_ice_gathering(
        self, pc: RTCPeerConnection, timeout: float = 5.0
    ) -> None:
        """Wait for ICE gathering to complete."""
        if pc.iceGatheringState == "complete":
            return

        done = asyncio.Event()

        @pc.on("icegatheringstatechange")
        def on_ice_state_change():
            if pc.iceGatheringState == "complete":
                done.set()

        try:
            await asyncio.wait_for(done.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("ICE gathering timed out")

    async def add_ice_candidate(self, session_id: str, candidate: dict) -> bool:
        """Add ICE candidate to session.

        Args:
            session_id: Session ID
            candidate: ICE candidate dict

        Returns:
            True if successful
        """
        session = self.sessions.get(session_id)
        if not session:
            return False

        from aiortc import RTCIceCandidate

        ice_candidate = RTCIceCandidate(
            candidate=candidate.get("candidate"),
            sdpMid=candidate.get("sdpMid"),
            sdpMLineIndex=candidate.get("sdpMLineIndex"),
        )
        await session.peer_connection.addIceCandidate(ice_candidate)
        return True

    async def _receive_audio_frames(
        self, session: WebRTCSession, track: MediaStreamTrack
    ) -> None:
        """Receive and process audio frames from client."""
        while True:
            try:
                frame = await track.recv()
                if session.stt_track:
                    session.stt_track.on_frame(frame)
            except Exception as e:
                logger.debug(f"Track receive ended: {e}")
                break

    async def _handle_data_message(self, session: WebRTCSession, message: str) -> None:
        """Handle incoming data channel message."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "stt_start":
                await self._start_stt(session, data)
            elif msg_type == "stt_stop":
                await self._stop_stt(session)
            elif msg_type == "tts_start":
                await self._start_tts(session, data)
            elif msg_type == "tts_stop":
                await self._stop_tts(session)
            elif msg_type == "batch_transcribe":
                await self._batch_transcribe(session, data)
            else:
                await session.send_message({
                    "type": "error",
                    "code": "UNKNOWN_MESSAGE_TYPE",
                    "message": f"Unknown message type: {msg_type}",
                })
        except json.JSONDecodeError:
            await session.send_message({
                "type": "error",
                "code": "INVALID_JSON",
                "message": "Invalid JSON message",
            })
        except Exception as e:
            logger.exception(f"Error handling message: {e}")
            await session.send_message({
                "type": "error",
                "code": "INTERNAL_ERROR",
                "message": str(e),
            })

    async def _start_stt(self, session: WebRTCSession, config: dict) -> None:
        """Start STT recording."""
        if not self.transcriber:
            await session.send_message({
                "type": "error",
                "code": "STT_NOT_AVAILABLE",
                "message": "STT is not available",
            })
            return

        session.stt_config = STTConfig(
            language=config.get("language"),
            identify_speakers=config.get("identify_speakers", False),
            profiles_path=config.get("profiles_path"),
        )
        session._stt_active = True

        if session.stt_track:
            session.stt_track.start_recording()

        # Start partial transcription task
        session._partial_task = asyncio.create_task(
            self._send_partial_transcriptions(session)
        )

        await session.send_message({"type": "stt_started"})

    async def _send_partial_transcriptions(self, session: WebRTCSession) -> None:
        """Periodically send partial transcription results."""
        last_text = ""
        while session._stt_active:
            await asyncio.sleep(2.0)  # Send partials every 2 seconds

            if not session._stt_active or not session.stt_track:
                break

            audio = session.stt_track.get_partial_audio()
            if len(audio) < WHISPER_SAMPLE_RATE:  # Less than 1 second
                continue

            try:
                # Run transcription in thread pool (blocking call)
                loop = asyncio.get_event_loop()
                text = await loop.run_in_executor(
                    None,
                    self._transcribe_partial,
                    audio,
                )

                if text and text != last_text:
                    last_text = text
                    await session.send_message({
                        "type": "stt_partial",
                        "text": text,
                        "is_final": False,
                    })
            except Exception as e:
                logger.warning(f"Partial transcription failed: {e}")

    def _transcribe_partial(self, audio: np.ndarray) -> str:
        """Transcribe audio with fast settings for partials."""
        if not self.transcriber:
            return ""

        segments, _ = self.transcriber.model.transcribe(
            audio,
            language=self.transcriber.config.language,
            beam_size=1,  # Fast greedy decoding for partials
            vad_filter=True,
        )
        return " ".join(seg.text.strip() for seg in segments)

    async def _stop_stt(self, session: WebRTCSession) -> None:
        """Stop STT recording and return final transcription."""
        session._stt_active = False

        if session._partial_task:
            session._partial_task.cancel()
            session._partial_task = None

        if not session.stt_track:
            await session.send_message({
                "type": "error",
                "code": "NO_AUDIO",
                "message": "No audio track available",
            })
            return

        audio = session.stt_track.stop_recording()

        if len(audio) < WHISPER_SAMPLE_RATE // 2:  # Less than 0.5 seconds
            await session.send_message({
                "type": "stt_final",
                "text": "",
                "segments": [],
            })
            return

        # Run final transcription in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._transcribe_final,
            audio,
            session.stt_config,
        )

        await session.send_message({
            "type": "stt_final",
            **result,
        })

    def _transcribe_final(self, audio: np.ndarray, config: STTConfig) -> dict:
        """Transcribe audio with full quality and optional speaker ID."""
        if not self.transcriber:
            return {"text": "", "segments": []}

        segments = self.transcriber.transcribe_audio_with_segments(
            audio, sample_rate=WHISPER_SAMPLE_RATE
        )

        # Build result
        result_segments = []
        for start, end, text in segments:
            result_segments.append({
                "start": start,
                "end": end,
                "text": text,
                "speaker": "Unknown",
                "speaker_confidence": 0.0,
            })

        # Add speaker identification if requested
        if config.identify_speakers and self.speaker_identifier:
            try:
                identified = self.speaker_identifier.identify_segments_from_array(
                    audio,
                    WHISPER_SAMPLE_RATE,
                    [(s["start"], s["end"], s["text"]) for s in result_segments],
                )
                for i, seg in enumerate(identified):
                    result_segments[i]["speaker"] = seg.speaker
                    result_segments[i]["speaker_confidence"] = seg.confidence
            except Exception as e:
                logger.warning(f"Speaker identification failed: {e}")

        full_text = " ".join(s["text"] for s in result_segments)
        return {"text": full_text, "segments": result_segments}

    async def _start_tts(self, session: WebRTCSession, config: dict) -> None:
        """Start TTS synthesis and streaming."""
        if not self.synthesizer:
            await session.send_message({
                "type": "error",
                "code": "TTS_NOT_AVAILABLE",
                "message": "TTS is not available",
            })
            return

        text = config.get("text", "")
        if not text:
            await session.send_message({
                "type": "error",
                "code": "NO_TEXT",
                "message": "No text provided for TTS",
            })
            return

        session.tts_config = TTSConfig(
            voice=config.get("voice"),
            cfg_scale=config.get("cfg_scale"),
        )
        session._tts_active = True

        if session.tts_track:
            await session.tts_track.start_streaming()

        await session.send_message({"type": "tts_started"})

        # Run TTS in background
        asyncio.create_task(self._run_tts_streaming(session, text))

    async def _run_tts_streaming(self, session: WebRTCSession, text: str) -> None:
        """Run TTS streaming in background."""
        try:
            loop = asyncio.get_event_loop()

            # Run streaming synthesis in thread pool
            def generate_chunks():
                return list(self.synthesizer.synthesize_streaming(
                    text,
                    voice=session.tts_config.voice,
                    cfg_scale=session.tts_config.cfg_scale,
                ))

            chunks = await loop.run_in_executor(None, generate_chunks)

            for chunk in chunks:
                if not session._tts_active:
                    break
                if session.tts_track:
                    await session.tts_track.push_audio(chunk)

            if session.tts_track:
                await session.tts_track.stop_streaming()

            await session.send_message({"type": "tts_finished"})
        except Exception as e:
            logger.exception(f"TTS streaming failed: {e}")
            await session.send_message({
                "type": "error",
                "code": "TTS_ERROR",
                "message": str(e),
            })
        finally:
            session._tts_active = False

    async def _stop_tts(self, session: WebRTCSession) -> None:
        """Stop TTS streaming."""
        session._tts_active = False
        if session.tts_track:
            await session.tts_track.stop_streaming()

    async def _batch_transcribe(self, session: WebRTCSession, data: dict) -> None:
        """Handle batch transcription request via data channel."""
        if not self.transcriber:
            await session.send_message({
                "type": "error",
                "code": "STT_NOT_AVAILABLE",
                "message": "STT is not available",
            })
            return

        audio_b64 = data.get("audio_base64", "")
        if not audio_b64:
            await session.send_message({
                "type": "error",
                "code": "NO_AUDIO",
                "message": "No audio data provided",
            })
            return

        try:
            # Decode base64 audio (expects WAV)
            audio_bytes = base64.b64decode(audio_b64)

            # Parse WAV
            import io
            import wave

            buffer = io.BytesIO(audio_bytes)
            with wave.open(buffer, "rb") as wav:
                sample_rate = wav.getframerate()
                n_frames = wav.getnframes()
                audio_raw = wav.readframes(n_frames)
                audio = np.frombuffer(audio_raw, dtype=np.int16).astype(np.float32)
                audio = audio / 32768.0

            # Resample to 16kHz if needed
            if sample_rate != WHISPER_SAMPLE_RATE:
                audio = resample_audio(audio, sample_rate, WHISPER_SAMPLE_RATE)

            # Run transcription
            config = STTConfig(
                language=data.get("language"),
                identify_speakers=data.get("identify_speakers", False),
                profiles_path=data.get("profiles_path"),
            )

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._transcribe_final,
                audio,
                config,
            )

            await session.send_message({
                "type": "batch_result",
                **result,
            })
        except Exception as e:
            logger.exception(f"Batch transcription failed: {e}")
            await session.send_message({
                "type": "error",
                "code": "BATCH_ERROR",
                "message": str(e),
            })

    async def _remove_session(self, session_id: str) -> None:
        """Remove and cleanup a session."""
        with self._lock:
            session = self.sessions.pop(session_id, None)
        if session:
            await session.close()

    async def close_all(self) -> None:
        """Close all sessions."""
        with self._lock:
            session_ids = list(self.sessions.keys())
        for session_id in session_ids:
            await self._remove_session(session_id)

    def get_session_count(self) -> int:
        """Get number of active sessions."""
        with self._lock:
            return len(self.sessions)
