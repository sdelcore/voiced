"""WebRTC client for real-time audio streaming."""

import asyncio
import base64
import io
import json
import logging
import queue
import threading
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import sounddevice as sd
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from av import AudioFrame
from scipy import signal

logger = logging.getLogger(__name__)

# Sample rates
WEBRTC_SAMPLE_RATE = 48000
WHISPER_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000

# Audio settings
FRAME_SAMPLES = 960  # 20ms at 48kHz
CHANNELS = 1


def resample_audio(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample audio from one sample rate to another."""
    if from_rate == to_rate:
        return audio
    num_samples = int(len(audio) * to_rate / from_rate)
    return signal.resample(audio, num_samples).astype(np.float32)


class LocalAudioSource(MediaStreamTrack):
    """Audio track that captures from local microphone."""

    kind = "audio"

    def __init__(self, device: int | str | None = None):
        super().__init__()
        self._device = device
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._pts = 0
        self._running = False

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags
    ) -> None:
        """Callback for sounddevice audio input."""
        if status:
            logger.warning(f"Audio input status: {status}")
        # Convert to mono float32
        audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        audio = audio.astype(np.float32)
        self._queue.put(audio.copy())

    def start(self) -> None:
        """Start audio capture."""
        if self._running:
            return

        self._stream = sd.InputStream(
            device=self._device,
            channels=CHANNELS,
            samplerate=WEBRTC_SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            dtype=np.float32,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._running = True
        logger.info("Local audio capture started")

    def stop(self) -> None:
        """Stop audio capture."""
        if not self._running:
            return

        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("Local audio capture stopped")

    async def recv(self) -> AudioFrame:
        """Generate audio frames for WebRTC."""
        # Get audio from queue (blocking in thread pool)
        loop = asyncio.get_event_loop()
        try:
            audio = await loop.run_in_executor(None, self._queue.get, True, 0.1)
        except queue.Empty:
            # Generate silence if no audio available
            audio = np.zeros(FRAME_SAMPLES, dtype=np.float32)

        # Ensure correct frame size
        if len(audio) < FRAME_SAMPLES:
            audio = np.pad(audio, (0, FRAME_SAMPLES - len(audio)))
        elif len(audio) > FRAME_SAMPLES:
            audio = audio[:FRAME_SAMPLES]

        # Convert to int16 for WebRTC
        audio_int16 = (audio * 32767).astype(np.int16)

        # Create audio frame
        frame = AudioFrame(format="s16", layout="mono", samples=FRAME_SAMPLES)
        frame.sample_rate = WEBRTC_SAMPLE_RATE
        frame.pts = self._pts
        frame.planes[0].update(audio_int16.tobytes())
        self._pts += FRAME_SAMPLES

        return frame


class RemoteAudioSink:
    """Plays received audio from WebRTC."""

    def __init__(self, device: int | str | None = None):
        self._device = device
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.OutputStream | None = None
        self._running = False

    def _audio_callback(
        self, outdata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags
    ) -> None:
        """Callback for sounddevice audio output."""
        if status:
            logger.warning(f"Audio output status: {status}")

        try:
            audio = self._queue.get_nowait()
            # Ensure correct frame size
            if len(audio) < frames:
                audio = np.pad(audio, (0, frames - len(audio)))
            elif len(audio) > frames:
                audio = audio[:frames]
            outdata[:, 0] = audio
        except queue.Empty:
            outdata.fill(0)  # Output silence

    def start(self) -> None:
        """Start audio playback."""
        if self._running:
            return

        self._stream = sd.OutputStream(
            device=self._device,
            channels=CHANNELS,
            samplerate=WEBRTC_SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            dtype=np.float32,
            callback=self._audio_callback,
        )
        self._stream.start()
        self._running = True
        logger.info("Remote audio playback started")

    def stop(self) -> None:
        """Stop audio playback."""
        if not self._running:
            return

        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("Remote audio playback stopped")

    def push_audio(self, audio: np.ndarray) -> None:
        """Push audio to playback queue."""
        self._queue.put(audio)


@dataclass
class STTResult:
    """Result from STT transcription."""

    text: str
    segments: list[dict] = field(default_factory=list)
    is_final: bool = True


@dataclass
class WebRTCClientConfig:
    """Configuration for WebRTC client."""

    server_url: str
    timeout: float = 30.0
    input_device: int | str | None = None
    output_device: int | str | None = None


class WebRTCClient:
    """Client for WebRTC-based voice streaming."""

    def __init__(self, config: WebRTCClientConfig):
        self.config = config
        self._pc: RTCPeerConnection | None = None
        self._session_id: str | None = None
        self._data_channel = None
        self._local_audio: LocalAudioSource | None = None
        self._remote_audio: RemoteAudioSink | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._connected = False

        # Callbacks
        self._on_stt_partial: Callable[[str], None] | None = None
        self._on_stt_final: Callable[[STTResult], None] | None = None
        self._on_tts_started: Callable[[], None] | None = None
        self._on_tts_finished: Callable[[], None] | None = None
        self._on_error: Callable[[str, str], None] | None = None

        # Result futures for synchronous API
        self._stt_result: asyncio.Future | None = None
        self._tts_done: asyncio.Future | None = None

    def set_callbacks(
        self,
        on_stt_partial: Callable[[str], None] | None = None,
        on_stt_final: Callable[[STTResult], None] | None = None,
        on_tts_started: Callable[[], None] | None = None,
        on_tts_finished: Callable[[], None] | None = None,
        on_error: Callable[[str, str], None] | None = None,
    ) -> None:
        """Set event callbacks."""
        self._on_stt_partial = on_stt_partial
        self._on_stt_final = on_stt_final
        self._on_tts_started = on_tts_started
        self._on_tts_finished = on_tts_finished
        self._on_error = on_error

    async def connect(self) -> None:
        """Establish WebRTC connection to server."""
        if self._connected:
            return

        self._pc = RTCPeerConnection()

        # Create local audio track
        self._local_audio = LocalAudioSource(device=self.config.input_device)
        self._pc.addTrack(self._local_audio)

        # Create data channel
        self._data_channel = self._pc.createDataChannel("control")
        self._data_channel_ready = asyncio.Event()

        @self._data_channel.on("open")
        def on_open():
            logger.info("Data channel opened")
            self._data_channel_ready.set()

        @self._data_channel.on("message")
        def on_message(message):
            asyncio.create_task(self._handle_message(message))

        # Handle incoming tracks (TTS audio)
        self._remote_audio = RemoteAudioSink(device=self.config.output_device)

        @self._pc.on("track")
        def on_track(track: MediaStreamTrack):
            logger.info(f"Received track: {track.kind}")
            if track.kind == "audio":
                asyncio.create_task(self._receive_remote_audio(track))

        # Create offer
        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)

        # Wait for ICE gathering
        await self._wait_for_ice_gathering()

        # Send offer to server
        answer_data = await self._send_offer(self._pc.localDescription.sdp)

        # Set remote description
        answer = RTCSessionDescription(sdp=answer_data["sdp"], type=answer_data["type"])
        await self._pc.setRemoteDescription(answer)

        self._session_id = answer_data.get("session_id")
        self._connected = True

        # Start audio
        self._local_audio.start()
        self._remote_audio.start()

        # Wait for data channel to open (with timeout)
        try:
            await asyncio.wait_for(self._data_channel_ready.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            raise RuntimeError("Data channel failed to open")

        logger.info(f"Connected to server, session: {self._session_id}")

    async def _wait_for_ice_gathering(self, timeout: float = 5.0) -> None:
        """Wait for ICE gathering to complete."""
        if self._pc.iceGatheringState == "complete":
            return

        done = asyncio.Event()

        @self._pc.on("icegatheringstatechange")
        def on_state_change():
            if self._pc.iceGatheringState == "complete":
                done.set()

        try:
            await asyncio.wait_for(done.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("ICE gathering timed out")

    async def _send_offer(self, sdp: str) -> dict:
        """Send SDP offer to server via HTTP."""
        url = f"{self.config.server_url.rstrip('/')}/webrtc/offer"
        data = json.dumps({"sdp": sdp}).encode("utf-8")

        request = Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: urlopen(request, timeout=self.config.timeout),
        )

        return json.loads(response.read().decode("utf-8"))

    async def _receive_remote_audio(self, track: MediaStreamTrack) -> None:
        """Receive and play remote audio."""
        while True:
            try:
                frame = await track.recv()
                # Convert frame to numpy
                audio = frame.to_ndarray()
                if audio.ndim > 1:
                    audio = audio.mean(axis=0)
                audio = audio.astype(np.float32) / 32768.0
                self._remote_audio.push_audio(audio)
            except Exception as e:
                logger.debug(f"Remote audio ended: {e}")
                break

    async def _handle_message(self, message: str) -> None:
        """Handle incoming data channel message."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "stt_partial":
                text = data.get("text", "")
                if self._on_stt_partial:
                    self._on_stt_partial(text)

            elif msg_type == "stt_final":
                result = STTResult(
                    text=data.get("text", ""),
                    segments=data.get("segments", []),
                    is_final=True,
                )
                if self._on_stt_final:
                    self._on_stt_final(result)
                if self._stt_result and not self._stt_result.done():
                    self._stt_result.set_result(result)

            elif msg_type == "batch_result":
                result = STTResult(
                    text=data.get("text", ""),
                    segments=data.get("segments", []),
                    is_final=True,
                )
                if self._on_stt_final:
                    self._on_stt_final(result)
                if self._stt_result and not self._stt_result.done():
                    self._stt_result.set_result(result)

            elif msg_type == "tts_started":
                if self._on_tts_started:
                    self._on_tts_started()

            elif msg_type == "tts_finished":
                if self._on_tts_finished:
                    self._on_tts_finished()
                if self._tts_done and not self._tts_done.done():
                    self._tts_done.set_result(True)

            elif msg_type == "error":
                code = data.get("code", "UNKNOWN")
                message = data.get("message", "Unknown error")
                logger.error(f"Server error: {code} - {message}")
                if self._on_error:
                    self._on_error(code, message)
                # Fail pending operations
                error = Exception(f"{code}: {message}")
                if self._stt_result and not self._stt_result.done():
                    self._stt_result.set_exception(error)
                if self._tts_done and not self._tts_done.done():
                    self._tts_done.set_exception(error)

            elif msg_type == "stt_started":
                logger.debug("STT recording started on server")

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON message: {message}")

    async def start_stt(
        self, language: str | None = None, identify_speakers: bool = False
    ) -> None:
        """Start STT recording."""
        if not self._connected or not self._data_channel:
            raise RuntimeError("Not connected")

        self._stt_result = asyncio.get_event_loop().create_future()

        message = {"type": "stt_start", "identify_speakers": identify_speakers}
        if language:
            message["language"] = language

        self._data_channel.send(json.dumps(message))
        logger.info("STT recording started")

    async def stop_stt(self) -> STTResult:
        """Stop STT recording and get result."""
        if not self._connected or not self._data_channel:
            raise RuntimeError("Not connected")

        self._data_channel.send(json.dumps({"type": "stt_stop"}))
        logger.info("STT recording stopped, waiting for result...")

        if self._stt_result:
            return await self._stt_result
        return STTResult(text="", segments=[], is_final=True)

    async def speak(
        self, text: str, voice: str | None = None, cfg_scale: float | None = None
    ) -> None:
        """Synthesize and play text via TTS."""
        if not self._connected or not self._data_channel:
            raise RuntimeError("Not connected")

        self._tts_done = asyncio.get_event_loop().create_future()

        message = {"type": "tts_start", "text": text}
        if voice:
            message["voice"] = voice
        if cfg_scale is not None:
            message["cfg_scale"] = cfg_scale

        self._data_channel.send(json.dumps(message))
        logger.info(f"TTS started: {text[:50]}...")

        await self._tts_done
        logger.info("TTS finished")

    async def stop_tts(self) -> None:
        """Stop TTS playback."""
        if not self._connected or not self._data_channel:
            return

        self._data_channel.send(json.dumps({"type": "tts_stop"}))

    async def batch_transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: str | None = None,
        identify_speakers: bool = False,
    ) -> STTResult:
        """Transcribe audio via batch mode (sends complete audio)."""
        if not self._connected or not self._data_channel:
            raise RuntimeError("Not connected")

        # Convert audio to WAV bytes
        wav_bytes = self._audio_to_wav(audio, sample_rate)
        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

        self._stt_result = asyncio.get_event_loop().create_future()

        message = {
            "type": "batch_transcribe",
            "audio_base64": audio_b64,
            "identify_speakers": identify_speakers,
        }
        if language:
            message["language"] = language

        self._data_channel.send(json.dumps(message))
        logger.info("Batch transcription started")

        return await self._stt_result

    def _audio_to_wav(self, audio: np.ndarray, sample_rate: int) -> bytes:
        """Convert numpy audio array to WAV bytes."""
        buffer = io.BytesIO()

        # Normalize to [-1, 1] if needed
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / 32768.0

        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)

        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio_int16.tobytes())

        return buffer.getvalue()

    async def disconnect(self) -> None:
        """Disconnect from server."""
        if not self._connected:
            return

        self._connected = False

        if self._local_audio:
            self._local_audio.stop()
            self._local_audio = None

        if self._remote_audio:
            self._remote_audio.stop()
            self._remote_audio = None

        if self._pc:
            await self._pc.close()
            self._pc = None

        self._session_id = None
        self._data_channel = None

        logger.info("Disconnected from server")

    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._connected

    @property
    def session_id(self) -> str | None:
        """Get current session ID."""
        return self._session_id


class SyncWebRTCClient:
    """Synchronous wrapper for WebRTCClient."""

    def __init__(self, config: WebRTCClientConfig):
        self._config = config
        self._client: WebRTCClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def _run_loop(self) -> None:
        """Run asyncio event loop in thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def connect(self) -> None:
        """Connect to server."""
        # Start event loop thread
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        # Wait for loop to be ready
        while self._loop is None:
            import time
            time.sleep(0.01)

        # Create client and connect
        self._client = WebRTCClient(self._config)
        future = asyncio.run_coroutine_threadsafe(self._client.connect(), self._loop)
        future.result(timeout=30.0)

    def start_stt(
        self, language: str | None = None, identify_speakers: bool = False
    ) -> None:
        """Start STT recording."""
        if not self._client or not self._loop:
            raise RuntimeError("Not connected")

        future = asyncio.run_coroutine_threadsafe(
            self._client.start_stt(language, identify_speakers),
            self._loop,
        )
        future.result(timeout=5.0)

    def stop_stt(self, timeout: float = 30.0) -> STTResult:
        """Stop STT and get result."""
        if not self._client or not self._loop:
            raise RuntimeError("Not connected")

        future = asyncio.run_coroutine_threadsafe(self._client.stop_stt(), self._loop)
        return future.result(timeout=timeout)

    def speak(
        self,
        text: str,
        voice: str | None = None,
        cfg_scale: float | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Speak text via TTS."""
        if not self._client or not self._loop:
            raise RuntimeError("Not connected")

        future = asyncio.run_coroutine_threadsafe(
            self._client.speak(text, voice, cfg_scale),
            self._loop,
        )
        future.result(timeout=timeout)

    def batch_transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language: str | None = None,
        identify_speakers: bool = False,
        timeout: float = 60.0,
    ) -> STTResult:
        """Transcribe audio via batch mode."""
        if not self._client or not self._loop:
            raise RuntimeError("Not connected")

        future = asyncio.run_coroutine_threadsafe(
            self._client.batch_transcribe(audio, sample_rate, language, identify_speakers),
            self._loop,
        )
        return future.result(timeout=timeout)

    def disconnect(self) -> None:
        """Disconnect from server."""
        if self._client and self._loop:
            future = asyncio.run_coroutine_threadsafe(
                self._client.disconnect(),
                self._loop,
            )
            try:
                future.result(timeout=5.0)
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")

        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

        self._client = None
        self._loop = None

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._client is not None and self._client.is_connected

    def set_callbacks(
        self,
        on_stt_partial: Callable[[str], None] | None = None,
        on_stt_final: Callable[[STTResult], None] | None = None,
        on_tts_started: Callable[[], None] | None = None,
        on_tts_finished: Callable[[], None] | None = None,
        on_error: Callable[[str, str], None] | None = None,
    ) -> None:
        """Set event callbacks."""
        if self._client:
            self._client.set_callbacks(
                on_stt_partial, on_stt_final, on_tts_started, on_tts_finished, on_error
            )
