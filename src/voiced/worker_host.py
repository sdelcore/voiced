"""Parent-side host for the disposable inference worker process.

``WorkerHost`` owns the worker's lifecycle: it spawns the child lazily on
the first operation, counts leases so the worker is never stopped while an
operation or stream is active, and terminates the process after the shared
idle timeout. Process termination — not ``torch.cuda.empty_cache()`` — is
what guarantees the VRAM comes back.

``WorkerTranscriber`` and ``WorkerSynthesizer`` mirror the public method
surfaces of ``Transcriber`` and ``Synthesizer`` so the daemon, HTTP server,
and WebRTC manager use them unchanged. This module must not import torch,
NeMo, or Kokoro.
"""

import itertools
import logging
import queue
import threading
from collections.abc import Iterator
from multiprocessing import get_context
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np

from voiced.config import Config, TranscriptionConfig
from voiced.model_host import ModelHost
from voiced.worker import worker_main

logger = logging.getLogger(__name__)

# Mirrors synthesizer.SAMPLE_RATE / TTS_MODEL and diarizer.SPEAKER_MODEL
# without importing those modules (they import torch at module level).
TTS_SAMPLE_RATE = 24000
TTS_MODEL = "hexgrad/Kokoro-82M"
SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

_START_TIMEOUT = 120.0
_GRACEFUL_TIMEOUT = 10.0
_KILL_TIMEOUT = 5.0


class WorkerCrashError(RuntimeError):
    """The inference worker died while an operation was in flight."""


class WorkerOperationError(RuntimeError):
    """An operation failed inside the worker with a non-builtin exception."""

    def __init__(self, original_type: str, message: str, original_traceback: str = ""):
        super().__init__(f"{original_type}: {message}")
        self.original_type = original_type
        self.original_traceback = original_traceback


def _rebuild_exception(info: dict) -> BaseException:
    """Reconstruct a builtin exception when possible, else wrap it."""
    import builtins

    cls = getattr(builtins, info["type"], None)
    if isinstance(cls, type) and issubclass(cls, Exception):
        try:
            return cls(info["message"])
        except TypeError:
            pass
    return WorkerOperationError(info["type"], info["message"], info.get("traceback", ""))


class _WorkerHandle:
    """One live worker process: its pipe, pending requests, and liveness."""

    def __init__(self, process, conn: Connection):
        self.process = process
        self.conn = conn
        self.stopping = False
        self._send_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: dict[int, queue.Queue] = {}
        self._ids = itertools.count()

    def open_request(self, op: str, kwargs: dict) -> tuple[int, queue.Queue]:
        req_id = next(self._ids)
        q: queue.Queue = queue.Queue()
        with self._pending_lock:
            self._pending[req_id] = q
        try:
            with self._send_lock:
                self.conn.send(("request", req_id, op, kwargs))
        except (OSError, ValueError) as exc:
            self.close_request(req_id)
            raise WorkerCrashError(f"inference worker is gone: {exc}") from exc
        return req_id, q

    def close_request(self, req_id: int) -> None:
        with self._pending_lock:
            self._pending.pop(req_id, None)

    def route(self, req_id: int, event: tuple) -> None:
        with self._pending_lock:
            q = self._pending.get(req_id)
        if q is not None:
            q.put(event)

    def fail_pending(self, message: str) -> None:
        with self._pending_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for q in pending:
            q.put(("crash", message))

    @property
    def alive(self) -> bool:
        return self.process.is_alive()


class WorkerHost:
    """Spawns, supervises, and retires the inference worker process.

    Reuses ModelHost for the lease/idle-timer machinery: the "model" is the
    worker process handle, the loader spawns it, and the unload hook is the
    graceful-then-forced termination sequence.
    """

    def __init__(
        self,
        config: Config,
        idle_timeout: float | None = None,
        *,
        transcriber_factory=None,
        synthesizer_factory=None,
        diarizer_factory=None,
        start_timeout: float = _START_TIMEOUT,
        graceful_timeout: float = _GRACEFUL_TIMEOUT,
        kill_timeout: float = _KILL_TIMEOUT,
    ):
        self._config = config
        self._transcriber_factory = transcriber_factory
        self._synthesizer_factory = synthesizer_factory
        self._diarizer_factory = diarizer_factory
        self._start_timeout = start_timeout
        self._graceful_timeout = graceful_timeout
        self._kill_timeout = kill_timeout

        self._stt_used = False
        self._tts_used = False
        self._flags_lock = threading.Lock()

        self._host: ModelHost[_WorkerHandle] = ModelHost(
            loader=self._spawn,
            idle_timeout=idle_timeout,
            on_unload=self._stop,
            name="inference-worker",
        )

    # ----- lifecycle -----

    def _spawn(self) -> _WorkerHandle:
        ctx = get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        process = ctx.Process(
            target=worker_main,
            args=(
                child_conn,
                self._config,
                self._transcriber_factory,
                self._synthesizer_factory,
                self._diarizer_factory,
            ),
            name="voiced-inference-worker",
            daemon=True,
        )
        process.start()
        child_conn.close()

        if not parent_conn.poll(self._start_timeout):
            process.kill()
            process.join()
            parent_conn.close()
            raise RuntimeError(
                f"Inference worker did not become ready within {self._start_timeout}s"
            )
        try:
            ready = parent_conn.recv()
        except (EOFError, OSError) as exc:
            process.join()
            parent_conn.close()
            raise RuntimeError(f"Inference worker died during startup: {exc}") from exc
        if ready != ("ready",):
            process.kill()
            process.join()
            parent_conn.close()
            raise RuntimeError(f"Unexpected worker handshake: {ready!r}")

        with self._flags_lock:
            self._stt_used = False
            self._tts_used = False

        handle = _WorkerHandle(process, parent_conn)
        threading.Thread(
            target=self._reader_loop,
            args=(handle,),
            daemon=True,
            name="voiced-worker-reader",
        ).start()
        logger.info(f"Inference worker started (pid {process.pid})")
        return handle

    def _reader_loop(self, handle: _WorkerHandle) -> None:
        """Sole reader of the worker pipe; routes messages to open requests."""
        while True:
            try:
                message = handle.conn.recv()
            except (EOFError, OSError):
                break
            kind = message[0]
            if kind == "result":
                handle.route(message[1], ("result", message[2]))
            elif kind == "chunk":
                handle.route(message[1], ("chunk", message[2]))
            elif kind == "end":
                handle.route(message[1], ("end", None))
            elif kind == "error":
                handle.route(message[1], ("error", message[2]))
            else:
                logger.warning(f"Unknown message from worker: {kind!r}")

        if not handle.stopping:
            self._on_worker_death(handle)

    def _on_worker_death(self, handle: _WorkerHandle) -> None:
        logger.error("Inference worker died unexpectedly")
        self._host.invalidate(handle)
        handle.fail_pending("inference worker died unexpectedly")
        handle.process.join(timeout=self._kill_timeout)
        try:
            handle.conn.close()
        except OSError:
            pass
        with self._flags_lock:
            self._stt_used = False
            self._tts_used = False

    def _stop(self, handle: _WorkerHandle) -> None:
        """Graceful-then-forced worker termination. Fired with zero leases."""
        handle.stopping = True
        try:
            handle.conn.send(("shutdown",))
        except (OSError, ValueError):
            pass

        handle.process.join(timeout=self._graceful_timeout)
        if handle.alive:
            logger.warning(f"Worker did not exit within {self._graceful_timeout}s; terminating")
            handle.process.terminate()
            handle.process.join(timeout=self._kill_timeout)
        if handle.alive:
            logger.warning("Worker ignored SIGTERM; killing")
            handle.process.kill()
            handle.process.join()

        try:
            handle.conn.close()
        except OSError:
            pass
        handle.fail_pending("inference worker was shut down")
        with self._flags_lock:
            self._stt_used = False
            self._tts_used = False
        logger.info("Inference worker stopped")

    # ----- requests -----

    def request(self, op: str, **kwargs) -> Any:
        """Run one operation on the worker, starting it if needed."""
        with self._host.use() as handle:
            req_id, q = handle.open_request(op, kwargs)
            try:
                result = self._wait_result(q)
            finally:
                handle.close_request(req_id)
            self._mark_used(op)
            return result

    def stream(self, op: str, **kwargs) -> Iterator[Any]:
        """Run a streaming operation, holding the worker lease until the
        stream is fully consumed (or the consumer stops iterating)."""
        with self._host.use() as handle:
            req_id, q = handle.open_request(op, kwargs)
            try:
                while True:
                    kind, payload = q.get()
                    if kind == "chunk":
                        self._mark_used(op)
                        yield payload
                    elif kind == "end":
                        break
                    elif kind == "error":
                        raise _rebuild_exception(payload)
                    elif kind == "crash":
                        raise WorkerCrashError(payload)
            finally:
                handle.close_request(req_id)

    def _wait_result(self, q: queue.Queue) -> Any:
        while True:
            kind, payload = q.get()
            if kind == "result":
                return payload
            if kind == "error":
                raise _rebuild_exception(payload)
            if kind == "crash":
                raise WorkerCrashError(payload)
            # "chunk"/"end" for a non-streaming op — protocol bug; ignore.
            logger.warning(f"Unexpected {kind!r} event for non-streaming request")

    def _mark_used(self, op: str) -> None:
        with self._flags_lock:
            if op.startswith("stt."):
                self._stt_used = True
            elif op.startswith("tts."):
                self._tts_used = True

    # ----- state -----

    @property
    def is_running(self) -> bool:
        return self._host.is_loaded

    @property
    def stt_model_loaded(self) -> bool:
        with self._flags_lock:
            return self._stt_used and self.is_running

    @property
    def tts_model_loaded(self) -> bool:
        with self._flags_lock:
            return self._tts_used and self.is_running

    def unload(self) -> None:
        """Stop the worker now. Raises if any operation is in flight."""
        self._host.unload()

    def shutdown(self) -> None:
        """Final shutdown: stop the worker and refuse further operations."""
        self._host.shutdown()


class WorkerTranscriber:
    """Transcriber facade that executes in the inference worker process.

    Mirrors ``Transcriber``'s public surface (transcribe_* methods, device,
    warmup, unload) for the daemon, HTTP server, and WebRTC manager.
    """

    def __init__(self, worker: WorkerHost, config: TranscriptionConfig | None = None):
        self.config = config or TranscriptionConfig()
        self._worker = worker
        self._device: str | None = None

    @property
    def device(self) -> str:
        """Resolved device once the worker has run an op; the configured
        device string before that (resolving earlier would need torch)."""
        return self._device or self.config.device

    def _request(self, op: str, **kwargs) -> Any:
        result = self._worker.request(op, **kwargs)
        self._device = result["device"]
        return result["value"]

    def warmup(self) -> None:
        self._request("stt.warmup")

    def transcribe_file(self, audio_path: str | Path) -> str:
        return self._request("stt.transcribe_file", path=str(audio_path))

    def transcribe_file_with_segments(
        self, audio_path: str | Path
    ) -> list[tuple[float, float, str]]:
        return self._request("stt.transcribe_file_with_segments", path=str(audio_path))

    def transcribe_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        return self._request("stt.transcribe_audio", audio=audio, sample_rate=sample_rate)

    def transcribe_audio_with_segments(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> list[tuple[float, float, str]]:
        return self._request(
            "stt.transcribe_audio_with_segments", audio=audio, sample_rate=sample_rate
        )

    def transcribe_audio_with_words(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> tuple[str, list[tuple[str, float, float, float]]]:
        return self._request(
            "stt.transcribe_audio_with_words", audio=audio, sample_rate=sample_rate
        )

    def transcribe_partial(self, audio: np.ndarray) -> str:
        return self._request("stt.transcribe_partial", audio=audio)

    def unload(self) -> None:
        """Stop the shared inference worker (releases TTS too)."""
        self._worker.unload()
        self._device = None


class WorkerSynthesizer:
    """Synthesizer facade that executes in the inference worker process."""

    def __init__(self, worker: WorkerHost, config: Config):
        self._worker = worker
        self._config = config

    @property
    def sample_rate(self) -> int:
        return TTS_SAMPLE_RATE

    @property
    def is_loaded(self) -> bool:
        return self._worker.tts_model_loaded

    @property
    def device(self) -> str | None:
        return self._config.tts.device if self.is_loaded else None

    def synthesize(
        self, text: str, voice: str | None = None, speed: float | None = None
    ) -> np.ndarray:
        return self._worker.request("tts.synthesize", text=text, voice=voice, speed=speed)

    def synthesize_streaming(
        self, text: str, voice: str | None = None, speed: float | None = None
    ) -> Iterator[np.ndarray]:
        yield from self._worker.stream(
            "tts.synthesize_streaming", text=text, voice=voice, speed=speed
        )

    def get_status(self) -> dict:
        if self._worker.tts_model_loaded:
            return self._worker.request("tts.get_status")
        from voiced.voice_manager import VoiceManager

        return {
            "model_loaded": False,
            "model": TTS_MODEL,
            "device": None,
            "default_voice": self._config.tts.default_voice,
            "speed": self._config.tts.speed,
            "unload_timeout_seconds": self._config.unload_timeout_minutes * 60,
            "last_used": None,
            "voices_downloaded": VoiceManager().list_downloaded(),
        }

    def shutdown(self) -> None:
        self._worker.shutdown()


class WorkerDiarizer:
    """Diarization facade that executes in the inference worker process.

    Serves both Voiced roles: ``speaker_diarizer`` (clustering + profile
    match for batch transcribe) and ``speaker_identifier`` (per-segment
    matching for WebRTC). Results come back as torch-free
    ``voiced.speaker_segments.IdentifiedSegment`` instances.
    """

    def __init__(self, worker: WorkerHost):
        self._worker = worker

    def diarize_and_match_profiles_from_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
        profiles=None,
        num_speakers: int | None = None,
    ):
        return self._worker.request(
            "diar.diarize_and_match",
            audio=audio,
            sample_rate=sample_rate,
            profiles=profiles,
            num_speakers=num_speakers,
        )

    def identify_segments_from_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
        transcription_segments: list[tuple[float, float, str]],
    ):
        return self._worker.request(
            "diar.identify_segments",
            audio=audio,
            sample_rate=sample_rate,
            segments=transcription_segments,
        )

    def unload(self) -> None:
        """No-op: the worker process lifecycle owns model release."""


class WorkerSpeakerEmbedder:
    """Embedder facade for profile enrolment, executing in the worker."""

    model_source = SPEAKER_MODEL

    def __init__(self, worker: WorkerHost):
        self._worker = worker

    def extract_embedding_from_array(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> np.ndarray:
        return self._worker.request("diar.embed", audio=audio, sample_rate=sample_rate)
