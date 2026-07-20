"""Inference worker — child-process entry point for GPU model execution.

The daemon keeps models out of its own process because PyTorch/NeMo/CUDA
retain process-level allocations that ``torch.cuda.empty_cache()`` cannot
reliably release. This module runs inside a disposable child process
(spawned by ``worker_host.WorkerHost``); when the worker exits, the OS and
CUDA driver reclaim all of its VRAM.

Protocol (tuples over a duplex ``multiprocessing.Pipe``):

    parent -> worker:  ("request", req_id, op, kwargs)
                       ("shutdown",)
    worker -> parent:  ("ready",)
                       ("result", req_id, value)     # non-streaming success
                       ("chunk", req_id, chunk)      # one streamed chunk
                       ("end", req_id)               # end of stream
                       ("error", req_id, error_dict) # structured failure

STT results are wrapped as ``{"value": ..., "device": ...}`` so the parent
can report the resolved device without a CUDA-touching import.

Torch/NeMo/Kokoro are imported lazily inside this process only. The module
itself must stay import-light: ``spawn`` re-imports it in the child, and the
parent imports it to reference ``worker_main``.
"""

import logging
import signal
import threading
import traceback
from multiprocessing.connection import Connection
from typing import Any

from voiced.config import Config

logger = logging.getLogger(__name__)

# Streamed ops send ("chunk", ...) messages instead of a single result.
STREAMING_OPS = {"tts.synthesize_streaming"}


def _describe_exception(exc: BaseException) -> dict:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(exc)),
    }


def default_transcriber_factory(config: Config) -> Any:
    from voiced.transcriber import Transcriber

    # Idle unload inside the worker is disabled: process termination (managed
    # by the parent's WorkerHost) is the VRAM-release mechanism.
    return Transcriber(config.transcription, unload_timeout_minutes=0)


def default_synthesizer_factory(config: Config) -> Any:
    from voiced.synthesizer import Synthesizer, TTSConfig

    return Synthesizer(
        TTSConfig(
            device=config.tts.device,
            default_voice=config.tts.default_voice,
            speed=config.tts.speed,
            unload_timeout_seconds=0,
        )
    )


class _DefaultDiarBackend:
    """Bundles the speechbrain-based diarization capabilities for the worker.

    Results are voiced.speaker_segments.IdentifiedSegment instances — that
    module is torch-free, so the parent can unpickle them without importing
    the diarization stack.
    """

    def __init__(self, config: Config):
        from voiced.diarizer import SpeakerDiarizer, SpeakerIdentifier

        self._diarizer = SpeakerDiarizer(config=config.diarization)
        self._identifier = SpeakerIdentifier(config=config.diarization)

    def diarize_and_match(self, audio, sample_rate, profiles, num_speakers):
        return self._diarizer.diarize_and_match_profiles_from_array(
            audio, sample_rate, profiles=profiles, num_speakers=num_speakers
        )

    def identify_segments(self, audio, sample_rate, segments):
        return self._identifier.identify_segments_from_array(audio, sample_rate, segments)

    def embed(self, audio, sample_rate):
        return self._identifier.embedder.extract_embedding_from_array(audio, sample_rate)


def default_diarizer_factory(config: Config) -> Any:
    return _DefaultDiarBackend(config)


class _WorkerState:
    """Lazily constructed model wrappers plus the outbound send lock."""

    def __init__(
        self,
        conn: Connection,
        config: Config,
        transcriber_factory,
        synthesizer_factory,
        diarizer_factory,
    ):
        self.conn = conn
        self.config = config
        self._transcriber_factory = transcriber_factory
        self._synthesizer_factory = synthesizer_factory
        self._diarizer_factory = diarizer_factory
        self._transcriber: Any = None
        self._synthesizer: Any = None
        self._diarizer: Any = None
        self._model_lock = threading.Lock()
        self._send_lock = threading.Lock()

    def send(self, message: tuple) -> None:
        with self._send_lock:
            self.conn.send(message)

    @property
    def transcriber(self) -> Any:
        with self._model_lock:
            if self._transcriber is None:
                self._transcriber = self._transcriber_factory(self.config)
            return self._transcriber

    @property
    def synthesizer(self) -> Any:
        with self._model_lock:
            if self._synthesizer is None:
                self._synthesizer = self._synthesizer_factory(self.config)
            return self._synthesizer

    @property
    def diarizer(self) -> Any:
        with self._model_lock:
            if self._diarizer is None:
                self._diarizer = self._diarizer_factory(self.config)
            return self._diarizer


def _dispatch(state: _WorkerState, op: str, kwargs: dict) -> Any:
    """Execute a non-streaming op and return its result payload."""
    if op == "stt.transcribe_file":
        value = state.transcriber.transcribe_file(kwargs["path"])
    elif op == "stt.transcribe_file_with_segments":
        value = state.transcriber.transcribe_file_with_segments(kwargs["path"])
    elif op == "stt.transcribe_audio":
        value = state.transcriber.transcribe_audio(kwargs["audio"], kwargs["sample_rate"])
    elif op == "stt.transcribe_audio_with_segments":
        value = state.transcriber.transcribe_audio_with_segments(
            kwargs["audio"], kwargs["sample_rate"]
        )
    elif op == "stt.transcribe_audio_with_words":
        value = state.transcriber.transcribe_audio_with_words(
            kwargs["audio"], kwargs["sample_rate"]
        )
    elif op == "stt.transcribe_partial":
        value = state.transcriber.transcribe_partial(kwargs["audio"])
    elif op == "stt.warmup":
        state.transcriber.warmup()
        value = None
    elif op == "stt.device":
        value = None
    elif op == "tts.synthesize":
        return state.synthesizer.synthesize(
            kwargs["text"], voice=kwargs.get("voice"), speed=kwargs.get("speed")
        )
    elif op == "tts.get_status":
        return state.synthesizer.get_status()
    elif op == "diar.diarize_and_match":
        return state.diarizer.diarize_and_match(
            kwargs["audio"], kwargs["sample_rate"], kwargs["profiles"], kwargs["num_speakers"]
        )
    elif op == "diar.identify_segments":
        return state.diarizer.identify_segments(
            kwargs["audio"], kwargs["sample_rate"], kwargs["segments"]
        )
    elif op == "diar.embed":
        return state.diarizer.embed(kwargs["audio"], kwargs["sample_rate"])
    else:
        raise ValueError(f"Unknown worker op: {op}")

    return {"value": value, "device": state.transcriber.device}


def _run_request(state: _WorkerState, req_id: int, op: str, kwargs: dict) -> None:
    try:
        if op in STREAMING_OPS:
            for chunk in state.synthesizer.synthesize_streaming(
                kwargs["text"], voice=kwargs.get("voice"), speed=kwargs.get("speed")
            ):
                state.send(("chunk", req_id, chunk))
            state.send(("end", req_id))
        else:
            state.send(("result", req_id, _dispatch(state, op, kwargs)))
    except Exception as exc:
        logger.exception(f"Worker op {op} failed")
        try:
            state.send(("error", req_id, _describe_exception(exc)))
        except OSError:
            logger.warning("Parent connection lost while reporting error")


def worker_main(
    conn: Connection,
    config: Config,
    transcriber_factory=None,
    synthesizer_factory=None,
    diarizer_factory=None,
) -> None:
    """Entry point of the inference worker process.

    Blocks reading requests from ``conn`` until a ("shutdown",) message
    arrives or the parent side of the pipe closes. Each request runs on its
    own daemon thread so concurrent operations (e.g. an in-flight TTS stream
    plus an STT partial) behave like they did in-process.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [voiced-worker] %(levelname)s %(name)s: %(message)s",
    )
    # The parent drives shutdown through the protocol (then SIGTERM/SIGKILL
    # as escalation); a terminal Ctrl+C must not race that sequence.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    state = _WorkerState(
        conn,
        config,
        transcriber_factory or default_transcriber_factory,
        synthesizer_factory or default_synthesizer_factory,
        diarizer_factory or default_diarizer_factory,
    )
    conn.send(("ready",))
    logger.info("Inference worker ready")

    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            logger.info("Parent connection closed; exiting")
            break

        if message[0] == "shutdown":
            logger.info("Shutdown requested; exiting")
            break
        if message[0] == "request":
            _, req_id, op, kwargs = message
            threading.Thread(
                target=_run_request,
                args=(state, req_id, op, kwargs),
                daemon=True,
                name=f"worker-op-{req_id}",
            ).start()
        else:
            logger.warning(f"Unknown message from parent: {message[0]!r}")

    conn.close()
