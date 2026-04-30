"""RecordingSession — owns the Recording Session state machine.

One Recording Session is the toggle-to-toggle cycle: idle → recording →
transcribing → idle. Sequential only — a toggle during transcribing is
rejected. Results are emitted via callbacks; the daemon is the dispatcher
that wires those callbacks to clipboard, history, tray, etc.
"""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from queue import Queue

import numpy as np

from voiced.config import AudioConfig
from voiced.recorder import Recorder
from voiced.transcriber import Transcriber

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Where a Recording Session is in its lifecycle."""

    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"


class ToggleOutcomeKind(Enum):
    """What happened on the most recent toggle."""

    STARTED = "started"
    STOPPED = "stopped"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ToggleOutcome:
    """Result of a toggle() call. Carries the resulting state and (if rejected) why."""

    kind: ToggleOutcomeKind
    state: SessionState
    reason: str | None = None

    @classmethod
    def started(cls) -> "ToggleOutcome":
        return cls(kind=ToggleOutcomeKind.STARTED, state=SessionState.RECORDING)

    @classmethod
    def stopped(cls) -> "ToggleOutcome":
        return cls(kind=ToggleOutcomeKind.STOPPED, state=SessionState.TRANSCRIBING)

    @classmethod
    def rejected(cls, current: SessionState, reason: str) -> "ToggleOutcome":
        return cls(kind=ToggleOutcomeKind.REJECTED, state=current, reason=reason)


@dataclass(frozen=True)
class TranscriptionResult:
    """Output of a completed Recording Session.

    Carries the text and timing only — not the raw audio. If a future
    feature needs the audio, add a separate accessor; storing it here
    inflates memory for retained results (history etc.).
    """

    text: str
    duration: float
    started_at: datetime


# Internal event types pushed onto the dispatch queue. Tagged tuples keep
# the queue typed without introducing a class hierarchy for two events.
_StateEvent = tuple[str, SessionState, SessionState]  # ("state", old, new)
_ResultEvent = tuple[str, TranscriptionResult]  # ("result", result)
_ErrorEvent = tuple[str, BaseException]  # ("error", exc)
_StopEvent = tuple[str]  # ("stop",) — dispatch loop sentinel


StateChangeCallback = Callable[[SessionState, SessionState], None]
ResultCallback = Callable[[TranscriptionResult], None]
ErrorCallback = Callable[[BaseException], None]


class RecordingSession:
    """Sequential record-then-transcribe state machine with a dispatch thread.

    The state machine: ``IDLE → RECORDING → TRANSCRIBING → IDLE``. Toggling
    during ``TRANSCRIBING`` returns ``ToggleOutcome.rejected`` instead of
    queueing — that's the load-bearing sequential decision.

    Callbacks fire on a single dispatch thread owned by this module, so
    subscribers (tray, history, clipboard injector) don't need to be
    thread-safe with respect to recorder/worker threads.
    """

    def __init__(
        self,
        transcriber: Transcriber,
        audio_config: AudioConfig,
        sample_rate: int,
    ):
        self._transcriber = transcriber
        self._audio_config = audio_config
        self._sample_rate = sample_rate

        self._state = SessionState.IDLE
        self._state_lock = threading.RLock()
        self._shutdown_flag = False

        self._recorder: Recorder | None = None
        self._worker: threading.Thread | None = None

        self._dispatch_queue: Queue = Queue()
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            daemon=True,
            name="RecordingSessionDispatch",
        )
        self._dispatch_thread.start()

        self._on_state_change_cbs: list[StateChangeCallback] = []
        self._on_result_cbs: list[ResultCallback] = []
        self._on_error_cbs: list[ErrorCallback] = []

    @property
    def state(self) -> SessionState:
        with self._state_lock:
            return self._state

    def on_state_change(self, cb: StateChangeCallback) -> None:
        self._on_state_change_cbs.append(cb)

    def on_result(self, cb: ResultCallback) -> None:
        self._on_result_cbs.append(cb)

    def on_error(self, cb: ErrorCallback) -> None:
        self._on_error_cbs.append(cb)

    def toggle(self) -> ToggleOutcome:
        """Advance the state machine. Returns a ToggleOutcome describing what happened."""
        with self._state_lock:
            if self._shutdown_flag:
                return ToggleOutcome.rejected(self._state, "session has been shut down")

            if self._state == SessionState.IDLE:
                return self._start_recording_locked()
            if self._state == SessionState.RECORDING:
                return self._stop_recording_locked()
            return ToggleOutcome.rejected(self._state, "transcription in progress")

    def shutdown(self, transcribe_timeout: float = 30.0) -> None:
        """Stop accepting toggles. Cancel recording immediately; wait for in-flight
        transcription up to ``transcribe_timeout`` seconds before forcing exit.
        """
        with self._state_lock:
            self._shutdown_flag = True
            recorder = self._recorder
            self._recorder = None
            if self._state == SessionState.RECORDING:
                old = self._state
                self._state = SessionState.IDLE
                self._enqueue_state_change(old, self._state)
            worker = self._worker

        # Stop the recorder outside the lock — its .stop() may block briefly.
        if recorder is not None and recorder.is_recording:
            try:
                recorder.stop()
            except Exception:
                logger.exception("Error stopping recorder during shutdown")

        # Wait for an in-flight transcription to finish (caller's text matters).
        if worker is not None and worker.is_alive():
            worker.join(timeout=transcribe_timeout)
            if worker.is_alive():
                logger.warning(f"Transcription worker did not finish within {transcribe_timeout}s")

        # Stop the dispatch thread.
        self._dispatch_queue.put(("stop",))
        self._dispatch_thread.join(timeout=2.0)

    # ----- locked transitions -----

    def _start_recording_locked(self) -> ToggleOutcome:
        try:
            self._recorder = Recorder(config=self._audio_config)
            self._recorder.start()
        except Exception as e:
            self._recorder = None
            self._enqueue_error(e)
            return ToggleOutcome.rejected(self._state, f"recorder failed to start: {e}")

        old = self._state
        self._state = SessionState.RECORDING
        self._enqueue_state_change(old, self._state)
        return ToggleOutcome.started()

    def _stop_recording_locked(self) -> ToggleOutcome:
        recorder = self._recorder
        self._recorder = None

        try:
            audio_data = recorder.stop() if recorder else np.array([], dtype=np.float32)
        except Exception as e:
            old = self._state
            self._state = SessionState.IDLE
            self._enqueue_state_change(old, self._state)
            self._enqueue_error(e)
            return ToggleOutcome.rejected(self._state, f"recorder failed to stop: {e}")

        if len(audio_data) == 0:
            old = self._state
            self._state = SessionState.IDLE
            self._enqueue_state_change(old, self._state)
            self._enqueue_error(RuntimeError("no audio recorded"))
            # Surface the empty-recording case via on_error rather than a
            # successful Stopped — callers that rely on a result wouldn't
            # get one anyway.
            return ToggleOutcome.rejected(self._state, "no audio recorded")

        old = self._state
        self._state = SessionState.TRANSCRIBING
        self._enqueue_state_change(old, self._state)

        started_at = datetime.now()
        self._worker = threading.Thread(
            target=self._transcribe,
            args=(audio_data, started_at),
            daemon=True,
            name="RecordingSessionWorker",
        )
        self._worker.start()
        return ToggleOutcome.stopped()

    # ----- worker thread -----

    def _transcribe(self, audio_data: np.ndarray, started_at: datetime) -> None:
        try:
            text = self._transcriber.transcribe_audio(
                audio_data,
                sample_rate=self._sample_rate,
            )
            duration = len(audio_data) / self._sample_rate if self._sample_rate else 0.0
            result = TranscriptionResult(
                text=text,
                duration=duration,
                started_at=started_at,
            )
            self._enqueue_result(result)
        except Exception as e:
            logger.exception("Transcription failed")
            self._enqueue_error(e)
        finally:
            with self._state_lock:
                old = self._state
                self._state = SessionState.IDLE
                self._worker = None
                if old != SessionState.IDLE:
                    self._enqueue_state_change(old, self._state)

    # ----- dispatch thread -----

    def _enqueue_state_change(self, old: SessionState, new: SessionState) -> None:
        self._dispatch_queue.put(("state", old, new))

    def _enqueue_result(self, result: TranscriptionResult) -> None:
        self._dispatch_queue.put(("result", result))

    def _enqueue_error(self, exc: BaseException) -> None:
        self._dispatch_queue.put(("error", exc))

    def _dispatch_loop(self) -> None:
        while True:
            event = self._dispatch_queue.get()
            kind = event[0]
            if kind == "stop":
                return
            try:
                if kind == "state":
                    _, old, new = event
                    for cb in list(self._on_state_change_cbs):
                        try:
                            cb(old, new)
                        except Exception:
                            logger.exception("on_state_change callback raised")
                elif kind == "result":
                    _, result = event
                    for cb in list(self._on_result_cbs):
                        try:
                            cb(result)
                        except Exception:
                            logger.exception("on_result callback raised")
                elif kind == "error":
                    _, exc = event
                    for cb in list(self._on_error_cbs):
                        try:
                            cb(exc)
                        except Exception:
                            logger.exception("on_error callback raised")
            except Exception:
                logger.exception("RecordingSession dispatch loop error")
