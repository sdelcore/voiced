"""Live capture of microphone and system-loopback audio for meetings.

System audio can use a PortAudio input or a native PulseAudio/PipeWire source
captured by parec or pw-record.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class MeetingCaptureError(RuntimeError):
    """A capture source failed to start or report usable audio."""


@dataclass(frozen=True)
class MeetingRecording:
    """A stopped meeting recording, retaining independent source tracks."""

    microphone: np.ndarray | None
    system_audio: np.ndarray | None
    sample_rate: int
    started_at: datetime
    duration: float

    def transcribe(
        self,
        transcriber: Any,
        *,
        identify_speakers: bool = True,
        num_speakers: int | None = None,
    ) -> MeetingTranscript:
        """Transcribe each available track independently via the adapter."""
        results: dict[str, Any] = {}
        errors: dict[str, BaseException] = {}
        for name, audio in (("microphone", self.microphone), ("system_audio", self.system_audio)):
            if audio is None or not audio.size:
                continue
            try:
                results[name] = transcriber.transcribe(
                    audio,
                    self.sample_rate,
                    identify_speakers=identify_speakers,
                    num_speakers=num_speakers,
                )
            except Exception as exc:
                errors[name] = exc
                logger.exception("Meeting %s transcription failed", name)
        return MeetingTranscript(
            microphone=results.get("microphone"),
            system_audio=results.get("system_audio"),
            sample_rate=self.sample_rate,
            duration=self.duration,
            errors=errors,
        )


@dataclass(frozen=True)
class MeetingTranscript:
    """Per-track outputs and failures from a local or remote adapter."""

    microphone: Any | None
    system_audio: Any | None
    sample_rate: int
    duration: float
    errors: dict[str, BaseException] | None = None


def list_audio_sources() -> list[dict[str, Any]]:
    """Return selectable PortAudio inputs and native Pulse/PipeWire monitors."""
    monitor_names = _monitor_source_names()
    result: list[dict[str, Any]] = []
    known_monitors: set[str] = set()
    for index, raw in enumerate(sd.query_devices()):
        channels = int(raw.get("max_input_channels", 0))
        if channels <= 0:
            continue
        name = str(raw["name"])
        is_monitor = any(token.casefold() in name.casefold() for token in monitor_names)
        if is_monitor:
            known_monitors.update(
                source for source in monitor_names if source.casefold() in name.casefold()
            )
        result.append(
            {
                "index": index,
                "name": name,
                "channels": channels,
                "sample_rate": float(raw["default_samplerate"]),
                "kind": "system" if is_monitor else "microphone",
                "is_monitor": is_monitor,
            }
        )
    native_backend = (
        "pulse" if shutil.which("parec") else ("pipewire" if shutil.which("pw-record") else None)
    )
    if native_backend:
        result.extend(
            {
                "index": None,
                "name": source,
                "channels": 2,
                "sample_rate": 16000,
                "kind": "system",
                "is_monitor": True,
                "backend": native_backend,
            }
            for source in sorted(monitor_names - known_monitors)
        )
    return result


def _monitor_source_names() -> set[str]:
    """Best-effort discover native PulseAudio/PipeWire monitor names."""
    if not shutil.which("pactl"):
        return set()
    try:
        completed = subprocess.run(
            ["pactl", "list", "short", "sources"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("Could not enumerate PulseAudio/PipeWire sources", exc_info=True)
        return set()
    return {
        fields[1]
        for line in completed.stdout.splitlines()
        if len(fields := line.split()) >= 2 and ".monitor" in fields[1].casefold()
    }


class MeetingCapture:
    """Capture selected microphone and system sources into separate tracks.

    PortAudio selections use integer indices. A native source can be selected
    using its source name or a dictionary returned by :func:`list_audio_sources`.
    """

    def __init__(
        self,
        microphone: int | str | dict[str, Any] | None = None,
        system_audio: int | str | dict[str, Any] | None = None,
        sample_rate: int = 16000,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.microphone = _device_index(microphone)
        native_source = _native_source(system_audio)
        if (
            isinstance(system_audio, str)
            and ".monitor" in system_audio.casefold()
            and native_source is None
        ):
            raise MeetingCaptureError(
                f"No native capture backend is available for {system_audio!r}"
            )
        if (
            isinstance(system_audio, dict)
            and system_audio.get("is_monitor")
            and system_audio.get("backend") not in {"pulse", "pipewire"}
        ):
            system_audio = system_audio.get("index")
        self._native_source = native_source
        self.system_audio = None if native_source else _device_index(system_audio)
        if self.microphone is None and self.system_audio is None and self._native_source is None:
            raise ValueError("Select at least one microphone or system-audio source")
        self.sample_rate = sample_rate
        self._streams: dict[str, sd.InputStream] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._reader_threads: dict[str, threading.Thread] = {}
        self._chunks: dict[str, list[np.ndarray]] = {}
        self._errors: dict[str, BaseException] = {}
        self._lock = threading.Lock()
        self._running = False
        self._started_at: datetime | None = None
        self._started_monotonic = 0.0

    @property
    def is_recording(self) -> bool:
        return self._running

    def start(self) -> None:
        """Open selected capture streams; rolls back partial setup on failure."""
        if self._running:
            raise MeetingCaptureError("Meeting capture is already running")
        self._chunks = {
            name: []
            for name, device in (
                ("microphone", self.microphone),
                ("system_audio", self.system_audio),
            )
            if device is not None or (name == "system_audio" and self._native_source)
        }
        self._errors.clear()
        self._running = True
        self._started_at = datetime.now(timezone.utc)
        self._started_monotonic = time.monotonic()
        try:
            sources = (("microphone", self.microphone), ("system_audio", self.system_audio))
            for name, device in sources:
                if device is None or (name == "system_audio" and self._native_source):
                    continue
                channels = _input_channels(device)
                stream = sd.InputStream(
                    device=device,
                    channels=channels,
                    samplerate=self.sample_rate,
                    dtype="float32",
                    callback=self._callback(name),
                )
                self._streams[name] = stream
                stream.start()
            if self._native_source:
                self._start_native_capture(self._native_source)
        except Exception as exc:
            cleanup_error = self._close_streams()
            self._running = False
            if cleanup_error:
                raise MeetingCaptureError(
                    f"Could not start meeting audio capture: {exc}; cleanup failed: {cleanup_error}"
                ) from exc
            raise MeetingCaptureError(f"Could not start meeting audio capture: {exc}") from exc

    def _start_native_capture(self, source: tuple[str, str]) -> None:
        backend, source_name = source
        if backend == "pulse":
            command = [
                "parec",
                "--device",
                source_name,
                "--format=float32le",
                f"--rate={self.sample_rate}",
                "--channels=2",
                "--raw",
            ]
        else:
            command = [
                "pw-record",
                "--target",
                source_name,
                "--format",
                "f32",
                "--rate",
                str(self.sample_rate),
                "--channels",
                "2",
                "-",
            ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if process.stdout is None:
            process.kill()
            process.wait()
            raise MeetingCaptureError("native monitor did not provide an audio stream")
        self._processes["system_audio"] = process
        thread = threading.Thread(target=self._read_native_audio, args=(process,), daemon=True)
        self._reader_threads["system_audio"] = thread
        thread.start()

    def _read_native_audio(self, process: subprocess.Popen) -> None:
        try:
            assert process.stdout is not None
            while self._running:
                data = process.stdout.read(4096 * 2 * 4)
                if not data:
                    break
                usable = len(data) - len(data) % 8
                if usable:
                    chunk = (
                        np.frombuffer(data[:usable], dtype="<f4")
                        .reshape(-1, 2)
                        .mean(axis=1, dtype=np.float32)
                        .copy()
                    )
                    with self._lock:
                        self._chunks["system_audio"].append(chunk)
            if self._running:
                returncode = process.poll()
                if returncode is None:
                    raise MeetingCaptureError("native monitor stream ended unexpectedly")
                if returncode:
                    raise MeetingCaptureError(f"native monitor exited with status {returncode}")
        except BaseException as exc:
            with self._lock:
                self._errors["system_audio"] = exc
            logger.error("Meeting system audio reader failed: %s", exc)

    def _callback(self, name: str):
        def capture(indata, frames, time_info, status) -> None:
            if status:
                logger.warning("Meeting %s capture status: %s", name, status)
            if not self._running:
                return
            try:
                chunk = np.asarray(indata, dtype=np.float32).copy()
                if chunk.ndim > 1:
                    chunk = chunk.mean(axis=1, dtype=np.float32)
                with self._lock:
                    self._chunks[name].append(chunk)
            except BaseException as exc:
                with self._lock:
                    self._errors[name] = exc
                logger.exception("Meeting %s audio callback failed", name)

        return capture

    def stop(self) -> MeetingRecording:
        """Stop capture and return distinct float32 mono tracks."""
        if not self._running:
            raise MeetingCaptureError("Meeting capture is not running")
        self._running = False
        close_error = self._close_streams()
        with self._lock:
            errors = dict(self._errors)
            chunks = {name: list(parts) for name, parts in self._chunks.items()}
        if errors or close_error:
            detail = "; ".join(f"{name}: {error}" for name, error in errors.items())
            if close_error:
                detail = f"{detail}; {close_error}" if detail else str(close_error)
            raise MeetingCaptureError(f"Meeting audio capture failed ({detail})")
        tracks: dict[str, np.ndarray | None] = {}
        for name, parts in chunks.items():
            tracks[name] = np.concatenate(parts).astype(np.float32, copy=False) if parts else None
        if all(track is None or track.size == 0 for track in tracks.values()):
            raise MeetingCaptureError("No audio was captured from the selected sources")
        started = self._started_at or datetime.now(timezone.utc)
        duration = max(0.0, time.monotonic() - self._started_monotonic)
        return MeetingRecording(
            microphone=tracks.get("microphone"),
            system_audio=tracks.get("system_audio"),
            sample_rate=self.sample_rate,
            started_at=started,
            duration=duration,
        )

    def _close_streams(self) -> BaseException | None:
        first_error = None
        streams, self._streams = self._streams, {}
        for stream in streams.values():
            try:
                if stream.active:
                    stream.stop()
            except BaseException as exc:
                first_error = first_error or exc
            try:
                stream.close()
            except BaseException as exc:
                first_error = first_error or exc
        processes, self._processes = self._processes, {}
        terminated: set[int] = set()
        for process in processes.values():
            try:
                process.terminate()
                terminated.add(id(process))
            except ProcessLookupError:
                pass
            except BaseException as exc:
                first_error = first_error or exc
        for process in processes.values():
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                try:
                    process.wait()
                except BaseException as exc:
                    first_error = first_error or exc
            except BaseException as exc:
                first_error = first_error or exc
        threads, self._reader_threads = self._reader_threads, {}
        for thread in threads.values():
            thread.join(timeout=2.0)
            if thread.is_alive():
                first_error = first_error or MeetingCaptureError(
                    "native system-audio reader did not stop"
                )
        for process in processes.values():
            if process.stdout:
                try:
                    process.stdout.close()
                except BaseException as exc:
                    first_error = first_error or exc
        for name, process in processes.items():
            if process.returncode and id(process) not in terminated:
                first_error = first_error or MeetingCaptureError(
                    f"{name} process exited with status {process.returncode}"
                )
        return first_error

    def close(self) -> None:
        """Idempotently stop streams without retaining their audio."""
        self._running = False
        close_error = self._close_streams()
        self._chunks.clear()
        if close_error:
            logger.warning("Meeting capture cleanup failed: %s", close_error)

    def __enter__(self) -> MeetingCapture:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _native_source(device: int | str | dict[str, Any] | None) -> tuple[str, str] | None:
    if isinstance(device, str) and device not in {"default"}:
        if ".monitor" not in device.casefold():
            return None
        if shutil.which("parec"):
            return "pulse", device
        if shutil.which("pw-record"):
            return "pipewire", device
        return None
    if not isinstance(device, dict):
        return None
    backend = device.get("backend")
    name = device.get("name")
    if backend in {"pulse", "pipewire"} and isinstance(name, str):
        return backend, name
    return None


def _device_index(device: int | str | dict[str, Any] | None) -> int | None:
    if device is None:
        return None
    if isinstance(device, dict):
        if device.get("backend") in {"pulse", "pipewire"}:
            return None
        device = device.get("index")
    if device == "default":
        return None
    if not isinstance(device, int):
        raise ValueError(f"Invalid audio device selection: {device!r}")
    return device


def _input_channels(device: int | None) -> int:
    try:
        info = sd.query_devices(device, "input")
        channels = int(info["max_input_channels"])
    except Exception as exc:
        raise MeetingCaptureError(f"Could not query input device {device!r}: {exc}") from exc
    if channels < 1:
        raise MeetingCaptureError(f"Audio device {device!r} has no input channels")
    return min(channels, 2)
