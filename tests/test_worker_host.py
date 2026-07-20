"""Tests for the inference worker host, proxies, and process lifecycle.

These use fake worker backends (injected factories, run in a real spawned
child process) — no GPU, no torch, no model downloads.
"""

import multiprocessing
import os
import signal
import threading
import time
from pathlib import Path

import numpy as np
import pytest

# For the forced-termination test the child must ignore SIGTERM, and signal
# handlers can only be installed from a process's main thread — which for the
# worker means at module import (during spawn unpickling), not inside a
# request thread. The env var scopes this to that one test's child processes.
if (
    os.environ.get("VOICED_TEST_IGNORE_SIGTERM")
    and multiprocessing.current_process().name == "voiced-inference-worker"
):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

from voiced.config import Config
from voiced.speaker_segments import IdentifiedSegment
from voiced.worker_host import (
    WorkerCrashError,
    WorkerDiarizer,
    WorkerHost,
    WorkerOperationError,
    WorkerSpeakerEmbedder,
    WorkerSynthesizer,
    WorkerTranscriber,
)

# ----- fake backends (module-level so spawn can pickle them by reference) -----


class CustomBackendError(Exception):
    pass


class FakeTranscriber:
    device = "cpu"

    def warmup(self):
        pass

    def transcribe_file(self, path):
        path = str(path)
        if path == "/crash":
            os._exit(1)
        if path == "/boom":
            raise ValueError("boom")
        if path == "/missing":
            raise FileNotFoundError("missing")
        if path == "/custom":
            raise CustomBackendError("custom failure")
        if path == "/pid":
            return str(os.getpid())
        return "file-ok"

    def transcribe_file_with_segments(self, path):
        return [(0.0, 1.0, "hello")]

    def transcribe_audio(self, audio, sample_rate):
        return f"text:{len(audio)}@{sample_rate}"

    def transcribe_audio_with_segments(self, audio, sample_rate):
        return [(0.0, 0.5, "hello"), (0.5, 1.0, "world")]

    def transcribe_audio_with_words(self, audio, sample_rate):
        return ("hi there", [("hi", 0.0, 0.4, 1.0), ("there", 0.5, 0.9, 1.0)])

    def transcribe_partial(self, audio):
        return "partial"


class FakeSynthesizer:
    def synthesize(self, text, voice=None, speed=None):
        return np.ones(240, dtype=np.float32)

    def synthesize_streaming(self, text, voice=None, speed=None):
        if text.startswith("gate:"):
            gate = Path(text.removeprefix("gate:"))
            yield np.full(10, 1.0, dtype=np.float32)
            deadline = time.monotonic() + 10.0
            while not gate.exists():
                if time.monotonic() > deadline:
                    raise RuntimeError("gate file never appeared")
                time.sleep(0.01)
            yield np.full(10, 2.0, dtype=np.float32)
        elif text == "boom-mid-stream":
            yield np.full(10, 1.0, dtype=np.float32)
            raise ValueError("stream boom")
        else:
            yield np.full(10, 1.0, dtype=np.float32)
            yield np.full(10, 2.0, dtype=np.float32)

    def get_status(self):
        return {"model_loaded": True, "model": "fake"}


class FakeDiarBackend:
    def diarize_and_match(self, audio, sample_rate, profiles, num_speakers):
        speaker = profiles[0].name if profiles else "SPEAKER_00"
        return [IdentifiedSegment(0.0, 1.0, "", speaker, 0.9 if profiles else 0.0)]

    def identify_segments(self, audio, sample_rate, segments):
        return [IdentifiedSegment(start, end, text, "alice", 0.8) for start, end, text in segments]

    def embed(self, audio, sample_rate):
        return np.full(192, float(len(audio)), dtype=np.float32)


def fake_transcriber_factory(config):
    return FakeTranscriber()


def fake_synthesizer_factory(config):
    return FakeSynthesizer()


def fake_diarizer_factory(config):
    return FakeDiarBackend()


def stubborn_transcriber_factory(config):
    """A backend that blocks graceful exit with a non-daemon thread.

    Combined with VOICED_TEST_IGNORE_SIGTERM (see module top), the worker
    survives both the protocol shutdown and terminate(), forcing kill().
    """
    threading.Thread(target=time.sleep, args=(300,), daemon=False).start()
    return FakeTranscriber()


# ----- helpers -----


def make_host(idle_timeout=None, transcriber_factory=None, **kwargs) -> WorkerHost:
    return WorkerHost(
        Config(),
        idle_timeout=idle_timeout,
        transcriber_factory=transcriber_factory or fake_transcriber_factory,
        synthesizer_factory=fake_synthesizer_factory,
        diarizer_factory=fake_diarizer_factory,
        start_timeout=30.0,
        **kwargs,
    )


def wait_until(predicate, timeout=10.0, interval=0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def pid_gone(pid: int) -> bool:
    """True once the process no longer exists (i.e. it was reaped, not zombied)."""
    try:
        os.kill(pid, 0)
        return False
    except ProcessLookupError:
        return True


@pytest.fixture
def host():
    h = make_host()
    yield h
    h.shutdown()


# ----- lazy startup -----


class TestLazyStartup:
    def test_no_worker_at_construction(self):
        h = make_host()
        try:
            assert not h.is_running
        finally:
            h.shutdown()

    def test_proxies_do_not_start_worker(self):
        h = make_host()
        try:
            WorkerTranscriber(h)
            WorkerSynthesizer(h, Config())
            assert not h.is_running
        finally:
            h.shutdown()

    def test_first_request_starts_worker(self, host):
        transcriber = WorkerTranscriber(host)
        assert transcriber.transcribe_file("/anything") == "file-ok"
        assert host.is_running


# ----- request dispatch and model operations -----


class TestOperations:
    def test_transcribe_audio(self, host):
        transcriber = WorkerTranscriber(host)
        audio = np.zeros(16000, dtype=np.float32)
        assert transcriber.transcribe_audio(audio, 16000) == "text:16000@16000"

    def test_transcribe_audio_with_segments(self, host):
        transcriber = WorkerTranscriber(host)
        audio = np.zeros(100, dtype=np.float32)
        segments = transcriber.transcribe_audio_with_segments(audio, 16000)
        assert segments == [(0.0, 0.5, "hello"), (0.5, 1.0, "world")]

    def test_transcribe_audio_with_words(self, host):
        transcriber = WorkerTranscriber(host)
        audio = np.zeros(100, dtype=np.float32)
        text, words = transcriber.transcribe_audio_with_words(audio, 16000)
        assert text == "hi there"
        assert words == [("hi", 0.0, 0.4, 1.0), ("there", 0.5, 0.9, 1.0)]

    def test_transcribe_partial(self, host):
        transcriber = WorkerTranscriber(host)
        assert transcriber.transcribe_partial(np.zeros(10, dtype=np.float32)) == "partial"

    def test_concurrent_requests(self, host):
        transcriber = WorkerTranscriber(host)
        results = []

        def call():
            results.append(transcriber.transcribe_file("/anything"))

        threads = [threading.Thread(target=call) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert results == ["file-ok"] * 4

    def test_device_resolved_after_first_op(self, host):
        transcriber = WorkerTranscriber(host)
        assert transcriber.device == "auto"  # config string before any op
        transcriber.warmup()
        assert transcriber.device == "cpu"

    def test_batch_tts(self, host):
        synth = WorkerSynthesizer(host, Config())
        audio = synth.synthesize("hello")
        assert audio.dtype == np.float32
        assert len(audio) == 240

    def test_tts_is_loaded_tracks_usage(self, host):
        synth = WorkerSynthesizer(host, Config())
        assert not synth.is_loaded
        synth.synthesize("hello")
        assert synth.is_loaded

    def test_tts_status_without_worker(self):
        h = make_host()
        try:
            synth = WorkerSynthesizer(h, Config())
            status = synth.get_status()
            assert status["model_loaded"] is False
            assert not h.is_running  # status must not wake the worker
        finally:
            h.shutdown()


class TestDiarization:
    def test_identify_segments_through_worker(self, host):
        diar = WorkerDiarizer(host)
        segments = diar.identify_segments_from_array(
            np.zeros(16000, dtype=np.float32), 16000, [(0.0, 1.0, "hello")]
        )
        assert segments == [IdentifiedSegment(0.0, 1.0, "hello", "alice", 0.8)]

    def test_diarize_and_match_through_worker(self, host):
        from voiced.profiles import VoiceProfile

        diar = WorkerDiarizer(host)
        profile = VoiceProfile(
            name="bob", embedding=[0.0] * 4, created_at="", audio_duration=1.0, model_version=""
        )
        segments = diar.diarize_and_match_profiles_from_array(
            np.zeros(16000, dtype=np.float32), 16000, profiles=[profile]
        )
        assert segments[0].speaker == "bob"
        assert segments[0].confidence == 0.9

    def test_embedding_through_worker(self, host):
        embedder = WorkerSpeakerEmbedder(host)
        embedding = embedder.extract_embedding_from_array(np.zeros(100, dtype=np.float32), 16000)
        assert embedding.shape == (192,)
        assert embedding[0] == 100.0

    def test_voiced_transcribe_with_speakers_stays_in_worker(self, host):
        """End-to-end Voiced.transcribe: STT + diarization + alignment, with
        all model work behind the worker and alignment torch-free in the parent."""
        from unittest.mock import MagicMock

        from voiced.capabilities import Voiced

        store = MagicMock()
        store.list.return_value = []
        v = Voiced(
            config=Config(),
            transcriber=WorkerTranscriber(host),
            profile_store=store,
            worker_host=host,
        )
        out = v.transcribe(np.zeros(16000, dtype=np.float32), 16000, identify_speakers=True)
        assert out.text == "hello world"
        assert [s.speaker for s in out.segments] == ["SPEAKER_00", "SPEAKER_00"]


# ----- streaming -----


class TestStreaming:
    def test_stream_yields_chunks_in_order(self, host):
        synth = WorkerSynthesizer(host, Config())
        chunks = list(synth.synthesize_streaming("hello"))
        assert len(chunks) == 2
        assert chunks[0][0] == 1.0
        assert chunks[1][0] == 2.0

    def test_chunks_arrive_before_generation_completes(self, host, tmp_path):
        """First chunk is delivered while the worker is still blocked
        producing the second — the stream is not buffered end-to-end."""
        gate = tmp_path / "gate"
        synth = WorkerSynthesizer(host, Config())
        stream = synth.synthesize_streaming(f"gate:{gate}")

        first = next(stream)
        assert first[0] == 1.0
        assert not gate.exists()  # generation of chunk 2 hasn't been allowed yet

        gate.touch()
        rest = list(stream)
        assert len(rest) == 1
        assert rest[0][0] == 2.0

    def test_mid_stream_error_propagates(self, host):
        synth = WorkerSynthesizer(host, Config())
        stream = synth.synthesize_streaming("boom-mid-stream")
        assert next(stream)[0] == 1.0
        with pytest.raises(ValueError, match="stream boom"):
            next(stream)


# ----- idle lifecycle -----


class TestIdleLifecycle:
    def test_idle_timeout_stops_worker(self):
        h = make_host(idle_timeout=0.2)
        try:
            transcriber = WorkerTranscriber(h)
            pid = int(transcriber.transcribe_file("/pid"))
            assert wait_until(lambda: not h.is_running, timeout=5.0)
            assert wait_until(lambda: pid_gone(pid), timeout=5.0)
        finally:
            h.shutdown()

    def test_active_stream_blocks_idle_shutdown(self, tmp_path):
        h = make_host(idle_timeout=0.2)
        try:
            synth = WorkerSynthesizer(h, Config())
            gate = tmp_path / "gate"
            stream = synth.synthesize_streaming(f"gate:{gate}")
            next(stream)

            time.sleep(0.6)  # well past the idle timeout
            assert h.is_running  # lease held by the open stream

            gate.touch()
            list(stream)
            assert wait_until(lambda: not h.is_running, timeout=5.0)
        finally:
            h.shutdown()

    def test_restart_after_idle_shutdown(self):
        h = make_host(idle_timeout=0.2)
        try:
            transcriber = WorkerTranscriber(h)
            pid1 = int(transcriber.transcribe_file("/pid"))
            assert wait_until(lambda: not h.is_running, timeout=5.0)

            pid2 = int(transcriber.transcribe_file("/pid"))
            assert h.is_running
            assert pid1 != pid2
        finally:
            h.shutdown()


# ----- crash recovery -----


class TestCrashRecovery:
    def test_crash_fails_request_with_useful_error(self, host):
        transcriber = WorkerTranscriber(host)
        with pytest.raises(WorkerCrashError):
            transcriber.transcribe_file("/crash")

    def test_next_request_starts_fresh_worker(self, host):
        transcriber = WorkerTranscriber(host)
        with pytest.raises(WorkerCrashError):
            transcriber.transcribe_file("/crash")
        assert wait_until(lambda: not host.is_running, timeout=5.0)

        pid = int(transcriber.transcribe_file("/pid"))
        assert host.is_running
        assert not pid_gone(pid)


# ----- shutdown -----


class TestShutdown:
    def test_clean_shutdown_reaps_worker(self):
        h = make_host()
        transcriber = WorkerTranscriber(h)
        pid = int(transcriber.transcribe_file("/pid"))

        h.shutdown()
        assert not h.is_running
        assert pid_gone(pid)

    def test_use_after_shutdown_raises(self):
        h = make_host()
        h.shutdown()
        with pytest.raises(RuntimeError, match="shut down"):
            WorkerTranscriber(h).transcribe_file("/anything")

    def test_forced_termination_is_bounded(self, monkeypatch):
        monkeypatch.setenv("VOICED_TEST_IGNORE_SIGTERM", "1")
        h = make_host(
            transcriber_factory=stubborn_transcriber_factory,
            graceful_timeout=0.5,
            kill_timeout=1.0,
        )
        transcriber = WorkerTranscriber(h)
        pid = int(transcriber.transcribe_file("/pid"))

        start = time.monotonic()
        h.shutdown()
        elapsed = time.monotonic() - start

        assert elapsed < 5.0
        assert pid_gone(pid)


# ----- structured error propagation -----


class TestErrorPropagation:
    def test_builtin_exception_is_reconstructed(self, host):
        transcriber = WorkerTranscriber(host)
        with pytest.raises(ValueError, match="boom"):
            transcriber.transcribe_file("/boom")

    def test_file_not_found_is_preserved(self, host):
        transcriber = WorkerTranscriber(host)
        with pytest.raises(FileNotFoundError, match="missing"):
            transcriber.transcribe_file("/missing")

    def test_non_builtin_exception_is_wrapped(self, host):
        transcriber = WorkerTranscriber(host)
        with pytest.raises(WorkerOperationError) as excinfo:
            transcriber.transcribe_file("/custom")
        assert excinfo.value.original_type == "CustomBackendError"
        assert "custom failure" in str(excinfo.value)

    def test_worker_survives_failed_request(self, host):
        transcriber = WorkerTranscriber(host)
        with pytest.raises(ValueError):
            transcriber.transcribe_file("/boom")
        assert transcriber.transcribe_file("/anything") == "file-ok"
