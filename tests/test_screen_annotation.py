"""Deterministic tests for screen annotation export and cancellation."""

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from voiced import screen_annotate as annotation


def _png_bytes(image: Image.Image) -> bytes:
    from io import BytesIO

    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_export_extracts_and_annotates_video_frames(tmp_path: Path, monkeypatch):
    """Export asks ffmpeg for stroke-end frames and burns strokes onto them."""

    baseline = Image.new("RGB", (32, 24), (10, 20, 30))
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=_png_bytes(baseline), stderr=b"")

    monkeypatch.setattr(annotation.subprocess, "run", run)
    session = annotation._OverlaySession((0, 0, 32, 24))
    session.points.extend(
        [
            {"t": 10.0, "x": 4, "y": 5, "action": "start"},
            {"t": 10.2, "x": 12, "y": 5, "action": "draw"},
            {"t": 10.5, "x": 20, "y": 5, "action": "end"},
            {"t": 11.0, "x": 8, "y": 15, "action": "start"},
            {"t": 11.25, "x": 8, "y": 15, "action": "end"},
        ]
    )
    keyframes = tmp_path / "clip.keyframes.json"
    video = tmp_path / "clip.mp4"

    session.export(keyframes, video, "ffmpeg-test", 10.0, 32, 24)

    data = json.loads(keyframes.read_text(encoding="utf-8"))
    assert data["width"] == 32
    assert data["height"] == 24
    assert [point["t"] for point in data["keyframes"]] == pytest.approx([0, 0.2, 0.5, 1.0, 1.25])
    assert len(calls) == 2
    assert [call[0][call[0].index("-ss") + 1] for call in calls] == ["0.5", "1.25"]
    assert all(call[0][0] == "ffmpeg-test" and call[1]["capture_output"] for call in calls)

    first = Image.open(tmp_path / "clip.keyframes-0000.500.png").convert("RGB")
    second = Image.open(tmp_path / "clip.keyframes-0001.250.png").convert("RGB")
    assert first.getpixel((12, 5)) == (255, 56, 20)
    assert first.getpixel((8, 15)) == (10, 20, 30)
    assert second.getpixel((8, 15)) == (255, 56, 20)
    assert second.getpixel((30, 22)) == (10, 20, 30)


def test_export_reports_frame_extraction_failure(tmp_path: Path, monkeypatch):
    def run(_command, **_kwargs):
        return SimpleNamespace(returncode=1, stdout=b"", stderr=b"decoder failed")

    monkeypatch.setattr(annotation.subprocess, "run", run)
    session = annotation._OverlaySession((0, 0, 10, 10))
    session.points.extend(
        [
            {"t": 1.0, "x": 1, "y": 1, "action": "start"},
            {"t": 1.1, "x": 2, "y": 2, "action": "end"},
        ]
    )

    with pytest.raises(annotation.ScreenAnnotationError, match="decoder failed"):
        session.export(tmp_path / "clip.keyframes.json", tmp_path / "clip.mp4", "ffmpeg", 1, 10, 10)


def test_run_annotation_cancel_removes_partial_outputs_and_finishes_overlay(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("WAYLAND_DISPLAY", "test-wayland")
    monkeypatch.setattr(annotation, "_require_program", lambda name: name)
    monkeypatch.setattr(
        annotation.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="0,0 16x12\n", stderr=""),
    )

    class FakeProcess:
        returncode = None
        terminated = False

        def poll(self):
            return None if not self.terminated else 0

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

    process = FakeProcess()
    monkeypatch.setattr(annotation.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(annotation.time, "sleep", lambda _duration: None)
    events = []

    class FakeOverlay:
        def __init__(self, geometry):
            self.geometry = geometry
            self.finished = False

        def start(self, _gtk, _glib):
            events.append("started")

        def finish(self):
            self.finished = True
            events.append("finished")

    monkeypatch.setattr(annotation, "_OverlaySession", FakeOverlay)
    cancel = threading.Event()
    cancel.set()

    result = annotation.run_annotation(tmp_path, cancel_event=cancel, gtk=object(), glib=object())
    assert result is None
    assert process.terminated
    assert events == ["started", "finished"]
    assert list(tmp_path.iterdir()) == []
