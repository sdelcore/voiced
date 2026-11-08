"""Wayland screen capture, live drawing overlay, and transcript export."""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import gcd
from pathlib import Path

import numpy as np

from voiced.config import get_server_url, load_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnnotationResult:
    """Exported screen recording, transcript, and drawing keyframes."""

    video_path: Path
    transcript_path: Path
    started_at: datetime
    duration: float
    keyframes_path: Path | None = None


class ScreenAnnotationError(RuntimeError):
    """Screen capture or transcript export failed."""


@dataclass
class _OverlaySession:
    """Overlay widget lifecycle dispatched onto the desktop GTK main loop."""

    geometry: tuple[int, int, int, int]
    stop: threading.Event = field(default_factory=threading.Event)
    points: list[dict[str, object]] = field(default_factory=list)
    error: BaseException | None = None
    ready: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    gtk: object | None = None
    glib: object | None = None
    window: object | None = None

    def start(self, gtk: object | None = None, glib: object | None = None) -> None:
        if gtk is None or glib is None:
            try:
                import gi

                gi.require_version("Gtk", "3.0")
                gi.require_version("GtkLayerShell", "0.1")
                from gi.repository import GLib, Gtk
            except BaseException as exc:
                raise ScreenAnnotationError(
                    f"Could not initialize GTK drawing overlay: {exc}"
                ) from exc
            gtk, glib = Gtk, GLib
        self.gtk, self.glib = gtk, glib
        glib.idle_add(self._create_window)
        if not self.ready.wait(10):
            self.finish()
            raise ScreenAnnotationError("Timed out starting GTK drawing overlay")
        if self.error is not None:
            raise ScreenAnnotationError(
                f"Could not start GTK drawing overlay: {self.error}"
            ) from self.error

    def finish(self) -> None:
        self.stop.set()
        if self.glib is None:
            return
        self.glib.idle_add(self._destroy_window)
        self.finished.wait(2)

    def _create_window(self) -> bool:
        try:
            import gi

            gi.require_version("Gtk", "3.0")
            gi.require_version("GtkLayerShell", "0.1")
            from gi.repository import Gdk, GtkLayerShell

            self._build_window(self.gtk, Gdk, self.glib, GtkLayerShell)
        except BaseException as exc:
            self.error = exc
            self.ready.set()
            self.finished.set()

    def _window_destroyed(self) -> None:
        self.window = None
        self.finished.set()

    def _destroy_window(self) -> bool:
        if self.window is not None:
            self.window.destroy()
            self.window = None
        self.finished.set()
        return False

    def _build_window(
        self, gtk: object, gdk: object, glib: object, gtk_layer_shell: object
    ) -> None:
        try:
            x, y, width, height = self.geometry
            window = gtk.Window(type=gtk.WindowType.TOPLEVEL)
            self.window = window
            window.set_title("Voiced annotation overlay")
            window.set_decorated(False)
            window.set_app_paintable(True)
            window.set_keep_above(True)
            window.set_accept_focus(False)
            window.set_default_size(width, height)
            gtk_layer_shell.init_for_window(window)
            gtk_layer_shell.set_layer(window, gtk_layer_shell.Layer.OVERLAY)
            gtk_layer_shell.set_keyboard_mode(window, gtk_layer_shell.KeyboardMode.NONE)
            gtk_layer_shell.set_anchor(window, gtk_layer_shell.Edge.LEFT, True)
            gtk_layer_shell.set_anchor(window, gtk_layer_shell.Edge.TOP, True)
            gtk_layer_shell.set_margin(window, gtk_layer_shell.Edge.LEFT, x)
            gtk_layer_shell.set_margin(window, gtk_layer_shell.Edge.TOP, y)
            window.set_visual(window.get_screen().get_rgba_visual())
            drawing = gtk.DrawingArea()
            drawing.add_events(
                gdk.EventMask.BUTTON_PRESS_MASK
                | gdk.EventMask.BUTTON_RELEASE_MASK
                | gdk.EventMask.POINTER_MOTION_MASK
            )
            active: list[tuple[float, float]] = []
            self._strokes: list[list[tuple[float, float]]] = []

            def draw(_widget: object, context: object) -> bool:
                context.set_operator(0)
                context.paint()
                context.set_operator(2)
                context.set_source_rgba(1.0, 0.22, 0.08, 0.95)
                context.set_line_width(4.0)
                context.set_line_cap(1)
                context.set_line_join(1)
                strokes = self._strokes + ([active] if active else [])
                for stroke in strokes:
                    if stroke:
                        context.move_to(*stroke[0])
                        for point in stroke[1:]:
                            context.line_to(*point)
                        context.stroke()
                return False

            def press(_widget: object, event: object) -> bool:
                active.clear()
                active.append((event.x, event.y))
                self.points.append(
                    {"t": time.monotonic(), "x": event.x, "y": event.y, "action": "start"}
                )
                return True

            def motion(_widget: object, event: object) -> bool:
                if event.state & gdk.ModifierType.BUTTON1_MASK:
                    point = (event.x, event.y)
                    active.append(point)
                    if self._strokes and self._strokes[-1] is active:
                        self._strokes.pop()
                    self.points.append(
                        {"t": time.monotonic(), "x": event.x, "y": event.y, "action": "draw"}
                    )
                    drawing.queue_draw()

            def release(_widget: object, event: object) -> bool:
                if active:
                    active.append((event.x, event.y))
                    self._strokes.append(list(active))
                    active.clear()
                    self.points.append(
                        {"t": time.monotonic(), "x": event.x, "y": event.y, "action": "end"}
                    )
                    drawing.queue_draw()
                return True

            drawing.connect("draw", draw)
            drawing.connect("button-press-event", press)
            drawing.connect("motion-notify-event", motion)
            drawing.connect("button-release-event", release)
            window.connect("destroy", lambda *_args: self._window_destroyed())
            window.add(drawing)
            window.show_all()
            self.ready.set()
        except BaseException as exc:
            self.error = exc
            self.ready.set()
            self.finished.set()

    def export(
        self,
        path: Path,
        video_path: Path,
        ffmpeg: str,
        started_monotonic: float,
        width: int,
        height: int,
    ) -> None:
        import json

        from PIL import Image, ImageDraw

        points = [
            dict(point, t=max(0.0, float(point["t"]) - started_monotonic)) for point in self.points
        ]
        path.write_text(
            json.dumps({"width": width, "height": height, "keyframes": points}, indent=2) + "\n",
            encoding="utf-8",
        )
        strokes: list[list[tuple[float, float]]] = []
        active: list[tuple[float, float]] = []
        for point in points:
            action = point.get("action")
            coordinate = (float(point["x"]), float(point["y"]))
            if action == "start":
                active = [coordinate]
            elif action == "draw" and active:
                active.append(coordinate)
            elif action == "end" and active:
                active.append(coordinate)
                strokes.append(active)
                active = []
                timestamp = float(point["t"])
                frame = subprocess.run(
                    [
                        ffmpeg,
                        "-nostdin",
                        "-v",
                        "error",
                        "-ss",
                        str(timestamp),
                        "-i",
                        str(video_path),
                        "-frames:v",
                        "1",
                        "-f",
                        "image2pipe",
                        "-vcodec",
                        "png",
                        "-",
                    ],
                    capture_output=True,
                    check=False,
                )
                if frame.returncode or not frame.stdout:
                    raise ScreenAnnotationError(
                        frame.stderr.decode(errors="replace").strip()
                        or f"Could not extract video frame at {timestamp:.3f}s"
                    )
                from io import BytesIO

                image = Image.open(BytesIO(frame.stdout)).convert("RGB")
                draw = ImageDraw.Draw(image)
                for stroke in strokes:
                    if len(stroke) == 1 or all(p == stroke[0] for p in stroke):
                        x, y = stroke[-1]
                        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 56, 20))
                    else:
                        draw.line(stroke, fill=(255, 56, 20), width=4)
                frame_path = path.with_name(f"{path.stem}-{timestamp:08.3f}.png")
                image.save(frame_path)


def _require_program(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise ScreenAnnotationError(f"Required program '{name}' was not found in PATH")
    return path


def _configured_transcriber(audio_path: Path) -> str:
    """Transcribe without importing or initializing inference models here."""
    config = load_config()
    server_url = get_server_url()
    if server_url != "http://127.0.0.1:8765":
        from voiced.http_client import TranscriptionClient

        client = TranscriptionClient(server_url, timeout=max(config.client.timeout, 300.0))
        return client.transcribe_text_only(
            _read_audio(audio_path), sample_rate=16000, language=config.transcription.language
        )

    from voiced.worker_host import get_worker

    return get_worker(config).transcriber.transcribe_file(audio_path)


def _read_audio(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    if rate != 16000:
        from scipy.signal import resample_poly

        divisor = gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
    return audio


def run_annotation(
    output_dir: Path | str,
    *,
    stop_event: threading.Event | None = None,
    cancel_event: threading.Event | None = None,
    transcribe: Callable[[Path], str] | None = None,
    on_state: Callable[[str], None] | None = None,
    gtk: object | None = None,
    glib: object | None = None,
) -> AnnotationResult | None:
    """Record a Wayland region with drawing, then export video and transcript.

    ``stop_event`` ends a recording; ``cancel_event`` ends it without export.
    A supplied ``transcribe`` callable receives a temporary 16 kHz mono WAV.
    The desktop may supply initialized ``gtk`` and ``glib`` modules so overlay
    creation and teardown run on its GTK main loop. Draw with the primary
    mouse button; keyframes record stroke start, movement, and end.
    """
    if not os.environ.get("WAYLAND_DISPLAY"):
        raise ScreenAnnotationError("Screen capture requires a Wayland session (WAYLAND_DISPLAY)")
    slurp = _require_program("slurp")
    recorder = _require_program("wf-recorder")
    ffmpeg = _require_program("ffmpeg")

    def state(message: str) -> None:
        if on_state is not None:
            on_state(message)

    state("selecting")
    selection = subprocess.run([slurp], capture_output=True, text=True, check=False)
    if selection.returncode:
        if selection.returncode == 1:
            return None
        raise ScreenAnnotationError(selection.stderr.strip() or "Region selection failed")
    geometry = selection.stdout.strip()
    if not geometry:
        return None

    try:
        import re

        match = re.fullmatch(r"\s*(-?\d+)\s*,\s*(-?\d+)\s+(\d+)x(\d+)\s*", geometry)
        if match is None:
            raise ValueError
        x, y, width, height = (int(value) for value in match.groups())
    except ValueError as exc:
        raise ScreenAnnotationError(f"Invalid slurp geometry: {geometry!r}") from exc
    if width <= 0 or height <= 0:
        raise ScreenAnnotationError(f"Invalid slurp geometry: {geometry!r}")
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    stem = started_at.strftime("screen-%Y%m%dT%H%M%S-%fZ")
    video_path = destination / f"{stem}.mp4"
    transcript_path = destination / f"{stem}.txt"
    keyframes_path = destination / f"{stem}.keyframes.json"

    state("recording")
    overlay = _OverlaySession((x, y, width, height))
    process: subprocess.Popen[str] | None = None
    try:
        overlay.start(gtk, glib)
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_file:
            process = subprocess.Popen(
                [recorder, "-g", geometry, "-f", str(video_path), "--audio"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
                text=True,
            )
            start_clock = time.monotonic()
            cancelled = False
            while process.poll() is None:
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    process.terminate()
                    break
                if stop_event is not None and stop_event.is_set():
                    process.send_signal(signal.SIGINT)
                    break
                time.sleep(0.05)
            try:
                process.wait(timeout=10 if cancelled else 20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
                video_path.unlink(missing_ok=True)
                raise ScreenAnnotationError("wf-recorder did not stop cleanly")
            stderr_file.seek(0)
            stderr = stderr_file.read()
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        overlay.finish()
        raise
    finally:
        overlay.finish()

    if cancelled:
        video_path.unlink(missing_ok=True)
        keyframes_path.unlink(missing_ok=True)
        transcript_path.unlink(missing_ok=True)
        return None
    duration = time.monotonic() - start_clock
    if (
        process.returncode not in (0, 130)
        or not video_path.is_file()
        or video_path.stat().st_size == 0
    ):
        video_path.unlink(missing_ok=True)
        raise ScreenAnnotationError(stderr.strip() or "wf-recorder failed to save a recording")
    try:
        overlay.export(keyframes_path, video_path, ffmpeg, start_clock, width, height)
    except Exception:
        for frame_path in destination.glob(f"{keyframes_path.stem}-*.png"):
            frame_path.unlink(missing_ok=True)
        video_path.unlink(missing_ok=True)
        keyframes_path.unlink(missing_ok=True)
        raise

    state("transcribing")
    try:
        with tempfile.TemporaryDirectory(prefix="voiced-screen-") as temporary:
            wav_path = Path(temporary) / "audio.wav"
            converted = subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-v",
                    "error",
                    "-i",
                    str(video_path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-y",
                    str(wav_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if converted.returncode:
                raise ScreenAnnotationError(
                    converted.stderr.strip() or "Could not extract recording audio"
                )
            text = (transcribe or _configured_transcriber)(wav_path)
        transcript_path.write_text(text.rstrip() + "\n", encoding="utf-8")
    except Exception:
        for frame_path in destination.glob(f"{keyframes_path.stem}-*.png"):
            frame_path.unlink(missing_ok=True)
        video_path.unlink(missing_ok=True)
        keyframes_path.unlink(missing_ok=True)
        transcript_path.unlink(missing_ok=True)
        raise
    state("complete")
    return AnnotationResult(video_path, transcript_path, started_at, duration, keyframes_path)
