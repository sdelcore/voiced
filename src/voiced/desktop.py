"""GTK desktop companion for the voiced daemon.

GTK is loaded lazily so command-line and daemon use do not require a display.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from voiced.client import ClientError, DaemonNotRunning, send_command
from voiced.config import (
    get_cache_dir,
    get_client_socket_path,
    get_config_path,
    get_server_url,
    load_config,
)
from voiced.desktop_history import TranscriptHistory

POLL_INTERVAL_MS = 1000

logger = logging.getLogger(__name__)


def _toml_key(value: str) -> str:
    if value and all(
        char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in value
    ):
        return value
    import json

    return json.dumps(value)


def _serialize_toml(data: dict[str, Any]) -> str:
    import json

    lines: list[str] = []

    def table(values: dict[str, Any], path: tuple[str, ...]) -> None:
        if path:
            if lines and lines[-1] != "":
                lines.append("")
            lines.append("[" + ".".join(_toml_key(part) for part in path) + "]")
        children = []
        for key, value in values.items():
            if isinstance(value, dict):
                children.append((key, value))
            else:
                lines.append(f"{_toml_key(key)} = {json.dumps(value, ensure_ascii=False)}")
        for key, value in children:
            table(value, path + (key,))

    scalars = {key: value for key, value in data.items() if not isinstance(value, dict)}
    for key, value in scalars.items():
        lines.append(f"{_toml_key(key)} = {json.dumps(value, ensure_ascii=False)}")
    for key, value in data.items():
        if isinstance(value, dict):
            table(value, (key,))
    return "\n".join(lines) + "\n"


def _gtk():
    try:
        import gi

        gi.require_version("Gtk", "3.0")
        gi.require_version("Gdk", "3.0")
        from gi.repository import Gdk, GLib, Gtk

        try:
            gi.require_version("GtkLayerShell", "0.1")
            from gi.repository import GtkLayerShell

            layer_shell = GtkLayerShell
        except (ImportError, ValueError):
            layer_shell = None
        return Gtk, GLib, layer_shell, Gdk
    except (ImportError, ValueError) as exc:
        raise RuntimeError("GTK 3 / PyGObject is required for the desktop companion") from exc


class DesktopWindow:
    """GTK companion backed by daemon IPC, with capture actions in local threads."""

    def __init__(self, socket_path=None, *, client: bool = False, persist_history: bool = False):
        self.socket_path = socket_path or (get_client_socket_path() if client else None)
        self._Gtk, self._GLib, self._layer_shell, self._Gdk = _gtk()
        self._last_status: dict[str, Any] = {}
        self._busy = False
        self._closed = False
        self._annotation_stop = None
        self._annotation_cancel = None
        self._annotation_thread: threading.Thread | None = None
        self._annotation_mode = "default"
        self._hidden_history_tokens: set[tuple[str, str]] = set()
        self._last_daemon_history: list[dict[str, Any]] = []
        self.history = TranscriptHistory(persist=persist_history)
        self._recording_started: float | None = None
        self._meeting_thread: threading.Thread | None = None
        self._meeting_capture = None
        self._meeting_started: float | None = None
        self._meeting_stop = threading.Event()
        self._meeting_mode = "default"
        self._local_transcriber = None
        self._poll_source = None
        self.window = self._Gtk.Window(title="Voiced")
        self.window.set_default_size(420, 600)
        self.window.set_accept_focus(True)
        self.window.set_focus_on_map(True)
        self.window.connect("destroy", self._on_destroy)
        self._build_ui()
        self.overlay = self._build_overlay()
        self.window.show_all()
        self._poll_source = self._GLib.timeout_add(POLL_INTERVAL_MS, self._poll)
        self._poll()

    def _build_overlay(self):
        gtk = self._Gtk
        overlay = gtk.Window(type=gtk.WindowType.TOPLEVEL)
        overlay.set_title("Voiced status")
        overlay.set_default_size(280, 72)
        overlay.set_resizable(False)
        overlay.set_decorated(False)
        overlay.set_accept_focus(False)
        overlay.set_focus_on_map(False)
        overlay.set_skip_taskbar_hint(True)
        overlay.set_skip_pager_hint(True)
        overlay.set_type_hint(self._Gdk.WindowTypeHint.NOTIFICATION)
        overlay.set_keep_above(True)
        box = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=4)
        box.set_border_width(10)
        overlay.add(box)
        self.overlay_state = gtk.Label(label="Idle", xalign=0)
        self.overlay_timer = gtk.Label(label="00:00", xalign=0)
        self.overlay_level = gtk.LevelBar()
        self.overlay_level.set_min_value(0.0)
        self.overlay_level.set_max_value(1.0)
        self.overlay_error = gtk.Label(label="", xalign=0)
        self.overlay_error.set_line_wrap(True)
        for child in (
            self.overlay_state,
            self.overlay_timer,
            self.overlay_level,
            self.overlay_error,
        ):
            box.pack_start(child, False, False, 0)
        layer_shell = self._layer_shell
        if layer_shell is not None:
            layer_shell.init_for_window(overlay)
            layer_shell.set_layer(overlay, layer_shell.Layer.OVERLAY)
            layer_shell.set_anchor(overlay, layer_shell.Edge.TOP, True)
            layer_shell.set_anchor(overlay, layer_shell.Edge.RIGHT, True)
            layer_shell.set_margin(overlay, layer_shell.Edge.TOP, 16)
            layer_shell.set_margin(overlay, layer_shell.Edge.RIGHT, 16)
            layer_shell.set_keyboard_mode(overlay, layer_shell.KeyboardMode.NONE)
        overlay.show_all()
        overlay.hide()
        return overlay

    def _build_ui(self):
        gtk = self._Gtk
        root = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=10)
        root.set_border_width(14)
        self.window.add(root)

        self.state_label = gtk.Label(label="Checking daemon…", xalign=0)
        root.pack_start(self.state_label, False, False, 0)
        self.timer_label = gtk.Label(label="00:00", xalign=0)
        root.pack_start(self.timer_label, False, False, 0)
        self.level = gtk.LevelBar()
        self.level.set_min_value(0.0)
        self.level.set_max_value(1.0)
        self.level.set_value(0.0)
        self.level.set_tooltip_text("Microphone input level")
        root.pack_start(self.level, False, False, 0)

        actions = gtk.Box(spacing=6)
        self.record_button = gtk.Button(label="Start recording")
        self.record_button.connect("clicked", self._toggle)
        actions.pack_start(self.record_button, True, True, 0)
        stop = gtk.Button(label="Stop")
        stop.connect("clicked", lambda *_: self._command("stop_recording"))
        actions.pack_start(stop, False, False, 0)
        cancel = gtk.Button(label="Cancel")
        cancel.connect("clicked", lambda *_: self._command("cancel"))
        actions.pack_start(cancel, False, False, 0)
        copy = gtk.Button(label="Copy last")
        copy.connect("clicked", lambda *_: self._command("copy_last"))
        actions.pack_start(copy, False, False, 0)
        root.pack_start(actions, False, False, 0)
        self.meeting_button = gtk.Button(label="Record meeting (mic + system audio)")
        self.meeting_button.connect("clicked", self._meeting_clicked)
        root.pack_start(self.meeting_button, False, False, 0)

        mode_row = gtk.Box(spacing=6)
        mode_row.pack_start(gtk.Label(label="Context", xalign=0), False, False, 0)
        self.mode_combo = gtk.ComboBoxText()
        self._context_modes = sorted(
            set(load_config().transcription.context_replacements) | {"default"}
        )
        for mode in self._context_modes:
            self.mode_combo.append(mode, mode.replace("_", " ").title())
        self.mode_combo.set_active_id("default")
        self.mode_combo.set_tooltip_text("Choose transcription replacements context")
        self.mode_combo.connect("changed", self._mode_changed)
        mode_row.pack_start(self.mode_combo, True, True, 0)
        root.pack_start(mode_row, False, False, 0)

        annotation_row = gtk.Box(spacing=6)
        self.annotation_button = gtk.Button(label="Capture screen region + transcript")
        self.annotation_button.connect("clicked", self._annotation_clicked)
        annotation_row.pack_start(self.annotation_button, True, True, 0)
        self.annotation_stop_button = gtk.Button(label="Stop capture")
        self.annotation_stop_button.connect("clicked", self._annotation_stop_clicked)
        annotation_row.pack_start(self.annotation_stop_button, False, False, 0)
        self.annotation_cancel_button = gtk.Button(label="Cancel capture")
        self.annotation_cancel_button.connect("clicked", self._annotation_cancel_clicked)
        annotation_row.pack_start(self.annotation_cancel_button, False, False, 0)
        root.pack_start(annotation_row, False, False, 0)
        self.annotation_stop_button.set_sensitive(False)
        self.annotation_cancel_button.set_sensitive(False)

        self.notebook = gtk.Notebook()
        root.pack_start(self.notebook, True, True, 0)
        history_page = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=6)
        self.history_store = gtk.ListStore(str, str, str)
        self.history_view = gtk.TreeView(model=self.history_store)
        for title, index in (("Time", 0), ("Transcript", 1)):
            renderer = gtk.CellRendererText()
            column = gtk.TreeViewColumn(title, renderer, text=index)
            column.set_resizable(True)
            self.history_view.append_column(column)
        self.history_view.connect("row-activated", self._edit_replacements)
        scroll = gtk.ScrolledWindow()
        scroll.add(self.history_view)
        history_page.pack_start(scroll, True, True, 0)
        history_actions = gtk.Box(spacing=6)
        clear_history = gtk.Button(label="Clear displayed history")
        clear_history.connect("clicked", self._clear_history)
        history_actions.pack_start(clear_history, False, False, 0)
        history_page.pack_start(history_actions, False, False, 0)
        self.notebook.append_page(history_page, gtk.Label(label="History"))

        self.transcript_buffer = gtk.TextBuffer()
        transcript_view = gtk.TextView(buffer=self.transcript_buffer)
        transcript_view.set_wrap_mode(gtk.WrapMode.WORD_CHAR)
        transcript_view.set_editable(False)
        transcript_scroll = gtk.ScrolledWindow()
        transcript_scroll.add(transcript_view)
        self.notebook.append_page(transcript_scroll, gtk.Label(label="Transcript"))

        settings = gtk.Box(orientation=gtk.Orientation.VERTICAL, spacing=6)
        settings.set_border_width(8)
        self.settings_label = gtk.Label(label=f"Configuration: {get_config_path()}", xalign=0)
        self.settings_label.set_line_wrap(True)
        settings.pack_start(self.settings_label, False, False, 0)
        config = load_config()
        self.device_entry = gtk.Entry()
        self.device_entry.set_text(str(config.transcription.device))
        settings.pack_start(gtk.Label(label="Transcription device", xalign=0), False, False, 0)
        settings.pack_start(self.device_entry, False, False, 0)
        self.input_entry = gtk.Entry()
        self.input_entry.set_text(str(config.audio.device or ""))
        settings.pack_start(
            gtk.Label(label="Input device (blank for default)", xalign=0), False, False, 0
        )
        settings.pack_start(self.input_entry, False, False, 0)
        self.beep_toggle = gtk.CheckButton(label="Enable recording beep")
        self.beep_toggle.set_active(config.audio.beep_enabled)
        settings.pack_start(self.beep_toggle, False, False, 0)
        replacement_button = gtk.Button(label="Edit selected context replacements…")
        replacement_button.connect("clicked", self._edit_replacements)
        settings.pack_start(replacement_button, False, False, 0)
        save_settings = gtk.Button(label="Save settings")
        save_settings.connect("clicked", self._save_settings)
        settings.pack_start(save_settings, False, False, 0)
        self.config_summary = gtk.Label(label="", xalign=0)
        settings.pack_start(self.config_summary, False, False, 0)
        diagnostics = gtk.Button(label="Refresh diagnostics")
        diagnostics.connect("clicked", lambda *_: self._poll())
        settings.pack_start(diagnostics, False, False, 0)
        self.notebook.append_page(settings, gtk.Label(label="Settings"))
        self.error_label = gtk.Label(label="", xalign=0)
        self.error_label.set_line_wrap(True)
        root.pack_start(self.error_label, False, False, 0)
        self._refresh_config_summary()

    def _refresh_config_summary(self):
        try:
            config = load_config()
            self.config_summary.set_text(
                f"Device: {config.transcription.device}\n"
                f"Input: {config.audio.device or 'default'}\n"
                f"Beep: {'on' if config.audio.beep_enabled else 'off'}\n"
                f"Remote URL: {get_server_url() or 'not configured'}\n"
                f"Cache: {get_cache_dir()}"
            )
        except Exception as exc:
            self.config_summary.set_text(f"Unable to read configuration: {exc}")

    def _mode_changed(self, *_):
        mode = self.mode_combo.get_active_id()
        if not mode or mode == self._last_status.get("context_mode"):
            return
        self._busy = True

        def run():
            try:
                response = self._request("set_context_mode", {"mode": mode})
                error = (
                    None
                    if response.get("status") == "ok"
                    else response.get("message", "Mode change failed")
                )
            except Exception as exc:
                response, error = None, str(exc)
            self._GLib.idle_add(self._settings_done, response, error)

        threading.Thread(target=run, name="voiced-desktop-mode", daemon=True).start()

    def _edit_replacements(self, *_):
        try:
            import tomllib

            path = get_config_path()
            raw = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            transcription = raw.setdefault("transcription", {})
            mode = self.mode_combo.get_active_id() or "default"
            base_replacements = dict(transcription.get("replacements", {}))
            context_replacements = dict(transcription.get("context_replacements", {}))
            if mode == "default":
                replacements = base_replacements
                title = "Global transcription replacements"
            else:
                replacements = dict(context_replacements.get(mode, {}))
                title = f"{mode.replace('_', ' ').title()} context replacements"
            dialog = self._Gtk.Dialog(title=title, transient_for=self.window, flags=0)
            dialog.add_button("Cancel", self._Gtk.ResponseType.CANCEL)
            dialog.add_button("Save", self._Gtk.ResponseType.OK)
            editor = self._Gtk.TextView()
            editor.get_buffer().set_text(
                "\n".join(f"{wrong} = {right}" for wrong, right in replacements.items())
            )
            editor.set_size_request(420, 140)
            dialog.get_content_area().pack_start(editor, True, True, 8)
            dialog.show_all()
            accepted = dialog.run() == self._Gtk.ResponseType.OK
            if accepted:
                buffer = editor.get_buffer()
                text = buffer.get_text(*buffer.get_bounds(), include_hidden_chars=True)
                parsed = {}
                for line in text.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        raise ValueError("Replacement rows must use 'wrong = right'")
                    wrong, right = (part.strip() for part in line.split("=", 1))
                    if not wrong or not right:
                        raise ValueError("Replacement keys and values cannot be empty")
                    parsed[wrong] = right
                if mode == "default":
                    base_replacements = parsed
                    transcription["replacements"] = parsed
                else:
                    context_replacements[mode] = parsed
                    transcription["context_replacements"] = context_replacements
            dialog.destroy()
            if not accepted:
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.tmp")
            temporary.write_text(_serialize_toml(raw), encoding="utf-8")
            temporary.replace(path)

            def apply_live():
                try:
                    response = self._request(
                        "update_replacements",
                        {
                            "replacements": base_replacements,
                            "context_replacements": context_replacements,
                        },
                    )
                    if response.get("status") != "ok":
                        message = response.get("message", "Daemon did not apply replacements")
                    elif mode != "default":
                        message = f"{title} saved and applied to future results."
                    elif self._last_status.get("mode") == "remote":
                        message = "Global replacements saved and applied to future remote results."
                    else:
                        message = "Global replacements saved and applied to future local results."
                except Exception as exc:
                    message = f"Replacements saved; live daemon update failed: {exc}"
                self._GLib.idle_add(self.error_label.set_text, message)

            threading.Thread(
                target=apply_live, name="voiced-desktop-replacements", daemon=True
            ).start()
        except Exception as exc:
            self.error_label.set_text(f"Unable to save replacements: {exc}")

    def _save_settings(self, *_):
        try:
            import tomllib

            path = get_config_path()
            raw = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            raw.setdefault("transcription", {})["device"] = (
                self.device_entry.get_text().strip() or "auto"
            )
            raw.setdefault("audio", {})["device"] = self.input_entry.get_text().strip() or "default"
            raw["audio"]["beep_enabled"] = self.beep_toggle.get_active()
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.tmp")
            temporary.write_text(_serialize_toml(raw), encoding="utf-8")
            temporary.replace(path)
            self.error_label.set_text("Settings saved. Restart daemon for device changes.")
            self._refresh_config_summary()
        except Exception as exc:
            self.error_label.set_text(f"Unable to save settings: {exc}")

    def _settings_done(self, response, error):
        self._busy = False
        if error:
            self.error_label.set_text(str(error))
        self._poll()
        return False

    def _request(self, command: str, args: dict | None = None) -> dict:
        return send_command(command, args=args, socket_path=self.socket_path)

    def _command(self, command: str):
        if self._busy:
            return
        self._busy = True
        self.error_label.set_text("")

        def run():
            try:
                response = self._request(command)
                self._GLib.idle_add(self._command_done, response, None)
            except Exception as exc:
                self._GLib.idle_add(self._command_done, None, str(exc))

        threading.Thread(target=run, name="voiced-desktop-command", daemon=True).start()

    def _command_done(self, response, error):
        self._busy = False
        if error:
            self.error_label.set_text(error)
        elif response and response.get("status") != "ok":
            self.error_label.set_text(
                response.get("message") or response.get("reason") or "Command failed"
            )
        self._poll()
        return False

    def _toggle(self, *_):
        self._command("toggle")

    def _poll(self):
        if self._closed:
            return False
        try:
            status = self._request("status")
            if status.get("status") != "ok":
                raise ClientError(status.get("message", "Status unavailable"))
            self._last_status = status
            self._render_status(status)
            history_response = self._request("history")
            daemon_entries = history_response.get("history", [])
            live_tokens = {
                (str(item.get("id", "")), str(item.get("timestamp", "")))
                for item in daemon_entries
                if isinstance(item, dict)
            }
            self._hidden_history_tokens.intersection_update(live_tokens)
            visible = [
                item
                for item in daemon_entries
                if (str(item.get("id", "")), str(item.get("timestamp", "")))
                not in self._hidden_history_tokens
            ]
            self._last_daemon_history = visible
            self._render_history(visible)
            self.error_label.set_text(status.get("error") or "")
        except (ClientError, DaemonNotRunning, OSError) as exc:
            self.state_label.set_text("Daemon unavailable")
            self.record_button.set_sensitive(False)
            self.error_label.set_text(str(exc))
            self.overlay_state.set_text("Daemon unavailable")
            self.overlay_error.set_text(str(exc))
            self.overlay.show_all()
        except Exception:
            logger.exception("Desktop status refresh failed")
        return True

    def _render_status(self, status):
        state = status.get("state", "unknown")
        self.state_label.set_text(
            f"{state.title()}  |  {status.get('model', 'model unavailable')}  |  "
            f"{status.get('device', 'device unavailable')}"
        )
        self.record_button.set_sensitive(state != "transcribing" and not self._busy)
        self.record_button.set_label(
            "Stop recording" if state == "recording" else "Start recording"
        )
        mode = status.get("context_mode", "default")

        if mode in self._context_modes and self.mode_combo.get_active_id() != mode:
            self.mode_combo.set_active_id(mode)
        self.mode_combo.set_tooltip_text("Choose transcription replacements context")
        level = status.get("input_level")
        level_value = max(0.0, min(1.0, float(level))) if level is not None else 0.0
        self.level.set_value(level_value)
        self.overlay_level.set_value(level_value)
        text = status.get("partial_text") or status.get("last_transcript") or ""
        self.transcript_buffer.set_text(text)
        if state == "recording":
            if self._recording_started is None:
                self._recording_started = time.monotonic()
            elapsed = max(0, int(time.monotonic() - self._recording_started))
            timer = f"{elapsed // 60:02d}:{elapsed % 60:02d}"
            self.timer_label.set_text(timer)
        else:
            self._recording_started = None
            timer = "00:00"
            self.timer_label.set_text(timer)
        self.overlay_timer.set_text(timer)
        self.overlay_state.set_text("Transcribing" if state == "transcribing" else state.title())
        self.overlay_error.set_text(str(status.get("error") or ""))
        if status.get("error") or state in ("recording", "transcribing"):
            self.overlay.show_all()
        else:
            self.overlay.hide()

    def _render_history(self, entries):
        self.history.sync(entries)
        local_entries = self.history.entries()
        self.history_store.clear()
        for entry in local_entries:
            timestamp = str(entry.get("timestamp", ""))
            text = entry.get("preview", entry.get("text", ""))
            self.history_store.append((timestamp, text, str(entry.get("id", ""))))

    def _clear_history(self, *_):
        self._hidden_history_tokens.update(
            (str(item.get("id", "")), str(item.get("timestamp", "")))
            for item in self._last_daemon_history
        )
        self.history.clear()
        self.history_store.clear()
        self._last_daemon_history = []

    def _get_transcriber(self, mode: str | None = None):
        config = load_config()
        if (mode or self._last_status.get("mode")) == "remote":
            from voiced.http_client import TranscriptionClient
            from voiced.transcribe import RemoteTranscribe

            server_url = config.client.server_url or get_server_url()
            return RemoteTranscribe(TranscriptionClient(server_url))
        from voiced.capabilities import Voiced

        self._local_transcriber = Voiced.from_config(config)
        return self._local_transcriber

    def _meeting_clicked(self, *_):
        if self._meeting_thread is not None and self._meeting_thread.is_alive():
            self._meeting_stop.set()
            self.meeting_button.set_sensitive(False)
            self.error_label.set_text("Stopping meeting recording…")
            return
        self._meeting_stop.clear()
        self._meeting_mode = self.mode_combo.get_active_id() or "default"
        self._meeting_transcriber_mode = self._last_status.get("mode")
        self._meeting_errors = {}
        self.meeting_button.set_label("Stop meeting recording")
        self._meeting_thread = threading.Thread(
            target=self._record_meeting, name="voiced-desktop-meeting", daemon=True
        )
        self._meeting_thread.start()

    def _meeting_poll(self):
        if self._meeting_capture is None:
            return False
        elapsed = max(0, int(time.monotonic() - self._meeting_started))
        self.timer_label.set_text(f"Meeting {elapsed // 60:02d}:{elapsed % 60:02d}")
        return True

    def _meeting_finished(self, message=None):
        self.meeting_button.set_sensitive(True)
        self.meeting_button.set_label("Record meeting (mic + system audio)")
        self.timer_label.set_text("00:00")
        if message:
            self.error_label.set_text(message)
        return False

    def _meeting_results(self, result):
        failures = []
        for key, title, transcript in (
            ("microphone", "Microphone", result.microphone),
            ("system_audio", "System audio", result.system_audio),
        ):
            error = (result.errors or {}).get(key)
            if error:
                failures.append(f"{title}: {error}")
            if transcript is None:
                continue
            segments = getattr(transcript, "segments", None)
            if segments:
                lines = []
                for segment in segments:
                    speaker = getattr(segment, "speaker", None)
                    text = getattr(segment, "text", "")
                    lines.append(f"{speaker}: {text}" if speaker else text)
                text = "\n".join(lines)
            else:
                text = getattr(transcript, "text", str(transcript))
            self.history.add_local(self._apply_context_replacements(text, self._meeting_mode))
        return (
            "Meeting transcription errors: " + "; ".join(failures)
            if failures
            else "Meeting recording transcribed"
        )

    def _record_meeting(self):
        capture = None
        final_message = None
        try:
            from voiced.meetings import MeetingCapture, list_audio_sources

            sources = list_audio_sources()
            microphones = [item for item in sources if item.get("kind") == "microphone"]
            system = [item for item in sources if item.get("kind") == "system"]
            mic = self._choose_source("Microphone", microphones)
            loopback = self._choose_source("System audio", system)
            if mic is None and loopback is None:
                raise RuntimeError("No microphone or system audio source selected")
            capture = MeetingCapture(microphone=mic, system_audio=loopback)
            self._meeting_capture = capture
            capture.start()
            self._meeting_started = time.monotonic()
            self._GLib.idle_add(self._meeting_poll)
            self._GLib.idle_add(self.error_label.set_text, "Recording meeting audio…")
            while capture.is_recording and not self._meeting_stop.is_set() and not self._closed:
                time.sleep(0.2)
            recording = capture.stop()
            if recording is not None:
                result = recording.transcribe(
                    self._get_transcriber(self._meeting_transcriber_mode), identify_speakers=True
                )
                final_message = self._meeting_results(result)
        except Exception as exc:
            logger.exception("Meeting recording failed")
            final_message = f"Meeting unavailable: {exc}"
        finally:
            if capture is not None:
                capture.close()
            self._meeting_capture = None
            self._GLib.idle_add(self._meeting_finished, final_message)
            self._GLib.idle_add(self._poll)

    def _apply_context_replacements(self, text: str, mode: str | None = None) -> str:
        import re

        mode = mode or "default"
        replacements = load_config().transcription.context_replacements.get(mode, {})
        for wrong, right in replacements.items():
            text = re.sub(rf"\b{re.escape(wrong)}\b", lambda _: right, text, flags=re.IGNORECASE)
        return text

    def _choose_source(self, title, sources):
        if not sources:
            return None
        event = threading.Event()
        selected: dict[str, Any] = {}
        self._GLib.idle_add(self._source_dialog, title, sources, event, selected)
        if not event.wait(120):
            return None
        return selected.get("source")

    def _source_dialog(self, title, sources, event, selected):
        gtk = self._Gtk
        dialog = gtk.Dialog(title=title, transient_for=self.window, flags=0)
        dialog.add_button("Skip", gtk.ResponseType.CANCEL)
        dialog.add_button("Select", gtk.ResponseType.OK)
        combo = gtk.ComboBoxText()
        for index, source in enumerate(sources):
            combo.append(str(index), f"{source.get('name')} ({source.get('channels')} ch)")
        combo.set_active(0)
        dialog.get_content_area().pack_start(combo, True, True, 8)
        dialog.show_all()
        if dialog.run() == gtk.ResponseType.OK:
            selected["source"] = sources[int(combo.get_active_id())]
        dialog.destroy()
        event.set()
        return False

    def _annotate_transcribe(self, audio_path):
        import soundfile as sf

        from voiced.transcribe import RemoteTranscribe

        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            import numpy as np

            audio = np.mean(audio, axis=1, dtype=np.float32)
        transcriber = self._get_transcriber(self._annotation_mode)
        if isinstance(transcriber, RemoteTranscribe) and sample_rate != 16000:
            from math import gcd

            from scipy.signal import resample_poly

            divisor = gcd(sample_rate, 16000)
            audio = resample_poly(audio, 16000 // divisor, sample_rate // divisor).astype("float32")
            sample_rate = 16000
        text = transcriber.transcribe(audio, sample_rate).text
        return self._apply_context_replacements(text, self._annotation_mode)

    def _annotation_stop_clicked(self, *_):
        if self._annotation_stop is not None:
            self._annotation_stop.set()
            self.error_label.set_text("Finishing screen capture…")

    def _annotation_cancel_clicked(self, *_):
        if self._annotation_cancel is not None:
            self._annotation_cancel.set()
            self.error_label.set_text("Cancelling screen capture…")

    def _annotation_clicked(self, *_):
        if self._annotation_thread is not None and self._annotation_thread.is_alive():
            return
        self.annotation_button.set_sensitive(False)
        self.annotation_stop_button.set_sensitive(True)
        self.annotation_cancel_button.set_sensitive(True)
        self._annotation_mode = self.mode_combo.get_active_id() or "default"
        self._annotation_mode = self.mode_combo.get_active_id() or "default"
        self._annotation_stop = threading.Event()
        self._annotation_cancel = threading.Event()

        def show_state(message):
            self._GLib.idle_add(self.error_label.set_text, f"Screen capture: {message}")

        def run():
            try:
                from voiced.screen_annotate import run_annotation

                result = run_annotation(
                    get_cache_dir() / "screen-captures",
                    gtk=self._Gtk,
                    glib=self._GLib,
                    stop_event=self._annotation_stop,
                    cancel_event=self._annotation_cancel,
                    transcribe=lambda audio_path: self._annotate_transcribe(audio_path),
                    on_state=show_state,
                )
                message = (
                    "Capture cancelled" if result is None else f"Exported: {result.video_path}"
                )
                self._GLib.idle_add(self.error_label.set_text, message)
            except Exception as exc:
                logger.exception("Screen capture failed")
                self._GLib.idle_add(self.error_label.set_text, f"Screen capture unavailable: {exc}")
            finally:
                self._annotation_stop = None
                self._annotation_cancel = None
                self._GLib.idle_add(self.annotation_button.set_sensitive, True)
                self._GLib.idle_add(self.annotation_stop_button.set_sensitive, False)
                self._GLib.idle_add(self.annotation_cancel_button.set_sensitive, False)

        self._annotation_thread = threading.Thread(target=run, daemon=True)
        self._annotation_thread.start()

    def _on_destroy(self, *_):
        self._closed = True
        if self._poll_source:
            self._GLib.source_remove(self._poll_source)
            self._poll_source = None
        if self._meeting_thread is not None and self._meeting_thread.is_alive():
            self._meeting_stop.set()
        if self._annotation_cancel is not None:
            self._annotation_cancel.set()
        if self._local_transcriber is not None:
            try:
                self._local_transcriber.shutdown()
            except Exception:
                logger.exception("Failed to stop local inference worker")
        self.overlay.destroy()
        self._Gtk.main_quit()

    def run(self):
        self.window.show_all()
        self._Gtk.main()


def main(*, client: bool = False):
    """Launch the GTK desktop companion."""
    DesktopWindow(client=client).run()
