"""Tests for CLI module."""

from click.testing import CliRunner

from voiced.cli import main


class TestCLI:
    """Tests for CLI commands."""

    def test_version(self):
        """Test version option."""
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "voiced" in result.output

    def test_help(self):
        """Test help output."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Voice Daemon" in result.output

    def test_status_not_running(self):
        """Test status when daemon is not running."""
        runner = CliRunner()
        result = runner.invoke(main, ["status"])
        # Should exit with code 1 if daemon is not running
        assert "not running" in result.output.lower() or result.exit_code == 1

    def test_toggle_not_running(self):
        """Test toggle when daemon is not running."""
        runner = CliRunner()
        result = runner.invoke(main, ["toggle"])
        assert result.exit_code == 1
        assert "not running" in result.output.lower()

    def test_remote_client_initializes_shutdown_event_for_control_lifecycle(self, monkeypatch):
        import sys
        from types import ModuleType

        from voiced.config import AudioConfig, Config

        tray = ModuleType("voiced.tray")
        tray.TrayIcon = object
        tray.TrayState = type("TrayState", (), {})
        monkeypatch.setitem(sys.modules, "voiced.tray", tray)
        from voiced.remote_daemon import ClientState, RemoteDaemon

        daemon = RemoteDaemon("http://example.test", config=Config(audio=AudioConfig()))
        assert daemon._shutdown_event.is_set() is False
        assert daemon.is_running is False
        assert daemon.state == ClientState.IDLE

        daemon._on_tray_quit()
        assert daemon._shutdown_event.is_set() is True
        assert daemon._handle_client_stop({}) == {"status": "ok"}

    def test_cleanup_stops_control_socket_before_removing_pid(self, tmp_path):
        from voiced.config import AudioConfig, Config
        from voiced.remote_daemon import RemoteDaemon

        class FakeControlServer:
            def __init__(self, events):
                self.events = events

            def stop(self):
                self.events.append("socket")

        daemon = RemoteDaemon("http://example.test", config=Config(audio=AudioConfig()))
        events = []
        daemon._control_server = FakeControlServer(events)
        daemon._remove_pid = lambda: events.append("pid")

        daemon._cleanup()

        assert events == ["socket", "pid"]

    def test_toggle_client_targets_remote_client_socket(self, monkeypatch, tmp_path):
        from voiced import client

        socket_path = tmp_path / "client-control.sock"
        sent = {}

        def send_command(command, socket_path=None):
            sent["command"] = command
            sent["socket_path"] = socket_path
            return {"status": "ok", "state": "recording"}

        monkeypatch.setattr(client, "send_command", send_command)
        monkeypatch.setattr("voiced.config.get_client_socket_path", lambda: socket_path)

        result = CliRunner().invoke(main, ["toggle", "--client"])

        assert result.exit_code == 0
        assert sent == {"command": "toggle", "socket_path": socket_path}
        assert "State: recording" in result.output

    def test_client_status_uses_client_socket(self, monkeypatch, tmp_path):
        from voiced import client

        socket_path = tmp_path / "client-control.sock"
        calls = []
        monkeypatch.setattr("voiced.config.get_client_socket_path", lambda: socket_path)
        monkeypatch.setattr(
            client,
            "send_command",
            lambda command, socket_path=None: (
                calls.append((command, socket_path))
                or {"status": "ok", "state": "idle", "server": "http://nightman.tap:8765"}
            ),
        )

        result = CliRunner().invoke(main, ["status", "--client"])

        assert result.exit_code == 0
        assert calls == [("status", socket_path)]
        assert "http://nightman.tap:8765" in result.output

    def test_client_stop_uses_client_socket(self, monkeypatch, tmp_path):
        from voiced import client

        socket_path = tmp_path / "client-control.sock"
        calls = []
        monkeypatch.setattr("voiced.config.get_client_socket_path", lambda: socket_path)
        monkeypatch.setattr(
            client,
            "send_command",
            lambda command, socket_path=None: (
                calls.append((command, socket_path)) or {"status": "ok"}
            ),
        )

        result = CliRunner().invoke(main, ["stop", "--client"])

        assert result.exit_code == 0
        assert calls == [("stop", socket_path)]
        assert "Remote client stopping" in result.output

    def test_devices_command(self):
        """Test devices listing."""
        runner = CliRunner()
        result = runner.invoke(main, ["devices"])
        # Should not error even if no devices
        assert result.exit_code == 0

    def test_config_show(self):
        """Test config show command."""
        runner = CliRunner()
        result = runner.invoke(main, ["config", "--show"])
        assert result.exit_code == 0

    def test_transcribe_missing_file(self):
        """Test transcribe with missing file."""
        runner = CliRunner()
        result = runner.invoke(main, ["transcribe", "/nonexistent/file.wav"])
        assert result.exit_code != 0
