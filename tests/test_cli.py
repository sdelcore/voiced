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

    def test_remote_client_socket_toggles_client_state(self, monkeypatch, tmp_path):
        import sys
        from types import ModuleType

        from voiced.client import send_command
        from voiced.config import AudioConfig, Config
        from voiced.server import Server

        tray = ModuleType("voiced.tray")
        tray.TrayIcon = object
        tray.TrayState = type("TrayState", (), {})
        monkeypatch.setitem(sys.modules, "voiced.tray", tray)
        from voiced.remote_daemon import ClientState, RemoteDaemon

        class FakeRecorder:
            is_recording = False

            def __init__(self, config):
                self.is_recording = False

            def start(self):
                self.is_recording = True

        monkeypatch.setattr("voiced.remote_daemon.Recorder", FakeRecorder)
        config = Config(audio=AudioConfig(beep_enabled=False))
        daemon = RemoteDaemon("http://example.test", config=config)
        socket_path = tmp_path / "client.sock"
        server = Server(socket_path)
        server.register_handler("toggle", daemon._handle_client_toggle)
        server.start()
        try:
            response = send_command("toggle", socket_path=socket_path)
            assert response == {"status": "ok", "state": "recording"}
            assert daemon.state == ClientState.RECORDING
        finally:
            server.stop()

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
