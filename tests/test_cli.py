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

    def test_status_not_running(self, monkeypatch, tmp_path):
        """Test status when the isolated daemon is not running."""
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        result = CliRunner().invoke(main, ["status"])
        assert result.exit_code == 1
        assert "not running" in result.output.lower()

    def test_toggle_not_running(self, monkeypatch, tmp_path):
        """Test toggle when the isolated daemon is not running."""
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        result = CliRunner().invoke(main, ["toggle"])
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

    def test_start_launches_gateway_client(self, monkeypatch):
        from types import SimpleNamespace

        from voiced import remote_daemon

        calls = []
        monkeypatch.setattr(remote_daemon, "is_gateway_running", lambda: False)
        monkeypatch.setattr(
            "voiced.cli.load_config",
            lambda: SimpleNamespace(client=SimpleNamespace(audio_base_url="")),
        )
        monkeypatch.setattr("voiced.config.get_server_url", lambda: "http://gateway.example")

        class FakeRemoteDaemon:
            def __init__(self, server_url, config):
                calls.append((server_url, config))

            def run(self):
                calls.append("run")

        monkeypatch.setattr(remote_daemon, "RemoteDaemon", FakeRemoteDaemon)
        result = CliRunner().invoke(main, ["start"])

        assert result.exit_code == 0
        assert calls[0][0] == "http://gateway.example"
        assert calls[1] == "run"
        assert "Starting gateway client" in result.output

    def test_default_control_socket_commands(self, monkeypatch, tmp_path):
        from voiced.server import Server

        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        server = Server()
        state = {"recording": False}

        def status(_args):
            return {
                "status": "ok",
                "state": "recording" if state["recording"] else "idle",
                "model": "test-model",
                "device": "cpu",
            }

        def toggle(_args):
            state["recording"] = not state["recording"]
            return {"status": "ok", "state": "recording" if state["recording"] else "idle"}

        def stop(_args):
            return {"status": "ok"}

        server.register_handler("status", status)
        server.register_handler("toggle", toggle)
        server.register_handler("stop", stop)
        server.start()
        try:
            status_result = CliRunner().invoke(main, ["status"])
            assert status_result.exit_code == 0
            assert "Status: running" in status_result.output
            assert "State: idle" in status_result.output
            assert "Model: test-model" in status_result.output

            toggle_result = CliRunner().invoke(main, ["toggle"])
            assert toggle_result.exit_code == 0
            assert "State: recording" in toggle_result.output

            stop_result = CliRunner().invoke(main, ["stop"])
            assert stop_result.exit_code == 0
            assert "Gateway client stopping" in stop_result.output
        finally:
            server.stop()

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
