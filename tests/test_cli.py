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
        assert "Speech-to-Text Daemon" in result.output

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
