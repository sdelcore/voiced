"""Command-line interface for sttd."""

import logging
import re
import sys
from pathlib import Path

import click

from sttd import __version__
from sttd.client import ClientError, DaemonNotRunning
from sttd.config import get_config_path, load_config, save_default_config


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.version_option(__version__, prog_name="sttd")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """sttd - Speech-to-Text Daemon.

    A CLI application for speech-to-text transcription using faster-whisper.
    Designed for Hyprland with hotkey support.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging(verbose)


@main.command()
@click.option("--daemon", "-d", is_flag=True, help="Run in background (daemonize)")
@click.pass_context
def start(ctx: click.Context, daemon: bool) -> None:
    """Start the sttd daemon."""
    from sttd.daemon import Daemon, daemonize, is_daemon_running

    if is_daemon_running():
        click.echo("Daemon is already running", err=True)
        sys.exit(1)

    config = load_config()

    click.echo(f"Starting sttd daemon (model: {config.transcription.model})")

    if daemon:
        click.echo("Daemonizing...")
        daemonize()

    try:
        d = Daemon(config)
        d.run()
    except KeyboardInterrupt:
        click.echo("\nDaemon interrupted")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
def stop() -> None:
    """Stop the sttd daemon."""
    from sttd import client

    try:
        response = client.stop_daemon()
        if response.get("status") == "ok":
            click.echo("Daemon stopping")
        else:
            click.echo(f"Error: {response.get('message', 'Unknown error')}", err=True)
            sys.exit(1)
    except DaemonNotRunning:
        click.echo("Daemon is not running", err=True)
        sys.exit(1)
    except ClientError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
def toggle() -> None:
    """Toggle recording on/off.

    This is the command to bind to a hotkey in Hyprland:

        bind = SUPER, R, exec, sttd toggle
    """
    from sttd import client

    try:
        response = client.toggle_recording()
        if response.get("status") == "ok":
            state = response.get("state", "unknown")
            click.echo(f"State: {state}")
        elif response.get("status") == "busy":
            click.echo(f"Daemon busy: {response.get('message', '')}", err=True)
        else:
            click.echo(f"Error: {response.get('message', 'Unknown error')}", err=True)
            sys.exit(1)
    except DaemonNotRunning:
        click.echo("Daemon is not running. Start it with: sttd start", err=True)
        sys.exit(1)
    except ClientError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
def status() -> None:
    """Show daemon status."""
    from sttd import client

    try:
        response = client.get_status()
        click.echo(f"Status: running")
        click.echo(f"State: {response.get('state', 'unknown')}")
        click.echo(f"Model: {response.get('model', 'unknown')}")
        click.echo(f"Device: {response.get('device', 'unknown')}")
    except DaemonNotRunning:
        click.echo("Status: not running")
        sys.exit(1)
    except ClientError as e:
        click.echo(f"Status: error ({e})", err=True)
        sys.exit(1)


@main.command()
@click.argument("audio_file", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), help="Output file (default: stdout)")
@click.option("--model", default=None, help="Model to use (overrides config)")
@click.option("--device", default=None, help="Device to use: auto, cuda, cpu")
def transcribe(
    audio_file: Path,
    output: Path | None,
    model: str | None,
    device: str | None,
) -> None:
    """Transcribe an audio file.

    Examples:

        sttd transcribe audio.wav

        sttd transcribe audio.mp3 -o transcript.txt

        sttd transcribe audio.wav --model large-v3 --device cuda
    """
    from sttd.config import TranscriptionConfig
    from sttd.transcriber import Transcriber

    config = load_config()

    # Override config with CLI options
    if model:
        config.transcription.model = model
    if device:
        config.transcription.device = device

    click.echo(f"Transcribing: {audio_file}", err=True)
    click.echo(f"Model: {config.transcription.model}", err=True)

    try:
        transcriber = Transcriber(config.transcription)
        text = transcriber.transcribe_file(audio_file)

        if output:
            with open(output, "w") as f:
                f.write(text)
            click.echo(f"Saved to: {output}", err=True)
        else:
            click.echo(text)

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option("--show", is_flag=True, help="Show current configuration")
@click.option("--init", is_flag=True, help="Create default config file")
@click.option("--model", help="Set transcription model")
@click.option("--device", help="Set device (auto, cuda, cpu)")
def config(show: bool, init: bool, model: str | None, device: str | None) -> None:
    """Show or modify configuration.

    Configuration file location: ~/.config/sttd/config.toml
    """
    config_path = get_config_path()

    if init:
        if config_path.exists():
            click.echo(f"Config already exists: {config_path}")
        else:
            save_default_config()
            click.echo(f"Created config: {config_path}")
        return

    if model or device:
        # Modify config
        if not config_path.exists():
            save_default_config()

        # Read, modify, write
        with open(config_path) as f:
            content = f.read()

        if model:
            content = re.sub(
                r'^model\s*=\s*"[^"]*"',
                f'model = "{model}"',
                content,
                flags=re.MULTILINE,
            )
            click.echo(f"Set model: {model}")

        if device:
            content = re.sub(
                r'^device\s*=\s*"[^"]*"',
                f'device = "{device}"',
                content,
                flags=re.MULTILINE,
            )
            click.echo(f"Set device: {device}")

        with open(config_path, "w") as f:
            f.write(content)

        click.echo(f"Updated: {config_path}")
        return

    # Show config
    if config_path.exists():
        click.echo(f"Config file: {config_path}\n")
        with open(config_path) as f:
            click.echo(f.read())
    else:
        click.echo(f"No config file found at: {config_path}")
        click.echo("\nCurrent defaults:")
        cfg = load_config()
        click.echo(f"  model: {cfg.transcription.model}")
        click.echo(f"  device: {cfg.transcription.device}")
        click.echo(f"  language: {cfg.transcription.language}")
        click.echo(f"  beep_enabled: {cfg.audio.beep_enabled}")
        click.echo(f"  output_method: {cfg.output.method}")
        click.echo(f"\nRun 'sttd config --init' to create a config file.")


@main.command()
def devices() -> None:
    """List available audio input devices."""
    from sttd.recorder import Recorder

    recorder = Recorder()
    devices = recorder.get_devices()

    if not devices:
        click.echo("No audio input devices found")
        return

    click.echo("Available audio input devices:\n")
    for device in devices:
        click.echo(f"  [{device['index']}] {device['name']}")
        click.echo(f"      Channels: {device['channels']}, Sample rate: {device['sample_rate']}")


if __name__ == "__main__":
    main()
