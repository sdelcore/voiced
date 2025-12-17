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
        click.echo("Status: running")
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
@click.option("-o", "--output", type=click.Path(path_type=Path), help="Output file")
@click.option("--model", default=None, help="Model to use (overrides config)")
@click.option("--device", default=None, help="Device to use: auto, cuda, cpu")
@click.option("--annotate", is_flag=True, help="Enable timestamps and speaker diarization")
def transcribe(
    audio_file: Path,
    output: Path | None,
    model: str | None,
    device: str | None,
    annotate: bool,
) -> None:
    """Transcribe an audio file.

    Examples:

        sttd transcribe audio.wav

        sttd transcribe audio.mp3 -o transcript.txt

        sttd transcribe audio.wav --model large-v3 --device cuda

        sttd transcribe meeting.wav --annotate
    """
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

        if annotate:
            from sttd.diarizer import SpeakerIdentifier
            from sttd.profiles import ProfileManager

            click.echo("Running speaker identification...", err=True)

            # Get transcription segments first
            trans_segments = transcriber.transcribe_file_with_segments(audio_file)
            click.echo(f"Found {len(trans_segments)} segment(s)", err=True)

            # Load profiles
            pm = ProfileManager()
            profiles = pm.load_all()
            if profiles:
                click.echo(f"Loaded {len(profiles)} voice profile(s)", err=True)

            # Identify speakers per segment
            identifier = SpeakerIdentifier(
                config=config.diarization,
                device=device or config.diarization.device,
            )

            result_segments = identifier.identify_segments(
                audio_file,
                trans_segments,
                profiles,
                threshold=config.diarization.similarity_threshold,
            )

            output_lines = []
            for seg in result_segments:
                time_str = f"[{seg.start:.2f}-{seg.end:.2f}]"
                output_lines.append(f"{time_str} {seg.speaker}: {seg.text}")

            text = "\n".join(output_lines)
        else:
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
        click.echo("\nRun 'sttd config --init' to create a config file.")


@main.command()
@click.argument("name")
@click.option(
    "-f",
    "--file",
    "audio_file",
    type=click.Path(exists=True, path_type=Path),
    help="Audio file containing the speaker's voice",
)
@click.option("-r", "--record", is_flag=True, help="Record audio from microphone")
@click.option(
    "-d",
    "--duration",
    type=float,
    default=10.0,
    help="Recording duration in seconds (default: 10)",
)
@click.option("--device", default=None, help="Device: auto, cuda, cpu")
@click.option("--force", is_flag=True, help="Overwrite existing profile")
def register(
    name: str,
    audio_file: Path | None,
    record: bool,
    duration: float,
    device: str | None,
    force: bool,
) -> None:
    """Register a voice profile for speaker identification.

    Provide either an audio file or use --record to capture from microphone.

    Examples:

        sttd register alice -f alice_sample.wav

        sttd register bob --record --duration 15

        sttd register alice -f new_sample.wav --force
    """
    import time
    from datetime import datetime

    from sttd.diarizer import ENROLLMENT_PROMPT, SpeakerEmbedder
    from sttd.profiles import ProfileManager, VoiceProfile

    config = load_config()

    if not audio_file and not record:
        click.echo("Error: Provide --file or --record", err=True)
        sys.exit(1)
    if audio_file and record:
        click.echo("Error: Use either --file or --record, not both", err=True)
        sys.exit(1)

    pm = ProfileManager()

    if pm.exists(name) and not force:
        click.echo(f"Error: Profile '{name}' already exists. Use --force to overwrite.", err=True)
        sys.exit(1)

    if record:
        click.echo("Please read aloud:")
        click.echo(f'"{ENROLLMENT_PROMPT}"')
        click.echo()
        click.echo(f"Recording for {duration} seconds... Press Ctrl+C to stop early.")

        from sttd.recorder import Recorder

        recorder = Recorder(config=config.audio)
        recorder.start()

        try:
            time.sleep(duration)
        except KeyboardInterrupt:
            click.echo("\nRecording stopped early")

        audio_data = recorder.stop()

        if len(audio_data) < config.audio.sample_rate * 2:
            click.echo("Error: Recording too short (minimum 2 seconds)", err=True)
            sys.exit(1)

        import tempfile

        import soundfile as sf

        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_file.name, audio_data, config.audio.sample_rate)
        audio_file = Path(temp_file.name)
        audio_duration = len(audio_data) / config.audio.sample_rate
    else:
        import soundfile as sf

        info = sf.info(str(audio_file))
        audio_duration = info.duration

    click.echo(f"Extracting voice embedding from: {audio_file}", err=True)

    try:
        dev = device or config.diarization.device
        embedder = SpeakerEmbedder(
            model_source=config.diarization.model,
            device=dev,
        )
        embedding = embedder.extract_embedding(audio_file)

        profile = VoiceProfile(
            name=name,
            embedding=embedding.tolist(),
            created_at=datetime.now().isoformat(),
            audio_duration=audio_duration,
            model_version=embedder.model_source,
        )

        path = pm.save(profile)
        click.echo(f"Profile '{name}' saved to: {path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        if record and audio_file and audio_file.exists():
            audio_file.unlink()


@main.command()
def profiles() -> None:
    """List registered voice profiles."""
    from sttd.profiles import ProfileManager

    pm = ProfileManager()
    profile_list = pm.load_all()

    if not profile_list:
        click.echo("No profiles registered. Use 'sttd register' to create one.")
        return

    click.echo("Registered voice profiles:\n")
    for p in profile_list:
        click.echo(f"  {p.name}")
        click.echo(f"    Created: {p.created_at}")
        click.echo(f"    Audio duration: {p.audio_duration:.1f}s")


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
