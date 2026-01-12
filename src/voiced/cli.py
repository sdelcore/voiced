"""Command-line interface for voiced."""

import logging
import re
import sys
from pathlib import Path

import click

from voiced import __version__
from voiced.client import ClientError, DaemonNotRunning
from voiced.config import get_config_path, load_config, save_default_config


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


@click.group()
@click.version_option(__version__, prog_name="voiced")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """voiced - Voice Daemon for STT and TTS.

    A CLI application for speech-to-text (STT) and text-to-speech (TTS).
    Uses faster-whisper for STT and VibeVoice for TTS.
    Designed for Hyprland/Wayland with hotkey support.

    \b
    STT Commands:
      voiced start          Start the daemon for hotkey recording
      voiced toggle         Toggle recording on/off
      voiced transcribe     Transcribe an audio file

    \b
    TTS Commands:
      voiced speak          Synthesize speech from text
      voiced voices         Manage TTS voice presets
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging(verbose)


@main.command()
@click.option("--daemon", "-d", is_flag=True, help="Run in background (daemonize)")
@click.option("--http", is_flag=True, help="Enable embedded HTTP server for remote transcription")
@click.option("--http-host", default=None, help="HTTP server host (default: from config)")
@click.option("--http-port", default=None, type=int, help="HTTP server port (default: from config)")
@click.option("--no-vad", is_flag=True, help="Disable voice activity detection")
@click.pass_context
def start(
    ctx: click.Context,
    daemon: bool,
    http: bool,
    http_host: str | None,
    http_port: int | None,
    no_vad: bool,
) -> None:
    """Start the voiced daemon.

    The daemon provides hotkey-triggered recording with tray icon.
    Use --http to also enable the HTTP transcription API.

    Examples:

        voiced start                      # Desktop mode only

        voiced start --http               # Desktop + HTTP API

        voiced start --http --http-host 0.0.0.0  # Accept remote connections

        voiced start -d                   # Run in background
    """
    from voiced.daemon import Daemon, daemonize, is_daemon_running

    if is_daemon_running():
        click.echo("Daemon is already running", err=True)
        sys.exit(1)

    config = load_config()

    # Override VAD if --no-vad is set
    if no_vad:
        config.vad.enabled = False

    # Determine effective HTTP settings
    http_enabled = http or config.daemon.http_enabled

    click.echo(f"Starting voiced daemon (model: {config.transcription.model})")
    if http_enabled:
        effective_host = http_host or config.daemon.http_host or config.server.host
        effective_port = http_port or config.daemon.http_port or config.server.port
        click.echo(f"HTTP server: {effective_host}:{effective_port}")

    if daemon:
        click.echo("Daemonizing...")
        daemonize()

    try:
        d = Daemon(
            config,
            http_enabled=http if http else None,  # Only override if explicitly set
            http_host=http_host,
            http_port=http_port,
        )
        d.run()
    except KeyboardInterrupt:
        click.echo("\nDaemon interrupted")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option("--host", default=None, help="Host to bind to (default: 127.0.0.1)")
@click.option("--port", default=None, type=int, help="Port to bind to (default: 8765)")
@click.option("--daemon", "-d", is_flag=True, help="Run in background")
@click.option("--no-vad", is_flag=True, help="Disable voice activity detection")
@click.pass_context
def server(
    ctx: click.Context,
    host: str | None,
    port: int | None,
    daemon: bool,
    no_vad: bool,
) -> None:
    """Start the transcription HTTP server.

    The server accepts audio via HTTP POST and returns transcribed text.
    WebRTC is always enabled for real-time streaming.
    Bind to 0.0.0.0 to accept connections from other machines.

    Examples:

        voiced server                     # Local only

        voiced server --host 0.0.0.0      # Network accessible

        voiced server --port 9000         # Custom port

        voiced server -d                  # Run in background
    """
    import signal

    from voiced.daemon import daemonize
    from voiced.http_server import TranscriptionServer

    config = load_config()

    # Override VAD if --no-vad is set
    if no_vad:
        config.vad.enabled = False

    effective_host = host or config.server.host
    effective_port = port or config.server.port

    click.echo(f"Starting transcription server on {effective_host}:{effective_port}")
    click.echo(f"Model: {config.transcription.model}")
    click.echo("WebRTC: enabled")

    if daemon:
        click.echo("Daemonizing...")
        daemonize()

    srv = TranscriptionServer(
        host=effective_host,
        port=effective_port,
        config=config,
    )

    def handle_sigterm(signum, frame):
        click.echo("\nShutting down server...")
        srv.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    try:
        srv.start()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command("client")
@click.option(
    "--server", "server_url", default=None, help="Server URL (e.g., http://192.168.1.100:8765)"
)
@click.option("--daemon", "-d", is_flag=True, help="Run in background")
@click.option("--timeout", type=float, default=None, help="Request timeout in seconds")
@click.pass_context
def client_cmd(
    ctx: click.Context, server_url: str | None, daemon: bool, timeout: float | None
) -> None:
    """Start the remote client daemon.

    Records audio locally and sends to a remote server for transcription.
    The server URL can be set via CLI, environment variable (STTD_SERVER_URL),
    or config file.

    Examples:

        voiced client --server http://192.168.1.100:8765

        voiced client -d                  # Run in background

        STTD_SERVER_URL=http://server:8765 voiced client
    """
    from voiced.config import get_server_url
    from voiced.daemon import daemonize
    from voiced.remote_daemon import RemoteDaemon, is_client_running

    if is_client_running():
        click.echo("Client is already running", err=True)
        sys.exit(1)

    config = load_config()
    effective_url = get_server_url(server_url)

    click.echo("Starting remote client")
    click.echo(f"Server: {effective_url}")

    if daemon:
        click.echo("Daemonizing...")
        daemonize()

    try:
        client = RemoteDaemon(
            server_url=effective_url,
            config=config,
            timeout=timeout,
        )
        client.run()
    except KeyboardInterrupt:
        click.echo("\nClient interrupted")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command("webrtc-client")
@click.option(
    "--server", "server_url", default=None, help="Server URL (e.g., http://192.168.1.100:8765)"
)
@click.option("--timeout", type=float, default=30.0, help="Connection timeout in seconds")
@click.pass_context
def webrtc_client_cmd(
    ctx: click.Context, server_url: str | None, timeout: float
) -> None:
    """Start interactive WebRTC client for real-time streaming.

    Connects to a server with WebRTC enabled for low-latency bidirectional
    audio streaming. Supports real-time STT and TTS.

    Examples:

        voiced webrtc-client --server http://192.168.1.100:8765

        # Interactive mode with real-time transcription
        voiced webrtc-client
    """
    import asyncio

    from voiced.config import get_server_url
    from voiced.webrtc_client import STTResult, WebRTCClient, WebRTCClientConfig

    effective_url = get_server_url(server_url)

    click.echo("Starting WebRTC client")
    click.echo(f"Server: {effective_url}")

    client_config = WebRTCClientConfig(
        server_url=effective_url,
        timeout=timeout,
    )

    async def run_interactive():
        client = WebRTCClient(client_config)

        # Set up callbacks for real-time feedback
        def on_partial(text: str):
            click.echo(f"\r[partial] {text}", nl=False)

        def on_final(result: STTResult):
            click.echo(f"\r[final] {result.text}")
            if result.segments:
                for seg in result.segments:
                    speaker = seg.get("speaker", "Unknown")
                    text = seg.get("text", "")
                    click.echo(f"  [{speaker}] {text}")

        def on_error(code: str, message: str):
            click.echo(f"Error: {code} - {message}", err=True)

        client.set_callbacks(
            on_stt_partial=on_partial,
            on_stt_final=on_final,
            on_error=on_error,
        )

        try:
            click.echo("Connecting...")
            await client.connect()
            click.echo("Connected! Commands: r=record, s=stop, t <text>=TTS, q=quit")

            while True:
                # Simple command loop
                loop = asyncio.get_event_loop()
                cmd = await loop.run_in_executor(None, input, "> ")
                cmd = cmd.strip()

                if cmd == "q":
                    break
                elif cmd == "r":
                    click.echo("Recording... (press 's' to stop)")
                    await client.start_stt(identify_speakers=True)
                elif cmd == "s":
                    result = await client.stop_stt()
                    click.echo(f"Transcription: {result.text}")
                elif cmd.startswith("t "):
                    text = cmd[2:].strip()
                    if text:
                        click.echo(f"Speaking: {text}")
                        await client.speak(text)
                        click.echo("Done speaking")
                elif cmd:
                    click.echo("Unknown command. Use 'r', 's', 't <text>', or 'q'")

        except Exception as e:
            click.echo(f"Error: {e}", err=True)
        finally:
            await client.disconnect()
            click.echo("Disconnected")

    try:
        asyncio.run(run_interactive())
    except KeyboardInterrupt:
        click.echo("\nClient interrupted")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
def stop() -> None:
    """Stop the voiced daemon."""
    from voiced import client

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

        bind = SUPER, R, exec, voiced toggle
    """
    from voiced import client

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
        click.echo("Daemon is not running. Start it with: voiced start", err=True)
        sys.exit(1)
    except ClientError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
def status() -> None:
    """Show daemon status."""
    from voiced import client

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
@click.option("--num-speakers", type=int, default=None, help="Number of speakers (auto if unset)")
@click.option("--server", "server_url", default=None, help="Server URL for remote transcription")
@click.option(
    "--timeout", type=float, default=300.0, help="Request timeout in seconds (default: 300)"
)
@click.option("--no-vad", is_flag=True, help="Disable voice activity detection")
def transcribe(
    audio_file: Path,
    output: Path | None,
    model: str | None,
    device: str | None,
    annotate: bool,
    num_speakers: int | None,
    server_url: str | None,
    timeout: float,
    no_vad: bool,
) -> None:
    """Transcribe an audio file.

    Examples:

        voiced transcribe audio.wav

        voiced transcribe audio.mp3 -o transcript.txt

        voiced transcribe audio.wav --model large-v3 --device cuda

        voiced transcribe meeting.wav --annotate

        voiced transcribe meeting.wav --annotate --num-speakers 3

        voiced transcribe audio.wav --server http://192.168.1.100:8765

        voiced transcribe meeting.wav --annotate --server http://server:8765
    """
    config = load_config()

    # Override config with CLI options
    if model:
        config.transcription.model = model
    if device:
        config.transcription.device = device
    if no_vad:
        config.vad.enabled = False

    click.echo(f"Transcribing: {audio_file}", err=True)

    try:
        # Use HTTP mode if server URL is provided
        if server_url:
            click.echo(f"Using remote server: {server_url}", err=True)
            _transcribe_remote(audio_file, output, server_url, timeout, annotate)
        else:
            click.echo(f"Model: {config.transcription.model}", err=True)
            _transcribe_local(audio_file, output, config, device, annotate, num_speakers)

    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _transcribe_remote(
    audio_file: Path,
    output: Path | None,
    server_url: str,
    timeout: float,
    annotate: bool,
) -> None:
    """Transcribe using remote server via WebRTC."""
    import soundfile as sf

    from voiced.webrtc_client import SyncWebRTCClient, WebRTCClientConfig

    # Load audio file
    audio, sample_rate = sf.read(str(audio_file), dtype="float32")

    # Convert stereo to mono if needed
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    config = WebRTCClientConfig(server_url=server_url, timeout=timeout)
    client = SyncWebRTCClient(config)

    try:
        click.echo("Connecting via WebRTC...", err=True)
        client.connect()

        # Server handles diarization when identify_speakers=True
        result = client.batch_transcribe(
            audio,
            sample_rate=sample_rate,
            identify_speakers=annotate,
            timeout=timeout,
        )

        if annotate and result.segments:
            # Format segments with speaker info
            output_lines = []
            for seg in result.segments:
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                speaker = seg.get("speaker", "Unknown")
                text = seg.get("text", "").strip()
                time_str = f"[{start:.2f}-{end:.2f}]"
                output_lines.append(f"{time_str} {speaker}: {text}")
            text = "\n".join(output_lines)
        else:
            text = result.text

        if output:
            with open(output, "w") as f:
                f.write(text)
            click.echo(f"Saved to: {output}", err=True)
        else:
            click.echo(text)

    except Exception as e:
        import traceback
        click.echo(f"Error ({type(e).__name__}): {e}", err=True)
        traceback.print_exc()
        sys.exit(1)
    finally:
        client.disconnect()


def _transcribe_local(
    audio_file: Path,
    output: Path | None,
    config,
    device: str | None,
    annotate: bool,
    num_speakers: int | None,
) -> None:
    """Transcribe using local model."""
    from voiced.transcriber import Transcriber

    transcriber = Transcriber(config.transcription, vad_config=config.vad)

    if annotate:
        from voiced.diarizer import (
            SpeakerDiarizer,
            align_transcription_with_diarization,
        )
        from voiced.profiles import ProfileManager

        click.echo("Running speaker diarization...", err=True)

        # Run diarization first
        diarizer = SpeakerDiarizer(
            config=config.diarization,
            device=device or config.diarization.device,
        )

        # Load profiles for matching
        pm = ProfileManager()
        profiles = pm.load_all()
        if profiles:
            click.echo(f"Loaded {len(profiles)} voice profile(s)", err=True)
            diar_segments = diarizer.diarize_and_match_profiles(
                audio_file,
                profiles,
                num_speakers=num_speakers,
            )
        else:
            diar_segments = diarizer.diarize_file(
                audio_file,
                num_speakers=num_speakers,
            )

        click.echo(f"Found {len(set(s.speaker for s in diar_segments))} speaker(s)", err=True)

        # Get transcription segments
        trans_segments = transcriber.transcribe_file_with_segments(audio_file)
        click.echo(f"Found {len(trans_segments)} text segment(s)", err=True)

        # Align transcription with diarization
        result_segments = align_transcription_with_diarization(
            trans_segments,
            diar_segments,
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


@main.command()
@click.option("-o", "--output", type=click.Path(path_type=Path), help="Output file")
@click.option("--model", default=None, help="Model to use (overrides config)")
@click.option("--device", default=None, help="Device to use: auto, cuda, cpu")
@click.option("--annotate", is_flag=True, help="Enable speaker diarization")
@click.option("--num-speakers", type=int, default=None, help="Number of speakers (auto if unset)")
def record(
    output: Path | None,
    model: str | None,
    device: str | None,
    annotate: bool,
    num_speakers: int | None,
) -> None:
    """Record from microphone and transcribe with timestamps.

    Recording stops with Ctrl+C. After stopping, the audio is transcribed
    with SRT-style timestamps.

    Examples:

        voiced record

        voiced record -o transcript.txt

        voiced record --annotate

        voiced record --annotate --num-speakers 2

        voiced record --model large-v3 --device cuda
    """
    import signal
    import tempfile
    import time

    from voiced.recorder import Recorder
    from voiced.transcriber import Transcriber

    config = load_config()

    # Override config with CLI options
    if model:
        config.transcription.model = model
    if device:
        config.transcription.device = device

    click.echo("Recording... Press Ctrl+C to stop.", err=True)

    recorder = Recorder(config=config.audio)
    recording_stopped = False

    def handle_sigint(signum, frame):
        nonlocal recording_stopped
        if not recording_stopped:
            recording_stopped = True
            click.echo("\nStopping recording...", err=True)

    # Install signal handler
    original_handler = signal.signal(signal.SIGINT, handle_sigint)

    try:
        recorder.start()

        # Wait until Ctrl+C
        while not recording_stopped:
            time.sleep(0.1)

        audio_data = recorder.stop()

    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)

    duration = len(audio_data) / config.audio.sample_rate
    click.echo(f"Recorded {duration:.2f} seconds of audio", err=True)

    if len(audio_data) < config.audio.sample_rate * 0.5:
        click.echo("Error: Recording too short (minimum 0.5 seconds)", err=True)
        sys.exit(1)

    click.echo(f"Transcribing with model: {config.transcription.model}", err=True)

    try:
        transcriber = Transcriber(config.transcription)
        segments = transcriber.transcribe_audio_with_segments(
            audio_data,
            config.audio.sample_rate,
        )

        click.echo(f"Found {len(segments)} segment(s)", err=True)

        if annotate:
            import soundfile as sf

            from voiced.diarizer import (
                SpeakerDiarizer,
                align_transcription_with_diarization,
            )
            from voiced.profiles import ProfileManager

            click.echo("Running speaker diarization...", err=True)

            # Save audio to temp file for diarization (SpeechBrain requires file path)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = Path(f.name)

            try:
                sf.write(str(temp_path), audio_data, config.audio.sample_rate)

                diarizer = SpeakerDiarizer(
                    config=config.diarization,
                    device=device or config.diarization.device,
                )

                pm = ProfileManager()
                profiles = pm.load_all()
                if profiles:
                    click.echo(f"Loaded {len(profiles)} voice profile(s)", err=True)
                    diar_segments = diarizer.diarize_and_match_profiles(
                        temp_path,
                        profiles,
                        num_speakers=num_speakers,
                    )
                else:
                    diar_segments = diarizer.diarize_file(
                        temp_path,
                        num_speakers=num_speakers,
                    )

                click.echo(
                    f"Found {len(set(s.speaker for s in diar_segments))} speaker(s)", err=True
                )

                # Align transcription with diarization
                result_segments = align_transcription_with_diarization(
                    segments,
                    diar_segments,
                )

                output_lines = []
                for seg in result_segments:
                    start_ts = format_srt_timestamp(seg.start)
                    end_ts = format_srt_timestamp(seg.end)
                    output_lines.append(f"[{start_ts} --> {end_ts}] {seg.speaker}: {seg.text}")

                text = "\n".join(output_lines)
            finally:
                temp_path.unlink(missing_ok=True)
        else:
            # Format without speaker diarization
            output_lines = []
            for start, end, segment_text in segments:
                start_ts = format_srt_timestamp(start)
                end_ts = format_srt_timestamp(end)
                output_lines.append(f"[{start_ts} --> {end_ts}] {segment_text}")

            text = "\n".join(output_lines)

        if output:
            with open(output, "w") as f:
                f.write(text)
            click.echo(f"Saved to: {output}", err=True)
        else:
            click.echo(text)

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

    Configuration file location: ~/.config/voiced/config.toml
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
        click.echo("\nRun 'voiced config --init' to create a config file.")


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
@click.option("--server", "server_url", default=None, help="Server URL for remote registration")
@click.option(
    "--timeout", type=float, default=60.0, help="Request timeout in seconds (default: 60)"
)
def register(
    name: str,
    audio_file: Path | None,
    record: bool,
    duration: float,
    device: str | None,
    force: bool,
    server_url: str | None,
    timeout: float,
) -> None:
    """Register a voice profile for speaker identification.

    Provide either an audio file or use --record to capture from microphone.

    Examples:

        voiced register alice -f alice_sample.wav

        voiced register bob --record --duration 15

        voiced register alice -f new_sample.wav --force

        voiced register alice -f alice_sample.wav --server http://192.168.1.100:8765
    """
    import time

    config = load_config()

    if not audio_file and not record:
        click.echo("Error: Provide --file or --record", err=True)
        sys.exit(1)
    if audio_file and record:
        click.echo("Error: Use either --file or --record, not both", err=True)
        sys.exit(1)

    # For remote registration, we don't check local profile existence
    if not server_url:
        from voiced.profiles import ProfileManager

        pm = ProfileManager()
        if pm.exists(name) and not force:
            click.echo(
                f"Error: Profile '{name}' already exists. Use --force to overwrite.", err=True
            )
            sys.exit(1)

    if record:
        from voiced.diarizer import ENROLLMENT_PROMPT
        from voiced.recorder import Recorder

        click.echo("Please read aloud:")
        click.echo(f'"{ENROLLMENT_PROMPT}"')
        click.echo()
        click.echo(f"Recording for {duration} seconds... Press Ctrl+C to stop early.")

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

    # At this point audio_file is guaranteed to be set
    assert audio_file is not None

    try:
        if server_url:
            # Remote registration
            _register_remote(name, audio_file, server_url, timeout)
        else:
            # Local registration
            _register_local(name, audio_file, audio_duration, device, config)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        if record and audio_file and audio_file.exists():
            audio_file.unlink()


def _register_remote(name: str, audio_file: Path, server_url: str, timeout: float) -> None:
    """Register a voice profile on a remote server."""
    import soundfile as sf

    from voiced.http_client import (
        HttpConnectionError,
        ServerError,
        TranscriptionClient,
    )

    click.echo(f"Registering profile '{name}' on server: {server_url}", err=True)

    # Load audio file
    audio, sample_rate = sf.read(str(audio_file), dtype="float32")

    # Convert stereo to mono if needed
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    client = TranscriptionClient(server_url, timeout=timeout)

    try:
        result = client.create_profile(name, audio, sample_rate)
        click.echo(f"Profile '{name}' registered on server")
        if "audio_duration" in result:
            click.echo(f"  Audio duration: {result['audio_duration']:.1f}s")
    except ServerError as e:
        click.echo(f"Server error: {e}", err=True)
        sys.exit(1)
    except HttpConnectionError as e:
        click.echo(f"Connection error: {e}", err=True)
        sys.exit(1)


def _register_local(
    name: str,
    audio_file: Path,
    audio_duration: float,
    device: str | None,
    config,
) -> None:
    """Register a voice profile locally."""
    from datetime import datetime

    from voiced.diarizer import SpeakerEmbedder
    from voiced.profiles import ProfileManager, VoiceProfile

    click.echo(f"Extracting voice embedding from: {audio_file}", err=True)

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

    pm = ProfileManager()
    path = pm.save(profile)
    click.echo(f"Profile '{name}' saved to: {path}")


# Profile management command group
@main.group(invoke_without_command=True)
@click.pass_context
def profiles(ctx: click.Context) -> None:
    """Manage voice profiles.

    When called without a subcommand, lists all profiles.

    Examples:

        voiced profiles                    # List local profiles

        voiced profiles list               # List local profiles

        voiced profiles list --server URL  # List remote profiles

        voiced profiles show alice         # Show profile details

        voiced profiles delete alice       # Delete a profile
    """
    # If no subcommand is invoked, default to listing profiles
    if ctx.invoked_subcommand is None:
        ctx.invoke(profiles_list)


@profiles.command("list")
@click.option("--server", "server_url", default=None, help="Server URL for remote profiles")
@click.option("--timeout", type=float, default=10.0, help="Request timeout in seconds")
def profiles_list(server_url: str | None, timeout: float = 10.0) -> None:
    """List all voice profiles (local or remote)."""
    if server_url:
        _list_profiles_remote(server_url, timeout)
    else:
        _list_profiles_local()


def _list_profiles_local() -> None:
    """List local voice profiles."""
    from voiced.profiles import ProfileManager

    pm = ProfileManager()
    profile_list = pm.load_all()

    if not profile_list:
        click.echo("No profiles registered. Use 'voiced register' to create one.")
        return

    click.echo("Registered voice profiles:\n")
    for p in profile_list:
        click.echo(f"  {p.name}")
        click.echo(f"    Created: {p.created_at}")
        click.echo(f"    Audio duration: {p.audio_duration:.1f}s")


def _list_profiles_remote(server_url: str, timeout: float) -> None:
    """List remote voice profiles."""
    from voiced.http_client import (
        HttpConnectionError,
        ServerError,
        TranscriptionClient,
    )

    client = TranscriptionClient(server_url, timeout=timeout)

    try:
        profile_list = client.list_profiles()

        if not profile_list:
            click.echo(f"No profiles on server: {server_url}")
            return

        click.echo(f"Voice profiles on {server_url}:\n")
        for p in profile_list:
            name = p.get("name", "unknown")
            created_at = p.get("created_at", "unknown")
            audio_duration = p.get("audio_duration", 0)
            click.echo(f"  {name}")
            click.echo(f"    Created: {created_at}")
            click.echo(f"    Audio duration: {audio_duration:.1f}s")

    except ServerError as e:
        click.echo(f"Server error: {e}", err=True)
        sys.exit(1)
    except HttpConnectionError as e:
        click.echo(f"Connection error: {e}", err=True)
        sys.exit(1)


@profiles.command("show")
@click.argument("name")
@click.option("--server", "server_url", default=None, help="Server URL for remote profile")
@click.option("--timeout", type=float, default=10.0, help="Request timeout in seconds")
def profiles_show(name: str, server_url: str | None, timeout: float) -> None:
    """Show details of a voice profile."""
    if server_url:
        _show_profile_remote(name, server_url, timeout)
    else:
        _show_profile_local(name)


def _show_profile_local(name: str) -> None:
    """Show local voice profile details."""
    from voiced.profiles import ProfileManager

    pm = ProfileManager()
    profile = pm.load(name)

    if not profile:
        click.echo(f"Profile '{name}' not found", err=True)
        sys.exit(1)

    click.echo(f"Profile: {profile.name}\n")
    click.echo(f"  Created: {profile.created_at}")
    click.echo(f"  Audio duration: {profile.audio_duration:.1f}s")
    click.echo(f"  Model version: {profile.model_version}")
    click.echo(f"  Embedding dimensions: {len(profile.embedding)}")


def _show_profile_remote(name: str, server_url: str, timeout: float) -> None:
    """Show remote voice profile details."""
    from voiced.http_client import (
        HttpConnectionError,
        ServerError,
        TranscriptionClient,
    )

    client = TranscriptionClient(server_url, timeout=timeout)

    try:
        profile = client.get_profile(name)

        if not profile:
            click.echo(f"Profile '{name}' not found on server", err=True)
            sys.exit(1)

        click.echo(f"Profile: {profile.get('name', name)}\n")
        click.echo(f"  Created: {profile.get('created_at', 'unknown')}")
        click.echo(f"  Audio duration: {profile.get('audio_duration', 0):.1f}s")
        if "model_version" in profile:
            click.echo(f"  Model version: {profile['model_version']}")
        if "embedding" in profile:
            click.echo(f"  Embedding dimensions: {len(profile['embedding'])}")

    except ServerError as e:
        click.echo(f"Server error: {e}", err=True)
        sys.exit(1)
    except HttpConnectionError as e:
        click.echo(f"Connection error: {e}", err=True)
        sys.exit(1)


@profiles.command("delete")
@click.argument("name")
@click.option("--server", "server_url", default=None, help="Server URL for remote profile")
@click.option("--force", is_flag=True, help="Skip confirmation")
@click.option("--timeout", type=float, default=10.0, help="Request timeout in seconds")
def profiles_delete(name: str, server_url: str | None, force: bool, timeout: float) -> None:
    """Delete a voice profile."""
    if not force:
        location = f"server {server_url}" if server_url else "local storage"
        if not click.confirm(f"Delete profile '{name}' from {location}?"):
            click.echo("Cancelled")
            return

    if server_url:
        _delete_profile_remote(name, server_url, timeout)
    else:
        _delete_profile_local(name)


def _delete_profile_local(name: str) -> None:
    """Delete local voice profile."""
    from voiced.profiles import ProfileManager

    pm = ProfileManager()
    if pm.delete(name):
        click.echo(f"Profile '{name}' deleted")
    else:
        click.echo(f"Profile '{name}' not found", err=True)
        sys.exit(1)


def _delete_profile_remote(name: str, server_url: str, timeout: float) -> None:
    """Delete remote voice profile."""
    from voiced.http_client import (
        HttpConnectionError,
        ServerError,
        TranscriptionClient,
    )

    client = TranscriptionClient(server_url, timeout=timeout)

    try:
        if client.delete_profile(name):
            click.echo(f"Profile '{name}' deleted from server")
        else:
            click.echo(f"Profile '{name}' not found on server", err=True)
            sys.exit(1)

    except ServerError as e:
        click.echo(f"Server error: {e}", err=True)
        sys.exit(1)
    except HttpConnectionError as e:
        click.echo(f"Connection error: {e}", err=True)
        sys.exit(1)


@main.command()
def devices() -> None:
    """List available audio input devices."""
    from voiced.recorder import Recorder

    recorder = Recorder()
    devices = recorder.get_devices()

    if not devices:
        click.echo("No audio input devices found")
        return

    click.echo("Available audio input devices:\n")
    for device in devices:
        click.echo(f"  [{device['index']}] {device['name']}")
        click.echo(f"      Channels: {device['channels']}, Sample rate: {device['sample_rate']}")


# =============================================================================
# TTS Commands
# =============================================================================


@main.command()
@click.argument("text", required=False)
@click.option("--stream", is_flag=True, help="Use low-latency streaming playback")
@click.option("--clipboard", is_flag=True, help="Speak text from clipboard")
@click.option("--stdin", is_flag=True, help="Read text from stdin")
@click.option(
    "-o", "--save", "output_file", type=click.Path(path_type=Path), help="Save audio to file"
)
@click.option("-v", "--voice", default=None, help="Voice preset (default: from config)")
@click.option("--server", "server_url", default=None, help="Server URL for remote TTS")
@click.option("--timeout", type=float, default=60.0, help="Request timeout in seconds")
@click.option("--cfg-scale", type=float, default=None, help="Classifier-free guidance scale")
@click.pass_context
def speak(
    ctx: click.Context,
    text: str | None,
    stream: bool,
    clipboard: bool,
    stdin: bool,
    output_file: Path | None,
    voice: str | None,
    server_url: str | None,
    timeout: float,
    cfg_scale: float | None,
) -> None:
    """Synthesize speech from text.

    Examples:

        voiced speak "Hello world"

        voiced speak --stream "Low latency streaming"

        voiced speak --clipboard                    # Speak clipboard contents

        echo "Hello" | voiced speak --stdin

        voiced speak "Save this" -o output.wav

        voiced speak "Hello" --voice mike

        voiced speak "Hello" --server http://192.168.1.100:8765
    """
    # Determine input source
    sources = [text is not None, clipboard, stdin]
    if sum(sources) > 1:
        click.echo("Error: Use only one input source (text, --clipboard, or --stdin)", err=True)
        sys.exit(1)

    if not any(sources):
        click.echo("Error: Provide text, --clipboard, or --stdin", err=True)
        sys.exit(1)

    # Get text from source
    if clipboard:
        import subprocess

        try:
            result = subprocess.run(
                ["wl-paste", "--no-newline"],
                capture_output=True,
                text=True,
                check=True,
            )
            text = result.stdout
        except FileNotFoundError:
            click.echo("Error: wl-paste not found (install wl-clipboard)", err=True)
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            click.echo(f"Error reading clipboard: {e}", err=True)
            sys.exit(1)

    elif stdin:
        text = sys.stdin.read()

    if not text or not text.strip():
        click.echo("Error: No text to speak", err=True)
        sys.exit(1)

    text = text.strip()
    config = load_config()

    if not config.tts.enabled:
        click.echo("Error: TTS is disabled in config", err=True)
        sys.exit(1)

    # Use server if URL provided
    if server_url:
        _speak_remote(text, server_url, voice, output_file, stream, timeout, cfg_scale)
    else:
        _speak_local(text, config, voice, output_file, stream, cfg_scale)


def _speak_local(
    text: str,
    config,
    voice: str | None,
    output_file: Path | None,
    stream: bool,
    cfg_scale: float | None,
) -> None:
    """Synthesize speech locally using VibeVoice."""
    from voiced.synthesizer import Synthesizer, TTSConfig, check_vibevoice_installed

    if not check_vibevoice_installed():
        click.echo(
            "Error: VibeVoice is not installed. Install it with:\n"
            "  pip install git+https://github.com/microsoft/VibeVoice.git",
            err=True,
        )
        sys.exit(1)

    # Create TTSConfig from app config
    tts_config = TTSConfig(
        model_path=config.tts.model,
        device=config.tts.device,
        default_voice=voice or config.tts.default_voice,
        cfg_scale=cfg_scale if cfg_scale is not None else config.tts.cfg_scale,
        unload_timeout_seconds=config.tts.unload_timeout_minutes * 60,
    )

    synthesizer = Synthesizer(tts_config)

    try:
        if stream and not output_file:
            # Streaming playback
            from voiced.tts_streamer import StreamingAudioPlayer

            click.echo(f"Streaming: {text[:50]}{'...' if len(text) > 50 else ''}", err=True)

            player = StreamingAudioPlayer(sample_rate=synthesizer.sample_rate)
            player.start()

            try:
                for chunk in synthesizer.synthesize_streaming(
                    text, voice=voice, cfg_scale=cfg_scale
                ):
                    player.write(chunk)
                player.finish()
                player.wait()
            except KeyboardInterrupt:
                player.stop()
                click.echo("\nPlayback interrupted", err=True)
        else:
            # Batch synthesis
            click.echo(f"Synthesizing: {text[:50]}{'...' if len(text) > 50 else ''}", err=True)
            audio = synthesizer.synthesize(text, voice=voice, cfg_scale=cfg_scale)

            if output_file:
                # Save to file
                import soundfile as sf

                sf.write(str(output_file), audio, synthesizer.sample_rate)
                click.echo(f"Saved to: {output_file}", err=True)
            else:
                # Play audio
                from voiced.tts_streamer import play_audio

                play_audio(audio, synthesizer.sample_rate)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        synthesizer.shutdown()


def _speak_remote(
    text: str,
    server_url: str,
    voice: str | None,
    output_file: Path | None,
    stream: bool,
    timeout: float,
    cfg_scale: float | None,
) -> None:
    """Synthesize speech using remote server via WebRTC."""
    from voiced.webrtc_client import SyncWebRTCClient, WebRTCClientConfig

    click.echo(f"Using remote server: {server_url}", err=True)
    click.echo(f"Synthesizing: {text[:50]}{'...' if len(text) > 50 else ''}", err=True)

    if output_file:
        click.echo("Note: --output not supported with remote TTS (audio plays directly)", err=True)

    config = WebRTCClientConfig(server_url=server_url, timeout=timeout)
    client = SyncWebRTCClient(config)

    try:
        click.echo("Connecting via WebRTC...", err=True)
        client.connect()

        # Speak via WebRTC - audio is streamed to local playback
        client.speak(text, voice=voice, cfg_scale=cfg_scale, timeout=timeout)
        click.echo("Done", err=True)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        client.disconnect()


# Voice management command group
@main.group(invoke_without_command=True)
@click.pass_context
def voices(ctx: click.Context) -> None:
    """Manage TTS voice presets.

    When called without a subcommand, lists available voices.

    Examples:

        voiced voices                    # List all voices

        voiced voices list               # List all voices

        voiced voices download emma      # Download voice preset

        voiced voices remove emma        # Remove cached voice
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(voices_list)


@voices.command("list")
def voices_list() -> None:
    """List available TTS voice presets."""
    from voiced.voice_manager import VoiceManager

    vm = VoiceManager()
    available = vm.list_available()
    downloaded = set(vm.list_downloaded())

    click.echo("Available voice presets:\n")
    for name in available:
        status = "[downloaded]" if name in downloaded else "[not downloaded]"
        click.echo(f"  {name:10} {status}")

    if downloaded:
        click.echo(f"\n{len(downloaded)} voice(s) cached locally")


@voices.command("download")
@click.argument("name")
@click.option("--force", is_flag=True, help="Re-download even if cached")
def voices_download(name: str, force: bool) -> None:
    """Download a voice preset.

    Examples:

        voiced voices download emma

        voiced voices download mike --force
    """
    from voiced.voice_manager import VoiceManager

    vm = VoiceManager()

    try:
        click.echo(f"Downloading voice preset: {name}")
        path = vm.download(name, force=force)
        click.echo(f"Downloaded to: {path}")
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error downloading voice: {e}", err=True)
        sys.exit(1)


@voices.command("remove")
@click.argument("name")
@click.option("--force", is_flag=True, help="Skip confirmation")
def voices_remove(name: str, force: bool) -> None:
    """Remove a cached voice preset.

    Examples:

        voiced voices remove emma

        voiced voices remove mike --force
    """
    from voiced.voice_manager import VoiceManager

    vm = VoiceManager()

    try:
        info = vm.get_voice_info(name)
        if not info.get("downloaded"):
            click.echo(f"Voice '{name}' is not downloaded", err=True)
            sys.exit(1)

        if not force:
            if not click.confirm(f"Remove voice preset '{name}'?"):
                click.echo("Cancelled")
                return

        if vm.remove(name):
            click.echo(f"Removed voice preset: {name}")
        else:
            click.echo(f"Voice '{name}' not found", err=True)
            sys.exit(1)

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@voices.command("info")
@click.argument("name")
def voices_info(name: str) -> None:
    """Show details about a voice preset.

    Examples:

        voiced voices info emma
    """
    from voiced.voice_manager import VoiceManager

    vm = VoiceManager()

    try:
        info = vm.get_voice_info(name)

        click.echo(f"Voice: {info['name']}\n")
        click.echo(f"  Filename: {info['filename']}")
        click.echo(f"  Downloaded: {'yes' if info['downloaded'] else 'no'}")

        if info.get("downloaded"):
            size_mb = info.get("size_bytes", 0) / (1024 * 1024)
            click.echo(f"  Size: {size_mb:.1f} MB")
            click.echo(f"  Path: {info.get('path', 'unknown')}")

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
