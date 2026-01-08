# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Enter development environment (NixOS with direnv)
direnv allow

# Or manually with nix
nix develop

# Run the daemon
voiced start              # Foreground
voiced start --daemon     # Background

# STT commands
voiced toggle             # Toggle recording
voiced status             # Check daemon state
voiced stop               # Stop daemon
voiced transcribe file.wav  # Transcribe audio file

# TTS commands
voiced speak "Hello world"           # Speak text
voiced speak --stream "Hello world"  # Low-latency streaming
voiced speak --clipboard             # Speak clipboard contents
voiced voices list                   # List available voices

# Lint
ruff check src/
ruff format src/

# Run tests
pytest tests/ -v
pytest tests/test_cli.py -v          # Single test file
pytest tests/test_cli.py::test_name  # Single test
```

## Architecture

voiced is a voice daemon for Wayland/Hyprland that provides both:
- **STT (Speech-to-Text)**: Using faster-whisper for transcription
- **TTS (Text-to-Speech)**: Using VibeVoice for synthesis

### Component Flow

```
CLI Command → Unix Socket IPC → Daemon
                                  ├── Server (server.py) - Unix socket handler
                                  ├── Recorder (recorder.py) - sounddevice audio capture
                                  ├── Transcriber (transcriber.py) - faster-whisper STT
                                  ├── Synthesizer (synthesizer.py) - VibeVoice TTS
                                  ├── Injector (injector.py) - wl-clipboard text injection
                                  └── TrayIcon (tray.py) - D-Bus StatusNotifierItem
```

### Key Design Patterns

**Record-then-Transcribe (STT)**: Toggle once to start recording (RED tray icon), toggle again to stop and begin batch transcription (YELLOW icon). When complete (BLUE icon), text is copied to clipboard.

**Lazy Model Loading (TTS)**: TTS model loaded on first request, auto-unloaded after 1 hour of inactivity to free GPU memory.

**HTTP Client-Server Mode**: For remote STT/TTS:
- `http_server.py` - HTTP server with `/transcribe`, `/synthesize`, `/health` endpoints
- `http_client.py` - HTTP client for remote connections
- WebSocket support for streaming TTS at `/synthesize/stream`

**State Machine**: Daemon states are `IDLE → RECORDING → TRANSCRIBING → IDLE`. The tray icon reflects state visually.

**IPC Protocol**: JSON over Unix domain socket at `~/.cache/voiced/control.sock`. Commands: `toggle`, `status`, `stop`.

**GPU Detection**: Uses `ctranslate2.get_cuda_device_count()` for CUDA detection (not torch).

### Config Locations

- Config: `~/.config/voiced/config.toml`
- Socket: `~/.cache/voiced/control.sock`
- PID: `~/.cache/voiced/daemon.pid`
- Voice presets: `~/.cache/voiced/voices/`

## Code Style

- Line length: 100 chars
- Ruff linting with rules: E, F, I, N, W, UP
- Type hints throughout
- Minimal comments - code should be self-explanatory
