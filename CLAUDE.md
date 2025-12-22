# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Enter development environment (NixOS with direnv)
direnv allow

# Or manually with nix
nix develop

# Run the daemon
sttd start              # Foreground
sttd start --daemon     # Background

# Test commands
sttd toggle             # Toggle recording
sttd status             # Check daemon state
sttd stop               # Stop daemon

# Lint
ruff check src/
ruff format src/

# Run tests
pytest tests/ -v
pytest tests/test_cli.py -v          # Single test file
pytest tests/test_cli.py::test_name  # Single test
```

## Architecture

sttd is a speech-to-text daemon for Wayland/Hyprland that uses faster-whisper for transcription.

### Component Flow

```
CLI Command → Unix Socket IPC → Daemon
                                  ├── Server (server.py) - Unix socket handler
                                  ├── Recorder (recorder.py) - sounddevice audio capture
                                  ├── Transcriber (transcriber.py) - faster-whisper
                                  ├── Injector (injector.py) - wtype/wl-clipboard text input
                                  └── TrayIcon (tray.py) - D-Bus StatusNotifierItem
```

### Key Design Patterns

**Streaming Transcription**: Uses a sliding window approach. Audio chunks accumulate up to `max_window` seconds, then oldest chunks are trimmed. Text from trimmed chunks is preserved as `initial_prompt` context for the model.

**State Machine**: Daemon states are `IDLE → RECORDING → TRANSCRIBING → IDLE`. The tray icon reflects state visually.

**IPC Protocol**: JSON over Unix domain socket at `~/.cache/sttd/control.sock`. Commands: `toggle`, `status`, `stop`.

**GPU Detection**: Uses `ctranslate2.get_cuda_device_count()` for CUDA detection (not torch).

### Config Locations

- Config: `~/.config/sttd/config.toml`
- Socket: `~/.cache/sttd/control.sock`
- PID: `~/.cache/sttd/daemon.pid`

## Code Style

- Line length: 100 chars
- Ruff linting with rules: E, F, I, N, W, UP
- Type hints throughout
- Minimal comments - code should be self-explanatory
