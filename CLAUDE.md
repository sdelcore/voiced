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
- **STT (Speech-to-Text)**: Using NVIDIA Parakeet-TDT (NeMo) for transcription
- **TTS (Text-to-Speech)**: Using Kokoro-82M for synthesis

### Component Flow

```
CLI Command → Unix Socket IPC → Daemon
                                  ├── Server (server.py) - Unix socket handler
                                  ├── Recorder (recorder.py) - sounddevice audio capture
                                  ├── WorkerHost (worker_host.py) - Inference Worker supervisor + proxies
                                  ├── Injector (injector.py) - wl-clipboard text injection
                                  └── TrayIcon (tray.py) - D-Bus StatusNotifierItem
                                        │ pipe IPC (spawn)
                                  Inference Worker (worker.py) - disposable child process
                                  ├── Transcriber (transcriber.py) - Parakeet-TDT STT
                                  ├── Synthesizer (synthesizer.py) - Kokoro TTS
                                  └── Diarizer (diarizer.py) - SpeechBrain speaker ID
```

### Key Design Patterns

**Record-then-Transcribe (STT)**: Toggle once to start recording (RED tray icon), toggle again to stop and begin batch transcription (YELLOW icon). When complete (BLUE icon), text is copied to clipboard. Starting a recording also warms the Inference Worker in the background so the stop-toggle doesn't pay worker-spawn + model-load latency.

**Disposable Inference Worker**: STT/TTS inference runs in a child process spawned lazily on the first request (`worker_host.py` parent side, `worker.py` child side). After the shared idle timeout (default 15 min) with no active operations, the worker process is terminated — process exit is what reliably releases VRAM; `torch.cuda.empty_cache()` in a live process does not. The next request transparently starts a fresh worker. The parent process must never import torch/NeMo/Kokoro (guarded by `tests/test_parent_imports.py`).

**HTTP Client-Server Mode**: For remote STT/TTS:
- `http_server.py` - HTTP server with `/transcribe`, `/synthesize`, `/health` endpoints
- `http_client.py` - HTTP client for remote connections
- WebSocket support for streaming TTS at `/synthesize/stream`

**State Machine**: Daemon states are `IDLE → RECORDING → TRANSCRIBING → IDLE`. The tray icon reflects state visually.

**IPC Protocol**: JSON over Unix domain socket at `~/.cache/voiced/control.sock`. Commands: `toggle`, `status`, `stop`.

**GPU Detection**: Uses `torch.cuda.is_available()` for CUDA detection (NeMo is torch-native).

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
