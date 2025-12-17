# sttd - Speech-to-Text Daemon

A speech-to-text daemon for Linux/Wayland, optimized for Hyprland. Bind a hotkey to toggle recording, and transcribed text is automatically typed at your cursor.

## Features

- **Real-time streaming transcription** - See text appear as you speak
- **Hotkey toggle** - Bind `sttd toggle` to any key (e.g., Super+R)
- **System tray integration** - Visual state indicator (idle/recording/transcribing)
- **GPU acceleration** - Automatic CUDA detection with CPU fallback
- **Multiple Whisper models** - From tiny (~75MB) to large-v3 (~3GB)
- **Audio feedback** - Beeps for start, stop, success, and error states
- **Smart text injection** - Uses `wtype` for native Wayland input, with clipboard fallback
- **File transcription** - Transcribe audio files from the command line
- **Speaker identification** - Register voice profiles and identify speakers in transcriptions

## Installation

### Prerequisites (NixOS)

This project uses Nix flakes for development. With direnv installed:

```bash
cd sttd
direnv allow
```

This will automatically set up the development environment with:
- Python 3.12
- uv (package manager)
- portaudio (for audio recording)
- wtype (for text injection)
- wl-clipboard (clipboard fallback)
- ffmpeg (audio format support)

### Setup

```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Usage

### Daemon Mode

Start the daemon:
```bash
sttd start           # Run in foreground
sttd start --daemon  # Run in background
```

Toggle recording (bind this to a hotkey):
```bash
sttd toggle
```

Check status:
```bash
sttd status
```

Stop the daemon:
```bash
sttd stop
```

### Hyprland Configuration

Add to `~/.config/hypr/hyprland.conf`:
```
bind = SUPER, R, exec, sttd toggle
```

### File Transcription

Transcribe an audio file:
```bash
sttd transcribe audio.wav                    # Output to stdout
sttd transcribe audio.mp3 -o transcript.txt  # Output to file
sttd transcribe audio.wav --model large-v3   # Use a specific model
```

### Speaker Identification

Register voice profiles to identify speakers in transcriptions:

```bash
# Register a speaker from an audio file
sttd register alice -f alice_sample.wav

# Or record directly from microphone
sttd register bob --record --duration 15

# List registered profiles
sttd profiles

# Transcribe with speaker labels
sttd transcribe meeting.wav --annotate
```

Output with `--annotate`:
```
[0.00-2.50] alice: Hello, how are you today?
[2.50-5.00] bob: I'm doing great, thanks for asking.
[5.00-8.20] alice: That's wonderful to hear.
```

### Configuration

Create a config file:
```bash
sttd config --init
```

Configuration file: `~/.config/sttd/config.toml`

```toml
[transcription]
model = "base"           # tiny, base, small, medium, large-v3
device = "auto"          # auto, cuda, cpu
compute_type = "auto"    # auto, float16, int8, float32
language = "en"
streaming = true         # Real-time transcription
chunk_duration = 2.0     # Seconds per streaming chunk

[audio]
sample_rate = 16000
channels = 1
device = "default"
beep_enabled = true

[output]
method = "wtype"         # wtype, clipboard, both

[diarization]
device = "auto"          # auto, cuda, cpu
similarity_threshold = 0.5  # Speaker matching threshold (0-1)
min_segment_duration = 0.5  # Min segment length for identification
```

### List Audio Devices

```bash
sttd devices
```

## Models

Available Whisper models (via faster-whisper):

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | ~75MB | Fastest | Lower |
| base | ~150MB | Fast | Good |
| small | ~500MB | Medium | Better |
| medium | ~1.5GB | Slower | High |
| large-v3 | ~3GB | Slowest | Highest |

## Architecture

```
CLI (sttd toggle) → Unix Socket → Daemon
                                    ├── Recorder (sounddevice)
                                    ├── Transcriber (faster-whisper)
                                    ├── Injector (wtype/wl-clipboard)
                                    └── Tray Icon (D-Bus SNI)

CLI (sttd transcribe --annotate) → Transcriber → SpeakerIdentifier
                                                   └── SpeechBrain ECAPA-TDNN
```

## License

MIT
