# sttd - Speech-to-Text Daemon

A speech-to-text daemon for Linux/Wayland, optimized for Hyprland. Bind a hotkey to toggle recording, and transcribed text is automatically typed at your cursor.

## Features

- Real-time streaming transcription
- Hotkey toggle - bind `sttd toggle` to any key
- System tray integration
- GPU acceleration with CPU fallback
- Multiple Whisper models (tiny to large-v3)
- Audio feedback on start/stop
- Text injection via wtype with clipboard fallback
- File transcription
- Speaker diarization with profile matching

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
chunk_duration = 1.0     # Seconds per streaming chunk
max_window = 30.0        # Max seconds in sliding window
beam_size = 1            # Beam size (1 = greedy for speed)
context_words = 200      # Context preserved after buffer trim

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

## Library Usage

sttd can be used as a Python library in your own projects:

```bash
pip install sttd
```

### Basic Transcription

```python
from sttd import Transcriber, TranscriptionConfig

# Use defaults (base model, auto device detection)
transcriber = Transcriber()
text = transcriber.transcribe_file("audio.wav")
print(text)

# Or with custom config
config = TranscriptionConfig(model="large-v3", device="cuda", language="en")
transcriber = Transcriber(config)
text = transcriber.transcribe_file("audio.wav")
```

### Transcribe with Timestamps

```python
from sttd import Transcriber

transcriber = Transcriber()
segments = transcriber.transcribe_file_with_segments("audio.wav")

for start, end, text in segments:
    print(f"[{start:.2f}-{end:.2f}] {text}")
```

### Speaker Identification

```python
from sttd import Transcriber, SpeakerIdentifier, ProfileManager

# First, transcribe with segments
transcriber = Transcriber()
segments = transcriber.transcribe_file_with_segments("meeting.wav")

# Then identify speakers
identifier = SpeakerIdentifier()
profiles = ProfileManager().load_all()
identified = identifier.identify_segments("meeting.wav", segments, profiles)

for seg in identified:
    print(f"[{seg.start:.1f}-{seg.end:.1f}] {seg.speaker}: {seg.text}")
```

### Audio Recording

```python
from sttd import Recorder, AudioConfig

config = AudioConfig(sample_rate=16000, channels=1)
recorder = Recorder(config)

recorder.start()
# ... recording ...
audio = recorder.stop()  # Returns numpy array
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
