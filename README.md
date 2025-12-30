# sttd - Speech-to-Text Daemon

A speech-to-text daemon for Linux/Wayland, optimized for Hyprland. Bind a hotkey to toggle recording, and transcribed text is copied to your clipboard.

## Features

- Hotkey toggle - bind `sttd toggle` to any key
- Record-then-transcribe workflow
- **Client-server mode** - run transcription on a remote GPU server
- System tray integration
- GPU acceleration with CPU fallback
- Multiple Whisper models (tiny to large-v3)
- Audio feedback on start/stop
- Clipboard text injection
- File transcription with timestamps
- Speaker diarization with spectral clustering
- Voice profile matching for speaker identification

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
- wl-clipboard (for clipboard injection)
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

### Client-Server Mode

Run transcription on a powerful remote machine (GPU server) while recording locally on an underpowered client.

**On the server (GPU machine):**
```bash
sttd server                     # Local only (127.0.0.1:8765)
sttd server --host 0.0.0.0      # Accept remote connections
sttd server --port 9000         # Custom port
sttd server -d                  # Run in background
```

**On the client:**
```bash
sttd client --server http://192.168.1.100:8765
sttd client -d                  # Run in background

# Or set server URL via environment
STTD_SERVER_URL=http://server:8765 sttd client
```

The client records audio locally, sends it to the server for transcription, and copies the result to clipboard.

**Test with curl:**
```bash
curl -X POST -H "Content-Type: audio/wav" \
  --data-binary @audio.wav \
  http://localhost:8765/transcribe
```

### Hyprland Configuration

Add to `~/.config/hypr/hyprland.conf`:
```
bind = SUPER, R, exec, sttd toggle
```

### Record and Transcribe

Record from microphone and transcribe with timestamps:
```bash
sttd record                         # Record until Ctrl+C, output timestamps
sttd record -o transcript.txt       # Save to file
sttd record --annotate              # With speaker diarization
sttd record --annotate --num-speakers 2
```

### File Transcription

Transcribe an audio file:
```bash
sttd transcribe audio.wav                    # Output to stdout
sttd transcribe audio.mp3 -o transcript.txt  # Output to file
sttd transcribe audio.wav --model large-v3   # Use a specific model
sttd transcribe meeting.wav --annotate --num-speakers 3  # With diarization
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

[audio]
sample_rate = 16000
channels = 1
device = "default"
beep_enabled = true

[diarization]
device = "auto"          # auto, cuda, cpu
similarity_threshold = 0.5  # Profile matching threshold (0-1)
min_segment_duration = 0.5  # Min segment length for embedding
# num_speakers = 2       # Set if known, leave unset for auto-detect
# clustering_threshold = 0.7  # Threshold when num_speakers is None

[server]
host = "127.0.0.1"       # 0.0.0.0 to accept remote connections
port = 8765

[client]
server_url = "http://127.0.0.1:8765"
timeout = 60.0           # Request timeout in seconds
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

### Speaker Diarization

```python
from sttd import Transcriber, SpeakerDiarizer, align_transcription_with_diarization

transcriber = Transcriber()
diarizer = SpeakerDiarizer()

# Get transcription segments and speaker segments
segments = transcriber.transcribe_file_with_segments("meeting.wav")
speaker_segments = diarizer.diarize("meeting.wav", num_speakers=2)

# Align transcription with speaker labels
aligned = align_transcription_with_diarization(segments, speaker_segments)

for seg in aligned:
    print(f"[{seg.start:.1f}-{seg.end:.1f}] {seg.speaker}: {seg.text}")
```

### Speaker Identification with Profiles

```python
from sttd import Transcriber, SpeakerIdentifier, ProfileManager

# First, transcribe with segments
transcriber = Transcriber()
segments = transcriber.transcribe_file_with_segments("meeting.wav")

# Then identify speakers using registered voice profiles
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

**Local mode** (`sttd start`):
```
CLI (sttd toggle) → Unix Socket → Daemon
                                    ├── Recorder (sounddevice)
                                    ├── Transcriber (faster-whisper)
                                    ├── Injector (wl-clipboard)
                                    └── Tray Icon (D-Bus SNI)
```

**Client-server mode** (`sttd server` + `sttd client`):
```
Client Machine                       Server Machine (GPU)
─────────────────                    ────────────────────
sttd client                          sttd server
  ├── Recorder ──── WAV ──── HTTP POST /transcribe ────→ Transcriber
  ├── Tray Icon                                              │
  └── Injector ←──── text ←───────────────────────────────────
```

**File transcription with diarization:**
```
CLI (sttd transcribe/record --annotate) → Transcriber → SpeakerDiarizer
                                                          └── SpeechBrain ECAPA-TDNN
                                                          └── Spectral Clustering
```

## License

MIT
