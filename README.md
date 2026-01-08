# voiced - Voice Daemon

A voice daemon for Linux/Wayland providing both **Speech-to-Text (STT)** and **Text-to-Speech (TTS)**. Optimized for Hyprland with hotkey support.

## Features

### STT (Speech-to-Text)
- Hotkey toggle - bind `voiced toggle` to any key
- Record-then-transcribe workflow
- **Client-server mode** - run transcription on a remote GPU server
- System tray integration with modern icons
- GPU acceleration with CPU fallback
- Multiple Whisper models (tiny to large-v3)
- Audio feedback on start/stop
- Clipboard text injection
- File transcription with timestamps
- Speaker diarization with spectral clustering
- Voice profile matching for speaker identification

### TTS (Text-to-Speech)
- High-quality synthesis using VibeVoice
- Multiple voice presets (emma, mike, carter, davis, frank, grace)
- Low-latency streaming playback
- Speak from clipboard or stdin
- Save audio to file
- HTTP API for remote synthesis

## Installation

### Prerequisites (NixOS)

This project uses Nix flakes for development. With direnv installed:

```bash
cd voiced
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

# For TTS support, also install VibeVoice:
pip install git+https://github.com/microsoft/VibeVoice.git
```

## Usage

### Daemon Mode (STT)

Start the daemon:
```bash
voiced start           # Run in foreground
voiced start --daemon  # Run in background
```

Toggle recording (bind this to a hotkey):
```bash
voiced toggle
```

Check status:
```bash
voiced status
```

Stop the daemon:
```bash
voiced stop
```

### Text-to-Speech

Speak text:
```bash
voiced speak "Hello world"
voiced speak --stream "Low latency streaming"   # Streaming playback
voiced speak --clipboard                         # Speak clipboard contents
echo "Hello" | voiced speak --stdin              # From stdin
voiced speak "Save this" -o output.wav           # Save to file
voiced speak "Hello" --voice mike                # Different voice
```

Manage voice presets:
```bash
voiced voices                    # List available voices
voiced voices download emma      # Download a voice preset
voiced voices info emma          # Show voice details
voiced voices remove emma        # Remove cached voice
```

### Daemon with HTTP API

The daemon can also expose an HTTP API for remote transcription and TTS requests:

```bash
voiced start --http                      # Enable HTTP API on 127.0.0.1:8765
voiced start --http --http-host 0.0.0.0  # Accept remote connections
voiced start --http --http-port 9000     # Custom port
```

Or enable via config (`~/.config/voiced/config.toml`):
```toml
[daemon]
http_enabled = true
http_host = "0.0.0.0"
http_port = 8765
```

### Standalone HTTP Server

For headless deployments without a display (e.g., dedicated GPU server):

```bash
voiced server                     # Local only (127.0.0.1:8765)
voiced server --host 0.0.0.0      # Accept remote connections
voiced server --port 9000         # Custom port
voiced server -d                  # Run in background
```

### Remote Client

Record locally on an underpowered client, send to a remote server for transcription:

```bash
voiced client --server http://192.168.1.100:8765
voiced client -d                  # Run in background

# Or set server URL via environment
VOICED_SERVER_URL=http://server:8765 voiced client
```

The client records audio locally, sends it to the server for transcription, and copies the result to clipboard.

**Test with curl:**
```bash
# Transcription
curl -X POST -H "Content-Type: audio/wav" \
  --data-binary @audio.wav \
  http://localhost:8765/transcribe

# TTS synthesis
curl -X POST -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "emma"}' \
  http://localhost:8765/synthesize --output speech.wav
```

### Hyprland Configuration

Add to `~/.config/hypr/hyprland.conf`:
```
bind = SUPER, R, exec, voiced toggle
```

### Record and Transcribe

Record from microphone and transcribe with timestamps:
```bash
voiced record                         # Record until Ctrl+C, output timestamps
voiced record -o transcript.txt       # Save to file
voiced record --annotate              # With speaker diarization
voiced record --annotate --num-speakers 2
```

### File Transcription

Transcribe an audio file:
```bash
voiced transcribe audio.wav                    # Output to stdout
voiced transcribe audio.mp3 -o transcript.txt  # Output to file
voiced transcribe audio.wav --model large-v3   # Use a specific model
voiced transcribe meeting.wav --annotate --num-speakers 3  # With diarization
```

### Speaker Identification

Register voice profiles to identify speakers in transcriptions:

```bash
# Register a speaker from an audio file
voiced register alice -f alice_sample.wav

# Or record directly from microphone
voiced register bob --record --duration 15

# List registered profiles
voiced profiles

# Transcribe with speaker labels
voiced transcribe meeting.wav --annotate
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
voiced config --init
```

Configuration file: `~/.config/voiced/config.toml`

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

[tts]
enabled = true           # Enable TTS (requires VibeVoice)
model = "microsoft/VibeVoice-Realtime-0.5B"
device = "auto"          # auto, cuda, mps, cpu
default_voice = "emma"   # carter, davis, emma, frank, grace, mike
cfg_scale = 1.5          # Classifier-free guidance scale
unload_timeout_minutes = 60  # Auto-unload after inactivity

[diarization]
device = "auto"          # auto, cuda, cpu
similarity_threshold = 0.5  # Profile matching threshold (0-1)
min_segment_duration = 0.5  # Min segment length for embedding
# num_speakers = 2       # Set if known, leave unset for auto-detect
# clustering_threshold = 0.7  # Threshold when num_speakers is None

[daemon]
http_enabled = false     # Start HTTP server alongside Unix socket
# http_host = "0.0.0.0"  # Override server.host for daemon HTTP
# http_port = 8765       # Override server.port for daemon HTTP

[server]
host = "127.0.0.1"       # 0.0.0.0 to accept remote connections
port = 8765

[client]
server_url = "http://127.0.0.1:8765"
timeout = 60.0           # Request timeout in seconds
```

### List Audio Devices

```bash
voiced devices
```

## Library Usage

voiced can be used as a Python library in your own projects:

```bash
pip install voiced
```

### Basic Transcription

```python
from voiced import Transcriber, TranscriptionConfig

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
from voiced import Transcriber

transcriber = Transcriber()
segments = transcriber.transcribe_file_with_segments("audio.wav")

for start, end, text in segments:
    print(f"[{start:.2f}-{end:.2f}] {text}")
```

### Text-to-Speech

```python
from voiced.synthesizer import Synthesizer, TTSConfig

# Create synthesizer with config
config = TTSConfig(default_voice="emma", device="cuda")
synth = Synthesizer(config)

# Generate audio
audio = synth.synthesize("Hello, world!")

# Or stream for low latency
for chunk in synth.synthesize_streaming("Hello, world!"):
    # Process audio chunks as they arrive
    play_audio(chunk)
```

### Speaker Diarization

```python
from voiced import Transcriber, SpeakerDiarizer, align_transcription_with_diarization

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
from voiced import Transcriber, SpeakerIdentifier, ProfileManager

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
from voiced import Recorder, AudioConfig

config = AudioConfig(sample_rate=16000, channels=1)
recorder = Recorder(config)

recorder.start()
# ... recording ...
audio = recorder.stop()  # Returns numpy array
```

## Models

### STT Models (faster-whisper)

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | ~75MB | Fastest | Lower |
| base | ~150MB | Fast | Good |
| small | ~500MB | Medium | Better |
| medium | ~1.5GB | Slower | High |
| large-v3 | ~3GB | Slowest | Highest |

### TTS Model (VibeVoice)

| Model | Size | Quality |
|-------|------|---------|
| VibeVoice-Realtime-0.5B | ~1GB | High quality, near real-time |

Available voices: `carter`, `davis`, `emma`, `frank`, `grace`, `mike`

## Architecture

**Desktop mode** (`voiced start`):
```
CLI (voiced toggle) → Unix Socket → Daemon
                                     ├── Recorder (sounddevice)
                                     ├── Transcriber (faster-whisper)
                                     ├── Injector (wl-clipboard)
                                     └── Tray Icon (D-Bus SNI)
```

**Desktop + HTTP mode** (`voiced start --http`):
```
CLI (voiced toggle) → Unix Socket → Daemon
                                     ├── Recorder (sounddevice)
                                     ├── Transcriber (faster-whisper) ←─┐
                                     ├── Synthesizer (VibeVoice) ←──────┤ shared
                                     ├── Injector (wl-clipboard)        │
                                     ├── Tray Icon (D-Bus SNI)          │
                                     └── HTTP Server ───────────────────┘
                                           ↑
                 HTTP /transcribe, /synthesize ─┘ (from other services)
```

**Headless server mode** (`voiced server`):
```
HTTP POST /transcribe → HTTP Server → Transcriber
HTTP POST /synthesize → HTTP Server → Synthesizer
```

**Remote client mode** (`voiced client`):
```
Client Machine                       Server Machine (GPU)
─────────────────                    ────────────────────
voiced client                        voiced server / voiced start --http
  ├── Recorder ──── WAV ──── HTTP POST /transcribe ────→ Transcriber
  ├── Tray Icon                                              │
  └── Injector ←──── text ←───────────────────────────────────
```

**File transcription with diarization:**
```
CLI (voiced transcribe/record --annotate) → Transcriber → SpeakerDiarizer
                                                           └── SpeechBrain ECAPA-TDNN
                                                           └── Spectral Clustering
```

## License

MIT
