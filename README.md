# voiced - Voice Daemon

A voice daemon for Linux/Wayland providing both **Speech-to-Text (STT)** and **Text-to-Speech (TTS)**. Optimized for Hyprland with hotkey support.

## Features

### STT (Speech-to-Text)
- Hotkey toggle - bind `voiced toggle` to any key
- Record-then-transcribe workflow
- Push-to-talk controls: separate start/release commands for compositor bindings
- Desktop recording window (`voiced desktop`)
- Local and remote transcription modes
- Optional screen annotation and meeting transcription workflows (see capabilities below)
- System tray integration with modern icons
- GPU acceleration with CPU fallback
- NVIDIA Parakeet-TDT v3 backend (low WER, hallucination-resistant on silence)
- Audio feedback on start/stop
- Clipboard text injection
- File transcription with timestamps
- Speaker diarization with spectral clustering
- Voice profile matching for speaker identification

### TTS (Text-to-Speech)
- High-quality synthesis using Kokoro-82M
- 28 English voice packs (American + British accents)
- Low-latency streaming playback
- Speak from clipboard or stdin
- Save audio to file
- HTTP API for remote synthesis

## Installation

### As a home-manager service

The module ships with the flake, so there is nothing to write:

```nix
# flake.nix
inputs.voiced.url = "github:sdelcore/voiced";

# home/<hostname>.nix
imports = [ inputs.voiced.homeModules.default ];
services.voiced.enable = true;
```

That installs voiced with `wtype` and `wl-clipboard`, and runs it as a
user unit with `--http`.

`settings` writes `~/.config/voiced/config.toml` and is empty by
default, which leaves the defaults in `src/voiced/config.py` in charge.
Set only the keys you want to differ — a restated default stops tracking
this repo the moment the value here changes, and nothing reports it.

Options are declared in [`nix/hm-module.nix`](nix/hm-module.nix); read
that rather than a copy here. The ones worth knowing: `settings`,
`http`, `startAtLogin` (turn off when a compositor `exec-once` owns the
lifecycle), `cudaDevice`, and `waylandDisplay`.

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
```

## Usage

### Desktop Client (STT)

Start the single desktop client, which records locally and uses the configured
OpenAI-compatible audio endpoint for transcription and speech:
```bash
voiced start           # Run in foreground
voiced start --daemon  # Run in background
```

Toggle dictation (bind this to a hotkey):
```bash
voiced toggle
```

For push-to-talk bindings, use one command on key press and the other on
release. Control commands target the default client socket; start the client first.
```bash
voiced start-recording   # key press: begin capture
voiced stop-recording    # key release: stop and transcribe
voiced cancel            # discard the current recording
voiced copy-last         # copy the most recent transcript
```

Open the GTK desktop interface with `voiced desktop`. Check status with
`voiced status` and stop the client with `voiced stop`.

### Voice Assistant

The assistant is a separate press-to-talk workflow from dictation. It sends
recognized text to the configured OpenAI-compatible chat-completions endpoint
and speaks the response using voiced's local TTS. Configure `[assistant]` in
`~/.config/voiced/config.toml`; export its `api_key_env` variable in the
environment of the voiced client before starting it. Keep the endpoint empty
to leave the assistant disabled.

Bind separate key-down and key-up actions in Hyprland:

```ini
bind = SUPER, A, exec, voiced assistant-start
bindr = SUPER, A, exec, voiced assistant-stop
```

Hold the key while speaking and release it to submit the prompt. Conversation
messages are saved locally and reused for context. Protect that file and review
the endpoint's data-handling policy before sending sensitive prompts.


### Text-to-Speech

Speak text:
```bash
voiced speak "Hello world"
voiced speak --stream "Low latency streaming"   # Streaming playback
voiced speak --clipboard                         # Speak clipboard contents
echo "Hello" | voiced speak --stdin              # From stdin
voiced speak "Save this" -o output.wav           # Save to file
voiced speak "Hello" --voice am_michael          # Different voice
```

Manage voice presets:
```bash
voiced voices                       # List available voices
voiced voices download af_heart     # Download a voice pack
voiced voices info af_heart         # Show voice details
voiced voices remove af_heart       # Remove cached voice
```

### Standalone HTTP Server

Run a headless backend, for example on Nightman, to serve LiteLLM audio routes:

```bash
voiced server --host 0.0.0.0 --port 8765
```

The desktop client can use this server as its configured `client.audio_base_url`
(for example, `http://nightman.tap:8765/v1`) while sending the configured
`stt_model` and `tts_model` to LiteLLM. Set `client.api_key_env` to the name of
an environment variable containing the API key; export it before starting the
client. These names must be models/routes available through your LiteLLM setup.

The standalone server remains separate from the desktop client and can also be
run locally with the defaults (`127.0.0.1:8765`). Use `--host 0.0.0.0` only
when remote clients need to reach it.

**Test with curl:**
```bash
# Transcription
curl -X POST -H "Content-Type: audio/wav" \
  --data-binary @audio.wav \
  http://localhost:8765/transcribe

# TTS synthesis
curl -X POST -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "af_heart"}' \
  http://localhost:8765/synthesize --output speech.wav
```

### OpenAI-compatible API

The server also answers the OpenAI audio endpoints, so any client that can talk
to OpenAI TTS/STT can use voiced by pointing its base URL at `/v1`:

| Endpoint | Notes |
|----------|-------|
| `POST /v1/audio/speech` | `input`, `voice`, `speed`, `response_format` (`mp3` default, plus `wav`, `opus`, `aac`, `flac`, `pcm`). `model` is accepted and ignored |
| `POST /v1/audio/transcriptions` | multipart with a `file` part, or JSON with a base64 `input_audio`. `response_format` `json` (default) or `text` |
| `GET /v1/audio/voices` | `{"voices": [{"id", "name"}]}` — the Kokoro voice list, for client voice pickers |

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "voice": "af_heart"}' \
  http://localhost:8765/v1/audio/speech --output speech.mp3

curl -X POST -F file=@audio.wav \
  http://localhost:8765/v1/audio/transcriptions
```

Speaker diarization is off on `/v1/audio/transcriptions` — the OpenAI response
has nowhere to carry speakers. Use `/transcribe` for segments and speaker IDs.

Example — Open WebUI, Admin → Settings → Audio:

```
STT Engine: OpenAI     Base URL: http://<host>:8765/v1     Key: any non-empty value
TTS Engine: OpenAI     Base URL: http://<host>:8765/v1     Voice: af_heart
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

### Desktop CLI, History, Annotation, and Meetings

Launch the GTK desktop companion with `voiced desktop`. It connects to the
running local daemon for recording controls and status; install GTK 3/PyGObject
and run it in a graphical session. It includes Start/Stop/Cancel, Copy last,
microphone level and elapsed-time indicators, transcript/history views, a
context selector, meeting capture, and screen annotation controls. Its config
page edits the local configuration file; it does not start a separate daemon.

The desktop UI connects to the default control socket used by `voiced start`.
Audio endpoint and model routing are configured under `[client]` in
`~/.config/voiced/config.toml`; it does not select a separate client daemon.

For Hyprland push-to-talk, bind start and stop separately so the release event
ends capture and starts transcription:

```ini
bind = SUPER, R, exec, voiced start-recording
bindr = SUPER, R, exec, voiced stop-recording
```

`bindr` runs on key release. `voiced cancel` discards a recording; `voiced
toggle` remains available for a single toggle binding.

The interactive experimental WebRTC client (`voiced webrtc-client`, optionally
with `--server URL`) streams bidirectional audio and prints partial/final STT
results; its commands are `r` to record, `s` to stop, `t <text>` to send TTS,
and `q` to quit. Treat streaming as provisional, not as the normal desktop
record-then-transcribe path. It requires a WebRTC-enabled server and adds
streaming/network overhead; the exact compute and bandwidth cost depends on
the server, codec, and session, so no fixed cost or latency is promised.

The context selector lists `default` and configured
`[transcription.context_replacements.<mode>]` modes. These are exact text
replacements applied during transcription, not prompts or model-training
contexts. Configure global replacements under `[transcription.replacements]`
and mode-specific replacements under a named table, for example:

```toml
[transcription.context_replacements.work]
"issue tracker" = "Linear"
```

History is displayed in the desktop but is not persisted by default. The
history store supports opt-in persistence, bounded to 200 entries, in
`~/.cache/voiced/desktop-history.json`; the file is mode 0600. Treat persisted
transcripts as sensitive local text. Clearing displayed history removes its
entries from the desktop history store; daemon history is distinct. Activating
a history row allows a local correction/replacement without changing the
daemon's canonical transcript.

Meeting capture is an explicit desktop action labeled “Record meeting (mic +
system audio)”. It captures microphone and system-loopback audio as separate
tracks; loopback uses a configured/native PulseAudio or PipeWire source. After
stopping, each available track is transcribed separately and results are
reported per source, with speaker segments when diarization is enabled. This
is not an automatic meeting-app integration or always-on recording; selecting
a remote backend sends the audio to that server.

Screen annotation is also explicitly started with “Capture screen region +
transcript”; it requires a Wayland session and `slurp`, `wf-recorder`, `ffmpeg`,
GTK 3/PyGObject, and GtkLayerShell. Select the region when prompted, draw with
the primary mouse button, then use Stop capture or Cancel capture. Successful
exports are an MP4 screen recording, a text transcript of its audio, JSON
drawing keyframes, and annotated PNG frames for completed strokes in the
annotation output directory. Transcription follows the configured local/remote
backend; canceling discards the capture.

### File Transcription

Transcribe an audio file:
```bash
voiced transcribe audio.wav                    # Output to stdout
voiced transcribe audio.mp3 -o transcript.txt  # Output to file
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
# Stop the STT/TTS inference worker process after this many idle minutes,
# releasing all model VRAM (0 = never). Shared by STT and TTS.
unload_timeout_minutes = 15

[transcription]
device = "auto"          # auto, cuda, cpu
language = "en"          # advisory only; Parakeet TDT v3 auto-detects

# Fix words the model habitually mishears (case-insensitive,
# word-boundary matched)
[transcription.replacements]
"cloud code" = "Claude Code"
"hyperland" = "Hyprland"

[audio]
sample_rate = 16000
channels = 1
device = "default"
beep_enabled = true

[tts]
enabled = true              # Enable TTS (requires kokoro)
device = "auto"             # auto, cuda, cpu
default_voice = "af_heart"  # see `voiced voices list`
speed = 1.0                 # Speech rate multiplier

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
audio_base_url = "http://127.0.0.1:8765/v1"
stt_model = "your-transcription-model"
tts_model = "your-speech-model"
api_key_env = "LITELLM_API_KEY"
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

# Use defaults (Parakeet-TDT v3, auto device detection)
transcriber = Transcriber()
text = transcriber.transcribe_file("audio.wav")
print(text)

# Or with custom config
config = TranscriptionConfig(device="cuda", language="en")
transcriber = Transcriber(config)
text = transcriber.transcribe_file("audio.wav")
print(text)
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
config = TTSConfig(default_voice="af_heart", device="cuda")
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

The backends are fixed — one STT model, one TTS model, not configurable.

| Role | Model | Params | Notes |
|------|-------|--------|-------|
| STT | `nvidia/parakeet-tdt-0.6b-v3` | 600M | 25 European languages, auto-detect, WER 6.3% on Open ASR, ~3300x realtime |
| TTS | `hexgrad/Kokoro-82M` | 82M | ~330MB, Apache 2.0, 24kHz output, runs on CPU too |

TTS voices are ~500KB packs downloaded on demand. Prefix encodes
accent + gender: `af_`/`am_` American, `bf_`/`bm_` British.
See `voiced voices list` for all 28.

## Architecture

**Desktop mode** (`voiced start`):
```
CLI (voiced toggle) → Unix Socket → Daemon
                                     ├── Recorder (sounddevice)
                                     ├── Transcriber (Parakeet-TDT)
                                     ├── Injector (wl-clipboard)
                                     └── Tray Icon (D-Bus SNI)
```

**Desktop client and standalone server**:
```
voiced start → local controls and recording → LiteLLM audio endpoint
voiced server → OpenAI-compatible audio routes → local STT/TTS backend
```

**Headless server mode** (`voiced server`):
```
HTTP POST /transcribe → HTTP Server → Transcriber
HTTP POST /synthesize → HTTP Server → Synthesizer
```

**LiteLLM desktop audio path**:
```
voiced start → local recorder → HTTP audio request → LiteLLM
voiced server on Nightman ← serves OpenAI-compatible audio routes
```

**File transcription with diarization:**
```
CLI (voiced transcribe/record --annotate) → Transcriber → SpeakerDiarizer
                                                           └── SpeechBrain ECAPA-TDNN
                                                           └── Spectral Clustering
```

## License

MIT
