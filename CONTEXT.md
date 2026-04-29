# Voiced

A voice daemon for Wayland that captures speech and synthesises it. The user toggles a recording, the daemon transcribes it when toggled again, and the text lands on the clipboard. The same daemon also exposes text-to-speech.

## Language

**Recording Session**:
One full toggle-to-toggle cycle: start recording → stop → transcribe → emit result. Sequential — only one session is in flight at a time.
_Avoid_: capture, dictation, take.

**Session State**:
Where a Recording Session currently is in its lifecycle. Three values: `IDLE`, `RECORDING`, `TRANSCRIBING`.
_Avoid_: status, mode.

**Toggle Outcome**:
What happened when the user pressed the toggle. `Started` (idle → recording), `Stopped` (recording → transcribing), or `Rejected` (toggle ignored, e.g. mid-transcription).
_Avoid_: result, response.

**Transcription Result**:
The text and timing data produced by a completed Recording Session. Carries `text`, `segments`, `duration`, `started_at`. Does *not* carry the raw audio.
_Avoid_: transcript, output.

**Voice Profile**:
A speaker enrolment used for diarisation. Stored embedding plus metadata.
_Avoid_: speaker, identity, user.

**Voice Preset**:
A reference voice used by the TTS engine to synthesise speech. Cached per voice name.
_Avoid_: speaker, model voice.

## Relationships

- A **Recording Session** has exactly one **Session State** at any time
- A **Recording Session** ends by emitting either a **Transcription Result** or an error
- Every user `toggle` returns a **Toggle Outcome**
- A **Transcription Result** may be enriched with **Voice Profile** matches when diarisation is enabled
- The TTS path is independent of Recording Sessions and operates over **Voice Presets**

## Example dialogue

> **Dev:** "If the user toggles while we're still transcribing, do we queue the new **Recording Session**?"
> **Domain expert:** "No — sequential only. The toggle returns a `Rejected` **Toggle Outcome** so the CLI can tell them why nothing started."
> **Dev:** "And the **Transcription Result** — does it ship the audio bytes?"
> **Domain expert:** "No. Just text, segments, duration, and the start timestamp. If we ever want re-transcription we'll add a separate accessor."

## Flagged ambiguities

- "speaker" was used for both **Voice Profile** (diarisation) and **Voice Preset** (TTS) — resolved: these are distinct concepts and do not share storage.
