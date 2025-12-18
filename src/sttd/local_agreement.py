"""LocalAgreement-2 algorithm for streaming transcription.

This module implements the LocalAgreement algorithm which only confirms text
when consecutive transcription iterations agree on a prefix. This eliminates
text flickering during streaming transcription.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

SENTENCE_ENDINGS = frozenset(".!?")


class WordInfo(NamedTuple):
    """Word with timestamp information from faster-whisper."""

    word: str
    start: float  # Start time in seconds relative to buffer start
    end: float  # End time in seconds relative to buffer start
    probability: float


@dataclass
class TranscriptResult:
    """Result of a single transcription pass."""

    words: list[WordInfo]
    text: str

    @property
    def word_texts(self) -> list[str]:
        """Extract just the word strings for comparison."""
        return [w.word.strip() for w in self.words]


@dataclass
class TrimInfo:
    """Information about where to trim the audio buffer."""

    should_trim: bool
    word_index: int  # Index in word list where to trim
    audio_timestamp: float  # Audio timestamp in seconds for trim point
    context_words: list[str]  # Words to preserve as initial_prompt


@dataclass
class LocalAgreementState:
    """State for the LocalAgreement-2 streaming transcription algorithm."""

    # Audio buffer with automatic size limit (1 chunk per entry)
    audio_buffer: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=30))

    # Previous transcription result for agreement comparison
    prev_result: TranscriptResult | None = None

    # Confirmed output tracking
    confirmed_text: str = ""
    confirmed_word_count: int = 0

    # Context for initial_prompt (text from trimmed audio)
    context_text: str = ""

    # Audio timing
    buffer_start_offset: float = 0.0  # Cumulative trimmed audio time

    def reset(self) -> None:
        """Reset all state for a new recording session."""
        self.audio_buffer.clear()
        self.prev_result = None
        self.confirmed_text = ""
        self.confirmed_word_count = 0
        self.context_text = ""
        self.buffer_start_offset = 0.0


def find_agreed_prefix(prev: TranscriptResult, curr: TranscriptResult) -> list[WordInfo]:
    """Find the longest common prefix between two transcription results.

    LocalAgreement-2: we need both transcriptions to agree on the prefix
    for it to be considered confirmed.

    Args:
        prev: Previous transcription result.
        curr: Current transcription result.

    Returns:
        List of WordInfo from current result that are agreed upon.
    """
    prev_words = prev.word_texts
    curr_words = curr.word_texts

    agreed_count = 0
    min_len = min(len(prev_words), len(curr_words))

    for i in range(min_len):
        # Normalize for comparison (lowercase, strip punctuation handling)
        prev_word = prev_words[i].lower().strip()
        curr_word = curr_words[i].lower().strip()

        if prev_word == curr_word:
            agreed_count = i + 1
        else:
            break

    return curr.words[:agreed_count]


def find_trim_point(
    confirmed_words: list[WordInfo],
    min_confirmed_seconds: float = 5.0,
    keep_overlap_seconds: float = 3.0,
) -> TrimInfo | None:
    """Find trim point based on confirmed audio duration.

    This function determines when to trim the audio buffer based on how much
    audio has been confirmed by LocalAgreement. Unlike sentence-boundary
    trimming, this works even when speech lacks punctuation.

    Strategy:
    1. Check if confirmed audio duration exceeds threshold
    2. Find trim point that keeps overlap seconds of context
    3. Return trim info with timestamp and context words

    Args:
        confirmed_words: List of confirmed words with timestamps.
        min_confirmed_seconds: Minimum confirmed audio duration before trimming.
        keep_overlap_seconds: How much confirmed audio to keep for context.

    Returns:
        TrimInfo if trim point found, None otherwise.
    """
    if not confirmed_words:
        return None

    last_word = confirmed_words[-1]
    confirmed_duration = last_word.end

    # Only trim if we have enough confirmed audio
    if confirmed_duration < min_confirmed_seconds:
        return None

    # Find trim point: keep last N seconds of confirmed audio
    target_trim_time = confirmed_duration - keep_overlap_seconds

    # Find word at or before target trim time
    trim_index = 0
    for i, word in enumerate(confirmed_words):
        if word.end <= target_trim_time:
            trim_index = i
        else:
            break

    if trim_index == 0:
        return None

    trim_word = confirmed_words[trim_index]
    context_words = [w.word.strip() for w in confirmed_words[: trim_index + 1]]

    return TrimInfo(
        should_trim=True,
        word_index=trim_index,
        audio_timestamp=trim_word.end,
        context_words=context_words,
    )


def find_sentence_boundary(
    words: list[WordInfo], min_words_after: int = 3, min_words_before: int = 5
) -> TrimInfo | None:
    """Find the last sentence boundary in confirmed words for buffer trimming.

    NOTE: This function is kept for backwards compatibility but find_trim_point()
    is preferred as it works even without punctuation in speech.

    Strategy:
    1. Look for sentence-ending punctuation (. ! ?)
    2. Ensure there are enough words before and after the boundary
    3. Return trim info with timestamp and context words

    Args:
        words: List of confirmed words with timestamps.
        min_words_after: Minimum words required after boundary to trigger trim.
        min_words_before: Minimum words required before boundary.

    Returns:
        TrimInfo if a valid trim point found, None otherwise.
    """
    if len(words) < min_words_before + min_words_after:
        return None

    # Find last sentence boundary
    last_sentence_end = -1
    for i, word in enumerate(words):
        word_text = word.word.strip()
        if word_text and word_text[-1] in SENTENCE_ENDINGS:
            last_sentence_end = i

    if last_sentence_end < 0:
        return None

    # Must have minimum words before and after
    if last_sentence_end < min_words_before - 1:
        return None
    if last_sentence_end >= len(words) - min_words_after:
        return None

    # Get the trim word and its end timestamp
    trim_word = words[last_sentence_end]

    # Gather context words for initial_prompt (all words up to and including boundary)
    context_words = [w.word.strip() for w in words[: last_sentence_end + 1]]

    return TrimInfo(
        should_trim=True,
        word_index=last_sentence_end,
        audio_timestamp=trim_word.end,
        context_words=context_words,
    )


def trim_audio_buffer(
    state: LocalAgreementState,
    trim_info: TrimInfo,
    sample_rate: int,
    chunk_duration: float,
    context_words_limit: int = 200,
) -> None:
    """Trim audio buffer at the specified timestamp.

    This removes confirmed audio that we no longer need to reprocess,
    keeping only audio from after the trim point.

    Args:
        state: LocalAgreement state to modify.
        trim_info: Information about where to trim.
        sample_rate: Audio sample rate (Hz).
        chunk_duration: Duration of each chunk in seconds.
        context_words_limit: Maximum words to keep as context.
    """
    if not state.audio_buffer:
        return

    # Calculate samples to remove
    samples_to_trim = int(trim_info.audio_timestamp * sample_rate)

    # Concatenate buffer and slice
    full_audio = np.concatenate(list(state.audio_buffer))

    if samples_to_trim >= len(full_audio):
        # Trim point is at or beyond buffer end, clear everything
        state.audio_buffer.clear()
        remaining_audio = np.array([], dtype=np.float32)
    else:
        remaining_audio = full_audio[samples_to_trim:]

    # Clear buffer and re-add remaining audio as chunks
    chunk_samples = int(chunk_duration * sample_rate)
    state.audio_buffer.clear()

    for i in range(0, len(remaining_audio), chunk_samples):
        chunk = remaining_audio[i : i + chunk_samples]
        if len(chunk) > 0:
            state.audio_buffer.append(chunk)

    # Update context for initial_prompt (keep last N words)
    context = trim_info.context_words[-context_words_limit:]
    state.context_text = " ".join(context)

    # Update timing offset
    state.buffer_start_offset += trim_info.audio_timestamp

    # Reset confirmed tracking since we trimmed the confirmed content
    state.confirmed_word_count = 0
    state.confirmed_text = ""
