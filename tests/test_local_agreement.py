"""Tests for local_agreement module."""

import numpy as np

from sttd.local_agreement import (
    LocalAgreementState,
    TranscriptResult,
    TrimInfo,
    WordInfo,
    find_agreed_prefix,
    find_sentence_boundary,
    find_trim_point,
    trim_audio_buffer,
)


class TestWordInfo:
    """Tests for WordInfo."""

    def test_word_info_creation(self):
        """Test creating a WordInfo."""
        word = WordInfo(word="hello", start=0.0, end=0.5, probability=0.95)
        assert word.word == "hello"
        assert word.start == 0.0
        assert word.end == 0.5
        assert word.probability == 0.95

    def test_word_info_tuple_unpacking(self):
        """Test WordInfo can be unpacked like a tuple."""
        word = WordInfo("world", 1.0, 1.5, 0.9)
        w, s, e, p = word
        assert w == "world"
        assert s == 1.0
        assert e == 1.5
        assert p == 0.9


class TestTranscriptResult:
    """Tests for TranscriptResult."""

    def test_word_texts_property(self):
        """Test word_texts extracts just the word strings."""
        words = [
            WordInfo(" Hello", 0.0, 0.3, 0.9),
            WordInfo(" world", 0.4, 0.8, 0.95),
        ]
        result = TranscriptResult(words=words, text="Hello world")
        assert result.word_texts == ["Hello", "world"]

    def test_empty_result(self):
        """Test empty TranscriptResult."""
        result = TranscriptResult(words=[], text="")
        assert result.word_texts == []


class TestLocalAgreementState:
    """Tests for LocalAgreementState."""

    def test_default_state(self):
        """Test default state values."""
        state = LocalAgreementState()
        assert len(state.audio_buffer) == 0
        assert state.prev_result is None
        assert state.confirmed_text == ""
        assert state.confirmed_word_count == 0
        assert state.context_text == ""
        assert state.buffer_start_offset == 0.0

    def test_reset(self):
        """Test state reset."""
        state = LocalAgreementState()
        state.confirmed_text = "test"
        state.confirmed_word_count = 5
        state.context_text = "context"
        state.buffer_start_offset = 10.0
        state.audio_buffer.append(np.array([0.1, 0.2]))

        state.reset()

        assert len(state.audio_buffer) == 0
        assert state.prev_result is None
        assert state.confirmed_text == ""
        assert state.confirmed_word_count == 0
        assert state.context_text == ""
        assert state.buffer_start_offset == 0.0

    def test_audio_buffer_maxlen(self):
        """Test audio buffer has maxlen limit."""
        state = LocalAgreementState()
        assert state.audio_buffer.maxlen == 30


class TestFindAgreedPrefix:
    """Tests for find_agreed_prefix function."""

    def test_full_agreement(self):
        """Test when both transcriptions fully agree."""
        prev = TranscriptResult(
            words=[
                WordInfo(" hello", 0.0, 0.5, 0.9),
                WordInfo(" world", 0.5, 1.0, 0.9),
            ],
            text="hello world",
        )
        curr = TranscriptResult(
            words=[
                WordInfo(" hello", 0.0, 0.5, 0.9),
                WordInfo(" world", 0.5, 1.0, 0.9),
            ],
            text="hello world",
        )

        agreed = find_agreed_prefix(prev, curr)
        assert len(agreed) == 2
        assert agreed[0].word == " hello"
        assert agreed[1].word == " world"

    def test_partial_agreement(self):
        """Test when transcriptions partially agree."""
        prev = TranscriptResult(
            words=[
                WordInfo(" thank", 0.0, 0.3, 0.9),
                WordInfo(" you", 0.3, 0.5, 0.9),
            ],
            text="thank you",
        )
        curr = TranscriptResult(
            words=[
                WordInfo(" thank", 0.0, 0.3, 0.9),
                WordInfo(" your", 0.3, 0.5, 0.9),
            ],
            text="thank your",
        )

        agreed = find_agreed_prefix(prev, curr)
        assert len(agreed) == 1
        assert agreed[0].word == " thank"

    def test_no_agreement(self):
        """Test when transcriptions don't agree at all."""
        prev = TranscriptResult(
            words=[WordInfo(" hello", 0.0, 0.5, 0.9)],
            text="hello",
        )
        curr = TranscriptResult(
            words=[WordInfo(" world", 0.0, 0.5, 0.9)],
            text="world",
        )

        agreed = find_agreed_prefix(prev, curr)
        assert len(agreed) == 0

    def test_case_insensitive(self):
        """Test agreement is case insensitive."""
        prev = TranscriptResult(
            words=[WordInfo(" Hello", 0.0, 0.5, 0.9)],
            text="Hello",
        )
        curr = TranscriptResult(
            words=[WordInfo(" hello", 0.0, 0.5, 0.9)],
            text="hello",
        )

        agreed = find_agreed_prefix(prev, curr)
        assert len(agreed) == 1

    def test_longer_prefix_in_curr(self):
        """Test when current has more words than previous."""
        prev = TranscriptResult(
            words=[WordInfo(" hello", 0.0, 0.5, 0.9)],
            text="hello",
        )
        curr = TranscriptResult(
            words=[
                WordInfo(" hello", 0.0, 0.5, 0.9),
                WordInfo(" world", 0.5, 1.0, 0.9),
            ],
            text="hello world",
        )

        agreed = find_agreed_prefix(prev, curr)
        assert len(agreed) == 1
        assert agreed[0].word == " hello"


class TestFindSentenceBoundary:
    """Tests for find_sentence_boundary function."""

    def test_finds_period(self):
        """Test finding sentence boundary at period."""
        words = [
            WordInfo(" Hello", 0.0, 0.3, 0.9),
            WordInfo(" world.", 0.3, 0.8, 0.9),
            WordInfo(" How", 0.9, 1.1, 0.9),
            WordInfo(" are", 1.2, 1.4, 0.9),
            WordInfo(" you", 1.5, 1.8, 0.9),
        ]

        trim_info = find_sentence_boundary(words, min_words_after=2, min_words_before=1)

        assert trim_info is not None
        assert trim_info.should_trim is True
        assert trim_info.word_index == 1
        assert trim_info.audio_timestamp == 0.8
        assert "Hello" in trim_info.context_words
        assert "world." in trim_info.context_words

    def test_finds_exclamation(self):
        """Test finding sentence boundary at exclamation mark."""
        words = [
            WordInfo(" Wow!", 0.0, 0.3, 0.9),
            WordInfo(" That", 0.4, 0.6, 0.9),
            WordInfo(" is", 0.7, 0.8, 0.9),
            WordInfo(" great", 0.9, 1.2, 0.9),
        ]

        trim_info = find_sentence_boundary(words, min_words_after=2, min_words_before=1)

        assert trim_info is not None
        assert trim_info.word_index == 0
        assert "Wow!" in trim_info.context_words

    def test_finds_question_mark(self):
        """Test finding sentence boundary at question mark."""
        words = [
            WordInfo(" Really?", 0.0, 0.4, 0.9),
            WordInfo(" Yes", 0.5, 0.7, 0.9),
            WordInfo(" it", 0.8, 0.9, 0.9),
            WordInfo(" is", 1.0, 1.1, 0.9),
        ]

        trim_info = find_sentence_boundary(words, min_words_after=2, min_words_before=1)

        assert trim_info is not None
        assert trim_info.word_index == 0

    def test_no_boundary_found(self):
        """Test when no sentence boundary exists."""
        words = [
            WordInfo(" Hello", 0.0, 0.3, 0.9),
            WordInfo(" world", 0.4, 0.7, 0.9),
            WordInfo(" how", 0.8, 1.0, 0.9),
            WordInfo(" are", 1.1, 1.3, 0.9),
        ]

        trim_info = find_sentence_boundary(words)
        assert trim_info is None

    def test_not_enough_words_after(self):
        """Test when not enough words after boundary."""
        words = [
            WordInfo(" Hello.", 0.0, 0.3, 0.9),
            WordInfo(" World", 0.4, 0.7, 0.9),
        ]

        trim_info = find_sentence_boundary(words, min_words_after=3)
        assert trim_info is None

    def test_not_enough_words_total(self):
        """Test when not enough words total."""
        words = [WordInfo(" Hi.", 0.0, 0.3, 0.9)]

        trim_info = find_sentence_boundary(words)
        assert trim_info is None


class TestTrimAudioBuffer:
    """Tests for trim_audio_buffer function."""

    def test_basic_trim(self):
        """Test basic buffer trimming."""
        state = LocalAgreementState()

        # Add some audio chunks (1 second each at 16kHz)
        chunk1 = np.zeros(16000, dtype=np.float32)
        chunk2 = np.zeros(16000, dtype=np.float32)
        chunk3 = np.zeros(16000, dtype=np.float32)
        state.audio_buffer.append(chunk1)
        state.audio_buffer.append(chunk2)
        state.audio_buffer.append(chunk3)

        trim_info = TrimInfo(
            should_trim=True,
            word_index=1,
            audio_timestamp=1.5,  # Trim at 1.5 seconds
            context_words=["hello", "world"],
        )

        trim_audio_buffer(
            state,
            trim_info,
            sample_rate=16000,
            chunk_duration=1.0,
            context_words_limit=200,
        )

        # Should have remaining audio (3 seconds - 1.5 = 1.5 seconds)
        total_samples = sum(len(chunk) for chunk in state.audio_buffer)
        assert total_samples == 24000  # 1.5 seconds at 16kHz

        # Context should be set
        assert state.context_text == "hello world"

        # Offset should be updated
        assert state.buffer_start_offset == 1.5

        # Confirmed tracking should be reset
        assert state.confirmed_word_count == 0
        assert state.confirmed_text == ""

    def test_trim_context_limit(self):
        """Test context words are limited."""
        state = LocalAgreementState()
        state.audio_buffer.append(np.zeros(16000, dtype=np.float32))

        # Context with many words
        context = [f"word{i}" for i in range(300)]
        trim_info = TrimInfo(
            should_trim=True,
            word_index=0,
            audio_timestamp=0.5,
            context_words=context,
        )

        trim_audio_buffer(
            state,
            trim_info,
            sample_rate=16000,
            chunk_duration=1.0,
            context_words_limit=100,
        )

        # Context should be limited to last 100 words
        words_in_context = state.context_text.split()
        assert len(words_in_context) == 100
        assert words_in_context[0] == "word200"  # Last 100 words

    def test_trim_empty_buffer(self):
        """Test trimming empty buffer doesn't crash."""
        state = LocalAgreementState()

        trim_info = TrimInfo(
            should_trim=True,
            word_index=0,
            audio_timestamp=1.0,
            context_words=["test"],
        )

        # Should not raise
        trim_audio_buffer(
            state,
            trim_info,
            sample_rate=16000,
            chunk_duration=1.0,
        )


class TestFindTrimPoint:
    """Tests for find_trim_point function."""

    def test_finds_trim_point_after_threshold(self):
        """Test finding trim point when confirmed audio exceeds threshold."""
        words = [
            WordInfo(" Hello", 0.0, 0.5, 0.9),
            WordInfo(" this", 0.5, 1.0, 0.9),
            WordInfo(" is", 1.0, 1.5, 0.9),
            WordInfo(" a", 1.5, 2.0, 0.9),
            WordInfo(" test", 2.0, 2.5, 0.9),
            WordInfo(" of", 2.5, 3.0, 0.9),
            WordInfo(" trimming", 3.0, 4.0, 0.9),
            WordInfo(" audio", 4.0, 5.0, 0.9),
            WordInfo(" buffers", 5.0, 6.0, 0.9),
        ]

        # 6 seconds of confirmed audio, threshold is 5s, overlap is 3s
        # Should trim at ~3s mark (6 - 3 = 3), keeping words after that
        trim_info = find_trim_point(words, min_confirmed_seconds=5.0, keep_overlap_seconds=3.0)

        assert trim_info is not None
        assert trim_info.should_trim is True
        # Should trim at word ending at or before 3.0s
        assert trim_info.audio_timestamp <= 3.0
        assert len(trim_info.context_words) > 0

    def test_no_trim_below_threshold(self):
        """Test no trim when confirmed audio is below threshold."""
        words = [
            WordInfo(" Hello", 0.0, 0.5, 0.9),
            WordInfo(" world", 0.5, 1.0, 0.9),
            WordInfo(" test", 1.0, 1.5, 0.9),
        ]

        # Only 1.5 seconds of audio, threshold is 5s
        trim_info = find_trim_point(words, min_confirmed_seconds=5.0, keep_overlap_seconds=3.0)

        assert trim_info is None

    def test_no_trim_empty_words(self):
        """Test no trim with empty words list."""
        trim_info = find_trim_point([], min_confirmed_seconds=5.0, keep_overlap_seconds=3.0)
        assert trim_info is None

    def test_trim_preserves_overlap(self):
        """Test that overlap is preserved after trim."""
        # Create words spanning 10 seconds
        words = [WordInfo(f" word{i}", float(i), float(i + 1), 0.9) for i in range(10)]

        # Threshold 5s, overlap 3s -> should trim at ~7s (10-3=7)
        trim_info = find_trim_point(words, min_confirmed_seconds=5.0, keep_overlap_seconds=3.0)

        assert trim_info is not None
        # Remaining audio after trim should be >= 3 seconds
        last_word_end = words[-1].end  # 10.0
        remaining_audio = last_word_end - trim_info.audio_timestamp
        assert remaining_audio >= 3.0

    def test_context_words_collected(self):
        """Test that context words are properly collected."""
        words = [
            WordInfo(" Hello", 0.0, 1.0, 0.9),
            WordInfo(" world", 1.0, 2.0, 0.9),
            WordInfo(" how", 2.0, 3.0, 0.9),
            WordInfo(" are", 3.0, 4.0, 0.9),
            WordInfo(" you", 4.0, 5.0, 0.9),
            WordInfo(" today", 5.0, 6.0, 0.9),
        ]

        # 6 seconds total, threshold 5s, overlap 3s
        trim_info = find_trim_point(words, min_confirmed_seconds=5.0, keep_overlap_seconds=3.0)

        assert trim_info is not None
        # Context should include trimmed words
        assert len(trim_info.context_words) > 0
        assert "Hello" in trim_info.context_words
