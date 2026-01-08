"""Speaker identification using SpeechBrain ECAPA-TDNN embeddings."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Compatibility shim: newer torchaudio removed list_audio_backends() but SpeechBrain still calls it
import torchaudio
from scipy.spatial.distance import cosine

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from voiced.config import DiarizationConfig
from voiced.profiles import ProfileManager, VoiceProfile

logger = logging.getLogger(__name__)

ENROLLMENT_PROMPT = (
    "The rainbow is a division of white light into many beautiful colors. "
    "These take the shape of a long round arch with its path high above "
    "and its two ends apparently beyond the horizon."
)


@dataclass
class IdentifiedSegment:
    """A transcription segment with speaker identification."""

    start: float
    end: float
    text: str
    speaker: str
    confidence: float


# Backward compatibility alias
DiarizedSegment = IdentifiedSegment


class SpeakerEmbedder:
    """Extract speaker embeddings using SpeechBrain ECAPA-TDNN."""

    def __init__(
        self,
        model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        device: str = "auto",
    ):
        self._classifier = None
        self.model_source = model_source
        self.device = self._resolve_device(device)

    def _resolve_device(self, device: str) -> str:
        """Determine device to use."""
        if device != "auto":
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def classifier(self):
        """Lazy-load the SpeechBrain encoder."""
        if self._classifier is None:
            from speechbrain.inference.speaker import EncoderClassifier

            logger.info(f"Loading SpeechBrain embedding model on {self.device}")
            run_opts = {"device": self.device}
            self._classifier = EncoderClassifier.from_hparams(
                source=self.model_source,
                run_opts=run_opts,
            )
        return self._classifier

    def extract_embedding(self, audio_path: str | Path) -> np.ndarray:
        """Extract speaker embedding from an audio file."""
        import soundfile as sf
        from scipy.signal import resample

        audio, sr = sf.read(str(audio_path))

        # Convert stereo to mono if needed
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample to 16kHz if needed
        if sr != 16000:
            num_samples = int(len(audio) * 16000 / sr)
            audio = resample(audio, num_samples)

        # Convert to torch tensor with shape (1, samples)
        signal = torch.from_numpy(audio).float().unsqueeze(0)

        embedding = self.classifier.encode_batch(signal)
        return embedding.squeeze().cpu().numpy()

    def extract_embedding_from_array(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """Extract embedding from numpy audio array."""
        from scipy.signal import resample

        # Ensure 1D array
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sample_rate != 16000:
            num_samples = int(len(audio) * 16000 / sample_rate)
            audio = resample(audio, num_samples)

        # Convert to torch tensor with shape (1, samples)
        signal = torch.from_numpy(audio).float().unsqueeze(0)

        embedding = self.classifier.encode_batch(signal)
        return embedding.squeeze().cpu().numpy()

    def unload(self) -> None:
        """Unload model to free memory."""
        if self._classifier is not None:
            del self._classifier
            self._classifier = None
        logger.info("SpeakerEmbedder model unloaded")


class SpeakerIdentifier:
    """Per-segment speaker identification using SpeechBrain embeddings."""

    def __init__(
        self,
        config: DiarizationConfig | None = None,
        device: str = "auto",
    ):
        self.config = config or DiarizationConfig()
        self.device = self._resolve_device(device if device != "auto" else self.config.device)
        self._embedder = None

    def _resolve_device(self, device: str) -> str:
        """Determine device to use."""
        if device != "auto":
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def embedder(self) -> SpeakerEmbedder:
        """Get or create speaker embedder."""
        if self._embedder is None:
            self._embedder = SpeakerEmbedder(
                model_source=self.config.model,
                device=self.device,
            )
        return self._embedder

    def identify_segments(
        self,
        audio_path: str | Path,
        transcription_segments: list[tuple[float, float, str]],
        profiles: list[VoiceProfile] | None = None,
        threshold: float | None = None,
    ) -> list[IdentifiedSegment]:
        """Identify speaker for each transcription segment.

        Args:
            audio_path: Path to the audio file.
            transcription_segments: (start, end, text) from whisper.
            profiles: Voice profiles to match against.
            threshold: Similarity threshold (uses config default if None).

        Returns:
            List of IdentifiedSegment with speaker labels.
        """
        import soundfile as sf

        if profiles is None:
            pm = ProfileManager()
            profiles = pm.load_all()

        threshold = threshold or self.config.similarity_threshold
        min_duration = self.config.min_segment_duration

        # Load full audio
        audio, sr = sf.read(str(audio_path))
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Prepare profile embeddings
        profile_embeddings = [(p.name, p.embedding_array()) for p in profiles] if profiles else []

        results = []

        for start, end, text in transcription_segments:
            duration = end - start

            # Extract segment audio
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment_audio = audio[start_sample:end_sample]

            speaker = "Unknown"
            confidence = 0.0

            if duration >= min_duration and len(segment_audio) > 0:
                try:
                    embedding = self.embedder.extract_embedding_from_array(segment_audio, sr)

                    # Match against profiles
                    if profile_embeddings:
                        best_match, best_score = self._match_embedding(
                            embedding, profile_embeddings, threshold
                        )
                        if best_match:
                            speaker = best_match
                            confidence = best_score
                            logger.debug(
                                f"Segment [{start:.2f}-{end:.2f}] matched {speaker} "
                                f"(score: {best_score:.3f})"
                            )
                except Exception as e:
                    logger.warning(f"Failed to extract embedding for segment [{start}-{end}]: {e}")

            results.append(
                IdentifiedSegment(
                    start=start,
                    end=end,
                    text=text,
                    speaker=speaker,
                    confidence=confidence,
                )
            )

        return results

    def identify_segments_from_array(
        self,
        audio: np.ndarray,
        sample_rate: int,
        transcription_segments: list[tuple[float, float, str]],
        profiles: list[VoiceProfile] | None = None,
        threshold: float | None = None,
    ) -> list[IdentifiedSegment]:
        """Identify speaker for each transcription segment from audio array.

        Args:
            audio: Audio data as numpy array (mono).
            sample_rate: Sample rate of the audio.
            transcription_segments: (start, end, text) from whisper.
            profiles: Voice profiles to match against.
            threshold: Similarity threshold (uses config default if None).

        Returns:
            List of IdentifiedSegment with speaker labels.
        """
        if profiles is None:
            pm = ProfileManager()
            profiles = pm.load_all()

        threshold = threshold or self.config.similarity_threshold
        min_duration = self.config.min_segment_duration

        # Ensure 1D array
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Prepare profile embeddings
        profile_embeddings = [(p.name, p.embedding_array()) for p in profiles] if profiles else []

        results = []

        for start, end, text in transcription_segments:
            duration = end - start

            # Extract segment audio
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            segment_audio = audio[start_sample:end_sample]

            speaker = "Unknown"
            confidence = 0.0

            if duration >= min_duration and len(segment_audio) > 0:
                try:
                    embedding = self.embedder.extract_embedding_from_array(
                        segment_audio, sample_rate
                    )

                    # Match against profiles
                    if profile_embeddings:
                        best_match, best_score = self._match_embedding(
                            embedding, profile_embeddings, threshold
                        )
                        if best_match:
                            speaker = best_match
                            confidence = best_score
                            logger.debug(
                                f"Segment [{start:.2f}-{end:.2f}] matched {speaker} "
                                f"(score: {best_score:.3f})"
                            )
                except Exception as e:
                    logger.warning(f"Failed to extract embedding for segment [{start}-{end}]: {e}")

            results.append(
                IdentifiedSegment(
                    start=start,
                    end=end,
                    text=text,
                    speaker=speaker,
                    confidence=confidence,
                )
            )

        return results

    def _match_embedding(
        self,
        embedding: np.ndarray,
        profile_embeddings: list[tuple[str, np.ndarray]],
        threshold: float,
    ) -> tuple[str | None, float]:
        """Match an embedding against profiles."""
        best_match = None
        best_score = -1.0

        for name, profile_emb in profile_embeddings:
            similarity = 1 - cosine(embedding, profile_emb)
            if similarity > best_score:
                best_score = similarity
                best_match = name

        if best_match and best_score >= threshold:
            return best_match, best_score
        return None, best_score

    def unload(self) -> None:
        """Unload models to free memory."""
        if self._embedder is not None:
            self._embedder.unload()
            self._embedder = None
        logger.info("SpeakerIdentifier models unloaded")


class SpeakerDiarizer:
    """Full speaker diarization using SpeechBrain embeddings and spectral clustering."""

    def __init__(
        self,
        config: DiarizationConfig | None = None,
        device: str = "auto",
        window_size: float = 1.5,
        step_size: float = 0.75,
    ):
        """Initialize diarizer.

        Args:
            config: Diarization configuration.
            device: Device to use (auto, cuda, cpu).
            window_size: Sliding window size in seconds for embedding extraction.
            step_size: Step size in seconds between windows.
        """
        self.config = config or DiarizationConfig()
        self.device = self._resolve_device(device if device != "auto" else self.config.device)
        self._embedder = None
        self.window_size = window_size
        self.step_size = step_size

    def _resolve_device(self, device: str) -> str:
        """Determine device to use."""
        if device != "auto":
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def embedder(self) -> SpeakerEmbedder:
        """Get embedder for embedding extraction."""
        if self._embedder is None:
            self._embedder = SpeakerEmbedder(
                model_source=self.config.model,
                device=self.device,
            )
        return self._embedder

    def _extract_windowed_embeddings(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> tuple[np.ndarray, list[tuple[float, float]]]:
        """Extract embeddings using sliding windows.

        Returns:
            Tuple of (embeddings array, list of (start, end) times for each window).
        """
        window_samples = int(self.window_size * sample_rate)
        step_samples = int(self.step_size * sample_rate)

        embeddings = []
        timestamps = []

        pos = 0
        while pos + window_samples <= len(audio):
            window = audio[pos : pos + window_samples]
            start_time = pos / sample_rate
            end_time = (pos + window_samples) / sample_rate

            try:
                emb = self.embedder.extract_embedding_from_array(window, sample_rate)
                embeddings.append(emb)
                timestamps.append((start_time, end_time))
            except Exception as e:
                logger.warning(f"Failed to extract embedding at {start_time:.2f}s: {e}")

            pos += step_samples

        if pos < len(audio) and len(audio) - pos >= sample_rate * 0.5:
            window = audio[pos:]
            start_time = pos / sample_rate
            end_time = len(audio) / sample_rate
            try:
                emb = self.embedder.extract_embedding_from_array(window, sample_rate)
                embeddings.append(emb)
                timestamps.append((start_time, end_time))
            except Exception:
                pass

        return np.array(embeddings), timestamps

    def _estimate_num_speakers(
        self,
        embeddings: np.ndarray,
        max_speakers: int = 10,
    ) -> int:
        """Estimate number of speakers using eigenvalue analysis.

        Uses the eigengap heuristic on the affinity matrix.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        affinity = cosine_similarity(embeddings)
        affinity = (affinity + 1) / 2  # ensure non-negative

        degree = np.sum(affinity, axis=1)
        degree_inv_sqrt = np.diag(1.0 / np.sqrt(degree + 1e-10))
        laplacian = np.eye(len(affinity)) - degree_inv_sqrt @ affinity @ degree_inv_sqrt

        eigenvalues = np.linalg.eigvalsh(laplacian)
        eigenvalues = np.sort(eigenvalues)

        gaps = np.diff(eigenvalues[:max_speakers])
        if len(gaps) > 0:
            n_clusters = int(np.argmax(gaps) + 1)
            n_clusters = max(1, min(n_clusters, max_speakers))
        else:
            n_clusters = 1

        return n_clusters

    def diarize_file(
        self,
        audio_path: str | Path,
        num_speakers: int | None = None,
    ) -> list[IdentifiedSegment]:
        """Run diarization on an audio file.

        Args:
            audio_path: Path to audio file.
            num_speakers: Number of speakers (None = auto-detect).

        Returns:
            List of IdentifiedSegment with speaker labels (SPEAKER_00, etc.).
        """
        import soundfile as sf
        from sklearn.cluster import SpectralClustering

        audio_path = Path(audio_path)
        n_speakers = num_speakers if num_speakers is not None else self.config.num_speakers

        logger.info(f"Diarizing {audio_path} (num_speakers={n_speakers})")

        audio, sr = sf.read(str(audio_path))
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        embeddings, timestamps = self._extract_windowed_embeddings(audio, sr)

        if len(embeddings) == 0:
            logger.warning("No embeddings extracted, audio may be too short")
            return []

        logger.info(f"Extracted {len(embeddings)} embeddings from sliding windows")

        if n_speakers is None:
            n_speakers = self._estimate_num_speakers(embeddings)
            logger.info(f"Auto-detected {n_speakers} speaker(s)")

        if len(embeddings) < n_speakers:
            n_speakers = len(embeddings)
            logger.warning(f"Reduced to {n_speakers} speakers due to limited embeddings")

        if n_speakers == 1:
            labels = np.zeros(len(embeddings), dtype=int)
        else:
            cluster = SpectralClustering(
                n_clusters=n_speakers,
                affinity="nearest_neighbors",
                n_neighbors=min(10, len(embeddings) - 1),
                random_state=42,
            )
            labels = cluster.fit_predict(embeddings)

        results = []
        if len(labels) > 0:
            current_speaker = int(labels[0])
            current_start = timestamps[0][0]
            current_end = timestamps[0][1]

            for i in range(1, len(labels)):
                speaker = int(labels[i])
                start, end = timestamps[i]

                if speaker == current_speaker:
                    current_end = end
                else:
                    results.append(
                        IdentifiedSegment(
                            start=current_start,
                            end=current_end,
                            text="",
                            speaker=f"SPEAKER_{current_speaker:02d}",
                            confidence=0.0,
                        )
                    )
                    current_speaker = speaker
                    current_start = start
                    current_end = end

            results.append(
                IdentifiedSegment(
                    start=current_start,
                    end=current_end,
                    text="",
                    speaker=f"SPEAKER_{current_speaker:02d}",
                    confidence=0.0,
                )
            )

        unique_speakers = len(set(s.speaker for s in results))
        logger.info(f"Found {unique_speakers} speaker(s) in {len(results)} segments")
        return results

    def diarize_and_match_profiles(
        self,
        audio_path: str | Path,
        profiles: list[VoiceProfile] | None = None,
        num_speakers: int | None = None,
    ) -> list[IdentifiedSegment]:
        """Run diarization and match speakers to registered profiles.

        Args:
            audio_path: Path to audio file.
            profiles: Voice profiles to match against.
            num_speakers: Number of speakers (None = auto-detect).

        Returns:
            List of IdentifiedSegment with profile names where matched.
        """
        import soundfile as sf

        diar_segments = self.diarize_file(audio_path, num_speakers)

        if not profiles:
            return diar_segments

        audio, sr = sf.read(str(audio_path))
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        speaker_segments: dict[str, list[IdentifiedSegment]] = {}
        for seg in diar_segments:
            speaker_segments.setdefault(seg.speaker, []).append(seg)

        profile_embeddings = [(p.name, p.embedding_array()) for p in profiles]
        speaker_mapping: dict[str, tuple[str, float]] = {}

        for speaker_label, segs in speaker_segments.items():
            segs_sorted = sorted(segs, key=lambda s: s.end - s.start, reverse=True)
            best_seg = segs_sorted[0]
            if (best_seg.end - best_seg.start) < self.config.min_segment_duration:
                continue

            start_sample = int(best_seg.start * sr)
            end_sample = int(best_seg.end * sr)
            segment_audio = audio[start_sample:end_sample]

            try:
                embedding = self.embedder.extract_embedding_from_array(segment_audio, sr)
                best_match = None
                best_score = -1.0

                for name, profile_emb in profile_embeddings:
                    similarity = 1 - cosine(embedding, profile_emb)
                    if similarity > best_score:
                        best_score = similarity
                        best_match = name

                if best_match and best_score >= self.config.similarity_threshold:
                    speaker_mapping[speaker_label] = (best_match, best_score)
                    logger.debug(
                        f"Matched {speaker_label} -> {best_match} (score={best_score:.3f})"
                    )
            except Exception as e:
                logger.warning(f"Failed to extract embedding for {speaker_label}: {e}")

        results = []
        for seg in diar_segments:
            if seg.speaker in speaker_mapping:
                name, confidence = speaker_mapping[seg.speaker]
                results.append(
                    IdentifiedSegment(
                        start=seg.start,
                        end=seg.end,
                        text=seg.text,
                        speaker=name,
                        confidence=confidence,
                    )
                )
            else:
                results.append(seg)

        return results

    def unload(self) -> None:
        """Unload models to free memory."""
        if self._embedder is not None:
            self._embedder.unload()
            self._embedder = None
        logger.info("SpeakerDiarizer models unloaded")


def align_transcription_with_diarization(
    transcription_segments: list[tuple[float, float, str]],
    diarization_segments: list[IdentifiedSegment],
) -> list[IdentifiedSegment]:
    """Align Whisper transcription with diarization speaker labels.

    For each transcription segment, assigns the speaker with maximum
    temporal overlap from diarization.

    Args:
        transcription_segments: (start, end, text) from Whisper.
        diarization_segments: IdentifiedSegment from diarization.

    Returns:
        List of IdentifiedSegment with speaker labels and text.
    """
    results = []

    for trans_start, trans_end, text in transcription_segments:
        overlaps: dict[str, float] = {}
        confidences: dict[str, float] = {}

        for diar_seg in diarization_segments:
            overlap_start = max(trans_start, diar_seg.start)
            overlap_end = min(trans_end, diar_seg.end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > 0:
                overlaps[diar_seg.speaker] = overlaps.get(diar_seg.speaker, 0) + overlap
                if (
                    diar_seg.speaker not in confidences
                    or diar_seg.confidence > confidences[diar_seg.speaker]
                ):
                    confidences[diar_seg.speaker] = diar_seg.confidence

        if overlaps:
            speaker = max(overlaps.keys(), key=lambda k: overlaps[k])
            confidence = confidences.get(speaker, 0.0)
        else:
            speaker = "Unknown"
            confidence = 0.0

        results.append(
            IdentifiedSegment(
                start=trans_start,
                end=trans_end,
                text=text,
                speaker=speaker,
                confidence=confidence,
            )
        )

    return results
