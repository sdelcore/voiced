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

from sttd.config import DiarizationConfig
from sttd.profiles import ProfileManager, VoiceProfile

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
