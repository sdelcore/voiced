"""Speaker-attributed segments and transcription/diarization alignment.

Torch-free on purpose: the daemon parent process exchanges these objects
with the inference worker (diarization runs worker-side), so this module
must be importable without pulling in the torch/speechbrain stack that
``voiced.diarizer`` requires.
"""

from dataclasses import dataclass


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
