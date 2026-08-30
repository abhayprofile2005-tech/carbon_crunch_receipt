"""
confidence_scoring.py
----------------------
Combines three signals into one 0-1 field-level confidence score, per the
assignment spec:
  1. OCR-level confidence (Tesseract's own per-word confidence)
  2. Pattern validation (does the value match the expected format?)
  3. Heuristics (was a relevant keyword found nearby, e.g. "Total")

Weights are deliberately simple and documented rather than tuned on data we
don't have access to (no ground-truth labels were provided) -- see the
project README for discussion of this limitation.
"""

from dataclasses import dataclass
from typing import Optional

W_OCR = 0.5
W_PATTERN = 0.3
W_HEURISTIC = 0.2

LOW_CONFIDENCE_THRESHOLD = 0.7


@dataclass
class ScoredField:
    value: Optional[str]
    confidence: float
    low_confidence: bool
    notes: list[str]


def score_field(
    value: Optional[str],
    ocr_conf: float,
    pattern_matched: bool,
    keyword_matched: bool,
    found: bool,
) -> ScoredField:
    """ocr_conf is expected in Tesseract's native 0-100 range."""
    if not found or value is None:
        return ScoredField(
            value=None, confidence=0.0, low_confidence=True,
            notes=["Field not found in OCR output."],
        )

    ocr_norm = max(0.0, min(ocr_conf, 100.0)) / 100.0
    pattern_score = 1.0 if pattern_matched else 0.0
    heuristic_score = 1.0 if keyword_matched else 0.5  # neutral if no keyword signal either way

    confidence = (W_OCR * ocr_norm) + (W_PATTERN * pattern_score) + (W_HEURISTIC * heuristic_score)
    confidence = round(min(max(confidence, 0.0), 1.0), 3)

    notes = []
    if ocr_norm < 0.5:
        notes.append("Low raw OCR confidence for source text.")
    if not pattern_matched:
        notes.append("Value did not match expected format.")
    if not keyword_matched:
        notes.append("No confirming keyword found nearby.")

    return ScoredField(
        value=value,
        confidence=confidence,
        low_confidence=confidence < LOW_CONFIDENCE_THRESHOLD,
        notes=notes,
    )


def resolve_conflicts(candidates: list[ScoredField]) -> ScoredField:
    """When multiple candidate values exist for the same field (e.g. two
    lines both looked like a total), keep the highest-confidence one and
    note the conflict for transparency rather than silently dropping data."""
    if not candidates:
        return ScoredField(value=None, confidence=0.0, low_confidence=True, notes=["No candidates."])
    best = max(candidates, key=lambda c: c.confidence)
    if len(candidates) > 1:
        others = [c.value for c in candidates if c is not best]
        best.notes.append(f"Resolved conflict: {len(candidates)} candidates found, others were {others}.")
    return best
