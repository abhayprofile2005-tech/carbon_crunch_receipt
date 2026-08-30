"""
ocr_engine.py
-------------
Thin wrapper around Tesseract (via pytesseract) that extracts not just text
but per-word bounding boxes and confidence scores, and groups words back
into lines. This line/word-level confidence is what field extraction and
confidence scoring build on top of.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pytesseract
from pytesseract import Output


@dataclass
class Word:
    text: str
    conf: float          # 0-100 from tesseract, -1 means "no confidence available"
    left: int
    top: int
    width: int
    height: int
    line_num: int
    block_num: int


@dataclass
class Line:
    text: str
    words: list[Word] = field(default_factory=list)
    avg_conf: float = 0.0
    top: int = 0


@dataclass
class OCRResult:
    full_text: str
    lines: list[Line]
    words: list[Word]
    avg_confidence: float  # 0-100, average of all word confidences
    success: bool = True
    error: Optional[str] = None


# Tesseract page segmentation mode 4 = "assume a single column of text",
# which fits the narrow, single-column layout of most receipts better than
# the default (fully automatic) mode.
TESS_CONFIG = "--oem 3 --psm 4"


def run_ocr(image: Optional[np.ndarray]) -> OCRResult:
    """Run Tesseract on a preprocessed image, returning structured word/line
    data with confidence scores. Returns an empty-but-valid OCRResult (not
    an exception) if OCR fails or finds no text, so downstream code can
    treat "no text found" as a normal edge case."""
    if image is None:
        return OCRResult(full_text="", lines=[], words=[], avg_confidence=0.0,
                          success=False, error="No image to OCR (preprocessing failed).")

    try:
        data = pytesseract.image_to_data(image, config=TESS_CONFIG, output_type=Output.DICT)
    except Exception as e:
        return OCRResult(full_text="", lines=[], words=[], avg_confidence=0.0,
                          success=False, error=f"Tesseract failed: {e}")

    words: list[Word] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = data["text"][i].strip()
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        if not text or conf < 0:
            continue
        words.append(Word(
            text=text,
            conf=conf,
            left=data["left"][i],
            top=data["top"][i],
            width=data["width"][i],
            height=data["height"][i],
            line_num=data["line_num"][i],
            block_num=data["block_num"][i],
        ))

    if not words:
        return OCRResult(full_text="", lines=[], words=[], avg_confidence=0.0,
                          success=True, error="No text detected in image.")

    # Group words into lines using (block_num, line_num), preserving reading order.
    lines_map: dict[tuple[int, int], list[Word]] = {}
    for w in words:
        key = (w.block_num, w.line_num)
        lines_map.setdefault(key, []).append(w)

    lines: list[Line] = []
    for key in sorted(lines_map.keys()):
        line_words = sorted(lines_map[key], key=lambda w: w.left)
        text = " ".join(w.text for w in line_words)
        avg_conf = sum(w.conf for w in line_words) / len(line_words)
        top = min(w.top for w in line_words)
        lines.append(Line(text=text, words=line_words, avg_conf=avg_conf, top=top))

    lines.sort(key=lambda ln: ln.top)
    full_text = "\n".join(ln.text for ln in lines)
    avg_confidence = sum(w.conf for w in words) / len(words)

    return OCRResult(
        full_text=full_text, lines=lines, words=words,
        avg_confidence=avg_confidence, success=True,
    )
