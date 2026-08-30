"""
pipeline.py
-----------
Orchestrates the full per-receipt flow:

    image -> preprocess -> OCR -> field extraction -> confidence scoring -> JSON

Output schema per receipt (matches the assignment's required format, with
confidence attached to each field):

{
  "source_file": "receipt_001.jpg",
  "store_name":   {"value": "...", "confidence": 0.93, "low_confidence": false},
  "date":         {"value": "...", "confidence": 0.88, "low_confidence": false},
  "items": [
      {"name": "...", "price": "...", "confidence": 0.81, "low_confidence": false}
  ],
  "total_amount": {"value": "...", "confidence": 0.96, "low_confidence": false},
  "ocr_avg_confidence": 87.4,
  "processing_notes": ["..."],
  "status": "ok" | "partial" | "failed"
}
"""

from dataclasses import asdict
from typing import Optional

from . import preprocessing, ocr_engine, field_extraction, confidence_scoring


def process_receipt(image_path: str) -> dict:
    notes: list[str] = []

    pre = preprocessing.preprocess(image_path)
    if not pre.success:
        return {
            "source_file": image_path,
            "store_name": {"value": None, "confidence": 0.0, "low_confidence": True},
            "date": {"value": None, "confidence": 0.0, "low_confidence": True},
            "items": [],
            "total_amount": {"value": None, "confidence": 0.0, "low_confidence": True},
            "ocr_avg_confidence": 0.0,
            "processing_notes": [pre.error or "Preprocessing failed."],
            "status": "failed",
        }

    if pre.skew_angle:
        notes.append(f"Corrected skew of {pre.skew_angle:.1f} degrees.")
    if pre.estimated_noise > 25:
        notes.append(f"High source-image noise detected (score={pre.estimated_noise:.1f}); "
                      f"results may be less reliable.")

    ocr_result = ocr_engine.run_ocr(pre.image)
    if not ocr_result.success or not ocr_result.lines:
        notes.append(ocr_result.error or "No text detected.")
        return {
            "source_file": image_path,
            "store_name": {"value": None, "confidence": 0.0, "low_confidence": True},
            "date": {"value": None, "confidence": 0.0, "low_confidence": True},
            "items": [],
            "total_amount": {"value": None, "confidence": 0.0, "low_confidence": True},
            "ocr_avg_confidence": 0.0,
            "processing_notes": notes,
            "status": "failed",
        }

    lines = ocr_result.lines

    store_c = field_extraction.extract_store_name(lines)
    date_c = field_extraction.extract_date(lines)
    total_c = field_extraction.extract_total(lines)
    item_candidates = field_extraction.extract_items(lines)

    store_scored = confidence_scoring.score_field(
        store_c.value, store_c.ocr_conf, store_c.pattern_matched, store_c.keyword_matched, store_c.found)
    date_scored = confidence_scoring.score_field(
        date_c.value, date_c.ocr_conf, date_c.pattern_matched, date_c.keyword_matched, date_c.found)
    total_scored = confidence_scoring.score_field(
        total_c.value, total_c.ocr_conf, total_c.pattern_matched, total_c.keyword_matched, total_c.found)

    items_out = []
    for item in item_candidates:
        scored = confidence_scoring.score_field(
            item.price, item.ocr_conf, item.pattern_matched, keyword_matched=False, found=True)
        items_out.append({
            "name": item.name,
            "price": item.price,
            "confidence": scored.confidence,
            "low_confidence": scored.low_confidence,
        })

    if not item_candidates:
        notes.append("No line items could be confidently separated from the receipt body.")
    if store_scored.low_confidence:
        notes.append("Store name is low-confidence: " + "; ".join(store_scored.notes))
    if date_scored.low_confidence:
        notes.append("Date is low-confidence: " + "; ".join(date_scored.notes))
    if total_scored.low_confidence:
        notes.append("Total amount is low-confidence: " + "; ".join(total_scored.notes))

    core_found = [store_c.found, date_c.found, total_c.found]
    if all(core_found) and not any([store_scored.low_confidence, date_scored.low_confidence, total_scored.low_confidence]):
        status = "ok"
    elif any(core_found):
        status = "partial"
    else:
        status = "failed"

    return {
        "source_file": image_path,
        "store_name": {"value": store_scored.value, "confidence": store_scored.confidence,
                        "low_confidence": store_scored.low_confidence},
        "date": {"value": date_scored.value, "confidence": date_scored.confidence,
                  "low_confidence": date_scored.low_confidence},
        "items": items_out,
        "total_amount": {"value": total_scored.value, "confidence": total_scored.confidence,
                          "low_confidence": total_scored.low_confidence},
        "ocr_avg_confidence": round(ocr_result.avg_confidence, 2),
        "processing_notes": notes,
        "status": status,
    }
