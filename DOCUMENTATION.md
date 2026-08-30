# Documentation

## 1. Approach

The pipeline follows the five stages the assignment lays out, implemented as
independent, testable modules wired together in `src/pipeline.py`:

1. **Preprocessing** (`src/preprocessing.py`) — Each image is denoised
   (`fastNlMeansDenoising`), contrast-corrected with CLAHE (adaptive
   histogram equalization, which copes with harsh shadows/uneven lighting
   better than a single global adjustment), deskewed (via `minAreaRect` on
   thresholded foreground pixels), upscaled if small, and finally binarized
   with adaptive thresholding before being handed to OCR.
2. **OCR** (`src/ocr_engine.py`) — Tesseract (`--psm 4`, single-column mode,
   which fits typical receipt layouts) is run through `pytesseract`, pulling
   per-word bounding boxes and confidence scores, which are then grouped
   back into lines in reading order.
3. **Field extraction** (`src/field_extraction.py`) — Rule-based heuristics
   per field:
   - *Store name*: the tallest (largest-font) line among the first six
     non-empty lines, excluding anything that looks like a date or is
     mostly digits.
   - *Date*: regex over common date formats (numeric, `DD Mon YYYY`, `Mon DD
     YYYY`), preferring lines with a "Date"/"Dt." keyword when multiple
     date-shaped tokens exist.
   - *Total*: searched bottom-up for a line containing a total-type keyword
     (excluding "subtotal") with a trailing currency amount; falls back to
     subtotal, then to the largest currency-shaped number on the receipt.
   - *Items*: any line ending in a price-shaped token that doesn't contain
     an excluded keyword (total/tax/cash/change/etc.) is treated as an item
     row, with the text before the price as the item name.
4. **Data structuring** — each field is emitted in the exact
   `{"value": ..., "confidence": ...}` shape the spec requests, plus a
   `low_confidence` flag and human-readable `processing_notes`.
5. **Financial summary** (`src/financial_summary.py`) — aggregates total
   spend, transaction count, and spend-per-store across all processed
   receipts, skipping/flagging receipts with no parseable total rather than
   silently dropping or crashing on them.

## 2. Confidence scoring

Per field, three signals are combined into one 0–1 score:

```
confidence = 0.5 * (OCR confidence / 100)   # Tesseract's own per-word confidence
           + 0.3 * pattern_match            # did the value match the expected format/shape?
           + 0.2 * keyword_match            # was a confirming keyword nearby (e.g. "Total")?
```

Fields below **0.7** are flagged `low_confidence: true`. When more than one
candidate exists for a field (e.g. two lines both look like a total), the
highest-scoring one is kept and the alternatives are recorded in
`processing_notes` rather than silently discarded, so conflicts stay
auditable. The 0.5/0.3/0.2 split is a reasoned default, not a value fit to
labeled data — the assignment provided no ground-truth confidence labels to
calibrate against (see Challenges).

## 3. Tools used

- **OpenCV** — preprocessing (denoise, CLAHE, deskew, adaptive threshold)
- **Tesseract OCR** via **pytesseract** — text detection & recognition,
  chosen over EasyOCR because it ships with native per-word confidence
  scores out of the box, which the confidence-scoring requirement depends
  on directly
- **NumPy** — array/image manipulation
- **Python stdlib** (`re`, `dataclasses`, `argparse`, `json`) — extraction
  logic, typed intermediate data, CLI, and output

## 4. Challenges faced

- **No access to the actual dataset.** The assignment's Google Drive folder
  wasn't reachable from this build environment (no network egress). To
  still validate the pipeline end-to-end, `gen_test_receipts.py` generates
  synthetic receipts with injected Gaussian noise, lighting gradients,
  Gaussian blur, and random rotation (±6°) to approximate the "noise, blur,
  skew, lighting issues" the assignment describes. The pipeline correctly
  recovered store name, date, all line items, and totals (matching the
  known ground truth exactly) across all four synthetic test receipts,
  including one deliberately sparse single-item receipt. **This should be
  re-validated against the real dataset**, since real photographs will have
  failure modes (crumpled paper, faded thermal print, handwriting) that a
  synthetic generator won't reproduce.
- **Ambiguous date formats.** `DD/MM/YYYY` vs `MM/DD/YYYY` is inherently
  ambiguous from a receipt alone without a locale assumption; the current
  regex captures the token but does not attempt to disambiguate day vs.
  month.
- **Item vs. non-item line disambiguation.** Distinguishing a real item row
  from a tax/subtotal/payment line by keyword blocklist is workable but
  brittle — receipts phrase these differently across regions and languages
  (e.g. "CGST/SGST" vs "VAT" vs "Sales Tax").
- **No labeled data to calibrate confidence weights.** The 0.5/0.3/0.2 OCR/
  pattern/keyword weighting is a reasoned default rather than one tuned
  against ground truth, since none was provided.

## 5. Improvements (given more time/data)

- Replace the rule-based store-name/item heuristics with a small layout-aware
  model (e.g. LayoutLM-style token classification) trained on labeled
  receipts, which would generalize far better than positional/font-size
  heuristics.
- Calibrate confidence-score weights against a labeled validation set
  (compare predicted confidence to actual field-level accuracy).
- Add locale detection (currency symbol, address format) to disambiguate
  date order and currency parsing per-region.
- Fine-tune Tesseract or swap in a receipt-specific OCR model for thermal
  printer fonts, which are a common real-world failure mode not covered by
  the synthetic test set.
- Expand the edge-case test suite with real corrupted/partial/rotated
  photos rather than only synthetic approximations.
