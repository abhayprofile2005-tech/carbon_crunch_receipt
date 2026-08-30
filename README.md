# Carbon Crunch — Receipt OCR & Financial Summary Pipeline

An OCR pipeline that extracts structured, confidence-scored data (store
name, date, line items, total amount) from real-world receipt photos and
rolls it up into an expense summary. Built for the Carbon Crunch ML Ops
shortlisting assignment.

## Project Structure

```
carbon_crunch_ocr/
├── main.py                    # CLI entry point
├── requirements.txt
├── src/
│   ├── preprocessing.py       # denoise, deskew, contrast/lighting correction
│   ├── ocr_engine.py          # Tesseract wrapper -> words/lines + confidence
│   ├── field_extraction.py    # store name / date / items / total heuristics
│   ├── confidence_scoring.py  # combines OCR conf + pattern + heuristic signals
│   ├── financial_summary.py   # aggregate spend across receipts
│   └── pipeline.py            # orchestrates one receipt end-to-end
├── sample_data/                # example receipt images (synthetic, see docs)
├── outputs/
│   ├── json/                  # one JSON file per processed receipt
│   └── summary.json           # aggregate expense summary
└── docs/
    └── DOCUMENTATION.md       # approach, tools, challenges, improvements
```

## Setup

```bash
pip install -r requirements.txt
# Tesseract OCR must also be installed as a system package (see requirements.txt)
```

## Usage

```bash
python main.py --input <folder_of_receipt_images> --output outputs
```

- Drop the Carbon Crunch dataset images into a folder (e.g. `data/`) and point
  `--input` at it: `python main.py --input data --output outputs`.
- Each receipt produces `outputs/json/<filename>.json` with per-field values
  and confidence scores.
- `outputs/summary.json` contains the aggregate expense summary (total
  spend, transaction count, spend per store, and any receipts flagged for
  manual review).

## Output schema (per receipt)

```json
{
  "source_file": "receipt_01.png",
  "store_name":   {"value": "GREEN LEAF GROCERY", "confidence": 0.88, "low_confidence": false},
  "date":         {"value": "14/03/2024", "confidence": 0.97, "low_confidence": false},
  "items": [
    {"name": "Organic Bananas", "price": "2.49", "confidence": 0.88, "low_confidence": false}
  ],
  "total_amount": {"value": "12.63", "confidence": 0.97, "low_confidence": false},
  "ocr_avg_confidence": 84.03,
  "processing_notes": ["Corrected skew of -3.6 degrees.", "..."],
  "status": "ok"
}
```

`status` is one of `ok` (all core fields found and confident), `partial`
(some fields missing/low-confidence), or `failed` (unreadable image or no
text detected) — see `docs/DOCUMENTATION.md` for edge-case handling details.

## Note on the dataset

This copy was built and tested against synthetic receipt images (generated
with realistic noise, blur, and skew — see `docs/DOCUMENTATION.md`) because
the assignment's Google Drive dataset wasn't reachable from the build
environment. Point `--input` at the real dataset folder to run it there; no
code changes are needed.
