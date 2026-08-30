#!/usr/bin/env python3
"""
main.py
-------
CLI entry point for the Carbon Crunch OCR receipt pipeline.

Usage:
    python main.py --input sample_data --output outputs

Processes every image in --input, writes one JSON file per receipt into
<output>/json/, and writes an aggregate expense summary to
<output>/summary.json.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import process_receipt
from src.financial_summary import build_summary

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_images(input_dir: str) -> list[str]:
    if not os.path.isdir(input_dir):
        return []
    paths = []
    for fname in sorted(os.listdir(input_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in VALID_EXTENSIONS:
            paths.append(os.path.join(input_dir, fname))
    return paths


def main():
    parser = argparse.ArgumentParser(description="Carbon Crunch receipt OCR pipeline")
    parser.add_argument("--input", default="sample_data", help="Folder of receipt images")
    parser.add_argument("--output", default="outputs", help="Folder to write JSON + summary into")
    args = parser.parse_args()

    json_dir = os.path.join(args.output, "json")
    os.makedirs(json_dir, exist_ok=True)

    image_paths = find_images(args.input)
    if not image_paths:
        print(f"No receipt images found in '{args.input}'. "
              f"Supported extensions: {sorted(VALID_EXTENSIONS)}", file=sys.stderr)
        # Still write an empty summary so downstream consumers don't break
        # on a missing file -- an empty batch is an edge case, not a crash.
        empty_summary = build_summary([])
        with open(os.path.join(args.output, "summary.json"), "w") as f:
            json.dump(asdict(empty_summary), f, indent=2)
        sys.exit(1)

    results = []
    print(f"Found {len(image_paths)} receipt image(s) in '{args.input}'.\n")

    for path in image_paths:
        fname = os.path.basename(path)
        print(f"Processing {fname} ...", end=" ")
        try:
            result = process_receipt(path)
        except Exception as e:
            # Last-resort safety net: one bad receipt should never crash the
            # whole batch. Record it as failed and keep going.
            result = {
                "source_file": path,
                "store_name": {"value": None, "confidence": 0.0, "low_confidence": True},
                "date": {"value": None, "confidence": 0.0, "low_confidence": True},
                "items": [],
                "total_amount": {"value": None, "confidence": 0.0, "low_confidence": True},
                "ocr_avg_confidence": 0.0,
                "processing_notes": [f"Unhandled error: {e}"],
                "status": "failed",
            }
        results.append(result)

        out_path = os.path.join(json_dir, os.path.splitext(fname)[0] + ".json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"status={result['status']} "
              f"store={result['store_name']['value']!r} "
              f"total={result['total_amount']['value']!r}")

    summary = build_summary(results)
    with open(os.path.join(args.output, "summary.json"), "w") as f:
        json.dump(asdict(summary), f, indent=2)

    print("\n--- Expense Summary ---")
    print(f"Receipts processed:        {summary.num_receipts_processed}")
    print(f"Transactions with a total: {summary.num_transactions}")
    print(f"Total spend:               {summary.total_spend}")
    print(f"Spend per store:           {summary.spend_per_store}")
    if summary.flagged_receipts:
        print(f"Flagged for review:        {len(summary.flagged_receipts)} receipt(s) -- see summary.json")
    print(f"\nPer-receipt JSON written to: {json_dir}/")
    print(f"Summary written to:          {os.path.join(args.output, 'summary.json')}")


if __name__ == "__main__":
    main()
