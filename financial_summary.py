"""
financial_summary.py
---------------------
Aggregates per-receipt structured JSON results into an overall expense
summary: total spend, transaction count, and spend per store. Designed to
be robust to receipts where fields are missing or low-confidence.
"""

from dataclasses import dataclass, field


@dataclass
class ExpenseSummary:
    total_spend: float
    num_transactions: int
    num_receipts_processed: int
    num_receipts_with_valid_total: int
    spend_per_store: dict
    flagged_receipts: list = field(default_factory=list)  # receipts with missing/low-confidence totals


def build_summary(receipts: list[dict]) -> ExpenseSummary:
    """`receipts` is a list of the per-receipt structured dicts produced by
    pipeline.py (see its docstring for the schema)."""
    total_spend = 0.0
    valid_total_count = 0
    spend_per_store: dict[str, float] = {}
    flagged = []

    for r in receipts:
        filename = r.get("source_file", "unknown")
        total_field = r.get("total_amount", {})
        store_field = r.get("store_name", {})

        total_value = total_field.get("value")
        store_value = store_field.get("value") or "Unknown store"

        if total_value is None:
            flagged.append({"file": filename, "reason": "total_amount missing"})
            continue

        try:
            amount = float(total_value)
        except (TypeError, ValueError):
            flagged.append({"file": filename, "reason": f"total_amount not numeric: {total_value!r}"})
            continue

        if total_field.get("confidence", 0) < 0.7:
            flagged.append({"file": filename, "reason": "total_amount is low-confidence", "value": amount})
            # Still counted below -- flagged for review, not silently dropped.

        total_spend += amount
        valid_total_count += 1
        spend_per_store[store_value] = spend_per_store.get(store_value, 0.0) + amount

    return ExpenseSummary(
        total_spend=round(total_spend, 2),
        num_transactions=valid_total_count,
        num_receipts_processed=len(receipts),
        num_receipts_with_valid_total=valid_total_count,
        spend_per_store={k: round(v, 2) for k, v in spend_per_store.items()},
        flagged_receipts=flagged,
    )
