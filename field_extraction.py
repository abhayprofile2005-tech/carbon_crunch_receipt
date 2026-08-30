"""
field_extraction.py
--------------------
Turns raw OCR'd lines into candidate values for the fields the assignment
asks for: store name, date, line items + prices, and total amount.

Every extractor returns not just a value but the "evidence" (which OCR
line(s) it came from, and whether a regex pattern / keyword heuristic
matched) so that confidence_scoring.py can score the field without
re-parsing text itself.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from ocr_engine import Line

# ---------------------------------------------------------------------------
# Shared patterns
# ---------------------------------------------------------------------------

CURRENCY_PRICE_RE = re.compile(
    r"(?:rs\.?|inr|₹|\$|€|£)?\s*"
    r"(\d{1,3}(?:[,.\s]\d{3})*(?:\.\d{1,2})?|\d+\.\d{1,2}|\d+)"
    r"\s*(?:rs\.?|inr|/-)?",
    re.IGNORECASE,
)

# A "money-like" token, standalone, used to find the price at the END of an item line.
TRAILING_PRICE_RE = re.compile(r"([\$€£₹]|rs\.?)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+\.\d{2})\s*$", re.IGNORECASE)

DATE_PATTERNS = [
    # DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY (also matches MM/DD/YYYY - ambiguous by design)
    re.compile(r"\b(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})\b"),
    # YYYY-MM-DD
    re.compile(r"\b(\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2})\b"),
    # 12 Jan 2024 / 12 January 2024
    re.compile(r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b", re.IGNORECASE),
    # Jan 12, 2024 / January 12 2024
    re.compile(r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})\b", re.IGNORECASE),
]

TOTAL_KEYWORDS = ["grand total", "total amount", "amount due", "balance due",
                   "net amount", "amount payable", "total"]
SUBTOTAL_KEYWORDS = ["subtotal", "sub total", "sub-total"]
EXCLUDE_FROM_ITEMS_KEYWORDS = [
    "total", "subtotal", "sub total", "tax", "vat", "gst", "cgst", "sgst",
    "cash", "change", "card", "balance", "due", "tender", "discount",
    "thank you", "receipt", "invoice", "date", "time", "cashier", "phone",
    "tel", "gstin", "www.", "http",
]
DATE_LINE_KEYWORDS = ["date", "dt.", "dt:", "time"]


@dataclass
class FieldCandidate:
    value: Optional[str]
    ocr_conf: float = 0.0          # average tesseract confidence (0-100) of the source line(s)
    pattern_matched: bool = False  # did a format regex confirm this looks right?
    keyword_matched: bool = False  # was a relevant keyword found nearby (e.g. "Total")?
    source_lines: list[str] = field(default_factory=list)
    found: bool = False


def _line_texts(lines: list[Line]) -> list[str]:
    return [ln.text for ln in lines]


# ---------------------------------------------------------------------------
# Store name
# ---------------------------------------------------------------------------

def extract_store_name(lines: list[Line]) -> FieldCandidate:
    """Heuristic: the store name is almost always in the first few lines,
    and is usually rendered in the largest font on the receipt (headers are
    printed bigger than item/price rows). We look at the first 6 non-empty
    lines, discard ones that are clearly not a name (pure numbers, dates,
    phone numbers, addresses with lots of digits), and pick the one with the
    tallest average word height as a proxy for "largest font"."""
    candidates = []
    for ln in lines[:6]:
        text = ln.text.strip()
        if not text or len(text) < 2:
            continue
        digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
        if digit_ratio > 0.4:
            continue  # looks like a phone number / address / date, not a name
        if any(re.search(p, text) for p in [d.pattern for d in DATE_PATTERNS]):
            continue
        avg_height = sum(w.height for w in ln.words) / max(len(ln.words), 1)
        candidates.append((avg_height, ln))

    if not candidates:
        return FieldCandidate(value=None, found=False)

    candidates.sort(key=lambda c: c[0], reverse=True)
    _, best_line = candidates[0]
    return FieldCandidate(
        value=best_line.text.strip(),
        ocr_conf=best_line.avg_conf,
        pattern_matched=True,  # "largest text in header zone" is the pattern here
        keyword_matched=False,
        source_lines=[best_line.text],
        found=True,
    )


# ---------------------------------------------------------------------------
# Date
# ---------------------------------------------------------------------------

def extract_date(lines: list[Line]) -> FieldCandidate:
    """Look for a date-shaped token anywhere on the receipt. Prefer lines
    that also contain a date-related keyword ("Date:", "Dt.") since receipts
    often print multiple date-like numbers (e.g. item codes)."""
    keyword_hits, plain_hits = [], []

    for ln in lines:
        text = ln.text
        for pat in DATE_PATTERNS:
            m = pat.search(text)
            if not m:
                continue
            has_keyword = any(kw in text.lower() for kw in DATE_LINE_KEYWORDS)
            hit = (m.group(1), ln)
            (keyword_hits if has_keyword else plain_hits).append(hit)

    chosen = keyword_hits[0] if keyword_hits else (plain_hits[0] if plain_hits else None)
    if chosen is None:
        return FieldCandidate(value=None, found=False)

    value, src_line = chosen
    return FieldCandidate(
        value=value,
        ocr_conf=src_line.avg_conf,
        pattern_matched=True,
        keyword_matched=bool(keyword_hits),
        source_lines=[src_line.text],
        found=True,
    )


# ---------------------------------------------------------------------------
# Total amount
# ---------------------------------------------------------------------------

def _parse_amount(token: str) -> Optional[float]:
    cleaned = re.sub(r"[^\d.]", "", token.replace(",", ""))
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_total(lines: list[Line]) -> FieldCandidate:
    """Search bottom-up (totals are usually near the end of a receipt) for a
    line containing a total-like keyword plus a trailing price. Falls back
    to subtotal, then to "largest number on the receipt" if no keyword line
    is found at all (common when OCR mangles the word 'Total')."""
    for ln in reversed(lines):
        low = ln.text.lower()
        if any(kw in low for kw in SUBTOTAL_KEYWORDS):
            continue  # skip subtotal when looking for the real total
        if any(kw in low for kw in TOTAL_KEYWORDS):
            m = TRAILING_PRICE_RE.search(ln.text)
            if not m:
                m = CURRENCY_PRICE_RE.search(ln.text)
            if m:
                amount = _parse_amount(m.group(0))
                if amount is not None:
                    return FieldCandidate(
                        value=f"{amount:.2f}", ocr_conf=ln.avg_conf,
                        pattern_matched=True, keyword_matched=True,
                        source_lines=[ln.text], found=True,
                    )

    # Fallback 1: subtotal line
    for ln in reversed(lines):
        if any(kw in ln.text.lower() for kw in SUBTOTAL_KEYWORDS):
            m = TRAILING_PRICE_RE.search(ln.text) or CURRENCY_PRICE_RE.search(ln.text)
            if m:
                amount = _parse_amount(m.group(0))
                if amount is not None:
                    return FieldCandidate(
                        value=f"{amount:.2f}", ocr_conf=ln.avg_conf,
                        pattern_matched=True, keyword_matched=True,
                        source_lines=[ln.text], found=True,
                    )

    # Fallback 2: largest currency-looking number anywhere (low confidence).
    best = None
    for ln in lines:
        for m in TRAILING_PRICE_RE.finditer(ln.text):
            amount = _parse_amount(m.group(0))
            if amount is not None and (best is None or amount > best[0]):
                best = (amount, ln)
    if best:
        amount, ln = best
        return FieldCandidate(
            value=f"{amount:.2f}", ocr_conf=ln.avg_conf,
            pattern_matched=True, keyword_matched=False,
            source_lines=[ln.text], found=True,
        )

    return FieldCandidate(value=None, found=False)


# ---------------------------------------------------------------------------
# Line items
# ---------------------------------------------------------------------------

@dataclass
class ItemCandidate:
    name: str
    price: Optional[str]
    ocr_conf: float
    pattern_matched: bool
    source_line: str


def extract_items(lines: list[Line]) -> list[ItemCandidate]:
    """Item lines are identified as: not matching any exclude keyword
    (total/tax/cash/etc.), and ending in a price-shaped token. The text
    before the price becomes the item name."""
    items: list[ItemCandidate] = []
    for ln in lines:
        low = ln.text.lower().strip()
        if not low:
            continue
        if any(kw in low for kw in EXCLUDE_FROM_ITEMS_KEYWORDS):
            continue

        m = TRAILING_PRICE_RE.search(ln.text)
        if not m:
            continue
        amount = _parse_amount(m.group(0))
        if amount is None:
            continue

        name = ln.text[: m.start()].strip(" -.:\t")
        if len(name) < 2:
            continue  # a lone number with no item description isn't a usable item row

        items.append(ItemCandidate(
            name=name,
            price=f"{amount:.2f}",
            ocr_conf=ln.avg_conf,
            pattern_matched=True,
            source_line=ln.text,
        ))
    return items
