"""
Lightweight unit tests for the pure-logic parts of the pipeline (regex-based
extraction and confidence scoring) that don't require an image or Tesseract.
Run with: python -m unittest discover tests
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ocr_engine import Line, Word
from src import field_extraction as fx
from src import confidence_scoring as cs


def make_line(text: str, conf: float = 90.0, height: int = 16) -> Line:
    """Build a minimal Line/Word pair for testing extraction logic without
    needing a real image or Tesseract call."""
    words = []
    x = 0
    for tok in text.split(" "):
        words.append(Word(text=tok, conf=conf, left=x, top=0, width=len(tok) * 8,
                           height=height, line_num=0, block_num=0))
        x += len(tok) * 8 + 8
    return Line(text=text, words=words, avg_conf=conf, top=0)


class TestDateExtraction(unittest.TestCase):
    def test_numeric_date_with_keyword(self):
        lines = [make_line("Date: 14/03/2024")]
        result = fx.extract_date(lines)
        self.assertTrue(result.found)
        self.assertEqual(result.value, "14/03/2024")
        self.assertTrue(result.keyword_matched)

    def test_month_name_date(self):
        lines = [make_line("22 Jun 2024")]
        result = fx.extract_date(lines)
        self.assertTrue(result.found)
        self.assertIn("Jun", result.value)

    def test_no_date_found(self):
        lines = [make_line("Thank you for shopping")]
        result = fx.extract_date(lines)
        self.assertFalse(result.found)


class TestTotalExtraction(unittest.TestCase):
    def test_finds_total_over_subtotal(self):
        lines = [make_line("Subtotal 20.00"), make_line("TOTAL 21.84")]
        result = fx.extract_total(lines)
        self.assertEqual(result.value, "21.84")
        self.assertTrue(result.keyword_matched)

    def test_falls_back_to_subtotal(self):
        lines = [make_line("Subtotal 20.00")]
        result = fx.extract_total(lines)
        self.assertEqual(result.value, "20.00")

    def test_no_keyword_falls_back_to_largest_number_low_confidence(self):
        # No "total"/"subtotal" keyword anywhere -> last-resort fallback
        # picks the largest currency-shaped number, but without a keyword
        # match, so confidence scoring will flag it as low-confidence.
        lines = [make_line("Item A 5.00")]
        result = fx.extract_total(lines)
        self.assertTrue(result.found)
        self.assertFalse(result.keyword_matched)

    def test_truly_empty_receipt_finds_nothing(self):
        lines = [make_line("Thank you for visiting")]
        result = fx.extract_total(lines)
        self.assertFalse(result.found)


class TestItemExtraction(unittest.TestCase):
    def test_extracts_item_and_price(self):
        lines = [make_line("Organic Bananas 2.49"), make_line("TOTAL 2.49")]
        items = fx.extract_items(lines)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].name, "Organic Bananas")
        self.assertEqual(items[0].price, "2.49")

    def test_excludes_tax_and_total_lines(self):
        lines = [make_line("Item A 5.00"), make_line("Tax 0.50"), make_line("Total 5.50")]
        items = fx.extract_items(lines)
        names = [i.name for i in items]
        self.assertIn("Item A", names)
        self.assertNotIn("Tax", names)
        self.assertNotIn("Total", names)


class TestConfidenceScoring(unittest.TestCase):
    def test_missing_field_gets_zero_confidence(self):
        scored = cs.score_field(None, 0, False, False, found=False)
        self.assertEqual(scored.confidence, 0.0)
        self.assertTrue(scored.low_confidence)

    def test_high_ocr_pattern_and_keyword_gives_high_confidence(self):
        scored = cs.score_field("21.84", 95.0, True, True, found=True)
        self.assertGreaterEqual(scored.confidence, 0.9)
        self.assertFalse(scored.low_confidence)

    def test_low_ocr_confidence_flagged(self):
        scored = cs.score_field("21.84", 30.0, True, False, found=True)
        self.assertTrue(scored.low_confidence)


if __name__ == "__main__":
    unittest.main()
