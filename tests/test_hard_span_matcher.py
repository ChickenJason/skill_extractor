from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import load_json  # noqa: E402
from self_consistent_annotation.hard_span_matcher import (  # noqa: E402
    anchor_coverage,
    dynamic_threshold,
    hard_match_family,
    normalize_text,
    word_units,
)


CONFIG = load_json(PROJECT_ROOT / "config" / "pipeline.json")["aggregation"]["hard_match"]


def member(label: str, text: str, start: int, end: int, indexes: list[int]) -> dict:
    return {
        "label_id": label,
        "text": text,
        "start": start,
        "end": end,
        "exact_votes": len(indexes),
        "sample_indexes": indexes,
    }


def match(
    members: list[dict],
    sample_indexes: list[int],
    *,
    valid_sample_count: int = 5,
) -> dict:
    return hard_match_family(
        members,
        sample_indexes=sample_indexes,
        valid_sample_count=valid_sample_count,
        exact_accept_votes=3,
        min_family_support=3,
        config=CONFIG,
    )


class HardSpanMatcherTests(unittest.TestCase):
    def test_normalization_length_units_and_thresholds(self) -> None:
        self.assertEqual(normalize_text("  C＋＋  NODE.JS "), "c++ node.js")
        self.assertEqual(word_units("C++ Node.js patient-centred"), 3)
        self.assertEqual(word_units("机器学习"), 2)
        self.assertEqual(dynamic_threshold(1, CONFIG), 0.45)
        self.assertEqual(dynamic_threshold(3, CONFIG), 0.55)
        self.assertEqual(dynamic_threshold(20, CONFIG), 0.85)

    def test_anchor_coverage_is_asymmetric(self) -> None:
        narrow = member("narrow", "planning", 12, 20, [0, 1])
        broad = member("broad", "project planning", 4, 20, [2])
        overlap, anchor_length, coverage = anchor_coverage(narrow, broad)
        self.assertEqual((overlap, anchor_length, coverage), (8, 8, 1.0))
        self.assertEqual(anchor_coverage(broad, narrow), (8, 16, 0.5))

    def test_most_frequent_span_is_anchor_and_full_coverage_passes(self) -> None:
        members = [
            member("anchor", "planning", 12, 20, [0, 1]),
            member("broad", "project planning", 4, 20, [2]),
        ]
        result = match(members, [0, 1, 2], valid_sample_count=3)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["winner_label_id"], "anchor")
        self.assertEqual(result["anchor_label_id"], "anchor")
        self.assertEqual(result["anchor_exact_votes"], 2)
        self.assertEqual(result["minimum_anchor_coverage"], 1.0)
        self.assertEqual(result["required_threshold"], 0.45)
        self.assertEqual(result["comparisons"][0]["overlap_characters"], 8)

    def test_long_anchor_uses_higher_threshold_and_rejects_partial_coverage(self) -> None:
        members = [
            member(
                "anchor",
                "atomic-scale model for SCR catalysis",
                12,
                48,
                [0, 1],
            ),
            member("partial", "SCR catalysis", 35, 48, [2]),
        ]
        result = match(members, [0, 1, 2])
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "family_anchor_overlap_failed")
        self.assertEqual(result["anchor_word_units"], 5)
        self.assertEqual(result["required_threshold"], 0.65)
        self.assertAlmostEqual(result["comparisons"][0]["anchor_coverage"], 13 / 36, 6)

    def test_equal_votes_use_maximin_then_compactness(self) -> None:
        members = [
            member("broad", "project planning", 4, 20, [0, 1]),
            member("compact", "planning", 12, 20, [2, 3]),
        ]
        result = match(members, [0, 1, 2, 3])
        self.assertTrue(result["accepted"])
        self.assertTrue(result["top_exact_vote_tie"])
        self.assertEqual(result["winner_label_id"], "compact")
        self.assertEqual(
            result["winner_selection_rule"][1:3],
            ["minimum_anchor_coverage_desc", "average_anchor_coverage_desc"],
        )

    def test_full_metric_tie_uses_start_then_end(self) -> None:
        members = [
            member("first", "aaa", 0, 3, [0, 1]),
            member("second", "aaa", 1, 4, [2, 3]),
        ]
        result = match(members, [0, 1, 2, 3])
        self.assertTrue(result["accepted"])
        self.assertEqual(result["winner_label_id"], "first")

    def test_overlap_chain_fails_through_zero_anchor_coverage(self) -> None:
        chain = [
            member("anchor", "alpha beta", 0, 10, [0, 1]),
            member("bridge", "beta gamma", 6, 16, [2]),
            member("tail", "gamma delta", 11, 22, [3]),
        ]
        result = match(chain, [0, 1, 2, 3], valid_sample_count=4)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "family_anchor_overlap_failed")
        tail = next(
            comparison
            for comparison in result["comparisons"]
            if comparison["other_label_id"] == "tail"
        )
        self.assertEqual(tail["anchor_coverage"], 0.0)

    def test_invalid_inputs_fail_closed(self) -> None:
        members = [
            member("anchor", "planning", 12, 20, [0, 1]),
            member("broad", "project planning", 4, 20, [2]),
        ]
        result = match(members, [0, 1, 2], valid_sample_count=2)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reason"], "hard_match_validation_failed")
        self.assertIn("three valid samples", result["validation_error"])


if __name__ == "__main__":
    unittest.main()
