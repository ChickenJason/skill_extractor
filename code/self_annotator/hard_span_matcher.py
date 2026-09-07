"""Deterministic length-aware anchor coverage for overlap families."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any


CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
TECH_WORD_RE = re.compile(
    r"(?iu)(?:\.[a-z0-9]+(?:\.[a-z0-9]+)*)|"
    r"(?:[a-z0-9]+(?:[.+#/\-][a-z0-9]+)*(?:\+{1,2}|#)?)"
)


def _rounded(value: float) -> float:
    return round(value, 6)


def normalize_text(text: str) -> str:
    """NFKC-normalize, lowercase, and collapse whitespace for length counting."""

    if not isinstance(text, str):
        raise TypeError("span text must be a string")
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def word_units(text: str) -> int:
    """Return the technical-word plus half-CJK length proxy for a span."""

    normalized = normalize_text(text)
    cjk_matches = list(CJK_RE.finditer(normalized))
    cjk_count = sum(len(match.group()) for match in cjk_matches)
    cjk_ranges = [match.span() for match in cjk_matches]
    word_count = sum(
        not any(start < match.end() and match.start() < end for start, end in cjk_ranges)
        for match in TECH_WORD_RE.finditer(normalized)
    )
    return max(1, word_count + math.ceil(cjk_count / 2))


def dynamic_threshold(unit_count: int, config: dict[str, Any]) -> float:
    """Return the unrounded threshold for the anchor's own word-unit length."""

    if isinstance(unit_count, bool) or not isinstance(unit_count, int) or unit_count < 1:
        raise ValueError("unit_count must be a positive integer")
    return min(
        float(config["threshold_cap"]),
        float(config["threshold_base"])
        + float(config["threshold_step"]) * (unit_count - 1),
    )


def positive_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return max(int(left["start"]), int(right["start"])) < min(
        int(left["end"]), int(right["end"])
    )


def anchor_coverage(anchor: dict[str, Any], other: dict[str, Any]) -> tuple[int, int, float]:
    """Return character overlap, anchor length, and overlap/anchor length."""

    anchor_start, anchor_end = int(anchor["start"]), int(anchor["end"])
    other_start, other_end = int(other["start"]), int(other["end"])
    anchor_characters = anchor_end - anchor_start
    if anchor_characters <= 0:
        raise ValueError("anchor span must have positive character length")
    overlap_characters = max(
        0,
        min(anchor_end, other_end) - max(anchor_start, other_start),
    )
    return (
        overlap_characters,
        anchor_characters,
        overlap_characters / anchor_characters,
    )


def _validate_member(member: dict[str, Any]) -> None:
    for field in ("label_id", "text", "start", "end", "exact_votes", "sample_indexes"):
        if field not in member:
            raise ValueError(f"hard-match member is missing {field}")
    if not isinstance(member["label_id"], str) or not member["label_id"]:
        raise TypeError("hard-match label_id must be a non-empty string")
    if not isinstance(member["text"], str) or not member["text"]:
        raise TypeError("hard-match text must be a non-empty string")
    start, end = member["start"], member["end"]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end)):
        raise TypeError("hard-match offsets must be integers")
    if start < 0 or end <= start:
        raise ValueError("hard-match span must have positive character length")
    exact_votes = member["exact_votes"]
    if isinstance(exact_votes, bool) or not isinstance(exact_votes, int) or exact_votes < 1:
        raise TypeError("hard-match exact_votes must be a positive integer")
    indexes = member["sample_indexes"]
    if not isinstance(indexes, list) or any(
        isinstance(index, bool) or not isinstance(index, int) for index in indexes
    ):
        raise TypeError("hard-match sample_indexes must be integer lists")
    if len(indexes) != len(set(indexes)):
        raise ValueError("hard-match sample_indexes must be unique")
    if exact_votes != len(indexes):
        raise ValueError("hard-match exact_votes must equal sample index support")


def _candidate_metrics(
    anchor: dict[str, Any],
    members: list[dict[str, Any]],
) -> dict[str, Any]:
    coverages = [
        anchor_coverage(anchor, other)[2]
        for other in members
        if other["label_id"] != anchor["label_id"]
    ]
    units = word_units(anchor["text"])
    return {
        "member": anchor,
        "exact_votes": int(anchor["exact_votes"]),
        "minimum_anchor_coverage": min(coverages),
        "average_anchor_coverage": sum(coverages) / len(coverages),
        "word_units": units,
        "character_length": int(anchor["end"]) - int(anchor["start"]),
    }


def _select_anchor(
    metrics: list[dict[str, Any]], epsilon: float
) -> tuple[dict[str, Any], list[str]]:
    top_votes = max(item["exact_votes"] for item in metrics)
    finalists = [item for item in metrics if item["exact_votes"] == top_votes]
    top_vote_label_ids = sorted(item["member"]["label_id"] for item in finalists)

    best_minimum = max(item["minimum_anchor_coverage"] for item in finalists)
    finalists = [
        item
        for item in finalists
        if item["minimum_anchor_coverage"] + epsilon >= best_minimum
    ]
    best_average = max(item["average_anchor_coverage"] for item in finalists)
    finalists = [
        item
        for item in finalists
        if item["average_anchor_coverage"] + epsilon >= best_average
    ]
    winner = min(
        finalists,
        key=lambda item: (
            item["word_units"],
            item["character_length"],
            int(item["member"]["start"]),
            int(item["member"]["end"]),
        ),
    )
    return winner, top_vote_label_ids


def hard_match_family(
    members: list[dict[str, Any]],
    *,
    sample_indexes: list[int],
    valid_sample_count: int,
    exact_accept_votes: int,
    min_family_support: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Choose the modal model span and require every member to cover its anchor."""

    selection_rule = [
        "exact_votes_desc",
        "minimum_anchor_coverage_desc",
        "average_anchor_coverage_desc",
        "word_units_asc",
        "character_length_asc",
        "start_end_asc",
    ]
    base: dict[str, Any] = {
        "accepted": False,
        "reason": "hard_match_validation_failed",
        "winner_label_id": None,
        "anchor_label_id": None,
        "anchor_exact_votes": None,
        "minimum_anchor_coverage": None,
        "average_anchor_coverage": None,
        "anchor_word_units": None,
        "comparisons": [],
        "candidate_rankings": [],
        "winner_selection_rule": selection_rule,
        "overlap_metric": "anchor_character_coverage",
        "length_unit": "word_units",
    }
    try:
        if isinstance(valid_sample_count, bool) or not isinstance(valid_sample_count, int):
            raise TypeError("valid_sample_count must be an integer")
        if valid_sample_count < 3 or valid_sample_count > 5:
            raise ValueError("hard match requires a strict majority of three valid samples")
        if min_family_support != 3:
            raise ValueError("min_family_support must be exactly 3")
        if (
            len(sample_indexes) != len(set(sample_indexes))
            or sample_indexes != sorted(sample_indexes)
            or any(index not in range(5) for index in sample_indexes)
        ):
            raise ValueError("family sample indexes must be a sorted unique subset of 0..4")
        if len(sample_indexes) < min_family_support:
            raise ValueError("family support is below the hard-match threshold")
        if len(sample_indexes) > valid_sample_count:
            raise ValueError("family support cannot exceed the valid sample count")
        if len(members) < 2:
            raise ValueError("hard match requires at least two unique spans")
        for member in members:
            _validate_member(member)
        label_ids = [member["label_id"] for member in members]
        if len(label_ids) != len(set(label_ids)):
            raise ValueError("hard-match label_id values must be unique")
        contributed = sorted(
            {index for member in members for index in member["sample_indexes"]}
        )
        if contributed != sample_indexes:
            raise ValueError("family support must equal the contributing sample indexes")
        if max(int(member["exact_votes"]) for member in members) >= exact_accept_votes:
            raise ValueError("hard match cannot override an exact accepted span")
        if config.get("overlap_metric") != "anchor_character_coverage":
            raise ValueError("hard-match overlap_metric must be anchor_character_coverage")
        if config.get("length_unit") != "word_units":
            raise ValueError("hard-match length_unit must be word_units")

        epsilon = float(config["comparison_epsilon"])
        metrics = [_candidate_metrics(member, members) for member in members]
        winner_metrics, top_vote_label_ids = _select_anchor(metrics, epsilon)
        anchor = winner_metrics["member"]
        threshold = dynamic_threshold(winner_metrics["word_units"], config)

        comparisons: list[dict[str, Any]] = []
        for other in members:
            if other["label_id"] == anchor["label_id"]:
                continue
            overlap_characters, anchor_characters, coverage = anchor_coverage(anchor, other)
            comparisons.append(
                {
                    "anchor_label_id": anchor["label_id"],
                    "other_label_id": other["label_id"],
                    "overlap_characters": overlap_characters,
                    "anchor_characters": anchor_characters,
                    "anchor_coverage": _rounded(coverage),
                    "anchor_word_units": winner_metrics["word_units"],
                    "required_threshold": _rounded(threshold),
                    "passed": coverage + epsilon >= threshold,
                }
            )

        passed = all(comparison["passed"] for comparison in comparisons)
        candidate_rankings = [
            {
                "label_id": item["member"]["label_id"],
                "exact_votes": item["exact_votes"],
                "eligible_as_anchor": item["exact_votes"]
                == winner_metrics["exact_votes"],
                "minimum_anchor_coverage": _rounded(item["minimum_anchor_coverage"]),
                "average_anchor_coverage": _rounded(item["average_anchor_coverage"]),
                "word_units": item["word_units"],
                "character_length": item["character_length"],
            }
            for item in sorted(
                metrics,
                key=lambda item: (
                    -item["exact_votes"],
                    -item["minimum_anchor_coverage"],
                    -item["average_anchor_coverage"],
                    item["word_units"],
                    item["character_length"],
                    int(item["member"]["start"]),
                    int(item["member"]["end"]),
                ),
            )
        ]
        return {
            **base,
            "accepted": passed,
            "reason": "family_hard_match" if passed else "family_anchor_overlap_failed",
            "winner_label_id": anchor["label_id"] if passed else None,
            "proposed_winner_label_id": anchor["label_id"],
            "anchor_label_id": anchor["label_id"],
            "anchor_exact_votes": winner_metrics["exact_votes"],
            "minimum_anchor_coverage": _rounded(
                winner_metrics["minimum_anchor_coverage"]
            ),
            "average_anchor_coverage": _rounded(
                winner_metrics["average_anchor_coverage"]
            ),
            "anchor_word_units": winner_metrics["word_units"],
            "required_threshold": _rounded(threshold),
            "top_exact_vote_tie": len(top_vote_label_ids) > 1,
            "top_exact_vote_label_ids": top_vote_label_ids,
            "comparisons": comparisons,
            "candidate_rankings": candidate_rankings,
        }
    except (KeyError, TypeError, ValueError, ArithmeticError) as error:
        return {**base, "validation_error": str(error)}
