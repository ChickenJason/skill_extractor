"""Stage 1: build the formal TRF corpus from immutable self-annotation decisions."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import atomic_write_json, atomic_write_jsonl, read_jsonl  # noqa: E402
from trf.common import (  # noqa: E402
    TRFError,
    execute_stage,
    initialize_run,
    load_trf_config,
    parse_config_argument,
    source_files,
)


FORMAL_STATUSES = {"accepted", "negative"}
EXCLUDED_STATUSES = {"unsolved", "abstained"}


def existence_score(decision: dict[str, Any]) -> float:
    consensus = decision.get("existence_consensus")
    if not isinstance(consensus, dict):
        raise TRFError(f"idx={decision.get('idx')} has no existence_consensus")
    valid_votes = consensus.get("valid_votes")
    if not isinstance(valid_votes, int) or valid_votes <= 0:
        raise TRFError(f"idx={decision.get('idx')} has invalid valid_votes")
    status = decision.get("status")
    numerator_key = "has_skill_votes" if status == "accepted" else "no_skill_votes"
    numerator = consensus.get(numerator_key)
    if not isinstance(numerator, int) or numerator < 0 or numerator > valid_votes:
        raise TRFError(f"idx={decision.get('idx')} has invalid {numerator_key}")
    return numerator / valid_votes


def validate_span(idx: int, sentence: str, span: dict[str, Any]) -> dict[str, Any]:
    required = ("text", "start", "end", "exact_votes", "accepted_by")
    if any(key not in span for key in required):
        raise TRFError(f"idx={idx} accepted span is missing required fields")
    text = span["text"]
    start = span["start"]
    end = span["end"]
    if not isinstance(text, str) or not isinstance(start, int) or not isinstance(end, int):
        raise TRFError(f"idx={idx} accepted span has invalid field types")
    if start < 0 or end <= start or end > len(sentence) or sentence[start:end] != text:
        raise TRFError(
            f"idx={idx} span offset mismatch: {start}:{end}={text!r}, "
            f"sentence slice={sentence[start:end]!r}"
        )
    if span["accepted_by"] not in {"exact_vote", "family_hard_match"}:
        raise TRFError(f"idx={idx} has unknown accepted_by={span['accepted_by']!r}")
    return {
        "text": text,
        "start": start,
        "end": end,
        "exact_votes": span["exact_votes"],
        "accepted_by": span["accepted_by"],
    }


def build_corpus_records(
    decisions: list[dict[str, Any]], expected: dict[str, int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    counts: Counter[str] = Counter()
    corpus: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    seen_indexes: set[int] = set()

    for decision in decisions:
        idx = decision.get("idx")
        sentence = decision.get("sentence")
        status = decision.get("status")
        if not isinstance(idx, int) or idx in seen_indexes:
            raise TRFError(f"Decision idx must be a unique integer: {idx!r}")
        seen_indexes.add(idx)
        if not isinstance(sentence, str) or not sentence:
            raise TRFError(f"idx={idx} sentence must be a non-empty string")
        if status not in FORMAL_STATUSES | EXCLUDED_STATUSES:
            raise TRFError(f"idx={idx} has unknown status={status!r}")
        counts[status] += 1

        score = existence_score(decision)
        if status == "accepted":
            raw_spans = decision.get("accepted_span_details")
            if not isinstance(raw_spans, list) or not raw_spans:
                raise TRFError(f"idx={idx} accepted decision has no accepted spans")
            spans = [validate_span(idx, sentence, span) for span in raw_spans]
            spans.sort(key=lambda span: (span["start"], span["end"], span["text"]))
            corpus.append(
                {
                    "idx": idx,
                    "sentence": sentence,
                    "status": status,
                    "pseudo_types": ["Skill"],
                    "skill_spans": spans,
                    "existence_score": score,
                    "eligible_for_statistics": True,
                    "eligible_as_demonstration": True,
                }
            )
        elif status == "negative":
            if decision.get("accepted_span_details") or decision.get("spans"):
                raise TRFError(f"idx={idx} negative decision unexpectedly contains spans")
            corpus.append(
                {
                    "idx": idx,
                    "sentence": sentence,
                    "status": status,
                    "pseudo_types": [],
                    "skill_spans": [],
                    "existence_score": score,
                    "eligible_for_statistics": True,
                    "eligible_as_demonstration": True,
                }
            )
        else:
            excluded.append(
                {
                    "idx": idx,
                    "sentence": sentence,
                    "status": status,
                    "pseudo_types": [],
                    "skill_spans": [],
                    "existence_score": score,
                    "eligible_for_statistics": False,
                    "eligible_as_demonstration": False,
                    "reasons": decision.get("reasons", []),
                }
            )

    actual = {
        "total": len(decisions),
        "accepted": counts["accepted"],
        "negative": counts["negative"],
        "unsolved": counts["unsolved"],
        "abstained": counts["abstained"],
        "formal": len(corpus),
    }
    if actual != expected:
        raise TRFError(f"Source decision counts violate the fixed contract: {actual} != {expected}")
    if len(excluded) != expected["unsolved"] + expected["abstained"]:
        raise TRFError("Excluded-record count is inconsistent")

    corpus.sort(key=lambda record: record["idx"])
    excluded.sort(key=lambda record: record["idx"])
    summary = {
        "schema_version": "trf-corpus-v1",
        "counts": actual,
        "accepted_spans": sum(len(record["skill_spans"]) for record in corpus),
        "accepted_by": dict(
            sorted(
                Counter(
                    span["accepted_by"]
                    for record in corpus
                    for span in record["skill_spans"]
                ).items()
            )
        ),
    }
    return corpus, excluded, summary


def build_manual_review_sample(corpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hard = [
        record
        for record in corpus
        if record["status"] == "accepted"
        and any(span["accepted_by"] == "family_hard_match" for span in record["skill_spans"])
    ][:5]
    hard_ids = {record["idx"] for record in hard}
    exact = [
        record
        for record in corpus
        if record["status"] == "accepted"
        and record["idx"] not in hard_ids
        and all(span["accepted_by"] == "exact_vote" for span in record["skill_spans"])
    ][:10]
    negative = [record for record in corpus if record["status"] == "negative"][:5]
    if len(exact) != 10 or len(hard) != 5 or len(negative) != 5:
        raise TRFError("Cannot construct the fixed 10/5/5 manual review sample")
    sample: list[dict[str, Any]] = []
    for category, records in (
        ("exact_accepted", exact),
        ("hard_match_accepted", hard),
        ("negative", negative),
    ):
        for record in records:
            sample.append({"review_category": category, **record})
    return sample


def write_corpus(config: dict[str, Any], run_root: Any) -> dict[str, Any]:
    decisions = read_jsonl(source_files(config).decisions)
    corpus, excluded, summary = build_corpus_records(
        decisions, config["source"]["expected_counts"]
    )
    run_root.corpus.mkdir(parents=True, exist_ok=False)
    run_root.audit.mkdir(parents=True, exist_ok=False)
    atomic_write_jsonl(run_root.corpus / "records.jsonl", corpus)
    atomic_write_jsonl(run_root.corpus / "excluded.jsonl", excluded)
    atomic_write_json(run_root.corpus / "summary.json", summary)
    atomic_write_jsonl(
        run_root.audit / "manual_review_sample.jsonl", build_manual_review_sample(corpus)
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/trf.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--self-annotator-run-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = parse_config_argument(args.config)
    paths = None
    try:
        config = load_trf_config(config_path, args.self_annotator_run_id)
        paths = initialize_run(config_path, config, args.run_id, sys.argv)
        execute_stage(
            config,
            paths,
            "build_corpus",
            None,
            lambda: write_corpus(config, paths),
        )
    except Exception as error:
        print(f"BuildTRFCorpus failed: {error}", file=sys.stderr)
        return 1
    print(f"TRF corpus completed: {paths.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
