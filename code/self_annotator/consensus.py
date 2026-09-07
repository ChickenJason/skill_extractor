"""Aggregate five inline-span samples with has-skill gating and hard matching."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    build_self_annotation_run_paths,
    load_json,
    read_jsonl,
    resolve_project_path,
    utc_now,
    validate_run_id,
)
from self_annotator.hard_span_matcher import (  # noqa: E402
    hard_match_family,
    positive_overlap,
)


CONSENSUS_SCHEMA = "span-xmlc-majority-anchor-overlap-v2"
EXPECTED_SAMPLE_INDEXES = list(range(5))
DECISION_STATUSES = ("accepted", "unsolved", "abstained", "negative")


def _entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -probability * math.log2(probability) - (1.0 - probability) * math.log2(1.0 - probability)


def _rounded(value: float) -> float:
    return round(value, 6)


def _group_samples(records: list[dict[str, Any]]) -> list[tuple[int, str, list[dict[str, Any]]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    sentences: dict[int, str] = {}
    for record in records:
        idx = record.get("idx")
        sentence = record.get("sentence")
        sample_index = record.get("sample_index")
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise TypeError("Parsed sample idx must be an integer")
        if not isinstance(sentence, str):
            raise TypeError(f"Parsed sentence must be a string for idx={idx}")
        if idx in sentences and sentences[idx] != sentence:
            raise ValueError(f"Conflicting sentences for idx={idx}")
        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise TypeError(f"sample_index must be an integer for idx={idx}")
        sentences[idx] = sentence
        grouped[idx].append(record)

    result: list[tuple[int, str, list[dict[str, Any]]]] = []
    for idx in sorted(grouped):
        samples = sorted(grouped[idx], key=lambda item: item["sample_index"])
        actual = [sample["sample_index"] for sample in samples]
        if actual != EXPECTED_SAMPLE_INDEXES:
            raise ValueError(
                f"idx={idx} must contain exactly sample_index 0..4; found {actual}"
            )
        result.append((idx, sentences[idx], samples))
    return result


def _validated_annotation(
    sample: dict[str, Any], sentence: str, idx: int
) -> tuple[dict[str, Any] | None, str | None]:
    if sample.get("parse_status") not in {"ok", "recovered"}:
        return None, str(sample.get("parse_error") or "parse_failed")
    annotation = sample.get("annotation")
    if not isinstance(annotation, dict):
        return None, "valid parse status without annotation"
    has_skill = annotation.get("has_skill")
    spans = annotation.get("spans")
    if isinstance(has_skill, bool) or has_skill not in (0, 1):
        return None, "annotation.has_skill must be 0 or 1"
    if not isinstance(spans, list):
        return None, "annotation.spans must be a list"
    derived_has_skill = 1 if spans else 0
    if has_skill != derived_has_skill:
        return None, "annotation.has_skill disagrees with annotation.spans"

    validated: list[dict[str, Any]] = []
    ranges: set[tuple[int, int]] = set()
    for position, span in enumerate(spans):
        if not isinstance(span, dict):
            return None, f"annotation.spans[{position}] must be an object"
        text, start, end = span.get("text"), span.get("start"), span.get("end")
        if not isinstance(text, str) or not text:
            return None, f"annotation.spans[{position}].text must be non-empty"
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end)):
            return None, f"annotation.spans[{position}] offsets must be integers"
        if start < 0 or end <= start or end > len(sentence):
            return None, f"annotation.spans[{position}] offsets are out of range"
        if sentence[start:end] != text:
            return None, f"annotation.spans[{position}] does not match the sentence"
        if (start, end) in ranges:
            return None, f"annotation.spans[{position}] duplicates an exact range"
        ranges.add((start, end))
        validated.append(
            {
                "label_id": f"span:{idx}:{start}:{end}",
                "text": text,
                "start": start,
                "end": end,
            }
        )
    validated.sort(key=lambda item: (item["start"], item["end"]))
    for left, right in zip(validated, validated[1:]):
        if positive_overlap(left, right):
            return None, "spans within one sample cannot overlap"
    return {"has_skill": derived_has_skill, "spans": validated}, None


def _label_evidence(
    valid_samples: list[tuple[int, dict[str, Any]]]
) -> list[dict[str, Any]]:
    evidence: dict[tuple[int, int], dict[str, Any]] = {}
    for sample_index, annotation in valid_samples:
        for span in annotation["spans"]:
            key = (span["start"], span["end"])
            label = evidence.setdefault(
                key,
                {
                    "label_id": span["label_id"],
                    "text": span["text"],
                    "start": span["start"],
                    "end": span["end"],
                    "sample_indexes": [],
                },
            )
            label["sample_indexes"].append(sample_index)
    labels = []
    for label in evidence.values():
        label["sample_indexes"].sort()
        label["exact_votes"] = len(label["sample_indexes"])
        labels.append(label)
    return sorted(labels, key=lambda item: (item["start"], item["end"]))


def _overlap_components(labels: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    unseen = set(range(len(labels)))
    components: list[list[dict[str, Any]]] = []
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        stack = [seed]
        component = [seed]
        while stack:
            current = stack.pop()
            neighbors = [
                candidate
                for candidate in sorted(unseen)
                if positive_overlap(labels[current], labels[candidate])
            ]
            for neighbor in neighbors:
                unseen.remove(neighbor)
                stack.append(neighbor)
                component.append(neighbor)
        components.append([labels[index] for index in sorted(component)])
    return sorted(
        components,
        key=lambda component: min((item["start"], item["end"]) for item in component),
    )


def _family_audit_member(label: dict[str, Any]) -> dict[str, Any]:
    return {
        "label_id": label["label_id"],
        "text": label["text"],
        "start": label["start"],
        "end": label["end"],
        "exact_votes": label["exact_votes"],
        "sample_indexes": label["sample_indexes"],
    }


def _uncertainty_record(
    idx: int,
    sentence: str,
    valid_samples: list[tuple[int, dict[str, Any]]],
    status: str,
) -> dict[str, Any]:
    sample_sets = {
        sample_index: {span["label_id"] for span in annotation["spans"]}
        for sample_index, annotation in valid_samples
    }
    pairwise: list[dict[str, Any]] = []
    for left, right in combinations(sorted(sample_sets), 2):
        union = sample_sets[left] | sample_sets[right]
        score = len(sample_sets[left] & sample_sets[right]) / len(union) if union else 1.0
        pairwise.append({"left": left, "right": right, "jaccard": _rounded(score)})
    all_labels = set().union(*sample_sets.values()) if sample_sets else set()
    entropies = [
        _entropy(sum(label in labels for labels in sample_sets.values()) / len(sample_sets))
        for label in all_labels
    ]
    cardinalities = [len(sample_sets[index]) for index in sorted(sample_sets)]
    return {
        "idx": idx,
        "sentence": sentence,
        "status": status,
        "valid_sample_indexes": sorted(sample_sets),
        "sample_label_ids": {
            str(index): sorted(sample_sets[index]) for index in sorted(sample_sets)
        },
        "pairwise_set_jaccard": pairwise,
        "mean_pairwise_set_jaccard": _rounded(
            sum(item["jaccard"] for item in pairwise) / len(pairwise)
        ) if pairwise else None,
        "label_entropy": _rounded(sum(entropies) / len(entropies)) if entropies else 0.0,
        "cardinality": {
            "per_sample": cardinalities,
            "mean": _rounded(sum(cardinalities) / len(cardinalities)) if cardinalities else None,
            "minimum": min(cardinalities) if cardinalities else None,
            "maximum": max(cardinalities) if cardinalities else None,
        },
    }


def _base_consensus(
    idx: int,
    sentence: str,
    status: str,
    has_skill: int | None,
    reasons: list[str],
    existence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "idx": idx,
        "sentence": sentence,
        "status": status,
        "has_skill": has_skill,
        "spans": [],
        "existence_consensus": existence,
        "accepted_span_details": [],
        "candidate_spans": [],
        "unresolved_families": [],
        "reasons": reasons,
    }


def aggregate_sentence(
    idx: int,
    sentence: str,
    samples: list[dict[str, Any]],
    aggregation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    valid_samples: list[tuple[int, dict[str, Any]]] = []
    invalid_samples: list[dict[str, Any]] = []
    for sample in samples:
        annotation, error = _validated_annotation(sample, sentence, idx)
        sample_index = sample["sample_index"]
        if annotation is None:
            invalid_samples.append({"sample_index": sample_index, "reason": error})
        else:
            valid_samples.append((sample_index, annotation))

    valid_count = len(valid_samples)
    has_votes = sum(annotation["has_skill"] == 1 for _, annotation in valid_samples)
    no_votes = valid_count - has_votes
    existence = {
        "valid_votes": valid_count,
        "invalid_votes": len(invalid_samples),
        "has_skill_votes": has_votes,
        "no_skill_votes": no_votes,
        "min_valid_samples": aggregation["min_valid_samples"],
        "min_has_skill_votes": aggregation["min_has_skill_votes"],
        "min_no_skill_votes": aggregation["min_no_skill_votes"],
        "outcome": None,
    }
    audit: dict[str, Any] = {
        "idx": idx,
        "sentence": sentence,
        "existence_consensus": existence,
        "invalid_samples": invalid_samples,
        "exact_labels": [],
        "explained_variants": [],
        "families": [],
        "provisional_accepted_spans": [],
    }

    if valid_count < aggregation["min_valid_samples"]:
        existence["outcome"] = "abstained"
        consensus = _base_consensus(
            idx, sentence, "abstained", None, ["insufficient_valid_samples"], existence
        )
        return consensus, audit, _uncertainty_record(idx, sentence, valid_samples, "abstained")
    if no_votes >= aggregation["min_no_skill_votes"]:
        existence["outcome"] = "negative"
        consensus = _base_consensus(
            idx, sentence, "negative", 0, ["high_confidence_no_skill"], existence
        )
        return consensus, audit, _uncertainty_record(idx, sentence, valid_samples, "negative")
    if has_votes < aggregation["min_has_skill_votes"]:
        existence["outcome"] = "unsolved"
        consensus = _base_consensus(
            idx, sentence, "unsolved", None, ["has_skill_no_consensus"], existence
        )
        return consensus, audit, _uncertainty_record(idx, sentence, valid_samples, "unsolved")

    existence["outcome"] = "positive"
    labels = _label_evidence(valid_samples)
    exact_accepted = [
        label for label in labels
        if label["exact_votes"] >= aggregation["min_exact_accept_votes"]
    ]
    accepted: list[dict[str, Any]] = [
        {**_family_audit_member(label), "accepted_by": "exact_vote"}
        for label in exact_accepted
    ]
    audit["exact_labels"] = [
        {
            **_family_audit_member(label),
            "is_candidate": label["exact_votes"] >= aggregation["min_exact_candidate_votes"],
            "is_accepted": label in exact_accepted,
        }
        for label in labels
    ]

    residual: list[dict[str, Any]] = []
    for label in labels:
        if label in exact_accepted:
            continue
        explained_by = [
            accepted_label["label_id"]
            for accepted_label in exact_accepted
            if positive_overlap(label, accepted_label)
        ]
        if explained_by:
            audit["explained_variants"].append(
                {**_family_audit_member(label), "explained_by_label_ids": explained_by}
            )
        else:
            residual.append(label)

    unresolved: list[dict[str, Any]] = []
    for family_index, members in enumerate(_overlap_components(residual)):
        family_id = f"family:{idx}:{family_index}"
        sample_indexes = sorted(
            {sample_index for member in members for sample_index in member["sample_indexes"]}
        )
        support = len(sample_indexes)
        family: dict[str, Any] = {
            "family_id": family_id,
            "support": support,
            "sample_indexes": sample_indexes,
            "members": [_family_audit_member(member) for member in members],
            "outcome": "audit_noise",
            "winner_label_id": None,
            "hard_match": None,
        }
        if support >= aggregation["hard_match_family_votes"]:
            result = hard_match_family(
                family["members"],
                sample_indexes=sample_indexes,
                valid_sample_count=valid_count,
                exact_accept_votes=aggregation["min_exact_accept_votes"],
                min_family_support=aggregation["hard_match_family_votes"],
                config=aggregation["hard_match"],
            )
            family["hard_match"] = result
            if result["accepted"]:
                winner = next(
                    member for member in members
                    if member["label_id"] == result["winner_label_id"]
                )
                accepted.append(
                    {**_family_audit_member(winner), "accepted_by": "family_hard_match"}
                )
                family["outcome"] = "accepted"
                family["winner_label_id"] = winner["label_id"]
            else:
                family["outcome"] = "unresolved"
                family["reason"] = result["reason"]
                unresolved.append(family)
        elif support >= aggregation["min_family_candidate_votes"]:
            family["outcome"] = "unresolved"
            family["reason"] = "unresolved_span_family"
            unresolved.append(family)
        audit["families"].append(family)

    accepted.sort(key=lambda item: (item["start"], item["end"]))
    audit["provisional_accepted_spans"] = accepted
    conflicts = [
        [left["label_id"], right["label_id"]]
        for left, right in combinations(accepted, 2)
        if positive_overlap(left, right)
    ]
    audit["accepted_span_conflicts"] = conflicts

    reasons: list[str] = []
    if unresolved:
        reasons.extend(
            sorted(
                {
                    family.get("reason", "unresolved_span_family")
                    for family in unresolved
                }
            )
        )
    if conflicts:
        reasons.append("accepted_span_conflict")
    if not accepted:
        reasons.append("positive_without_stable_span")
    if reasons:
        consensus = _base_consensus(idx, sentence, "unsolved", None, reasons, existence)
        consensus["candidate_spans"] = [
            _family_audit_member(label)
            for label in labels
            if label["exact_votes"] >= aggregation["min_exact_candidate_votes"]
        ]
        consensus["unresolved_families"] = [
            {
                "family_id": family["family_id"],
                "support": family["support"],
                "reason": family.get("reason", "unresolved_span_family"),
            }
            for family in unresolved
        ]
    else:
        consensus = _base_consensus(
            idx, sentence, "accepted", 1, ["all_span_families_resolved"], existence
        )
        consensus["spans"] = [item["text"] for item in accepted]
        consensus["accepted_span_details"] = accepted
        consensus["candidate_spans"] = [
            _family_audit_member(label)
            for label in labels
            if label["exact_votes"] >= aggregation["min_exact_candidate_votes"]
        ]
    uncertainty = _uncertainty_record(idx, sentence, valid_samples, consensus["status"])
    return consensus, audit, uncertainty


def aggregate_records(
    records: list[dict[str, Any]], aggregation: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    consensus_records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    uncertainty_records: list[dict[str, Any]] = []
    for idx, sentence, samples in _group_samples(records):
        consensus, audit, uncertainty = aggregate_sentence(
            idx, sentence, samples, aggregation
        )
        consensus_records.append(consensus)
        audit_records.append(audit)
        uncertainty_records.append(uncertainty)
    return consensus_records, audit_records, uncertainty_records


def build_summary(consensus_records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "updated_at": utc_now(),
        "consensus_schema": CONSENSUS_SCHEMA,
        "records": len(consensus_records),
        "statuses": {
            status: sum(record["status"] == status for record in consensus_records)
            for status in DECISION_STATUSES
        },
        "accepted_spans": sum(
            len(record["accepted_span_details"])
            for record in consensus_records
            if record["status"] == "accepted"
        ),
        "acceptance_sources": {
            source: sum(
                detail["accepted_by"] == source
                for record in consensus_records
                for detail in record["accepted_span_details"]
            )
            for source in ("exact_vote", "family_hard_match")
        },
    }


def write_aggregation_outputs(
    output_dir: Path,
    consensus_records: list[dict[str, Any]],
    audit_records: list[dict[str, Any]],
    uncertainty_records: list[dict[str, Any]],
) -> None:
    atomic_write_jsonl(output_dir / "consensus.jsonl", consensus_records)
    atomic_write_jsonl(output_dir / "aggregation_audit.jsonl", audit_records)
    atomic_write_jsonl(output_dir / "uncertainty_audit.jsonl", uncertainty_records)
    atomic_write_json(output_dir / "summary.json", build_summary(consensus_records))


def _validate_active_run_schema(config: dict[str, Any], manifest: dict[str, Any]) -> None:
    compatibility = manifest.get("compatibility", {})
    if config.get("consensus_schema") != CONSENSUS_SCHEMA:
        raise ValueError(f"Active config consensus_schema must be {CONSENSUS_SCHEMA}")
    if compatibility.get("annotation_schema") != config.get("annotation_schema"):
        raise ValueError("Run annotation schema does not match the active configuration")
    if compatibility.get("consensus_schema") != config.get("consensus_schema"):
        raise ValueError("Run consensus schema does not match the active configuration")
    if compatibility.get("aggregation") != config.get("aggregation"):
        raise ValueError("Run aggregation config does not match the active configuration")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "self_annotator.json"
    )
    args = parser.parse_args()

    run_id = validate_run_id(args.run_id)
    config_path = resolve_project_path(PROJECT_ROOT, args.config)
    config = load_json(config_path)
    paths = build_self_annotation_run_paths(PROJECT_ROOT, config["output"], run_id)
    _validate_active_run_schema(config, load_json(paths.manifest))
    records = read_jsonl(paths.parsed / "samples.jsonl")
    consensus, audit, uncertainty = aggregate_records(records, config["aggregation"])
    write_aggregation_outputs(paths.aggregated, consensus, audit, uncertainty)
    print(f"aggregated {len(consensus)} sentences -> {paths.aggregated}")


if __name__ == "__main__":
    main()
