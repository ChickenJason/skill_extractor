"""Pure data, retrieval, prompt, parsing, and diagnostic logic for target TRFs."""

from __future__ import annotations

import json
import math
import sys
import unicodedata
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.contracts import record_identity  # noqa: E402
from common.io_utils import load_json, read_jsonl, sha256_json  # noqa: E402
from trf.target.common import TargetTRFError, source_paths  # noqa: E402


def load_source_bundle(config: dict[str, Any]) -> dict[str, Any]:
    paths = source_paths(config)
    corpus = read_jsonl(paths["corpus"])
    pseudo = read_jsonl(paths["pseudo_trfs"])
    candidates = load_json(paths["candidates"])
    decisions = read_jsonl(paths["demonstration_records"])
    if len(decisions) != config["demonstrations"]["expected_count"]:
        raise TargetTRFError("Demonstrations must contain exactly 326 records")

    decision_indexes = [record.get("idx") for record in decisions]
    if any(
        not isinstance(idx, int) or isinstance(idx, bool) for idx in decision_indexes
    ):
        raise TargetTRFError("Every source decision idx must be an integer")
    if len(set(decision_indexes)) != len(decision_indexes):
        raise TargetTRFError("Source decision idx values must be unique")
    decision_by_idx = {record["idx"]: record for record in decisions}
    formal_indexes = {
        record["idx"]
        for record in decisions
        if record.get("status") in {"accepted", "negative"}
    }
    if any(
        record.get("status") not in {"accepted", "negative", "unsolved", "abstained"}
        for record in decisions
    ):
        raise TargetTRFError("Source decisions contain an unsupported status")

    corpus_indexes = [record.get("idx") for record in corpus]
    pseudo_indexes = [record.get("idx") for record in pseudo]
    if (
        len(set(corpus_indexes)) != len(corpus_indexes)
        or len(set(pseudo_indexes)) != len(pseudo_indexes)
    ):
        raise TargetTRFError("Source corpus and pseudo TRF indexes must be unique")
    if set(corpus_indexes) != formal_indexes or set(pseudo_indexes) != formal_indexes:
        raise TargetTRFError(
            "Source corpus and pseudo TRFs must exactly match formal decision indexes"
        )
    expected_main_trfs = int(config["candidates"]["expected_main_trfs"])
    if candidates.get("max_trfs") != expected_main_trfs:
        raise TargetTRFError(
            "Source candidate max_trfs does not match "
            "candidates.expected_main_trfs"
        )
    if len(candidates.get("trfs", [])) != expected_main_trfs:
        raise TargetTRFError(
            f"Source main candidate bank must contain exactly {expected_main_trfs} TRFs"
        )
    if len(candidates.get("context_only_trfs", [])) != expected_main_trfs:
        raise TargetTRFError(
            "Source context-only candidate bank must contain exactly "
            f"{expected_main_trfs} TRFs"
        )
    pseudo_by_idx = {record["idx"]: record for record in pseudo}
    demonstrations: list[dict[str, Any]] = []
    for record in corpus:
        decision = decision_by_idx[record["idx"]]
        if (
            record.get("status") != decision.get("status")
            or record.get("sentence") != decision.get("sentence")
        ):
            raise TargetTRFError(
                f"Corpus record differs from its formal decision for idx={record['idx']}"
            )
        pseudo_record = pseudo_by_idx.get(record["idx"])
        if pseudo_record is None or pseudo_record["status"] != record["status"]:
            raise TargetTRFError(f"Pseudo TRF record mismatch for idx={record['idx']}")
        trfs = [item["text"] for item in pseudo_record["trfs"]]
        expected_count = 5 if record["status"] == "accepted" else 0
        if len(trfs) != expected_count:
            raise TargetTRFError(f"Unexpected main pseudo TRF count for idx={record['idx']}")
        demonstrations.append(
            {
                "schema_version": "trf-demonstration-v1",
                "dataset_id": record["dataset_id"],
                "record_id": record["record_id"],
                "source_sha256": record["source_sha256"],
                "idx": record["idx"],
                "sentence": record["sentence"],
                "status": record["status"],
                "existence_score": record["existence_score"],
                "trfs": trfs,
                "skill_spans": list(decision.get("spans", [])),
                "accepted_span_details": list(
                    decision.get("accepted_span_details", [])
                ),
            }
        )
    return {
        "demonstrations": sorted(demonstrations, key=lambda item: item["idx"]),
        "decisions": sorted(decisions, key=lambda item: item["idx"]),
        "main_trfs": list(candidates["trfs"]),
        "pseudo_by_idx": pseudo_by_idx,
    }


def validate_independent_targets(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TargetTRFError("Independent target input must be a JSON array")
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for position, item in enumerate(value):
        if not isinstance(item, dict) or {"idx", "sentence"} - set(item):
            raise TargetTRFError(
                f"Target record {position} must contain idx and sentence"
            )
        idx, sentence = item["idx"], item["sentence"]
        if not isinstance(idx, int) or isinstance(idx, bool) or idx in seen:
            raise TargetTRFError(f"Target idx must be a unique integer: {idx!r}")
        if not isinstance(sentence, str) or not sentence.strip():
            raise TargetTRFError(f"Target idx={idx} has an empty sentence")
        seen.add(idx)
        records.append({"idx": idx, "sentence": sentence})
    if not records:
        raise TargetTRFError("Target input must not be empty")
    records.sort(key=lambda record: record["idx"])
    source_sha256 = sha256_json(records)
    dataset_id = f"sentence-dataset-{source_sha256[:16]}"
    return [
        {
            "schema_version": "sentence-record-v1",
            "dataset_id": dataset_id,
            "record_id": str(record["idx"]),
            "source_sha256": source_sha256,
            **record,
        }
        for record in records
    ]


def build_targets(
    mode: str,
    bundle: dict[str, Any],
    independent_value: Any | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    if mode == "independent":
        records = validate_independent_targets(independent_value)
    elif mode == "leave-one-out":
        records = [
            {
                "schema_version": "sentence-record-v1",
                "dataset_id": item["dataset_id"],
                "record_id": item["record_id"],
                "source_sha256": item["source_sha256"],
                "idx": item["idx"],
                "sentence": item["sentence"],
            }
            for item in bundle["decisions"]
        ]
    else:
        raise TargetTRFError(f"Unknown target mode: {mode}")
    if limit is not None:
        if limit < 1:
            raise TargetTRFError("Limit must be a positive integer")
        records = records[:limit]
    return records


def sentence_hash(sentence: str) -> str:
    import hashlib

    return hashlib.sha256(sentence.encode("utf-8")).hexdigest()


def validate_embedding_vector(vector: Any, dimensions: int) -> list[float]:
    if not isinstance(vector, list) or len(vector) != dimensions:
        raise TargetTRFError(f"Embedding must contain exactly {dimensions} values")
    values = [float(number) for number in vector]
    if not all(math.isfinite(number) for number in values):
        raise TargetTRFError("Embedding contains NaN or infinite values")
    if math.sqrt(sum(number * number for number in values)) == 0.0:
        raise TargetTRFError("Embedding vector has zero norm")
    return values


def retrieve_demonstrations(
    target: dict[str, Any],
    demonstrations: list[dict[str, Any]],
    target_vector: list[float],
    demo_vectors: dict[int, list[float]],
    *,
    leave_one_out: bool,
    nearest_neighbors: int,
    selected_count: int,
    similarity_decimals: int,
) -> dict[str, Any]:
    import numpy as np

    target_array = np.asarray(target_vector, dtype=np.float64)
    target_array /= np.linalg.norm(target_array)
    ranked: list[tuple[float, int, dict[str, Any]]] = []
    target_identity = record_identity(target, "target")
    for demo in demonstrations:
        if leave_one_out and record_identity(demo, "demonstration") == target_identity:
            continue
        vector = np.asarray(demo_vectors[demo["idx"]], dtype=np.float64)
        vector /= np.linalg.norm(vector)
        similarity = float(np.dot(target_array, vector))
        if not math.isfinite(similarity):
            raise TargetTRFError("Cosine similarity is not finite")
        ranked.append((similarity, demo["idx"], demo))
    if len(ranked) < nearest_neighbors:
        raise TargetTRFError(
            f"Only {len(ranked)} demonstrations are available; K={nearest_neighbors} is required"
        )
    ranked.sort(key=lambda item: (-item[0], item[1]))
    neighbors = ranked[:nearest_neighbors]
    selected = sorted(
        neighbors,
        key=lambda item: (-float(item[2]["existence_score"]), -item[0], item[1]),
    )[:selected_count]
    selected_ranks = {idx: rank for rank, (_, idx, _) in enumerate(selected, start=1)}
    neighbor_records = []
    for neighbor_rank, (similarity, idx, demo) in enumerate(neighbors, start=1):
        neighbor_records.append(
            {
                "demo_dataset_id": demo["dataset_id"],
                "demo_record_id": demo["record_id"],
                "demo_source_sha256": demo["source_sha256"],
                "demo_idx": idx,
                "neighbor_rank": neighbor_rank,
                "selected_rank": selected_ranks.get(idx),
                "selected": idx in selected_ranks,
                "similarity": round(similarity, similarity_decimals),
                "status": demo["status"],
                "existence_score": demo["existence_score"],
            }
        )
    selected_records = [
        {
            "demo_dataset_id": demo["dataset_id"],
            "demo_record_id": demo["record_id"],
            "demo_source_sha256": demo["source_sha256"],
            "demo_idx": idx,
            "selected_rank": rank,
            "similarity": round(similarity, similarity_decimals),
            "status": demo["status"],
            "existence_score": demo["existence_score"],
            "sentence": demo["sentence"],
            "trfs": list(demo["trfs"]),
            "skill_spans": list(demo["skill_spans"]),
            "accepted_span_details": list(demo["accepted_span_details"]),
        }
        for rank, (similarity, idx, demo) in enumerate(selected, start=1)
    ]
    if len(selected_records) != selected_count:
        raise TargetTRFError("Selected demonstration count is inconsistent")
    return {
        "schema_version": "candidate-instances-v1",
        "dataset_id": target["dataset_id"],
        "record_id": target["record_id"],
        "source_sha256": target["source_sha256"],
        "idx": target["idx"],
        "sentence": target["sentence"],
        "leave_one_out": leave_one_out,
        "self_excluded": leave_one_out
        and any(
            record_identity(demo, "demonstration") == target_identity
            for demo in demonstrations
        ),
        "ranking_contract": {
            "neighbor_order": "unrounded_cosine_desc_then_demo_idx_asc",
            "selected_order": (
                "existence_score_desc_then_unrounded_cosine_desc_then_demo_idx_asc"
            ),
            "class_quota": None,
        },
        "neighbors": neighbor_records,
        "selected": selected_records,
    }


def build_prompts(
    target: dict[str, Any], retrieval: dict[str, Any], max_characters: int
) -> dict[str, Any]:
    demonstrations = [
        {
            "sentence_id": item["selected_rank"],
            "sentence": item["sentence"],
            "trfs": item["trfs"],
        }
        for item in retrieval["selected"]
    ]
    stage1 = (
        'Given entity label set: ["Skill"] and the target sentence below, determine which '
        "entity types are present. Return only a JSON object with key entity_types. The value "
        'must be either ["Skill"] or [].\nTarget sentence: '
        + json.dumps(target["sentence"], ensure_ascii=False)
    )
    stage2 = (
        "Here are example sentences and corresponding type-related features (TRFs). TRFs are "
        "tokens or phrases strongly associated with Skill entities and relevant to the sentence. "
        "Using the examples and the entity-type judgment from the previous turn, identify relevant "
        "TRFs for the target sentence. TRFs may be open-vocabulary and need not occur verbatim in "
        "the target. Return only a JSON object with key trfs and a list of strings.\n"
        "Demonstrations:\n"
        + json.dumps(demonstrations, ensure_ascii=False, separators=(",", ":"))
        + "\nTarget sentence: "
        + json.dumps(target["sentence"], ensure_ascii=False)
    )
    if len(stage1) > max_characters or len(stage2) > max_characters:
        raise TargetTRFError("Generated prompt exceeds max_prompt_characters")
    return {
        "schema_version": "trf-prompt-record-v1",
        "dataset_id": target["dataset_id"],
        "record_id": target["record_id"],
        "source_sha256": target["source_sha256"],
        "idx": target["idx"],
        "sentence": target["sentence"],
        "stage1_user": stage1,
        "stage2_user": stage2,
        "stage1_sha256": sha256_json(stage1),
        "stage2_sha256": sha256_json(stage2),
        "demo_indexes": [item["demo_idx"] for item in retrieval["selected"]],
    }


def _parse_json_object(content: str, expected_key: str) -> Any:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        raise TargetTRFError(f"Invalid JSON response: {error}") from error
    if not isinstance(value, dict) or set(value) != {expected_key}:
        raise TargetTRFError(f"Response must contain exactly the key {expected_key!r}")
    return value[expected_key]


def parse_entity_types(content: str) -> list[str]:
    values = _parse_json_object(content, "entity_types")
    if values not in ([], ["Skill"]):
        raise TargetTRFError('entity_types must be exactly [] or ["Skill"]')
    return list(values)


def parse_target_trfs(
    content: str, sentence: str, main_bank: list[str]
) -> list[dict[str, Any]]:
    values = _parse_json_object(content, "trfs")
    if not isinstance(values, list):
        raise TargetTRFError("trfs must be a JSON list")
    seen: set[str] = set()
    parsed: list[dict[str, Any]] = []
    bank_casefold = {item.casefold() for item in main_bank}
    sentence_casefold = sentence.casefold()
    for raw in values:
        if not isinstance(raw, str):
            raise TargetTRFError("Every TRF must be a string")
        text = raw.strip()
        normalized = unicodedata.normalize("NFKC", text).strip()
        if not normalized:
            raise TargetTRFError("TRF strings must not be empty")
        if normalized in seen:
            continue
        seen.add(normalized)
        parsed.append(
            {
                "raw_text": raw,
                "text": text,
                "normalized_text": normalized,
                "first_seen_order": len(parsed) + 1,
                "in_main_bank_exact": normalized in main_bank,
                "in_main_bank_casefold": normalized.casefold() in bank_casefold,
                "appears_in_target_exact": normalized in sentence,
                "appears_in_target_casefold": normalized.casefold() in sentence_casefold,
            }
        )
    return parsed


def build_parsed_record(
    target: dict[str, Any],
    entity_types: list[str],
    trfs: list[dict[str, Any]],
    embedding_model: str,
    chat_model: str,
) -> dict[str, Any]:
    review_reasons = []
    if not entity_types and trfs:
        review_reasons.append("type_absent_but_trfs_nonempty")
    return {
        "schema_version": "feature-records-v1",
        "dataset_id": target["dataset_id"],
        "record_id": target["record_id"],
        "source_sha256": target["source_sha256"],
        "idx": target["idx"],
        "sentence": target["sentence"],
        "status": "needs_review" if review_reasons else "complete",
        "entity_types": entity_types,
        "trfs": trfs,
        "retrieval_count": 16,
        "review_reasons": review_reasons,
        "models": {"embedding": embedding_model, "chat": chat_model},
    }


def jaccard_and_recall(predicted: set[str], reference: set[str]) -> tuple[float, float | None]:
    union = predicted | reference
    jaccard = len(predicted & reference) / len(union) if union else 1.0
    recall = len(predicted & reference) / len(reference) if reference else None
    return jaccard, recall
