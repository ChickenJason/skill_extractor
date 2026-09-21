"""Strict gate-spec loading and deterministic selection from the frozen R0 base."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from common.io_utils import load_json
from second_layer.collaborative_reflection.common import ReflectionError
from second_layer.collaborative_reflection_json_schema.common import PROTOCOL_VERSION
from second_layer.collaborative_reflection_json_schema.pipeline import build_reflection_inputs


SPEC_VERSION = "collaborative-reflection-stratified-gate-spec-v1"
STRATA = {
    "nonempty_baseline_trf",
    "trf_needs_review",
    "exemplar_needs_review",
    "candidate_positive_only",
    "candidate_negative_heavy",
    "candidate_mixed",
    "baseline_zero_selected",
    "baseline_one_selected",
    "empty_baseline_trf_control",
}
TARGET_KEYS = {
    "dataset_id", "record_id", "idx", "sentence", "source_sha256", "strata",
    "trf_anchors", "anchor_rationale",
}
ACCEPTANCE_KEYS = {
    "target_count", "minimum_strata_counts", "required_keep_anchors",
    "required_drop_anchors", "required_wrapper_repairs", "maximum_total_repairs",
    "maximum_model_calls", "required_failed_records",
}


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if type(value) is not int or value < minimum:
        raise ReflectionError(f"{label} must be an integer >= {minimum}")
    return value


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ReflectionError(f"{label} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ReflectionError(f"{label} contains duplicates")
    return value


def load_gate_spec(path: Path) -> dict[str, Any]:
    spec = load_json(path)
    expected_top = {
        "schema_version", "gate_id", "base_concurrent_run_id", "underlying_protocol",
        "maximum_targets", "targets", "acceptance",
    }
    if not isinstance(spec, dict) or set(spec) != expected_top:
        raise ReflectionError("Stratified gate spec has missing or unknown top-level fields")
    if spec["schema_version"] != SPEC_VERSION:
        raise ReflectionError("Unsupported stratified gate spec version")
    if spec["underlying_protocol"] != PROTOCOL_VERSION:
        raise ReflectionError("Stratified gate must use the immutable JSON-Schema v2 protocol")
    for key in ("gate_id", "base_concurrent_run_id"):
        if not isinstance(spec[key], str) or not spec[key]:
            raise ReflectionError(f"Gate spec {key} must be a non-empty string")
    maximum = _positive_int(spec["maximum_targets"], "maximum_targets")
    if maximum > 16:
        raise ReflectionError("Stratified small-sample gates cannot allow more than 16 targets")
    targets = spec["targets"]
    if not isinstance(targets, list) or not targets or len(targets) > maximum:
        raise ReflectionError("Gate target list is empty or exceeds maximum_targets")
    indexes: list[int] = []
    identities: set[tuple[str, str]] = set()
    keep_count = 0
    drop_count = 0
    for position, target in enumerate(targets, start=1):
        if not isinstance(target, dict) or set(target) != TARGET_KEYS:
            raise ReflectionError(f"Gate target {position} has missing or unknown fields")
        idx = _positive_int(target["idx"], f"targets[{position}].idx")
        indexes.append(idx)
        for key in ("dataset_id", "record_id", "sentence", "source_sha256", "anchor_rationale"):
            if not isinstance(target[key], str) or not target[key]:
                raise ReflectionError(f"targets[{position}].{key} must be non-empty")
        identity = (target["dataset_id"], target["record_id"])
        if identity in identities:
            raise ReflectionError(f"Duplicate gate identity: {identity!r}")
        identities.add(identity)
        strata = _string_list(target["strata"], f"targets[{position}].strata")
        if set(strata) - STRATA:
            raise ReflectionError(f"targets[{position}].strata contains an unknown stratum")
        anchors = target["trf_anchors"]
        if not isinstance(anchors, dict) or set(anchors) != {"must_keep", "must_drop"}:
            raise ReflectionError(f"targets[{position}].trf_anchors is invalid")
        keep = _string_list(anchors["must_keep"], f"targets[{position}].must_keep")
        drop = _string_list(anchors["must_drop"], f"targets[{position}].must_drop")
        if set(keep) & set(drop):
            raise ReflectionError(f"targets[{position}] has contradictory TRF anchors")
        keep_count += len(keep)
        drop_count += len(drop)
    if indexes != sorted(indexes) or len(indexes) != len(set(indexes)):
        raise ReflectionError("Gate target idx values must be unique and strictly increasing")
    acceptance = spec["acceptance"]
    if not isinstance(acceptance, dict) or set(acceptance) != ACCEPTANCE_KEYS:
        raise ReflectionError("Gate acceptance contract is invalid")
    for key in ACCEPTANCE_KEYS - {"minimum_strata_counts"}:
        _positive_int(
            acceptance[key], f"acceptance.{key}",
            allow_zero=key in {"required_wrapper_repairs", "required_failed_records"},
        )
    minimums = acceptance["minimum_strata_counts"]
    if not isinstance(minimums, dict) or set(minimums) != STRATA:
        raise ReflectionError("Gate minimum_strata_counts must cover every supported stratum")
    for name, value in minimums.items():
        _positive_int(value, f"acceptance.minimum_strata_counts.{name}")
    if acceptance["target_count"] != len(targets):
        raise ReflectionError("Gate acceptance target_count does not match targets")
    if acceptance["required_keep_anchors"] != keep_count:
        raise ReflectionError("Gate keep-anchor count contract is wrong")
    if acceptance["required_drop_anchors"] != drop_count:
        raise ReflectionError("Gate drop-anchor count contract is wrong")
    return spec


def compute_strata(base: dict[str, Any], candidates: dict[str, Any]) -> list[str]:
    result: set[str] = set()
    trfs = base["trf_context"]["trfs"]
    if trfs:
        result.add("nonempty_baseline_trf")
    else:
        result.add("empty_baseline_trf_control")
    if base["trf_context"]["status"] == "needs_review":
        result.add("trf_needs_review")
    if base["exemplar_context"]["status"] == "needs_review":
        result.add("exemplar_needs_review")
    statuses = Counter(item["status"] for item in candidates["candidates"])
    accepted = statuses["accepted"]
    negative = statuses["negative"]
    if accepted == len(candidates["candidates"]) and negative == 0:
        result.add("candidate_positive_only")
    if negative >= 8:
        result.add("candidate_negative_heavy")
    if accepted > 0 and negative > 0:
        result.add("candidate_mixed")
    selected_count = base["exemplar_context"]["selected_count"]
    if selected_count == 0:
        result.add("baseline_zero_selected")
    if selected_count == 1:
        result.add("baseline_one_selected")
    return sorted(result)


def _map_by_idx(records: list[dict[str, Any]], label: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for record in records:
        idx = record.get("idx")
        if type(idx) is not int or idx in result:
            raise ReflectionError(f"{label} has invalid or duplicate idx={idx!r}")
        result[idx] = record
    return result


def select_gate_sources(
    contexts: list[dict[str, Any]], candidates: list[dict[str, Any]],
    judgments: list[dict[str, Any]], spec: dict[str, Any], *, candidate_count: int = 16,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
    list[dict[str, Any]], dict[str, int],
]:
    context_by_idx = _map_by_idx(contexts, "base context")
    candidate_by_idx = _map_by_idx(candidates, "candidate records")
    judgment_by_idx = _map_by_idx(judgments, "judgment records")
    selected_contexts: list[dict[str, Any]] = []
    selected_candidates: list[dict[str, Any]] = []
    selected_judgments: list[dict[str, Any]] = []
    coverage: Counter[str] = Counter()
    for target in spec["targets"]:
        idx = target["idx"]
        if idx not in context_by_idx or idx not in candidate_by_idx or idx not in judgment_by_idx:
            raise ReflectionError(f"Gate target idx={idx} is missing from a base artifact")
        base = context_by_idx[idx]
        candidate = candidate_by_idx[idx]
        judgment = judgment_by_idx[idx]
        for key in ("dataset_id", "record_id", "idx", "sentence", "source_sha256"):
            if base.get(key) != target[key]:
                raise ReflectionError(f"Gate target idx={idx} mismatches base {key}")
            if candidate.get(key) != target[key] or judgment.get(key) != target[key]:
                raise ReflectionError(f"Gate target idx={idx} mismatches child {key}")
        actual_strata = compute_strata(base, candidate)
        if set(actual_strata) != set(target["strata"]):
            raise ReflectionError(
                f"Gate target idx={idx} strata mismatch: expected={target['strata']}, actual={actual_strata}"
            )
        coverage.update(actual_strata)
        original = {item["normalized_text"] for item in base["trf_context"]["trfs"]}
        anchors = set(target["trf_anchors"]["must_keep"]) | set(target["trf_anchors"]["must_drop"])
        if anchors - original:
            raise ReflectionError(f"Gate target idx={idx} anchors a non-baseline TRF")
        selected_contexts.append(base)
        selected_candidates.append(candidate)
        selected_judgments.append(judgment)
    minimums = spec["acceptance"]["minimum_strata_counts"]
    for name, minimum in minimums.items():
        if coverage[name] < minimum:
            raise ReflectionError(f"Gate stratum {name} has {coverage[name]} targets; needs {minimum}")
    inputs = build_reflection_inputs(
        selected_contexts, selected_candidates, selected_judgments,
        candidate_count=candidate_count,
    )
    return (
        selected_contexts,
        selected_candidates,
        selected_judgments,
        inputs,
        {name: coverage[name] for name in sorted(STRATA)},
    )


def identity_contract(inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dataset_id": item["dataset_id"],
            "record_id": item["record_id"],
            "source_sha256": item["source_sha256"],
            "idx": item["idx"],
        }
        for item in inputs
    ]
