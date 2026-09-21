"""Offline gold evaluation and paired comparison for aggregator ablations."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from aggregator.common import AggregatorError, aggregator_run_paths, load_aggregator_config, load_records  # noqa: E402
from common.io_utils import atomic_write_json, load_json, read_jsonl, resolve_project_path  # noqa: E402


def _gold_records(path: Path) -> list[dict[str, Any]]:
    records = load_records(path)
    output: list[dict[str, Any]] = []
    seen: set[int] = set()
    for position, record in enumerate(records):
        idx, sentence, texts = record.get("idx"), record.get("sentence"), record.get("spans")
        if not isinstance(idx, int) or isinstance(idx, bool) or idx in seen:
            raise AggregatorError(f"gold[{position}].idx is invalid or duplicated")
        seen.add(idx)
        if not isinstance(sentence, str) or not isinstance(texts, list):
            raise AggregatorError(f"gold[{position}] has invalid sentence or spans")
        spans = []
        for text in texts:
            if not isinstance(text, str) or sentence.count(text) != 1:
                raise AggregatorError(
                    f"Gold span must occur exactly once for idx={idx}: {text!r}"
                )
            start = sentence.index(text)
            spans.append({"text": text, "start": start, "end": start + len(text)})
        spans.sort(key=lambda item: (item["start"], item["end"]))
        if record.get("has_skill") != (1 if spans else 0):
            raise AggregatorError(f"gold[{position}].has_skill disagrees with spans")
        output.append({"idx": idx, "sentence": sentence, "spans": spans})
    return output


def _span_set(record: dict[str, Any]) -> set[tuple[int, int, str]]:
    sentence = record.get("sentence")
    spans = record.get("spans")
    if not isinstance(sentence, str) or not isinstance(spans, list):
        raise AggregatorError("Evaluation record has invalid sentence or spans")
    result: set[tuple[int, int, str]] = set()
    ordered: list[tuple[int, int, str]] = []
    for item in spans:
        if not isinstance(item, dict):
            raise AggregatorError("Evaluation span must be an object")
        start, end, text = item.get("start"), item.get("end"), item.get("text")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not isinstance(text, str)
            or not 0 <= start < end <= len(sentence)
            or sentence[start:end] != text
        ):
            raise AggregatorError("Evaluation span is not exact end-exclusive source text")
        value = (start, end, text)
        result.add(value)
        ordered.append(value)
    if len(result) != len(spans):
        raise AggregatorError("Evaluation spans contain duplicates")
    if ordered != sorted(ordered):
        raise AggregatorError("Evaluation spans are not in source order")
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
        raise AggregatorError("Evaluation spans overlap")
    return result


def _prediction_index(
    records: list[dict[str, Any]], expected_branch: str | None = None
) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    seen: set[int] = set()
    for position, record in enumerate(records):
        idx = record.get("idx")
        schema = record.get("schema_version")
        if schema not in {"skill-prediction-v1", "skill-prediction-result-v1"}:
            raise AggregatorError(f"prediction[{position}] has an invalid schema")
        if expected_branch is not None and record.get("branch") != expected_branch:
            raise AggregatorError(f"prediction[{position}] has the wrong branch")
        if not isinstance(idx, int) or isinstance(idx, bool) or idx in seen:
            raise AggregatorError(f"prediction[{position}].idx is invalid or duplicated")
        seen.add(idx)
        if schema == "skill-prediction-result-v1":
            if record.get("outcome") not in {"exact", "recovered"}:
                continue
            prediction = record.get("prediction")
            if not isinstance(prediction, dict):
                raise AggregatorError(f"prediction result[{position}] lacks a formal prediction")
            record = prediction
        _span_set(record)
        output[idx] = record
    return output


def _metrics(
    gold: list[dict[str, Any]], predictions: dict[int, dict[str, Any]], indexes: list[int]
) -> dict[str, Any]:
    gold_by_idx = {item["idx"]: item for item in gold}
    tp = fp = fn = exact = presence_tp = presence_fp = presence_fn = negative_fp = 0
    predicted = 0
    for idx in indexes:
        truth = gold_by_idx[idx]
        prediction = predictions.get(idx)
        if prediction is None:
            predicted_spans: set[tuple[int, int, str]] = set()
        else:
            if prediction.get("sentence") != truth["sentence"]:
                raise AggregatorError(f"Prediction sentence mismatch for idx={idx}")
            predicted += 1
            predicted_spans = _span_set(prediction)
        gold_spans = _span_set(truth)
        tp += len(predicted_spans & gold_spans)
        fp += len(predicted_spans - gold_spans)
        fn += len(gold_spans - predicted_spans)
        exact += prediction is not None and predicted_spans == gold_spans
        gold_positive, predicted_positive = bool(gold_spans), bool(predicted_spans)
        presence_tp += gold_positive and predicted_positive
        presence_fp += not gold_positive and predicted_positive
        presence_fn += gold_positive and not predicted_positive
        negative_fp += not gold_positive and predicted_positive

    def safe(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    precision = safe(tp, tp + fp)
    recall = safe(tp, tp + fn)
    presence_precision = safe(presence_tp, presence_tp + presence_fp)
    presence_recall = safe(presence_tp, presence_tp + presence_fn)
    negative_count = sum(not _span_set(gold_by_idx[idx]) for idx in indexes)
    return {
        "target_count": len(indexes),
        "prediction_count": predicted,
        "coverage": safe(predicted, len(indexes)),
        "exact_span": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": safe(2 * precision * recall, precision + recall),
        },
        "sentence_exact_set_match": safe(exact, len(indexes)),
        "skill_presence": {
            "precision": presence_precision,
            "recall": presence_recall,
            "f1": safe(
                2 * presence_precision * presence_recall,
                presence_precision + presence_recall,
            ),
        },
        "negative_false_positive_rate": safe(negative_fp, negative_count),
    }


def _run_bundle(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    paths = aggregator_run_paths(config, run_id)
    manifest = load_json(paths.manifest)
    if manifest.get("status") != "completed":
        raise AggregatorError(f"Evaluation requires a completed run: {run_id}")
    predictions = read_jsonl(paths.prediction)
    results = read_jsonl(paths.results)
    validation_issues = read_jsonl(paths.validation_issues)
    raw = {item["idx"]: item for item in read_jsonl(paths.raw)}
    failures = read_jsonl(paths.failures)
    return {
        "manifest": manifest,
        "inputs": {item["idx"]: item for item in read_jsonl(paths.inputs)},
        "predictions": _prediction_index(predictions),
        "operational": {
            "coverage": len(predictions) / manifest["compatibility"]["targets"]["record_count"],
            "repair_rate": (
                sum(bool(item.get("repair")) for item in raw.values()) / len(raw)
                if raw else 0.0
            ),
            "failure_rate": len(failures) / manifest["compatibility"]["targets"]["record_count"],
            "validation_issue_rate": (
                len(validation_issues)
                / manifest["compatibility"]["targets"]["record_count"]
            ),
            "provisional_rate": (
                sum(item.get("outcome") == "provisional" for item in results)
                / manifest["compatibility"]["targets"]["record_count"]
            ),
            "result_coverage": (
                len(results) / manifest["compatibility"]["targets"]["record_count"]
            ),
        },
    }


def _bootstrap(
    gold: list[dict[str, Any]],
    evidence: dict[int, dict[str, Any]],
    experts: dict[int, dict[str, Any]],
    indexes: list[int],
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = random.Random(seed)
    deltas: dict[str, list[float]] = {"exact_span_f1": [], "sentence_exact_set_match": []}
    for _ in range(samples):
        selected = [rng.choice(indexes) for _ in indexes]
        left = _metrics(gold, evidence, selected)
        right = _metrics(gold, experts, selected)
        deltas["exact_span_f1"].append(right["exact_span"]["f1"] - left["exact_span"]["f1"])
        deltas["sentence_exact_set_match"].append(
            right["sentence_exact_set_match"] - left["sentence_exact_set_match"]
        )
    result: dict[str, list[float]] = {}
    for name, values in deltas.items():
        values.sort()
        low = values[int(0.025 * (samples - 1))]
        high = values[int(0.975 * (samples - 1))]
        result[name] = [low, high]
    return result


def evaluate(
    config_path: Path,
    evidence_run_id: str,
    expert_run_id: str,
    gold_path: Path,
    trf_predictions_path: Path,
    exemplar_predictions_path: Path,
    *,
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    config = load_aggregator_config(config_path)
    from aggregator.validator import validate as validate_run

    validate_run(config_path, evidence_run_id)
    validate_run(config_path, expert_run_id)
    gold = _gold_records(gold_path)
    indexes = [item["idx"] for item in gold]
    evidence = _run_bundle(config, evidence_run_id)
    experts = _run_bundle(config, expert_run_id)
    if evidence["manifest"].get("mode") != "evidence_only":
        raise AggregatorError("The evidence run has the wrong mode")
    if experts["manifest"].get("mode") != "with_expert_results":
        raise AggregatorError("The expert-aware run has the wrong mode")
    for name, bundle in (("evidence_only", evidence), ("with_expert_results", experts)):
        for item in gold:
            run_input = bundle["inputs"].get(item["idx"])
            if run_input is None or run_input.get("target_sentence") != item["sentence"]:
                raise AggregatorError(f"{name} input does not align with gold idx={item['idx']}")
    evidence_metrics = _metrics(gold, evidence["predictions"], indexes)
    expert_metrics = _metrics(gold, experts["predictions"], indexes)
    both_available = [
        item["idx"]
        for item in read_jsonl(aggregator_run_paths(config, expert_run_id).inputs)
        if all(value["available"] for value in item["expert_proposals"].values())
        and item["idx"] in set(indexes)
    ]
    delta_f1 = expert_metrics["exact_span"]["f1"] - evidence_metrics["exact_span"]["f1"]
    delta_exact = (
        expert_metrics["sentence_exact_set_match"]
        - evidence_metrics["sentence_exact_set_match"]
    )
    negative_delta = (
        expert_metrics["negative_false_positive_rate"]
        - evidence_metrics["negative_false_positive_rate"]
    )
    per_sentence_changes = []
    gold_by_idx = {item["idx"]: item for item in gold}
    for idx in indexes:
        left = evidence["predictions"].get(idx)
        right = experts["predictions"].get(idx)
        left_spans = _span_set(left) if left else set()
        right_spans = _span_set(right) if right else set()
        truth = _span_set(gold_by_idx[idx])
        per_sentence_changes.append(
            {
                "idx": idx,
                "changed": left_spans != right_spans,
                "evidence_only_exact": left_spans == truth,
                "with_expert_results_exact": right_spans == truth,
                "evidence_only_spans": sorted(left_spans),
                "with_expert_results_spans": sorted(right_spans),
            }
        )
    result: dict[str, Any] = {
        "schema_version": "aggregator-evaluation-v1",
        "gold": {"path": str(gold_path.resolve()), "target_count": len(gold)},
        "evidence_only": {"metrics": evidence_metrics, "operational": evidence["operational"]},
        "with_expert_results": {
            "metrics": expert_metrics,
            "both_experts_available_subset": _metrics(
                gold, experts["predictions"], both_available
            ),
            "operational": experts["operational"],
        },
        "paired_comparison": {
            "exact_span_f1_delta": delta_f1,
            "sentence_exact_set_match_delta": delta_exact,
            "negative_false_positive_rate_delta": negative_delta,
            "paired_bootstrap_95_percent_ci": _bootstrap(
                gold,
                evidence["predictions"],
                experts["predictions"],
                indexes,
                samples=bootstrap_samples,
                seed=seed,
            ),
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
            "both_experts_available_subset": {
                "evidence_only": _metrics(gold, evidence["predictions"], both_available),
                "with_expert_results": _metrics(gold, experts["predictions"], both_available),
            },
            "per_sentence_changes": per_sentence_changes,
            "positive_contribution_criterion_met": (
                (delta_f1 > 0 and delta_exact >= 0)
                or (delta_exact > 0 and delta_f1 >= 0)
            )
            and negative_delta <= 0,
        },
        "single_expert_baselines": {},
    }
    for name, path in (("trf", trf_predictions_path), ("exemplar", exemplar_predictions_path)):
        records = load_records(path.resolve())
        result["single_expert_baselines"][name] = _metrics(
            gold, _prediction_index(records, expected_branch=name), indexes
        )
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/aggregator.json")
    parser.add_argument("--evidence-run-id", required=True)
    parser.add_argument("--expert-run-id", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--trf-predictions", required=True)
    parser.add_argument("--exemplar-predictions", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        if args.bootstrap_samples < 1:
            raise AggregatorError("--bootstrap-samples must be positive")
        result = evaluate(
            resolve_project_path(PROJECT_ROOT, args.config).resolve(),
            args.evidence_run_id,
            args.expert_run_id,
            resolve_project_path(PROJECT_ROOT, args.gold).resolve(),
            resolve_project_path(PROJECT_ROOT, args.trf_predictions).resolve(),
            resolve_project_path(PROJECT_ROOT, args.exemplar_predictions).resolve(),
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        if args.output:
            atomic_write_json(resolve_project_path(PROJECT_ROOT, args.output).resolve(), result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        print(f"EvaluateAggregator failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
