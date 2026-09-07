"""Sample skill-annotation prompts with Qwen and preserve immutable raw responses."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import (  # noqa: E402
    SelfAnnotationRunPaths,
    append_jsonl,
    atomic_write_json,
    build_self_annotation_run_paths,
    default_run_id,
    expand_environment_references,
    load_json,
    read_jsonl,
    redact_secrets,
    resolve_project_path,
    sha256_file,
    sha256_json,
    utc_now,
    validate_run_id,
)
from common.qwen_client import (  # noqa: E402
    QwenClient,
    QwenRequestError,
    QwenSettings,
)


logger = logging.getLogger("skill_sentence.ask_qwen")


def validate_prompt_records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("Prompt file root must be a JSON list")

    indexes: set[int] = set()
    versions: set[str] = set()
    records: list[dict[str, Any]] = []
    required = {"idx", "sentence", "prompt", "prompt_version"}
    for position, item in enumerate(value):
        if not isinstance(item, dict):
            raise TypeError(f"prompt[{position}] must be an object")
        missing = required - set(item)
        if missing:
            raise ValueError(f"prompt[{position}] is missing fields: {sorted(missing)}")

        idx = item["idx"]
        sentence = item["sentence"]
        prompt = item["prompt"]
        prompt_version = item["prompt_version"]
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise TypeError(f"prompt[{position}].idx must be an integer")
        if idx in indexes:
            raise ValueError(f"Duplicate prompt idx: {idx}")
        if not isinstance(sentence, str) or not sentence.strip():
            raise ValueError(f"prompt[{position}].sentence must be non-empty")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"prompt[{position}].prompt must be non-empty")
        if not isinstance(prompt_version, str) or not prompt_version.strip():
            raise ValueError(f"prompt[{position}].prompt_version must be non-empty")

        indexes.add(idx)
        versions.add(prompt_version)
        records.append(
            {
                "idx": idx,
                "sentence": sentence,
                "prompt": prompt,
                "prompt_version": prompt_version,
            }
        )

    if len(versions) > 1:
        raise ValueError(f"Prompt file contains multiple versions: {sorted(versions)}")
    return records


def validate_config(config: dict[str, Any]) -> None:
    required_fields = {
        "schema_version",
        "module",
        "version",
        "annotation_schema",
        "consensus_schema",
        "input",
        "prompt",
        "provider",
        "model",
        "generation",
        "aggregation",
        "output",
    }
    for field in required_fields:
        if field not in config:
            raise KeyError(f"Missing config field: {field}")
    unexpected_fields = set(config) - required_fields
    if unexpected_fields:
        raise ValueError(f"Unexpected config fields: {sorted(unexpected_fields)}")

    if config["schema_version"] != 1 or config["module"] != "self_annotator":
        raise ValueError("Expected a schema_version 1 self_annotator module config")

    if not isinstance(config["version"], str) or not config["version"].strip():
        raise ValueError("version must be non-empty")
    if config["annotation_schema"] != "skill-inline-span-v1":
        raise ValueError("annotation_schema must be 'skill-inline-span-v1'")
    if config.get("consensus_schema") != "span-xmlc-majority-anchor-overlap-v2":
        raise ValueError(
            "consensus_schema must be 'span-xmlc-majority-anchor-overlap-v2'"
        )
    provider = config["provider"]
    for field in ("api_key", "base_url"):
        if not isinstance(provider.get(field), str) or not provider[field].strip():
            raise ValueError(f"provider.{field} must be non-empty")

    model = config["model"]
    if not isinstance(model.get("name"), str) or not model["name"].strip():
        raise ValueError("model.name must be non-empty")

    generation = config["generation"]
    if generation.get("samples") != 5:
        raise ValueError("generation.samples must be exactly 5")
    if not isinstance(generation.get("max_tokens"), int) or generation["max_tokens"] < 1:
        raise ValueError("generation.max_tokens must be a positive integer")
    if not isinstance(generation.get("max_retries"), int) or generation["max_retries"] < 1:
        raise ValueError("generation.max_retries must be a positive integer")
    if not isinstance(generation.get("temperature"), (int, float)):
        raise TypeError("generation.temperature must be numeric")
    if generation["temperature"] < 0:
        raise ValueError("generation.temperature must be non-negative")
    if not isinstance(generation.get("timeout_seconds"), (int, float)) or generation["timeout_seconds"] <= 0:
        raise ValueError("generation.timeout_seconds must be positive")

    aggregation = config["aggregation"]
    vote_fields = (
        "min_valid_samples",
        "min_has_skill_votes",
        "min_no_skill_votes",
        "min_exact_candidate_votes",
        "min_exact_accept_votes",
        "min_family_candidate_votes",
        "hard_match_family_votes",
    )
    for field in vote_fields:
        if not isinstance(aggregation.get(field), int) or aggregation[field] < 1:
            raise ValueError(f"aggregation.{field} must be a positive integer")
        if aggregation[field] > generation["samples"]:
            raise ValueError(f"aggregation.{field} cannot exceed generation.samples")
    if aggregation["min_exact_candidate_votes"] > aggregation["min_exact_accept_votes"]:
        raise ValueError(
            "aggregation.min_exact_candidate_votes cannot exceed min_exact_accept_votes"
        )
    majority_fields = {
        "min_valid_samples": 3,
        "min_has_skill_votes": 3,
        "min_exact_candidate_votes": 3,
        "min_exact_accept_votes": 3,
        "min_family_candidate_votes": 3,
        "hard_match_family_votes": 3,
    }
    for field, expected in majority_fields.items():
        if aggregation[field] != expected:
            raise ValueError(f"aggregation.{field} must be exactly {expected}")
    if aggregation["min_no_skill_votes"] != 4:
        raise ValueError("aggregation.min_no_skill_votes must be exactly 4")

    hard_match = aggregation.get("hard_match")
    if not isinstance(hard_match, dict):
        raise TypeError("aggregation.hard_match must be an object")
    expected_hard_match_fields = {
        "overlap_metric",
        "length_unit",
        "threshold_base",
        "threshold_step",
        "threshold_cap",
        "comparison_epsilon",
    }
    if set(hard_match) != expected_hard_match_fields:
        raise ValueError(
            "aggregation.hard_match fields must be exactly "
            f"{sorted(expected_hard_match_fields)}"
        )
    if hard_match.get("overlap_metric") != "anchor_character_coverage":
        raise ValueError(
            "aggregation.hard_match.overlap_metric must be "
            "'anchor_character_coverage'"
        )
    if hard_match.get("length_unit") != "word_units":
        raise ValueError("aggregation.hard_match.length_unit must be 'word_units'")
    for field in ("threshold_base", "threshold_step", "threshold_cap"):
        value = hard_match.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"aggregation.hard_match.{field} must be in [0, 1]")
    if hard_match["threshold_base"] > hard_match["threshold_cap"]:
        raise ValueError("hard-match threshold_base cannot exceed threshold_cap")
    epsilon = hard_match.get("comparison_epsilon")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or epsilon <= 0:
        raise ValueError("aggregation.hard_match.comparison_epsilon must be positive")

    output = config["output"]
    if not isinstance(output.get("runs_root"), str) or not output["runs_root"].strip():
        raise ValueError("output.runs_root must be non-empty")


def load_runtime_config(path: Path) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise TypeError("Qwen config root must be a JSON object")
    expanded = expand_environment_references(value)
    validate_config(expanded)
    return expanded


def build_client(config: dict[str, Any], *, client: Any | None = None) -> QwenClient:
    provider = config["provider"]
    model = config["model"]
    generation = config["generation"]
    settings = QwenSettings(
        chat_model=model["name"],
        base_url=provider["base_url"],
        enable_thinking=bool(model.get("enable_thinking", False)),
        structured_output=bool(model.get("structured_output", True)),
        timeout_seconds=float(generation["timeout_seconds"]),
        max_retries=int(generation["max_retries"]),
    )
    return QwenClient(api_key=provider["api_key"], settings=settings, client=client)


def build_run_paths(config: dict[str, Any], run_id: str) -> SelfAnnotationRunPaths:
    return build_self_annotation_run_paths(PROJECT_ROOT, config["output"], run_id)


def compatibility_payload(
    *,
    input_path: Path,
    source_input_path: Path | None = None,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    limit: int | None,
) -> dict[str, Any]:
    prompt_versions = sorted({record["prompt_version"] for record in records})
    source_path = (source_input_path or input_path).resolve()
    input_sha256 = sha256_file(source_path)
    return {
        "config_version": config["version"],
        "annotation_schema": config["annotation_schema"],
        "consensus_schema": config["consensus_schema"],
        "input_path": str(source_path),
        "input_sha256": input_sha256,
        "prompt_input_path": str(input_path.resolve()),
        "prompt_input_sha256": sha256_file(input_path),
        "dataset_id": f"sentence-dataset-{input_sha256[:16]}",
        "selected_count": len(records),
        "selected_indexes_sha256": sha256_json([record["idx"] for record in records]),
        "prompt_versions": prompt_versions,
        "limit": limit,
        "model": {
            "name": config["model"]["name"],
            "enable_thinking": bool(config["model"].get("enable_thinking", False)),
            "structured_output": bool(config["model"].get("structured_output", True)),
        },
        "generation": config["generation"],
        "aggregation": config["aggregation"],
    }


def validate_source_alignment(
    source_path: Path,
    prompt_records: list[dict[str, Any]],
    limit: int | None,
) -> None:
    value = load_json(source_path)
    if not isinstance(value, list):
        raise TypeError("Source sentence file root must be a JSON list")
    source = value[:limit] if limit is not None else value
    expected = [(record["idx"], record["sentence"]) for record in prompt_records]
    actual = [
        (record.get("idx"), record.get("sentence"))
        if isinstance(record, dict)
        else (None, None)
        for record in source
    ]
    if actual != expected:
        raise ValueError("Source sentences do not align with the prompt snapshot")


def create_or_validate_manifest(
    paths: SelfAnnotationRunPaths,
    *,
    run_id: str,
    compatibility: dict[str, Any],
    base_url: str,
    resume: bool,
) -> dict[str, Any]:
    if resume:
        if not paths.manifest.is_file():
            raise FileNotFoundError(f"Cannot resume without manifest: {paths.manifest}")
        manifest = load_json(paths.manifest)
        if manifest.get("compatibility") != compatibility:
            raise ValueError("Resume configuration does not match the existing run manifest")
        manifest["status"] = "running"
        manifest["resumed_at"] = utc_now()
        atomic_write_json(paths.manifest, manifest)
        return manifest

    if paths.root.exists():
        unexpected = [path for path in paths.root.iterdir() if path != paths.prompts]
        if unexpected:
            raise FileExistsError(
                f"Run directory already contains files: {paths.root}. "
                "Use --resume or another run-id."
            )
    paths.root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "created_at": utc_now(),
        "provider": {
            "name": "aliyun-model-studio",
            "base_url": base_url,
            "api_key": "<configured>",
        },
        "compatibility": compatibility,
    }
    atomic_write_json(paths.manifest, manifest)
    return manifest


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)


def close_logging() -> None:
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()


def generate_response_per_query(
    client: QwenClient,
    query: dict[str, Any],
    generation: dict[str, Any],
    *,
    sample_index: int = 0,
) -> dict[str, Any]:
    try:
        response = client.chat(
            [{"role": "user", "content": query["prompt"]}],
            temperature=float(generation["temperature"]),
            max_tokens=int(generation["max_tokens"]),
        )
        return {
            "sample_index": sample_index,
            "status": "ok",
            **response,
            "error": None,
        }
    except QwenRequestError as error:
        return {
            "sample_index": sample_index,
            "status": "failed",
            "content": None,
            "finish_reason": None,
            "latency_ms": None,
            "attempts": error.attempts,
            "response_id": None,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "error": {
                "type": error.__class__.__name__,
                "message": str(error),
                "retriable": error.retriable,
            },
        }


def generate_responses_per_query_multiquery(
    client: QwenClient,
    query: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    samples = [
        generate_response_per_query(client, query, generation, sample_index=sample_index)
        for sample_index in range(int(generation["samples"]))
    ]
    success_count = sum(sample["status"] == "ok" for sample in samples)
    if success_count == len(samples):
        status = "complete"
    elif success_count == 0:
        status = "failed"
    else:
        status = "partial_failed"
    return {
        "idx": query["idx"],
        "sentence": query["sentence"],
        "prompt": query["prompt"],
        "prompt_version": query["prompt_version"],
        "model": client.settings.chat_model,
        "status": status,
        "created_at": utc_now(),
        "samples": samples,
    }


def load_latest_query_records(path: Path) -> dict[int, dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    for record in read_jsonl(path):
        idx = record.get("idx")
        if isinstance(idx, int) and not isinstance(idx, bool):
            latest[idx] = record
    return latest


def summarize_responses(path: Path, expected_count: int) -> dict[str, Any]:
    latest = load_latest_query_records(path)
    statuses: dict[str, int] = {}
    total_calls = 0
    total_tokens = 0
    for record in latest.values():
        status = str(record.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
        for sample in record.get("samples", []):
            total_calls += 1
            total_tokens += int(sample.get("usage", {}).get("total_tokens", 0) or 0)
    return {
        "updated_at": utc_now(),
        "expected_records": expected_count,
        "latest_records": len(latest),
        "statuses": statuses,
        "total_calls": total_calls,
        "total_tokens": total_tokens,
    }


def generate_responses_batch(
    client: QwenClient,
    records: list[dict[str, Any]],
    generation: dict[str, Any],
    response_path: Path,
    *,
    resume: bool,
    fail_fast: bool,
) -> dict[str, Any]:
    latest = load_latest_query_records(response_path) if resume else {}
    completed = {idx for idx, record in latest.items() if record.get("status") == "complete"}
    logger.info("Starting %s records; %s already complete", len(records), len(completed))

    for position, query in enumerate(records, start=1):
        if query["idx"] in completed:
            continue
        logger.info("Query %s/%s idx=%s", position, len(records), query["idx"])
        response_record = generate_responses_per_query_multiquery(client, query, generation)
        append_jsonl(response_path, response_record)
        if fail_fast and response_record["status"] != "complete":
            raise RuntimeError(f"Qwen sampling failed for idx={query['idx']}")
    return summarize_responses(response_path, len(records))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "self_annotator.json",
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--source-input", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    config_path = resolve_project_path(PROJECT_ROOT, args.config)
    config = load_runtime_config(config_path)
    input_path = (
        resolve_project_path(PROJECT_ROOT, args.input)
        if args.input is not None
        else resolve_project_path(PROJECT_ROOT, config["input"])
    )
    input_path = input_path.resolve()
    records = validate_prompt_records(load_json(input_path))
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")
        records = records[: args.limit]
    source_input_path = (
        resolve_project_path(PROJECT_ROOT, args.source_input).resolve()
        if args.source_input is not None
        else input_path
    )
    validate_source_alignment(source_input_path, records, args.limit)

    run_id = validate_run_id(args.run_id or default_run_id())
    paths = build_run_paths(config, run_id)
    compatibility = compatibility_payload(
        input_path=input_path,
        source_input_path=source_input_path,
        records=records,
        config=config,
        limit=args.limit,
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry-run",
                    "run_id": run_id,
                    "records": len(records),
                    "model": config["model"]["name"],
                    "samples": config["generation"]["samples"],
                    "input_sha256": compatibility["input_sha256"],
                    "response_path": str(paths.raw / "responses.jsonl"),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    manifest = create_or_validate_manifest(
        paths,
        run_id=run_id,
        compatibility=compatibility,
        base_url=config["provider"]["base_url"],
        resume=args.resume,
    )
    configure_logging(paths.log)
    client = build_client(config)
    try:
        summary = generate_responses_batch(
            client,
            records,
            config["generation"],
            paths.raw / "responses.jsonl",
            resume=args.resume,
            fail_fast=args.fail_fast,
        )
        atomic_write_json(paths.raw / "summary.json", summary)
        has_incomplete = any(
            summary["statuses"].get(status, 0) > 0
            for status in ("partial_failed", "failed", "unknown")
        )
        manifest["status"] = "completed_with_errors" if has_incomplete else "completed"
        manifest["completed_at"] = utc_now()
        manifest["summary"] = summary
        atomic_write_json(paths.manifest, manifest)
        logger.info("Finished raw responses: %s", paths.raw / "responses.jsonl")
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failed_at"] = utc_now()
        safe_message = redact_secrets(str(error), [config["provider"]["api_key"]])
        manifest["error"] = {
            "type": error.__class__.__name__,
            "message": safe_message,
        }
        atomic_write_json(paths.manifest, manifest)
        raise


if __name__ == "__main__":
    main()
