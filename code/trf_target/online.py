"""Checkpointed Qwen embedding and two-turn target TRF online operations."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import (  # noqa: E402
    MissingEnvironmentVariable,
    append_jsonl,
    atomic_write_jsonl,
    read_jsonl,
    redact_secrets,
    utc_now,
)
from common.qwen_client import QwenClient  # noqa: E402
from trf_target.common import TargetTRFError, latest_records  # noqa: E402
from trf_target.pipeline import (  # noqa: E402
    build_parsed_record,
    parse_entity_types,
    parse_target_trfs,
    sentence_hash,
    validate_embedding_vector,
)


class NetworkRequiredError(TargetTRFError):
    """Raised when missing online artifacts cannot be created without permission."""


def _append_embedding_records_atomic(
    path: Path, records: list[dict[str, Any]]
) -> None:
    if records:
        atomic_write_jsonl(path, [*read_jsonl(path), *records])


def load_embedding_records(
    path: Path,
    expected: list[dict[str, Any]],
    model: str,
    dimensions: int,
) -> dict[int, dict[str, Any]]:
    latest = latest_records(path)
    expected_by_idx = {item["idx"]: item for item in expected}
    valid: dict[int, dict[str, Any]] = {}
    for idx, record in latest.items():
        item = expected_by_idx.get(idx)
        if item is None:
            raise TargetTRFError(f"Embedding cache contains unexpected idx={idx}")
        if record.get("sentence_sha256") != sentence_hash(item["sentence"]):
            raise TargetTRFError(f"Embedding sentence hash mismatch for idx={idx}")
        if record.get("model") != model or record.get("dimensions") != dimensions:
            raise TargetTRFError(f"Embedding model contract mismatch for idx={idx}")
        record["vector"] = validate_embedding_vector(record.get("vector"), dimensions)
        valid[idx] = record
    return valid


def load_reuse_embeddings(
    reuse_root: Path | None,
    filename: str,
    expected: list[dict[str, Any]],
    model: str,
    dimensions: int,
) -> dict[int, dict[str, Any]]:
    if reuse_root is None:
        return {}
    manifest_path = reuse_root / "manifest.json"
    if not manifest_path.is_file():
        raise TargetTRFError(f"Embedding reuse run has no manifest: {reuse_root}")
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    compatible = manifest.get("compatibility", {}).get("embedding")
    if compatible != {
        "model": model,
        "dimensions": dimensions,
        "batch_size": 20,
    }:
        raise TargetTRFError("Embedding reuse run uses a different embedding contract")
    path = reuse_root / "embeddings" / filename
    return load_embedding_records(path, expected, model, dimensions) if path.is_file() else {}


def ensure_embeddings(
    items: list[dict[str, Any]],
    path: Path,
    *,
    model: str,
    dimensions: int,
    batch_size: int,
    allow_network: bool,
    client_factory: Callable[[], QwenClient] | None,
    reused: dict[int, dict[str, Any]] | None = None,
    same_run_by_sentence: dict[str, dict[str, Any]] | None = None,
    reuse_label: str | None = None,
    on_network_call: Callable[[], None] | None = None,
) -> dict[int, list[float]]:
    cached = load_embedding_records(path, items, model, dimensions)
    reused = reused or {}
    by_sentence = same_run_by_sentence or {}
    copied_records: list[dict[str, Any]] = []
    for item in items:
        idx = item["idx"]
        if idx in cached:
            continue
        source = reused.get(idx)
        if source and source["sentence_sha256"] == sentence_hash(item["sentence"]):
            record = {
                "idx": idx,
                "sentence_sha256": source["sentence_sha256"],
                "model": model,
                "dimensions": dimensions,
                "vector": source["vector"],
                "reused_from": reuse_label,
                "created_at": utc_now(),
            }
            copied_records.append(record)
            cached[idx] = record
            continue
        sentence_source = by_sentence.get(sentence_hash(item["sentence"]))
        if sentence_source:
            record = {
                "idx": idx,
                "sentence_sha256": sentence_source["sentence_sha256"],
                "model": model,
                "dimensions": dimensions,
                "vector": sentence_source["vector"],
                "reused_from": f"current-run:{sentence_source['idx']}",
                "created_at": utc_now(),
            }
            copied_records.append(record)
            cached[idx] = record
    _append_embedding_records_atomic(path, copied_records)

    missing = [item for item in items if item["idx"] not in cached]
    if missing and (not allow_network or client_factory is None):
        raise NetworkRequiredError(
            f"{len(missing)} embeddings are missing; -AllowNetwork or reusable embeddings are required"
        )
    client: QwenClient | None = None
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        if client is None:
            client = client_factory()
            if on_network_call:
                on_network_call()
        detailed = client.embeddings_detailed(
            [item["sentence"] for item in batch],
            model=model,
            dimensions=dimensions,
            batch_size=batch_size,
        )
        vectors = detailed["vectors"]
        if len(vectors) != len(batch) or len(detailed["batches"]) != 1:
            raise TargetTRFError("Embedding API batch audit is inconsistent")
        batch_audit = detailed["batches"][0]
        new_records: list[dict[str, Any]] = []
        for item, raw_vector in zip(batch, vectors):
            vector = validate_embedding_vector(raw_vector, dimensions)
            record = {
                "idx": item["idx"],
                "sentence_sha256": sentence_hash(item["sentence"]),
                "model": model,
                "dimensions": dimensions,
                "vector": vector,
                "batch_audit": batch_audit,
                "reused_from": None,
                "created_at": utc_now(),
            }
            new_records.append(record)
            cached[item["idx"]] = record
        _append_embedding_records_atomic(path, new_records)
    return {idx: record["vector"] for idx, record in cached.items()}


def _safe_error(error: Exception, secrets: list[str]) -> dict[str, str]:
    return {
        "type": error.__class__.__name__,
        "message": redact_secrets(str(error), secrets),
    }


def run_two_turn_extraction(
    targets: list[dict[str, Any]],
    prompts_by_idx: dict[int, dict[str, Any]],
    raw_path: Path,
    *,
    main_bank: list[str],
    embedding_model: str,
    chat_model: str,
    temperature: float,
    stage1_max_tokens: int,
    stage2_max_tokens: int,
    allow_network: bool,
    retry_failed: bool,
    client_factory: Callable[[], QwenClient] | None,
    secrets: list[str],
    on_network_call: Callable[[], None] | None = None,
) -> dict[int, dict[str, Any]]:
    latest = latest_records(raw_path)
    successful = {idx for idx, record in latest.items() if record.get("status") in {"complete", "needs_review"}}
    failed = {idx for idx, record in latest.items() if record.get("status") == "failed"}
    pending = [
        target
        for target in targets
        if target["idx"] not in successful
        and (target["idx"] not in failed or retry_failed)
    ]
    if pending and (not allow_network or client_factory is None):
        raise NetworkRequiredError(
            f"{len(pending)} target conversations are pending; -AllowNetwork is required"
        )
    client: QwenClient | None = None
    for target in pending:
        idx = target["idx"]
        prompt = prompts_by_idx[idx]
        logical_attempt = int(latest.get(idx, {}).get("logical_attempt", 0)) + 1
        record: dict[str, Any] = {
            "idx": idx,
            "sentence_sha256": sentence_hash(target["sentence"]),
            "logical_attempt": logical_attempt,
            "created_at": utc_now(),
            "model": chat_model,
            "status": "failed",
            "stage1": None,
            "stage2": None,
            "error": None,
        }
        try:
            if client is None:
                client = client_factory()
                if on_network_call:
                    on_network_call()
            stage1 = client.chat(
                [{"role": "user", "content": prompt["stage1_user"]}],
                temperature=temperature,
                max_tokens=stage1_max_tokens,
                model=chat_model,
            )
            record["stage1"] = stage1
            entity_types = parse_entity_types(stage1["content"])
            messages = [
                {"role": "user", "content": prompt["stage1_user"]},
                {"role": "assistant", "content": stage1["content"]},
                {"role": "user", "content": prompt["stage2_user"]},
            ]
            stage2 = client.chat(
                messages,
                temperature=temperature,
                max_tokens=stage2_max_tokens,
                model=chat_model,
            )
            record["stage2"] = stage2
            trfs = parse_target_trfs(stage2["content"], target["sentence"], main_bank)
            parsed = build_parsed_record(
                target, entity_types, trfs, embedding_model, chat_model
            )
            record["status"] = parsed["status"]
        except MissingEnvironmentVariable:
            raise
        except Exception as error:
            record["error"] = _safe_error(error, secrets)
        append_jsonl(raw_path, record)
        latest[idx] = record
    return latest


def parse_successful_raw_records(
    targets: list[dict[str, Any]],
    latest_raw: dict[int, dict[str, Any]],
    main_bank: list[str],
    embedding_model: str,
    chat_model: str,
) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for target in targets:
        record = latest_raw.get(target["idx"])
        if not record or record.get("status") not in {"complete", "needs_review"}:
            continue
        entity_types = parse_entity_types(record["stage1"]["content"])
        trfs = parse_target_trfs(record["stage2"]["content"], target["sentence"], main_bank)
        parsed.append(
            build_parsed_record(target, entity_types, trfs, embedding_model, chat_model)
        )
    return parsed
