"""Run one strict JSON-Schema reflection branch over frozen R0 inputs."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import (  # noqa: E402
    append_jsonl,
    atomic_write_jsonl,
    expand_environment_references,
    load_json,
    read_jsonl,
    redact_secrets,
    resolve_project_path,
    sha256_file,
    utc_now,
)
from common.qwen_client import QwenSettings  # noqa: E402
from second_layer.collaborative_reflection_json_schema.common import (  # noqa: E402
    BRANCHES,
    PROTOCOL_VERSION,
    ReflectionError,
    file_snapshot,
    implementation_hashes,
    load_config,
    write_manifest,
)
from second_layer.collaborative_reflection_json_schema.online import reflect_one  # noqa: E402
from second_layer.collaborative_reflection_json_schema.pipeline import (  # noqa: E402
    build_exemplar_records,
    build_trf_record,
    parse_exemplar_response,
    parse_trf_response,
)
from second_layer.collaborative_reflection_json_schema.schema_client import JsonSchemaQwenClient  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--parent-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--branch", choices=BRANCHES, required=True)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args(argv)


def _latest(records: list[dict[str, Any]], branch: str) -> dict[int, dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    identities: dict[int, tuple[Any, ...]] = {}
    for position, record in enumerate(records, start=1):
        if record.get("schema_version") != "collaborative-reflection-raw-v2" or record.get("branch") != branch:
            raise ReflectionError(f"Invalid {branch} JSON-Schema raw record at line {position}")
        idx = record.get("idx")
        if type(idx) is not int:
            raise ReflectionError(f"Invalid {branch} raw idx at line {position}")
        identity = (
            record.get("dataset_id"), record.get("record_id"),
            record.get("source_sha256"), record.get("sentence"),
        )
        if idx in identities and identities[idx] != identity:
            raise ReflectionError(f"Append-only {branch} raw identity changed for idx={idx}")
        if idx in latest and latest[idx].get("status") == "complete":
            raise ReflectionError(f"Append-only {branch} raw retried completed idx={idx}")
        identities[idx] = identity
        latest[idx] = record
    return latest


def _branch_paths(root: Path, branch: str) -> dict[str, Path]:
    branch_root = root / ("trf-reflection" if branch == "trf" else "exemplar-reflection")
    return {
        "root": branch_root,
        "manifest": branch_root / "manifest.json",
        "prompts": branch_root / "prompts" / "records.jsonl",
        "raw": branch_root / "raw" / "responses.jsonl",
        "parsed": branch_root / "parsed" / ("records.jsonl" if branch == "trf" else "judgments.jsonl"),
        "selected": branch_root / "selected" / "records.jsonl",
    }


def _output_hashes(branch_root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(branch_root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file() and path.name != "manifest.json":
            result[path.relative_to(branch_root).as_posix()] = {
                "sha256": sha256_file(path), "bytes": path.stat().st_size
            }
    return result


def _compatibility(
    config_path: Path, parent_root: Path, run_id: str, branch: str,
    paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "config": file_snapshot(config_path),
        "parent_run_id": load_json(parent_root / "manifest.json")["run_id"],
        "branch_run_id": run_id,
        "branch": branch,
        "reflection_inputs": file_snapshot(parent_root / "source" / "reflection-inputs.jsonl"),
        "prompts": file_snapshot(paths["prompts"]),
        "frozen_source": load_json(parent_root / "source" / "snapshot.json"),
    }


def _initialize(
    paths: dict[str, Path], *, compatibility: dict[str, Any], run_id: str,
    branch: str, resume: bool,
) -> dict[str, Any]:
    implementation = implementation_hashes()
    if paths["manifest"].exists():
        if not resume:
            raise ReflectionError(f"{branch} JSON-Schema reflection manifest already exists")
        manifest = load_json(paths["manifest"])
        if manifest.get("compatibility") != compatibility:
            raise ReflectionError(f"{branch} JSON-Schema reflection resume compatibility mismatch")
        if manifest.get("implementation") != implementation:
            raise ReflectionError(f"{branch} JSON-Schema reflection implementation changed")
        if manifest.get("status") == "completed":
            return manifest
        manifest.update({"status": "running", "completed_at": None, "error": None})
    else:
        paths["root"].mkdir(parents=True, exist_ok=True)
        paths["raw"].parent.mkdir(exist_ok=True)
        paths["parsed"].parent.mkdir(exist_ok=True)
        if branch == "exemplar":
            paths["selected"].parent.mkdir(exist_ok=True)
        manifest = {
            "schema_version": 1,
            "pipeline_version": f"{branch}-reflection-json-schema-v2",
            "protocol_version": PROTOCOL_VERSION,
            "schedule_type": "collaborative_reflection",
            "run_id": run_id,
            "branch": branch,
            "status": "running",
            "created_at": utc_now(),
            "completed_at": None,
            "network_called": False,
            "compatibility": compatibility,
            "implementation": implementation,
            "environment": {"python": platform.python_version(), "platform": platform.platform()},
            "counts": {},
            "metrics": {},
            "outputs": {},
            "error": None,
        }
    write_manifest(paths["manifest"], manifest)
    return manifest


def _make_client(config: dict[str, Any]) -> tuple[JsonSchemaQwenClient, dict[str, Any]]:
    expanded = expand_environment_references(config)
    provider = expanded["provider"]
    chat = expanded["chat"]
    settings = QwenSettings(
        chat_model=chat["model"],
        base_url=provider["base_url"],
        enable_thinking=chat["enable_thinking"],
        structured_output=True,
        timeout_seconds=chat["timeout_seconds"],
        max_retries=chat["max_retries"],
    )
    return JsonSchemaQwenClient(api_key=provider["api_key"], settings=settings), expanded


def run_branch(
    *, config_path: Path, parent_root: Path, run_id: str, branch: str,
    allow_network: bool, resume: bool, retry_failed: bool,
    client_factory: Callable[[dict[str, Any]], tuple[Any, dict[str, Any]]] = _make_client,
) -> dict[str, Any]:
    if retry_failed and not resume:
        raise ReflectionError("--retry-failed requires --resume")
    if not allow_network:
        raise ReflectionError("JSON-Schema reflection inference requires --allow-network")
    config = load_config(config_path)
    parent_root = parent_root.resolve()
    paths = _branch_paths(parent_root, branch)
    compatibility = _compatibility(config_path, parent_root, run_id, branch, paths)
    manifest = _initialize(
        paths, compatibility=compatibility, run_id=run_id, branch=branch, resume=resume
    )
    if manifest.get("status") == "completed":
        return manifest
    inputs = read_jsonl(parent_root / "source" / "reflection-inputs.jsonl")
    prompts = read_jsonl(paths["prompts"])
    if len(prompts) != len(inputs):
        raise ReflectionError(f"{branch} prompt count does not match reflection inputs")
    prompts_by_idx = {item["idx"]: item for item in prompts}
    if len(prompts_by_idx) != len(prompts):
        raise ReflectionError(f"Duplicate {branch} prompt idx")
    raw_records = read_jsonl(paths["raw"])
    latest = _latest(raw_records, branch)
    if set(latest) - {item["idx"] for item in inputs}:
        raise ReflectionError(f"{branch} raw output contains unknown targets")
    client, expanded = client_factory(config)
    manifest["runtime"] = redact_secrets(
        {"provider": expanded["provider"], "chat": expanded["chat"]},
        [expanded["provider"]["api_key"]],
    )
    manifest["network_called"] = True
    write_manifest(paths["manifest"], manifest)
    by_idx = {item["idx"]: item for item in inputs}
    for idx in sorted(by_idx):
        prior = latest.get(idx)
        if prior and prior.get("status") == "complete":
            continue
        if prior and not retry_failed:
            continue
        raw, _ = reflect_one(
            branch=branch,
            record=by_idx[idx],
            prompt=prompts_by_idx[idx],
            client=client,
            chat=config["chat"],
            max_repairs=config["reflection"]["max_repair_attempts"],
        )
        append_jsonl(paths["raw"], raw)
        latest[idx] = raw

    raw_records = read_jsonl(paths["raw"])
    latest = _latest(raw_records, branch)
    parsed_records: list[dict[str, Any]] = []
    selected_records: list[dict[str, Any]] = []
    for idx in sorted(by_idx):
        raw = latest.get(idx)
        if not raw or raw.get("status") != "complete":
            continue
        final = raw["final_response"]["content"]
        if branch == "trf":
            parsed = parse_trf_response(
                final, by_idx[idx], max_reason_characters=config["chat"]["max_reason_characters"]
            )
            parsed_records.append(build_trf_record(by_idx[idx], parsed, config["chat"]["model"]))
        else:
            parsed = parse_exemplar_response(
                final, by_idx[idx], max_reason_characters=config["chat"]["max_reason_characters"]
            )
            parsed_record, selected_record = build_exemplar_records(
                by_idx[idx], parsed, config["gate"], config["chat"]["model"]
            )
            parsed_records.append(parsed_record)
            selected_records.append(selected_record)
    atomic_write_jsonl(paths["parsed"], parsed_records)
    if branch == "exemplar":
        atomic_write_jsonl(paths["selected"], selected_records)
    failed = len(inputs) - len(parsed_records)
    attempts = [attempt for raw in raw_records for attempt in raw.get("attempts", [])]
    metrics = {
        "initial_requests": sum(item.get("kind") == "initial" for item in attempts),
        "repair_requests": sum(item.get("kind") == "repair" for item in attempts),
        "transport_attempts": sum(item.get("transport_attempts") or 0 for item in attempts),
        "outer_retries": sum(max((item.get("transport_attempts") or 0) - 1, 0) for item in attempts),
        "model_calls": sum(item.get("response") is not None for item in attempts),
        "prompt_tokens": sum((item.get("response") or {}).get("usage", {}).get("prompt_tokens", 0) for item in attempts),
        "completion_tokens": sum((item.get("response") or {}).get("usage", {}).get("completion_tokens", 0) for item in attempts),
        "total_tokens": sum((item.get("response") or {}).get("usage", {}).get("total_tokens", 0) for item in attempts),
        "latency_ms": sum((item.get("response") or {}).get("latency_ms", 0) for item in attempts),
        "failed_records": failed,
    }
    manifest = load_json(paths["manifest"])
    manifest.update(
        {
            "status": "partial" if failed else "completed",
            "completed_at": utc_now(),
            "counts": {"target": len(inputs), "completed": len(parsed_records), "failed": failed},
            "metrics": metrics,
            "error": None if not failed else f"{failed} record(s) failed",
        }
    )
    write_manifest(paths["manifest"], manifest)
    manifest["outputs"] = _output_hashes(paths["root"])
    write_manifest(paths["manifest"], manifest)
    return manifest


def main() -> int:
    args = parse_args()
    try:
        result = run_branch(
            config_path=resolve_project_path(PROJECT_ROOT, args.config).resolve(),
            parent_root=Path(args.parent_root).resolve(),
            run_id=args.run_id,
            branch=args.branch,
            allow_network=args.allow_network,
            resume=args.resume,
            retry_failed=args.retry_failed,
        )
        print(json.dumps({"run_id": result["run_id"], "branch": args.branch, "status": result["status"]}))
        return 0
    except Exception as error:
        print(f"RunJsonSchemaReflectionBranch failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
