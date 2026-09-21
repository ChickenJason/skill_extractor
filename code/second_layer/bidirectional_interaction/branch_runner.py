"""Run one frozen interaction branch for one round."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import (  # noqa: E402
    append_jsonl,
    atomic_write_json,
    atomic_write_jsonl,
    expand_environment_references,
    load_json,
    read_jsonl,
    resolve_project_path,
    sha256_file,
    utc_now,
)
from common.qwen_client import QwenSettings  # noqa: E402
from second_layer.bidirectional_interaction.common import (  # noqa: E402
    InteractionError,
    branch_run_id,
    load_config,
)
from second_layer.bidirectional_interaction.online import reflect_one  # noqa: E402
from second_layer.bidirectional_interaction.schema_client import JsonSchemaQwenClient  # noqa: E402


def _now_microseconds() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--parent-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--branch", choices=["trf", "exemplar"], required=True)
    parser.add_argument("--round", type=int, choices=[1, 2], required=True)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args(argv)


def _identity(item: dict[str, Any]) -> tuple[str, str]:
    return item["dataset_id"], item["record_id"]


def _latest(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for item in read_jsonl(path):
        result[_identity(item)] = item
    return result


def _metrics(raw_records: list[dict[str, Any]]) -> dict[str, int]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in raw_records:
        latest[_identity(raw)] = raw
    values = {
        "model_responses": 0,
        "repair_responses": 0,
        "failed_records": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    values["failed_records"] = sum(
        raw.get("status") == "failed" for raw in latest.values()
    )
    for raw in raw_records:
        for attempt in raw.get("attempts", []):
            response = attempt.get("response")
            if response is None:
                continue
            values["model_responses"] += 1
            if attempt.get("kind") == "repair":
                values["repair_responses"] += 1
            usage = response.get("usage", {})
            values["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
            values["completion_tokens"] += int(usage.get("completion_tokens", 0))
    return values


def _write_manifest(path: Path, value: dict[str, Any]) -> None:
    atomic_write_json(path, value)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.retry_failed and not args.resume:
        raise InteractionError("--retry-failed requires --resume")
    parent_root = Path(args.parent_root).resolve()
    round_root = parent_root / "rounds" / f"round-{args.round:02d}"
    branch_root = round_root / "branches" / args.branch
    manifest_path = branch_root / "manifest.json"
    input_path = round_root / "inputs" / "records.jsonl"
    prompt_path = branch_root / "prompts" / "records.jsonl"
    raw_path = branch_root / "raw" / "records.jsonl"
    parsed_path = branch_root / "parsed" / "proposals.jsonl"
    config_path = resolve_project_path(PROJECT_ROOT, args.config).resolve()
    config = load_config(config_path)
    expected_run_id = branch_run_id(parent_root.name, args.round, args.branch)
    if args.run_id != expected_run_id:
        raise InteractionError("Branch run-id does not match parent/round/branch")
    if not input_path.is_file() or not prompt_path.is_file():
        raise InteractionError("Frozen round inputs/prompts are missing")
    inputs = read_jsonl(input_path)
    prompts = read_jsonl(prompt_path)
    prompt_by_identity = {_identity(item): item for item in prompts}
    input_ids = [_identity(item) for item in inputs]
    if len(input_ids) != len(set(input_ids)) or set(prompt_by_identity) != set(input_ids):
        raise InteractionError("Frozen branch input/prompt identities mismatch")
    for record in inputs:
        prompt = prompt_by_identity[_identity(record)]
        if prompt.get("branch") != args.branch or prompt.get("round") != args.round:
            raise InteractionError("Frozen prompt branch/round mismatch")

    started_at = _now_microseconds()
    started_ns = time.time_ns()
    if manifest_path.exists():
        if not args.resume:
            raise InteractionError("Existing branch requires --resume")
        manifest = load_json(manifest_path)
        if manifest.get("status") == "completed":
            manifest.update(
                {
                    "last_started_at": started_at,
                    "last_started_unix_ns": started_ns,
                    "completed_at": _now_microseconds(),
                    "completed_unix_ns": time.time_ns(),
                }
            )
            _write_manifest(manifest_path, manifest)
            return manifest
        if manifest.get("input_sha256") != sha256_file(input_path) or manifest.get("prompt_sha256") != sha256_file(prompt_path):
            raise InteractionError("Branch frozen artifacts changed")
        manifest.update({"status": "running", "error": None, "last_started_at": started_at, "last_started_unix_ns": started_ns})
    else:
        branch_root.mkdir(parents=True, exist_ok=True)
        (branch_root / "raw").mkdir(exist_ok=True)
        (branch_root / "parsed").mkdir(exist_ok=True)
        manifest = {
            "schema_version": 1,
            "run_id": args.run_id,
            "branch": args.branch,
            "round": args.round,
            "status": "running",
            "started_at": started_at,
            "started_unix_ns": started_ns,
            "last_started_at": started_at,
            "last_started_unix_ns": started_ns,
            "completed_at": None,
            "completed_unix_ns": None,
            "network_called": False,
            "input_sha256": sha256_file(input_path),
            "prompt_sha256": sha256_file(prompt_path),
            "metrics": {},
            "error": None,
        }
    _write_manifest(manifest_path, manifest)

    latest = _latest(raw_path)
    unexpected = set(latest) - set(input_ids)
    if unexpected:
        raise InteractionError("Raw branch audit contains unexpected target identities")
    pending = [
        item
        for item in inputs
        if _identity(item) not in latest
        or (
            latest[_identity(item)].get("status") == "failed"
            and args.retry_failed
        )
    ]
    if pending and not args.allow_network:
        manifest.update(
            {
                "status": "partial",
                "completed_at": _now_microseconds(),
                "completed_unix_ns": time.time_ns(),
                "metrics": _metrics(read_jsonl(raw_path)),
                "error": f"{len(pending)} targets pending; --allow-network is required",
            }
        )
        _write_manifest(manifest_path, manifest)
        return manifest

    secrets: list[str] = []
    client: JsonSchemaQwenClient | None = None
    try:
        if pending:
            provider = expand_environment_references(config["provider"])
            secrets.append(provider["api_key"])
            settings = QwenSettings(
                chat_model=config["chat"]["model"],
                base_url=provider["base_url"],
                enable_thinking=config["chat"]["enable_thinking"],
                structured_output=True,
                timeout_seconds=config["chat"]["timeout_seconds"],
                max_retries=config["chat"]["max_retries"],
            )
            client = JsonSchemaQwenClient(api_key=provider["api_key"], settings=settings)
        for record in pending:
            prompt = prompt_by_identity[_identity(record)]

            def mark_network() -> None:
                if manifest.get("network_called") is not True:
                    manifest["network_called"] = True
                    _write_manifest(manifest_path, manifest)

            raw, _ = reflect_one(
                branch=args.branch,
                record=record,
                previous_r=record["previous_r"],
                previous_h=record["previous_h"],
                prompt=prompt,
                client=client,
                chat=config["chat"],
                max_repairs=config["interaction"]["max_repair_attempts"],
                secrets=secrets,
                before_call=mark_network,
            )
            append_jsonl(raw_path, raw)
            latest[_identity(record)] = raw
    except Exception as error:
        manifest.update(
            {
                "status": "partial",
                "completed_at": _now_microseconds(),
                "completed_unix_ns": time.time_ns(),
                "metrics": _metrics(read_jsonl(raw_path)),
                "error": str(error),
            }
        )
        _write_manifest(manifest_path, manifest)
        return manifest

    latest = _latest(raw_path)
    parsed = [latest[_identity(item)]["proposal"] for item in inputs if latest.get(_identity(item), {}).get("status") == "complete"]
    atomic_write_jsonl(parsed_path, parsed)
    complete = len(parsed) == len(inputs)
    manifest.update(
        {
            "status": "completed" if complete else "partial",
            "completed_at": _now_microseconds(),
            "completed_unix_ns": time.time_ns(),
            "metrics": _metrics(read_jsonl(raw_path)),
            "error": None if complete else "At least one target has no completed proposal",
        }
    )
    _write_manifest(manifest_path, manifest)
    return manifest


def main() -> int:
    args = parse_args()
    try:
        manifest = run(args)
        print(json.dumps({"run_id": args.run_id, "status": manifest["status"]}, ensure_ascii=False))
        return 0
    except Exception as error:
        print(f"RunInteractionBranch failed: {error}", file=sys.stderr)
        return 2 if isinstance(error, InteractionError) else 1


if __name__ == "__main__":
    raise SystemExit(main())
