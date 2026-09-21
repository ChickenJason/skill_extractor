"""Read-only deterministic validator for one reflection branch."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import load_json, read_jsonl, sha256_json  # noqa: E402
from second_layer.collaborative_reflection.branch_runner import (  # noqa: E402
    _branch_paths,
    _compatibility,
    _latest,
    _output_hashes,
)
from second_layer.collaborative_reflection.common import (  # noqa: E402
    ReflectionError,
    implementation_hashes,
    load_reflection_config,
)
from second_layer.collaborative_reflection.pipeline import (  # noqa: E402
    build_exemplar_records,
    build_prompt,
    build_repair_messages,
    build_trf_record,
    parse_exemplar_response,
    parse_trf_response,
)


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ReflectionError(f"Validation mismatch: {label}")


def _validate_raw(
    raw: dict[str, Any], prompt: dict[str, Any], record: dict[str, Any],
    branch: str, chat: dict[str, Any], max_repairs: int
) -> dict[str, Any] | None:
    for key in ("dataset_id", "record_id", "source_sha256", "idx", "sentence"):
        _equal(raw.get(key), record.get(key), f"{branch} raw {key} for idx={record['idx']}")
    _equal(raw.get("prompt_sha256"), prompt["prompt_sha256"], f"{branch} raw prompt hash")
    attempts = raw.get("attempts")
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= max_repairs + 1:
        raise ReflectionError(f"Invalid {branch} attempt count for idx={record['idx']}")
    prior_content = ""
    prior_error: str | None = None
    parsed: dict[str, Any] | None = None
    for position, attempt in enumerate(attempts, start=1):
        _equal(attempt.get("attempt"), position, f"{branch} attempt number")
        expected_kind = "initial" if position == 1 else "repair"
        _equal(attempt.get("kind"), expected_kind, f"{branch} attempt kind")
        if position == 1:
            messages = prompt["messages"]
        else:
            if prior_error is None:
                raise ReflectionError(f"Unnecessary {branch} repair for idx={record['idx']}")
            messages = build_repair_messages(
                prompt,
                branch,
                prior_error,
                prior_content,
                chat["max_reason_characters"],
            )
        _equal(attempt.get("request_sha256"), sha256_json(messages), f"{branch} request hash")
        response = attempt.get("response")
        if response is None:
            if not attempt.get("transport_error") or position != len(attempts):
                raise ReflectionError(f"Invalid {branch} transport audit for idx={record['idx']}")
            prior_error = None
            continue
        if attempt.get("transport_error") is not None:
            raise ReflectionError(f"{branch} response and transport error coexist")
        prior_content = response.get("content", "")
        try:
            parser = parse_trf_response if branch == "trf" else parse_exemplar_response
            parsed = parser(
                prior_content,
                record,
                max_reason_characters=chat["max_reason_characters"],
            )
            computed_error = None
        except Exception as error:
            parsed = None
            computed_error = str(error)
        _equal(attempt.get("parse_error"), computed_error, f"{branch} parse error audit")
        prior_error = computed_error
        if parsed is not None and position != len(attempts):
            raise ReflectionError(f"{branch} audit continued after a valid response")
    expected_status = "complete" if parsed is not None else "failed"
    _equal(raw.get("status"), expected_status, f"{branch} raw status")
    if expected_status == "complete":
        _equal(raw.get("final_response"), attempts[-1]["response"], f"{branch} final response")
        _equal(raw.get("error"), None, f"{branch} final error")
    else:
        if not raw.get("error"):
            raise ReflectionError(f"Failed {branch} raw record has no error")
    return parsed


def validate_branch(
    *, config_path: Path, parent_root: Path, run_id: str, branch: str
) -> dict[str, Any]:
    config = load_reflection_config(config_path)
    paths = _branch_paths(parent_root.resolve(), branch)
    manifest = load_json(paths["manifest"])
    if manifest.get("status") != "completed":
        raise ReflectionError(f"Only completed {branch} reflection runs can pass validation")
    _equal(manifest.get("run_id"), run_id, f"{branch} run ID")
    _equal(manifest.get("branch"), branch, f"{branch} manifest branch")
    _equal(manifest.get("implementation"), implementation_hashes(), f"{branch} implementation")
    _equal(
        manifest.get("compatibility"),
        _compatibility(config_path, parent_root.resolve(), run_id, branch, paths),
        f"{branch} compatibility",
    )
    inputs = read_jsonl(parent_root / "source" / "reflection-inputs.jsonl")
    prompts = read_jsonl(paths["prompts"])
    expected_prompts = [
        build_prompt(
            item,
            branch,
            config["chat"]["max_prompt_characters"],
            config["chat"]["max_reason_characters"],
        )
        for item in inputs
    ]
    _equal(prompts, expected_prompts, f"{branch} deterministic prompts")
    prompts_by_idx = {item["idx"]: item for item in prompts}
    inputs_by_idx = {item["idx"]: item for item in inputs}
    raw_records = read_jsonl(paths["raw"])
    latest = _latest(raw_records, branch)
    if set(latest) != set(inputs_by_idx):
        raise ReflectionError(f"{branch} latest raw records do not cover all inputs")
    parsed_by_raw: dict[int, dict[str, Any]] = {}
    for raw in raw_records:
        idx = raw["idx"]
        if idx not in inputs_by_idx:
            raise ReflectionError(f"{branch} raw record contains unknown idx={idx}")
        parsed = _validate_raw(
            raw, prompts_by_idx[idx], inputs_by_idx[idx], branch,
            config["chat"], config["reflection"]["max_repair_attempts"],
        )
        if raw is latest[idx] and parsed is not None:
            parsed_by_raw[idx] = parsed
    if set(parsed_by_raw) != set(inputs_by_idx):
        raise ReflectionError(f"{branch} has unresolved latest failures")
    if branch == "trf":
        expected_parsed = [
            build_trf_record(inputs_by_idx[idx], parsed_by_raw[idx], config["chat"]["model"])
            for idx in sorted(inputs_by_idx)
        ]
        _equal(read_jsonl(paths["parsed"]), expected_parsed, "TRF reflected records")
    else:
        expected_parsed = []
        expected_selected = []
        for idx in sorted(inputs_by_idx):
            parsed, selected = build_exemplar_records(
                inputs_by_idx[idx], parsed_by_raw[idx], config["gate"], config["chat"]["model"]
            )
            expected_parsed.append(parsed)
            expected_selected.append(selected)
        _equal(read_jsonl(paths["parsed"]), expected_parsed, "exemplar reflected judgments")
        _equal(read_jsonl(paths["selected"]), expected_selected, "exemplar reflected selection")
    _equal(manifest.get("outputs"), _output_hashes(paths["root"]), f"{branch} output hashes")
    _equal(manifest.get("counts", {}).get("target"), len(inputs), f"{branch} target count")
    _equal(manifest.get("counts", {}).get("completed"), len(inputs), f"{branch} completed count")
    _equal(manifest.get("counts", {}).get("failed"), 0, f"{branch} failed count")
    return {
        "schema_version": "collaborative-reflection-branch-validation-v1",
        "branch": branch,
        "run_id": run_id,
        "status": "valid",
        "targets": len(inputs),
        "metrics": manifest.get("metrics"),
    }
