"""Read-only reconstruction of prepared or completed discriminator runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import load_json, read_jsonl, resolve_project_path  # noqa: E402
from instance_discriminator.common import (  # noqa: E402
    InstanceDiscriminatorError,
    assert_sources_unchanged,
    collect_output_hashes,
    discriminator_run_paths,
    implementation_hashes,
    latest_records,
    load_discriminator_config,
    load_source_bundle,
    target_descriptor,
)
from instance_discriminator.diagnostics import (  # noqa: E402
    build_diagnostics,
    prepared_summary,
)
from instance_discriminator.online import parse_successful_raw_records  # noqa: E402
from instance_discriminator.pipeline import (  # noqa: E402
    build_candidate_record,
    build_discriminator_prompt,
    build_selected_record,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/instance_discriminator.json")
    parser.add_argument("--run-id", required=True)
    return parser.parse_args(argv)


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise InstanceDiscriminatorError(f"Validation mismatch: {label}")


def validate(config_path: Path, run_id: str) -> dict[str, Any]:
    config = load_discriminator_config(config_path)
    paths = discriminator_run_paths(config, run_id)
    manifest = load_json(paths.manifest)
    if manifest.get("status") not in {"partial", "completed"}:
        raise InstanceDiscriminatorError("Only prepared or completed runs can pass validation")
    source = manifest.get("compatibility", {}).get("source", {})
    target_run_id = source.get("target_trf_run_id")
    if not isinstance(target_run_id, str):
        raise InstanceDiscriminatorError("Manifest has no target TRF run-id")
    source_snapshot = load_json(paths.root / "source_snapshot.json")
    _equal(source_snapshot, source, "source snapshot")
    assert_sources_unchanged(config, target_run_id, source_snapshot)
    if manifest.get("source_unchanged") is not True:
        raise InstanceDiscriminatorError("Manifest does not attest source immutability")
    _equal(manifest.get("implementation"), implementation_hashes(), "implementation")
    _equal(collect_output_hashes(paths), manifest.get("outputs"), "output SHA256 ledger")

    descriptor = manifest["compatibility"]["targets"]
    bundle = load_source_bundle(config, target_run_id, descriptor.get("limit"))
    _equal(target_descriptor(bundle["targets"], descriptor.get("limit")), descriptor, "targets")
    expected_candidates = [
        build_candidate_record(
            target,
            bundle["retrieval_by_idx"][target["idx"]],
            bundle["decisions_by_idx"],
            config["source"]["candidate_count"],
        )
        for target in bundle["targets"]
    ]
    actual_candidates = read_jsonl(paths.candidates / "records.jsonl")
    _equal(actual_candidates, expected_candidates, "candidate records")
    expected_prompts = [
        build_discriminator_prompt(item, config["chat"]["max_prompt_characters"])
        for item in expected_candidates
    ]
    actual_prompts = read_jsonl(paths.prompts / "records.jsonl")
    _equal(actual_prompts, expected_prompts, "prompt records")

    summary = load_json(paths.audit / "summary.json")
    if summary.get("status") == "prepared":
        _equal(manifest.get("status"), "partial", "prepared manifest status")
        _equal(
            manifest.get("stages"),
            {"prepare": "completed", "discriminate": "pending", "diagnostics": "pending"},
            "prepared stages",
        )
        _equal(summary, prepared_summary(len(expected_candidates)), "prepared summary")
        _equal(manifest.get("summary"), summary, "manifest prepared summary")
        if manifest.get("network_called") is not False:
            raise InstanceDiscriminatorError("Prepare-only run must not call the network")
        for path in (
            paths.raw / "responses.jsonl",
            paths.parsed / "judgments.jsonl",
            paths.selected / "records.jsonl",
            paths.audit / "manual_review.jsonl",
        ):
            if path.exists() and path.stat().st_size:
                raise InstanceDiscriminatorError(
                    f"Prepare-only run has an unexpected downstream artifact: {path}"
                )
        return {
            "run_id": run_id,
            "status": "valid_prepared",
            "targets": len(expected_candidates),
            "candidate_count_per_target": config["source"]["candidate_count"],
            "network_called": False,
        }

    if manifest.get("status") != "completed":
        raise InstanceDiscriminatorError(
            "A non-prepare partial online run cannot pass complete validation"
        )
    if set(manifest.get("stages", {}).values()) != {"completed"}:
        raise InstanceDiscriminatorError("Every completed-run stage must be completed")
    latest_raw = latest_records(paths.raw / "responses.jsonl")
    expected_ids = {item["idx"] for item in expected_candidates}
    if set(latest_raw) != expected_ids or any(
        item.get("status") != "complete" for item in latest_raw.values()
    ):
        raise InstanceDiscriminatorError(
            "Completed run needs one successful latest raw response per target"
        )
    prompts_by_idx = {item["idx"]: item for item in expected_prompts}
    expected_parsed = parse_successful_raw_records(
        expected_candidates,
        prompts_by_idx,
        latest_raw,
        chat_model=config["chat"]["model"],
        max_reason_characters=config["chat"]["max_reason_characters"],
    )
    actual_parsed = read_jsonl(paths.parsed / "judgments.jsonl")
    _equal(actual_parsed, expected_parsed, "parsed judgments")
    parsed_by_idx = {item["idx"]: item for item in expected_parsed}
    expected_selected = [
        build_selected_record(
            item,
            parsed_by_idx[item["idx"]]["judgments"],
            config["gate"],
            config["chat"]["model"],
        )
        for item in expected_candidates
    ]
    actual_selected = read_jsonl(paths.selected / "records.jsonl")
    _equal(actual_selected, expected_selected, "selected records")
    expected_summary, expected_review = build_diagnostics(
        expected_candidates,
        expected_parsed,
        expected_selected,
        latest_raw,
        config["diagnostics"]["review_sample_size"],
    )
    _equal(summary, expected_summary, "summary")
    _equal(read_jsonl(paths.audit / "manual_review.jsonl"), expected_review, "review sample")
    _equal(manifest.get("summary"), expected_summary, "manifest summary")
    return {
        "run_id": run_id,
        "status": "valid_completed",
        "targets": len(expected_candidates),
        "selected_total": expected_summary["selection"]["selected_total"],
        "needs_review": expected_summary["needs_review_count"],
        "accuracy_improvement": "not_evaluated",
    }


def main() -> int:
    args = parse_args()
    config_path = resolve_project_path(PROJECT_ROOT, args.config).resolve()
    try:
        result = validate(config_path, args.run_id)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        print(f"ValidateInstanceDiscriminatorRun failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
