"""Fail-closed validation for a completed target TRF extraction run."""

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
from trf_target.common import (  # noqa: E402
    TargetTRFError,
    assert_sources_unchanged,
    collect_output_hashes,
    latest_records,
    load_target_config,
    target_run_paths,
)
from trf_target.diagnostics import build_diagnostics  # noqa: E402
from trf_target.online import (  # noqa: E402
    load_embedding_records,
    parse_successful_raw_records,
)
from trf_target.pipeline import (  # noqa: E402
    build_prompts,
    load_source_bundle,
    retrieve_demonstrations,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/trf_target.json")
    parser.add_argument("--run-id", required=True)
    return parser.parse_args(argv)


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise TargetTRFError(f"Validation mismatch: {label}")


def validate(config_path: Path, run_id: str) -> dict[str, Any]:
    config = load_target_config(config_path)
    paths = target_run_paths(config, run_id)
    manifest = load_json(paths.manifest)
    if manifest.get("status") != "completed":
        raise TargetTRFError("Only completed target TRF runs can pass validation")
    if set(manifest.get("stages", {}).values()) != {"completed"}:
        raise TargetTRFError("Every target TRF stage must be completed")
    source_snapshot = load_json(paths.root / "source_snapshot.json")
    assert_sources_unchanged(config, source_snapshot)
    if manifest.get("source_unchanged") is not True:
        raise TargetTRFError("Manifest does not attest source immutability")
    _equal(collect_output_hashes(paths), manifest.get("outputs"), "output SHA256 ledger")

    bundle = load_source_bundle(config)
    targets = read_jsonl(paths.targets / "records.jsonl")
    descriptor = manifest["compatibility"]["targets"]
    _equal(len(targets), descriptor["record_count"], "target count")
    _equal([item["idx"] for item in targets], descriptor["indexes"], "target indexes")

    embedding = config["embedding"]
    demo_cache = load_embedding_records(
        paths.embeddings / "demonstrations.jsonl",
        bundle["demonstrations"],
        embedding["model"],
        embedding["dimensions"],
    )
    target_cache = load_embedding_records(
        paths.embeddings / "targets.jsonl",
        targets,
        embedding["model"],
        embedding["dimensions"],
    )
    demo_vectors = {idx: item["vector"] for idx, item in demo_cache.items()}
    target_vectors = {idx: item["vector"] for idx, item in target_cache.items()}
    _equal(
        len(demo_vectors),
        len(bundle["demonstrations"]),
        "demonstration embedding count",
    )
    _equal(len(target_vectors), len(targets), "target embedding count")

    mode = descriptor["mode"]
    expected_retrieval = [
        retrieve_demonstrations(
            target,
            bundle["demonstrations"],
            target_vectors[target["idx"]],
            demo_vectors,
            leave_one_out=mode == "leave-one-out",
            nearest_neighbors=config["retrieval"]["nearest_neighbors"],
            selected_count=config["retrieval"]["demonstrations"],
            similarity_decimals=config["retrieval"]["similarity_decimals"],
        )
        for target in targets
    ]
    actual_retrieval = read_jsonl(paths.retrieval / "records.jsonl")
    _equal(actual_retrieval, expected_retrieval, "retrieval records")
    retrieval_by_idx = {item["idx"]: item for item in actual_retrieval}

    expected_prompts = [
        build_prompts(
            target,
            retrieval_by_idx[target["idx"]],
            config["chat"]["max_prompt_characters"],
        )
        for target in targets
    ]
    _equal(read_jsonl(paths.prompts / "records.jsonl"), expected_prompts, "prompts")

    latest_raw = latest_records(paths.raw / "responses.jsonl")
    if set(latest_raw) != {item["idx"] for item in targets}:
        raise TargetTRFError("A completed run needs one latest raw result per target")
    if any(item.get("status") not in {"complete", "needs_review"} for item in latest_raw.values()):
        raise TargetTRFError("A completed run contains a latest failed raw result")
    expected_parsed = parse_successful_raw_records(
        targets,
        latest_raw,
        bundle["main_trfs"],
        embedding["model"],
        config["chat"]["model"],
    )
    actual_parsed = read_jsonl(paths.parsed / "records.jsonl")
    _equal(actual_parsed, expected_parsed, "parsed TRFs")
    expected_summary, expected_review = build_diagnostics(
        mode,
        targets,
        actual_parsed,
        retrieval_by_idx,
        bundle,
        config["diagnostics"]["review_sample_size"],
    )
    _equal(load_json(paths.audit / "summary.json"), expected_summary, "summary")
    _equal(read_jsonl(paths.audit / "manual_review.jsonl"), expected_review, "review sample")
    _equal(manifest.get("summary"), expected_summary, "manifest summary")
    return {
        "run_id": run_id,
        "status": "valid",
        "mode": mode,
        "targets": len(targets),
        "retrieval_k": config["retrieval"]["nearest_neighbors"],
        "prompt_demonstrations": config["retrieval"]["demonstrations"],
        "semantic_acceptance": "pending_manual_review",
    }


def main() -> int:
    args = parse_args()
    config_path = resolve_project_path(PROJECT_ROOT, args.config).resolve()
    try:
        result = validate(config_path, args.run_id)
    except Exception as error:
        print(f"ValidateTargetTRFRun failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
