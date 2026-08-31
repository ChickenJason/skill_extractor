"""Reparse and reaggregate an archived five-sample run without network access."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    build_self_annotation_run_paths,
    load_json,
    resolve_project_path,
    sha256_file,
    utc_now,
    validate_run_id,
)
from self_consistent_annotation.AskQwen import validate_config  # noqa: E402
from self_consistent_annotation.ParseAnswers import (  # noqa: E402
    build_parse_summary,
    latest_raw_records,
    parse_raw_records,
)
from self_consistent_annotation.SelectAnnotations import (  # noqa: E402
    select_annotations,
    write_selection_outputs,
)
from self_consistent_annotation.SpanXMLCConsensus import (  # noqa: E402
    aggregate_records,
    write_aggregation_outputs,
)


def _validate_source(
    source_root: Path, config: dict[str, Any]
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    manifest_path = source_root / "manifest.json"
    raw_path = source_root / "raw" / "responses.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Source manifest is missing: {manifest_path}")
    if not raw_path.is_file():
        raise FileNotFoundError(f"Source raw responses are missing: {raw_path}")
    source_manifest = load_json(manifest_path)
    if source_manifest.get("status") not in {"completed", "completed_with_errors"}:
        raise ValueError("Source run must be completed before offline reaggregation")
    source_compatibility = source_manifest.get("compatibility", {})
    source_schema = source_compatibility.get("annotation_schema")
    if source_schema != config["annotation_schema"]:
        raise ValueError("Source annotation schema is incompatible with the active config")
    if source_compatibility.get("generation", {}).get("samples") not in (None, 5):
        raise ValueError("Source run must use the fixed five-sample protocol")

    records = latest_raw_records(raw_path)
    expected = source_manifest.get("compatibility", {}).get("selected_count")
    if isinstance(expected, int) and len(records) != expected:
        raise ValueError(
            f"Source manifest expects {expected} records but raw file contains {len(records)}"
        )
    seen: set[int] = set()
    for record in records:
        idx = record.get("idx")
        if isinstance(idx, bool) or not isinstance(idx, int) or idx in seen:
            raise ValueError("Source raw records must have unique integer indexes")
        seen.add(idx)
        samples = record.get("samples")
        if not isinstance(samples, list):
            raise TypeError(f"Source samples must be a list for idx={idx}")
        indexes = sorted(sample.get("sample_index") for sample in samples)
        if indexes != list(range(5)):
            raise ValueError(
                f"Source idx={idx} must contain exactly sample_index 0..4; found {indexes}"
            )
    return raw_path, records, source_manifest


def reaggregate_run(
    *, source_root: Path, run_id: str, config: dict[str, Any]
) -> dict[str, Any]:
    """Create an immutable derived run from archived raw responses."""

    validate_config(config)
    source_root = source_root.resolve()
    raw_path, raw_records, source_manifest = _validate_source(source_root, config)
    source_manifest_path = source_root / "manifest.json"
    raw_sha_before = sha256_file(raw_path)
    manifest_sha_before = sha256_file(source_manifest_path)

    paths = build_self_annotation_run_paths(PROJECT_ROOT, config["output"], run_id)
    if paths.root.exists() and any(paths.root.iterdir()):
        raise FileExistsError(f"Target run directory is not empty: {paths.root}")
    paths.root.mkdir(parents=True, exist_ok=True)
    compatibility = {
        "config_version": config["version"],
        "annotation_schema": config["annotation_schema"],
        "consensus_schema": config["consensus_schema"],
        "generation": config["generation"],
        "aggregation": config["aggregation"],
        "source_run_id": source_manifest.get("run_id"),
        "source_raw_path": str(raw_path),
        "source_raw_sha256": raw_sha_before,
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": manifest_sha_before,
        "network_called": False,
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "run_kind": "offline_reaggregation",
        "status": "running",
        "created_at": utc_now(),
        "network_called": False,
        "compatibility": compatibility,
    }
    atomic_write_json(paths.manifest, manifest)
    atomic_write_json(
        paths.raw / "source_reference.json",
        {
            "source_run_root": str(source_root),
            "source_raw_path": str(raw_path),
            "source_raw_sha256": raw_sha_before,
            "source_manifest_path": str(source_manifest_path),
            "source_manifest_sha256": manifest_sha_before,
            "source_record_count": len(raw_records),
            "network_called": False,
        },
    )

    try:
        parsed = parse_raw_records(raw_records)
        atomic_write_jsonl(paths.parsed / "samples.jsonl", parsed)
        atomic_write_json(paths.parsed / "summary.json", build_parse_summary(parsed))

        consensus, audit, uncertainty = aggregate_records(parsed, config["aggregation"])
        write_aggregation_outputs(paths.aggregated, consensus, audit, uncertainty)
        formal, decisions = select_annotations(consensus)
        write_selection_outputs(
            paths.selected,
            formal,
            decisions,
            annotation_schema=config["annotation_schema"],
            consensus_schema=config["consensus_schema"],
        )

        if (
            sha256_file(raw_path) != raw_sha_before
            or sha256_file(source_manifest_path) != manifest_sha_before
        ):
            raise RuntimeError("Source files changed during offline reaggregation")
        manifest.update(
            {
                "status": "completed",
                "completed_at": utc_now(),
                "source_unchanged": True,
                "summary": {
                    "source_records": len(raw_records),
                    "parsed_samples": len(parsed),
                    "consensus_records": len(consensus),
                    "formal_annotations": len(formal),
                },
            }
        )
        atomic_write_json(paths.manifest, manifest)
        return manifest
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "failed_at": utc_now(),
                "error": {"type": error.__class__.__name__, "message": str(error)},
            }
        )
        atomic_write_json(paths.manifest, manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument(
        "--run-id", default="span-xmlc-base5-replay-majority3-anchor-v1"
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "pipeline.json"
    )
    args = parser.parse_args()

    config_path = resolve_project_path(PROJECT_ROOT, args.config)
    config = load_json(config_path)
    source_root = resolve_project_path(PROJECT_ROOT, args.source_run_root)
    run_id = validate_run_id(args.run_id)
    manifest = reaggregate_run(
        source_root=source_root, run_id=run_id, config=config
    )
    output_root = build_self_annotation_run_paths(
        PROJECT_ROOT, config["output"], run_id
    ).root
    print(
        f"offline reaggregation completed: {manifest['summary']['consensus_records']} "
        f"sentences -> {output_root}"
    )


if __name__ == "__main__":
    main()
