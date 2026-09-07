"""Project span XMLC consensus decisions into formal and reviewable outputs."""

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
    read_jsonl,
    resolve_project_path,
    utc_now,
    validate_run_id,
)


DECISION_STATUSES = ("accepted", "unsolved", "abstained", "negative")


def select_annotations(
    consensus_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    formal: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    seen_indexes: set[int] = set()

    for record in consensus_records:
        idx = record.get("idx")
        sentence = record.get("sentence")
        status = record.get("status")
        if not isinstance(idx, int) or isinstance(idx, bool):
            raise TypeError("Consensus idx must be an integer")
        if idx in seen_indexes:
            raise ValueError(f"Duplicate consensus idx: {idx}")
        if not isinstance(sentence, str):
            raise TypeError(f"Consensus sentence must be a string for idx={idx}")
        if status not in DECISION_STATUSES:
            raise ValueError(f"Unknown decision status for idx={idx}: {status!r}")
        seen_indexes.add(idx)

        has_skill = record.get("has_skill")
        spans = record.get("spans")
        if not isinstance(spans, list) or not all(isinstance(span, str) for span in spans):
            raise TypeError(f"Consensus spans must be strings for idx={idx}")
        if status == "accepted":
            if has_skill != 1 or not spans:
                raise ValueError(f"Accepted decision requires positive spans for idx={idx}")
            formal.append(
                {"idx": idx, "sentence": sentence, "has_skill": 1, "spans": spans}
            )
        elif status == "negative":
            if has_skill != 0 or spans:
                raise ValueError(f"Negative decision requires empty spans for idx={idx}")
            formal.append(
                {"idx": idx, "sentence": sentence, "has_skill": 0, "spans": []}
            )
        elif has_skill is not None or spans:
            raise ValueError(
                f"Unsolved and abstained decisions require null has_skill and no formal spans for idx={idx}"
            )

        decisions.append(record)

    formal.sort(key=lambda item: item["idx"])
    decisions.sort(key=lambda item: item["idx"])
    return formal, decisions


def write_selection_outputs(
    output_dir: Path,
    formal: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    *,
    annotation_schema: str,
    consensus_schema: str,
    dataset_id: str,
    source_sha256: str,
) -> None:
    formal = [
        {
            "schema_version": "skill-annotation-record-v1",
            "dataset_id": dataset_id,
            "record_id": str(item["idx"]),
            "source_sha256": source_sha256,
            **item,
        }
        for item in formal
    ]
    decisions = [
        {
            "schema_version": "demonstration-record-v1",
            "dataset_id": dataset_id,
            "record_id": str(item["idx"]),
            "source_sha256": source_sha256,
            **item,
        }
        for item in decisions
    ]
    atomic_write_json(output_dir / "annotations.json", formal)
    atomic_write_jsonl(output_dir / "decisions.jsonl", decisions)
    counts = {
        status: sum(item["status"] == status for item in decisions)
        for status in DECISION_STATUSES
    }
    atomic_write_json(
        output_dir / "summary.json",
        {
            "updated_at": utc_now(),
            "annotation_schema": annotation_schema,
            "consensus_schema": consensus_schema,
            "decisions": len(decisions),
            "statuses": counts,
            "formal_annotations": len(formal),
            "formal_positive_annotations": sum(item["has_skill"] == 1 for item in formal),
            "formal_negative_annotations": sum(item["has_skill"] == 0 for item in formal),
            "formal_positive_spans": sum(
                len(item["spans"]) for item in formal if item["has_skill"] == 1
            ),
        },
    )


def _validate_active_run_schema(config: dict[str, Any], manifest: dict[str, Any]) -> None:
    compatibility = manifest.get("compatibility", {})
    if compatibility.get("annotation_schema") != config.get("annotation_schema"):
        raise ValueError("Run annotation schema does not match the active configuration")
    if compatibility.get("consensus_schema") != config.get("consensus_schema"):
        raise ValueError("Run consensus schema does not match the active configuration")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "self_annotator.json",
    )
    args = parser.parse_args()

    run_id = validate_run_id(args.run_id)
    config_path = resolve_project_path(PROJECT_ROOT, args.config)
    config = load_json(config_path)
    paths = build_self_annotation_run_paths(PROJECT_ROOT, config["output"], run_id)
    manifest = load_json(paths.manifest)
    _validate_active_run_schema(config, manifest)
    formal, decisions = select_annotations(
        read_jsonl(paths.aggregated / "consensus.jsonl")
    )
    write_selection_outputs(
        paths.selected,
        formal,
        decisions,
        annotation_schema=config["annotation_schema"],
        consensus_schema=config["consensus_schema"],
        dataset_id=manifest["compatibility"]["dataset_id"],
        source_sha256=manifest["compatibility"]["input_sha256"],
    )
    print(f"selected {len(formal)}/{len(decisions)} formal annotations -> {paths.selected}")


if __name__ == "__main__":
    main()
