"""Join completed TRF and exemplar evidence into neutral aggregator context records."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.contracts import (  # noqa: E402
    record_identity,
    record_source_sha256,
    validate_unique_identities,
)
from second_layer.common import SecondLayerError  # noqa: E402


def _index(records: list[dict[str, Any]], label: str) -> dict[tuple[str, str], dict[str, Any]]:
    validate_unique_identities(records, label)
    return {record_identity(record, label): record for record in records}


def _validate_alignment(
    target: dict[str, Any],
    record: dict[str, Any],
    label: str,
) -> None:
    identity = record_identity(target, "target")
    if record_identity(record, label) != identity:
        raise SecondLayerError(f"{label} identity mismatch for {identity!r}")
    if record.get("sentence") != target.get("sentence"):
        raise SecondLayerError(f"{label} sentence mismatch for {identity!r}")
    if record_source_sha256(record, label) != record_source_sha256(target, "target"):
        raise SecondLayerError(f"{label} source hash mismatch for {identity!r}")
    if record.get("idx") != target.get("idx"):
        raise SecondLayerError(f"{label} idx mismatch for {identity!r}")


def assemble_context_records(
    targets: list[dict[str, Any]],
    trf_records: list[dict[str, Any]],
    exemplar_records: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build one ready context record per target, failing on any missing branch."""

    validate_unique_identities(targets, "targets")
    trf_by_identity = _index(trf_records, "TRF records")
    exemplar_by_identity = _index(exemplar_records, "exemplar records")
    target_identities = {record_identity(item, "target") for item in targets}
    if set(trf_by_identity) != target_identities:
        raise SecondLayerError("TRF records do not exactly cover the targets")
    if set(exemplar_by_identity) != target_identities:
        raise SecondLayerError("Exemplar records do not exactly cover the targets")

    output: list[dict[str, Any]] = []
    for target in sorted(targets, key=lambda item: item["idx"]):
        identity = record_identity(target, "target")
        trf = trf_by_identity[identity]
        exemplar = exemplar_by_identity[identity]
        _validate_alignment(target, trf, "TRF record")
        _validate_alignment(target, exemplar, "exemplar record")
        if trf.get("schema_version") != "feature-records-v1":
            raise SecondLayerError(f"Unsupported TRF record schema for {identity!r}")
        if trf.get("status") not in {"complete", "needs_review"}:
            raise SecondLayerError(f"TRF branch is incomplete for {identity!r}")
        if exemplar.get("schema_version") != "instance-selection-v1":
            raise SecondLayerError(f"Unsupported exemplar record schema for {identity!r}")
        if exemplar.get("status") not in {"complete", "needs_review"}:
            raise SecondLayerError(f"Exemplar branch is incomplete for {identity!r}")
        evidence = exemplar.get("target_evidence")
        if not isinstance(evidence, dict) or evidence.get("feature_context") != "absent":
            raise SecondLayerError(
                f"Exemplar branch must remain independent of target TRFs for {identity!r}"
            )
        if evidence.get("entity_types") or evidence.get("trfs"):
            raise SecondLayerError(
                f"Feature-absent exemplar evidence must not contain TRF features for {identity!r}"
            )
        selected = exemplar.get("selected")
        if not isinstance(selected, list) or len(selected) != exemplar.get("selected_count"):
            raise SecondLayerError(f"Invalid selected exemplar list for {identity!r}")
        if any(item.get("role") not in {"supporting", "contrastive"} for item in selected):
            raise SecondLayerError(f"Selected exemplars contain an invalid role for {identity!r}")

        output.append(
            {
                "schema_version": "second-layer-context-v1",
                "dataset_id": identity[0],
                "record_id": identity[1],
                "source_sha256": target["source_sha256"],
                "idx": target["idx"],
                "sentence": target["sentence"],
                "assembly_status": "ready",
                "trf_context": {
                    "status": trf["status"],
                    "entity_types": trf["entity_types"],
                    "trfs": trf["trfs"],
                    "retrieval_count": trf["retrieval_count"],
                    "review_reasons": trf["review_reasons"],
                    "models": trf["models"],
                },
                "exemplar_context": {
                    "status": exemplar["status"],
                    "feature_context": "absent",
                    "candidate_count": exemplar["candidate_count"],
                    "eligible_count": exemplar["eligible_count"],
                    "selected_count": exemplar["selected_count"],
                    "selected": selected,
                    "rejected_demo_ids": exemplar["rejected_demo_ids"],
                    "review_reasons": exemplar["review_reasons"],
                    "feature_review_reasons": exemplar["feature_review_reasons"],
                    "models": exemplar["models"],
                },
                "provenance": provenance,
            }
        )
    return output


def context_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "second-layer-summary-v1",
        "status": "completed",
        "target_count": len(records),
        "ready_count": sum(item.get("assembly_status") == "ready" for item in records),
        "trf_total": sum(len(item["trf_context"]["trfs"]) for item in records),
        "selected_exemplar_total": sum(
            item["exemplar_context"]["selected_count"] for item in records
        ),
        "trf_needs_review": sum(
            item["trf_context"]["status"] == "needs_review" for item in records
        ),
        "exemplar_needs_review": sum(
            item["exemplar_context"]["status"] == "needs_review" for item in records
        ),
    }
