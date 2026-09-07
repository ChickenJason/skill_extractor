"""Temporary, output-independent fixtures for TRF integration tests."""

from __future__ import annotations

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
    load_json,
    read_jsonl,
    sha256_file,
)
from trf.offline.build_corpus import write_corpus  # noqa: E402
from trf.offline.extract_candidates import write_candidate_bank  # noqa: E402
from trf.offline.common import (  # noqa: E402
    execute_stage,
    finalize_run,
    initialize_run,
    load_trf_config,
)
from trf.orchestration.runner import build_generated_target_config  # noqa: E402
from trf.target.common import load_target_config  # noqa: E402


DEMONSTRATION_MANIFEST = (
    PROJECT_ROOT / "data" / "processed" / "demonstrations" / "v1" / "manifest.json"
)
DEMONSTRATIONS = str(DEMONSTRATION_MANIFEST)


def prepare_v06_trf_config(root: Path) -> tuple[Path, dict[str, Any]]:
    """Bind a temporary TRF config to the canonical neutral demonstration set."""
    config = load_json(PROJECT_ROOT / "config" / "trf.json")
    config["offline"]["source"]["demonstrations"] = str(DEMONSTRATION_MANIFEST)
    config["output"]["runs_root"] = str(root / "trf-runs")
    config_path = root / "trf.json"
    atomic_write_json(config_path, config)
    return config_path, load_trf_config(config_path)


def _first_visible_character(sentence: str) -> tuple[str, int, int]:
    start = next((index for index, value in enumerate(sentence) if not value.isspace()), 0)
    return sentence[start : start + 1], start, start + 1


def _fake_assign_pseudo_trfs(config: dict[str, Any], paths: Any) -> dict[str, Any]:
    """Write structurally valid deterministic pseudo TRFs without loading BERT."""

    paths.demonstrations.mkdir(parents=True, exist_ok=False)
    corpus = read_jsonl(paths.corpus / "records.jsonl")
    selected = load_json(paths.candidates / "selected_trfs.json")
    top_k = int(config["model"]["top_k"])

    def assignments(record: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
        token, start, end = _first_visible_character(record["sentence"])
        return [
            {
                "text": text,
                "rank": rank,
                "distance": float(rank),
                "nearest_token": token,
                "nearest_token_start": start,
                "nearest_token_end": end,
                "candidate_rank": names.index(text) + 1,
            }
            for rank, text in enumerate(names[:top_k], start=1)
        ]

    pseudo = []
    for record in corpus:
        accepted = record["status"] == "accepted"
        pseudo.append(
            {
                "schema_version": "trf-pseudo-record-v1",
                "dataset_id": record["dataset_id"],
                "record_id": record["record_id"],
                "source_sha256": record["source_sha256"],
                "idx": record["idx"],
                "sentence": record["sentence"],
                "status": record["status"],
                "trfs": assignments(record, selected["trfs"]) if accepted else [],
                "context_only_trfs": (
                    assignments(record, selected["context_only_trfs"])
                    if accepted
                    else []
                ),
            }
        )
    atomic_write_jsonl(paths.demonstrations / "pseudo_trfs.jsonl", pseudo)

    fake_snapshot = paths.root.parent / f"{paths.root.name}-fake-model"
    fake_snapshot.mkdir(parents=True, exist_ok=False)
    weight_path = fake_snapshot / "model.safetensors"
    weight_path.write_bytes(b"deterministic-test-weight")
    atomic_write_json(
        paths.demonstrations / "model_files.json",
        {
            "model": {"revision": config["model"]["revision"]},
            "snapshot_path": str(fake_snapshot),
            "weight_files": [
                {
                    "file": weight_path.name,
                    "sha256": sha256_file(weight_path),
                    "bytes": weight_path.stat().st_size,
                }
            ],
        },
    )
    summary = {
        "records": len(pseudo),
        "semantic_acceptance": "pending_manual_review",
    }
    atomic_write_json(paths.demonstrations / "summary.json", summary)
    return summary


def create_fake_completed_offline(
    config_path: Path,
    demonstrations: str | Path | None,
    run_id: str,
    allow_model_download: bool = False,
    command: list[str] | None = None,
) -> None:
    """Create a validator-compatible offline child entirely in a temp directory."""

    if allow_model_download:
        raise AssertionError("The test fixture must never download a model")
    config = load_trf_config(config_path, demonstrations)
    paths = initialize_run(config_path, config, run_id, command or ["test-fixture"])
    execute_stage(config, paths, "build_corpus", None, lambda: write_corpus(config, paths))
    execute_stage(
        config,
        paths,
        "extract_candidates",
        "build_corpus",
        lambda: write_candidate_bank(config, paths),
    )
    execute_stage(
        config,
        paths,
        "assign_pseudo_trfs",
        "extract_candidates",
        lambda: _fake_assign_pseudo_trfs(config, paths),
    )
    finalize_run(config, paths)


def prepare_generated_target_config(
    root: Path,
    offline_config: dict[str, Any],
    offline_run_id: str,
) -> tuple[Path, dict[str, Any]]:
    module_path = root / "trf.json"
    if not module_path.is_file():
        module_config = load_json(PROJECT_ROOT / "config" / "trf.json")
        module_config["output"]["runs_root"] = offline_config["output"]["runs_root"]
        atomic_write_json(module_path, module_config)
    template = load_target_config(module_path)
    generated = build_generated_target_config(template, offline_config, offline_run_id)
    generated["output"]["runs_root"] = offline_config["output"]["runs_root"]
    config_path = root / "target.generated.json"
    atomic_write_json(config_path, generated)
    return config_path, generated
