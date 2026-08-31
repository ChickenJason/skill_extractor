"""Read-only validation for a completed three-stage TRF run."""

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

from common.io_utils import load_json, read_jsonl, sha256_file  # noqa: E402
from trf.AssignPseudoTRFs import _validate_pseudo_outputs  # noqa: E402
from trf.common import (  # noqa: E402
    STAGE_NAMES,
    TRFError,
    collect_output_hashes,
    load_trf_config,
    parse_config_argument,
    run_paths,
    verify_run_source,
)


def validate_completed_run(config: dict[str, Any], run_id: str) -> dict[str, Any]:
    paths = run_paths(config, run_id)
    if not paths.manifest.is_file():
        raise TRFError(f"TRF run manifest does not exist: {paths.manifest}")
    manifest = load_json(paths.manifest)
    if manifest.get("status") != "completed":
        raise TRFError(f"TRF run is not completed: {manifest.get('status')!r}")
    if manifest.get("source_unchanged") is not True:
        raise TRFError("TRF manifest does not confirm immutable source files")
    if any(manifest["stages"][stage]["status"] != "completed" for stage in STAGE_NAMES):
        raise TRFError("At least one TRF stage is not completed")
    verify_run_source(config, paths)
    actual_hashes = collect_output_hashes(paths)
    if actual_hashes != manifest.get("outputs"):
        raise TRFError("TRF output hashes differ from the completed manifest")

    corpus = read_jsonl(paths.corpus / "records.jsonl")
    excluded = read_jsonl(paths.corpus / "excluded.jsonl")
    selected = load_json(paths.candidates / "selected_trfs.json")
    pseudo = read_jsonl(paths.demonstrations / "pseudo_trfs.jsonl")
    expected = config["source"]["expected_counts"]
    counts = {
        "accepted": sum(item["status"] == "accepted" for item in corpus),
        "negative": sum(item["status"] == "negative" for item in corpus),
        "unsolved": sum(item["status"] == "unsolved" for item in excluded),
        "abstained": sum(item["status"] == "abstained" for item in excluded),
    }
    if len(corpus) != expected["formal"] or len(corpus) + len(excluded) != expected["total"]:
        raise TRFError("Corpus/exclusion row counts violate the fixed contract")
    if any(counts[key] != expected[key] for key in counts):
        raise TRFError(f"Class counts violate the fixed contract: {counts}")
    maximum = int(config["candidates"]["max_trfs"])
    if len(selected["trfs"]) != maximum or len(selected["context_only_trfs"]) != maximum:
        raise TRFError("Candidate banks do not each contain the fixed number of TRFs")
    _validate_pseudo_outputs(pseudo, corpus, selected, config)

    model_files = load_json(paths.demonstrations / "model_files.json")
    if model_files["model"]["revision"] != config["model"]["revision"]:
        raise TRFError("Model revision differs from config/trf.json")
    for weight in model_files["weight_files"]:
        weight_path = Path(model_files["snapshot_path"]) / weight["file"]
        if not weight_path.is_file() or sha256_file(weight_path) != weight["sha256"]:
            raise TRFError(f"Model weight hash mismatch: {weight_path}")

    return {
        "run_id": run_id,
        "status": "validated",
        "counts": counts,
        "formal": len(corpus),
        "excluded": len(excluded),
        "main_trfs": selected["trfs"],
        "context_only_trfs": selected["context_only_trfs"],
        "pseudo_records": len(pseudo),
        "model_revision": model_files["model"]["revision"],
        "weight_sha256": [item["sha256"] for item in model_files["weight_files"]],
        "semantic_acceptance": load_json(paths.demonstrations / "summary.json")[
            "semantic_acceptance"
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/trf.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--self-annotator-run-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_trf_config(
            parse_config_argument(args.config), args.self_annotator_run_id
        )
        summary = validate_completed_run(config, args.run_id)
    except Exception as error:
        print(f"ValidateTRFRun failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
