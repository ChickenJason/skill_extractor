"""Read-only validation for a completed self-annotation run."""

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

from common.io_utils import (  # noqa: E402
    build_self_annotation_run_paths,
    load_json,
    read_jsonl,
    resolve_project_path,
    sha256_file,
)


def validate(config_path: Path, run_id: str) -> dict[str, Any]:
    config = load_json(config_path)
    paths = build_self_annotation_run_paths(PROJECT_ROOT, config["output"], run_id)
    manifest = load_json(paths.manifest)
    if manifest.get("status") not in {"completed", "completed_with_errors"}:
        raise ValueError("Self-annotation run is not completed")
    compatibility = manifest.get("compatibility", {})
    decisions = read_jsonl(paths.selected / "decisions.jsonl")
    annotations = load_json(paths.selected / "annotations.json")
    summary = load_json(paths.selected / "summary.json")
    if len(decisions) != compatibility.get("selected_count"):
        raise ValueError("Decision count differs from the run manifest")
    if len(annotations) != summary.get("formal_annotations"):
        raise ValueError("Formal annotation count differs from selected summary")
    dataset_id = compatibility.get("dataset_id")
    source_sha256 = compatibility.get("input_sha256")
    if any(
        item.get("dataset_id") != dataset_id
        or item.get("record_id") != str(item.get("idx"))
        or item.get("source_sha256") != source_sha256
        for item in decisions
    ):
        raise ValueError("Selected decisions do not carry the run identity contract")
    return {
        "run_id": run_id,
        "status": "valid",
        "records": len(decisions),
        "formal": len(annotations),
        "dataset_id": dataset_id,
        "decisions_sha256": sha256_file(paths.selected / "decisions.jsonl"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/self_annotator.json")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        result = validate(
            resolve_project_path(PROJECT_ROOT, args.config).resolve(), args.run_id
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        print(f"Self annotator validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
