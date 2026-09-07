"""Run all three deterministic offline TRF stages under one manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from trf.offline.assign_pseudo_trfs import assign_pseudo_trfs  # noqa: E402
from trf.offline.build_corpus import write_corpus  # noqa: E402
from trf.offline.extract_candidates import write_candidate_bank  # noqa: E402
from trf.offline.common import (  # noqa: E402
    execute_stage,
    finalize_run,
    initialize_run,
    load_trf_config,
    mark_failed,
    parse_config_argument,
    read_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/trf.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--demonstrations")
    parser.add_argument("--allow-model-download", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = parse_config_argument(args.config)
    paths = None
    try:
        config = load_trf_config(config_path, args.demonstrations)
        paths = initialize_run(config_path, config, args.run_id, sys.argv)
        execute_stage(
            config,
            paths,
            "build_corpus",
            None,
            lambda: write_corpus(config, paths),
        )
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
            lambda: assign_pseudo_trfs(config, paths, args.allow_model_download),
        )
        finalize_run(config, paths)
    except Exception as error:
        if paths is not None and paths.manifest.is_file():
            manifest = read_manifest(paths)
            if manifest.get("status") != "failed":
                active = next(
                    (
                        name
                        for name, details in manifest["stages"].items()
                        if details["status"] == "running"
                    ),
                    None,
                )
                mark_failed(paths, active, error)
        print(f"RunTRFOffline failed: {error}", file=sys.stderr)
        return 1
    print(f"TRF offline pipeline completed: {paths.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
