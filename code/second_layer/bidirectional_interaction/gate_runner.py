"""Run the independent frozen-anchor interaction gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import atomic_write_json, load_json, read_jsonl, resolve_project_path  # noqa: E402
from second_layer.bidirectional_interaction.common import collect_output_hashes, file_snapshot, load_config, paths_for, update_manifest  # noqa: E402
from second_layer.bidirectional_interaction.gate import evaluate_gate, load_gate_spec, target_identities  # noqa: E402
from second_layer.bidirectional_interaction.runner import run as run_interaction  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/second_layer_bidirectional_interaction.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gate-spec", required=True)
    parser.add_argument("--mode", choices=["none", "trf_to_exemplar", "exemplar_to_trf", "bidirectional"], default="bidirectional")
    parser.add_argument("--max-rounds", choices=[1, 2], type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    config_path = resolve_project_path(PROJECT_ROOT, args.config).resolve()
    gate_path = resolve_project_path(PROJECT_ROOT, args.gate_spec).resolve()
    spec = load_gate_spec(gate_path)
    runner_args = argparse.Namespace(
        config=str(config_path),
        run_id=args.run_id,
        base_concurrent_run_id=spec["base_concurrent_run_id"],
        mode=args.mode,
        max_rounds=args.max_rounds,
        limit=None,
        prepare_only=args.prepare_only,
        resume=args.resume,
        retry_failed=args.retry_failed,
        allow_network=args.allow_network,
        confirm_full_run=True,
    )
    manifest = run_interaction(
        runner_args,
        selected_identities=target_identities(spec),
        gate_spec_snapshot=file_snapshot(gate_path),
    )
    config = load_config(config_path)
    paths = paths_for(config, args.run_id)
    if manifest["status"] != "completed":
        update_manifest(paths, gate_status="not_evaluated", gate_spec=file_snapshot(gate_path))
        update_manifest(paths, outputs=collect_output_hashes(paths))
        return load_json(paths.manifest)
    evaluation = evaluate_gate(
        spec,
        read_jsonl(paths.context / "records.jsonl"),
        load_json(paths.context / "summary.json"),
    )
    atomic_write_json(paths.audit / "gate-evaluation.json", evaluation)
    update_manifest(
        paths,
        gate_status=evaluation["status"],
        gate_spec=file_snapshot(gate_path),
    )
    update_manifest(paths, outputs=collect_output_hashes(paths))
    return load_json(paths.manifest)


def main() -> int:
    args = parse_args()
    try:
        manifest = run(args)
        print(json.dumps({"run_id": manifest["run_id"], "status": manifest["status"], "gate_status": manifest.get("gate_status")}, ensure_ascii=False))
        return 0 if manifest.get("gate_status") in {"passed", "not_evaluated"} else 1
    except Exception as error:
        print(f"RunBidirectionalInteractionGate failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
