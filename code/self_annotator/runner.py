"""Single entry point for prompt generation and the self-annotation stages."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import build_self_annotation_run_paths, load_json, resolve_project_path  # noqa: E402
from self_annotator.generate_prompts import (  # noqa: E402
    generate_prompts,
    load_prompt_template,
    save_json,
    validate_records,
)
from self_annotator.validator import validate  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/self_annotator.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def _run_stage(path: Path, arguments: list[str]) -> None:
    completed = subprocess.run([sys.executable, str(path), *arguments], check=False)
    if completed.returncode:
        raise RuntimeError(f"Stage failed with exit code {completed.returncode}: {path.name}")


def run(args: argparse.Namespace) -> dict[str, object]:
    config_path = resolve_project_path(PROJECT_ROOT, args.config).resolve()
    config = load_json(config_path)
    input_path = resolve_project_path(
        PROJECT_ROOT, args.input or config["input"]
    ).resolve()
    records = validate_records(load_json(input_path))
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")
        records = records[: args.limit]
    template = load_prompt_template(
        resolve_project_path(PROJECT_ROOT, config["prompt"]["template"])
    )
    if args.validate_only:
        return {"status": "validated", "records": len(records), "input": str(input_path)}

    paths = build_self_annotation_run_paths(PROJECT_ROOT, config["output"], args.run_id)
    prompt_path = paths.prompts / "records.json"
    if not args.resume:
        prompts = generate_prompts(records, template, config["prompt"]["version"])
        save_json(prompt_path, prompts)
    elif not prompt_path.is_file():
        raise FileNotFoundError(f"Cannot resume without prompt snapshot: {prompt_path}")

    shared = ["--config", str(config_path), "--run-id", args.run_id]
    ask = [
        *shared,
        "--input",
        str(prompt_path),
        "--source-input",
        str(input_path),
    ]
    if args.limit is not None:
        ask.extend(["--limit", str(args.limit)])
    if args.resume:
        ask.append("--resume")
    if args.fail_fast:
        ask.append("--fail-fast")
    _run_stage(Path(__file__).with_name("ask_qwen.py"), ask)
    for stage in ("parse_answers.py", "consensus.py", "selection.py"):
        _run_stage(Path(__file__).with_name(stage), shared)
    return validate(config_path, args.run_id)


def main() -> int:
    args = parse_args()
    try:
        result = run(args)
        print(result)
        return 0
    except Exception as error:
        print(f"Self annotator failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
