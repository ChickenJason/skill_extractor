"""Render the active inline skill-annotation prompt for every source sentence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESERVED_TAGS = ("<skill>", "</skill>")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_prompt_template(path: Path) -> str:
    template = path.read_text(encoding="utf-8").rstrip("\r\n")
    if not template:
        raise ValueError(f"Prompt template is empty: {path}")
    if template.count("<sentence_json>") != 1:
        raise ValueError(
            "Prompt template must contain exactly one <sentence_json> placeholder"
        )
    return template


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def validate_records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("Dataset root must be a JSON list")

    indexes: set[int] = set()
    records: list[dict[str, Any]] = []
    for position, item in enumerate(value):
        if not isinstance(item, dict):
            raise TypeError(f"record[{position}] must be a JSON object")
        if {"idx", "sentence"} - set(item):
            raise ValueError(f"record[{position}] must contain idx and sentence")
        idx = item["idx"]
        sentence = item["sentence"]
        if not isinstance(idx, int):
            raise TypeError(f"record[{position}].idx must be an integer")
        if idx in indexes:
            raise ValueError(f"Duplicate idx: {idx}")
        if not isinstance(sentence, str) or not sentence.strip():
            raise ValueError(f"record[{position}].sentence must be non-empty")
        if any(tag in sentence for tag in RESERVED_TAGS):
            raise ValueError(
                f"record[{position}].sentence contains a reserved inline tag"
            )
        indexes.add(idx)
        records.append({"idx": idx, "sentence": sentence})
    return records


def render_prompt(template: str, sentence: str) -> str:
    if template.count("<sentence_json>") != 1:
        raise ValueError("Prompt template must contain exactly one <sentence_json> placeholder")
    sentence_json = json.dumps(sentence, ensure_ascii=False)
    return template.replace("<sentence_json>", sentence_json)


def generate_prompts(
    records: list[dict[str, Any]], template: str, version: str
) -> list[dict[str, Any]]:
    return [
        {
            "idx": item["idx"],
            "sentence": item["sentence"],
            "prompt": render_prompt(template, item["sentence"]),
            "prompt_version": version,
        }
        for item in records
    ]


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "self_annotator.json",
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    config = load_json(config_path)
    input_path = (
        resolve_project_path(args.input)
        if args.input is not None
        else resolve_project_path(config["input"])
    )
    prompt_config = config["prompt"]
    prompt_path = resolve_project_path(prompt_config["template"])
    records = validate_records(load_json(input_path))
    template = load_prompt_template(prompt_path)

    if args.validate_only:
        print(f"validated {len(records)} records from {input_path}")
        return
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")
        records = records[: args.limit]

    if args.output is None:
        raise ValueError("--output is required unless --validate-only is used")
    output_path = resolve_project_path(args.output)
    prompts = generate_prompts(records, template, prompt_config["version"])
    save_json(output_path, prompts)
    print(
        f"generated {len(prompts)} prompts from {prompt_path} -> {output_path}"
    )


if __name__ == "__main__":
    main()
