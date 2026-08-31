"""Parse and strictly validate inline skill-span annotations."""

from __future__ import annotations

import argparse
import difflib
import json
import re
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
OPEN_TAG = "<skill>"
CLOSE_TAG = "</skill>"
JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
PARSER_VERSION = "inline-parser-v5.0"


def _json_object(value: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _balanced_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(text):
        if start is None:
            if character == "{":
                start = index
                depth = 1
            continue
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                objects.append(text[start : index + 1])
                start = None
    return objects


def parse_json_response(content: str) -> tuple[dict[str, Any], str]:
    value, status, _ = parse_json_response_detailed(content)
    return value, status


def parse_json_response_detailed(
    content: str,
) -> tuple[dict[str, Any], str, str | None]:
    stripped = content.strip()
    direct = _json_object(stripped)
    if direct is not None:
        return direct, "ok", None

    fenced = [
        parsed
        for block in JSON_FENCE.findall(content)
        if (parsed := _json_object(block.strip())) is not None
    ]
    if len(fenced) == 1:
        return fenced[0], "recovered", "json_code_fence"
    if len(fenced) > 1:
        raise ValueError("Response contains multiple JSON objects in code fences")

    recovered = [
        parsed
        for candidate in _balanced_json_objects(content)
        if (parsed := _json_object(candidate)) is not None
    ]
    if len(recovered) == 1:
        return recovered[0], "recovered", "embedded_json_object"
    if len(recovered) > 1:
        raise ValueError("Response contains multiple JSON objects")
    raise ValueError("Response does not contain a valid JSON object")


def scan_inline_tags(
    annotated_sentence: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Validate tag structure and return payloads in reconstructed-text offsets."""

    reconstructed: list[str] = []
    raw_spans: list[tuple[int, int]] = []
    active_start: int | None = None
    cursor = 0

    while cursor < len(annotated_sentence):
        if annotated_sentence.startswith(OPEN_TAG, cursor):
            if active_start is not None:
                raise ValueError("Skill tags cannot be nested or overlapping")
            active_start = len(reconstructed)
            cursor += len(OPEN_TAG)
            continue
        if annotated_sentence.startswith(CLOSE_TAG, cursor):
            if active_start is None:
                raise ValueError("Closing skill tag has no matching opening tag")
            end = len(reconstructed)
            if end == active_start:
                raise ValueError("Skill tags cannot be empty")
            raw_spans.append((active_start, end))
            active_start = None
            cursor += len(CLOSE_TAG)
            continue
        reconstructed.append(annotated_sentence[cursor])
        cursor += 1

    if active_start is not None:
        raise ValueError("Opening skill tag has no matching closing tag")

    restored_sentence = "".join(reconstructed)
    payloads = [
        {"text": restored_sentence[start:end], "start": start, "end": end}
        for start, end in raw_spans
    ]
    return restored_sentence, payloads


def extract_inline_spans(
    annotated_sentence: str, sentence: str
) -> tuple[str, list[dict[str, Any]]]:
    """Remove inline tags, prove exact reconstruction, and return source offsets."""

    restored_sentence, spans = scan_inline_tags(annotated_sentence)
    if restored_sentence != sentence:
        raise ValueError(
            "Removing skill tags must reconstruct the original sentence exactly"
        )
    return restored_sentence, spans


def _unique_monotonic_payload_assignment(
    sentence: str, payloads: list[str]
) -> list[tuple[int, int]] | None:
    """Return the only ordered non-overlapping exact assignment, otherwise None."""

    solutions: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []

    def search(position: int, minimum_start: int) -> None:
        if len(solutions) > 1:
            return
        if position == len(payloads):
            solutions.append(list(current))
            return
        payload = payloads[position]
        start = sentence.find(payload, minimum_start)
        while start >= 0:
            end = start + len(payload)
            current.append((start, end))
            search(position + 1, end)
            current.pop()
            if len(solutions) > 1:
                return
            start = sentence.find(payload, start + 1)

    search(0, 0)
    return solutions[0] if len(solutions) == 1 else None


def _source_edit_summary(source: str, reconstructed: str) -> dict[str, Any]:
    operations = [
        (operation, source_start, source_end, model_start, model_end)
        for operation, source_start, source_end, model_start, model_end in difflib.SequenceMatcher(
            a=source, b=reconstructed
        ).get_opcodes()
        if operation != "equal"
    ]
    operation_counts: dict[str, int] = {}
    for operation, *_ in operations:
        operation_counts[operation] = operation_counts.get(operation, 0) + 1
    return {
        "operation_count": len(operations),
        "operation_counts": operation_counts,
        "source_characters_changed": sum(end - start for _, start, end, _, _ in operations),
        "model_characters_changed": sum(end - start for _, _, _, start, end in operations),
    }


def recover_exact_tag_payloads(
    annotated_sentence: str, sentence: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Project unchanged tag payloads onto their unique exact source positions."""

    reconstructed, tagged_spans = scan_inline_tags(annotated_sentence)
    if reconstructed == sentence:
        raise ValueError("Exact reconstruction does not require payload projection")
    if not tagged_spans:
        raise ValueError("A modified negative response cannot be recovered safely")

    payloads = [span["text"] for span in tagged_spans]
    assignment = _unique_monotonic_payload_assignment(sentence, payloads)
    if assignment is None:
        raise ValueError(
            "Tagged payloads do not have one unique ordered exact mapping in the source sentence"
        )
    spans = [
        {"text": payload, "start": start, "end": end}
        for payload, (start, end) in zip(payloads, assignment)
    ]
    return spans, _source_edit_summary(sentence, reconstructed)


def validate_inline_annotation(
    value: dict[str, Any], sentence: str
) -> dict[str, Any]:
    if set(value) != {"annotated_sentence"}:
        missing = {"annotated_sentence"} - set(value)
        unexpected = set(value) - {"annotated_sentence"}
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {sorted(missing)}")
        if unexpected:
            details.append(f"unexpected fields: {sorted(unexpected)}")
        raise ValueError("Inline annotation has " + "; ".join(details))

    annotated_sentence = value["annotated_sentence"]
    if not isinstance(annotated_sentence, str):
        raise TypeError("annotated_sentence must be a string")
    _, spans = extract_inline_spans(annotated_sentence, sentence)
    return {
        "has_skill": 1 if spans else 0,
        "spans": spans,
    }


def parse_response_content_detailed(content: Any, sentence: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "annotation": None,
        "parse_status": "failed",
        "parse_error": None,
        "recovery_methods": [],
        "reconstruction_status": None,
        "source_edit_summary": None,
    }
    if not isinstance(content, str) or not content.strip():
        result["parse_error"] = "Response content is empty"
        return result
    try:
        value, json_status, json_recovery = parse_json_response_detailed(content)
        if set(value) != {"annotated_sentence"}:
            missing = {"annotated_sentence"} - set(value)
            unexpected = set(value) - {"annotated_sentence"}
            details: list[str] = []
            if missing:
                details.append(f"missing fields: {sorted(missing)}")
            if unexpected:
                details.append(f"unexpected fields: {sorted(unexpected)}")
            raise ValueError("Inline annotation has " + "; ".join(details))
        annotated_sentence = value["annotated_sentence"]
        if not isinstance(annotated_sentence, str):
            raise TypeError("annotated_sentence must be a string")

        methods = [json_recovery] if json_recovery else []
        try:
            _, spans = extract_inline_spans(annotated_sentence, sentence)
            reconstruction_status = "exact"
            edit_summary = None
        except ValueError as exact_error:
            if str(exact_error) != (
                "Removing skill tags must reconstruct the original sentence exactly"
            ):
                raise
            spans, edit_summary = recover_exact_tag_payloads(
                annotated_sentence, sentence
            )
            methods.append("exact_tag_payload_projection")
            reconstruction_status = "projected"

        result.update(
            {
                "annotation": {"has_skill": 1 if spans else 0, "spans": spans},
                "parse_status": "recovered" if methods or json_status == "recovered" else "ok",
                "parse_error": None,
                "recovery_methods": methods,
                "reconstruction_status": reconstruction_status,
                "source_edit_summary": edit_summary,
            }
        )
    except (TypeError, ValueError) as error:
        result["parse_error"] = str(error)
    return result


def parse_response_content(
    content: Any, sentence: str
) -> tuple[dict[str, Any] | None, str, str | None]:
    result = parse_response_content_detailed(content, sentence)
    return result["annotation"], result["parse_status"], result["parse_error"]


def latest_raw_records(path: Path) -> list[dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    for record in read_jsonl(path):
        idx = record.get("idx")
        if isinstance(idx, int) and not isinstance(idx, bool):
            latest[idx] = record
    return [latest[idx] for idx in sorted(latest)]


def parse_raw_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parsed_records: list[dict[str, Any]] = []
    for record in records:
        idx = record.get("idx")
        sentence = record.get("sentence")
        samples = record.get("samples")
        if not isinstance(idx, int) or isinstance(idx, bool):
            raise TypeError("Raw response idx must be an integer")
        if not isinstance(sentence, str):
            raise TypeError(f"Raw response sentence must be a string for idx={idx}")
        if not isinstance(samples, list):
            raise TypeError(f"Raw response samples must be a list for idx={idx}")

        for sample in samples:
            sample_index = sample.get("sample_index")
            if sample.get("status") != "ok":
                parsed = {
                    "annotation": None,
                    "parse_status": "failed",
                    "parse_error": sample.get("error", {}).get(
                        "message", "Model request failed"
                    ),
                    "recovery_methods": [],
                    "reconstruction_status": None,
                    "source_edit_summary": None,
                }
            else:
                parsed = parse_response_content_detailed(
                    sample.get("content"), sentence
                )
            annotation = parsed["annotation"]
            parse_status = parsed["parse_status"]
            parse_error = parsed["parse_error"]
            if annotation is not None:
                annotation = {
                    "has_skill": annotation["has_skill"],
                    "spans": [
                        {
                            **span,
                            "label_id": (
                                f"span:{idx}:{span['start']}:{span['end']}"
                            ),
                        }
                        for span in annotation["spans"]
                    ],
                }
            parsed_records.append(
                {
                    "idx": idx,
                    "sentence": sentence,
                    "prompt_version": record.get("prompt_version"),
                    "model": record.get("model"),
                    "sample_index": sample_index,
                    "parse_status": parse_status,
                    "parse_error": parse_error,
                    "parser_version": PARSER_VERSION,
                    "recovery_methods": parsed["recovery_methods"],
                    "reconstruction_status": parsed["reconstruction_status"],
                    "source_edit_summary": parsed["source_edit_summary"],
                    "source_format": "inline_tags",
                    "annotation": annotation,
                }
            )
    return parsed_records


def build_parse_summary(parsed_records: list[dict[str, Any]]) -> dict[str, Any]:
    methods = sorted(
        {
            method
            for record in parsed_records
            for method in record.get("recovery_methods", [])
        }
    )
    errors: dict[str, int] = {}
    for record in parsed_records:
        if record.get("parse_status") != "failed":
            continue
        error = str(record.get("parse_error") or "unknown")
        errors[error] = errors.get(error, 0) + 1
    return {
        "updated_at": utc_now(),
        "annotation_schema": "skill-inline-span-v1",
        "parser_version": PARSER_VERSION,
        "records": len(parsed_records),
        "statuses": {
            status: sum(record["parse_status"] == status for record in parsed_records)
            for status in ("ok", "recovered", "failed")
        },
        "recovery_methods": {
            method: sum(
                method in record.get("recovery_methods", [])
                for record in parsed_records
            )
            for method in methods
        },
        "failure_reasons": errors,
    }


def _validate_active_run_schema(config: dict[str, Any], manifest: dict[str, Any]) -> None:
    expected = config.get("annotation_schema")
    actual = manifest.get("compatibility", {}).get("annotation_schema")
    if expected != "skill-inline-span-v1" or actual != expected:
        raise ValueError(
            "Run manifest is not compatible with the active skill-inline-span-v1 pipeline"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "pipeline.json",
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_id = validate_run_id(args.run_id)
    config_path = resolve_project_path(PROJECT_ROOT, args.config)
    config = load_json(config_path)
    paths = build_self_annotation_run_paths(PROJECT_ROOT, config["output"], run_id)
    _validate_active_run_schema(config, load_json(paths.manifest))
    input_path = (
        resolve_project_path(PROJECT_ROOT, args.input)
        if args.input is not None
        else paths.raw / "responses.jsonl"
    )
    output_path = (
        resolve_project_path(PROJECT_ROOT, args.output)
        if args.output is not None
        else paths.parsed / "samples.jsonl"
    )
    parsed_records = parse_raw_records(latest_raw_records(input_path))
    atomic_write_jsonl(output_path, parsed_records)
    summary = build_parse_summary(parsed_records)
    summary["annotation_schema"] = config["annotation_schema"]
    atomic_write_json(output_path.parent / "summary.json", summary)
    print(f"parsed {len(parsed_records)} inline samples -> {output_path}")


if __name__ == "__main__":
    main()
