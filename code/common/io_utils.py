"""Deterministic JSON, JSONL, hashing, and configuration helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ENVIRONMENT_REFERENCE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class MissingEnvironmentVariable(ValueError):
    """Raised when a required ${NAME} configuration value is unavailable."""


@dataclass(frozen=True)
class SelfAnnotationRunPaths:
    """Canonical directories and metadata files for one self-annotation run."""

    root: Path
    raw: Path
    parsed: Path
    aggregated: Path
    selected: Path
    manifest: Path
    log: Path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise TypeError(f"JSONL record at {path}:{line_number} must be an object")
            records.append(value)
    return records


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    lines = [json.dumps(record, ensure_ascii=False, separators=(",", ":")) for record in records]
    _atomic_write(path, "\n".join(lines) + ("\n" if lines else ""))


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def expand_environment_references(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_environment_references(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_environment_references(item) for item in value]
    if not isinstance(value, str):
        return value

    match = ENVIRONMENT_REFERENCE.fullmatch(value)
    if not match:
        return value
    name = match.group(1)
    resolved = os.getenv(name)
    if not resolved:
        raise MissingEnvironmentVariable(f"Required environment variable is not set: {name}")
    return resolved


def redact_secrets(value: Any, secrets: Iterable[str]) -> Any:
    secret_values = [secret for secret in secrets if secret]
    if isinstance(value, dict):
        return {key: redact_secrets(item, secret_values) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_secrets(item, secret_values) for item in value]
    if isinstance(value, str):
        redacted = value
        for secret in secret_values:
            redacted = redacted.replace(secret, "<redacted>")
        return redacted
    return value


def validate_run_id(value: str) -> str:
    if not SAFE_RUN_ID.fullmatch(value):
        raise ValueError(
            "run-id must start with a letter or digit and contain only letters, "
            "digits, dots, underscores, or hyphens"
        )
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_run_id(prefix: str = "qwen") -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{timestamp}"


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def build_self_annotation_run_paths(
    project_root: Path,
    output_config: dict[str, Any],
    run_id: str,
) -> SelfAnnotationRunPaths:
    """Build the run-first output layout from the configured runs root."""

    runs_root = output_config.get("runs_root")
    if not isinstance(runs_root, str) or not runs_root.strip():
        raise ValueError("output.runs_root must be a non-empty string")
    root = resolve_project_path(project_root, runs_root) / validate_run_id(run_id)
    return SelfAnnotationRunPaths(
        root=root,
        raw=root / "raw",
        parsed=root / "parsed",
        aggregated=root / "aggregated",
        selected=root / "selected",
        manifest=root / "manifest.json",
        log=root / "run.log",
    )
