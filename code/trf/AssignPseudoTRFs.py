"""Stage 3: assign token-level nearest BERT TRFs to positive demonstrations."""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import os
import random
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
    load_json,
    read_jsonl,
    sha256_file,
)
from trf.ExtractTRFCandidates import TOKEN_PATTERN, clean_technical_token  # noqa: E402
from trf.common import (  # noqa: E402
    TRFError,
    execute_stage,
    finalize_run,
    load_run,
    load_trf_config,
    parse_config_argument,
    read_manifest,
    write_manifest,
)


EXPECTED_DISTRIBUTIONS = {
    "numpy": "1.26.4",
    "scikit-learn": "1.6.1",
    "torch": "2.5.1+cpu",
    "transformers": "4.57.1",
}


def validate_runtime_environment() -> dict[str, str]:
    if sys.version_info[:2] != (3, 10):
        raise TRFError(
            f"TRF Stage 3 requires Python 3.10 exactly; running {sys.version.split()[0]}"
        )
    versions: dict[str, str] = {}
    for distribution, expected in EXPECTED_DISTRIBUTIONS.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise TRFError(f"Required distribution is missing: {distribution}=={expected}") from error
        if actual != expected:
            raise TRFError(
                f"Dependency mismatch for {distribution}: expected {expected}, got {actual}"
            )
        versions[distribution] = actual

    import torch

    if torch.__version__ != "2.5.1+cpu":
        raise TRFError(
            f"TRF Stage 3 requires the CPU torch build 2.5.1+cpu; got {torch.__version__}"
        )
    versions["torch_runtime"] = torch.__version__
    versions["python"] = sys.version.split()[0]
    return versions


def configure_determinism(seed: int) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # It can only be set once in an interpreter; the existing value is harmless on CPU.
        pass
    torch.use_deterministic_algorithms(True)


def _repo_cache_directory(model_name: str) -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE) / f"models--{model_name.replace('/', '--')}"


def resolve_model_snapshot(model: dict[str, Any], allow_download: bool) -> Path:
    model_name = model["name"]
    revision = model["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise TRFError("BERT revision must be a full 40-character commit hash")
    snapshot = _repo_cache_directory(model_name) / "snapshots" / revision

    if allow_download:
        from huggingface_hub import snapshot_download

        resolved = Path(
            snapshot_download(
                repo_id=model_name,
                revision=revision,
                allow_patterns=[
                    "config.json",
                    "model.safetensors",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.txt",
                ],
                local_files_only=False,
            )
        ).resolve()
        if resolved.name != revision:
            raise TRFError(f"Downloaded snapshot is not the pinned revision: {resolved}")
        snapshot = resolved

    if not snapshot.is_dir():
        raise TRFError(
            "Pinned bert-base-cased snapshot is absent. Re-run a new run-id with "
            "-AllowModelDownload to download the fixed revision."
        )
    if not (snapshot / "model.safetensors").is_file():
        raise TRFError(
            "Pinned BERT tokenizer/config is cached but model.safetensors is missing. "
            "Use -AllowModelDownload with a new run-id."
        )
    return snapshot


def hash_weight_files(snapshot: Path) -> list[dict[str, Any]]:
    weight_path = snapshot / "model.safetensors"
    if not weight_path.is_file():
        raise TRFError("No BERT weight file found in the pinned snapshot")
    return [{
        "file": weight_path.name,
        "sha256": sha256_file(weight_path),
        "bytes": weight_path.stat().st_size,
    }]


def load_pinned_bert(snapshot: Path, expected_hidden_size: int) -> tuple[Any, Any]:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), use_fast=True, local_files_only=True
    )
    if not tokenizer.is_fast:
        raise TRFError("bert-base-cased fast tokenizer is required for offset mapping")
    model = AutoModel.from_pretrained(
        str(snapshot), local_files_only=True, use_safetensors=True
    )
    if int(model.config.hidden_size) != expected_hidden_size:
        raise TRFError(
            f"BERT hidden size mismatch: expected {expected_hidden_size}, "
            f"got {model.config.hidden_size}"
        )
    model.to("cpu")
    model.eval()
    return tokenizer, model


def lexical_tokens_with_offsets(
    sentence: str, placeholder_pattern: str
) -> list[dict[str, int | str]]:
    placeholder_ranges = [
        (match.start(), match.end()) for match in re.finditer(placeholder_pattern, sentence)
    ]
    tokens: list[dict[str, int | str]] = []
    for match in TOKEN_PATTERN.finditer(sentence):
        start, end = match.span()
        raw_text = match.group(0)
        text = clean_technical_token(raw_text)
        leading = len(raw_text) - len(raw_text.lstrip("-/"))
        start += leading
        end = start + len(text)
        if not any(character.isalnum() for character in text):
            continue
        if any(start < placeholder_end and end > placeholder_start for placeholder_start, placeholder_end in placeholder_ranges):
            continue
        tokens.append({"text": text, "start": start, "end": end})
    return tokens


def _ensure_model_length(input_ids: Any, maximum: int, description: str) -> None:
    length = int(input_ids.shape[1])
    if length > maximum:
        raise TRFError(
            f"{description} encodes to {length} subwords including special tokens; "
            f"maximum is {maximum}; truncation is forbidden"
        )


def embed_candidate(text: str, tokenizer: Any, model: Any, maximum: int) -> Any:
    import numpy as np
    import torch

    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        return_special_tokens_mask=True,
        truncation=False,
    )
    _ensure_model_length(encoded["input_ids"], maximum, f"Candidate {text!r}")
    special_mask = encoded.pop("special_tokens_mask")[0].bool()
    with torch.inference_mode():
        hidden = model(**encoded).last_hidden_state[0]
    usable = hidden[~special_mask]
    if usable.shape[0] == 0:
        raise TRFError(f"Candidate {text!r} has no non-special BERT subwords")
    vector = usable.mean(dim=0).cpu().numpy().astype(np.float32, copy=False)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise TRFError(f"Candidate {text!r} produced an invalid embedding")
    return vector


def embed_sentence_tokens(
    sentence: str,
    tokenizer: Any,
    model: Any,
    maximum: int,
    placeholder_pattern: str,
) -> list[dict[str, Any]]:
    import numpy as np
    import torch

    words = lexical_tokens_with_offsets(sentence, placeholder_pattern)
    if not words:
        raise TRFError("Positive demonstration has no eligible lexical tokens")
    encoded = tokenizer(
        sentence,
        return_tensors="pt",
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        truncation=False,
    )
    _ensure_model_length(encoded["input_ids"], maximum, "Demonstration sentence")
    offsets = encoded.pop("offset_mapping")[0].tolist()
    special_mask = encoded.pop("special_tokens_mask")[0].bool()
    with torch.inference_mode():
        hidden = model(**encoded).last_hidden_state[0]

    aligned: list[dict[str, Any]] = []
    for word in words:
        indexes = [
            index
            for index, (start, end) in enumerate(offsets)
            if not bool(special_mask[index])
            and start < int(word["end"])
            and end > int(word["start"])
        ]
        if not indexes:
            raise TRFError(
                f"BERT offset mapping did not align token {word['text']!r} at "
                f"{word['start']}:{word['end']}"
            )
        vector = hidden[indexes].mean(dim=0).cpu().numpy().astype(np.float32, copy=False)
        if not np.isfinite(vector).all():
            raise TRFError(f"Token {word['text']!r} produced a non-finite embedding")
        aligned.append({**word, "vector": vector})
    return aligned


def rank_nearest_candidates(
    candidates: list[dict[str, Any]],
    candidate_vectors: dict[str, Any],
    sentence_tokens: list[dict[str, Any]],
    top_k: int,
    decimals: int,
) -> list[dict[str, Any]]:
    import numpy as np

    if len(candidates) < top_k:
        raise TRFError(f"Candidate bank has fewer than required Top-{top_k}")
    ranked: list[tuple[float, int, str, dict[str, Any]]] = []
    for candidate in candidates:
        text = candidate["text"]
        candidate_rank = int(candidate["rank"])
        vector = candidate_vectors[text]
        nearest: tuple[float, dict[str, Any]] | None = None
        for token in sentence_tokens:
            distance = float(np.linalg.norm(vector - token["vector"]))
            if not math.isfinite(distance):
                raise TRFError(f"Non-finite distance for candidate {text!r}")
            if nearest is None or distance < nearest[0]:
                nearest = (distance, token)
        assert nearest is not None
        distance, token = nearest
        ranked.append((distance, candidate_rank, text, token))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    output: list[dict[str, Any]] = []
    for rank, (distance, candidate_rank, text, token) in enumerate(ranked[:top_k], start=1):
        start, end = int(token["start"]), int(token["end"])
        output.append(
            {
                "text": text,
                "rank": rank,
                "distance": round(distance, decimals),
                "nearest_token": token["text"],
                "nearest_token_start": start,
                "nearest_token_end": end,
                "candidate_rank": candidate_rank,
            }
        )
    return output


def _validate_selected_candidates(selected: dict[str, Any], maximum: int) -> None:
    for list_key, details_key in (
        ("trfs", "candidates"),
        ("context_only_trfs", "context_only_candidates"),
    ):
        names = selected.get(list_key)
        details = selected.get(details_key)
        if not isinstance(names, list) or len(names) != maximum or len(set(names)) != maximum:
            raise TRFError(f"{list_key} must contain exactly {maximum} unique strings")
        if not isinstance(details, list) or [item["text"] for item in details] != names:
            raise TRFError(f"{details_key} does not match {list_key}")
        if [item["rank"] for item in details] != list(range(1, maximum + 1)):
            raise TRFError(f"{details_key} ranks are not contiguous from 1")


def assign_pseudo_trfs(
    config: dict[str, Any], paths: Any, allow_model_download: bool
) -> dict[str, Any]:
    versions = validate_runtime_environment()
    model_config = config["model"]
    configure_determinism(int(model_config["seed"]))

    if allow_model_download:
        manifest = read_manifest(paths)
        manifest["network_called"] = True
        manifest["network_policy"] = "pinned-model-download-allowed"
        write_manifest(paths, manifest)
    snapshot = resolve_model_snapshot(model_config, allow_model_download)
    weights = hash_weight_files(snapshot)
    tokenizer, model = load_pinned_bert(snapshot, int(model_config["hidden_size"]))

    manifest = read_manifest(paths)
    manifest["dependencies"].update(versions)
    manifest["model"].update(
        {
            "snapshot_path": str(snapshot),
            "hidden_size": int(model.config.hidden_size),
            "weight_files": weights,
        }
    )
    write_manifest(paths, manifest)

    corpus = read_jsonl(paths.corpus / "records.jsonl")
    selected = load_json(paths.candidates / "selected_trfs.json")
    maximum = int(config["candidates"]["max_trfs"])
    _validate_selected_candidates(selected, maximum)
    main_candidates = selected["candidates"]
    context_candidates = selected["context_only_candidates"]
    all_candidate_names = sorted(set(selected["trfs"]) | set(selected["context_only_trfs"]))
    candidate_vectors = {
        text: embed_candidate(text, tokenizer, model, int(model_config["max_subwords"]))
        for text in all_candidate_names
    }
    import numpy as np

    for text, vector in candidate_vectors.items():
        if vector.shape != (int(model_config["hidden_size"]),) or not np.isfinite(vector).all():
            raise TRFError(f"Candidate vector contract failed for {text!r}")

    top_k = int(model_config["top_k"])
    decimals = int(model_config["distance_decimals"])
    placeholder_pattern = config["tokenization"]["anonymous_placeholder_pattern"]
    model_metadata = {
        "name": model_config["name"],
        "revision": model_config["revision"],
        "hidden_size": int(model.config.hidden_size),
    }
    outputs: list[dict[str, Any]] = []
    accepted_count = 0
    negative_count = 0
    for record in corpus:
        base = {
            "idx": record["idx"],
            "status": record["status"],
            "type": "Skill",
        }
        if record["status"] == "accepted":
            accepted_count += 1
            sentence_tokens = embed_sentence_tokens(
                record["sentence"],
                tokenizer,
                model,
                int(model_config["max_subwords"]),
                placeholder_pattern,
            )
            trfs = rank_nearest_candidates(
                main_candidates, candidate_vectors, sentence_tokens, top_k, decimals
            )
            context_trfs = rank_nearest_candidates(
                context_candidates, candidate_vectors, sentence_tokens, top_k, decimals
            )
        elif record["status"] == "negative":
            negative_count += 1
            trfs = []
            context_trfs = []
        else:
            raise TRFError(f"Unexpected formal corpus status: {record['status']!r}")
        outputs.append(
            {
                **base,
                "trfs": trfs,
                "context_only_trfs": context_trfs,
                "model": model_metadata,
            }
        )

    expected = config["source"]["expected_counts"]
    if accepted_count != expected["accepted"] or negative_count != expected["negative"]:
        raise TRFError("Pseudo-TRF output class counts do not match the fixed corpus contract")
    _validate_pseudo_outputs(outputs, corpus, selected, config)

    paths.demonstrations.mkdir(parents=True, exist_ok=False)
    atomic_write_jsonl(paths.demonstrations / "pseudo_trfs.jsonl", outputs)
    atomic_write_json(
        paths.demonstrations / "model_files.json",
        {
            "model": model_metadata,
            "snapshot_path": str(snapshot),
            "weight_files": weights,
        },
    )
    review_categories = {
        item["idx"]: item["review_category"]
        for item in read_jsonl(paths.audit / "manual_review_sample.jsonl")
    }
    review_output = [
        {"review_category": review_categories[item["idx"]], **item}
        for item in outputs
        if item["idx"] in review_categories
    ]
    atomic_write_jsonl(paths.audit / "semantic_review_sample.jsonl", review_output)
    summary = {
        "schema_version": "trf-pseudo-label-summary-v1",
        "counts": {"accepted": accepted_count, "negative": negative_count},
        "top_k": top_k,
        "main_assignments": accepted_count * top_k,
        "context_only_assignments": accepted_count * top_k,
        "semantic_acceptance": {
            "status": "pending_manual_review",
            "passed": False,
            "sample_size": len(review_output),
        },
    }
    atomic_write_json(paths.demonstrations / "summary.json", summary)
    return summary


def _validate_pseudo_outputs(
    outputs: list[dict[str, Any]],
    corpus: list[dict[str, Any]],
    selected: dict[str, Any],
    config: dict[str, Any],
) -> None:
    if len(outputs) != len(corpus):
        raise TRFError("Pseudo-TRF output row count differs from the corpus")
    main_names = set(selected["trfs"])
    context_names = set(selected["context_only_trfs"])
    top_k = int(config["model"]["top_k"])
    corpus_by_idx = {record["idx"]: record for record in corpus}
    for output in outputs:
        source = corpus_by_idx.get(output["idx"])
        if source is None or source["status"] != output["status"]:
            raise TRFError(f"Pseudo-TRF record does not match corpus idx={output['idx']}")
        if output["status"] == "negative":
            if output["trfs"] or output["context_only_trfs"]:
                raise TRFError(f"Negative idx={output['idx']} must have empty TRF sets")
            continue
        if len(output["trfs"]) != top_k or len(output["context_only_trfs"]) != top_k:
            raise TRFError(f"Accepted idx={output['idx']} must have two Top-{top_k} lists")
        for key, allowed in (("trfs", main_names), ("context_only_trfs", context_names)):
            ranks = [item["rank"] for item in output[key]]
            if ranks != list(range(1, top_k + 1)):
                raise TRFError(f"idx={output['idx']} {key} ranks are invalid")
            for item in output[key]:
                if item["text"] not in allowed or not math.isfinite(item["distance"]):
                    raise TRFError(f"idx={output['idx']} contains an invalid {key} candidate")
                start, end = item["nearest_token_start"], item["nearest_token_end"]
                if source["sentence"][start:end] != item["nearest_token"]:
                    raise TRFError(f"idx={output['idx']} nearest-token offset is invalid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/trf.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--self-annotator-run-id", required=True)
    parser.add_argument("--allow-model-download", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = parse_config_argument(args.config)
    try:
        config = load_trf_config(config_path, args.self_annotator_run_id)
        paths = load_run(config, args.run_id, sys.argv)
        execute_stage(
            config,
            paths,
            "assign_pseudo_trfs",
            "extract_candidates",
            lambda: assign_pseudo_trfs(config, paths, args.allow_model_download),
        )
        finalize_run(config, paths)
    except Exception as error:
        print(f"AssignPseudoTRFs failed: {error}", file=sys.stderr)
        return 1
    print(f"TRF pseudo-label assignment completed: {paths.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
