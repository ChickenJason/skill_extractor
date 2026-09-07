"""Stage 2: extract deterministic unigram TRF candidate banks."""

from __future__ import annotations

import argparse
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import atomic_write_json, atomic_write_jsonl, read_jsonl  # noqa: E402
from trf.offline.common import (  # noqa: E402
    TRFError,
    execute_stage,
    load_run,
    load_trf_config,
    parse_config_argument,
)


TOKEN_PATTERN = re.compile(r"(?u)(?<![\w.+#/-])(?:[^\W_]|[.+#/-])+(?![\w.+#/-])")


def clean_technical_token(token: str) -> str:
    """Keep technical punctuation only where it is lexical (plus C++/C# suffixes)."""

    return token.strip("-/").rstrip(".")


def remove_anonymous_placeholders(text: str, placeholder_pattern: str) -> str:
    """Replace placeholders with equal-length spaces so character positions stay stable."""

    pattern = re.compile(placeholder_pattern)
    return pattern.sub(lambda match: " " * (match.end() - match.start()), text)


def mask_skill_spans(sentence: str, spans: Iterable[dict[str, Any]]) -> str:
    characters = list(sentence)
    occupied: set[int] = set()
    for span in spans:
        start, end, text = span["start"], span["end"], span["text"]
        if sentence[start:end] != text:
            raise TRFError(f"Cannot mask invalid span {start}:{end}={text!r}")
        for position in range(start, end):
            if position in occupied:
                raise TRFError("Accepted skill spans overlap; context-only masking is ambiguous")
            occupied.add(position)
            characters[position] = " "
    masked = "".join(characters)
    if len(masked) != len(sentence):
        raise AssertionError("Context-only span masking changed sentence length")
    return masked


def tokenize_unigrams(
    text: str,
    *,
    placeholder_pattern: str,
    stopwords: frozenset[str],
    exclude_digits: bool,
) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = remove_anonymous_placeholders(normalized, placeholder_pattern)
    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(normalized):
        token = clean_technical_token(match.group(0))
        if not any(character.isalnum() for character in token):
            continue
        if exclude_digits and any(character.isdigit() for character in token):
            continue
        if token.lower() in stopwords:
            continue
        tokens.append(token)
    return tokens


def binary_mutual_information(
    positive_df: int,
    negative_df: int,
    positive_documents: int,
    negative_documents: int,
) -> float:
    """MI between binary word presence and the binary Skill class, in nats."""

    if not (0 <= positive_df <= positive_documents):
        raise ValueError("positive_df is outside its document range")
    if not (0 <= negative_df <= negative_documents):
        raise ValueError("negative_df is outside its document range")
    total = positive_documents + negative_documents
    cells = (
        (positive_df, positive_df + negative_df, positive_documents),
        (negative_df, positive_df + negative_df, negative_documents),
        (
            positive_documents - positive_df,
            total - positive_df - negative_df,
            positive_documents,
        ),
        (
            negative_documents - negative_df,
            total - positive_df - negative_df,
            negative_documents,
        ),
    )
    value = 0.0
    for cell, word_state_total, class_total in cells:
        if cell == 0:
            continue
        probability = cell / total
        value += probability * math.log((cell * total) / (word_state_total * class_total))
    return value


def _class_tokens(
    corpus: list[dict[str, Any]],
    *,
    context_only: bool,
    placeholder_pattern: str,
    stopwords: frozenset[str],
    exclude_digits: bool,
) -> tuple[list[list[str]], list[list[str]]]:
    positive: list[list[str]] = []
    negative: list[list[str]] = []
    for record in corpus:
        sentence = record["sentence"]
        if context_only and record["status"] == "accepted":
            sentence = mask_skill_spans(sentence, record["skill_spans"])
        tokens = tokenize_unigrams(
            sentence,
            placeholder_pattern=placeholder_pattern,
            stopwords=stopwords,
            exclude_digits=exclude_digits,
        )
        if record["status"] == "accepted":
            positive.append(tokens)
        elif record["status"] == "negative":
            negative.append(tokens)
        else:
            raise TRFError(f"Formal corpus contains unexpected status {record['status']!r}")
    return positive, negative


def calculate_candidate_metrics(
    positive_documents: list[list[str]],
    negative_documents: list[list[str]],
    rho: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    positive_raw = Counter(token for document in positive_documents for token in document)
    negative_raw = Counter(token for document in negative_documents for token in document)
    positive_df = Counter(token for document in positive_documents for token in set(document))
    negative_df = Counter(token for document in negative_documents for token in set(document))
    positive_count = len(positive_documents)
    negative_count = len(negative_documents)
    if positive_count == 0 or negative_count == 0:
        raise TRFError("Candidate statistics require both positive and negative documents")

    all_metrics: list[dict[str, Any]] = []
    for token in sorted(set(positive_raw) | set(negative_raw)):
        raw_positive = positive_raw[token]
        raw_negative = negative_raw[token]
        df_positive = positive_df[token]
        df_negative = negative_df[token]
        ratio = raw_negative / raw_positive if raw_positive > 0 else None
        positive_rate = df_positive / positive_count
        negative_rate = df_negative / negative_count
        paper_eligible = raw_positive > 0 and ratio is not None and ratio <= rho
        direction_gate = positive_rate > negative_rate
        all_metrics.append(
            {
                "text": token,
                "raw_count": {
                    "positive": raw_positive,
                    "negative": raw_negative,
                    "total": raw_positive + raw_negative,
                },
                "document_frequency": {
                    "positive": df_positive,
                    "negative": df_negative,
                    "total": df_positive + df_negative,
                },
                "mi": binary_mutual_information(
                    df_positive, df_negative, positive_count, negative_count
                ),
                "frequency_ratio_negative_to_positive": ratio,
                "normalized_document_rate": {
                    "positive": positive_rate,
                    "negative": negative_rate,
                },
                "paper_formula_eligible": paper_eligible,
                "direction_gate_passed": direction_gate,
                "paper_rank": None,
                "rank": None,
            }
        )

    def rank_key(record: dict[str, Any]) -> tuple[float, int, str]:
        return (
            -record["mi"],
            -record["document_frequency"]["positive"],
            record["text"],
        )

    paper = sorted(
        (record for record in all_metrics if record["paper_formula_eligible"]), key=rank_key
    )
    for rank, record in enumerate(paper, start=1):
        record["paper_rank"] = rank
    directional = sorted(
        (record for record in paper if record["direction_gate_passed"]), key=rank_key
    )
    for rank, record in enumerate(directional, start=1):
        record["rank"] = rank
    return all_metrics, paper, directional


def _load_stopwords(name: str) -> frozenset[str]:
    if name != "scikit-learn-english":
        raise TRFError(f"Unsupported stopword set: {name}")
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    except ImportError as error:
        raise TRFError("scikit-learn is required for the fixed English stopword set") from error
    return frozenset(ENGLISH_STOP_WORDS)


def extract_bank(
    corpus: list[dict[str, Any]], config: dict[str, Any], *, context_only: bool
) -> dict[str, Any]:
    token_config = config["tokenization"]
    stopwords = _load_stopwords(token_config["stopwords"])
    positive, negative = _class_tokens(
        corpus,
        context_only=context_only,
        placeholder_pattern=token_config["anonymous_placeholder_pattern"],
        stopwords=stopwords,
        exclude_digits=token_config["exclude_tokens_containing_digits"],
    )
    all_metrics, paper, directional = calculate_candidate_metrics(
        positive, negative, float(config["candidates"]["rho"])
    )
    max_trfs = int(config["candidates"]["max_trfs"])
    if len(paper) < max_trfs:
        raise TRFError(
            f"Paper-formula {'context' if context_only else 'main'} bank has fewer than "
            f"{max_trfs} candidates"
        )
    if len(directional) < max_trfs:
        raise TRFError(
            f"Directional {'context' if context_only else 'main'} bank has fewer than "
            f"{max_trfs} candidates"
        )
    return {
        "all": all_metrics,
        "paper": paper,
        "directional": directional,
        "selected": directional[:max_trfs],
        "document_counts": {"positive": len(positive), "negative": len(negative)},
    }


def write_candidate_bank(config: dict[str, Any], paths: Any) -> dict[str, Any]:
    corpus = read_jsonl(paths.corpus / "records.jsonl")
    expected_formal = config["source"]["expected_counts"]["formal"]
    if len(corpus) != expected_formal:
        raise TRFError(f"Formal corpus has {len(corpus)} rows; expected {expected_formal}")
    main = extract_bank(corpus, config, context_only=False)
    context = extract_bank(corpus, config, context_only=True)
    paths.candidates.mkdir(parents=True, exist_ok=False)

    for prefix, bank in (("main", main), ("context_only", context)):
        atomic_write_jsonl(paths.candidates / f"{prefix}_all_metrics.jsonl", bank["all"])
        atomic_write_jsonl(paths.candidates / f"{prefix}_paper_formula.jsonl", bank["paper"])
        atomic_write_jsonl(paths.candidates / f"{prefix}_directional.jsonl", bank["directional"])

    candidate_config = config["candidates"]
    common_contract = {
        "schema_version": "trf-candidate-bank-v1",
        "type": candidate_config["type"],
        "max_trfs": candidate_config["max_trfs"],
        "rho": candidate_config["rho"],
        "direction_gate": candidate_config["direction_gate"],
    }
    main_formal = {
        **common_contract,
        "trfs": [record["text"] for record in main["selected"]],
    }
    context_formal = {
        **common_contract,
        "context_only": True,
        "trfs": [record["text"] for record in context["selected"]],
    }
    selected = {
        **common_contract,
        "trfs": main_formal["trfs"],
        "context_only_trfs": context_formal["trfs"],
        "candidates": main["selected"],
        "context_only_candidates": context["selected"],
        "paper_formula_audit": {
            "trfs": [record["text"] for record in main["paper"][: candidate_config["max_trfs"]]],
            "context_only_trfs": [
                record["text"] for record in context["paper"][: candidate_config["max_trfs"]]
            ],
        },
    }
    atomic_write_json(paths.candidates / "trfs.json", main_formal)
    atomic_write_json(paths.candidates / "context_only_trfs.json", context_formal)
    atomic_write_json(paths.candidates / "selected_trfs.json", selected)
    summary = {
        "schema_version": "trf-candidate-summary-v1",
        "documents": main["document_counts"],
        "main": {
            "all_terms": len(main["all"]),
            "paper_eligible": len(main["paper"]),
            "directional_eligible": len(main["directional"]),
            "selected": main_formal["trfs"],
        },
        "context_only": {
            "all_terms": len(context["all"]),
            "paper_eligible": len(context["paper"]),
            "directional_eligible": len(context["directional"]),
            "selected": context_formal["trfs"],
        },
    }
    atomic_write_json(paths.candidates / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/trf.json")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--self-annotator-run-id", required=True)
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
            "extract_candidates",
            "build_corpus",
            lambda: write_candidate_bank(config, paths),
        )
    except Exception as error:
        print(f"ExtractTRFCandidates failed: {error}", file=sys.stderr)
        return 1
    print(f"TRF candidate extraction completed: {paths.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
