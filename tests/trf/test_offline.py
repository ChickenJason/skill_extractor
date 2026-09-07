from __future__ import annotations

import copy
import math
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import (  # noqa: E402
    atomic_write_json,
    load_json,
    read_jsonl,
    sha256_file,
)
from trf.offline.assign_pseudo_trfs import (  # noqa: E402
    _repo_cache_directory,
    embed_candidate,
    embed_sentence_tokens,
    lexical_tokens_with_offsets,
    load_pinned_bert,
    rank_nearest_candidates,
    resolve_model_snapshot,
)
from trf.offline.build_corpus import build_corpus_records, validate_span  # noqa: E402
from trf.offline.extract_candidates import (  # noqa: E402
    _class_tokens,
    binary_mutual_information,
    calculate_candidate_metrics,
    extract_bank,
    mask_skill_spans,
    tokenize_unigrams,
)
from trf.offline.common import (  # noqa: E402
    TRFError,
    execute_stage,
    initialize_run,
    load_trf_config,
    snapshot_source,
    source_files,
)
from trf.offline.build_corpus import write_corpus  # noqa: E402
from trf.offline.extract_candidates import write_candidate_bank  # noqa: E402


CONFIG = load_json(PROJECT_ROOT / "config" / "trf.json")

from tests.trf.fixtures import prepare_v06_trf_config  # noqa: E402


_SOURCE_WORKSPACE: tempfile.TemporaryDirectory[str] | None = None


def setUpModule() -> None:
    global _SOURCE_WORKSPACE
    _SOURCE_WORKSPACE = tempfile.TemporaryDirectory()
    _, config = prepare_v06_trf_config(Path(_SOURCE_WORKSPACE.name))
    CONFIG.clear()
    CONFIG.update(config)


def tearDownModule() -> None:
    global _SOURCE_WORKSPACE
    if _SOURCE_WORKSPACE is not None:
        _SOURCE_WORKSPACE.cleanup()
        _SOURCE_WORKSPACE = None


class TRFSourceBindingTests(unittest.TestCase):
    def test_neutral_demonstration_manifest_is_the_only_source_binding(self) -> None:
        config = load_trf_config(PROJECT_ROOT / "config" / "trf.json")
        snapshot = snapshot_source(config)
        self.assertEqual(snapshot["dataset_id"], "skill-sentences-326-v1")
        self.assertEqual(set(snapshot["files"]), {"manifest", "records", "audit"})

    def test_missing_demonstration_manifest_fails_closed(self) -> None:
        config = load_trf_config(
            PROJECT_ROOT / "config" / "trf.json", "missing-demonstrations.json"
        )
        with self.assertRaises(Exception):
            snapshot_source(config)


class TRFCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.decisions = read_jsonl(source_files(CONFIG).records)

    def test_fixed_source_contract_and_span_sources(self) -> None:
        corpus, excluded, summary = build_corpus_records(
            self.decisions, CONFIG["source"]["expected_counts"]
        )
        self.assertEqual(len(corpus), 313)
        self.assertEqual(len(excluded), 13)
        self.assertEqual(summary["counts"], CONFIG["source"]["expected_counts"])
        self.assertEqual(summary["accepted_spans"], 497)
        self.assertEqual(
            summary["accepted_by"], {"exact_vote": 491, "family_hard_match": 6}
        )
        self.assertTrue(any(len(item["skill_spans"]) > 1 for item in corpus))
        self.assertTrue(
            any(
                span["accepted_by"] == "family_hard_match"
                for item in corpus
                for span in item["skill_spans"]
            )
        )
        for item in corpus:
            expected_types = ["Skill"] if item["status"] == "accepted" else []
            self.assertEqual(item["pseudo_types"], expected_types)
            self.assertTrue(item["eligible_for_statistics"])
            self.assertTrue(item["eligible_as_demonstration"])
            for span in item["skill_spans"]:
                self.assertEqual(
                    item["sentence"][span["start"] : span["end"]], span["text"]
                )

    def test_span_validation_fails_closed_on_bad_offset(self) -> None:
        sentence = "Use project management."
        good = {
            "text": "project management",
            "start": 4,
            "end": 22,
            "exact_votes": 5,
            "accepted_by": "exact_vote",
        }
        self.assertEqual(validate_span(1, sentence, good)["text"], "project management")
        bad = {**good, "end": 21}
        with self.assertRaises(TRFError):
            validate_span(1, sentence, bad)


class TRFTokenizerAndStatisticsTests(unittest.TestCase):
    def test_technical_tokens_case_digits_stopwords_and_placeholders(self) -> None:
        text = (
            "C++ C# CI/CD patient-centred .NET English english skill2 and "
            "<ORGANIZATION> <PROFESSION> <LOCATION> <NAME> <CONTACT>"
        )
        tokens = tokenize_unigrams(
            text,
            placeholder_pattern=CONFIG["tokenization"]["anonymous_placeholder_pattern"],
            stopwords=frozenset({"and"}),
            exclude_digits=True,
        )
        self.assertEqual(
            tokens,
            ["C++", "C#", "CI/CD", "patient-centred", ".NET", "English", "english"],
        )

    def test_context_mask_preserves_length_and_only_masks_positive_spans(self) -> None:
        sentence = "Use project management today."
        start = sentence.index("project management")
        masked = mask_skill_spans(
            sentence,
            [{"text": "project management", "start": start, "end": start + 18}],
        )
        self.assertEqual(len(masked), len(sentence))
        self.assertEqual(masked[:start], sentence[:start])
        self.assertEqual(masked[start + 18 :], sentence[start + 18 :])
        records = [
            {
                "status": "accepted",
                "sentence": sentence,
                "skill_spans": [
                    {"text": "project management", "start": start, "end": start + 18}
                ],
            },
            {"status": "negative", "sentence": sentence, "skill_spans": []},
        ]
        positive, negative = _class_tokens(
            records,
            context_only=True,
            placeholder_pattern=CONFIG["tokenization"]["anonymous_placeholder_pattern"],
            stopwords=frozenset(),
            exclude_digits=True,
        )
        self.assertNotIn("project", positive[0])
        self.assertIn("project", negative[0])

    def test_binary_mi_rho_direction_and_tie_breaking(self) -> None:
        self.assertAlmostEqual(binary_mutual_information(2, 0, 2, 2), math.log(2.0))
        positive = [
            ["good", "tieA", "reverse", "rho"],
            ["good", "tieB"],
        ]
        negative = [
            ["reverse", "rho", "rho"],
            ["reverse", "rho", "rho"],
        ]
        all_metrics, paper, directional = calculate_candidate_metrics(positive, negative, 3.0)
        by_text = {record["text"]: record for record in all_metrics}
        self.assertEqual(by_text["rho"]["frequency_ratio_negative_to_positive"], 4.0)
        self.assertFalse(by_text["rho"]["paper_formula_eligible"])
        self.assertTrue(by_text["reverse"]["paper_formula_eligible"])
        self.assertFalse(by_text["reverse"]["direction_gate_passed"])
        self.assertNotIn("reverse", [record["text"] for record in directional])
        self.assertLess(
            [record["text"] for record in paper].index("tieA"),
            [record["text"] for record in paper].index("tieB"),
        )

    def test_real_candidate_banks_have_fixed_top_twenty(self) -> None:
        corpus, _, _ = build_corpus_records(
            read_jsonl(source_files(CONFIG).records), CONFIG["source"]["expected_counts"]
        )
        main = extract_bank(corpus, CONFIG, context_only=False)
        context = extract_bank(corpus, CONFIG, context_only=True)
        self.assertEqual(len(main["selected"]), 20)
        self.assertEqual(len(context["selected"]), 20)
        self.assertEqual(
            [item["text"] for item in main["selected"][:10]],
            [
                "English",
                "projects",
                "skills",
                "management",
                "sales",
                "communication",
                "software",
                "technical",
                "tools",
                "years",
            ],
        )
        self.assertEqual(len({item["text"] for item in main["selected"]}), 20)
        self.assertEqual(len({item["text"] for item in context["selected"]}), 20)
        self.assertTrue(all(item["direction_gate_passed"] for item in main["selected"]))
        self.assertTrue(all(item["direction_gate_passed"] for item in context["selected"]))


class FakeTokenizer:
    is_fast = True

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        import torch

        if kwargs.get("return_offsets_mapping"):
            spans = [(match.start(), match.end()) for match in re.finditer(r"\w+", text)]
            ids = [101] + list(range(11, 11 + len(spans))) + [102]
            offsets = [(0, 0)] + spans + [(0, 0)]
            special = [1] + [0] * len(spans) + [1]
            return {
                "input_ids": torch.tensor([ids]),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
                "offset_mapping": torch.tensor([offsets]),
                "special_tokens_mask": torch.tensor([special]),
            }
        return {
            "input_ids": torch.tensor([[101, 11, 12, 102]]),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "special_tokens_mask": torch.tensor([[1, 0, 0, 1]]),
        }


class FakeModel:
    def __call__(self, input_ids: object, attention_mask: object) -> object:
        import torch

        values = input_ids.to(dtype=torch.float32)
        hidden = torch.stack([values, values + 1, values + 2, values + 3], dim=-1)
        return SimpleNamespace(last_hidden_state=hidden)


class TRFBertAlgorithmTests(unittest.TestCase):
    def test_placeholder_is_excluded_and_offsets_point_back(self) -> None:
        sentence = "Use <ORGANIZATION> C++ and CI/CD."
        tokens = lexical_tokens_with_offsets(
            sentence, CONFIG["tokenization"]["anonymous_placeholder_pattern"]
        )
        self.assertEqual([item["text"] for item in tokens], ["Use", "C++", "and", "CI/CD"])
        for token in tokens:
            self.assertEqual(
                sentence[int(token["start"]) : int(token["end"])], token["text"]
            )

    def test_embedding_pooling_and_distance_tie_contract(self) -> None:
        import numpy as np

        candidate = embed_candidate("skill", FakeTokenizer(), FakeModel(), 512)
        self.assertEqual(candidate.shape, (4,))
        sentence_tokens = embed_sentence_tokens(
            "alpha beta",
            FakeTokenizer(),
            FakeModel(),
            512,
            CONFIG["tokenization"]["anonymous_placeholder_pattern"],
        )
        self.assertEqual(len(sentence_tokens), 2)
        candidates = [{"text": "b", "rank": 2}, {"text": "a", "rank": 1}]
        vectors = {
            "a": np.array([11, 12, 13, 14], dtype=np.float32),
            "b": np.array([11, 12, 13, 14], dtype=np.float32),
        }
        ranked = rank_nearest_candidates(candidates, vectors, sentence_tokens, 2, 8)
        self.assertEqual([item["text"] for item in ranked], ["a", "b"])
        self.assertEqual(ranked[0]["nearest_token"], "alpha")

    def test_offline_resolution_does_not_call_download_client(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "models--bert-base-cased"
            snapshot = cache / "snapshots" / CONFIG["model"]["revision"]
            snapshot.mkdir(parents=True)
            (snapshot / "model.safetensors").write_bytes(b"test")
            with patch("trf.offline.assign_pseudo_trfs._repo_cache_directory", return_value=cache):
                with patch("huggingface_hub.snapshot_download", side_effect=AssertionError):
                    self.assertEqual(resolve_model_snapshot(CONFIG["model"], False), snapshot)


class TRFStageIntegrationTests(unittest.TestCase):
    def test_stage_one_two_manifest_source_safety_and_no_overwrite(self) -> None:
        before = snapshot_source(CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            config = copy.deepcopy(CONFIG)
            config["output"]["runs_root"] = str(Path(directory) / "runs")
            config_path = Path(directory) / "trf.json"
            atomic_write_json(config_path, config)
            paths = initialize_run(config_path, config, "test-run", ["unit-test"])
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
            self.assertEqual(len(read_jsonl(paths.corpus / "records.jsonl")), 313)
            self.assertEqual(len(load_json(paths.candidates / "trfs.json")["trfs"]), 20)
            with self.assertRaises(TRFError):
                initialize_run(config_path, config, "test-run", ["unit-test"])
        self.assertEqual(before, snapshot_source(CONFIG))


def _real_bert_smoke_ready() -> bool:
    if sys.version_info[:2] != (3, 10):
        return False
    model_config = CONFIG.get("model") or CONFIG["offline"]["model"]
    snapshot = (
        _repo_cache_directory(model_config["name"])
        / "snapshots"
        / model_config["revision"]
    )
    return (snapshot / "model.safetensors").is_file()


@unittest.skipUnless(_real_bert_smoke_ready(), "Pinned BERT weights/Python 3.10 unavailable")
class TRFRealBertSmokeTests(unittest.TestCase):
    def test_real_model_hidden_size_vectors_and_offsets(self) -> None:
        import numpy as np

        snapshot = resolve_model_snapshot(CONFIG["model"], False)
        tokenizer, model = load_pinned_bert(snapshot, 768)
        candidate = embed_candidate("communication", tokenizer, model, 512)
        tokens = embed_sentence_tokens(
            "Clear and inclusive communication.",
            tokenizer,
            model,
            512,
            CONFIG["tokenization"]["anonymous_placeholder_pattern"],
        )
        self.assertEqual(candidate.shape, (768,))
        self.assertTrue(np.isfinite(candidate).all())
        self.assertTrue(all(item["vector"].shape == (768,) for item in tokens))
        sentence = "Clear and inclusive communication."
        for token in tokens:
            self.assertEqual(
                sentence[int(token["start"]) : int(token["end"])], token["text"]
            )


if __name__ == "__main__":
    unittest.main()
