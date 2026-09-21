from __future__ import annotations

import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
import sys

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from aggregator.common import AggregatorError, aggregator_run_paths, load_aggregator_config
from aggregator.evaluation import _bootstrap, _metrics, evaluate
from aggregator.pipeline import (
    build_aggregator_inputs,
    build_prompt,
    expert_indexes,
    review_reasons,
)
from aggregator.runner import run
from aggregator.validator import validate
from common.io_utils import atomic_write_json, atomic_write_jsonl, load_json, read_jsonl, sha256_file


SOURCE = "a" * 64


def selected_examples() -> list[dict]:
    return [
        {
            "demo_dataset_id": "demo",
            "demo_record_id": "7",
            "demo_idx": 7,
            "sentence": "Use Python daily.",
            "status": "accepted",
            "skill_spans": ["Python"],
            "accepted_span_details": [
                {
                    "label_id": "ignored",
                    "text": "Python",
                    "start": 4,
                    "end": 10,
                    "exact_votes": 5,
                    "sample_indexes": [0, 1],
                    "accepted_by": "exact_vote",
                }
            ],
            "pseudo_trfs": ["must-not-leak"],
            "similarity": 0.9,
            "existence_score": 1.0,
            "helpfulness_score": 5,
            "role": "supporting",
            "reason_codes": ["must-not-leak"],
            "gate_rank": 1,
        },
        {
            "demo_dataset_id": "demo",
            "demo_record_id": "8",
            "demo_idx": 8,
            "sentence": "Attend weekly meetings.",
            "status": "negative",
            "skill_spans": [],
            "accepted_span_details": [],
            "pseudo_trfs": [],
            "similarity": 0.8,
            "existence_score": 0.0,
            "helpfulness_score": 4,
            "role": "contrastive",
            "reason_codes": ["NEGATIVE_CONTRAST"],
            "gate_rank": 2,
        },
    ]


def context(*, trf_status: str = "complete", exemplar_status: str = "complete") -> dict:
    chosen = selected_examples()
    return {
        "schema_version": "second-layer-context-v1",
        "dataset_id": "targets",
        "record_id": "1",
        "source_sha256": SOURCE,
        "idx": 1,
        "sentence": "Build Python services.",
        "assembly_status": "ready",
        "trf_context": {
            "status": trf_status,
            "entity_types": ["Skill"],
            "trfs": [
                {"normalized_text": "tools", "raw": "ignored"},
                {"normalized_text": "technical"},
                {"normalized_text": "tools"},
            ],
            "retrieval_count": 3,
            "review_reasons": [],
            "models": {},
        },
        "exemplar_context": {
            "status": exemplar_status,
            "feature_context": "absent",
            "candidate_count": 16,
            "eligible_count": 2,
            "selected_count": len(chosen),
            "selected": chosen,
            "rejected_demo_ids": [],
            "review_reasons": [],
            "feature_review_reasons": [],
            "models": {},
        },
        "provenance": {},
    }


def expert(branch: str, spans: list[dict] | None = None, *, status: str = "complete") -> dict:
    spans = spans if spans is not None else [{"text": "Python", "start": 6, "end": 12}]
    return {
        "schema_version": "skill-prediction-v1",
        "branch": branch,
        "dataset_id": "targets",
        "record_id": "1",
        "source_sha256": SOURCE,
        "idx": 1,
        "sentence": "Build Python services.",
        "status": status,
        "has_skill": 1 if spans else 0,
        "spans": spans,
        "review_reasons": ["advisory"] if status == "needs_review" else [],
        "evidence": {},
        "models": {"chat": "qwen3.7-plus-2026-05-26"},
        "prompt_sha256": "b" * 64,
    }


def provisional_expert_result(branch: str) -> dict:
    return {
        "schema_version": "skill-prediction-result-v1",
        "branch": branch,
        "dataset_id": "targets",
        "record_id": "1",
        "source_sha256": SOURCE,
        "idx": 1,
        "sentence": "Build Python services.",
        "outcome": "provisional",
        "validation_issue": True,
        "coordinate_space": "model_sentence",
        "prediction": None,
        "model_output": '{"annotated_sentence":"Build Pyth0n services."}',
        "provisional_extraction": {
            "schema_version": "skill-prediction-provisional-v1",
            "model_sentence": "Build Pyth0n services.",
            "has_skill": 1,
            "spans": [{"text": "Pyth0n", "start": 6, "end": 12}],
        },
        "failure": {},
        "review_reasons": ["prediction_validation_failed"],
    }


class FakeClient:
    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)

    def chat(self, messages, **kwargs):
        return {"content": self.contents.pop(0), "usage": {}}


class PipelineTests(unittest.TestCase):
    def test_modes_share_base_evidence_and_evidence_prompt_has_no_expert_leakage(self) -> None:
        source = context()
        evidence_input = build_aggregator_inputs(
            [source], mode="evidence_only", maximum_examples=8
        )[0]
        indexes = expert_indexes([expert("trf")], [expert("exemplar")])
        expert_input = build_aggregator_inputs(
            [source],
            mode="with_expert_results",
            maximum_examples=8,
            experts=indexes,
        )[0]
        for key in ("dataset_id", "record_id", "idx", "target_sentence", "target_trfs", "examples"):
            self.assertEqual(evidence_input[key], expert_input[key])
        self.assertEqual(evidence_input["target_trfs"], ["tools", "technical"])
        self.assertEqual(evidence_input["examples"][0]["skill_spans"], [{"text": "Python", "start": 4, "end": 10}])
        self.assertEqual(evidence_input["examples"][1]["has_skill"], 0)
        serialized_input = json.dumps(evidence_input, ensure_ascii=False)
        for forbidden in ("pseudo_trfs", "similarity", "existence_score", "reason_codes", "role"):
            self.assertNotIn(forbidden, serialized_input)
        evidence_prompt = build_prompt(evidence_input, source, maximum_characters=60000)
        serialized_prompt = json.dumps(evidence_prompt, ensure_ascii=False)
        self.assertNotIn("expert_proposals", serialized_prompt)
        self.assertNotIn("must-not-leak", serialized_prompt)
        expert_prompt = build_prompt(expert_input, source, maximum_characters=60000)
        self.assertIn("may both be wrong", expert_prompt["messages"][0]["content"])
        self.assertIn("expert_proposals", expert_prompt["messages"][1]["content"])

    def test_missing_one_expert_is_review_but_disagreement_is_not(self) -> None:
        source = context()
        indexes = expert_indexes([expert("trf")], [])
        item = build_aggregator_inputs(
            [source], mode="with_expert_results", maximum_examples=8, experts=indexes
        )[0]
        self.assertFalse(item["expert_proposals"]["exemplar"]["available"])
        self.assertEqual(review_reasons(item, source), ["expert_prediction_missing:exemplar"])
        disagreeing = expert_indexes([expert("trf")], [expert("exemplar", [])])
        item = build_aggregator_inputs(
            [source], mode="with_expert_results", maximum_examples=8, experts=disagreeing
        )[0]
        self.assertEqual(review_reasons(item, source), [])
        boundary_disagreement = expert_indexes(
            [expert("trf")],
            [expert("exemplar", [{"text": "Build", "start": 0, "end": 5}])],
        )
        item = build_aggregator_inputs(
            [source],
            mode="with_expert_results",
            maximum_examples=8,
            experts=boundary_disagreement,
        )[0]
        self.assertEqual(review_reasons(item, source), [])

    def test_both_missing_falls_back_to_base_evidence_but_misalignment_fails(self) -> None:
        source = context()
        item = build_aggregator_inputs(
            [source],
            mode="with_expert_results",
            maximum_examples=8,
            experts=expert_indexes([], []),
        )[0]
        self.assertEqual(
            review_reasons(item, source),
            ["expert_prediction_missing:trf", "expert_prediction_missing:exemplar"],
        )
        prompt = build_prompt(item, source, maximum_characters=60000)
        self.assertIn("target_trfs", prompt["messages"][1]["content"])
        bad = expert("trf")
        bad["sentence"] = "Build Python systems."
        with self.assertRaisesRegex(AggregatorError, "sentence mismatch"):
            build_aggregator_inputs(
                [source],
                mode="with_expert_results",
                maximum_examples=8,
                experts=expert_indexes([bad], [expert("exemplar")]),
            )
        bad_source = expert("trf")
        bad_source["source_sha256"] = "c" * 64
        with self.assertRaisesRegex(AggregatorError, "source hash mismatch"):
            build_aggregator_inputs(
                [source],
                mode="with_expert_results",
                maximum_examples=8,
                experts=expert_indexes([bad_source], [expert("exemplar")]),
            )
        bad_identity = expert("trf")
        bad_identity["record_id"] = "wrong"
        with self.assertRaisesRegex(AggregatorError, "identity mismatch"):
            build_aggregator_inputs(
                [source],
                mode="with_expert_results",
                maximum_examples=8,
                experts=expert_indexes([bad_identity], [expert("exemplar")]),
            )

    def test_provisional_expert_is_retained_in_model_coordinate_space(self) -> None:
        source = context()
        item = build_aggregator_inputs(
            [source],
            mode="with_expert_results",
            maximum_examples=8,
            experts=expert_indexes(
                [provisional_expert_result("trf")], [expert("exemplar")]
            ),
        )[0]
        proposal = item["expert_proposals"]["trf"]
        self.assertTrue(proposal["available"])
        self.assertEqual(proposal["outcome"], "provisional")
        self.assertEqual(proposal["coordinate_space"], "model_sentence")
        self.assertEqual(proposal["model_sentence"], "Build Pyth0n services.")
        bad_branch = expert("not-trf")
        with self.assertRaisesRegex(AggregatorError, "wrong branch"):
            expert_indexes([bad_branch], [expert("exemplar")])
        bad_boundary = expert("trf")
        bad_boundary["spans"][0]["end"] = 13
        with self.assertRaisesRegex(AggregatorError, "exact end-exclusive"):
            expert_indexes([bad_boundary], [expert("exemplar")])

    def test_review_rules_do_not_use_empty_trfs_or_expert_review_status(self) -> None:
        source = context()
        source["trf_context"]["trfs"] = []
        source["exemplar_context"]["selected"] = source["exemplar_context"]["selected"][:1]
        source["exemplar_context"]["selected_count"] = 1
        indexes = expert_indexes(
            [expert("trf", status="needs_review")],
            [expert("exemplar", status="needs_review")],
        )
        item = build_aggregator_inputs(
            [source], mode="with_expert_results", maximum_examples=8, experts=indexes
        )[0]
        self.assertEqual(review_reasons(item, source), ["insufficient_helpful_examples"])

    def test_example_cap_and_hard_gate_are_enforced(self) -> None:
        source = context()
        source["exemplar_context"]["selected"] = [
            {**copy.deepcopy(selected_examples()[1]), "demo_record_id": str(i), "demo_idx": i, "gate_rank": i}
            for i in range(1, 10)
        ]
        source["exemplar_context"]["selected_count"] = 9
        item = build_aggregator_inputs([source], mode="evidence_only", maximum_examples=8)[0]
        self.assertEqual(len(item["examples"]), 8)
        source["exemplar_context"]["selected"][0]["helpfulness_score"] = 3
        with self.assertRaisesRegex(AggregatorError, "hard-gated"):
            build_aggregator_inputs([source], mode="evidence_only", maximum_examples=8)

    def test_zero_and_all_positive_examples(self) -> None:
        empty = context()
        empty["exemplar_context"]["selected"] = []
        empty["exemplar_context"]["selected_count"] = 0
        empty_input = build_aggregator_inputs(
            [empty], mode="evidence_only", maximum_examples=8
        )[0]
        self.assertEqual(review_reasons(empty_input, empty), ["no_helpful_examples"])

        positive = context()
        first = copy.deepcopy(selected_examples()[0])
        second = copy.deepcopy(first)
        second.update({"demo_record_id": "9", "demo_idx": 9, "gate_rank": 2})
        positive["exemplar_context"]["selected"] = [first, second]
        positive["exemplar_context"]["selected_count"] = 2
        positive_input = build_aggregator_inputs(
            [positive], mode="evidence_only", maximum_examples=8
        )[0]
        self.assertTrue(all(item["has_skill"] == 1 for item in positive_input["examples"]))
        self.assertEqual(review_reasons(positive_input, positive), [])

    def test_offline_metric_and_bootstrap_contract(self) -> None:
        gold = [
            {
                "idx": 1,
                "sentence": "Use SQL.",
                "spans": [{"text": "SQL", "start": 4, "end": 7}],
            },
            {"idx": 2, "sentence": "Attend meeting.", "spans": []},
        ]
        evidence = {
            1: {"idx": 1, "sentence": "Use SQL.", "spans": []},
            2: {"idx": 2, "sentence": "Attend meeting.", "spans": []},
        }
        experts = {
            1: {
                "idx": 1,
                "sentence": "Use SQL.",
                "spans": [{"text": "SQL", "start": 4, "end": 7}],
            },
            2: {"idx": 2, "sentence": "Attend meeting.", "spans": []},
        }
        self.assertEqual(_metrics(gold, experts, [1, 2])["exact_span"]["f1"], 1.0)
        interval = _bootstrap(gold, evidence, experts, [1, 2], samples=50, seed=42)
        self.assertEqual(set(interval), {"exact_span_f1", "sentence_exact_set_match"})


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.context_records = self.root / "context" / "records.jsonl"
        atomic_write_jsonl(self.context_records, [context()])
        self.context_manifest = self.root / "manifest.json"
        atomic_write_json(
            self.context_manifest,
            {
                "status": "completed",
                "outputs": {
                    "context/records.jsonl": {
                        "sha256": sha256_file(self.context_records),
                        "bytes": self.context_records.stat().st_size,
                    }
                },
            },
        )
        config = copy.deepcopy(load_json(PROJECT_ROOT / "config" / "aggregator.json"))
        config["output"]["runs_root"] = str(self.root / "runs")
        self.config = self.root / "aggregator.json"
        atomic_write_json(self.config, config)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def args(self, run_id: str, **updates) -> argparse.Namespace:
        values = {
            "config": str(self.config),
            "mode": "EvidenceOnly",
            "run_id": run_id,
            "context_manifest": str(self.context_manifest),
            "context_records": str(self.context_records),
            "trf_predictions": None,
            "exemplar_predictions": None,
            "limit": 1,
            "prepare_only": False,
            "allow_network": True,
            "confirm_full_run": False,
            "resume": False,
            "retry_failed": False,
        }
        values.update(updates)
        return argparse.Namespace(**values)

    def test_prepare_validate_resume_repair_and_tamper_detection(self) -> None:
        args = self.args("integration", prepare_only=True, allow_network=False)
        self.assertEqual(run(args), "partial")
        self.assertEqual(validate(self.config, "integration")["run_status"], "partial")
        args = self.args("integration", resume=True)
        client = FakeClient(
            [
                "not-json",
                json.dumps({"annotated_sentence": "Build Python <skill>services</skill>."}),
            ]
        )
        self.assertEqual(run(args, client_factory=lambda: client), "completed")
        self.assertEqual(validate(self.config, "integration")["run_status"], "completed")
        config = load_aggregator_config(self.config)
        paths = aggregator_run_paths(config, "integration")
        prediction = read_jsonl(paths.prediction)[0]
        self.assertEqual(prediction["spans"], [{"text": "services", "start": 13, "end": 21}])
        self.assertIn("repair", read_jsonl(paths.raw)[0])
        self.assertEqual(read_jsonl(paths.results)[0]["outcome"], "exact")
        self.assertEqual(read_jsonl(paths.validation_issues), [])

        inputs = read_jsonl(paths.inputs)
        original_inputs = copy.deepcopy(inputs)
        inputs[0]["target_trfs"] = ["tampered"]
        atomic_write_jsonl(paths.inputs, inputs)
        with self.assertRaisesRegex(AggregatorError, "normalized inputs"):
            validate(self.config, "integration")
        atomic_write_jsonl(paths.inputs, original_inputs)

        predictions = read_jsonl(paths.prediction)
        original_predictions = copy.deepcopy(predictions)
        predictions[0]["spans"] = []
        atomic_write_jsonl(paths.prediction, predictions)
        with self.assertRaisesRegex(AggregatorError, "predictions"):
            validate(self.config, "integration")
        atomic_write_jsonl(paths.prediction, original_predictions)

        results = read_jsonl(paths.results)
        original_results = copy.deepcopy(results)
        results[0]["outcome"] = "provisional"
        atomic_write_jsonl(paths.results, results)
        with self.assertRaisesRegex(AggregatorError, "prediction results"):
            validate(self.config, "integration")
        atomic_write_jsonl(paths.results, original_results)

        manifest = load_json(paths.manifest)
        original_manifest = copy.deepcopy(manifest)
        manifest["compatibility"]["chat"]["max_tokens"] = 513
        atomic_write_json(paths.manifest, manifest)
        with self.assertRaisesRegex(AggregatorError, "compatibility contract"):
            validate(self.config, "integration")
        atomic_write_json(paths.manifest, original_manifest)

        prompts = read_jsonl(paths.prompts)
        prompts[0]["review_reasons"] = ["tampered"]
        atomic_write_jsonl(paths.prompts, prompts)
        with self.assertRaisesRegex(AggregatorError, "prompts"):
            validate(self.config, "integration")

    def test_evidence_only_rejects_expert_paths(self) -> None:
        predictions = self.root / "expert.jsonl"
        atomic_write_jsonl(predictions, [])
        args = self.args(
            "leak",
            prepare_only=True,
            allow_network=False,
            trf_predictions=str(predictions),
        )
        with self.assertRaisesRegex(AggregatorError, "forbids expert"):
            run(args)

    def test_terminal_format_failure_validates_and_retry_failed_resumes(self) -> None:
        args = self.args("retry")
        invalid = FakeClient(["bad", "bad", "bad"])
        self.assertEqual(run(args, client_factory=lambda: invalid), "partial")
        self.assertEqual(validate(self.config, "retry")["run_status"], "partial")
        paths = aggregator_run_paths(load_aggregator_config(self.config), "retry")
        result = read_jsonl(paths.results)[0]
        self.assertEqual(result["outcome"], "validation_failed")
        self.assertTrue(result["validation_issue"])
        self.assertEqual([item["idx"] for item in read_jsonl(paths.validation_issues)], [1])
        args = self.args("retry", resume=True, retry_failed=True)
        repaired = FakeClient(
            [json.dumps({"annotated_sentence": "Build <skill>Python</skill> services."})]
        )
        self.assertEqual(run(args, client_factory=lambda: repaired), "completed")
        self.assertEqual(validate(self.config, "retry")["run_status"], "completed")
        self.assertEqual(read_jsonl(paths.validation_issues), [])

    def test_expert_mode_can_ignore_proposal_and_emit_new_span(self) -> None:
        trf_path = self.root / "trf.jsonl"
        exemplar_path = self.root / "exemplar.jsonl"
        atomic_write_jsonl(trf_path, [expert("trf")])
        atomic_write_jsonl(exemplar_path, [])
        args = self.args(
            "experts",
            mode="WithExpertResults",
            trf_predictions=str(trf_path),
            exemplar_predictions=str(exemplar_path),
        )
        client = FakeClient(
            [json.dumps({"annotated_sentence": "Build Python <skill>services</skill>."})]
        )
        self.assertEqual(run(args, client_factory=lambda: client), "completed")
        paths = aggregator_run_paths(load_aggregator_config(self.config), "experts")
        prediction = read_jsonl(paths.prediction)[0]
        self.assertEqual(prediction["branch"], "aggregator_with_expert_results")
        self.assertEqual(prediction["status"], "needs_review")
        self.assertEqual(prediction["review_reasons"], ["expert_prediction_missing:exemplar"])
        self.assertEqual(prediction["spans"][0]["text"], "services")
        self.assertEqual(validate(self.config, "experts")["run_status"], "completed")

    def test_offline_evaluator_compares_completed_matched_runs(self) -> None:
        evidence_client = FakeClient(
            [json.dumps({"annotated_sentence": "Build Python services."})]
        )
        self.assertEqual(
            run(self.args("eval-evidence"), client_factory=lambda: evidence_client),
            "completed",
        )
        trf_path = self.root / "eval-trf.jsonl"
        exemplar_path = self.root / "eval-exemplar.jsonl"
        atomic_write_jsonl(trf_path, [expert("trf")])
        atomic_write_jsonl(exemplar_path, [expert("exemplar")])
        expert_client = FakeClient(
            [json.dumps({"annotated_sentence": "Build Python <skill>services</skill>."})]
        )
        expert_args = self.args(
            "eval-experts",
            mode="WithExpertResults",
            trf_predictions=str(trf_path),
            exemplar_predictions=str(exemplar_path),
        )
        self.assertEqual(run(expert_args, client_factory=lambda: expert_client), "completed")
        gold_path = self.root / "gold.json"
        atomic_write_json(
            gold_path,
            [
                {
                    "idx": 1,
                    "sentence": "Build Python services.",
                    "has_skill": 1,
                    "spans": ["services"],
                }
            ],
        )
        result = evaluate(
            self.config,
            "eval-evidence",
            "eval-experts",
            gold_path,
            trf_path,
            exemplar_path,
            bootstrap_samples=20,
            seed=42,
        )
        self.assertTrue(result["paired_comparison"]["positive_contribution_criterion_met"])
        self.assertEqual(result["with_expert_results"]["metrics"]["exact_span"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
