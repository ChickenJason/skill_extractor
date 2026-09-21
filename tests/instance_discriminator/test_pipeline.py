from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
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
from instance_discriminator.runner import run  # noqa: E402
from instance_discriminator.validator import validate  # noqa: E402
from instance_discriminator.common import (  # noqa: E402
    InstanceDiscriminatorError,
    assert_sources_unchanged,
    load_discriminator_config,
    load_source_bundle,
    snapshot_sources,
)
from instance_discriminator.pipeline import (  # noqa: E402
    REASON_CODES,
    apply_hard_gate,
    build_candidate_record,
    build_discriminator_prompt,
    build_selected_record,
    build_skill_prediction_prompt,
    parse_judgments,
)


CHAT_MODEL = "qwen3.7-plus-2026-05-26"


def _base_config() -> dict[str, Any]:
    return load_json(PROJECT_ROOT / "config" / "instance_discriminator.json")


def _decision(idx: int) -> dict[str, Any]:
    accepted = idx % 2 == 1
    span = f"skill-{idx}"
    return {
        "schema_version": "demonstration-record-v1",
        "dataset_id": "demo-set",
        "record_id": str(idx),
        "source_sha256": "a" * 64,
        "idx": idx,
        "sentence": f"DEMO-ID:{idx} sentence with {span}",
        "status": "accepted" if accepted else "negative",
        "has_skill": 1 if accepted else 0,
        "spans": [span] if accepted else [],
        "existence_consensus": {"outcome": "positive" if accepted else "negative"},
        "accepted_span_details": (
            [{"text": span, "start": 24, "end": 24 + len(span), "exact_votes": 5}]
            if accepted
            else []
        ),
    }


def _target(idx: int, *, review: bool = False) -> dict[str, Any]:
    return {
        "schema_version": "feature-records-v1",
        "dataset_id": "target-set",
        "record_id": str(idx),
        "source_sha256": "b" * 64,
        "idx": idx,
        "sentence": f"TARGET-ID:{idx} requires a transferable skill",
        "status": "needs_review" if review else "complete",
        "entity_types": ["Skill"],
        "trfs": [] if review else [{"normalized_text": "transferable skill"}],
        "retrieval_count": 16,
        "review_reasons": ["upstream_fixture_review"] if review else [],
        "models": {"embedding": "fixture", "chat": CHAT_MODEL},
    }


def _retrieval(target: dict[str, Any]) -> dict[str, Any]:
    selected = []
    for rank, idx in enumerate(range(1, 17), start=1):
        decision = _decision(idx)
        selected.append(
            {
                "demo_dataset_id": decision["dataset_id"],
                "demo_record_id": decision["record_id"],
                "demo_source_sha256": decision["source_sha256"],
                "demo_idx": idx,
                "selected_rank": rank,
                "similarity": round(0.9 - rank / 100, 10),
                "status": decision["status"],
                "existence_score": 1.0 if rank <= 8 else 0.8,
                "sentence": decision["sentence"],
                "trfs": ["transferable skill"] if decision["status"] == "accepted" else [],
                "skill_spans": list(decision["spans"]),
                "accepted_span_details": list(decision["accepted_span_details"]),
            }
        )
    return {
        "schema_version": "candidate-instances-v1",
        "dataset_id": target["dataset_id"],
        "record_id": target["record_id"],
        "source_sha256": target["source_sha256"],
        "idx": target["idx"],
        "sentence": target["sentence"],
        "leave_one_out": True,
        "self_excluded": True,
        "selected": selected,
    }


def _prepare_workspace(
    root: Path, target_count: int = 2
) -> tuple[Path, Path, Path, Path]:
    targets = [
        _target(1001 + offset, review=offset == 1)
        for offset in range(target_count)
    ]
    target_records = [
        {
            "schema_version": "sentence-record-v1",
            "dataset_id": item["dataset_id"],
            "record_id": item["record_id"],
            "source_sha256": item["source_sha256"],
            "idx": item["idx"],
            "sentence": item["sentence"],
        }
        for item in targets
    ]
    targets_path = root / "targets.jsonl"
    candidates_path = root / "candidates.jsonl"
    features_path = root / "features.jsonl"
    atomic_write_jsonl(targets_path, target_records)
    atomic_write_jsonl(candidates_path, [_retrieval(item) for item in targets])
    atomic_write_jsonl(features_path, targets)

    config = _base_config()
    config["output"]["runs_root"] = str(root / "discriminator-runs")
    config_path = root / "instance_discriminator.json"
    atomic_write_json(config_path, config)
    return config_path, targets_path, candidates_path, features_path


class FakeDiscriminatorClient:
    def __init__(
        self,
        fail_once: set[int] | None = None,
        fail_always: set[int] | None = None,
        invalid_role_once: set[int] | None = None,
        prediction_invalid_once: set[int] | None = None,
        prediction_fail_always: set[int] | None = None,
    ) -> None:
        self.fail_once = set(fail_once or set())
        self.fail_always = set(fail_always or set())
        self.invalid_role_once = set(invalid_role_once or set())
        self.prediction_invalid_once = set(prediction_invalid_once or set())
        self.prediction_fail_always = set(prediction_fail_always or set())
        self.failed: set[int] = set()
        self.role_failed: set[int] = set()
        self.calls: list[int] = []
        self.prediction_failed: set[int] = set()
        self.prediction_calls: list[int] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> dict[str, Any]:
        if "exemplar expert's final exact Skill span extractor" in messages[0]["content"]:
            payload = json.loads(messages[1]["content"])
            sentence = payload["target_sentence"]
            idx = int(sentence.split("TARGET-ID:", 1)[1].split()[0])
            self.prediction_calls.append(idx)
            if idx in self.prediction_fail_always:
                content = "invalid-json"
            elif (
                idx in self.prediction_invalid_once
                and idx not in self.prediction_failed
            ):
                self.prediction_failed.add(idx)
                content = "invalid-json"
            else:
                annotated = sentence.replace(
                    "transferable skill", "<skill>transferable skill</skill>"
                )
                content = json.dumps({"annotated_sentence": annotated})
            return {
                "content": content,
                "finish_reason": "stop",
                "latency_ms": 0,
                "attempts": 1,
                "response_id": f"fake-prediction-{len(self.prediction_calls)}",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

        payload = json.loads(messages[1]["content"])["input"]
        idx = int(payload["target"]["sentence"].split("TARGET-ID:", 1)[1].split()[0])
        self.calls.append(idx)
        if idx in self.fail_always:
            content = "invalid-json"
        elif idx in self.fail_once and idx not in self.failed:
            self.failed.add(idx)
            content = "invalid-json"
        else:
            judgments = []
            for rank, demo in enumerate(payload["demonstrations"], start=1):
                useful = rank <= 8
                accepted = demo["status"] == "accepted"
                judgments.append(
                    {
                        "demo_idx": demo["demo_idx"],
                        "helpfulness_score": 5 if useful else 2,
                        "role": (
                            "supporting"
                            if useful and accepted
                            else "contrastive"
                            if useful
                            else "irrelevant"
                        ),
                        "reason_codes": [
                            "BOUNDARY_TRANSFERABLE" if accepted else "NEGATIVE_CONTRAST"
                        ],
                    }
                )
            if idx in self.invalid_role_once and idx not in self.role_failed:
                self.role_failed.add(idx)
                accepted = next(
                    item
                    for item, demo in zip(judgments, payload["demonstrations"])
                    if demo["status"] == "accepted"
                )
                accepted["role"] = "contrastive"
            content = json.dumps({"judgments": judgments})
        return {
            "content": content,
            "finish_reason": "stop",
            "latency_ms": 0,
            "attempts": 1,
            "response_id": f"fake-{len(self.calls)}",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }


class InstanceDiscriminatorTests(unittest.TestCase):
    @staticmethod
    def _args(
        config: Path,
        targets: Path,
        candidates: Path,
        features: Path | None,
        **overrides: Any,
    ) -> Namespace:
        values = {
            "config": str(config),
            "targets": str(targets),
            "candidates": str(candidates),
            "features": str(features) if features else None,
            "run_id": "test-run",
            "limit": None,
            "prepare_only": False,
            "allow_network": True,
            "resume": False,
            "retry_failed": False,
            "confirm_full_run": True,
        }
        values.update(overrides)
        return Namespace(**values)

    @staticmethod
    def _candidate_record() -> dict[str, Any]:
        target = _target(1001)
        return build_candidate_record(target, _retrieval(target), target, 16)

    @staticmethod
    def _judgments(record: dict[str, Any], score: int = 5) -> list[dict[str, Any]]:
        return [
            {
                "demo_idx": item["demo_idx"],
                "helpfulness_score": score,
                "role": "supporting" if item["status"] == "accepted" else "contrastive",
                "reason_codes": [
                    "BOUNDARY_TRANSFERABLE"
                    if item["status"] == "accepted"
                    else "NEGATIVE_CONTRAST"
                ],
            }
            for item in record["candidates"]
        ]

    def test_source_hashes_and_alignment_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path, targets, candidates, features = _prepare_workspace(Path(directory))
            load_discriminator_config(config_path)
            snapshot = snapshot_sources(targets, candidates, features)
            with targets.open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(InstanceDiscriminatorError, "changed"):
                assert_sources_unchanged(snapshot)

    def test_candidate_contract_rejects_count_status_and_sentence_mismatch(self) -> None:
        target = _target(1001)
        retrieval = _retrieval(target)
        with self.assertRaisesRegex(InstanceDiscriminatorError, "exactly 16"):
            build_candidate_record(
                target, {**retrieval, "selected": retrieval["selected"][:-1]}, target, 16
            )
        broken = json.loads(json.dumps(retrieval))
        broken["selected"][0]["status"] = "negative"
        with self.assertRaisesRegex(InstanceDiscriminatorError, "empty skill spans"):
            build_candidate_record(target, broken, target, 16)
        broken = json.loads(json.dumps(retrieval))
        broken["sentence"] = "different"
        with self.assertRaisesRegex(InstanceDiscriminatorError, "sentence mismatch"):
            build_candidate_record(target, broken, target, 16)

    def test_prompt_target_is_leave_one_out_and_contains_only_allowed_evidence(self) -> None:
        record = self._candidate_record()
        prompt = build_discriminator_prompt(record, 60000)
        payload = json.loads(prompt["messages"][1]["content"])["input"]
        self.assertEqual(
            set(payload["target"]),
            {"sentence", "feature_context", "entity_types", "target_trfs"},
        )
        self.assertNotIn(record["idx"], prompt["demo_indexes"])
        serialized_target = json.dumps(payload["target"])
        for forbidden in ("decision", "status", "spans", "has_skill"):
            self.assertNotIn(forbidden, serialized_target)

        selected = build_selected_record(
            record, self._judgments(record), _base_config()["gate"], CHAT_MODEL
        )
        prediction_prompt = build_skill_prediction_prompt(selected, 60000)
        prediction_payload = json.loads(
            prediction_prompt["messages"][1]["content"]
        )
        self.assertEqual(
            set(prediction_payload),
            {"target_sentence", "selected_examples", "required_output"},
        )
        self.assertEqual(
            set(prediction_payload["selected_examples"][0]),
            {"demo_idx", "sentence", "skill_spans", "helpfulness_score", "role"},
        )
        serialized_prediction = json.dumps(prediction_payload)
        for forbidden in ("pseudo_trfs", "similarity", "existence_score", "target_trfs"):
            self.assertNotIn(forbidden, serialized_prediction)

    def test_zero_and_one_selected_examples_still_build_prediction_prompts(self) -> None:
        record = self._candidate_record()
        zero = build_selected_record(
            record, self._judgments(record, score=2), _base_config()["gate"], CHAT_MODEL
        )
        zero_prompt = build_skill_prediction_prompt(zero, 60000)
        zero_payload = json.loads(zero_prompt["messages"][1]["content"])
        self.assertEqual(zero_payload["selected_examples"], [])
        self.assertIn("no_helpful_examples", zero_prompt["review_reasons"])

        judgments = self._judgments(record, score=2)
        judgments[0]["helpfulness_score"] = 5
        one = build_selected_record(
            record, judgments, _base_config()["gate"], CHAT_MODEL
        )
        one_prompt = build_skill_prediction_prompt(one, 60000)
        one_payload = json.loads(one_prompt["messages"][1]["content"])
        self.assertEqual(len(one_payload["selected_examples"]), 1)
        self.assertIn("insufficient_helpful_examples", one_prompt["review_reasons"])

    def test_same_numeric_idx_in_different_datasets_is_not_self_leakage(self) -> None:
        target = {
            **_target(1),
            "dataset_id": "target-set",
            "record_id": "1",
        }
        candidates = _retrieval(target)
        record = build_candidate_record(target, candidates, target, 16)
        self.assertEqual(record["candidates"][0]["demo_idx"], 1)
        broken = json.loads(json.dumps(candidates))
        broken["selected"][0]["demo_dataset_id"] = "target-set"
        with self.assertRaisesRegex(InstanceDiscriminatorError, "Leave-one-out"):
            build_candidate_record(target, broken, target, 16)

    def test_prepare_only_without_features_records_absent_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, targets, candidates, _ = _prepare_workspace(root)
            args = self._args(
                config_path,
                targets,
                candidates,
                None,
                prepare_only=True,
                allow_network=False,
            )
            self.assertEqual(run(args), "partial")
            result = validate(config_path, "test-run")
            self.assertEqual(result["status"], "valid_prepared")
            manifest = load_json(
                root / "discriminator-runs" / "test-run" / "manifest.json"
            )
            self.assertEqual(
                manifest["compatibility"]["source"]["feature_context"], "absent"
            )
            prompt = read_jsonl(
                root / "discriminator-runs" / "test-run" / "prompts" / "records.jsonl"
            )[0]
            payload = json.loads(prompt["messages"][1]["content"])["input"]
            self.assertEqual(
                set(payload["target"]), {"sentence", "feature_context"}
            )

    def test_strict_parser_enforces_ids_types_roles_codes_and_forbids_reason(self) -> None:
        record = self._candidate_record()
        good = self._judgments(record)
        parsed = parse_judgments(json.dumps({"judgments": good}), record["candidates"])
        self.assertEqual(len(parsed), 16)
        mutations = []
        missing = json.loads(json.dumps(good[:-1]))
        mutations.append(missing)
        duplicate = json.loads(json.dumps(good))
        duplicate[-1]["demo_idx"] = duplicate[0]["demo_idx"]
        mutations.append(duplicate)
        unknown = json.loads(json.dumps(good))
        unknown[0]["demo_idx"] = 999
        mutations.append(unknown)
        bool_score = json.loads(json.dumps(good))
        bool_score[0]["helpfulness_score"] = True
        mutations.append(bool_score)
        wrong_role = json.loads(json.dumps(good))
        wrong_role[0]["role"] = "contrastive"
        mutations.append(wrong_role)
        wrong_code = json.loads(json.dumps(good))
        wrong_code[0]["reason_codes"] = ["UNKNOWN"]
        mutations.append(wrong_code)
        unexpected_reason = json.loads(json.dumps(good))
        unexpected_reason[0]["reason"] = "The model must not emit prose reasons."
        mutations.append(unexpected_reason)
        for value in mutations:
            with self.subTest(value=value[0]), self.assertRaises(InstanceDiscriminatorError):
                parse_judgments(json.dumps({"judgments": value}), record["candidates"])
        self.assertIn("TRF_ALIGNED", REASON_CODES)

    def test_discriminator_prompt_never_requests_free_text_reason(self) -> None:
        prompt = build_discriminator_prompt(self._candidate_record(), 60000)
        payload = json.loads(prompt["messages"][1]["content"])
        judgment = payload["required_output"]["judgments"][0]
        self.assertEqual(
            set(judgment),
            {"demo_idx", "helpfulness_score", "role", "reason_codes"},
        )
        self.assertNotIn('"reason"', prompt["messages"][1]["content"])

    def test_hard_gate_never_backfills_low_scores_and_accepts_negative_contrast(self) -> None:
        record = self._candidate_record()
        judgments = self._judgments(record, score=2)
        judgments[1]["helpfulness_score"] = 4
        gate = _base_config()["gate"]
        result = apply_hard_gate(record["candidates"], judgments, gate)
        self.assertEqual([item["demo_idx"] for item in result["selected"]], [2])
        self.assertEqual(result["selected"][0]["role"], "contrastive")
        selected = build_selected_record(record, judgments, gate, CHAT_MODEL)
        self.assertEqual(selected["status"], "needs_review")
        self.assertEqual(selected["review_reasons"], ["insufficient_helpful_examples"])
        judgments[1]["helpfulness_score"] = 3
        selected = build_selected_record(record, judgments, gate, CHAT_MODEL)
        self.assertEqual(selected["selected_count"], 0)
        self.assertEqual(selected["review_reasons"], ["no_helpful_examples"])

    def test_hard_gate_only_filters_eligible_and_preserves_input_order(self) -> None:
        record = self._candidate_record()
        judgments = self._judgments(record)
        judgments[0]["helpfulness_score"] = 4
        result = apply_hard_gate(record["candidates"], judgments, _base_config()["gate"])
        self.assertEqual(
            [item["demo_idx"] for item in result["selected"]],
            [item["demo_idx"] for item in record["candidates"]],
        )
        self.assertEqual(
            [item["gate_rank"] for item in result["selected"]],
            list(range(1, 17)),
        )
        self.assertEqual(result["eligible_count"], 16)
        self.assertEqual(
            result["role_counts"],
            {
                role: sum(item["role"] == role for item in judgments)
                for role in ("supporting", "contrastive")
            },
        )

    def test_upstream_review_and_empty_skill_trf_reasons_are_preserved(self) -> None:
        target = _target(1002, review=True)
        record = build_candidate_record(
            target, _retrieval(target), target, 16
        )
        result = build_selected_record(
            record, self._judgments(record), _base_config()["gate"], CHAT_MODEL
        )
        self.assertEqual(
            result["review_reasons"][:2],
            ["feature_context_needs_review", "skill_type_without_target_trfs"],
        )
        self.assertEqual(result["feature_review_reasons"], ["upstream_fixture_review"])

    def test_prepare_only_never_creates_client_and_validator_rebuilds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, targets, candidates, features = _prepare_workspace(root)

            def forbidden_client() -> Any:
                raise AssertionError("prepare-only instantiated a client")

            args = self._args(
                config_path,
                targets,
                candidates,
                features,
                prepare_only=True,
                allow_network=True,
            )
            self.assertEqual(run(args, client_factory=forbidden_client), "partial")
            result = validate(config_path, "test-run")
            self.assertEqual(result["status"], "valid_prepared")
            self.assertFalse(result["network_called"])
            manifest = load_json(root / "discriminator-runs" / "test-run" / "manifest.json")
            self.assertFalse(manifest["network_called"])

    def test_fake_full_run_and_validator_reconstruct_all_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, targets, candidates, features = _prepare_workspace(root)
            client = FakeDiscriminatorClient()
            args = self._args(config_path, targets, candidates, features)
            self.assertEqual(run(args, client_factory=lambda: client), "completed")
            self.assertEqual(client.calls, [1001, 1002])
            self.assertEqual(client.prediction_calls, [1001, 1002])
            result = validate(config_path, "test-run")
            self.assertEqual(result["status"], "valid_completed")
            selected = read_jsonl(
                root / "discriminator-runs" / "test-run" / "selected" / "records.jsonl"
            )
            # All eight eligible candidates remain in their original input order.
            self.assertEqual([item["selected_count"] for item in selected], [8, 8])
            self.assertEqual(selected[1]["status"], "needs_review")
            predictions = read_jsonl(
                root
                / "discriminator-runs"
                / "test-run"
                / "prediction"
                / "records.jsonl"
            )
            self.assertEqual([item["has_skill"] for item in predictions], [1, 1])
            self.assertEqual(predictions[0]["spans"][0]["text"], "transferable skill")

    def test_prediction_invalid_output_is_repaired_and_audited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, targets, candidates, features = _prepare_workspace(root)
            client = FakeDiscriminatorClient(prediction_invalid_once={1001})
            args = self._args(config_path, targets, candidates, features)
            self.assertEqual(run(args, client_factory=lambda: client), "completed")
            self.assertEqual(client.prediction_calls, [1001, 1001, 1002])
            raw = read_jsonl(
                root
                / "discriminator-runs"
                / "test-run"
                / "prediction"
                / "raw.jsonl"
            )
            self.assertEqual(len(raw[0]["repair"]["attempts"]), 2)
            self.assertEqual(validate(config_path, "test-run")["status"], "valid_completed")

    def test_failed_prediction_resume_does_not_repeat_discrimination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, targets, candidates, features = _prepare_workspace(root)
            client = FakeDiscriminatorClient(prediction_fail_always={1001})
            args = self._args(config_path, targets, candidates, features)
            self.assertEqual(run(args, client_factory=lambda: client), "partial")
            self.assertEqual(client.calls, [1001, 1002])
            self.assertEqual(client.prediction_calls, [1001, 1001, 1001, 1002])

            no_retry = self._args(
                config_path,
                targets,
                candidates,
                features,
                allow_network=False,
                resume=True,
                confirm_full_run=False,
            )
            self.assertEqual(run(no_retry, client_factory=lambda: client), "partial")
            self.assertEqual(client.calls, [1001, 1002])
            self.assertEqual(client.prediction_calls, [1001, 1001, 1001, 1002])

            client.prediction_fail_always.clear()
            retry = self._args(
                config_path,
                targets,
                candidates,
                features,
                resume=True,
                retry_failed=True,
            )
            self.assertEqual(run(retry, client_factory=lambda: client), "completed")
            self.assertEqual(client.calls, [1001, 1002])
            self.assertEqual(
                client.prediction_calls, [1001, 1001, 1001, 1002, 1001]
            )
            self.assertEqual(validate(config_path, "test-run")["status"], "valid_completed")

    def test_one_failed_prediction_within_three_percent_can_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, targets, candidates, features = _prepare_workspace(
                root, target_count=100
            )
            client = FakeDiscriminatorClient(prediction_fail_always={1001})
            args = self._args(config_path, targets, candidates, features)
            self.assertEqual(run(args, client_factory=lambda: client), "completed")

            run_root = root / "discriminator-runs" / "test-run"
            predictions = read_jsonl(run_root / "prediction" / "records.jsonl")
            self.assertEqual(len(predictions), 99)
            summary = load_json(run_root / "audit" / "summary.json")
            self.assertEqual(summary["prediction"]["failed_indexes"], [1001])
            self.assertEqual(summary["failure_tolerance"]["maximum_failure_rate"], 0.03)
            self.assertEqual(summary["failure_tolerance"]["allowed_failure_count"], 3)
            self.assertEqual(summary["failure_tolerance"]["validation_issue_count"], 1)
            self.assertTrue(summary["failure_tolerance"]["within_tolerance"])
            result = validate(config_path, "test-run")
            self.assertEqual(result["status"], "valid_completed")
            self.assertEqual(result["allowed_failures"], 3)
            self.assertEqual(result["failed_predictions"], 1)

    def test_failed_judgment_is_retained_as_empty_context_under_shared_quota(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, targets, candidates, features = _prepare_workspace(
                root, target_count=100
            )
            client = FakeDiscriminatorClient(fail_always={1001})
            args = self._args(config_path, targets, candidates, features)
            self.assertEqual(run(args, client_factory=lambda: client), "completed")

            run_root = root / "discriminator-runs" / "test-run"
            selected = read_jsonl(run_root / "selected" / "records.jsonl")
            predictions = read_jsonl(run_root / "prediction" / "records.jsonl")
            self.assertEqual(len(selected), 100)
            self.assertEqual(len(predictions), 100)
            placeholder = next(item for item in selected if item["idx"] == 1001)
            self.assertEqual(placeholder["selected"], [])
            self.assertEqual(placeholder["status"], "needs_review")
            self.assertIn("exemplar_judgment_validation_failed", placeholder["review_reasons"])
            self.assertIn(1001, client.prediction_calls)
            summary = load_json(run_root / "audit" / "summary.json")
            self.assertEqual(summary["failed_indexes"], [])
            self.assertEqual(summary["validation_issue_indexes"], [1001])
            self.assertEqual(summary["prediction"]["failed_indexes"], [])
            self.assertTrue(summary["failure_tolerance"]["within_tolerance"])
            self.assertEqual(validate(config_path, "test-run")["status"], "valid_completed")

    def test_invalid_structured_output_is_repaired_and_audited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, targets, candidates, features = _prepare_workspace(root)
            client = FakeDiscriminatorClient(fail_once={1001})
            args = self._args(config_path, targets, candidates, features)
            self.assertEqual(run(args, client_factory=lambda: client), "completed")
            self.assertEqual(client.calls, [1001, 1001, 1002])
            raw_path = root / "discriminator-runs" / "test-run" / "raw" / "responses.jsonl"
            raw = read_jsonl(raw_path)
            self.assertEqual(len(raw), 2)
            self.assertEqual(raw[0]["status"], "complete")
            self.assertEqual(len(raw[0]["repair"]["attempts"]), 2)
            self.assertEqual(
                raw[0]["repair"]["attempts"][0]["parse_error"]["type"],
                "InstanceDiscriminatorError",
            )
            self.assertIsNone(raw[0]["repair"]["attempts"][1]["parse_error"])
            self.assertEqual(validate(config_path, "test-run")["status"], "valid_completed")

    def test_invalid_accepted_role_is_repaired_without_relaxing_parser(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, targets, candidates, features = _prepare_workspace(root)
            client = FakeDiscriminatorClient(invalid_role_once={1001})
            args = self._args(config_path, targets, candidates, features)
            self.assertEqual(run(args, client_factory=lambda: client), "completed")
            self.assertEqual(client.calls, [1001, 1001, 1002])
            raw = read_jsonl(
                root / "discriminator-runs" / "test-run" / "raw" / "responses.jsonl"
            )
            first_error = raw[0]["repair"]["attempts"][0]["parse_error"]["message"]
            self.assertIn("cannot be contrastive", first_error)
            self.assertEqual(validate(config_path, "test-run")["status"], "valid_completed")

    def test_partial_resume_and_retry_failed_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, targets, candidates, features = _prepare_workspace(root)
            client = FakeDiscriminatorClient(fail_always={1001})
            args = self._args(config_path, targets, candidates, features)
            self.assertEqual(run(args, client_factory=lambda: client), "partial")
            self.assertEqual(client.calls, [1001, 1001, 1001, 1002])
            raw_path = root / "discriminator-runs" / "test-run" / "raw" / "responses.jsonl"
            first_raw = read_jsonl(raw_path)
            self.assertEqual(len(first_raw), 2)
            self.assertEqual(len(first_raw[0]["repair"]["attempts"]), 3)
            self.assertIsNotNone(first_raw[0]["response"])

            no_retry = self._args(
                config_path,
                targets,
                candidates,
                features,
                allow_network=False,
                resume=True,
                confirm_full_run=False,
            )
            self.assertEqual(run(no_retry, client_factory=lambda: client), "partial")
            self.assertEqual(client.calls, [1001, 1001, 1001, 1002])
            self.assertEqual(len(read_jsonl(raw_path)), 2)

            manifest_path = root / "discriminator-runs" / "test-run" / "manifest.json"
            manifest = load_json(manifest_path)
            online_key = "code/instance_discriminator/online.py"
            manifest["implementation"][online_key]["sha256"] = "0" * 64
            atomic_write_json(manifest_path, manifest)
            client.fail_always.clear()

            retry = self._args(
                config_path,
                targets,
                candidates,
                features,
                resume=True,
                retry_failed=True,
            )
            self.assertEqual(run(retry, client_factory=lambda: client), "completed")
            self.assertEqual(client.calls, [1001, 1001, 1001, 1002, 1001])
            self.assertEqual(len(read_jsonl(raw_path)), 3)
            upgraded = load_json(manifest_path)
            self.assertEqual(
                upgraded["implementation_upgrades"][0]["id"],
                "structured-output-repair-v1",
            )
            self.assertEqual(validate(config_path, "test-run")["status"], "valid_completed")

    def test_safety_flags_fail_before_run_creation(self) -> None:
        args = self._args(
            Path("missing.json"),
            Path("missing-targets.json"),
            Path("missing-candidates.json"),
            None,
            confirm_full_run=False,
        )
        with self.assertRaisesRegex(InstanceDiscriminatorError, "confirm-full-run"):
            run(args)
        args.confirm_full_run = True
        args.retry_failed = True
        with self.assertRaisesRegex(InstanceDiscriminatorError, "requires --resume"):
            run(args)


if __name__ == "__main__":
    unittest.main()
