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


def _prepare_workspace(root: Path) -> tuple[Path, Path, Path, Path]:
    targets = [_target(1001), _target(1002, review=True)]
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
    def __init__(self, fail_once: set[int] | None = None) -> None:
        self.fail_once = set(fail_once or set())
        self.failed: set[int] = set()
        self.calls: list[int] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> dict[str, Any]:
        payload = json.loads(messages[1]["content"])["input"]
        idx = int(payload["target"]["sentence"].split("TARGET-ID:", 1)[1].split()[0])
        self.calls.append(idx)
        if idx in self.fail_once and idx not in self.failed:
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
                        "reason": "Deterministic fake judgment.",
                    }
                )
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
                "reason": "Useful exact-boundary evidence.",
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

    def test_strict_parser_enforces_ids_types_roles_codes_and_reason_length(self) -> None:
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
        long_reason = json.loads(json.dumps(good))
        long_reason[0]["reason"] = "x" * 241
        mutations.append(long_reason)
        for value in mutations:
            with self.subTest(value=value[0]), self.assertRaises(InstanceDiscriminatorError):
                parse_judgments(json.dumps({"judgments": value}), record["candidates"])
        self.assertIn("TRF_ALIGNED", REASON_CODES)

    def test_hard_gate_never_backfills_low_scores_and_accepts_negative_contrast(self) -> None:
        record = self._candidate_record()
        judgments = self._judgments(record, score=2)
        judgments[1]["helpfulness_score"] = 5
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

    def test_hard_gate_sorting_and_caps_are_deterministic(self) -> None:
        record = self._candidate_record()
        judgments = self._judgments(record)
        result = apply_hard_gate(record["candidates"], judgments, _base_config()["gate"])
        self.assertEqual(len(result["selected"]), 8)
        self.assertLessEqual(result["role_counts"]["supporting"], 6)
        self.assertLessEqual(result["role_counts"]["contrastive"], 3)
        self.assertEqual(
            [item["demo_idx"] for item in result["selected"]],
            [1, 2, 3, 4, 5, 6, 7, 9],
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
            result = validate(config_path, "test-run")
            self.assertEqual(result["status"], "valid_completed")
            selected = read_jsonl(
                root / "discriminator-runs" / "test-run" / "selected" / "records.jsonl"
            )
            # The first eight fake positives contain four negatives; the
            # max_contrastive=3 cap intentionally leaves seven selected.
            self.assertEqual([item["selected_count"] for item in selected], [7, 7])
            self.assertEqual(selected[1]["status"], "needs_review")

    def test_partial_resume_and_retry_failed_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, targets, candidates, features = _prepare_workspace(root)
            client = FakeDiscriminatorClient(fail_once={1001})
            args = self._args(config_path, targets, candidates, features)
            self.assertEqual(run(args, client_factory=lambda: client), "partial")
            self.assertEqual(client.calls, [1001, 1002])
            raw_path = root / "discriminator-runs" / "test-run" / "raw" / "responses.jsonl"
            self.assertEqual(len(read_jsonl(raw_path)), 2)

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
            self.assertEqual(len(read_jsonl(raw_path)), 2)

            retry = self._args(
                config_path,
                targets,
                candidates,
                features,
                resume=True,
                retry_failed=True,
            )
            self.assertEqual(run(retry, client_factory=lambda: client), "completed")
            self.assertEqual(client.calls, [1001, 1002, 1001])
            self.assertEqual(len(read_jsonl(raw_path)), 3)
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
