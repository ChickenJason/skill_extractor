from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
from instance_discriminator.RunInstanceDiscriminator import run  # noqa: E402
from instance_discriminator.ValidateInstanceDiscriminatorRun import validate  # noqa: E402
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
                "demo_idx": idx,
                "selected_rank": rank,
                "similarity": round(0.9 - rank / 100, 10),
                "status": decision["status"],
                "existence_score": 1.0 if rank <= 8 else 0.8,
                "sentence": decision["sentence"],
                "trfs": ["transferable skill"] if decision["status"] == "accepted" else [],
            }
        )
    return {
        "idx": target["idx"],
        "sentence": target["sentence"],
        "leave_one_out": True,
        "self_excluded": True,
        "selected": selected,
    }


def _prepare_workspace(root: Path) -> tuple[Path, Path, str]:
    target_run_id = "fixture-target"
    target_root = root / "target-runs" / target_run_id
    (target_root / "retrieval").mkdir(parents=True)
    (target_root / "parsed").mkdir()
    decisions_path = root / "self-run" / "selected" / "decisions.jsonl"
    decisions = [_decision(idx) for idx in range(1, 17)]
    atomic_write_jsonl(decisions_path, decisions)
    targets = [_target(1001), _target(1002, review=True)]
    atomic_write_jsonl(target_root / "retrieval" / "records.jsonl", [_retrieval(x) for x in targets])
    atomic_write_jsonl(target_root / "parsed" / "records.jsonl", targets)
    decision_item = {
        "path": str(decisions_path.resolve()),
        "sha256": sha256_file(decisions_path),
        "bytes": decisions_path.stat().st_size,
    }
    target_source = {"trf_run_id": "fixture-offline", "files": {"decisions": decision_item}}
    atomic_write_json(target_root / "source_snapshot.json", target_source)
    output_paths = {
        "source_snapshot.json": target_root / "source_snapshot.json",
        "retrieval/records.jsonl": target_root / "retrieval" / "records.jsonl",
        "parsed/records.jsonl": target_root / "parsed" / "records.jsonl",
    }
    target_manifest = {
        "schema_version": 1,
        "pipeline_version": "trf-target-extractor-v1",
        "run_id": target_run_id,
        "status": "completed",
        "source_unchanged": True,
        "compatibility": {
            "source": {"files": {"decisions": decision_item}},
            "targets": {"indexes": [item["idx"] for item in targets]},
        },
        "outputs": {
            relative: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for relative, path in output_paths.items()
        },
    }
    atomic_write_json(target_root / "manifest.json", target_manifest)

    config = _base_config()
    config["source"]["target_runs_root"] = str(root / "target-runs")
    config["output"]["runs_root"] = str(root / "discriminator-runs")
    config_path = root / "instance_discriminator.json"
    atomic_write_json(config_path, config)
    return config_path, decisions_path, target_run_id


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
    def _args(config: Path, target_run_id: str, **overrides: Any) -> Namespace:
        values = {
            "config": str(config),
            "target_trf_run_id": target_run_id,
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
        decisions = {idx: _decision(idx) for idx in range(1, 17)}
        return build_candidate_record(target, _retrieval(target), decisions, 16)

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
            config_path, decisions_path, target_run_id = _prepare_workspace(Path(directory))
            config = load_discriminator_config(config_path)
            snapshot = snapshot_sources(config, target_run_id)
            with decisions_path.open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(InstanceDiscriminatorError, "decisions hash mismatch"):
                assert_sources_unchanged(config, target_run_id, snapshot)

    def test_candidate_contract_rejects_count_status_and_sentence_mismatch(self) -> None:
        target = _target(1001)
        retrieval = _retrieval(target)
        decisions = {idx: _decision(idx) for idx in range(1, 17)}
        with self.assertRaisesRegex(InstanceDiscriminatorError, "exactly 16"):
            build_candidate_record(target, {**retrieval, "selected": retrieval["selected"][:-1]}, decisions, 16)
        broken = json.loads(json.dumps(retrieval))
        broken["selected"][0]["status"] = "negative"
        with self.assertRaisesRegex(InstanceDiscriminatorError, "status mismatch"):
            build_candidate_record(target, broken, decisions, 16)
        broken = json.loads(json.dumps(retrieval))
        broken["sentence"] = "different"
        with self.assertRaisesRegex(InstanceDiscriminatorError, "sentence mismatch"):
            build_candidate_record(target, broken, decisions, 16)

    def test_prompt_target_is_leave_one_out_and_contains_only_allowed_evidence(self) -> None:
        record = self._candidate_record()
        prompt = build_discriminator_prompt(record, 60000)
        payload = json.loads(prompt["messages"][1]["content"])["input"]
        self.assertEqual(
            set(payload["target"]), {"sentence", "entity_types", "target_trfs"}
        )
        self.assertNotIn(record["idx"], prompt["demo_indexes"])
        serialized_target = json.dumps(payload["target"])
        for forbidden in ("decision", "status", "spans", "has_skill"):
            self.assertNotIn(forbidden, serialized_target)

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
            target, _retrieval(target), {idx: _decision(idx) for idx in range(1, 17)}, 16
        )
        result = build_selected_record(
            record, self._judgments(record), _base_config()["gate"], CHAT_MODEL
        )
        self.assertEqual(
            result["review_reasons"][:2],
            ["upstream_target_trf_needs_review", "skill_type_without_target_trfs"],
        )
        self.assertEqual(result["upstream_review_reasons"], ["upstream_fixture_review"])

    def test_prepare_only_never_creates_client_and_validator_rebuilds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path, _, target_run_id = _prepare_workspace(root)

            def forbidden_client() -> Any:
                raise AssertionError("prepare-only instantiated a client")

            args = self._args(
                config_path,
                target_run_id,
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
            config_path, _, target_run_id = _prepare_workspace(root)
            client = FakeDiscriminatorClient()
            args = self._args(config_path, target_run_id)
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
            config_path, _, target_run_id = _prepare_workspace(root)
            client = FakeDiscriminatorClient(fail_once={1001})
            args = self._args(config_path, target_run_id)
            self.assertEqual(run(args, client_factory=lambda: client), "partial")
            self.assertEqual(client.calls, [1001, 1002])
            raw_path = root / "discriminator-runs" / "test-run" / "raw" / "responses.jsonl"
            self.assertEqual(len(read_jsonl(raw_path)), 2)

            no_retry = self._args(
                config_path,
                target_run_id,
                allow_network=False,
                resume=True,
                confirm_full_run=False,
            )
            self.assertEqual(run(no_retry, client_factory=lambda: client), "partial")
            self.assertEqual(client.calls, [1001, 1002])
            self.assertEqual(len(read_jsonl(raw_path)), 2)

            retry = self._args(
                config_path,
                target_run_id,
                resume=True,
                retry_failed=True,
            )
            self.assertEqual(run(retry, client_factory=lambda: client), "completed")
            self.assertEqual(client.calls, [1001, 1002, 1001])
            self.assertEqual(len(read_jsonl(raw_path)), 3)
            self.assertEqual(validate(config_path, "test-run")["status"], "valid_completed")

    def test_safety_flags_fail_before_run_creation(self) -> None:
        args = self._args(Path("missing.json"), "source", confirm_full_run=False)
        with self.assertRaisesRegex(InstanceDiscriminatorError, "confirm-full-run"):
            run(args)
        args.confirm_full_run = True
        args.retry_failed = True
        with self.assertRaisesRegex(InstanceDiscriminatorError, "requires --resume"):
            run(args)


if __name__ == "__main__":
    unittest.main()
