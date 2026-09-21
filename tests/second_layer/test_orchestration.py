from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from typing import Any
from unittest.mock import patch


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
from second_layer.assembler import assemble_context_records  # noqa: E402
from second_layer.common import SecondLayerError  # noqa: E402
from second_layer.runner import (  # noqa: E402
    build_exemplar_command,
    build_trf_command,
    execute_parallel_processes,
    run,
)
from second_layer.validator import validate  # noqa: E402


def _target() -> dict[str, Any]:
    return {
        "schema_version": "sentence-record-v1",
        "dataset_id": "targets",
        "record_id": "1",
        "source_sha256": "a" * 64,
        "idx": 1,
        "sentence": "需要熟练使用 Python 进行数据分析。",
    }


def _trf_record() -> dict[str, Any]:
    return {
        **_target(),
        "schema_version": "feature-records-v1",
        "status": "complete",
        "entity_types": ["Skill"],
        "trfs": [
            {
                "raw_text": "Python",
                "text": "Python",
                "normalized_text": "Python",
                "first_seen_order": 1,
                "in_main_bank_exact": True,
                "in_main_bank_casefold": True,
                "appears_in_target_exact": True,
                "appears_in_target_casefold": True,
            }
        ],
        "retrieval_count": 16,
        "review_reasons": [],
        "models": {"embedding": "fixture-bert", "chat": "fixture-qwen"},
    }


def _selected_record() -> dict[str, Any]:
    selected = [
        {
            "demo_dataset_id": "demos",
            "demo_record_id": "7",
            "demo_source_sha256": "b" * 64,
            "demo_idx": 7,
            "retrieval_rank": 1,
            "sentence": "熟练使用 Python 完成数据清洗。",
            "status": "accepted",
            "skill_spans": ["Python"],
            "accepted_span_details": [
                {"text": "Python", "start": 5, "end": 11, "exact_votes": 5}
            ],
            "pseudo_trfs": ["Python"],
            "similarity": 0.91,
            "existence_score": 1.0,
            "helpfulness_score": 5,
            "role": "supporting",
            "reason_codes": ["span_pattern_match"],
            "reason": "技能边界和目标句相近。",
            "gate_rank": 1,
        }
    ]
    return {
        **_target(),
        "schema_version": "instance-selection-v1",
        "status": "complete",
        "target_evidence": {
            "entity_types": [],
            "trfs": [],
            "feature_context": "absent",
        },
        "candidate_count": 16,
        "eligible_count": 1,
        "selected_count": 1,
        "selected": selected,
        "rejected_demo_ids": list(range(1, 7)) + list(range(8, 17)),
        "review_reasons": [],
        "feature_review_reasons": [],
        "models": {"chat": "fixture-qwen"},
    }


def _provenance() -> dict[str, Any]:
    return {
        "trf_run_id": "run-trf",
        "exemplar_run_id": "run-examples",
        "artifacts": {},
    }


class CommandAndParallelTests(unittest.TestCase):
    @staticmethod
    def _args(**overrides: Any) -> Namespace:
        values = {
            "target_mode": "independent",
            "limit": 3,
            "reuse_embeddings_from": "prior-embeddings",
            "allow_model_download": True,
            "allow_network": True,
            "confirm_full_run": False,
            "retry_failed": True,
        }
        values.update(overrides)
        return Namespace(**values)

    def test_commands_derive_child_ids_and_keep_features_absent(self) -> None:
        args = self._args()
        trf = build_trf_command(
            args,
            Path("trf.json"),
            "parent-trf",
            Path("targets.json"),
            prepare_only=False,
            resume=True,
        )
        exemplar = build_exemplar_command(
            args,
            Path("disc.json"),
            "parent-examples",
            Path("frozen-targets.jsonl"),
            Path("frozen-candidates.jsonl"),
            resume=True,
        )
        self.assertIn("parent-trf", trf)
        self.assertIn("--resume", trf)
        self.assertIn("--retry-failed", trf)
        self.assertIn("parent-examples", exemplar)
        self.assertIn("frozen-targets.jsonl", exemplar)
        self.assertIn("frozen-candidates.jsonl", exemplar)
        self.assertNotIn("--features", exemplar)
        self.assertNotIn("--prepare-only", exemplar)

    def test_parallel_executor_starts_both_before_waiting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.started"
            second = root / "second.started"

            def barrier_command(own: Path, peer: Path) -> list[str]:
                source = (
                    "from pathlib import Path\n"
                    "import sys, time\n"
                    f"own = Path({str(own)!r})\n"
                    f"peer = Path({str(peer)!r})\n"
                    "own.write_text('started', encoding='utf-8')\n"
                    "deadline = time.time() + 3\n"
                    "while not peer.exists() and time.time() < deadline:\n"
                    "    time.sleep(0.01)\n"
                    "sys.exit(0 if peer.exists() else 9)\n"
                )
                return [sys.executable, "-c", source]

            results = execute_parallel_processes(
                {
                    "first": (barrier_command(first, second), root / "first.log"),
                    "second": (barrier_command(second, first), root / "second.log"),
                }
            )
            self.assertEqual({item["returncode"] for item in results.values()}, {0})
            self.assertTrue(first.is_file())
            self.assertTrue(second.is_file())


class AssemblyTests(unittest.TestCase):
    def test_context_contains_both_independent_experts(self) -> None:
        records = assemble_context_records(
            [_target()], [_trf_record()], [_selected_record()], _provenance()
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["schema_version"], "second-layer-context-v1")
        self.assertEqual(records[0]["assembly_status"], "ready")
        self.assertEqual(records[0]["trf_context"]["entity_types"], ["Skill"])
        self.assertEqual(
            records[0]["exemplar_context"]["feature_context"], "absent"
        )
        self.assertEqual(
            records[0]["exemplar_context"]["selected"][0]["skill_spans"],
            ["Python"],
        )

    def test_assembly_fails_closed_on_identity_text_hash_or_duplicate(self) -> None:
        mutations = []
        for key, value in (
            ("record_id", "other"),
            ("sentence", "另一句话"),
            ("source_sha256", "c" * 64),
        ):
            record = _trf_record()
            record[key] = value
            mutations.append(([record], [_selected_record()]))
        mutations.append(([], [_selected_record()]))
        mutations.append(([_trf_record(), _trf_record()], [_selected_record()]))
        for trf_records, exemplar_records in mutations:
            with self.subTest(trf_count=len(trf_records)):
                with self.assertRaises(Exception):
                    assemble_context_records(
                        [_target()], trf_records, exemplar_records, _provenance()
                    )

    def test_assembly_rejects_target_features_in_exemplar_branch(self) -> None:
        exemplar = _selected_record()
        exemplar["target_evidence"] = {
            "feature_context": "present",
            "entity_types": ["Skill"],
            "trfs": [{"normalized_text": "Python"}],
        }
        with self.assertRaises(SecondLayerError):
            assemble_context_records(
                [_target()], [_trf_record()], [exemplar], _provenance()
            )


class OrchestrationLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input_path = self.root / "input.json"
        atomic_write_json(self.input_path, [_target()["sentence"]])
        self.trf_runs = self.root / "trf-runs"
        self.exemplar_runs = self.root / "exemplar-runs"
        self.parent_runs = self.root / "second-layer-runs"
        self.trf_config = self.root / "trf.json"
        self.exemplar_config = self.root / "exemplar.json"
        self.config = self.root / "second-layer.json"
        atomic_write_json(
            self.trf_config,
            {
                "output": {"runs_root": str(self.trf_runs)},
                "target": {
                    "targets": {"independent_input": str(self.input_path)}
                },
            },
        )
        atomic_write_json(
            self.exemplar_config,
            {"output": {"runs_root": str(self.exemplar_runs)}},
        )
        atomic_write_json(
            self.config,
            {
                "schema_version": 1,
                "module": "second_layer",
                "pipeline_version": "second-layer-parallel-v1",
                "children": {
                    "trf_config": str(self.trf_config),
                    "instance_discriminator_config": str(self.exemplar_config),
                },
                "output": {"runs_root": str(self.parent_runs)},
            },
        )
        self.run_id = "fixture-layer2"
        self.trf_root = self.trf_runs / f"{self.run_id}-trf"
        self.exemplar_root = self.exemplar_runs / f"{self.run_id}-examples"
        self.parent_root = self.parent_runs / self.run_id
        self.child_validation = {
            "schema_version": "second-layer-child-validation-v1",
            "trf": {"run_id": f"{self.run_id}-trf", "status": "valid"},
            "exemplar": {
                "run_id": f"{self.run_id}-examples",
                "status": "valid_completed",
            },
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _args(self, **overrides: Any) -> Namespace:
        values = {
            "config": str(self.config),
            "run_id": self.run_id,
            "target_mode": "independent",
            "input": str(self.input_path),
            "limit": 1,
            "prepare_only": False,
            "allow_model_download": False,
            "allow_network": True,
            "confirm_full_run": False,
            "resume": False,
            "retry_failed": False,
            "reuse_embeddings_from": None,
        }
        values.update(overrides)
        return Namespace(**values)

    def _fake_prepare(
        self, command: list[str], log_path: Path, name: str
    ) -> dict[str, Any]:
        self.assertEqual(name, "prepare_trf")
        self.assertIn("--prepare-only", command)
        atomic_write_json(
            self.trf_root / "manifest.json",
            {"status": "partial", "network_called": False},
        )
        atomic_write_json(
            self.trf_root / "target" / "manifest.json",
            {"stages": {"embed_and_retrieve": "completed"}},
        )
        atomic_write_jsonl(
            self.trf_root / "target" / "targets" / "records.jsonl", [_target()]
        )
        atomic_write_jsonl(
            self.trf_root / "target" / "retrieval" / "records.jsonl",
            [
                {
                    **_target(),
                    "schema_version": "retrieval-record-v1",
                    "selected": [],
                }
            ],
        )
        return {
            "name": name,
            "command": command,
            "started_at": "fixture-start",
            "completed_at": "fixture-end",
            "duration_seconds": 0.0,
            "returncode": 0,
            "log": str(log_path),
        }

    def test_prepare_partial_retry_and_deterministic_validation(self) -> None:
        status = run(
            self._args(prepare_only=True), command_executor=self._fake_prepare
        )
        self.assertEqual(status, "partial")
        manifest = load_json(self.parent_root / "manifest.json")
        self.assertEqual(manifest["stages"]["prepare_shared"], "completed")
        self.assertEqual(manifest["summary"]["reason"], "prepare_only")
        frozen_targets = self.parent_root / "shared" / "targets.jsonl"
        frozen_candidates = self.parent_root / "shared" / "candidates.jsonl"
        target_hash = sha256_file(frozen_targets)
        candidate_hash = sha256_file(frozen_candidates)

        calls = 0

        def fake_parallel(
            specifications: dict[str, tuple[list[str], Path]],
        ) -> dict[str, dict[str, Any]]:
            nonlocal calls
            calls += 1
            if calls == 1:
                self.assertEqual(set(specifications), {"trf", "exemplar"})
                self.assertNotIn("--features", specifications["exemplar"][0])
                atomic_write_jsonl(
                    self.trf_root / "target" / "targets" / "records.jsonl",
                    [{**_target(), "sentence": "TRF Resume 重写的原始文件"}],
                )
                atomic_write_jsonl(
                    self.trf_root / "target" / "retrieval" / "records.jsonl",
                    [{**_target(), "selected": ["rewritten"]}],
                )
                atomic_write_jsonl(
                    self.trf_root / "target" / "parsed" / "records.jsonl",
                    [_trf_record()],
                )
                atomic_write_json(
                    self.trf_root / "manifest.json",
                    {"status": "completed", "network_called": True},
                )
                atomic_write_json(
                    self.exemplar_root / "manifest.json",
                    {"status": "partial", "network_called": True},
                )
            else:
                self.assertEqual(set(specifications), {"exemplar"})
                exemplar_command = specifications["exemplar"][0]
                self.assertIn("--resume", exemplar_command)
                self.assertIn("--retry-failed", exemplar_command)
                self.assertNotIn("--features", exemplar_command)
                atomic_write_jsonl(
                    self.exemplar_root / "selected" / "records.jsonl",
                    [_selected_record()],
                )
                atomic_write_json(
                    self.exemplar_root / "manifest.json",
                    {"status": "completed", "network_called": True},
                )
            return {
                name: {
                    "name": name,
                    "command": command,
                    "started_at": "fixture-start",
                    "completed_at": "fixture-end",
                    "duration_seconds": 0.0,
                    "returncode": 0,
                    "log": str(log_path),
                }
                for name, (command, log_path) in specifications.items()
            }

        status = run(
            self._args(resume=True),
            parallel_executor=fake_parallel,
            child_validator=lambda *_: self.child_validation,
        )
        self.assertEqual(status, "partial")
        self.assertFalse((self.parent_root / "context" / "records.jsonl").exists())
        self.assertEqual(sha256_file(frozen_targets), target_hash)
        self.assertEqual(sha256_file(frozen_candidates), candidate_hash)

        status = run(
            self._args(resume=True, retry_failed=True),
            parallel_executor=fake_parallel,
            child_validator=lambda *_: self.child_validation,
        )
        self.assertEqual(status, "completed")
        context = read_jsonl(self.parent_root / "context" / "records.jsonl")
        self.assertEqual(len(context), 1)
        self.assertEqual(context[0]["assembly_status"], "ready")
        self.assertEqual(context[0]["exemplar_context"]["feature_context"], "absent")
        self.assertEqual(sha256_file(frozen_targets), target_hash)
        self.assertEqual(sha256_file(frozen_candidates), candidate_hash)

        with patch(
            "second_layer.validator.validate_trf",
            return_value=self.child_validation["trf"],
        ), patch(
            "second_layer.validator.validate_exemplar",
            return_value=self.child_validation["exemplar"],
        ):
            result = validate(self.config, self.run_id)
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["ready"], 1)

        with self.assertRaises(SecondLayerError):
            run(
                self._args(resume=True),
                parallel_executor=fake_parallel,
                child_validator=lambda *_: self.child_validation,
            )

        context_path = self.parent_root / "context" / "records.jsonl"
        atomic_write_jsonl(
            context_path,
            [{**context[0], "assembly_status": "tampered"}],
        )
        with patch(
            "second_layer.validator.validate_trf",
            return_value=self.child_validation["trf"],
        ), patch(
            "second_layer.validator.validate_exemplar",
            return_value=self.child_validation["exemplar"],
        ):
            with self.assertRaises(SecondLayerError):
                validate(self.config, self.run_id)

    def test_retry_failed_requires_resume(self) -> None:
        with self.assertRaises(SecondLayerError):
            run(self._args(retry_failed=True), command_executor=self._fake_prepare)


if __name__ == "__main__":
    unittest.main()
