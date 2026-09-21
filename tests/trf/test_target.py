from __future__ import annotations

import hashlib
import json
import shutil
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

from common.io_utils import load_json, read_jsonl  # noqa: E402
from trf.target.runner import run  # noqa: E402
from trf.target.validator import validate  # noqa: E402
from trf.target.common import (  # noqa: E402
    TargetTRFError,
    assert_sources_unchanged,
    snapshot_sources,
)
from trf.target.pipeline import (  # noqa: E402
    build_parsed_record,
    build_prompts,
    build_skill_prediction_prompt,
    load_source_bundle,
    parse_entity_types,
    parse_target_trfs,
    retrieve_demonstrations,
    validate_embedding_vector,
    validate_independent_targets,
)
from tests.trf.fixtures import (  # noqa: E402
    DEMONSTRATIONS,
    create_fake_completed_offline,
    prepare_generated_target_config,
    prepare_v06_trf_config,
)


class FakeTargetClient:
    def __init__(
        self,
        fail_once: set[int] | None = None,
        prediction_invalid_once: set[int] | None = None,
        prediction_fail_always: set[int] | None = None,
    ) -> None:
        self.embedding_calls: list[list[str]] = []
        self.chat_calls: list[list[dict[str, str]]] = []
        self.fail_once = set(fail_once or set())
        self.prediction_invalid_once = set(prediction_invalid_once or set())
        self.prediction_fail_always = set(prediction_fail_always or set())
        self.failed: set[int] = set()
        self.prediction_failed: set[int] = set()
        self.prediction_calls: list[int] = []

    @staticmethod
    def _vector(text: str, dimensions: int) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector = [0.0] * dimensions
        vector[0] = 1.0
        for offset, value in enumerate(digest, start=1):
            vector[offset] = (value + 1) / 256.0
        return vector

    def embeddings_detailed(
        self,
        texts: list[str],
        *,
        model: str,
        dimensions: int,
        batch_size: int,
    ) -> dict[str, Any]:
        self.embedding_calls.append(list(texts))
        return {
            "vectors": [self._vector(text, dimensions) for text in texts],
            "batches": [
                {
                    "start": 0,
                    "count": len(texts),
                    "attempts": 1,
                    "latency_ms": 0,
                    "response_id": f"mock-embedding-{len(self.embedding_calls)}",
                    "model": model,
                    "usage": {"prompt_tokens": len(texts), "total_tokens": len(texts)},
                }
            ],
        }

    @staticmethod
    def _target_idx(content: str) -> int:
        marker = "TARGET-ID:"
        if marker not in content:
            return -1
        suffix = content.split(marker, 1)[1]
        return int(suffix.split()[0])

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> dict[str, Any]:
        self.chat_calls.append(messages)
        combined = "\n".join(item["content"] for item in messages)
        idx = self._target_idx(combined)
        if "TRF expert's final exact Skill span extractor" in combined:
            self.prediction_calls.append(idx)
            if idx in self.prediction_fail_always:
                content = "not-json"
            elif (
                idx in self.prediction_invalid_once
                and idx not in self.prediction_failed
            ):
                self.prediction_failed.add(idx)
                content = "not-json"
            else:
                payload = json.loads(messages[1]["content"])
                content = json.dumps(
                    {"annotated_sentence": payload["target_sentence"]}, ensure_ascii=False
                )
        elif idx in self.fail_once and idx not in self.failed:
            self.failed.add(idx)
            content = "not-json"
        elif len(messages) == 1:
            content = '{"entity_types":["Skill"]}'
        else:
            content = '{"trfs":["communication","open generated feature","communication"]}'
        return {
            "content": content,
            "finish_reason": "stop",
            "latency_ms": 0,
            "attempts": 1,
            "response_id": f"mock-chat-{len(self.chat_calls)}",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }


class TargetTRFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_workspace = tempfile.TemporaryDirectory()
        source_root = Path(cls.source_workspace.name)
        offline_path, cls.offline_config = prepare_v06_trf_config(source_root)
        cls.offline_run_id = "target-test-offline"
        create_fake_completed_offline(
            offline_path, DEMONSTRATIONS, cls.offline_run_id
        )
        _, cls.base_target_config = prepare_generated_target_config(
            source_root, cls.offline_config, cls.offline_run_id
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.source_workspace.cleanup()

    def test_template_is_rejected_by_direct_target_runner(self) -> None:
        args = self._args(PROJECT_ROOT / "config" / "trf.json")
        with self.assertRaisesRegex(TargetTRFError, "module template"):
            run(args)

    def test_formal_index_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Path(self.base_target_config["source_trf"]["run_root"])
            copied = root / "offline"
            shutil.copytree(source, copied)
            config = json.loads(json.dumps(self.base_target_config))
            config["source_trf"]["run_root"] = str(copied)
            pseudo_path = copied / config["source_trf"]["files"]["pseudo_trfs"]["path"]
            pseudo = read_jsonl(pseudo_path)
            pseudo_path.write_text(
                "\n".join(json.dumps(item) for item in pseudo[:-1]) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(TargetTRFError, "formal decision indexes"):
                load_source_bundle(config)

    def _workspace(self, directory: str, targets: list[dict[str, Any]] | None = None):
        root = Path(directory)
        config = json.loads(json.dumps(self.base_target_config))
        config["output"]["runs_root"] = str(root / "runs")
        config_path = root / "target.generated.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        input_path = root / "targets.json"
        input_path.write_text(
            json.dumps(
                targets
                or [
                    {"idx": 9001, "sentence": "TARGET-ID:9001 communication role"},
                    {"idx": 9002, "sentence": "TARGET-ID:9002 project role"},
                ]
            ),
            encoding="utf-8",
        )
        return config_path, input_path, root / "runs"

    @staticmethod
    def _args(config: Path, **overrides: Any) -> Namespace:
        values = {
            "config": str(config),
            "run_id": "test-run",
            "mode": "independent",
            "input": None,
            "feedback": None,
            "limit": None,
            "prepare_only": False,
            "allow_network": True,
            "resume": False,
            "retry_failed": False,
            "reuse_embeddings_from": None,
            "confirm_full_run": True,
        }
        values.update(overrides)
        return Namespace(**values)

    def test_optional_feedback_is_validated_and_hash_locked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _, _ = self._workspace(directory)
            feedback = root / "feedback.jsonl"
            feedback.write_text(
                json.dumps(
                    {
                        "schema_version": "trf-feedback-v1",
                        "dataset_id": "feedback-set",
                        "record_id": "1",
                        "source_sha256": "c" * 64,
                        "idx": 1,
                        "decision": "retain",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            loaded = load_json(config)
            snapshot = snapshot_sources(loaded, feedback)
            self.assertEqual(snapshot["feedback"]["context"], "present")
            feedback.write_text(feedback.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(TargetTRFError, "locked"):
                assert_sources_unchanged(loaded, snapshot)

    def test_independent_input_is_strict(self) -> None:
        self.assertEqual(
            validate_independent_targets([{"idx": 2, "sentence": "x"}])[0]["idx"],
            2,
        )
        invalid = [
            [{"idx": 1}],
            [{"idx": 1, "sentence": " "}],
            [{"idx": 1, "sentence": "a"}, {"idx": 1, "sentence": "b"}],
        ]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(TargetTRFError):
                validate_independent_targets(value)

    def test_retrieval_uses_k_then_reliability_and_self_exclusion(self) -> None:
        demos = [
            {
                "schema_version": "trf-demonstration-v1",
                "dataset_id": "demo-set",
                "record_id": str(idx),
                "source_sha256": "a" * 64,
                "idx": idx,
                "sentence": str(idx),
                "status": "negative" if idx == 2 else "accepted",
                "existence_score": score,
                "trfs": [] if idx == 2 else ["x"],
                "skill_spans": [] if idx == 2 else ["x"],
                "accepted_span_details": [],
            }
            for idx, score in [(1, 0.8), (2, 1.0), (3, 1.0), (4, 0.8)]
        ]
        vectors = {idx: [1.0, float(5 - idx)] for idx in range(1, 5)}
        result = retrieve_demonstrations(
            {
                "dataset_id": "demo-set",
                "record_id": "1",
                "source_sha256": "a" * 64,
                "idx": 1,
                "sentence": "target",
            },
            demos,
            [1.0, 4.0],
            vectors,
            leave_one_out=True,
            nearest_neighbors=3,
            selected_count=2,
            similarity_decimals=10,
        )
        self.assertTrue(result["self_excluded"])
        self.assertEqual(len(result["neighbors"]), 3)
        self.assertEqual([item["demo_idx"] for item in result["selected"]], [2, 3])
        self.assertEqual(result["selected"][0]["trfs"], [])
        with self.assertRaisesRegex(TargetTRFError, "K=4"):
            retrieve_demonstrations(
                {
                    "dataset_id": "demo-set",
                    "record_id": "1",
                    "source_sha256": "a" * 64,
                    "idx": 1,
                    "sentence": "target",
                },
                demos,
                [1.0, 4.0],
                vectors,
                leave_one_out=True,
                nearest_neighbors=4,
                selected_count=2,
                similarity_decimals=10,
            )

    def test_embedding_contract_rejects_bad_vectors(self) -> None:
        with self.assertRaises(TargetTRFError):
            validate_embedding_vector([1.0], 2)
        with self.assertRaises(TargetTRFError):
            validate_embedding_vector([float("nan"), 1.0], 2)
        with self.assertRaises(TargetTRFError):
            validate_embedding_vector([0.0, 0.0], 2)

    def test_prompt_and_strict_open_trf_parsing(self) -> None:
        retrieval = {
            "selected": [
                {
                    "demo_idx": idx,
                    "selected_rank": idx,
                    "sentence": f'demo "{idx}"',
                    "trfs": [] if idx == 2 else ["skill"],
                }
                for idx in range(1, 17)
            ]
        }
        target = {
            "dataset_id": "target-set",
            "record_id": "7",
            "source_sha256": "b" * 64,
            "idx": 7,
            "sentence": 'target "quoted"',
        }
        prompt = build_prompts(target, retrieval, 60000)
        self.assertEqual(prompt["demo_indexes"], list(range(1, 17)))
        self.assertIn("target \\\"quoted\\\"", prompt["stage1_user"])
        self.assertIn('"trfs":[]', prompt["stage2_user"])
        self.assertEqual(parse_entity_types('{"entity_types":[]}'), [])
        values = parse_target_trfs(
            '{"trfs":["  Ｃ＋＋  ","C++","open OOV"]}',
            "C++ target",
            ["C++"],
        )
        self.assertEqual([item["normalized_text"] for item in values], ["C++", "open OOV"])
        self.assertEqual(values[0]["raw_text"], "  Ｃ＋＋  ")
        self.assertTrue(values[0]["in_main_bank_exact"])
        self.assertFalse(values[1]["appears_in_target_exact"])
        for content in ('not-json', '{"entity_types":["Other"]}', '{"trfs":[1]}'):
            with self.subTest(content=content), self.assertRaises(TargetTRFError):
                if "entity_types" in content or content == "not-json":
                    parse_entity_types(content)
                else:
                    parse_target_trfs(content, "x", ["x"])
        conflict = build_parsed_record(
            target,
            [],
            values,
            "qwen3.7-text-embedding",
            "qwen3.7-plus-2026-05-26",
        )
        self.assertEqual(conflict["status"], "needs_review")
        self.assertEqual(conflict["review_reasons"], ["type_absent_but_trfs_nonempty"])
        prediction_prompt = build_skill_prediction_prompt(target, conflict, 60000)
        payload = json.loads(prediction_prompt["messages"][1]["content"])
        self.assertEqual(payload["target_sentence"], target["sentence"])
        self.assertEqual(payload["inferred_trfs"], ["C++", "open OOV"])
        self.assertNotIn("demonstrations", payload)
        self.assertEqual(
            prediction_prompt["review_reasons"],
            ["type_absent_but_trfs_nonempty", "no_entity_type"],
        )

    def test_online_safety_flags_fail_before_creating_a_run(self) -> None:
        args = self._args(
            Path("missing-config.json"),
            confirm_full_run=False,
        )
        with self.assertRaisesRegex(TargetTRFError, "confirm-full-run"):
            run(args)
        args.confirm_full_run = True
        args.retry_failed = True
        with self.assertRaisesRegex(TargetTRFError, "requires --resume"):
            run(args)

    def test_mock_independent_integration_and_validator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, input_path, runs = self._workspace(directory)
            client = FakeTargetClient()
            args = self._args(config, input=str(input_path))
            self.assertEqual(run(args, client_factory=lambda: client), "completed")
            run_root = runs / "test-run" / "target"
            manifest = load_json(run_root / "manifest.json")
            self.assertEqual(manifest["status"], "completed")
            self.assertTrue(manifest["network_called"])
            parsed = read_jsonl(run_root / "parsed" / "records.jsonl")
            self.assertEqual(len(parsed), 2)
            self.assertEqual(len(parsed[0]["trfs"]), 2)
            predictions = read_jsonl(run_root / "prediction" / "records.jsonl")
            self.assertEqual(len(predictions), 2)
            self.assertTrue(all(item["branch"] == "trf" for item in predictions))
            retrieval = read_jsonl(run_root / "retrieval" / "records.jsonl")
            self.assertTrue(all(len(item["neighbors"]) == 50 for item in retrieval))
            self.assertTrue(all(len(item["selected"]) == 16 for item in retrieval))
            self.assertEqual(
                [len(item) for item in client.embedding_calls[:16]],
                [20] * 15 + [13],
            )
            stage2_calls = [messages for messages in client.chat_calls if len(messages) == 3]
            self.assertEqual(len(stage2_calls), 2)
            self.assertEqual(stage2_calls[0][1]["role"], "assistant")
            result = validate(config, "test-run")
            self.assertEqual(result["status"], "valid")

    def test_leave_one_out_excludes_self(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, _, runs = self._workspace(directory)
            client = FakeTargetClient()
            args = self._args(
                config,
                mode="leave-one-out",
                limit=3,
                input=None,
                confirm_full_run=False,
            )
            self.assertEqual(run(args, client_factory=lambda: client), "completed")
            retrieval = read_jsonl(
                runs / "test-run" / "target" / "retrieval" / "records.jsonl"
            )
            for record in retrieval:
                self.assertTrue(record["self_excluded"])
                self.assertNotIn(record["idx"], [item["demo_idx"] for item in record["neighbors"]])
            self.assertEqual(validate(config, "test-run")["mode"], "leave-one-out")

    def test_partial_resume_and_explicit_failed_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, input_path, runs = self._workspace(directory)
            base = self._args(
                config,
                input=str(input_path),
                allow_network=False,
            )
            self.assertEqual(run(base), "partial")
            run_root = runs / "test-run" / "target"
            initial_manifest = load_json(run_root / "manifest.json")
            self.assertEqual(initial_manifest["status"], "partial")
            self.assertFalse(initial_manifest["network_called"])

            client = FakeTargetClient(fail_once={9001})
            resumed = self._args(
                config,
                input=str(input_path),
                resume=True,
                confirm_full_run=True,
            )
            self.assertEqual(run(resumed, client_factory=lambda: client), "partial")
            raw = read_jsonl(run_root / "raw" / "responses.jsonl")
            calls_after_failure = len(raw)
            parsed = read_jsonl(run_root / "parsed" / "records.jsonl")
            placeholder = next(item for item in parsed if item["idx"] == 9001)
            self.assertEqual(placeholder["trfs"], [])
            self.assertEqual(placeholder["status"], "needs_review")
            self.assertEqual(
                placeholder["review_reasons"], ["trf_extraction_validation_failed"]
            )
            self.assertIn(9001, client.prediction_calls)

            no_retry = self._args(
                config,
                input=str(input_path),
                allow_network=False,
                resume=True,
            )
            self.assertEqual(run(no_retry), "partial")
            self.assertEqual(
                len(read_jsonl(run_root / "raw" / "responses.jsonl")),
                calls_after_failure,
            )

            retry = self._args(
                config,
                input=str(input_path),
                resume=True,
                retry_failed=True,
                confirm_full_run=True,
            )
            self.assertEqual(run(retry, client_factory=lambda: client), "completed")
            latest = {
                item["idx"]: item
                for item in read_jsonl(run_root / "raw" / "responses.jsonl")
            }
            self.assertEqual(set(latest), {9001, 9002})
            self.assertIn(latest[9001]["status"], {"complete", "needs_review"})
            self.assertEqual(validate(config, "test-run")["status"], "valid")

    def test_failed_prediction_resume_does_not_repeat_trf_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, input_path, runs = self._workspace(directory)
            client = FakeTargetClient(prediction_fail_always={9001})
            args = self._args(config, input=str(input_path))
            self.assertEqual(run(args, client_factory=lambda: client), "partial")
            trf_calls = [
                messages for messages in client.chat_calls if len(messages) in {1, 3}
            ]
            self.assertEqual(len(trf_calls), 4)
            self.assertEqual(client.prediction_calls, [9001, 9001, 9001, 9002])
            failures = read_jsonl(
                runs / "test-run" / "target" / "prediction" / "failures.jsonl"
            )
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["idx"], 9001)
            self.assertEqual(
                failures[0]["sentence"], "TARGET-ID:9001 communication role"
            )
            self.assertEqual(failures[0]["failure_kind"], "validation_exhausted")
            self.assertTrue(failures[0]["tolerance_eligible"])
            self.assertEqual(failures[0]["model_output"], "not-json")

            no_retry = self._args(
                config,
                input=str(input_path),
                allow_network=False,
                resume=True,
                confirm_full_run=False,
            )
            self.assertEqual(run(no_retry, client_factory=lambda: client), "partial")
            self.assertEqual(
                len([m for m in client.chat_calls if len(m) in {1, 3}]), 4
            )
            self.assertEqual(client.prediction_calls, [9001, 9001, 9001, 9002])

            client.prediction_fail_always.clear()
            retry = self._args(
                config,
                input=str(input_path),
                resume=True,
                retry_failed=True,
            )
            self.assertEqual(run(retry, client_factory=lambda: client), "completed")
            self.assertEqual(
                len([m for m in client.chat_calls if len(m) in {1, 3}]), 4
            )
            self.assertEqual(
                client.prediction_calls, [9001, 9001, 9001, 9002, 9001]
            )
            self.assertEqual(
                read_jsonl(
                    runs / "test-run" / "target" / "prediction" / "failures.jsonl"
                ),
                [],
            )
            self.assertEqual(validate(config, "test-run")["status"], "valid")

    def test_terminal_prediction_failure_within_three_percent_can_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            targets = [
                {
                    "idx": 9000 + offset,
                    "sentence": f"TARGET-ID:{9000 + offset} skill sentence",
                }
                for offset in range(1, 101)
            ]
            config, input_path, runs = self._workspace(directory, targets)
            client = FakeTargetClient(prediction_fail_always={9001})
            args = self._args(config, input=str(input_path))
            self.assertEqual(run(args, client_factory=lambda: client), "completed")
            run_root = runs / "test-run" / "target"
            failures = read_jsonl(run_root / "prediction" / "failures.jsonl")
            self.assertEqual([item["idx"] for item in failures], [9001])
            self.assertEqual(
                load_json(run_root / "manifest.json")["summary"][
                    "prediction_tolerance"
                ]["failure_rate"],
                0.01,
            )
            self.assertEqual(
                load_json(run_root / "manifest.json")["summary"][
                    "prediction_tolerance"
                ]["allowed_failure_count"],
                3,
            )
            result = validate(config, "test-run")
            self.assertEqual(result["status"], "valid")
            self.assertEqual(result["predictions"], 99)
            self.assertEqual(result["prediction_failures"], 1)

    def test_embedding_reuse_requires_exact_target_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, input_path, runs = self._workspace(
                directory,
                [{"idx": 9001, "sentence": "TARGET-ID:9001 unique target"}],
            )
            client = FakeTargetClient()
            first = self._args(
                config,
                run_id="first",
                input=str(input_path),
            )
            self.assertEqual(run(first, client_factory=lambda: client), "completed")

            reused = self._args(
                config,
                run_id="same-target",
                input=str(input_path),
                prepare_only=True,
                allow_network=False,
                reuse_embeddings_from="first",
            )
            self.assertEqual(run(reused), "partial")
            reused_root = runs / "same-target" / "target"
            reused_manifest = load_json(reused_root / "manifest.json")
            self.assertFalse(reused_manifest["network_called"])
            target_cache = read_jsonl(
                reused_root / "embeddings" / "targets.jsonl"
            )
            self.assertEqual(target_cache[0]["reused_from"], "first")

            changed_input = Path(directory) / "changed.json"
            changed_input.write_text(
                json.dumps([{"idx": 9002, "sentence": "a completely new target"}]),
                encoding="utf-8",
            )
            changed = self._args(
                config,
                run_id="changed-target",
                input=str(changed_input),
                prepare_only=True,
                allow_network=False,
                reuse_embeddings_from="first",
            )
            self.assertEqual(run(changed), "partial")
            self.assertEqual(
                len(
                    read_jsonl(
                        runs
                        / "changed-target"
                        / "target"
                        / "embeddings"
                        / "demonstrations.jsonl"
                    )
                ),
                313,
            )
            self.assertFalse(
                (
                    runs
                    / "changed-target"
                    / "target"
                    / "embeddings"
                    / "targets.jsonl"
                ).exists()
            )


if __name__ == "__main__":
    unittest.main()
