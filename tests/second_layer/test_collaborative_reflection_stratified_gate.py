from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import atomic_write_json, read_jsonl  # noqa: E402
from second_layer.collaborative_reflection_json_schema.online import reflect_one  # noqa: E402
from second_layer.collaborative_reflection_json_schema.pipeline import (  # noqa: E402
    assemble_context_records,
    build_exemplar_records,
    build_prompt,
    build_trf_record,
)
from second_layer.collaborative_reflection_stratified_gate.evaluator import evaluate_gate  # noqa: E402
from second_layer.collaborative_reflection_stratified_gate.selection import (  # noqa: E402
    load_gate_spec,
    select_gate_sources,
)
from tests.second_layer.test_collaborative_reflection import (  # noqa: E402
    _exemplar_payload,
    _input,
    _trf_payload,
)


SPEC_PATH = PROJECT_ROOT / "config" / "gates" / "collaborative_reflection_stratified12_v1.json"
CONTEXT_PATH = PROJECT_ROOT / "output" / "second_layer" / "skill-concurrent-v1" / "context" / "records.jsonl"
CANDIDATE_PATH = PROJECT_ROOT / "output" / "instance_discriminator" / "skill-concurrent-v1-examples" / "candidates" / "records.jsonl"
JUDGMENT_PATH = PROJECT_ROOT / "output" / "instance_discriminator" / "skill-concurrent-v1-examples" / "parsed" / "judgments.jsonl"
CHAT = {"model": "fixture", "temperature": 0.0, "max_tokens": 4096, "max_reason_characters": 240}


class SelectionTests(unittest.TestCase):
    def test_real_base_reconstructs_exact_twelve_strata(self) -> None:
        spec = load_gate_spec(SPEC_PATH)
        selected = select_gate_sources(
            read_jsonl(CONTEXT_PATH), read_jsonl(CANDIDATE_PATH), read_jsonl(JUDGMENT_PATH), spec
        )
        self.assertEqual([item["idx"] for item in selected[3]], [13, 17, 50, 85, 100, 115, 139, 147, 180, 275, 312, 323])
        self.assertEqual(selected[4], spec["acceptance"]["minimum_strata_counts"])
        self.assertEqual(sum(len(item["baseline_context"]["trf_context"]["trfs"]) for item in selected[3]), 48)

    def test_selection_fails_closed_on_identity_or_strata_mismatch(self) -> None:
        spec = load_gate_spec(SPEC_PATH)
        contexts = read_jsonl(CONTEXT_PATH)
        candidates = read_jsonl(CANDIDATE_PATH)
        judgments = read_jsonl(JUDGMENT_PATH)
        for mutation in ("sentence", "source", "strata", "anchor"):
            changed = copy.deepcopy(spec)
            if mutation == "sentence":
                changed["targets"][0]["sentence"] += " changed"
            elif mutation == "source":
                changed["targets"][0]["source_sha256"] = "0" * 64
            elif mutation == "strata":
                changed["targets"][0]["strata"].remove("candidate_mixed")
            else:
                changed["targets"][0]["trf_anchors"]["must_keep"] = ["not-a-baseline-trf"]
            with self.subTest(mutation=mutation), self.assertRaises(Exception):
                select_gate_sources(contexts, candidates, judgments, changed)

    def test_spec_rejects_duplicates_and_more_than_sixteen(self) -> None:
        spec = load_gate_spec(SPEC_PATH)
        variants = []
        duplicate = copy.deepcopy(spec)
        duplicate["targets"].append(copy.deepcopy(duplicate["targets"][-1]))
        duplicate["acceptance"]["target_count"] += 1
        variants.append(duplicate)
        too_many = copy.deepcopy(spec)
        too_many["maximum_targets"] = 17
        variants.append(too_many)
        with tempfile.TemporaryDirectory() as directory:
            for position, value in enumerate(variants):
                path = Path(directory) / f"bad-{position}.json"
                atomic_write_json(path, value)
                with self.subTest(position=position), self.assertRaises(Exception):
                    load_gate_spec(path)

    def test_gate_oracle_does_not_enter_prompts(self) -> None:
        spec = load_gate_spec(SPEC_PATH)
        selected = select_gate_sources(
            read_jsonl(CONTEXT_PATH), read_jsonl(CANDIDATE_PATH), read_jsonl(JUDGMENT_PATH), spec
        )
        prompts = [build_prompt(item, branch, 60000, 240) for item in selected[3] for branch in ("trf", "exemplar")]
        serialized = json.dumps(prompts, ensure_ascii=False)
        self.assertNotIn(spec["gate_id"], serialized)
        self.assertNotIn("must_keep", serialized)
        self.assertNotIn("must_drop", serialized)
        for target in spec["targets"]:
            self.assertNotIn(target["anchor_rationale"], serialized)


class EvaluationTests(unittest.TestCase):
    @staticmethod
    def _fake_client(payload: dict[str, Any]) -> Any:
        class FakeClient:
            def chat(self, _messages: Any, **_kwargs: Any) -> dict[str, Any]:
                return {
                    "content": json.dumps(payload),
                    "finish_reason": "stop",
                    "latency_ms": 1,
                    "attempts": 1,
                    "response_id": "fixture",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
        return FakeClient()

    def _fixture(
        self,
    ) -> tuple[
        dict[str, Any], list[dict[str, Any]], list[dict[str, Any]],
        list[dict[str, Any]], dict[str, Any],
    ]:
        record = _input()
        second = copy.deepcopy(record["baseline_context"]["trf_context"]["trfs"][0])
        second.update({"raw_text": "tools", "text": "tools", "normalized_text": "tools", "first_seen_order": 2})
        record["baseline_context"]["trf_context"]["trfs"].append(second)
        trf_payload = _trf_payload()
        trf_payload["original_trf_decisions"].append(
            {
                "normalized_text": "tools",
                "action": "drop",
                "confidence": 5,
                "evidence_sources": ["TARGET_TEXT"],
                "evidence_demo_ids": [],
                "reason_codes": ["TOO_GENERIC"],
                "reason": "This generic relation is not supported by the target.",
            }
        )
        trf_prompt = build_prompt(record, "trf", 60000, 240)
        exemplar_prompt = build_prompt(record, "exemplar", 60000, 240)
        trf_raw, trf_parsed = reflect_one(
            branch="trf", record=record, prompt=trf_prompt,
            client=self._fake_client(trf_payload), chat=CHAT, max_repairs=2,
        )
        exemplar_raw, exemplar_parsed = reflect_one(
            branch="exemplar", record=record, prompt=exemplar_prompt,
            client=self._fake_client(_exemplar_payload()), chat=CHAT, max_repairs=2,
        )
        assert trf_parsed is not None and exemplar_parsed is not None
        trf_record = build_trf_record(record, trf_parsed, "fixture")
        exemplar_record, selected_record = build_exemplar_records(
            record, exemplar_parsed,
            {"minimum_helpfulness": 3, "max_selected": 8, "max_supporting": 6, "max_contrastive": 3, "minimum_for_complete": 2},
            "fixture",
        )
        contexts, _ = assemble_context_records(
            [record], [trf_record], [exemplar_record], [selected_record], {"artifacts": {}}
        )
        target = {
            "dataset_id": record["dataset_id"], "record_id": record["record_id"],
            "source_sha256": record["source_sha256"], "idx": record["idx"],
            "sentence": record["sentence"], "strata": ["nonempty_baseline_trf"],
            "trf_anchors": {"must_keep": ["Python"], "must_drop": ["tools"]},
            "anchor_rationale": "fixture",
        }
        spec = {
            "gate_id": "fixture-gate",
            "targets": [target],
            "acceptance": {
                "target_count": 1,
                "minimum_strata_counts": {"nonempty_baseline_trf": 1},
                "required_keep_anchors": 1,
                "required_drop_anchors": 1,
                "required_wrapper_repairs": 0,
                "maximum_total_repairs": 0,
                "maximum_model_calls": 2,
                "required_failed_records": 0,
            },
        }
        return record, contexts, [trf_raw], [exemplar_raw], spec

    def test_gate_passes_valid_keep_drop_and_request_contracts(self) -> None:
        record, contexts, trf_raw, exemplar_raw, spec = self._fixture()
        report, manual = evaluate_gate(
            spec=spec, inputs=[record], contexts=contexts,
            trf_raw=trf_raw, exemplar_raw=exemplar_raw,
            branch_metrics={
                "trf": {"repair_requests": 0, "model_calls": 1, "failed_records": 0},
                "exemplar": {"repair_requests": 0, "model_calls": 1, "failed_records": 0},
            },
            processes={
                "trf": {"started_at": "2026-01-01T00:00:00+00:00", "ended_at": "2026-01-01T00:00:02+00:00"},
                "exemplar": {"started_at": "2026-01-01T00:00:01+00:00", "ended_at": "2026-01-01T00:00:03+00:00"},
            },
            coverage={"nonempty_baseline_trf": 1},
        )
        self.assertEqual(report["gate_status"], "passed")
        self.assertEqual(report["action_counts"], {"keep": 1, "drop": 1})
        self.assertEqual(report["request_audit"]["valid_initial_roots"], 2)
        self.assertEqual(report["request_audit"]["wrapper_repairs"], 0)
        self.assertEqual(len(manual), 1)

    def test_anchor_mismatch_fails_but_keeps_evaluation(self) -> None:
        record, contexts, trf_raw, exemplar_raw, spec = self._fixture()
        contexts[0]["reflected_context"]["trf_context"]["original_trf_decisions"][1]["action"] = "keep"
        report, _ = evaluate_gate(
            spec=spec, inputs=[record], contexts=contexts,
            trf_raw=trf_raw, exemplar_raw=exemplar_raw,
            branch_metrics={
                "trf": {"repair_requests": 0, "model_calls": 1, "failed_records": 0},
                "exemplar": {"repair_requests": 0, "model_calls": 1, "failed_records": 0},
            },
            processes={
                "trf": {"started_at": "2026-01-01T00:00:00+00:00", "ended_at": "2026-01-01T00:00:02+00:00"},
                "exemplar": {"started_at": "2026-01-01T00:00:01+00:00", "ended_at": "2026-01-01T00:00:03+00:00"},
            },
            coverage={"nonempty_baseline_trf": 1},
        )
        self.assertEqual(report["gate_status"], "failed")
        self.assertTrue(any(item.startswith("anchor_mismatch") for item in report["hard_failures"]))


class InterfaceTests(unittest.TestCase):
    def test_powershell_interface_is_exact_selection_not_limit(self) -> None:
        text = (PROJECT_ROOT / "scripts" / "second_layer" / "run_reflection_stratified_gate.ps1").read_text(encoding="utf-8")
        for parameter in ("RunId", "BaseConcurrentRunId", "GateSpecPath", "PrepareOnly", "AllowNetwork", "Resume", "RetryFailed", "PythonExecutable"):
            self.assertIn(f"${parameter}", text)
        self.assertNotIn("[int]$Limit", text)
        self.assertNotIn("ConfirmFullRun", text)


if __name__ == "__main__":
    unittest.main()
