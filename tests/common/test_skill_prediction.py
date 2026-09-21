from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.skill_prediction import (  # noqa: E402
    RECOVERY_REVIEW_REASON,
    SkillPredictionError,
    build_prediction_failure_records,
    build_prediction_result_records,
    evaluate_result_completion,
    merge_validation_issue_records,
    parse_skill_prediction,
    parse_successful_prediction_records,
    provisional_skill_prediction,
    prediction_validation_issue_events,
    recover_skill_prediction,
    run_skill_predictions,
)
from common.io_utils import sha256_json  # noqa: E402


class FakePredictionClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        return {"content": self.content, "usage": {}}


class SkillPredictionContractTests(unittest.TestCase):
    def test_three_percent_completion_uses_unique_targets_and_blocks_runtime_failures(self) -> None:
        def result(idx: int, outcome: str = "exact") -> dict:
            return {"idx": idx, "outcome": outcome}

        for total, accepted_count, rejected_count in ((100, 3, 4), (326, 9, 10)):
            accepted = evaluate_result_completion(
                expected_count=total,
                results=[result(idx) for idx in range(total)],
                validation_issues=[{"idx": idx} for idx in range(accepted_count)],
                maximum_failure_rate=0.03,
            )
            self.assertTrue(accepted["within_tolerance"])
            self.assertEqual(accepted["maximum_failure_count"], accepted_count)
            rejected = evaluate_result_completion(
                expected_count=total,
                results=[result(idx) for idx in range(total)],
                validation_issues=[{"idx": idx} for idx in range(rejected_count)],
                maximum_failure_rate=0.03,
            )
            self.assertFalse(rejected["within_tolerance"])
        blocked = evaluate_result_completion(
            expected_count=100,
            results=[
                result(idx, "runtime_failed" if idx == 1 else "exact")
                for idx in range(100)
            ],
            validation_issues=[],
            maximum_failure_rate=0.03,
        )
        self.assertFalse(blocked["within_tolerance"])

    def test_exact_positive_multiple_and_empty_predictions(self) -> None:
        sentence = "Use Python and 项目管理 ."
        positive = json.dumps(
            {
                "annotated_sentence": (
                    "Use <skill>Python</skill> and <skill>项目管理</skill> ."
                )
            },
            ensure_ascii=False,
        )
        self.assertEqual(
            parse_skill_prediction(positive, sentence),
            {
                "has_skill": 1,
                "spans": [
                    {"text": "Python", "start": 4, "end": 10},
                    {"text": "项目管理", "start": 15, "end": 19},
                ],
            },
        )
        self.assertEqual(
            parse_skill_prediction(
                json.dumps({"annotated_sentence": sentence}, ensure_ascii=False), sentence
            ),
            {"has_skill": 0, "spans": []},
        )
        spaced = "Use\tC++\u00a0carefully ."
        spaced_content = json.dumps(
            {"annotated_sentence": "Use\t<skill>C++</skill>\u00a0carefully ."},
            ensure_ascii=False,
        )
        self.assertEqual(
            parse_skill_prediction(spaced_content, spaced)["spans"],
            [{"text": "C++", "start": 4, "end": 7}],
        )

    def test_strict_contract_rejects_recovery_and_invalid_boundaries(self) -> None:
        sentence = "Use Python ."
        invalid = [
            "```json\n{\"annotated_sentence\":\"Use Python .\"}\n```",
            json.dumps({"annotated_sentence": sentence, "extra": 1}),
            json.dumps({"annotated_sentence": "Use  Python ."}),
            json.dumps({"annotated_sentence": "Use <skill></skill>Python ."}),
            json.dumps(
                {
                    "annotated_sentence": (
                        "Use <skill>Py<skill>th</skill>on</skill> ."
                    )
                }
            ),
            json.dumps({"annotated_sentence": "Use <skill>Python ."}),
        ]
        for content in invalid:
            with self.subTest(content=content):
                with self.assertRaises(SkillPredictionError):
                    parse_skill_prediction(content, sentence)

    def test_whitelisted_character_recovery_projects_exact_source_spans(self) -> None:
        sentence = "research\u0097teaching"
        content = json.dumps(
            {
                "annotated_sentence": (
                    "<skill>research</skill>\u2014<skill>teaching</skill>"
                )
            },
            ensure_ascii=False,
        )
        strict_error = {
            "type": "SkillPredictionError",
            "message": "Removing skill tags must reconstruct the target sentence exactly",
        }
        with self.assertRaises(SkillPredictionError):
            parse_skill_prediction(content, sentence)
        recovery = recover_skill_prediction(content, sentence, strict_error)
        self.assertEqual(
            recovery["annotation"]["spans"],
            [
                {"text": "research", "start": 0, "end": 8},
                {"text": "teaching", "start": 9, "end": 17},
            ],
        )
        self.assertEqual(
            recovery["character_substitutions"],
            [
                {
                    "position": 8,
                    "source_character": "\u0097",
                    "model_character": "\u2014",
                    "source_codepoint": "U+0097",
                    "model_codepoint": "U+2014",
                }
            ],
        )

    def test_unsafe_rewrite_is_only_provisional(self) -> None:
        content = json.dumps(
            {"annotated_sentence": "Use <skill>Pyth0n</skill> ."}
        )
        provisional = provisional_skill_prediction(content)
        self.assertEqual(provisional["model_sentence"], "Use Pyth0n .")
        self.assertEqual(
            provisional["spans"],
            [{"text": "Pyth0n", "start": 4, "end": 10}],
        )
        with self.assertRaisesRegex(SkillPredictionError, "non-whitelisted"):
            recover_skill_prediction(
                content,
                "Use Python .",
                {"type": "SkillPredictionError", "message": "strict failure"},
            )

    def test_runner_publishes_safe_recovery_and_retains_unsafe_extraction(self) -> None:
        def prompt(sentence: str, idx: int) -> dict:
            messages = [{"role": "user", "content": sentence}]
            return {
                "schema_version": "skill-prediction-prompt-v1",
                "branch": "test",
                "dataset_id": "dataset",
                "record_id": str(idx),
                "source_sha256": "a" * 64,
                "idx": idx,
                "sentence": sentence,
                "messages": messages,
                "review_reasons": [],
                "evidence": {},
                "prompt_sha256": sha256_json(messages),
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            safe_prompt = prompt("research\u0097teaching", 1)
            safe_content = json.dumps(
                {
                    "annotated_sentence": (
                        "<skill>research</skill>\u2014<skill>teaching</skill>"
                    )
                },
                ensure_ascii=False,
            )
            safe_client = FakePredictionClient(safe_content)
            safe_raw = run_skill_predictions(
                [safe_prompt],
                root / "safe.jsonl",
                chat_model="test-model",
                temperature=0.0,
                max_tokens=512,
                maximum_repairs=2,
                allow_network=True,
                retry_failed=False,
                client_factory=lambda: safe_client,
                secrets=[],
            )
            self.assertEqual(safe_client.calls, 3)
            self.assertEqual(safe_raw[1]["status"], "complete")
            self.assertIn("recovery", safe_raw[1])
            predictions = parse_successful_prediction_records(
                [safe_prompt],
                safe_raw,
                chat_model="test-model",
                maximum_repairs=2,
            )
            self.assertEqual(predictions[0]["status"], "needs_review")
            self.assertEqual(predictions[0]["review_reasons"], [RECOVERY_REVIEW_REASON])
            self.assertEqual(
                build_prediction_failure_records(
                    [safe_prompt], safe_raw, maximum_repairs=2
                ),
                [],
            )
            safe_results = build_prediction_result_records(
                [safe_prompt], predictions, [], safe_raw
            )
            self.assertEqual(safe_results[0]["outcome"], "recovered")
            self.assertTrue(safe_results[0]["validation_issue"])
            self.assertEqual(safe_results[0]["coordinate_space"], "target_sentence")
            tampered = copy.deepcopy(safe_raw)
            tampered[1]["recovery"]["annotation"]["spans"][0]["end"] = 7
            with self.assertRaisesRegex(SkillPredictionError, "recovery audit mismatch"):
                parse_successful_prediction_records(
                    [safe_prompt],
                    tampered,
                    chat_model="test-model",
                    maximum_repairs=2,
                )

            unsafe_prompt = prompt("Use Python .", 2)
            unsafe_content = json.dumps(
                {"annotated_sentence": "Use <skill>Pyth0n</skill> ."}
            )
            unsafe_client = FakePredictionClient(unsafe_content)
            unsafe_raw = run_skill_predictions(
                [unsafe_prompt],
                root / "unsafe.jsonl",
                chat_model="test-model",
                temperature=0.0,
                max_tokens=512,
                maximum_repairs=2,
                allow_network=True,
                retry_failed=False,
                client_factory=lambda: unsafe_client,
                secrets=[],
            )
            self.assertEqual(unsafe_raw[2]["status"], "failed")
            failures = build_prediction_failure_records(
                [unsafe_prompt], unsafe_raw, maximum_repairs=2
            )
            self.assertEqual(failures[0]["failure_kind"], "validation_exhausted")
            self.assertEqual(
                failures[0]["provisional_extraction"]["spans"],
                [{"text": "Pyth0n", "start": 4, "end": 10}],
            )
            results = build_prediction_result_records(
                [unsafe_prompt], [], failures, unsafe_raw
            )
            self.assertEqual(results[0]["outcome"], "provisional")
            self.assertEqual(results[0]["coordinate_space"], "model_sentence")
            events = prediction_validation_issue_events(
                results, stage="test_prediction"
            )
            issues = merge_validation_issue_records(
                [
                    {
                        "dataset_id": "dataset",
                        "record_id": "2",
                        "source_sha256": "a" * 64,
                        "idx": 2,
                        "sentence": "Use Python .",
                    }
                ],
                [events[0], {**events[0], "stage": "earlier_stage"}],
            )
            self.assertEqual(len(issues), 1)
            self.assertEqual(issues[0]["event_count"], 2)
            self.assertEqual(
                issues[0]["stages"], ["test_prediction", "earlier_stage"]
            )


if __name__ == "__main__":
    unittest.main()
