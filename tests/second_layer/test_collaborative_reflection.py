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

from second_layer.collaborative_reflection.online import reflect_one  # noqa: E402
from second_layer.collaborative_reflection.pipeline import (  # noqa: E402
    assemble_context_records,
    build_exemplar_records,
    build_interaction,
    build_prompt,
    build_repair_messages,
    build_reflection_inputs,
    build_trf_record,
    parse_exemplar_response,
    parse_trf_response,
)
from second_layer.collaborative_reflection.branch_runner import _latest  # noqa: E402
from second_layer.collaborative_reflection.runner import (  # noqa: E402
    _branch_command,
    execute_parallel_processes,
)


def _candidate(demo_idx: int) -> dict[str, Any]:
    accepted = demo_idx <= 8
    return {
        "demo_dataset_id": "demos",
        "demo_record_id": str(demo_idx),
        "demo_source_sha256": "b" * 64,
        "demo_idx": demo_idx,
        "retrieval_rank": demo_idx,
        "sentence": f"Demo {demo_idx} uses Python." if accepted else f"Demo {demo_idx} has no skill.",
        "status": "accepted" if accepted else "negative",
        "skill_spans": ["Python"] if accepted else [],
        "accepted_span_details": [],
        "pseudo_trfs": ["Python"] if accepted else [],
        "similarity": round(1.0 - demo_idx / 100, 2),
        "existence_score": 1.0,
    }


def _judgment(demo_idx: int, score: int | None = None) -> dict[str, Any]:
    accepted = demo_idx <= 8
    return {
        "demo_idx": demo_idx,
        "helpfulness_score": score if score is not None else (5 if demo_idx <= 2 else 2),
        "role": "supporting" if accepted else "contrastive",
        "reason_codes": ["TYPE_ALIGNED" if accepted else "NEGATIVE_CONTRAST"],
        "reason": "Useful frozen evidence for testing.",
    }


def _fixtures() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    identity = {
        "dataset_id": "targets",
        "record_id": "1",
        "source_sha256": "a" * 64,
        "idx": 1,
        "sentence": "Use Python for analysis.",
    }
    candidates = [_candidate(index) for index in range(1, 17)]
    judgments = [_judgment(index) for index in range(1, 17)]
    selected = [
        {**candidates[index - 1], **judgments[index - 1], "gate_rank": index}
        for index in (1, 2)
    ]
    trf = {
        "raw_text": "Python",
        "text": "Python",
        "normalized_text": "Python",
        "first_seen_order": 1,
        "in_main_bank_exact": True,
        "in_main_bank_casefold": True,
        "appears_in_target_exact": True,
        "appears_in_target_casefold": True,
    }
    base = {
        "schema_version": "second-layer-context-v1",
        **identity,
        "assembly_status": "ready",
        "trf_context": {
            "status": "complete",
            "entity_types": ["Skill"],
            "trfs": [trf],
            "retrieval_count": 16,
            "review_reasons": [],
            "models": {"embedding": "fixture", "chat": "fixture"},
        },
        "exemplar_context": {
            "status": "complete",
            "feature_context": "absent",
            "candidate_count": 16,
            "eligible_count": 2,
            "selected_count": 2,
            "selected": selected,
            "rejected_demo_ids": list(range(3, 17)),
            "review_reasons": [],
            "feature_review_reasons": [],
            "models": {"chat": "fixture"},
        },
        "provenance": {},
    }
    candidate_record = {
        "schema_version": "candidate-instances-v1",
        **identity,
        "target_evidence": {"entity_types": [], "trfs": [], "feature_context": "absent"},
        "feature_status": {"status": "complete", "review_reasons": []},
        "candidate_count": 16,
        "candidates": candidates,
    }
    judgment_record = {
        "schema_version": "instance-judgments-v1",
        **identity,
        "status": "complete",
        "candidate_count": 16,
        "judgments": judgments,
        "models": {"chat": "fixture"},
    }
    return base, candidate_record, judgment_record


def _input() -> dict[str, Any]:
    base, candidates, judgments = _fixtures()
    return build_reflection_inputs([base], [candidates], [judgments])[0]


def _trf_payload(
    action: str = "keep",
    addition: str | None = None,
    semantic_class: str = "domain_feature",
) -> dict[str, Any]:
    additions = []
    if addition is not None:
        additions.append(
            {
                "text": addition,
                "semantic_class": semantic_class,
                "confidence": 4,
                "evidence_sources": ["TARGET_TEXT"],
                "evidence_demo_ids": [],
                "reason_codes": ["OPEN_VOCABULARY_INFERENCE"],
                "reason": "The target provides evidence for this relation.",
            }
        )
    return {
        "revised_entity_types": ["Skill"],
        "original_trf_decisions": [
            {
                "normalized_text": "Python",
                "action": action,
                "confidence": 5,
                "evidence_sources": ["TARGET_TEXT", "EXEMPLARS"],
                "evidence_demo_ids": [1],
                "reason_codes": ["TARGET_EXPLICIT_SUPPORT", "EXEMPLAR_SUPPORT"],
                "reason": "Both the target and candidate one support this relation.",
            }
        ],
        "added_trfs": additions,
        "overall_reason": "All frozen evidence was considered once.",
        "unresolved_issues": [],
    }


def _exemplar_payload(*, promote: bool = False, support_python: bool = False) -> dict[str, Any]:
    judgments = []
    for demo_idx in range(1, 17):
        accepted = demo_idx <= 8
        score = 5 if demo_idx <= 2 or (promote and demo_idx == 3) else 2
        relations = (
            [{"normalized_text": "Python", "relation": "supports"}]
            if support_python and demo_idx == 1
            else []
        )
        judgments.append(
            {
                "demo_idx": demo_idx,
                "helpfulness_score": score,
                "role": "supporting" if accepted else "contrastive",
                "trf_relations": relations,
                "reason_codes": ["TYPE_ALIGNED" if accepted else "NEGATIVE_CONTRAST"],
                "reason": "This candidate was reconsidered using frozen baseline evidence.",
            }
        )
    return {
        "revised_judgments": judgments,
        "overall_reason": "All sixteen candidates were reconsidered.",
        "unresolved_issues": [],
    }


class InputAndPromptTests(unittest.TestCase):
    def test_minimal_snapshots_reconstruct_exact_input(self) -> None:
        record = _input()
        self.assertEqual(record["schema_version"], "collaborative-reflection-input-v1")
        self.assertEqual(len(record["candidates"]), 16)
        self.assertEqual(len(record["baseline_judgments"]), 16)
        self.assertEqual(record["baseline_selected_demo_ids"], [1, 2])
        self.assertEqual(record["baseline_context"]["exemplar_context"]["feature_context"], "absent")

    def test_accepts_v2_baseline_judgments_without_free_text_reason(self) -> None:
        base, candidates, judgments = _fixtures()
        judgments["schema_version"] = "instance-judgments-v2"
        for judgment in judgments["judgments"]:
            judgment.pop("reason")

        record = build_reflection_inputs([base], [candidates], [judgments])[0]

        self.assertTrue(all("reason" not in item for item in record["baseline_judgments"]))
        build_prompt(record, "trf", 60000)
        build_prompt(record, "exemplar", 60000)

    def test_join_fails_closed_on_identity_sentence_hash_and_demo_ids(self) -> None:
        base, candidates, judgments = _fixtures()
        variants = []
        for key, value in (("record_id", "2"), ("sentence", "other"), ("source_sha256", "c" * 64)):
            bad = copy.deepcopy(candidates)
            bad[key] = value
            variants.append((bad, judgments))
        bad_judgments = copy.deepcopy(judgments)
        bad_judgments["judgments"][0]["demo_idx"] = 999
        variants.append((candidates, bad_judgments))
        for candidate_variant, judgment_variant in variants:
            with self.subTest(candidate=candidate_variant.get("record_id")):
                with self.assertRaises(Exception):
                    build_reflection_inputs([base], [candidate_variant], [judgment_variant])

    def test_both_prompts_use_only_the_same_r0_record(self) -> None:
        record = _input()
        trf_prompt = build_prompt(record, "trf", 60000)
        exemplar_prompt = build_prompt(record, "exemplar", 60000)
        for prompt in (trf_prompt, exemplar_prompt):
            text = json.dumps(prompt, ensure_ascii=False)
            self.assertIn("Use Python for analysis.", text)
            self.assertIn("baseline_judgment", text)
            self.assertNotIn("trf-reflection-v1", text)
            self.assertNotIn("exemplar-reflection-v1", text)

    def test_empty_baseline_forces_empty_exemplar_relations(self) -> None:
        base, candidates, judgments = _fixtures()
        base["trf_context"]["entity_types"] = []
        base["trf_context"]["trfs"] = []
        record = build_reflection_inputs([base], [candidates], [judgments])[0]
        prompt = build_prompt(record, "exemplar", 60000, 240)
        body = json.loads(prompt["messages"][1]["content"])
        self.assertEqual(
            body["constraints"]["trf_relations_contract"],
            "every_list_must_be_empty",
        )

    def test_repair_contract_repeats_reason_limit_and_empty_relation_rule(self) -> None:
        base, candidates, judgments = _fixtures()
        base["trf_context"]["entity_types"] = []
        base["trf_context"]["trfs"] = []
        record = build_reflection_inputs([base], [candidates], [judgments])[0]
        prompt = build_prompt(record, "exemplar", 60000, 240)
        messages = build_repair_messages(prompt, "exemplar", "too long", "{}", 240)
        repair = json.loads(messages[-1]["content"])["repair_request"]["contract"]
        self.assertEqual(repair["all_reason_and_overall_reason_max_characters"], 240)
        self.assertEqual(repair["repair_reason_target_characters"], 160)
        self.assertEqual(
            repair["trf_relations_when_allowed_baseline_trfs_empty"],
            "every_list_must_be_empty",
        )

    def test_trf_prompt_contains_human_semantic_boundary(self) -> None:
        prompt = build_prompt(_input(), "trf", 60000, 240)
        contract = prompt["trf_semantic_contract"]
        labels = {item["text"]: item for item in contract["human_labeled_examples"]}
        self.assertFalse(labels["patient-centred"]["is_trf"])
        self.assertFalse(labels["inclusive"]["is_trf"])
        self.assertFalse(labels["safe and effective"]["is_trf"])
        self.assertFalse(labels["course of treatment"]["is_trf"])
        self.assertTrue(labels["IT systems"]["is_trf"])
        self.assertEqual(labels["IT systems"]["semantic_class"], "domain_feature")


class ParsingAndAssemblyTests(unittest.TestCase):
    gate = {
        "minimum_helpfulness": 3,
        "max_selected": 8,
        "max_supporting": 6,
        "max_contrastive": 3,
        "minimum_for_complete": 2,
    }

    def test_trf_keep_drop_add_normalize_and_deduplicate(self) -> None:
        record = _input()
        parsed = parse_trf_response(json.dumps(_trf_payload(addition="  Data Analysis  ")), record)
        reflected = build_trf_record(record, parsed, "fixture")
        self.assertEqual([item["normalized_text"] for item in reflected["revised_trfs"]], ["Python", "Data Analysis"])
        duplicate = _trf_payload(addition=" Python ")
        with self.assertRaises(Exception):
            parse_trf_response(json.dumps(duplicate), record)

    def test_empty_baseline_can_be_supplemented(self) -> None:
        base, candidates, judgments = _fixtures()
        base["trf_context"]["entity_types"] = []
        base["trf_context"]["trfs"] = []
        record = build_reflection_inputs([base], [candidates], [judgments])[0]
        payload = _trf_payload(addition="Data Analysis")
        payload["original_trf_decisions"] = []
        reflected = build_trf_record(record, parse_trf_response(json.dumps(payload), record), "fixture")
        self.assertEqual(reflected["revised_trfs"][0]["origin"], "added")

    def test_unseen_added_trf_semantics_require_review(self) -> None:
        record = _input()
        parsed = parse_trf_response(
            json.dumps(_trf_payload(addition="analysis")), record
        )
        reflected = build_trf_record(record, parsed, "fixture")
        self.assertEqual(reflected["status"], "needs_review")
        self.assertIn(
            "added_trf_semantics_unverified", reflected["review_reasons"]
        )

    def test_human_semantic_labels_fail_closed_and_accept_it_systems(self) -> None:
        record = _input()
        for text, semantic_class in (
            ("patient-centred", "domain_feature"),
            ("inclusive", "skill_type_cue"),
            ("safe and effective", "domain_feature"),
            ("course of treatment", "domain_feature"),
        ):
            with self.subTest(text=text):
                with self.assertRaises(Exception):
                    parse_trf_response(
                        json.dumps(_trf_payload(addition=text, semantic_class=semantic_class)),
                        record,
                    )

        parsed = parse_trf_response(
            json.dumps(_trf_payload(addition="IT systems", semantic_class="domain_feature")),
            record,
        )
        reflected = build_trf_record(record, parsed, "fixture")
        addition = reflected["revised_trfs"][-1]
        self.assertEqual(addition["semantic_class"], "domain_feature")
        self.assertEqual(addition["semantic_validation"], "human_validated")
        self.assertNotIn("added_trf_semantics_unverified", reflected["review_reasons"])

        with self.assertRaises(Exception):
            parse_trf_response(
                json.dumps(_trf_payload(addition="IT systems", semantic_class="skill_type_cue")),
                record,
            )

    def test_human_labeled_non_trf_cannot_be_kept_from_baseline(self) -> None:
        base, candidates, judgments = _fixtures()
        base_trf = base["trf_context"]["trfs"][0]
        for key in ("raw_text", "text", "normalized_text"):
            base_trf[key] = "patient-centred"
        record = build_reflection_inputs([base], [candidates], [judgments])[0]
        payload = _trf_payload(action="keep")
        payload["original_trf_decisions"][0]["normalized_text"] = "patient-centred"
        with self.assertRaises(Exception):
            parse_trf_response(json.dumps(payload), record)
        payload["original_trf_decisions"][0]["action"] = "drop"
        parsed = parse_trf_response(json.dumps(payload), record)
        self.assertEqual(parsed["original_trf_decisions"][0]["action"], "drop")

    def test_exemplar_requires_exactly_sixteen_and_can_promote(self) -> None:
        record = _input()
        invalid = _exemplar_payload()
        invalid["revised_judgments"].pop()
        with self.assertRaises(Exception):
            parse_exemplar_response(json.dumps(invalid), record)
        parsed = parse_exemplar_response(json.dumps(_exemplar_payload(promote=True)), record)
        _, selected = build_exemplar_records(record, parsed, self.gate, "fixture")
        self.assertIn(3, selected["promoted_demo_ids"])
        self.assertEqual(selected["selected_count"], 3)

    def test_interaction_conflicts_are_deterministic(self) -> None:
        record = _input()
        trf_parsed = parse_trf_response(json.dumps(_trf_payload(action="drop")), record)
        trf = build_trf_record(record, trf_parsed, "fixture")
        exemplar_parsed = parse_exemplar_response(
            json.dumps(_exemplar_payload(support_python=True)), record
        )
        parsed_record, selected = build_exemplar_records(record, exemplar_parsed, self.gate, "fixture")
        interaction = build_interaction(record, trf, {**parsed_record, **selected})
        self.assertEqual(interaction["status"], "conflicted")
        self.assertIn("selected_demo_supports_dropped_trf", {item["type"] for item in interaction["conflicts"]})

    def test_final_context_preserves_baseline_and_has_no_final_spans(self) -> None:
        record = _input()
        trf = build_trf_record(record, parse_trf_response(json.dumps(_trf_payload()), record), "fixture")
        parsed, selected = build_exemplar_records(
            record,
            parse_exemplar_response(json.dumps(_exemplar_payload()), record),
            self.gate,
            "fixture",
        )
        contexts, interactions = assemble_context_records([record], [trf], [parsed], [selected], {"run": "fixture"})
        self.assertEqual(contexts[0]["schema_version"], "collaborative-reflection-context-v1")
        self.assertEqual(contexts[0]["baseline_context"], record["baseline_context"])
        self.assertEqual(contexts[0]["interaction"], interactions[0])
        self.assertNotIn("final_skill_spans", contexts[0])


class RepairAndInterfaceTests(unittest.TestCase):
    def test_reflection_processes_overlap_and_failure_does_not_cancel_peer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_started = root / "first.started"
            second_started = root / "second.started"
            first_finished = root / "first.finished"

            def command(own: Path, peer: Path, *, fail: bool) -> list[str]:
                source = (
                    "from pathlib import Path\n"
                    "import sys, time\n"
                    f"own=Path({str(own)!r}); peer=Path({str(peer)!r})\n"
                    "own.write_text('started', encoding='utf-8')\n"
                    "deadline=time.time()+3\n"
                    "while not peer.exists() and time.time()<deadline: time.sleep(0.01)\n"
                    + (
                        "sys.exit(7 if peer.exists() else 9)\n"
                        if fail
                        else f"Path({str(first_finished)!r}).write_text('finished', encoding='utf-8')\n"
                    )
                )
                return [sys.executable, "-c", source]

            states = execute_parallel_processes(
                {
                    "trf": (command(first_started, second_started, fail=False), root / "trf.log"),
                    "exemplar": (command(second_started, first_started, fail=True), root / "exemplar.log"),
                },
                cwd=root,
            )
            self.assertEqual(states["trf"]["exit_code"], 0)
            self.assertEqual(states["exemplar"]["exit_code"], 7)
            self.assertTrue(first_finished.is_file())
            self.assertLessEqual(states["trf"]["started_at"], states["exemplar"]["ended_at"])
            self.assertLessEqual(states["exemplar"]["started_at"], states["trf"]["ended_at"])

    def test_append_only_raw_allows_failed_retry_but_not_completed_retry(self) -> None:
        base = {
            "schema_version": "collaborative-reflection-raw-v1",
            "branch": "trf",
            "dataset_id": "d",
            "record_id": "1",
            "source_sha256": "a" * 64,
            "idx": 1,
            "sentence": "s",
            "status": "failed",
        }
        completed = {**base, "status": "complete"}
        self.assertEqual(_latest([base, completed], "trf")[1]["status"], "complete")
        with self.assertRaises(Exception):
            _latest([completed, completed], "trf")

    def test_invalid_initial_response_is_repaired_with_a_strict_audit_chain(self) -> None:
        record = _input()
        valid = json.dumps(_trf_payload())

        class FakeClient:
            def __init__(self) -> None:
                self.responses = ["{}", valid]

            def chat(self, messages: list[dict[str, str]], **_: Any) -> dict[str, Any]:
                content = self.responses.pop(0)
                return {
                    "content": content,
                    "finish_reason": "stop",
                    "latency_ms": 1,
                    "attempts": 1,
                    "response_id": "fixture",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }

        raw, parsed = reflect_one(
            branch="trf",
            record=record,
            prompt=build_prompt(record, "trf", 60000),
            client=FakeClient(),
            chat={"temperature": 0.0, "max_tokens": 4096, "model": "fixture", "max_reason_characters": 240},
            max_repairs=2,
        )
        self.assertIsNotNone(parsed)
        self.assertEqual(raw["status"], "complete")
        self.assertEqual([item["kind"] for item in raw["attempts"]], ["initial", "repair"])
        self.assertIsNotNone(raw["attempts"][0]["parse_error"])

    def test_branch_command_propagates_resume_retry_and_ids(self) -> None:
        command = _branch_command(
            python="python",
            config_path=Path("reflection.json"),
            paths=type("Paths", (), {"root": Path("parent")})(),
            run_id="run",
            branch="exemplar",
            allow_network=True,
            resume=True,
            retry_failed=True,
        )
        self.assertIn("run-exemplar-reflection", command)
        self.assertIn("--allow-network", command)
        self.assertIn("--resume", command)
        self.assertIn("--retry-failed", command)

    def test_concurrent_alias_exposes_the_legacy_parameter_surface(self) -> None:
        text = (PROJECT_ROOT / "scripts" / "second_layer" / "run_concurrent.ps1").read_text(encoding="utf-8")
        for parameter in (
            "RunId", "TargetMode", "Input", "Limit", "PrepareOnly", "AllowModelDownload",
            "AllowNetwork", "ConfirmFullRun", "Resume", "RetryFailed", "ReuseEmbeddingsFrom",
            "ConfigPath", "PythonExecutable",
        ):
            self.assertIn(f"${parameter}", text)


if __name__ == "__main__":
    unittest.main()
