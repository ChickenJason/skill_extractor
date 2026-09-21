from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from instance_discriminator.pipeline import apply_hard_gate
from common.qwen_client import QwenRequestError, QwenSettings
from second_layer.bidirectional_interaction.common import InteractionError
from second_layer.bidirectional_interaction.branch_runner import _metrics
from second_layer.bidirectional_interaction.branch_validator import _cumulative_metrics
from second_layer.bidirectional_interaction.contracts import (
    parse_exemplar_proposal,
    parse_trf_proposal,
    rebuild_inputs,
)
from second_layer.bidirectional_interaction.online import reflect_one
from second_layer.bidirectional_interaction.pipeline import (
    active_branches,
    apply_round,
    effective_signature,
)
from second_layer.bidirectional_interaction.prompts import build_prompt, build_repair_messages
from second_layer.bidirectional_interaction.runner import _overlap
from second_layer.bidirectional_interaction.schema_client import JsonSchemaQwenClient, validate_response_format
from second_layer.bidirectional_interaction.schemas import response_format


SHA = "0" * 64
GATE = {
    "minimum_helpfulness": 3,
    "max_selected": 8,
    "max_supporting": 6,
    "max_contrastive": 3,
    "minimum_for_complete": 2,
}


def fixture() -> tuple[dict, dict, dict]:
    candidates = []
    judgments = []
    for position in range(1, 17):
        accepted = position <= 8
        candidates.append(
            {
                "demo_dataset_id": "d",
                "demo_record_id": str(position + 100),
                "demo_source_sha256": SHA,
                "demo_idx": position + 100,
                "retrieval_rank": position,
                "sentence": f"demo {position}",
                "status": "accepted" if accepted else "negative",
                "skill_spans": ["skill"] if accepted else [],
                "accepted_span_details": [],
                "pseudo_trfs": ["tools"] if accepted else [],
                "similarity": 1.0 - position / 100,
                "existence_score": 1.0,
            }
        )
        judgments.append(
            {
                "demo_idx": position + 100,
                "helpfulness_score": 5 if position <= 2 else 2,
                "role": "supporting" if accepted and position <= 2 else ("irrelevant" if accepted else "contrastive"),
                "reason_codes": ["TYPE_ALIGNED"] if accepted else ["NEGATIVE_CONTRAST"],
                "reason": "baseline reason",
            }
        )
    selection = apply_hard_gate(candidates, judgments, GATE)
    context = {
        "schema_version": "second-layer-context-v1",
        "dataset_id": "d",
        "record_id": "1",
        "source_sha256": SHA,
        "idx": 1,
        "sentence": "Use Excel tools",
        "assembly_status": "ready",
        "trf_context": {
            "status": "complete",
            "entity_types": ["Skill"],
            "trfs": [
                {
                    "raw_text": "tools",
                    "text": "tools",
                    "normalized_text": "tools",
                    "first_seen_order": 1,
                    "in_main_bank_exact": True,
                    "in_main_bank_casefold": True,
                    "appears_in_target_exact": True,
                    "appears_in_target_casefold": True,
                }
            ],
            "retrieval_count": 16,
            "review_reasons": [],
            "models": {},
        },
        "exemplar_context": {
            "status": "complete",
            "feature_context": "absent",
            "candidate_count": 16,
            "eligible_count": selection["eligible_count"],
            "selected_count": len(selection["selected"]),
            "selected": selection["selected"],
            "rejected_demo_ids": selection["rejected_demo_ids"],
            "review_reasons": [],
            "feature_review_reasons": [],
            "models": {},
        },
        "provenance": {},
    }
    candidate = {
        "schema_version": "candidate-instances-v1",
        "dataset_id": "d",
        "record_id": "1",
        "source_sha256": SHA,
        "idx": 1,
        "sentence": "Use Excel tools",
        "candidate_count": 16,
        "candidates": candidates,
    }
    judgment = {
        "schema_version": "instance-judgments-v1",
        "dataset_id": "d",
        "record_id": "1",
        "source_sha256": SHA,
        "idx": 1,
        "sentence": "Use Excel tools",
        "candidate_count": 16,
        "judgments": judgments,
    }
    return context, candidate, judgment


def rebuilt() -> dict:
    context, candidate, judgment = fixture()
    return rebuild_inputs([context], [candidate], [judgment], GATE)[0]


def trf_proposal(record: dict, action: str = "keep", addition: str | None = None) -> dict:
    evidence = {
        "confidence": 4,
        "evidence_sources": ["TARGET_TEXT"],
        "evidence_demo_ids": [],
        "reason_codes": ["TARGET_EXPLICIT_SUPPORT"],
        "reason": "grounded",
    }
    return {
        "schema_version": "trf-interaction-proposal-v1",
        "revised_entity_types": ["Skill"],
        "known_trf_decisions": [
            {"normalized_text": item["normalized_text"], "action": action, **evidence}
            for item in record["r0"]["known_trfs"]
        ],
        "added_trfs": (
            [{"text": addition, "semantic_class": "domain_feature", **evidence}]
            if addition else []
        ),
        "overall_reason": "complete review",
        "unresolved_issues": [],
    }


def exemplar_proposal(record: dict, score_delta: int = 0, reason: str = "rejudged") -> dict:
    values = []
    for item in record["h0"]["judgments"]:
        value = copy.deepcopy(item)
        previous_score = value["helpfulness_score"]
        final_score = max(1, min(5, previous_score + score_delta))
        value["helpfulness_score"] = final_score
        value["trf_relations"] = []
        value["reason"] = reason
        values.append(value)
    return {"schema_version": "exemplar-interaction-proposal-v1", "revised_judgments": values, "overall_reason": "complete review", "unresolved_issues": []}


class FakeClient:
    def __init__(self, contents: list[str]):
        self.contents = iter(contents)
        self.calls = 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        return {
            "content": next(self.contents),
            "attempts": 1,
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }


class BidirectionalInteractionTests(unittest.TestCase):
    def test_rebuilds_full_r0_h0_and_hard_gate(self):
        record = rebuilt()
        self.assertEqual(len(record["candidates"]), 16)
        self.assertEqual(len(record["h0"]["judgments"]), 16)
        self.assertEqual(record["h0"]["selection"], apply_hard_gate(record["candidates"], record["baseline_judgments"], GATE))

    def test_accepts_v2_baseline_judgments_without_free_text_reason(self):
        context, candidate, judgment = fixture()
        judgment["schema_version"] = "instance-judgments-v2"
        for item in judgment["judgments"]:
            item.pop("reason")
        selection = apply_hard_gate(candidate["candidates"], judgment["judgments"], GATE)
        context["exemplar_context"].update(
            {
                "eligible_count": selection["eligible_count"],
                "selected_count": len(selection["selected"]),
                "selected": selection["selected"],
                "rejected_demo_ids": selection["rejected_demo_ids"],
            }
        )

        record = rebuild_inputs([context], [candidate], [judgment], GATE)[0]

        self.assertTrue(all("reason" not in item for item in record["h0"]["judgments"]))
        build_prompt(
            record,
            "exemplar",
            record["r0"],
            record["h0"],
            1,
            max_characters=60000,
            max_reason_characters=240,
        )

    def test_identity_and_source_hash_fail_closed(self):
        context, candidate, judgment = fixture()
        candidate["source_sha256"] = "1" * 64
        with self.assertRaises(InteractionError):
            rebuild_inputs([context], [candidate], [judgment], GATE)

    def test_modes_and_single_direction_carry_forward(self):
        record = rebuilt()
        self.assertEqual(active_branches("bidirectional"), ("trf", "exemplar"))
        state = apply_round(record, record["r0"], record["h0"], {"trf": trf_proposal(record, "drop")}, "exemplar_to_trf", 1, GATE)
        self.assertEqual(state["exemplar_state"], record["h0"])
        state = apply_round(record, record["r0"], record["h0"], {"exemplar": exemplar_proposal(record)}, "trf_to_exemplar", 1, GATE)
        self.assertEqual(state["trf_state"], record["r0"])

    def test_trf_drop_rekeep_addition_and_duplicate_gate(self):
        record = rebuilt()
        first = apply_round(record, record["r0"], record["h0"], {"trf": trf_proposal(record, "drop", "Excel")}, "exemplar_to_trf", 1, GATE)
        self.assertFalse(first["trf_state"]["known_trfs"][0]["active"])
        self.assertEqual(first["trf_state"]["known_trfs"][1]["origin"], "added")
        record2 = copy.deepcopy(record)
        record2["r0"] = first["trf_state"]
        proposal = trf_proposal(record2, "keep")
        second = apply_round(record, first["trf_state"], record["h0"], {"trf": proposal}, "exemplar_to_trf", 2, GATE)
        self.assertTrue(all(item["active"] for item in second["trf_state"]["known_trfs"]))
        duplicate = trf_proposal(record, addition=" Tools ")
        with self.assertRaises(InteractionError):
            parse_trf_proposal(json.dumps(duplicate), record, record["r0"], max_reason=240)

    def test_complete_exemplar_rejudgment_and_reason_only_early_stop(self):
        record = rebuilt()
        for candidate in record["candidates"]:
            candidate["similarity"] = 0.5
        proposal = exemplar_proposal(record, reason="new audit reason")
        parsed = parse_exemplar_proposal(json.dumps(proposal), record, record["r0"], record["h0"], max_reason=240)
        state = apply_round(record, record["r0"], record["h0"], {"exemplar": parsed}, "trf_to_exemplar", 1, GATE)
        self.assertFalse(state["effective_changed"])
        self.assertEqual(effective_signature(state["trf_state"], state["exemplar_state"]), effective_signature(record["r0"], record["h0"]))

    def test_dynamic_closed_schema_has_no_unique_items(self):
        record = rebuilt()
        fmt = response_format(record, "exemplar", record["r0"], max_reason_characters=240)
        validate_response_format(fmt)
        encoded = json.dumps(fmt)
        self.assertNotIn("uniqueItems", encoded)
        self.assertIn('"additionalProperties": false', encoded)
        judgment_codes = fmt["json_schema"]["schema"]["properties"]["revised_judgments"]["items"]["properties"]["reason_codes"]
        self.assertEqual(judgment_codes["minItems"], 1)

        trf_format = response_format(record, "trf", record["r0"], max_reason_characters=240)
        properties = trf_format["json_schema"]["schema"]["properties"]
        decision_sources = properties["known_trf_decisions"]["items"]["properties"]["evidence_sources"]
        addition_sources = properties["added_trfs"]["items"]["properties"]["evidence_sources"]
        self.assertEqual(decision_sources["minItems"], 1)
        self.assertEqual(addition_sources["minItems"], 1)

    def test_prompt_reads_frozen_previous_state(self):
        record = rebuilt()
        prompt = build_prompt(record, "exemplar", record["r0"], record["h0"], 1, max_characters=60000, max_reason_characters=240)
        changed_h = copy.deepcopy(record["h0"])
        changed_h["judgments"][0]["reason"] = "changed previous reason"
        changed = build_prompt(record, "exemplar", record["r0"], changed_h, 1, max_characters=60000, max_reason_characters=240)
        self.assertNotEqual(prompt["prompt_sha256"], changed["prompt_sha256"])

    def test_strict_repair_is_bounded_to_two(self):
        record = rebuilt()
        prompt = build_prompt(record, "trf", record["r0"], record["h0"], 1, max_characters=60000, max_reason_characters=240)
        valid = json.dumps(trf_proposal(record))
        client = FakeClient(["{}", "{}", valid])
        raw, parsed = reflect_one(branch="trf", record=record, previous_r=record["r0"], previous_h=record["h0"], prompt=prompt, client=client, chat={"temperature": 0.0, "max_tokens": 100, "model": "fake", "max_reason_characters": 240}, max_repairs=2, secrets=[])
        self.assertEqual(raw["status"], "complete")
        self.assertEqual(len(raw["attempts"]), 3)
        self.assertIsNotNone(parsed)

        repair_text = build_repair_messages(prompt, "invalid", "{}")[ -1]["content"]
        self.assertIn("MUST contain at least one", repair_text)
        self.assertIn("no longer than 240 characters", repair_text)
        self.assertIn("no more than 120 characters", repair_text)

        prompt_body = json.loads(prompt["messages"][1]["content"])
        self.assertEqual(
            prompt_body["constraints"]["overall_reason_max_characters"], 120
        )

    def test_exemplar_parser_rejects_incomplete_batch(self):
        record = rebuilt()
        proposal = exemplar_proposal(record)
        proposal["revised_judgments"].pop()
        with self.assertRaises(InteractionError):
            parse_exemplar_proposal(json.dumps(proposal), record, record["r0"], record["h0"], max_reason=240)

    def test_exemplar_counterfactual_and_relation_semantics(self):
        record = rebuilt()
        record["candidates"][0]["skill_spans"] = ["Excel tools"]
        proposal = exemplar_proposal(record)
        first = proposal["revised_judgments"][0]
        first["trf_relations"] = [{"normalized_text": "tools", "relation": "supports"}]
        first["helpfulness_score"] = 3
        parsed = parse_exemplar_proposal(
            json.dumps(proposal), record, record["r0"], record["h0"], max_reason=240
        )
        first = parsed["revised_judgments"][0]
        self.assertEqual(first["previous_helpfulness_score"], 5)
        self.assertEqual(first["previous_role"], "supporting")
        self.assertEqual(first["helpfulness_score"], 4)
        self.assertEqual(first["score_adjustment"], -1)
        self.assertEqual(first["trf_effect"], "penalize")

    def test_sentence_only_trf_support_cannot_promote_accepted_demo(self):
        record = rebuilt()
        record["candidates"][2]["sentence"] = "generic tools context"
        record["candidates"][2]["similarity"] = 0.5
        proposal = exemplar_proposal(record)
        item = proposal["revised_judgments"][2]
        item["helpfulness_score"] = 5
        item["role"] = "supporting"
        item["trf_relations"] = [{"normalized_text": "tools", "relation": "supports"}]
        item["reason_codes"] = ["TRF_ALIGNED"]
        parsed = parse_exemplar_proposal(
            json.dumps(proposal), record, record["r0"], record["h0"], max_reason=240
        )
        item = parsed["revised_judgments"][2]
        self.assertEqual((item["helpfulness_score"], item["role"]), (3, "irrelevant"))
        self.assertEqual(item["trf_relations"], [])
        self.assertIn("unsupported_trf_promotion_rejected", item["calibration_actions"])

    def test_accepted_target_alignment_promotes_and_weak_h0_four_drops(self):
        record = rebuilt()
        proposal = exemplar_proposal(record)
        strong = proposal["revised_judgments"][2]
        strong["helpfulness_score"] = 2
        strong["role"] = "irrelevant"
        record["candidates"][2]["similarity"] = 0.6
        parsed = parse_exemplar_proposal(
            json.dumps(proposal), record, record["r0"], record["h0"], max_reason=240
        )
        strong = parsed["revised_judgments"][2]
        self.assertEqual((strong["helpfulness_score"], strong["role"]), (4, "supporting"))

        record = rebuilt()
        record["h0"]["judgments"][2]["helpfulness_score"] = 4
        record["h0"]["judgments"][2]["role"] = "supporting"
        record["candidates"][2]["similarity"] = 0.56
        proposal = exemplar_proposal(record)
        parsed = parse_exemplar_proposal(
            json.dumps(proposal), record, record["r0"], record["h0"], max_reason=240
        )
        weak = parsed["revised_judgments"][2]
        self.assertEqual((weak["helpfulness_score"], weak["role"]), (3, "irrelevant"))
        self.assertIn("unsubstantiated_hard_gate_score_rejected", weak["calibration_actions"])

    def test_high_negative_requires_boundary_or_relation(self):
        record = rebuilt()
        negative = record["h0"]["judgments"][8]
        negative["helpfulness_score"] = 4
        negative["role"] = "contrastive"
        record["candidates"][8]["similarity"] = 0.4
        proposal = exemplar_proposal(record)
        parsed = parse_exemplar_proposal(
            json.dumps(proposal), record, record["r0"], record["h0"], max_reason=240
        )
        calibrated = parsed["revised_judgments"][8]
        self.assertEqual((calibrated["helpfulness_score"], calibrated["role"]), (3, "irrelevant"))
        self.assertEqual(calibrated["score_adjustment"], -1)
        self.assertNotIn("NEGATIVE_CONTRAST", calibrated["reason_codes"])
        proposal["revised_judgments"][8]["reason_codes"].append("BOUNDARY_TRANSFERABLE")
        parsed = parse_exemplar_proposal(
            json.dumps(proposal), record, record["r0"], record["h0"], max_reason=240
        )
        self.assertEqual(parsed["revised_judgments"][8]["helpfulness_score"], 4)

    def test_strong_prior_negative_boundary_is_retained(self):
        record = rebuilt()
        negative = record["h0"]["judgments"][8]
        negative["helpfulness_score"] = 4
        negative["role"] = "contrastive"
        record["candidates"][8]["similarity"] = 0.7
        proposal = exemplar_proposal(record)
        proposed = proposal["revised_judgments"][8]
        proposed["helpfulness_score"] = 2
        proposed["role"] = "irrelevant"
        proposed["reason_codes"] = ["SEMANTIC_ONLY"]
        parsed = parse_exemplar_proposal(
            json.dumps(proposal), record, record["r0"], record["h0"], max_reason=240
        )
        retained = parsed["revised_judgments"][8]
        self.assertEqual((retained["helpfulness_score"], retained["role"]), (4, "contrastive"))
        self.assertIn("BOUNDARY_TRANSFERABLE", retained["reason_codes"])

    def test_negative_contrast_code_deterministically_sets_role(self):
        record = rebuilt()
        proposal = exemplar_proposal(record)
        negative = proposal["revised_judgments"][8]
        negative["role"] = "irrelevant"
        negative["reason_codes"] = ["NEGATIVE_CONTRAST"]
        parsed = parse_exemplar_proposal(
            json.dumps(proposal), record, record["r0"], record["h0"], max_reason=240
        )
        self.assertEqual(parsed["revised_judgments"][8]["role"], "contrastive")

    def test_trf_evidence_consistency_and_empty_type_gate(self):
        record = rebuilt()
        proposal = trf_proposal(record)
        proposal["known_trf_decisions"][0]["evidence_sources"] = ["EXEMPLARS"]
        with self.assertRaisesRegex(InteractionError, "evidence_demo_id"):
            parse_trf_proposal(json.dumps(proposal), record, record["r0"], max_reason=240)

        proposal = trf_proposal(record)
        proposal["revised_entity_types"] = []
        with self.assertRaisesRegex(InteractionError, "No TRF may remain active"):
            parse_trf_proposal(json.dumps(proposal), record, record["r0"], max_reason=240)

        proposal = trf_proposal(record, action="drop")
        decision = proposal["known_trf_decisions"][0]
        decision["evidence_sources"] = ["EXEMPLARS"]
        decision["evidence_demo_ids"] = [record["candidates"][0]["demo_idx"]]
        decision["reason_codes"] = ["BOUNDARY_EVIDENCE"]
        parsed = parse_trf_proposal(json.dumps(proposal), record, record["r0"], max_reason=240)
        self.assertEqual(parsed["known_trf_decisions"][0]["reason_codes"], ["BOUNDARY_EVIDENCE"])

    def test_obvious_adverbial_modifier_is_forced_out_of_trfs(self):
        record = rebuilt()
        record["sentence"] = "Work independently"
        record["r0"]["known_trfs"][0]["normalized_text"] = "independently"
        record["r0"]["active_trfs"] = ["independently"]
        proposal = trf_proposal(record)
        proposal["known_trf_decisions"][0]["normalized_text"] = "independently"
        parsed = parse_trf_proposal(json.dumps(proposal), record, record["r0"], max_reason=240)
        decision = parsed["known_trf_decisions"][0]
        self.assertEqual(decision["action"], "drop")
        self.assertEqual(decision["model_action"], "keep")
        self.assertEqual(decision["semantic_gate_action"], "forced_drop_descriptive_modifier")

    def test_explicit_skill_type_cue_is_forced_active(self):
        record = rebuilt()
        record["sentence"] = "Excellent communication skills"
        record["r0"]["known_trfs"][0]["normalized_text"] = "skills"
        record["r0"]["active_trfs"] = ["skills"]
        proposal = trf_proposal(record, action="drop")
        proposal["known_trf_decisions"][0]["normalized_text"] = "skills"
        parsed = parse_trf_proposal(json.dumps(proposal), record, record["r0"], max_reason=240)
        decision = parsed["known_trf_decisions"][0]
        self.assertEqual(decision["action"], "keep")
        self.assertEqual(decision["model_action"], "drop")
        self.assertEqual(decision["semantic_gate_action"], "forced_keep_explicit_skill_type_cue")

    def test_missing_skill_type_cue_is_forced_inactive(self):
        record = rebuilt()
        record["sentence"] = "Read the Latin alphabet"
        record["r0"]["known_trfs"][0]["normalized_text"] = "skills"
        record["r0"]["active_trfs"] = ["skills"]
        proposal = trf_proposal(record, action="keep")
        proposal["known_trf_decisions"][0]["normalized_text"] = "skills"
        parsed = parse_trf_proposal(json.dumps(proposal), record, record["r0"], max_reason=240)
        decision = parsed["known_trf_decisions"][0]
        self.assertEqual(decision["action"], "drop")
        self.assertEqual(decision["model_action"], "keep")
        self.assertEqual(decision["semantic_gate_action"], "forced_drop_missing_skill_type_cue")

    def test_nanosecond_overlap_is_enforced(self):
        audit = _overlap(
            {
                "trf": {"started_unix_ns": 10, "completed_unix_ns": 30},
                "exemplar": {"started_unix_ns": 20, "completed_unix_ns": 40},
            }
        )
        self.assertTrue(audit["overlapped"])
        with self.assertRaises(InteractionError):
            _overlap(
                {
                    "trf": {"started_unix_ns": 10, "completed_unix_ns": 19},
                    "exemplar": {"started_unix_ns": 20, "completed_unix_ns": 40},
                }
            )

    def test_provider_400_does_not_downgrade_or_retry(self):
        class BadRequest(Exception):
            status_code = 400

        class Endpoint:
            def __init__(self):
                self.calls = []

            def create(self, **kwargs):
                self.calls.append(kwargs)
                raise BadRequest("bad schema")

        endpoint = Endpoint()
        transport = SimpleNamespace(chat=SimpleNamespace(completions=endpoint))
        settings = QwenSettings(chat_model="fake", base_url="https://example.invalid", max_retries=5)
        client = JsonSchemaQwenClient(api_key="secret", settings=settings, client=transport)
        record = rebuilt()
        fmt = response_format(record, "trf", record["r0"], max_reason_characters=240)
        with self.assertRaises(QwenRequestError):
            client.chat([], temperature=0.0, max_tokens=10, response_format=fmt)
        self.assertEqual(len(endpoint.calls), 1)
        self.assertEqual(endpoint.calls[0]["response_format"]["type"], "json_schema")

    def test_call_budget_counts_complete_append_only_history(self):
        response = {
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }
        raw_records = [
            {
                "dataset_id": "d",
                "record_id": "1",
                "status": "failed",
                "attempts": [
                    {"kind": "initial", "response": response},
                    {"kind": "repair", "response": response},
                ],
            },
            {
                "dataset_id": "d",
                "record_id": "1",
                "status": "complete",
                "attempts": [{"kind": "initial", "response": response}],
            },
        ]
        expected = {
            "model_responses": 3,
            "repair_responses": 1,
            "failed_records": 0,
            "prompt_tokens": 30,
            "completion_tokens": 6,
        }
        self.assertEqual(_metrics(raw_records), expected)
        self.assertEqual(_cumulative_metrics(raw_records), expected)


if __name__ == "__main__":
    unittest.main()
