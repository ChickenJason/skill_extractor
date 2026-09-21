from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from second_layer.bidirectional_interaction.gate import (
    evaluate_gate,
    load_gate_spec,
    target_identities,
)
from second_layer.bidirectional_interaction.prompts import build_prompt
from tests.second_layer.test_bidirectional_interaction import rebuilt


FULL_GATE = PROJECT_ROOT / "config" / "gates" / "bidirectional_interaction_stratified_v1.json"
SMOKE_GATE = PROJECT_ROOT / "config" / "gates" / "bidirectional_interaction_smoke3_v1.json"
LITE_GATE = PROJECT_ROOT / "config" / "gates" / "bidirectional_interaction_smoke_lite_v1.json"


class BidirectionalInteractionGateTests(unittest.TestCase):
    def test_full_gate_frozen_anchor_counts(self):
        spec = load_gate_spec(FULL_GATE)
        self.assertEqual(len(spec["targets"]), 12)
        self.assertEqual(sum(len(item["exemplar_anchors"]) for item in spec["targets"]), 24)
        self.assertEqual(sum(len(item["trf_anchors"]["must_keep"]) for item in spec["targets"]), 4)
        self.assertEqual(sum(len(item["trf_anchors"]["must_drop"]) for item in spec["targets"]), 14)
        self.assertEqual(sum(anchor["expected_change_from_baseline"] is True for item in spec["targets"] for anchor in item["exemplar_anchors"]), 5)

    def test_smoke_gate_exact_targets_and_budget(self):
        spec = load_gate_spec(SMOKE_GATE)
        self.assertEqual([item["idx"] for item in spec["targets"]], [50, 115, 312])
        self.assertEqual(spec["acceptance"]["maximum_model_responses"], 15)
        self.assertEqual(spec["acceptance"]["maximum_repair_responses"], 3)
        self.assertEqual(len(target_identities(spec)), 3)

    def test_lite_gate_uses_structural_checks_without_semantic_anchors(self):
        spec = load_gate_spec(LITE_GATE)
        self.assertEqual([item["idx"] for item in spec["targets"]], [50, 312])
        self.assertEqual(
            sum(len(item["exemplar_anchors"]) for item in spec["targets"]), 0
        )
        self.assertEqual(spec["acceptance"]["minimum_changed_exemplar_anchors"], 0)
        contexts = []
        for position, target in enumerate(spec["targets"]):
            contexts.append(
                {
                    **{
                        key: target[key]
                        for key in (
                            "dataset_id", "record_id", "idx", "sentence", "source_sha256"
                        )
                    },
                    "assembly_status": "ready",
                    "baseline_state": {
                        "exemplar": {"judgments": [], "selection": {"selected": []}}
                    },
                    "final_state": {
                        "trf": {"known_trfs": [], "active_trfs": ["tools"] if position == 0 else []},
                        "exemplar": {
                            "judgments": [],
                            "selection": {"selected": [{"demo_idx": 176}]} if position == 0 else {"selected": []},
                        },
                    },
                }
            )
        result = evaluate_gate(
            spec,
            contexts,
            {"model_responses": 4, "repair_responses": 0, "failed_records": 0},
        )
        self.assertEqual(result["status"], "passed")

    def test_oracle_edits_do_not_enter_prompt_or_hash(self):
        record = rebuilt()
        spec = load_gate_spec(SMOKE_GATE)
        before = build_prompt(record, "trf", record["r0"], record["h0"], 1, max_characters=60000, max_reason_characters=240)
        edited = copy.deepcopy(spec)
        edited["targets"][0]["exemplar_anchors"][0]["minimum_score"] = 1
        edited["targets"][0]["exemplar_anchors"].pop()
        edited["targets"][0]["exemplar_anchors"].append(copy.deepcopy(spec["targets"][0]["exemplar_anchors"][0]))
        after = build_prompt(record, "trf", record["r0"], record["h0"], 1, max_characters=60000, max_reason_characters=240)
        self.assertEqual(before["messages"], after["messages"])
        self.assertEqual(before["prompt_sha256"], after["prompt_sha256"])
        self.assertEqual(before["request_contract_sha256"], after["request_contract_sha256"])


if __name__ == "__main__":
    unittest.main()
