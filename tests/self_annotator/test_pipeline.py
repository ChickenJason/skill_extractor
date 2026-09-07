from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


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
from self_annotator.generate_prompts import (  # noqa: E402
    generate_prompts,
    load_prompt_template,
    validate_records,
)
from self_annotator.parse_answers import (  # noqa: E402
    extract_inline_spans,
    parse_raw_records,
    parse_response_content,
    parse_response_content_detailed,
    validate_inline_annotation,
)
from self_annotator.reaggregate import reaggregate_run  # noqa: E402
from self_annotator.selection import select_annotations  # noqa: E402
from self_annotator.consensus import (  # noqa: E402
    aggregate_records,
)


AGGREGATION = load_json(PROJECT_ROOT / "config" / "self_annotator.json")["aggregation"]


def annotation(sentence: str, *ranges: tuple[int, int]) -> dict:
    spans = [
        {"text": sentence[start:end], "start": start, "end": end}
        for start, end in ranges
    ]
    return {"has_skill": 1 if spans else 0, "spans": spans}


def annotation_for_text(sentence: str, *texts: str) -> dict:
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for text in texts:
        start = sentence.index(text, cursor)
        ranges.append((start, start + len(text)))
        cursor = start + len(text)
    return annotation(sentence, *ranges)


def parsed_sample(
    sample_index: int,
    sentence: str,
    value: dict | None,
    *,
    idx: int = 1,
    status: str = "ok",
) -> dict:
    return {
        "idx": idx,
        "sentence": sentence,
        "sample_index": sample_index,
        "parse_status": status,
        "parse_error": None if status != "failed" else "failure",
        "annotation": value,
    }


def aggregate(samples: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    return aggregate_records(samples, AGGREGATION)


class PromptAndPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_json(PROJECT_ROOT / "config" / "self_annotator.json")
        self.template = load_prompt_template(
            PROJECT_ROOT / self.config["prompt"]["template"]
        )

    def test_prompt_contract_is_unchanged(self) -> None:
        self.assertEqual(
            self.config["prompt"]["version"], "skill-inline-annotation-v4.1"
        )
        self.assertEqual(self.template.count("<sentence_json>"), 1)
        self.assertIn('"annotated_sentence"', self.template)
        self.assertIn("<skill>", self.template)
        self.assertIn("character-for-character identical", self.template)

    def test_prompt_generation_json_quotes_sentence(self) -> None:
        records = validate_records([{"idx": 7, "sentence": 'Use "C++".'}])
        prompts = generate_prompts(
            records, self.template, self.config["prompt"]["version"]
        )
        self.assertIn(json.dumps('Use "C++".', ensure_ascii=False), prompts[0]["prompt"])

    def test_dataset_hash_and_external_working_directory(self) -> None:
        dataset_path = PROJECT_ROOT / "data" / "raw" / "skill_sentences.json"
        self.assertEqual(len(load_json(dataset_path)), 326)
        self.assertEqual(
            sha256_file(dataset_path),
            "0c426247d64c23db7228edb20400565161e671f715abe1351fe609ed813d27a8",
        )
        script = PROJECT_ROOT / "code" / "self_annotator" / "generate_prompts.py"
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--config",
                    "config/self_annotator.json",
                    "--input",
                    "data/raw/skill_sentences.json",
                    "--validate-only",
                ],
                cwd=directory,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)


class InlineParserTests(unittest.TestCase):
    def test_positive_negative_and_offsets(self) -> None:
        sentence = "Experience in project planning."
        positive = {"annotated_sentence": "Experience in <skill>project planning</skill>."}
        self.assertEqual(
            validate_inline_annotation(positive, sentence),
            annotation_for_text(sentence, "project planning"),
        )
        self.assertEqual(
            validate_inline_annotation({"annotated_sentence": sentence}, sentence),
            {"has_skill": 0, "spans": []},
        )

    def test_direct_fenced_and_projected_json(self) -> None:
        sentence = "Use Python ."
        direct = json.dumps({"annotated_sentence": "Use <skill>Python</skill> ."})
        self.assertEqual(parse_response_content(direct, sentence)[1], "ok")
        self.assertEqual(parse_response_content(f"```json\n{direct}\n```", sentence)[1], "recovered")
        projected = parse_response_content_detailed(
            json.dumps({"annotated_sentence": "Use <skill>Python</skill>."}), sentence
        )
        self.assertEqual(projected["parse_status"], "recovered")
        self.assertEqual(projected["reconstruction_status"], "projected")

    def test_malformed_and_ambiguous_payloads_fail(self) -> None:
        sentence = "research and research"
        ambiguous = parse_response_content_detailed(
            json.dumps({"annotated_sentence": "Do <skill>research</skill>"}), sentence
        )
        self.assertEqual(ambiguous["parse_status"], "failed")
        with self.assertRaises(ValueError):
            validate_inline_annotation(
                {"annotated_sentence": "<skill><skill>x</skill></skill>"}, "x"
            )

    def test_repeated_surface_forms_and_parser_label_ids(self) -> None:
        sentence = "research and research"
        _, spans = extract_inline_spans(
            "<skill>research</skill> and <skill>research</skill>", sentence
        )
        self.assertEqual([span["start"] for span in spans], [0, 13])
        raw = [{
            "idx": 9,
            "sentence": sentence,
            "prompt_version": "skill-inline-annotation-v4.1",
            "model": "fake",
            "samples": [{
                "sample_index": sample_index,
                "status": "ok",
                "content": json.dumps({
                    "annotated_sentence": "<skill>research</skill> and <skill>research</skill>"
                }),
            } for sample_index in range(5)],
        }]
        parsed = parse_raw_records(raw)
        self.assertEqual(
            [span["label_id"] for span in parsed[0]["annotation"]["spans"]],
            ["span:9:0:8", "span:9:13:21"],
        )
        self.assertNotIn("bio_tags", parsed[0])
        self.assertNotIn("alignment_warnings", parsed[0])


class ExistenceGateTests(unittest.TestCase):
    def test_has_skill_vote_counts_zero_through_five(self) -> None:
        sentence = "Use Python."
        positive = annotation_for_text(sentence, "Python")
        negative = annotation(sentence)
        expected = {
            0: "negative",
            1: "negative",
            2: "unsolved",
            3: "accepted",
            4: "accepted",
            5: "accepted",
        }
        for has_votes, status in expected.items():
            with self.subTest(has_votes=has_votes):
                samples = [
                    parsed_sample(index, sentence, positive if index < has_votes else negative)
                    for index in range(5)
                ]
                consensus, _, _ = aggregate(samples)
                self.assertEqual(consensus[0]["status"], status)

    def test_failed_samples_are_not_no_skill_votes(self) -> None:
        sentence = "Use Python."
        positive = annotation_for_text(sentence, "Python")
        negative = annotation(sentence)
        samples = [
            parsed_sample(0, sentence, positive),
            parsed_sample(1, sentence, negative),
            parsed_sample(2, sentence, negative),
            parsed_sample(3, sentence, negative),
            parsed_sample(4, sentence, None, status="failed"),
        ]
        consensus, _, _ = aggregate(samples)
        self.assertEqual(consensus[0]["status"], "unsolved")
        self.assertEqual(consensus[0]["existence_consensus"]["no_skill_votes"], 3)

    def test_insufficient_valid_samples_abstain(self) -> None:
        sentence = "Use Python."
        positive = annotation_for_text(sentence, "Python")
        samples = [
            parsed_sample(index, sentence, positive if index < 2 else None,
                          status="ok" if index < 2 else "failed")
            for index in range(5)
        ]
        consensus, _, _ = aggregate(samples)
        self.assertEqual(consensus[0]["status"], "abstained")

    def test_four_valid_no_skill_votes_are_negative(self) -> None:
        sentence = "Use Python."
        samples = [parsed_sample(index, sentence, annotation(sentence)) for index in range(4)]
        samples.append(parsed_sample(4, sentence, None, status="failed"))
        consensus, _, _ = aggregate(samples)
        self.assertEqual(consensus[0]["status"], "negative")

    def test_missing_or_extra_sample_index_is_structural_error(self) -> None:
        sentence = "Use Python."
        samples = [parsed_sample(index, sentence, annotation(sentence)) for index in range(5)]
        samples[-1]["sample_index"] = 5
        with self.assertRaisesRegex(ValueError, "exactly sample_index 0..4"):
            aggregate(samples)


class SpanConsensusTests(unittest.TestCase):
    def test_exact_three_of_five_and_formal_string_interface(self) -> None:
        sentence = "Use Python."
        positive = annotation_for_text(sentence, "Python")
        samples = [
            parsed_sample(index, sentence, positive if index < 3 else annotation(sentence))
            for index in range(5)
        ]
        consensus, _, uncertainty = aggregate(samples)
        self.assertEqual(consensus[0]["status"], "accepted")
        self.assertEqual(consensus[0]["spans"], ["Python"])
        self.assertEqual(consensus[0]["accepted_span_details"][0]["accepted_by"], "exact_vote")
        formal, _ = select_annotations(consensus)
        self.assertEqual(formal[0]["spans"], ["Python"])
        self.assertIn("pairwise_set_jaccard", uncertainty[0])
        self.assertNotIn("tokens", uncertainty[0])

    def test_multi_target_explained_variant(self) -> None:
        sentence = "Python SQL"
        separate = annotation_for_text(sentence, "Python", "SQL")
        broad = annotation_for_text(sentence, "Python SQL")
        samples = [parsed_sample(index, sentence, separate) for index in range(4)]
        samples.append(parsed_sample(4, sentence, broad))
        consensus, audit, _ = aggregate(samples)
        self.assertEqual(consensus[0]["status"], "accepted")
        self.assertEqual(consensus[0]["spans"], ["Python", "SQL"])
        explained = audit[0]["explained_variants"][0]
        self.assertEqual(len(explained["explained_by_label_ids"]), 2)

    def test_repeated_text_at_distinct_offsets_is_preserved(self) -> None:
        sentence = "research and research"
        value = annotation_for_text(sentence, "research", "research")
        consensus, _, _ = aggregate(
            [parsed_sample(index, sentence, value) for index in range(5)]
        )
        self.assertEqual(consensus[0]["spans"], ["research", "research"])
        details = consensus[0]["accepted_span_details"]
        self.assertNotEqual(details[0]["start"], details[1]["start"])

    def test_family_support_one_two_are_noise(self) -> None:
        sentence = "Python and SQL"
        python = annotation_for_text(sentence, "Python")
        both = annotation_for_text(sentence, "Python", "SQL")
        for support in (1, 2):
            with self.subTest(support=support):
                samples = [
                    parsed_sample(index, sentence, both if index < support else python)
                    for index in range(5)
                ]
                consensus, audit, _ = aggregate(samples)
                self.assertEqual(consensus[0]["status"], "accepted")
                sql_family = next(
                    family for family in audit[0]["families"]
                    if family["members"][0]["text"] == "SQL"
                )
                self.assertEqual(sql_family["support"], support)

    def test_family_support_three_uses_hard_match(self) -> None:
        sentence = "Python and project planning"
        python = annotation_for_text(sentence, "Python")
        broad = annotation_for_text(sentence, "Python", "project planning")
        narrow = annotation_for_text(sentence, "Python", "planning")
        samples = [
            parsed_sample(0, sentence, broad),
            parsed_sample(1, sentence, broad),
            parsed_sample(2, sentence, narrow),
            parsed_sample(3, sentence, python),
            parsed_sample(4, sentence, python),
        ]
        consensus, audit, _ = aggregate(samples)
        self.assertEqual(consensus[0]["status"], "accepted")
        self.assertEqual(consensus[0]["spans"], ["Python", "project planning"])
        detail = consensus[0]["accepted_span_details"][1]
        self.assertEqual(detail["accepted_by"], "family_hard_match")
        family = next(family for family in audit[0]["families"] if family["support"] == 3)
        self.assertEqual(family["hard_match"]["anchor_label_id"], detail["label_id"])

    def test_family_support_four_boundary_variants_use_hard_match(self) -> None:
        sentence = "Python and project planning"
        python = annotation_for_text(sentence, "Python")
        broad = annotation_for_text(sentence, "Python", "project planning")
        narrow = annotation_for_text(sentence, "Python", "planning")
        samples = [
            parsed_sample(0, sentence, broad),
            parsed_sample(1, sentence, broad),
            parsed_sample(2, sentence, narrow),
            parsed_sample(3, sentence, narrow),
            parsed_sample(4, sentence, python),
        ]
        consensus, _, _ = aggregate(samples)
        self.assertEqual(consensus[0]["status"], "accepted")
        self.assertEqual(consensus[0]["spans"], ["Python", "planning"])
        self.assertEqual(
            consensus[0]["accepted_span_details"][1]["accepted_by"],
            "family_hard_match",
        )

    def test_support_five_boundary_family_hard_matches(self) -> None:
        sentence = "Use project planning skills."
        broad = annotation_for_text(sentence, "project planning")
        narrow = annotation_for_text(sentence, "planning")
        extended = annotation_for_text(sentence, "planning skills")
        samples = [
            parsed_sample(0, sentence, broad),
            parsed_sample(1, sentence, broad),
            parsed_sample(2, sentence, narrow),
            parsed_sample(3, sentence, narrow),
            parsed_sample(4, sentence, extended),
        ]
        consensus, _, _ = aggregate(samples)
        self.assertEqual(consensus[0]["status"], "accepted")
        self.assertEqual(consensus[0]["spans"], ["planning"])
        self.assertEqual(
            consensus[0]["accepted_span_details"][0]["accepted_by"],
            "family_hard_match",
        )

    def test_long_family_and_overlap_chain_are_unsolved(self) -> None:
        long_sentence = "Develop our atomic-scale model for SCR catalysis."
        broad = annotation_for_text(long_sentence, "atomic-scale model for SCR catalysis")
        middle = annotation_for_text(long_sentence, "model for SCR catalysis")
        short = annotation_for_text(long_sentence, "SCR catalysis")
        long_samples = [
            parsed_sample(0, long_sentence, broad),
            parsed_sample(1, long_sentence, broad),
            parsed_sample(2, long_sentence, middle),
            parsed_sample(3, long_sentence, middle),
            parsed_sample(4, long_sentence, short),
        ]
        consensus, _, _ = aggregate(long_samples)
        self.assertEqual(consensus[0]["status"], "unsolved")
        self.assertIn("family_anchor_overlap_failed", consensus[0]["reasons"])

        chain_sentence = "alpha beta gamma delta"
        ranges = ((0, 10), (6, 16), (11, 22))
        chain_samples = [
            parsed_sample(index, chain_sentence, annotation(chain_sentence, ranges[0 if index < 2 else 1 if index < 4 else 2]))
            for index in range(5)
        ]
        consensus, _, _ = aggregate(chain_samples)
        self.assertEqual(consensus[0]["status"], "unsolved")
        self.assertIn("family_anchor_overlap_failed", consensus[0]["reasons"])

    def test_three_valid_samples_can_hard_match(self) -> None:
        sentence = "Use project planning."
        anchor = annotation_for_text(sentence, "planning")
        broad = annotation_for_text(sentence, "project planning")
        samples = [
            parsed_sample(0, sentence, anchor),
            parsed_sample(1, sentence, anchor),
            parsed_sample(2, sentence, broad),
            parsed_sample(3, sentence, None, status="failed"),
            parsed_sample(4, sentence, None, status="failed"),
        ]
        consensus, audit, _ = aggregate(samples)
        self.assertEqual(consensus[0]["status"], "accepted")
        self.assertEqual(consensus[0]["spans"], ["planning"])
        self.assertEqual(audit[0]["families"][0]["support"], 3)


def _raw_record(idx: int, sentence: str, positive: bool) -> dict:
    annotated = sentence.replace("Python", "<skill>Python</skill>") if positive else sentence
    return {
        "idx": idx,
        "sentence": sentence,
        "prompt": "unchanged prompt",
        "prompt_version": "skill-inline-annotation-v4.1",
        "model": "fake-model",
        "status": "complete",
        "samples": [
            {
                "sample_index": index,
                "status": "ok",
                "content": json.dumps({"annotated_sentence": annotated}),
            }
            for index in range(5)
        ],
    }


class OfflineReplayTests(unittest.TestCase):
    def test_reaggregation_is_offline_immutable_and_records_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            raw_path = source / "raw" / "responses.jsonl"
            records = [
                _raw_record(1, "Use Python.", True),
                _raw_record(2, "No explicit skill.", False),
            ]
            atomic_write_jsonl(raw_path, records)
            atomic_write_json(
                source / "manifest.json",
                {
                    "run_id": "source-five",
                    "status": "completed",
                    "compatibility": {
                        "annotation_schema": "skill-inline-span-v1",
                        "selected_count": 2,
                    },
                },
            )
            before = sha256_file(raw_path)
            config = load_json(PROJECT_ROOT / "config" / "self_annotator.json")
            config["output"]["runs_root"] = str(root / "derived")
            manifest = reaggregate_run(
                source_root=source, run_id="offline-test", config=config
            )
            target = root / "derived" / "offline-test"
            self.assertEqual(manifest["status"], "completed")
            self.assertFalse(manifest["network_called"])
            self.assertTrue(manifest["source_unchanged"])
            self.assertEqual(before, sha256_file(raw_path))
            self.assertEqual(
                load_json(target / "raw" / "source_reference.json")["source_raw_sha256"],
                before,
            )
            decisions = read_jsonl(target / "selected" / "decisions.jsonl")
            self.assertEqual([item["status"] for item in decisions], ["accepted", "negative"])


if __name__ == "__main__":
    unittest.main()
