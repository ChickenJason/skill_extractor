from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.io_utils import MissingEnvironmentVariable, sha256_file  # noqa: E402
from common.qwen_client import QwenClient, QwenSettings  # noqa: E402


ASK_PATH = PROJECT_ROOT / "code" / "self_annotator" / "ask_qwen.py"
ASK_SPEC = importlib.util.spec_from_file_location("span_xmlc_ask_qwen", ASK_PATH)
assert ASK_SPEC and ASK_SPEC.loader
ASK = importlib.util.module_from_spec(ASK_SPEC)
sys.modules[ASK_SPEC.name] = ASK
ASK_SPEC.loader.exec_module(ASK)


class FakeCompletions:
    def __init__(self) -> None:
        self.calls = 0
        self.kwargs = None

    def create(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        return SimpleNamespace(
            id=f"response-{self.calls}",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content='{"annotated_sentence":"No explicit skill."}'
                    ),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


class FakeSDK:
    def __init__(self) -> None:
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class FakeEmbeddings:
    def __init__(self) -> None:
        self.calls = 0
        self.kwargs = []

    def create(self, **kwargs):
        self.calls += 1
        self.kwargs.append(kwargs)
        dimensions = kwargs["dimensions"]
        return SimpleNamespace(
            id=f"embedding-{self.calls}",
            model=kwargs["model"],
            data=[
                SimpleNamespace(index=index, embedding=[float(index + 1)] * dimensions)
                for index, _ in reversed(list(enumerate(kwargs["input"])))
            ],
            usage=SimpleNamespace(prompt_tokens=len(kwargs["input"]), total_tokens=len(kwargs["input"])),
        )


class FakeEmbeddingSDK(FakeSDK):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = FakeEmbeddings()


def fake_client(fake: FakeSDK) -> QwenClient:
    return QwenClient(
        api_key="sk-test",
        settings=QwenSettings(
            chat_model="qwen-test",
            base_url="https://example.invalid/v1",
            max_retries=1,
        ),
        client=fake,
        sleep=lambda _: None,
    )


class QwenTests(unittest.TestCase):
    def test_embedding_details_preserve_vector_only_interface(self) -> None:
        fake = FakeEmbeddingSDK()
        client = QwenClient(
            api_key="sk-test",
            settings=QwenSettings(
                chat_model="qwen-test",
                embedding_model="embed-test",
                base_url="https://example.invalid/v1",
                max_retries=1,
            ),
            client=fake,
            sleep=lambda _: None,
        )
        detailed = client.embeddings_detailed(
            ["a", "b", "c"], dimensions=4, batch_size=2
        )
        self.assertEqual(len(detailed["vectors"]), 3)
        self.assertEqual(detailed["vectors"], [[1.0] * 4, [2.0] * 4, [1.0] * 4])
        self.assertEqual([item["count"] for item in detailed["batches"]], [2, 1])
        self.assertEqual(detailed["batches"][0]["response_id"], "embedding-1")
        self.assertEqual(client.embeddings(["a"], dimensions=4), [[1.0] * 4])

    def test_embedding_response_indexes_must_be_complete_and_unique(self) -> None:
        fake = FakeEmbeddingSDK()
        fake.embeddings.create = lambda **kwargs: SimpleNamespace(
            id="bad-indexes",
            model=kwargs["model"],
            data=[
                SimpleNamespace(index=0, embedding=[1.0] * kwargs["dimensions"]),
                SimpleNamespace(index=0, embedding=[2.0] * kwargs["dimensions"]),
            ],
            usage=SimpleNamespace(prompt_tokens=2, total_tokens=2),
        )
        client = QwenClient(
            api_key="sk-test",
            settings=QwenSettings(
                chat_model="qwen-test",
                embedding_model="embed-test",
                base_url="https://example.invalid/v1",
                max_retries=1,
            ),
            client=fake,
            sleep=lambda _: None,
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            client.embeddings_detailed(["a", "b"], dimensions=4)

    def test_five_samples_and_structured_json_are_preserved(self) -> None:
        fake = FakeSDK()
        query = {
            "idx": 1,
            "sentence": "No explicit skill.",
            "prompt": "annotate",
            "prompt_version": "skill-inline-annotation-v4.1",
        }
        result = ASK.generate_responses_per_query_multiquery(
            fake_client(fake),
            query,
            {"samples": 5, "temperature": 0.7, "max_tokens": 128},
        )
        self.assertEqual(len(result["samples"]), 5)
        self.assertEqual(fake.completions.calls, 5)
        self.assertEqual(fake.completions.kwargs["response_format"], {"type": "json_object"})

    def test_missing_credentials_fail_before_any_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = json.loads(
                (PROJECT_ROOT / "config" / "self_annotator.json").read_text(
                    encoding="utf-8"
                )
            )
            path.write_text(json.dumps(config), encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(MissingEnvironmentVariable):
                    ASK.load_runtime_config(path)

    def test_manifest_compatibility_contains_schema_and_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompt_path = Path(directory) / "prompts.json"
            prompt_path.write_text("[]", encoding="utf-8")
            config = json.loads(
                (PROJECT_ROOT / "config" / "self_annotator.json").read_text(
                    encoding="utf-8"
                )
            )
            payload = ASK.compatibility_payload(
                input_path=prompt_path, records=[], config=config, limit=None
            )
            self.assertEqual(
                payload["consensus_schema"],
                "span-xmlc-majority-anchor-overlap-v2",
            )
            self.assertEqual(payload["aggregation"], config["aggregation"])
            self.assertEqual(payload["input_sha256"], payload["prompt_input_sha256"])

    def test_source_sentences_are_separate_from_prompt_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sentences.json"
            prompts = root / "prompts.json"
            source.write_text(
                json.dumps([{"idx": 1, "sentence": "source sentence"}]),
                encoding="utf-8",
            )
            prompt_records = [
                {
                    "idx": 1,
                    "sentence": "source sentence",
                    "prompt": "rendered prompt",
                    "prompt_version": "v1",
                }
            ]
            prompts.write_text(json.dumps(prompt_records), encoding="utf-8")
            config = json.loads(
                (PROJECT_ROOT / "config" / "self_annotator.json").read_text(
                    encoding="utf-8"
                )
            )
            ASK.validate_source_alignment(source, prompt_records, None)
            payload = ASK.compatibility_payload(
                input_path=prompts,
                source_input_path=source,
                records=prompt_records,
                config=config,
                limit=None,
            )
            self.assertEqual(payload["input_sha256"], sha256_file(source))
            self.assertEqual(payload["prompt_input_sha256"], sha256_file(prompts))
            source.write_text(
                json.dumps([{"idx": 1, "sentence": "different"}]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "align"):
                ASK.validate_source_alignment(source, prompt_records, None)

    def test_config_rejects_non_five_sampling_and_bad_anchor_contract(self) -> None:
        config = json.loads(
            (PROJECT_ROOT / "config" / "self_annotator.json").read_text(encoding="utf-8")
        )
        config["provider"]["api_key"] = "test"
        config["provider"]["base_url"] = "https://example.invalid"
        ASK.validate_config(config)
        config["generation"]["samples"] = 4
        with self.assertRaisesRegex(ValueError, "exactly 5"):
            ASK.validate_config(config)
        config["generation"]["samples"] = 5
        config["aggregation"]["hard_match_family_votes"] = 5
        with self.assertRaisesRegex(ValueError, "exactly 3"):
            ASK.validate_config(config)
        config["aggregation"]["hard_match_family_votes"] = 3
        config["aggregation"]["hard_match"]["char_iou_weight"] = 0.2
        with self.assertRaisesRegex(ValueError, "fields must be exactly"):
            ASK.validate_config(config)
        del config["aggregation"]["hard_match"]["char_iou_weight"]
        config["aggregation"]["hard_match"]["overlap_metric"] = "char_iou"
        with self.assertRaisesRegex(ValueError, "anchor_character_coverage"):
            ASK.validate_config(config)


if __name__ == "__main__":
    unittest.main()
