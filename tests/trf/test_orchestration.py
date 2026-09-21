from __future__ import annotations

import hashlib
import json
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

from common.io_utils import atomic_write_json, load_json, read_jsonl  # noqa: E402
from trf.orchestration.runner import FullTRFError, run  # noqa: E402
from trf.orchestration.validator import validate  # noqa: E402
from tests.trf.fixtures import (  # noqa: E402
    DEMONSTRATIONS,
    create_fake_completed_offline,
    prepare_v06_trf_config,
)


class FullRunFakeClient:
    def __init__(self) -> None:
        self.embedding_calls = 0
        self.chat_calls = 0

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
        self.embedding_calls += 1
        return {
            "vectors": [self._vector(text, dimensions) for text in texts],
            "batches": [
                {
                    "start": 0,
                    "count": len(texts),
                    "attempts": 1,
                    "latency_ms": 0,
                    "response_id": f"full-embedding-{self.embedding_calls}",
                    "model": model,
                    "usage": {"prompt_tokens": len(texts), "total_tokens": len(texts)},
                }
            ],
        }

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        model: str,
    ) -> dict[str, Any]:
        self.chat_calls += 1
        combined = "\n".join(item["content"] for item in messages)
        if "TRF expert's final exact Skill span extractor" in combined:
            payload = json.loads(messages[1]["content"])
            content = json.dumps(
                {"annotated_sentence": payload["target_sentence"]}, ensure_ascii=False
            )
        elif len(messages) == 1:
            content = '{"entity_types":["Skill"]}'
        else:
            content = '{"trfs":["communication","open full-run feature"]}'
        return {
            "content": content,
            "finish_reason": "stop",
            "latency_ms": 0,
            "attempts": 1,
            "response_id": f"full-chat-{self.chat_calls}",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }


class CompleteTRFRunTests(unittest.TestCase):
    def _configs(self, directory: str) -> tuple[Path, Path]:
        root = Path(directory)
        config_path, _ = prepare_v06_trf_config(root)
        return config_path, root / "trf-runs"

    @staticmethod
    def _args(
        config: Path,
        **overrides: Any,
    ) -> Namespace:
        values = {
            "run_id": "full-test",
            "demonstrations": DEMONSTRATIONS,
            "mode": "leave-one-out",
            "input": None,
            "feedback": None,
            "limit": 2,
            "prepare_only": True,
            "allow_model_download": False,
            "allow_network": True,
            "confirm_full_run": False,
            "resume": False,
            "retry_failed": False,
            "reuse_embeddings_from": None,
            "config": str(config),
        }
        values.update(overrides)
        return Namespace(**values)

    @staticmethod
    def _build_validated_offline(
        config_path: Path,
        demonstrations: str,
        run_id: str,
        allow_model_download: bool,
        command: list[str],
    ) -> None:
        create_fake_completed_offline(
            config_path,
            demonstrations,
            run_id,
            allow_model_download,
            command,
        )

    def test_prepare_resume_and_joint_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, full_root = self._configs(directory)
            client = FullRunFakeClient()
            prepare = self._args(config)
            status = run(
                prepare,
                target_client_factory=lambda: client,
                offline_executor=self._build_validated_offline,
            )
            self.assertEqual(status, "partial")
            parent = full_root / "full-test"
            manifest = load_json(parent / "manifest.json")
            self.assertEqual(parent.name, "full-test")
            self.assertEqual(
                manifest["compatibility"]["demonstrations"]["dataset_id"],
                "skill-sentences-326-v1",
            )
            self.assertEqual(manifest["stages"]["offline_trf"], "completed")
            self.assertEqual(manifest["stages"]["target_trf"], "partial")
            self.assertEqual(
                manifest["network"],
                {"offline_model": False, "target_qwen": True},
            )
            self.assertTrue(manifest["network_called"])
            generated = load_json(parent / "target-config.generated.json")
            self.assertEqual(generated["source_trf"]["run_id"], "full-test")
            self.assertEqual(
                Path(generated["source_trf"]["run_root"]).name,
                "offline",
            )

            resume = self._args(
                config,
                prepare_only=False,
                resume=True,
            )
            self.assertEqual(
                run(
                    resume,
                    target_client_factory=lambda: client,
                    offline_executor=self._build_validated_offline,
                ),
                "completed",
            )
            result = validate("full-test", config)
            self.assertEqual(result["status"], "valid")
            self.assertEqual(result["targets"], 2)
            self.assertEqual(client.chat_calls, 6)
            target_records = read_jsonl(
                Path(manifest["children"]["target"]["root"])
                / "parsed"
                / "records.jsonl"
            )
            self.assertEqual(len(target_records), 2)
            predictions = read_jsonl(
                Path(manifest["children"]["target"]["root"])
                / "prediction"
                / "records.jsonl"
            )
            self.assertEqual(len(predictions), 2)

    def test_unrestricted_online_run_requires_confirmation(self) -> None:
        args = Namespace(
            retry_failed=False,
            resume=False,
            allow_network=True,
            prepare_only=False,
            limit=None,
            confirm_full_run=False,
        )
        with self.assertRaisesRegex(FullTRFError, "confirm-full-run"):
            run(args)


if __name__ == "__main__":
    unittest.main()
