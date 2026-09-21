from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common.qwen_client import QwenSettings  # noqa: E402
from second_layer.collaborative_reflection_json_schema.branch_validator import (  # noqa: E402
    _validate_raw,
)
from second_layer.collaborative_reflection_json_schema.online import reflect_one  # noqa: E402
from second_layer.collaborative_reflection_json_schema.pipeline import (  # noqa: E402
    build_prompt,
    build_response_schema,
    parse_trf_response,
    request_sha256,
)
from second_layer.collaborative_reflection_json_schema.runner import (  # noqa: E402
    _branch_command,
)
from second_layer.collaborative_reflection_json_schema.schema_client import (  # noqa: E402
    JsonSchemaQwenClient,
)
from tests.second_layer.test_collaborative_reflection import (  # noqa: E402
    _exemplar_payload,
    _input,
    _trf_payload,
)


CHAT = {
    "model": "fixture-qwen",
    "temperature": 0.0,
    "max_tokens": 4096,
    "max_reason_characters": 240,
}


def _assert_closed(test: unittest.TestCase, schema: dict[str, Any]) -> None:
    test.assertNotIn("uniqueItems", schema)
    if schema.get("type") == "object":
        test.assertIs(schema.get("additionalProperties"), False)
        test.assertEqual(set(schema.get("required", [])), set(schema.get("properties", {})))
        for child in schema["properties"].values():
            _assert_closed(test, child)
    items = schema.get("items")
    if isinstance(items, dict):
        _assert_closed(test, items)


class DynamicSchemaTests(unittest.TestCase):
    def test_trf_root_is_exact_and_every_object_is_closed(self) -> None:
        schema = build_response_schema(_input(), "trf", 240)
        self.assertEqual(
            list(schema["properties"]),
            [
                "revised_entity_types",
                "original_trf_decisions",
                "added_trfs",
                "overall_reason",
                "unresolved_issues",
            ],
        )
        decisions = schema["properties"]["original_trf_decisions"]
        self.assertEqual((decisions["minItems"], decisions["maxItems"]), (1, 1))
        self.assertEqual(decisions["items"]["properties"]["confidence"]["maximum"], 5)
        self.assertEqual(decisions["items"]["properties"]["reason"]["maxLength"], 240)
        _assert_closed(self, schema)

    def test_exemplar_schema_requires_exactly_sixteen_and_known_enums(self) -> None:
        schema = build_response_schema(_input(), "exemplar", 240)
        judgments = schema["properties"]["revised_judgments"]
        self.assertEqual((judgments["minItems"], judgments["maxItems"]), (16, 16))
        item = judgments["items"]["properties"]
        self.assertEqual(item["helpfulness_score"]["minimum"], 1)
        self.assertEqual(item["helpfulness_score"]["maximum"], 5)
        self.assertEqual(set(item["role"]["enum"]), {"supporting", "contrastive", "irrelevant"})
        _assert_closed(self, schema)

    def test_empty_baseline_forbids_trf_relations(self) -> None:
        record = _input()
        record["baseline_context"]["trf_context"]["trfs"] = []
        schema = build_response_schema(record, "exemplar", 240)
        relations = schema["properties"]["revised_judgments"]["items"]["properties"]["trf_relations"]
        self.assertEqual(relations["maxItems"], 0)

    def test_prompt_uses_guidance_not_required_output_property(self) -> None:
        prompt = build_prompt(_input(), "trf", 60000, 240)
        body = json.loads(prompt["messages"][1]["content"])
        self.assertNotIn("required_output", body)
        self.assertIn("output_field_guidance", body)
        self.assertEqual(prompt["schema_version"], "trf-reflection-prompt-v2")
        self.assertEqual(prompt["response_format"]["type"], "json_schema")
        self.assertIs(prompt["response_format"]["json_schema"]["strict"], True)
        self.assertEqual(
            prompt["request_contract_sha256"],
            request_sha256(prompt["messages"], prompt["response_format"]),
        )


class JsonSchemaClientTests(unittest.TestCase):
    def test_sdk_receives_strict_schema_and_hash_covers_it(self) -> None:
        captured: list[dict[str, Any]] = []

        def create(**kwargs: Any) -> Any:
            captured.append(kwargs)
            return SimpleNamespace(
                id="response-1",
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(_trf_payload())), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            )

        sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        client = JsonSchemaQwenClient(
            api_key="fixture-key",
            settings=QwenSettings(chat_model="fixture", base_url="https://example.invalid", max_retries=1),
            client=sdk,
            sleep=lambda _: None,
        )
        prompt = build_prompt(_input(), "trf", 60000, 240)
        response = client.chat(
            prompt["messages"],
            temperature=0.0,
            max_tokens=4096,
            response_format=prompt["response_format"],
        )
        self.assertEqual(response["response_id"], "response-1")
        self.assertEqual(captured[0]["response_format"], prompt["response_format"])
        self.assertIs(captured[0]["response_format"]["json_schema"]["strict"], True)
        altered = copy.deepcopy(prompt["response_format"])
        altered["json_schema"]["name"] = "tampered"
        self.assertNotEqual(
            request_sha256(prompt["messages"], altered),
            prompt["request_contract_sha256"],
        )

    def test_provider_400_fails_once_without_json_object_fallback(self) -> None:
        captured: list[dict[str, Any]] = []

        class ProviderError(RuntimeError):
            status_code = 400

        def create(**kwargs: Any) -> Any:
            captured.append(kwargs)
            raise ProviderError("schema rejected")

        sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        client = JsonSchemaQwenClient(
            api_key="fixture-key",
            settings=QwenSettings(chat_model="fixture", base_url="https://example.invalid", max_retries=5),
            client=sdk,
            sleep=lambda _: None,
        )
        record = _input()
        prompt = build_prompt(record, "trf", 60000, 240)
        raw, parsed = reflect_one(
            branch="trf",
            record=record,
            prompt=prompt,
            client=client,
            chat=CHAT,
            max_repairs=2,
        )
        self.assertIsNone(parsed)
        self.assertEqual(raw["status"], "failed")
        self.assertEqual(len(raw["attempts"]), 1)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["response_format"]["type"], "json_schema")


class RepairAndAuditTests(unittest.TestCase):
    def test_wrapped_initial_is_rejected_and_repair_reuses_schema(self) -> None:
        record = _input()
        prompt = build_prompt(record, "trf", 60000, 240)
        formats: list[dict[str, Any]] = []
        contents = [
            json.dumps({"required_output": _trf_payload()}),
            json.dumps(_trf_payload()),
        ]

        class FakeClient:
            def chat(self, _messages: Any, **kwargs: Any) -> dict[str, Any]:
                formats.append(copy.deepcopy(kwargs["response_format"]))
                return {
                    "content": contents.pop(0),
                    "finish_reason": "stop",
                    "latency_ms": 1,
                    "attempts": 1,
                    "response_id": "fixture",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }

        raw, parsed = reflect_one(
            branch="trf",
            record=record,
            prompt=prompt,
            client=FakeClient(),
            chat=CHAT,
            max_repairs=2,
        )
        self.assertIsNotNone(parsed)
        self.assertEqual(len(raw["attempts"]), 2)
        self.assertIsNotNone(raw["attempts"][0]["parse_error"])
        self.assertEqual(formats[0], formats[1])
        with self.assertRaises(Exception):
            parse_trf_response(json.dumps({"required_output": _trf_payload()}), record, 240)

    def test_validator_audit_rejects_schema_message_or_hash_tampering(self) -> None:
        record = _input()
        prompt = build_prompt(record, "trf", 60000, 240)

        class FakeClient:
            def chat(self, _messages: Any, **_kwargs: Any) -> dict[str, Any]:
                return {
                    "content": json.dumps(_trf_payload()),
                    "finish_reason": "stop",
                    "latency_ms": 1,
                    "attempts": 1,
                    "response_id": "fixture",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }

        raw, _ = reflect_one(
            branch="trf", record=record, prompt=prompt, client=FakeClient(), chat=CHAT, max_repairs=2
        )
        for mutation in ("schema", "messages", "hash"):
            changed_prompt = copy.deepcopy(prompt)
            changed_raw = copy.deepcopy(raw)
            if mutation == "schema":
                changed_prompt["response_format"]["json_schema"]["name"] = "tampered"
            elif mutation == "messages":
                changed_prompt["messages"][0]["content"] += " tampered"
            else:
                changed_raw["attempts"][0]["request_sha256"] = "0" * 64
            with self.subTest(mutation=mutation):
                with self.assertRaises(Exception):
                    _validate_raw(changed_raw, changed_prompt, record, "trf", CHAT, 2)


class InterfaceTests(unittest.TestCase):
    def test_branch_command_uses_versioned_runner_and_ids(self) -> None:
        command = _branch_command(
            python=sys.executable,
            config_path=Path("reflection-json-schema.json"),
            paths=SimpleNamespace(root=Path("parent")),
            run_id="gate-v3",
            branch="trf",
            allow_network=True,
            resume=True,
            retry_failed=True,
        )
        self.assertTrue(command[1].endswith("collaborative_reflection_json_schema\\branch_runner.py"))
        self.assertIn("gate-v3-trf-reflection-jsonschema", command)
        self.assertIn("--allow-network", command)
        self.assertIn("--resume", command)
        self.assertIn("--retry-failed", command)

    def test_powershell_entry_points_to_versioned_protocol(self) -> None:
        text = (PROJECT_ROOT / "scripts" / "second_layer" / "run_reflection_json_schema.ps1").read_text(encoding="utf-8")
        self.assertIn("second_layer_reflection_json_schema.json", text)
        self.assertIn("collaborative_reflection_json_schema\\runner.py", text)
        for parameter in ("RunId", "BaseConcurrentRunId", "Limit", "PrepareOnly", "AllowNetwork", "Resume", "RetryFailed"):
            self.assertIn(f"${parameter}", text)


if __name__ == "__main__":
    unittest.main()
