"""Qwen client surface that permits only named strict JSON Schemas."""

from __future__ import annotations

import copy
import time
from typing import Any, Sequence

from common.qwen_client import QwenClient, QwenRequestError


def _value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def validate_response_format(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"type", "json_schema"}:
        raise ValueError("Interaction response_format must contain type and json_schema")
    schema = value.get("json_schema")
    if value.get("type") != "json_schema" or not isinstance(schema, dict):
        raise ValueError("Interaction response_format must use json_schema")
    if set(schema) != {"name", "strict", "schema"} or schema.get("strict") is not True:
        raise ValueError("Interaction JSON Schema must be named and strict")
    root = schema.get("schema")
    if not isinstance(root, dict) or root.get("type") != "object" or root.get("additionalProperties") is not False:
        raise ValueError("Interaction JSON Schema root must be a closed object")
    return copy.deepcopy(value)


class JsonSchemaQwenClient(QwenClient):
    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any],
        model: str | None = None,
    ) -> dict[str, Any]:
        request = {
            "model": model or self.settings.chat_model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "extra_body": {"enable_thinking": self.settings.enable_thinking},
            "response_format": validate_response_format(response_format),
        }
        started = time.perf_counter()
        response, attempts = self._retry(
            lambda: self.client.chat.completions.create(**request),
            "Chat completion",
        )
        latency_ms = round((time.perf_counter() - started) * 1000)
        choices = _value(response, "choices", [])
        if not choices:
            raise QwenRequestError("Chat completion returned no choices", attempts=attempts, retriable=False)
        message = _value(choices[0], "message")
        usage = _value(response, "usage")
        return {
            "content": _value(message, "content", "") or "",
            "finish_reason": _value(choices[0], "finish_reason"),
            "latency_ms": latency_ms,
            "attempts": attempts,
            "response_id": _value(response, "id"),
            "usage": {
                "prompt_tokens": _value(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": _value(usage, "completion_tokens", 0) or 0,
                "total_tokens": _value(usage, "total_tokens", 0) or 0,
            },
        }
