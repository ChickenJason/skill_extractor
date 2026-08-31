"""Retrying Alibaba Cloud Model Studio client using OpenAI-compatible APIs."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence, TypeVar


logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True)
class QwenSettings:
    chat_model: str
    base_url: str
    embedding_model: str | None = None
    enable_thinking: bool = False
    structured_output: bool = True
    timeout_seconds: float = 120.0
    max_retries: int = 5
    retry_initial_seconds: float = 1.0


class QwenRequestError(RuntimeError):
    def __init__(self, message: str, *, attempts: int, retriable: bool) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.retriable = retriable


def _object_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _safe_error_text(error: BaseException, api_key: str) -> str:
    text = str(error)
    return text.replace(api_key, "<redacted>") if api_key else text


def _is_retriable(error: BaseException) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code == 429 or (isinstance(status_code, int) and status_code >= 500):
        return True
    if isinstance(status_code, int) and 400 <= status_code < 500:
        return False

    name = error.__class__.__name__
    return name in {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
        "ServiceUnavailableError",
        "TimeoutError",
        "ConnectionError",
    }


class QwenClient:
    """Small client wrapper with bounded retries and serializable responses."""

    def __init__(
        self,
        *,
        api_key: str,
        settings: QwenSettings,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be non-empty")
        if not settings.base_url:
            raise ValueError("base_url must be non-empty")
        if settings.max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        self._api_key = api_key
        self.settings = settings
        self._client = client
        self._sleep = sleep

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise RuntimeError(
                    "The online stage requires the 'openai' package. "
                    "Install environment/requirements.txt first."
                ) from error
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self.settings.base_url.rstrip("/"),
                timeout=self.settings.timeout_seconds,
                max_retries=0,
            )
        return self._client

    def _retry(self, operation: Callable[[], T], operation_name: str) -> tuple[T, int]:
        delay = self.settings.retry_initial_seconds
        for attempt in range(1, self.settings.max_retries + 1):
            try:
                return operation(), attempt
            except Exception as error:
                retriable = _is_retriable(error)
                safe_text = _safe_error_text(error, self._api_key)
                if not retriable or attempt == self.settings.max_retries:
                    raise QwenRequestError(
                        f"{operation_name} failed after {attempt} attempt(s): {safe_text}",
                        attempts=attempt,
                        retriable=retriable,
                    ) from error
                logger.warning(
                    "%s failed on attempt %s/%s: %s",
                    operation_name,
                    attempt,
                    self.settings.max_retries,
                    safe_text,
                )
                self._sleep(delay)
                delay = min(delay * 2, 16.0)
        raise AssertionError("retry loop exited unexpectedly")

    def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        model: str | None = None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": model or self.settings.chat_model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "extra_body": {"enable_thinking": self.settings.enable_thinking},
        }
        if self.settings.structured_output:
            request["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        response, attempts = self._retry(
            lambda: self.client.chat.completions.create(**request),
            "Chat completion",
        )
        latency_ms = round((time.perf_counter() - started) * 1000)

        choices = _object_value(response, "choices", [])
        if not choices:
            raise QwenRequestError(
                "Chat completion returned no choices",
                attempts=attempts,
                retriable=False,
            )
        choice = choices[0]
        message = _object_value(choice, "message")
        content = _object_value(message, "content", "") or ""
        usage = _object_value(response, "usage")
        usage_value = {
            "prompt_tokens": _object_value(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": _object_value(usage, "completion_tokens", 0) or 0,
            "total_tokens": _object_value(usage, "total_tokens", 0) or 0,
        }
        return {
            "content": content,
            "finish_reason": _object_value(choice, "finish_reason"),
            "latency_ms": latency_ms,
            "attempts": attempts,
            "response_id": _object_value(response, "id"),
            "usage": usage_value,
        }

    def embeddings(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
        dimensions: int = 1024,
        batch_size: int = 20,
    ) -> list[list[float]]:
        """Return vectors only, preserving the original public interface."""

        return self.embeddings_detailed(
            texts,
            model=model,
            dimensions=dimensions,
            batch_size=batch_size,
        )["vectors"]

    def embeddings_detailed(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
        dimensions: int = 1024,
        batch_size: int = 20,
    ) -> dict[str, Any]:
        """Return embedding vectors together with deterministic per-batch audit metadata."""

        if not texts:
            return {"vectors": [], "batches": []}
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        vectors: list[list[float]] = []
        batches: list[dict[str, Any]] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])

            def request() -> Any:
                embedding_model = model or self.settings.embedding_model
                if not embedding_model:
                    raise ValueError("An embedding model must be provided")
                return self.client.embeddings.create(
                    model=embedding_model,
                    input=batch,
                    dimensions=dimensions,
                    encoding_format="float",
                )

            started = time.perf_counter()
            response, attempts = self._retry(request, "Embedding request")
            latency_ms = round((time.perf_counter() - started) * 1000)
            data = sorted(
                _object_value(response, "data", []),
                key=lambda item: _object_value(item, "index", 0),
            )
            indexes = [_object_value(item, "index") for item in data]
            if indexes != list(range(len(batch))):
                raise ValueError(
                    "Embedding API returned missing, duplicate, or out-of-range indexes"
                )
            batch_vectors = [list(_object_value(item, "embedding", [])) for item in data]
            if len(batch_vectors) != len(batch):
                raise ValueError(
                    f"Embedding API returned {len(batch_vectors)} vectors for {len(batch)} texts"
                )
            for vector in batch_vectors:
                if len(vector) != dimensions:
                    raise ValueError(
                        f"Expected {dimensions}-dimensional embedding, got {len(vector)}"
                    )
                if not all(math.isfinite(number) for number in vector):
                        raise ValueError("Embedding API returned NaN or infinite values")
            vectors.extend(batch_vectors)
            usage = _object_value(response, "usage")
            batches.append(
                {
                    "start": start,
                    "count": len(batch),
                    "attempts": attempts,
                    "latency_ms": latency_ms,
                    "response_id": _object_value(response, "id"),
                    "model": _object_value(
                        response,
                        "model",
                        model or self.settings.embedding_model,
                    ),
                    "usage": {
                        "prompt_tokens": _object_value(usage, "prompt_tokens", 0) or 0,
                        "total_tokens": _object_value(usage, "total_tokens", 0) or 0,
                    },
                }
            )
        return {"vectors": vectors, "batches": batches}
