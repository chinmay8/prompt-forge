"""OpenAI-compatible Chat Completions client configured via .env."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Literal, overload
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
)

from core.config import (
    DEFAULT_API_BASE_URL,
    DEFAULT_API_KEY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_REQUEST_TIMEOUT,
    LOCAL_NUM_CTX,
    MAX_LOCAL_NUM_CTX,
    PROJECT_ROOT,
)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_CODE_FENCE_RE = re.compile(
    r"```(?:json)?\s*(.*?)\s*```",
    re.IGNORECASE | re.DOTALL,
)


class OpenAIClientError(RuntimeError):
    """Raised when the OpenAI-compatible API fails, is misconfigured, or returns invalid JSON."""


def _is_local_endpoint(url: str) -> bool:
    """Return True when base_url points to localhost/loopback (key optional)."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def extract_json_payload(raw_text: str) -> dict[str, Any]:
    """Parse JSON safely from LLM output, handling think tags, fences, and extra text."""
    if not raw_text or not raw_text.strip():
        raise OpenAIClientError("Model returned an empty response when JSON was expected.")

    cleaned = _THINK_BLOCK_RE.sub("", raw_text).strip()

    # 1. Direct parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2. Extract from fenced code blocks (```json ... ``` or ``` ... ```)
    for match in _CODE_FENCE_RE.finditer(cleaned):
        candidate = match.group(1).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    # 3. Scan from any '{' using JSONDecoder.raw_decode to ignore surrounding prose
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(cleaned):
        if ch != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[idx:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    preview = cleaned[:280].replace("\n", " ")
    raise OpenAIClientError(f"Failed to parse valid JSON object from model output: {preview!r}")


class OpenAIClient:
    """Chat Completions client reading base_url and api_key from .env by default."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        # Refresh .env so updating URL/KEY in .env takes effect cleanly
        load_dotenv(PROJECT_ROOT / ".env", override=False)

        resolved_url = (
            base_url
            or os.environ.get("API_BASE_URL")
            or DEFAULT_API_BASE_URL
        ).strip()
        if api_key is not None:
            resolved_key = api_key.strip()
        elif "API_KEY" in os.environ:
            resolved_key = os.environ["API_KEY"].strip()
        else:
            resolved_key = DEFAULT_API_KEY

        self.base_url: str = resolved_url
        self.api_key: str = resolved_key
        self.timeout: float = timeout
        self.last_error: str | None = None

        # OpenAI SDK requires a non-empty api_key string even for local endpoints
        effective_key = self.api_key or ("local-no-key" if _is_local_endpoint(self.base_url) else "missing-api-key")
        self._client: OpenAI = OpenAI(
            base_url=self.base_url,
            api_key=effective_key,
            timeout=self.timeout,
        )

    @property
    def is_local_endpoint(self) -> bool:
        """Return True when the configured base_url targets localhost/loopback."""
        return _is_local_endpoint(self.base_url)

    @property
    def has_api_key(self) -> bool:
        """Return True if an API key is configured or endpoint is local."""
        return bool(self.api_key) or self.is_local_endpoint

    @property
    def default_model(self) -> str:
        """Return active DEFAULT_MODEL from environment or config fallback."""
        return (os.environ.get("DEFAULT_MODEL") or DEFAULT_MODEL).strip()

    def check_connection(self) -> bool:
        """Verify credentials and reachability of the configured API endpoint."""
        if not self.has_api_key:
            self.last_error = "Missing API key. Set API_KEY in .env."
            return False

        try:
            self._client.models.list()
            self.last_error = None
            return True
        except NotFoundError:
            # Some OpenAI-compatible gateways do not expose /v1/models
            self.last_error = None
            return True
        except APIStatusError as exc:
            if exc.status_code in {404, 405, 501}:
                self.last_error = None
                return True
            self.last_error = f"API error ({exc.status_code}): {exc.message}"
            return False
        except (AuthenticationError, APIConnectionError, APITimeoutError) as exc:
            self.last_error = str(exc)
            return False
        except Exception as exc:
            self.last_error = str(exc)
            return False

    def get_available_models(self) -> list[str]:
        """Return model IDs from /v1/models, falling back to DEFAULT_MODEL if unsupported."""
        if not self.has_api_key:
            return []

        try:
            response = self._client.models.list()
            self.last_error = None
        except NotFoundError:
            self.last_error = None
            return [self.default_model] if self.default_model else []
        except APIStatusError as exc:
            if exc.status_code in {404, 405, 501}:
                self.last_error = None
                return [self.default_model] if self.default_model else []
            self.last_error = f"API error ({exc.status_code}): {exc.message}"
            return []
        except Exception as exc:
            self.last_error = str(exc)
            return []

        raw_items = getattr(response, "data", None)
        if raw_items is None and isinstance(response, dict):
            raw_items = response.get("data", [])

        models: list[str] = []
        seen: set[str] = set()
        for item in raw_items or []:
            model_id = (
                item.get("id")
                if isinstance(item, dict)
                else getattr(item, "id", None)
            )
            if model_id:
                tag = str(model_id).strip()
                if tag and tag not in seen:
                    seen.add(tag)
                    models.append(tag)

        if not models and not self.is_local_endpoint and self.default_model:
            return [self.default_model]
        return models

    @overload
    def chat_completion(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        json_mode: Literal[False] = False,
    ) -> str: ...

    @overload
    def chat_completion(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        json_mode: Literal[True] = True,
    ) -> dict[str, Any]: ...

    def chat_completion(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> dict[str, Any] | str:
        """Execute a chat completion via OpenAI Chat Completions API."""
        if not model or not model.strip():
            raise OpenAIClientError("No model ID specified.")
        if not self.has_api_key:
            raise OpenAIClientError("Missing API key. Set API_KEY in .env.")

        kwargs: dict[str, Any] = {
            "model": model.strip(),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": float(temperature),
            "max_tokens": DEFAULT_MAX_TOKENS,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.is_local_endpoint:
            # Cap KV cache on local Ollama /v1 endpoints at <=2048 to avoid RAM exhaustion
            raw_ctx = os.environ.get("LOCAL_NUM_CTX")
            requested_ctx = int(raw_ctx) if raw_ctx else LOCAL_NUM_CTX
            capped_ctx = min(max(1, requested_ctx), MAX_LOCAL_NUM_CTX)
            kwargs["extra_body"] = {"options": {"num_ctx": capped_ctx}}

        text = self._request_text(model=model, kwargs=kwargs)
        if not json_mode:
            return _THINK_BLOCK_RE.sub("", text).strip()

        try:
            return extract_json_payload(text)
        except OpenAIClientError:
            # Small 1B-1.5B models occasionally emit malformed JSON; attempt 1 repair pass
            repair_kwargs = dict(kwargs)
            repair_kwargs["temperature"] = 0.0
            repair_kwargs["messages"] = [
                *kwargs["messages"],
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was not valid JSON. "
                        "Reply ONLY with a single valid JSON object matching the requested schema."
                    ),
                },
            ]
            repaired_text = self._request_text(model=model, kwargs=repair_kwargs)
            return extract_json_payload(repaired_text)

    def _request_text(self, model: str, kwargs: dict[str, Any]) -> str:
        """Execute chat completion request and extract non-empty text content."""
        try:
            response = self._invoke_chat(kwargs)
        except Exception as exc:
            self.last_error = str(exc)
            raise OpenAIClientError(
                f"Chat completion failed for model '{model}': {exc}"
            ) from exc

        choices = getattr(response, "choices", None)
        if not choices:
            raise OpenAIClientError("API returned a response with no choices.")

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message is not None else None
        if content is None:
            raise OpenAIClientError("API returned a response without message content.")

        return str(content).strip()

    def _invoke_chat(self, kwargs: dict[str, Any]) -> Any:
        """Call chat.completions.create, gracefully dropping unsupported optional params."""
        current = dict(kwargs)
        for _ in range(3):
            try:
                return self._client.chat.completions.create(**current)
            except BadRequestError as exc:
                err_lower = str(exc).lower()
                changed = False
                if "response_format" in current and (
                    "response_format" in err_lower or "json" in err_lower
                ):
                    current.pop("response_format", None)
                    changed = True
                if "temperature" in current and "temperature" in err_lower:
                    current.pop("temperature", None)
                    changed = True
                if "max_tokens" in current and "max_tokens" in err_lower:
                    current.pop("max_tokens", None)
                    changed = True
                if not changed and "extra_body" in current:
                    current.pop("extra_body", None)
                    changed = True
                if not changed:
                    raise
        return self._client.chat.completions.create(**current)
