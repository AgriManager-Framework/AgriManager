"""
OpenRouter provider model implementation.
"""
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from json import JSONDecodeError
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

from ..base.model import BaseModel
from .model_config import OpenRouterConfig


class OpenRouterModel(BaseModel):
    """OpenRouter model provider implementation."""

    def __init__(self, config: OpenRouterConfig):
        """
        Initialize OpenRouter model.

        Args:
            config: OpenRouter configuration instance
        """
        super().__init__(config)
        self.config: OpenRouterConfig = config

        api_key = config.api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Set it in config or OPENROUTER_API_KEY environment variable."
            )

        self.default_headers = self._build_default_headers(config)

        client_kwargs = {
            "api_key": api_key,
            "base_url": config.base_url,
            "timeout": config.timeout,
        }
        if self.default_headers:
            client_kwargs["default_headers"] = self.default_headers
        self.client = OpenAI(**client_kwargs)
        self._usage_lock = threading.Lock()
        self._usage_records: List[Dict[str, Any]] = []
        self._attempt_count = 0
        self._failed_attempt_count = 0
        self._parse_failure_count = 0

    @staticmethod
    def _build_default_headers(config: OpenRouterConfig) -> Dict[str, str]:
        headers = {}

        site_url = (
            config.site_url
            or os.getenv("OPENROUTER_HTTP_REFERER")
            or os.getenv("OPENROUTER_SITE_URL")
        )
        if site_url:
            headers["HTTP-Referer"] = site_url

        app_title = config.app_title or os.getenv("OPENROUTER_APP_TITLE")
        if app_title:
            headers["X-OpenRouter-Title"] = app_title

        headers.update(config.extra_headers or {})
        return headers

    @staticmethod
    def _read_extra_field(obj: Any, name: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(name)

        value = getattr(obj, name, None)
        if value is not None:
            return value

        model_extra = getattr(obj, "model_extra", None)
        if isinstance(model_extra, dict):
            return model_extra.get(name)

        return None

    @classmethod
    def _to_plain_data(cls, value: Any) -> Any:
        """Convert SDK response objects into JSON-safe Python containers."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): cls._to_plain_data(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_plain_data(item) for item in value]

        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return cls._to_plain_data(model_dump(exclude_none=True))
        return str(value)

    @staticmethod
    def _numeric(value: Any, cast, default=0):
        try:
            return cast(value)
        except (TypeError, ValueError):
            return default

    def _record_response_usage(self, response: Any) -> None:
        usage = self._to_plain_data(self._read_extra_field(response, "usage"))
        if not isinstance(usage, dict):
            usage = {}

        record = {
            "generation_id": self._read_extra_field(response, "id"),
            "model": self._read_extra_field(response, "model") or self.config.model_name,
            "provider": self._read_extra_field(response, "provider"),
            "usage": usage,
        }
        with self._usage_lock:
            self._usage_records.append(self._to_plain_data(record))

    def reset_usage(self) -> None:
        """Reset accumulated per-response OpenRouter usage accounting."""
        with self._usage_lock:
            self._usage_records = []
            self._attempt_count = 0
            self._failed_attempt_count = 0
            self._parse_failure_count = 0

    def get_usage_report(self) -> Dict[str, Any]:
        """Return OpenRouter-reported usage plus local request-attempt counts."""
        with self._usage_lock:
            records = [dict(record) for record in self._usage_records]
            attempt_count = self._attempt_count
            failed_attempt_count = self._failed_attempt_count
            parse_failure_count = self._parse_failure_count

        summary = {
            "request_count": len(records),
            "attempt_count": attempt_count,
            "failed_attempt_count": failed_attempt_count,
            "parse_failure_count": parse_failure_count,
            "requests_with_cost": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
            "per_model": {},
        }
        for record in records:
            usage = record.get("usage") or {}
            completion_details = usage.get("completion_tokens_details") or {}
            prompt_details = usage.get("prompt_tokens_details") or {}
            values = {
                "prompt_tokens": self._numeric(usage.get("prompt_tokens"), int),
                "completion_tokens": self._numeric(
                    usage.get("completion_tokens"), int
                ),
                "reasoning_tokens": self._numeric(
                    completion_details.get("reasoning_tokens"), int
                ),
                "cached_tokens": self._numeric(
                    prompt_details.get("cached_tokens"), int
                ),
                "total_tokens": self._numeric(usage.get("total_tokens"), int),
                "cost": self._numeric(usage.get("cost"), float, 0.0),
            }
            if usage.get("cost") is not None:
                summary["requests_with_cost"] += 1
            for key, value in values.items():
                summary[key] += value

            model_name = str(record.get("model") or self.config.model_name)
            model_summary = summary["per_model"].setdefault(
                model_name,
                {
                    "request_count": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "reasoning_tokens": 0,
                    "total_tokens": 0,
                    "cost": 0.0,
                },
            )
            model_summary["request_count"] += 1
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "reasoning_tokens",
                "total_tokens",
                "cost",
            ):
                model_summary[key] += values[key]

        summary["missing_cost_requests"] = (
            summary["request_count"] - summary["requests_with_cost"]
        )
        summary["unreported_attempt_count"] = max(
            0, summary["attempt_count"] - summary["request_count"]
        )
        return {"summary": summary, "requests": records}

    def _record_attempt(self) -> None:
        with self._usage_lock:
            self._attempt_count += 1

    def _record_attempt_failure(self, *, parse_failure: bool = False) -> None:
        with self._usage_lock:
            self._failed_attempt_count += 1
            if parse_failure:
                self._parse_failure_count += 1

    @staticmethod
    def _extract_tagged_reasoning(content: Optional[str]) -> tuple[Optional[str], str]:
        if not content:
            return None, content or ""

        for tag in ("think", "analysis"):
            match = re.search(
                rf"<{tag}>\s*(.*?)\s*</{tag}>",
                content,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not match:
                continue
            reasoning = match.group(1).strip() or None
            clean_content = re.sub(
                rf"<{tag}>\s*.*?\s*</{tag}>",
                "",
                content,
                count=1,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()
            return reasoning, clean_content

        return None, content

    @staticmethod
    def _stringify_reasoning_details(reasoning_details: Any) -> Optional[str]:
        if reasoning_details is None:
            return None
        if isinstance(reasoning_details, str):
            return reasoning_details
        if isinstance(reasoning_details, list):
            return "\n".join(str(item) for item in reasoning_details)
        return str(reasoning_details)

    def _build_extra_body(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        extra_body = {}
        configured_params = dict(self.config.extra_params or {})

        if self.config.provider is not None:
            configured_params["provider"] = self.config.provider
        if self.config.models is not None:
            configured_params["models"] = self.config.models
        if self.config.route is not None:
            configured_params["route"] = self.config.route
        if self.config.include_reasoning is not None:
            configured_params["include_reasoning"] = self.config.include_reasoning
        if self.config.reasoning is not None:
            configured_params["reasoning"] = self.config.reasoning
        elif self.config.reasoning_effort:
            configured_params["reasoning"] = {"effort": self.config.reasoning_effort}

        explicit_extra_body = kwargs.pop("extra_body", None)
        if explicit_extra_body:
            extra_body.update(explicit_extra_body)

        extra_body.update(configured_params)
        extra_body.update(kwargs)
        return extra_body

    def _call_openrouter_api(self, args):
        """
        Helper function for multithreading API calls to OpenRouter.

        Args:
            args: Tuple of (messages, config_dict)

        Returns:
            Generated response string or metadata dict.
        """
        messages, config_dict = args
        extra_kwargs = dict(config_dict.get("extra_kwargs", {}))

        api_params = {
            "model": config_dict["model_name"],
            "messages": messages,
            "temperature": config_dict["temperature"],
            "top_p": config_dict["top_p"],
            "max_tokens": config_dict["max_tokens"],
        }

        extra_headers = extra_kwargs.pop("extra_headers", None)
        if extra_headers:
            api_params["extra_headers"] = extra_headers

        extra_query = extra_kwargs.pop("extra_query", None)
        if extra_query:
            api_params["extra_query"] = extra_query

        request_timeout = extra_kwargs.pop("timeout", None)
        if request_timeout is not None:
            api_params["timeout"] = request_timeout

        extra_body = self._build_extra_body(extra_kwargs)
        if extra_body:
            api_params["extra_body"] = extra_body

        parse_retries = max(0, int(self.config.response_parse_retries))
        for attempt in range(parse_retries + 1):
            self._record_attempt()
            try:
                response = self.client.chat.completions.create(**api_params)
                break
            except JSONDecodeError:
                self._record_attempt_failure(parse_failure=True)
                if attempt >= parse_retries:
                    raise
                backoff = max(
                    0.0,
                    float(self.config.response_parse_retry_backoff_seconds),
                ) * (2**attempt)
                print(
                    "OpenRouter returned malformed JSON; "
                    f"retrying response parse request ({attempt + 1}/{parse_retries}) "
                    f"after {backoff:.1f}s"
                )
                if backoff:
                    time.sleep(backoff)
            except Exception:
                self._record_attempt_failure()
                raise
        self._record_response_usage(response)

        message = response.choices[0].message
        content = self._read_extra_field(message, "content") or ""
        if not config_dict.get("return_metadata"):
            return content

        reasoning = self._read_extra_field(message, "reasoning")
        if reasoning is None:
            reasoning = self._read_extra_field(message, "reasoning_content")
        if reasoning is None:
            reasoning = self._stringify_reasoning_details(
                self._read_extra_field(message, "reasoning_details")
            )
        if reasoning is None:
            reasoning, content = self._extract_tagged_reasoning(content)

        metadata = {"content": content, "reasoning": reasoning}
        for field in ("id", "model", "provider"):
            value = self._read_extra_field(response, field)
            if value is not None:
                metadata[field] = self._to_plain_data(value)
        usage = self._read_extra_field(response, "usage")
        if usage is not None:
            metadata["usage"] = self._to_plain_data(usage)
        return metadata

    def generate(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 4096,
        return_metadata: bool = False,
        **kwargs
    ) -> List[str]:
        """
        Generate responses for a batch of message lists using multithreading.

        Args:
            messages_list: List of message lists in OpenAI format
            temperature: Temperature parameter (default: 1.0)
            top_p: Top-p sampling parameter (default: 1.0)
            max_tokens: Maximum tokens to generate (default: 4096)
            return_metadata: If True, return dicts with content/reasoning
            **kwargs: Additional OpenRouter request body parameters

        Returns:
            List of generated response strings or metadata dicts
        """
        if not messages_list:
            return []

        return_metadata = kwargs.pop("return_metadata", return_metadata)
        if not return_metadata and (
            self.config.reasoning is not None
            or self.config.reasoning_effort
            or self.config.include_reasoning
        ):
            return_metadata = True

        config_dict = {
            "model_name": self.config.model_name,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "extra_kwargs": kwargs,
            "return_metadata": return_metadata,
        }

        args_list = [(messages, config_dict) for messages in messages_list]

        results = [None] * len(messages_list)
        errors = []
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            future_to_idx = {
                executor.submit(self._call_openrouter_api, args): idx
                for idx, args in enumerate(args_list)
            }

            for future in tqdm(
                as_completed(future_to_idx),
                total=len(messages_list),
                desc="Generating",
            ):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Error processing request {idx}: {str(e)}")
                    errors.append((idx, e))

        if errors:
            first_idx, first_error = errors[0]
            raise RuntimeError(
                f"OpenRouter failed for {len(errors)} of {len(messages_list)} "
                f"requests; first failure at index {first_idx}: {first_error}"
            ) from first_error

        return results

    def convert_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Convert OpenAI format messages to OpenRouter format.

        OpenRouter chat completions are OpenAI-compatible, so no conversion is
        required for the current AgriManager message format.
        """
        return messages
