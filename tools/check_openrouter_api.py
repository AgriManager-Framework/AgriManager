#!/usr/bin/env python3
"""Safely verify an OpenRouter API key and optionally run a free inference."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


BASE_URL = "https://openrouter.ai/api/v1"


def request_json(
    path: str,
    api_key: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        f"{BASE_URL}{path}", headers=headers, data=data
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        message = f"HTTP {exc.code}"
        try:
            error_body = json.load(exc)
            detail = error_body.get("error", {}).get("message")
            if detail:
                message += f": {detail}"
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        raise RuntimeError(message) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error: {exc.reason}") from exc


def verify_key(api_key: str, timeout: float) -> None:
    result = request_json("/key", api_key, timeout=timeout)
    key_info = result.get("data")
    if not isinstance(key_info, dict):
        raise RuntimeError("OpenRouter returned an unexpected key response")

    print("Authentication: OK")
    print(f"Free tier: {bool(key_info.get('is_free_tier'))}")
    if key_info.get("limit_remaining") is not None:
        print(f"Credit limit remaining: {key_info['limit_remaining']}")


def run_chat_check(api_key: str, timeout: float) -> None:
    result = request_json(
        "/chat/completions",
        api_key,
        timeout=timeout,
        payload={
            "model": "openrouter/free",
            "messages": [
                {"role": "user", "content": "Reply with exactly: OK"},
            ],
            "max_tokens": 16,
            "temperature": 0,
        },
    )
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("OpenRouter returned no completion choices")

    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise RuntimeError("OpenRouter returned an unexpected chat response")

    print("Inference: OK")
    print(f"Model used: {result.get('model', 'unknown')}")
    print(f"Response: {content.strip()!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify OPENROUTER_API_KEY without printing the key."
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="also make a minimal request through the free-model router",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds (default: 30)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print(
            "OPENROUTER_API_KEY is not set. Run `source ~/.bashrc` first.",
            file=sys.stderr,
        )
        return 2

    try:
        verify_key(api_key, args.timeout)
        if args.chat:
            run_chat_check(api_key, args.timeout)
    except RuntimeError as exc:
        # Never include the API key in diagnostic output.
        safe_message = str(exc).replace(api_key, "[REDACTED]")
        print(f"OpenRouter test failed: {safe_message}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
