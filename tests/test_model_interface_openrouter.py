from json import JSONDecodeError
from pathlib import Path

import pytest

from agrimanager.model_interface.model_factory import create_model


class _FakeMessage:
    content = "ok"
    reasoning = "because"


class _FakeChoice:
    message = _FakeMessage()


class _FakeResponse:
    choices = [_FakeChoice()]
    id = "gen-test"
    model = "moonshotai/kimi-k2.6"
    provider = "test-provider"
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
        "cost": 0.00012,
        "completion_tokens_details": {"reasoning_tokens": 2},
        "prompt_tokens_details": {"cached_tokens": 3},
    }


class _FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _FakeResponse()


class _FailingCompletions:
    def create(self, **kwargs):
        raise RuntimeError("payment required")


class _MalformedOnceCompletions(_FakeCompletions):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise JSONDecodeError("truncated response", "", 0)
        return super().create(**kwargs)


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self):
        self.chat = _FakeChat()


def test_openrouter_factory_routes_chat_completion_params(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    config_path = tmp_path / "openrouter.yaml"
    config_path.write_text(
        "\n".join(
            [
                'model_provider: "openrouter"',
                'model_name: "moonshotai/kimi-k2.6"',
                "num_workers: 1",
                'reasoning_effort: "low"',
                "provider:",
                '  sort: "latency"',
            ]
        ),
        encoding="utf-8",
    )

    model = create_model(config_path)
    fake_client = _FakeClient()
    model.client = fake_client

    messages = [[{"role": "user", "content": "2+2?"}]]
    responses = model.generate(
        messages,
        temperature=0.2,
        max_tokens=12,
        return_metadata=True,
        extra_body={"seed": 7},
    )

    assert responses == [
        {
            "content": "ok",
            "reasoning": "because",
            "id": "gen-test",
            "model": "moonshotai/kimi-k2.6",
            "provider": "test-provider",
            "usage": _FakeResponse.usage,
        }
    ]

    request = fake_client.chat.completions.kwargs
    assert request["model"] == "moonshotai/kimi-k2.6"
    assert request["messages"] == messages[0]
    assert request["temperature"] == 0.2
    assert request["top_p"] == 1.0
    assert request["max_tokens"] == 12
    assert request["extra_body"]["seed"] == 7
    assert request["extra_body"]["provider"] == {"sort": "latency"}
    assert request["extra_body"]["reasoning"] == {"effort": "low"}

    usage = model.get_usage_report()["summary"]
    assert usage["request_count"] == 1
    assert usage["attempt_count"] == 1
    assert usage["failed_attempt_count"] == 0
    assert usage["parse_failure_count"] == 0
    assert usage["requests_with_cost"] == 1
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 4
    assert usage["reasoning_tokens"] == 2
    assert usage["cached_tokens"] == 3
    assert usage["total_tokens"] == 14
    assert usage["cost"] == 0.00012


def test_openrouter_free_eval_configs_are_wired(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    model = create_model(
        Path("agrimanager/model_interface/configs/openrouter/free.yaml")
    )

    assert model.config.model_provider == "openrouter"
    assert model.config.model_name == "openrouter/free"

    eval_config = Path("entrypoints/eval/config/openrouter.yaml").read_text(
        encoding="utf-8"
    )
    assert "configs/openrouter/free.yaml" in eval_config
    assert "max_samples: null" in eval_config

    smoke_model = create_model(
        Path(
            "agrimanager/model_interface/configs/openrouter/"
            "nemotron-3-nano-30b-a3b-free_no_reasoning.yaml"
        )
    )
    assert smoke_model.config.model_name == "nvidia/nemotron-3-nano-30b-a3b:free"
    assert smoke_model.config.reasoning == {"effort": "none", "exclude": True}


def test_openrouter_generate_raises_instead_of_returning_error_text(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    config_path = tmp_path / "openrouter.yaml"
    config_path.write_text(
        'model_provider: "openrouter"\nmodel_name: "test/model"\nnum_workers: 1\n',
        encoding="utf-8",
    )
    model = create_model(config_path)
    model.client = _FakeClient()
    model.client.chat.completions = _FailingCompletions()

    with pytest.raises(RuntimeError, match="OpenRouter failed for 1 of 1"):
        model.generate([[{"role": "user", "content": "test"}]])

    usage = model.get_usage_report()["summary"]
    assert usage["request_count"] == 0
    assert usage["attempt_count"] == 1
    assert usage["failed_attempt_count"] == 1
    assert usage["unreported_attempt_count"] == 1


def test_openrouter_retries_truncated_json_response(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    config_path = tmp_path / "openrouter.yaml"
    config_path.write_text(
        "\n".join(
            [
                'model_provider: "openrouter"',
                'model_name: "test/model"',
                "num_workers: 1",
                "response_parse_retries: 1",
                "response_parse_retry_backoff_seconds: 0",
            ]
        ),
        encoding="utf-8",
    )
    model = create_model(config_path)
    malformed_once = _MalformedOnceCompletions()
    model.client = _FakeClient()
    model.client.chat.completions = malformed_once

    responses = model.generate([[{"role": "user", "content": "test"}]])

    assert responses == ["ok"]
    assert malformed_once.calls == 2
    usage = model.get_usage_report()["summary"]
    assert usage["request_count"] == 1
    assert usage["attempt_count"] == 2
    assert usage["failed_attempt_count"] == 1
    assert usage["parse_failure_count"] == 1
    assert usage["unreported_attempt_count"] == 1
