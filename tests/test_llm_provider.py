"""Verify LLM client construction for openrouter / openai / custom providers.

These tests do not hit the network. They check that resolve_llm_settings produces
the correct base_url, api_key_env, and headers; and that create_llm_client
constructs the OpenAI client with the matching constructor arguments.
"""

from dataclasses import asdict

import pytest

from autoconcourse.config import LLMConfig, GenerationConfig, PipelineConfig, TopicConfig
from autoconcourse.generation import (
    PROVIDER_DEFAULTS,
    create_llm_client,
    resolve_llm_settings,
)


def _make_config(llm: LLMConfig) -> PipelineConfig:
    """Build a minimal PipelineConfig wrapping the given LLMConfig."""
    return PipelineConfig(
        topic=TopicConfig(name="t", id="t", languages=["en"]),
        llm=llm,
        corpus=None,                # unused by LLM resolution
        anchors=None,
        embedding=None,
        retrieval=None,
        groups=None,
        generation=GenerationConfig(),
    )


def test_openrouter_default_resolution():
    cfg = _make_config(LLMConfig(provider="openrouter"))
    s = resolve_llm_settings(cfg)
    assert s["base_url"] == "https://openrouter.ai/api/v1"
    assert s["api_key_env"] == "OPENROUTER_API_KEY"
    assert s["headers"] == {}


def test_openrouter_with_ranking_headers():
    cfg = _make_config(LLMConfig(
        provider="openrouter",
        site_url="https://github.com/example/autoconcourse",
        site_name="AutoConcourse",
    ))
    s = resolve_llm_settings(cfg)
    assert s["headers"]["HTTP-Referer"] == "https://github.com/example/autoconcourse"
    assert s["headers"]["X-Title"] == "AutoConcourse"


def test_openrouter_user_headers_not_overwritten():
    cfg = _make_config(LLMConfig(
        provider="openrouter",
        site_name="AutoConcourse",
        extra_headers={"X-Title": "Explicit override"},
    ))
    s = resolve_llm_settings(cfg)
    # extra_headers wins over the site_name shorthand
    assert s["headers"]["X-Title"] == "Explicit override"


def test_openai_default_resolution():
    cfg = _make_config(LLMConfig(provider="openai"))
    s = resolve_llm_settings(cfg)
    assert s["base_url"] is None      # SDK default (api.openai.com)
    assert s["api_key_env"] == "OPENAI_API_KEY"
    # Ranking headers are openrouter-only
    assert s["headers"] == {}


def test_openai_does_not_attach_ranking_headers():
    cfg = _make_config(LLMConfig(
        provider="openai",
        site_url="x",
        site_name="y",
    ))
    s = resolve_llm_settings(cfg)
    assert "HTTP-Referer" not in s["headers"]
    assert "X-Title" not in s["headers"]


def test_custom_provider_requires_base_url_and_api_key_env():
    cfg = _make_config(LLMConfig(provider="custom"))
    with pytest.raises(ValueError, match="base_url"):
        resolve_llm_settings(cfg)

    cfg = _make_config(LLMConfig(provider="custom", base_url="http://localhost:8000/v1"))
    with pytest.raises(ValueError, match="api_key_env"):
        resolve_llm_settings(cfg)


def test_custom_provider_passthrough():
    cfg = _make_config(LLMConfig(
        provider="custom",
        base_url="http://localhost:8000/v1",
        api_key_env="LOCAL_LLM_KEY",
        extra_headers={"X-Custom": "value"},
    ))
    s = resolve_llm_settings(cfg)
    assert s["base_url"] == "http://localhost:8000/v1"
    assert s["api_key_env"] == "LOCAL_LLM_KEY"
    assert s["headers"] == {"X-Custom": "value"}


def test_unknown_provider_errors():
    cfg = _make_config(LLMConfig(provider="bogus"))
    with pytest.raises(ValueError, match="Unknown llm.provider"):
        resolve_llm_settings(cfg)


def test_explicit_base_url_overrides_provider_default():
    cfg = _make_config(LLMConfig(provider="openrouter", base_url="https://proxy.example/v1"))
    s = resolve_llm_settings(cfg)
    assert s["base_url"] == "https://proxy.example/v1"


def test_create_llm_client_constructs_openai_with_correct_args(monkeypatch):
    """Mirror the user's JS snippet: client is built with base_url and headers."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")

    captured = {}

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None, default_headers=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["default_headers"] = default_headers

    monkeypatch.setattr("autoconcourse.generation.OpenAI", FakeOpenAI)

    cfg = _make_config(LLMConfig(
        provider="openrouter",
        site_url="https://autoconcourse.org",
        site_name="AutoConcourse",
    ))
    create_llm_client(cfg)

    assert captured["api_key"] == "sk-test-fake"
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["default_headers"]["HTTP-Referer"] == "https://autoconcourse.org"
    assert captured["default_headers"]["X-Title"] == "AutoConcourse"


def test_create_llm_client_missing_key_raises(monkeypatch):
    # create_llm_client calls load_dotenv() which would repopulate keys from .env
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = _make_config(LLMConfig(provider="openrouter"))
    with pytest.raises(ValueError, match="No API key found in env var"):
        create_llm_client(cfg)
