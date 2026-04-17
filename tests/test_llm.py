"""Tests for the LLM provider abstraction layer.

Three test groups:
1. Factory — correct client returned per LLM_PROVIDER value, error on bad/missing value.
2. Cache — hit returns same response without calling provider (provider is mocked).
3. JSON mode — response is parseable JSON (live API call, skipped unless env vars set).
"""

import json
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_response(text: str = '{"ok": true}', provider: str = "openai") -> "LLMResponse":
    """Build a fake LLMResponse for testing."""
    from llm.base import LLMResponse
    return LLMResponse(
        text=text,
        model="test-model",
        provider=provider,
        input_tokens=10,
        output_tokens=5,
        raw={"test": True},
    )


# ---------------------------------------------------------------------------
# 1. Factory tests
# ---------------------------------------------------------------------------

class TestFactory:
    def test_returns_openai_client(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from llm.factory import get_client
        from llm.openai_client import OpenAIClient
        client = get_client()
        assert isinstance(client, OpenAIClient)

    def test_returns_anthropic_client(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        from llm.factory import get_client
        from llm.anthropic_client import AnthropicClient
        client = get_client()
        assert isinstance(client, AnthropicClient)

    def test_provider_override_arg(self, monkeypatch):
        """Explicit provider arg overrides LLM_PROVIDER env var."""
        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from llm.factory import get_client
        from llm.openai_client import OpenAIClient
        client = get_client(provider="openai")
        assert isinstance(client, OpenAIClient)

    def test_missing_provider_raises(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        from llm.factory import get_client
        with pytest.raises(ValueError, match="LLM_PROVIDER"):
            get_client()

    def test_unknown_provider_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "cohere")
        from llm.factory import get_client
        with pytest.raises(ValueError, match="Unrecognised"):
            get_client()

    def test_model_override_env(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from llm.factory import get_client
        client = get_client()
        assert client.model == "gpt-4o"

    def test_model_override_arg(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from llm.factory import get_client
        client = get_client(model="gpt-4-turbo")
        assert client.model == "gpt-4-turbo"


# ---------------------------------------------------------------------------
# 2. Cache tests
# ---------------------------------------------------------------------------

class TestCache:
    """Cache is tested on a concrete client to avoid testing the abstract base."""

    @pytest.fixture()
    def temp_cache(self, monkeypatch, tmp_path):
        """Redirect CACHE_DIR to a temporary directory for isolation."""
        import llm.base as base_module
        monkeypatch.setattr(base_module, "CACHE_DIR", tmp_path / "cache")
        return tmp_path / "cache"

    @pytest.fixture()
    def openai_client(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from llm.openai_client import OpenAIClient
        client = OpenAIClient(model="gpt-4o-mini")
        return client

    def test_cache_miss_calls_provider(self, openai_client, temp_cache, monkeypatch):
        """On first call the provider is invoked and response is cached."""
        mock_response = _make_mock_response()
        monkeypatch.setattr(openai_client, "_complete_uncached", MagicMock(return_value=mock_response))

        result = openai_client.complete("Hello")

        openai_client._complete_uncached.assert_called_once()
        assert result.text == mock_response.text
        # Cache file should now exist
        assert any(temp_cache.glob("*.json"))

    def test_cache_hit_skips_provider(self, openai_client, temp_cache, monkeypatch):
        """Second call with same prompt returns cached response without calling provider."""
        mock_response = _make_mock_response()
        mock_uncached = MagicMock(return_value=mock_response)
        monkeypatch.setattr(openai_client, "_complete_uncached", mock_uncached)

        # First call — populates cache
        openai_client.complete("Hello")
        # Second call — should hit cache
        result = openai_client.complete("Hello")

        assert mock_uncached.call_count == 1  # provider called only once
        assert result.text == mock_response.text

    def test_different_prompts_different_cache_entries(self, openai_client, temp_cache, monkeypatch):
        """Different prompts each generate their own cache entry."""
        monkeypatch.setattr(
            openai_client,
            "_complete_uncached",
            MagicMock(side_effect=[_make_mock_response("a"), _make_mock_response("b")]),
        )

        r1 = openai_client.complete("Prompt A")
        r2 = openai_client.complete("Prompt B")

        assert r1.text == "a"
        assert r2.text == "b"
        assert len(list(temp_cache.glob("*.json"))) == 2

    def test_json_mode_affects_cache_key(self, openai_client, temp_cache, monkeypatch):
        """json_mode=True and json_mode=False are cached separately."""
        monkeypatch.setattr(
            openai_client,
            "_complete_uncached",
            MagicMock(side_effect=[_make_mock_response("plain"), _make_mock_response('{"x":1}')]),
        )

        openai_client.complete("Same prompt", json_mode=False)
        openai_client.complete("Same prompt", json_mode=True)

        assert len(list(temp_cache.glob("*.json"))) == 2

    def test_system_prompt_affects_cache_key(self, openai_client, temp_cache, monkeypatch):
        """Different system prompts produce different cache keys."""
        monkeypatch.setattr(
            openai_client,
            "_complete_uncached",
            MagicMock(side_effect=[_make_mock_response("r1"), _make_mock_response("r2")]),
        )

        openai_client.complete("prompt", system="You are A")
        openai_client.complete("prompt", system="You are B")

        assert len(list(temp_cache.glob("*.json"))) == 2


# ---------------------------------------------------------------------------
# 3. Live JSON mode tests (skipped unless real API keys are present)
# ---------------------------------------------------------------------------

SKIP_LIVE = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "").startswith("sk-test"),
    reason="OPENAI_API_KEY not set or is a test key — skipping live API call",
)

SKIP_LIVE_ANTHROPIC = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "").startswith("sk-ant-test"),
    reason="ANTHROPIC_API_KEY not set or is a test key — skipping live API call",
)


class TestLiveJsonMode:
    @SKIP_LIVE
    def test_openai_json_mode_parseable(self):
        from llm.openai_client import OpenAIClient
        client = OpenAIClient()
        response = client.complete(
            'Return a JSON object with a single key "status" set to "ok".',
            json_mode=True,
        )
        parsed = json.loads(response.text)
        assert "status" in parsed

    @SKIP_LIVE_ANTHROPIC
    def test_anthropic_json_mode_parseable(self):
        from llm.anthropic_client import AnthropicClient
        client = AnthropicClient()
        response = client.complete(
            'Return a JSON object with a single key "status" set to "ok".',
            json_mode=True,
        )
        parsed = json.loads(response.text)
        assert "status" in parsed
