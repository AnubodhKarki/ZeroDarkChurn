"""LLM client factory.

Reads LLM_PROVIDER env var and returns the appropriate LLMClient subclass.
Supported values: "openai", "anthropic".

Optional LLM_MODEL env var overrides the provider's default model.

Usage:
    from llm import get_client
    client = get_client()
    response = client.complete("Summarize this text...", json_mode=True)
"""

import logging
import os
from typing import Optional

from .base import LLMClient

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = ("openai", "anthropic")


def get_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> LLMClient:
    """Build and return the correct LLMClient for the configured provider.

    Args:
        provider: Provider name. Overrides LLM_PROVIDER env var if given.
        model: Model ID. Overrides LLM_MODEL env var if given.

    Returns:
        A concrete LLMClient ready to call.

    Raises:
        ValueError: If the provider is missing or not recognised.
    """
    resolved_provider = (provider or os.environ.get("LLM_PROVIDER", "")).strip().lower()
    resolved_model = model or os.environ.get("LLM_MODEL") or None  # None → use provider default

    if not resolved_provider:
        raise ValueError(
            "LLM provider not configured. "
            "Set the LLM_PROVIDER environment variable to one of: "
            + ", ".join(SUPPORTED_PROVIDERS)
        )

    if resolved_provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unrecognised LLM provider: '{resolved_provider}'. "
            "Supported providers: " + ", ".join(SUPPORTED_PROVIDERS)
        )

    logger.info("Using LLM provider: %s model: %s", resolved_provider, resolved_model or "(default)")

    if resolved_provider == "openai":
        from .openai_client import OpenAIClient
        return OpenAIClient(model=resolved_model)

    if resolved_provider == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(model=resolved_model)

    # Unreachable given the check above, but makes type checkers happy.
    raise ValueError(f"Provider '{resolved_provider}' not implemented.")
