"""Anthropic implementation of LLMClient.

Uses the official `anthropic` SDK. JSON mode is achieved via assistant prefill:
we start the assistant turn with `{` to force a JSON response, then prepend it
back to the returned text. This is more reliable than prompt instructions alone.

Default model: claude-sonnet-4-5.
"""

import logging
import os
from typing import Optional

from .base import LLMClient, LLMResponse

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-5"


class AnthropicClient(LLMClient):
    """LLMClient backed by the Anthropic Messages API."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        """
        Args:
            model: Model ID to use. Defaults to DEFAULT_MODEL.
            api_key: API key. Falls back to ANTHROPIC_API_KEY env var if not provided.
        """
        try:
            import anthropic as anthropic_sdk
        except ImportError as exc:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            ) from exc

        self.model = model or DEFAULT_MODEL
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable."
            )
        self._client = anthropic_sdk.Anthropic(api_key=key)

    def _complete_uncached(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        json_mode: bool = False,
    ) -> LLMResponse:
        messages = [{"role": "user", "content": prompt}]

        if json_mode:
            # Prefill the assistant turn with `{` to force a JSON object response.
            # Anthropic's API allows partial assistant turns that the model continues.
            messages.append({"role": "assistant", "content": "{"})

        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)

        raw_text = response.content[0].text if response.content else ""

        # Re-attach the prefilled `{` since Anthropic only returns the continuation.
        if json_mode:
            raw_text = "{" + raw_text

        usage = response.usage

        return LLMResponse(
            text=raw_text,
            model=response.model,
            provider="anthropic",
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            raw={
                "id": response.id,
                "stop_reason": response.stop_reason,
                "model": response.model,
            },
        )

    def _provider_name(self) -> str:
        return "anthropic"

    def _model_name(self) -> str:
        return self.model
