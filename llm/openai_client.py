"""OpenAI implementation of LLMClient.

Uses the official `openai` SDK. JSON mode is handled via response_format.
Default model: gpt-4o-mini (cheap, fast, sufficient for classification + generation).
"""

import logging
import os
from typing import Optional

from .base import LLMClient, LLMResponse

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIClient(LLMClient):
    """LLMClient backed by OpenAI's chat completions API."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        """
        Args:
            model: Model ID to use. Defaults to DEFAULT_MODEL.
            api_key: API key. Falls back to OPENAI_API_KEY env var if not provided.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            ) from exc

        self.model = model or DEFAULT_MODEL
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
            )
        self._client = OpenAI(api_key=key)

    def _complete_uncached(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        json_mode: bool = False,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            # Native JSON mode — OpenAI guarantees valid JSON output.
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        text = choice.message.content or ""
        usage = response.usage

        return LLMResponse(
            text=text,
            model=response.model,
            provider="openai",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            raw={
                "id": response.id,
                "finish_reason": choice.finish_reason,
                "model": response.model,
            },
        )

    def _provider_name(self) -> str:
        return "openai"

    def _model_name(self) -> str:
        return self.model
