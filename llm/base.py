"""Abstract LLM client interface and disk-caching wrapper.

Every provider implementation subclasses LLMClient and implements complete().
The caching decorator is applied at instantiation time in the factory so the
cache is transparent to all callers.
"""

import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    raw: dict  # full provider response for debugging


class LLMClient(ABC):
    """Abstract base for all LLM provider clients.

    Subclasses implement _complete_uncached(). Callers use complete(), which
    adds transparent disk caching on top.
    """

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Send a one-shot prompt and return a response.

        Checks the disk cache first. On a miss, calls the provider and writes
        the result to cache before returning.

        Args:
            prompt: The user-facing prompt text.
            system: Optional system prompt / instruction preamble.
            max_tokens: Hard cap on output tokens.
            temperature: Sampling temperature (0 = deterministic).
            json_mode: If True, instruct the provider to return valid JSON.
                       OpenAI uses response_format; Anthropic uses prefill.

        Returns:
            LLMResponse with text, token counts, provider metadata, and raw response.
        """
        cache_key = self._cache_key(prompt, system, max_tokens, temperature, json_mode)
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.info("[cache hit] key=%s", cache_key[:12])
            return cached

        start = time.monotonic()
        response = self._complete_uncached(
            prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
        )
        elapsed = time.monotonic() - start
        logger.debug(
            "[live call] provider=%s model=%s tokens_in=%d tokens_out=%d elapsed=%.2fs",
            response.provider,
            response.model,
            response.input_tokens,
            response.output_tokens,
            elapsed,
        )
        self._save_cache(cache_key, response)
        return response

    @abstractmethod
    def _complete_uncached(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Provider-specific implementation. Do not call directly — use complete()."""
        ...

    # ------------------------------------------------------------------ cache

    def _cache_key(
        self,
        prompt: str,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> str:
        """Stable SHA-256 key over all inputs that affect the response."""
        payload = json.dumps(
            {
                "provider": self._provider_name(),
                "model": self._model_name(),
                "system": system,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "json_mode": json_mode,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return CACHE_DIR / f"{key}.json"

    def _load_cache(self, key: str) -> Optional[LLMResponse]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return LLMResponse(**data)
        except Exception as exc:
            logger.warning("Cache read failed for %s: %s", key[:12], exc)
            return None

    def _save_cache(self, key: str, response: LLMResponse) -> None:
        path = self._cache_path(key)
        try:
            path.write_text(json.dumps(asdict(response), indent=2))
        except Exception as exc:
            logger.warning("Cache write failed for %s: %s", key[:12], exc)

    # ------------------------------------------------------------------ meta (override in subclasses)

    def _provider_name(self) -> str:
        return self.__class__.__name__

    def _model_name(self) -> str:
        return getattr(self, "model", "unknown")
