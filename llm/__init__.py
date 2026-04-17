"""LLM provider abstraction for Silent Churn Detection System.

All LLM calls go through this package. Swap providers by setting LLM_PROVIDER env var.
"""

from .base import LLMClient, LLMResponse
from .factory import get_client

__all__ = ["LLMClient", "LLMResponse", "get_client"]
