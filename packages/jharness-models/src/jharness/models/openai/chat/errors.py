"""OpenAI Chat adapter errors."""

from __future__ import annotations

from jharness.models._json import JsonValues


class OpenAIChatError(ValueError):
    """The OpenAI Chat adapter could not encode or decode a request."""


OPENAI_CHAT_JSON = JsonValues(OpenAIChatError)
