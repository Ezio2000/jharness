"""OpenAI Responses adapter errors."""

from __future__ import annotations

from jharness.models._json import JsonValues


class OpenAIResponsesError(ValueError):
    """The OpenAI Responses adapter could not encode or decode a request."""


OPENAI_RESPONSES_JSON = JsonValues(OpenAIResponsesError)
