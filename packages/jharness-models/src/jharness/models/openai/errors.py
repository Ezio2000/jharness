"""OpenAI protocol adapter errors."""

from __future__ import annotations

from jharness.models._json import JsonValues


class OpenAIChatCompletionsError(ValueError):
    """The OpenAI Chat Completions adapter could not encode or decode a request."""


class OpenAIResponsesError(ValueError):
    """The OpenAI Responses adapter could not encode or decode a request."""


OPENAI_JSON = JsonValues(OpenAIChatCompletionsError)
OPENAI_RESPONSES_JSON = JsonValues(OpenAIResponsesError)
