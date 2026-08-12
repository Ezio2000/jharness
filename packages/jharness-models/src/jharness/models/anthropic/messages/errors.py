"""Anthropic Messages adapter errors."""

from __future__ import annotations

from jharness.models._json import JsonValues


class AnthropicMessagesError(ValueError):
    """The Anthropic Messages adapter could not encode or decode a request."""


ANTHROPIC_MESSAGES_JSON = JsonValues(AnthropicMessagesError)
