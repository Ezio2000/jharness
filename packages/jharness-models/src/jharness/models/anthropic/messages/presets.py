"""Official Anthropic Messages profile and hosted-tool presets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from jharness.kernel import ProviderToolId, ProviderToolSpec
from jharness.models.anthropic.messages.profile import AnthropicMessagesProfile
from jharness.models.anthropic.messages.server_tools import (
    AnthropicMessagesServerToolRegistry,
    anthropic_messages_web_search_codec,
)

ANTHROPIC_MESSAGES_WEB_SEARCH = ProviderToolId(
    "anthropic.messages",
    "web_search",
)
_ANTHROPIC_MESSAGES_WEB_SEARCH_VARIANTS = frozenset(
    {
        "web_search_20250305",
        "web_search_20260209",
        "web_search_20260318",
    }
)


def anthropic_messages_web_search(
    configuration: Mapping[str, Any] | None = None,
) -> ProviderToolSpec:
    """Declare Anthropic's hosted web-search tool for one request."""

    return ProviderToolSpec(
        ANTHROPIC_MESSAGES_WEB_SEARCH,
        {} if configuration is None else configuration,
    )


def anthropic_messages_profile() -> AnthropicMessagesProfile:
    """Return the official Anthropic Messages profile with hosted tools installed."""

    base = AnthropicMessagesProfile()
    return replace(
        base,
        capabilities=replace(
            base.capabilities,
            tool_choice_types=base.capabilities.tool_choice_types | {"provider"},
            provider_tools=frozenset({ANTHROPIC_MESSAGES_WEB_SEARCH}),
        ),
        server_tools=AnthropicMessagesServerToolRegistry(
            (
                anthropic_messages_web_search_codec(
                    ANTHROPIC_MESSAGES_WEB_SEARCH,
                    variants=_ANTHROPIC_MESSAGES_WEB_SEARCH_VARIANTS,
                ),
            )
        ),
    )
