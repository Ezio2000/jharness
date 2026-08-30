"""Official Anthropic Messages profile and hosted-tool presets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from jharness.kernel import ProviderToolSpec
from jharness.models.anthropic.messages.profile import AnthropicMessagesProfile
from jharness.models.anthropic.messages.server_tools import ANTHROPIC_MESSAGES_WEB_SEARCH


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
    )
