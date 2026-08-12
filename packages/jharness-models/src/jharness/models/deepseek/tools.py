"""Typed provider-tool declarations supported by DeepSeek profiles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jharness.kernel import ProviderToolId, ProviderToolSpec

DEEPSEEK_ANTHROPIC_WEB_SEARCH = ProviderToolId(
    "deepseek.anthropic",
    "web_search",
)
DEEPSEEK_RESPONSES_WEB_SEARCH = ProviderToolId(
    "deepseek.responses",
    "web_search",
)


def deepseek_anthropic_web_search(
    configuration: Mapping[str, Any] | None = None,
) -> ProviderToolSpec:
    """Declare DeepSeek's Anthropic-compatible hosted web-search tool."""

    return ProviderToolSpec(
        DEEPSEEK_ANTHROPIC_WEB_SEARCH,
        {} if configuration is None else configuration,
    )


def deepseek_responses_web_search(
    configuration: Mapping[str, Any] | None = None,
) -> ProviderToolSpec:
    """Declare DeepSeek's Responses-compatible hosted web-search tool."""

    return ProviderToolSpec(
        DEEPSEEK_RESPONSES_WEB_SEARCH,
        {} if configuration is None else configuration,
    )
