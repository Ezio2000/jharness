"""Official OpenAI Responses profile and hosted-tool declarations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from jharness.kernel import ProviderToolId, ProviderToolSpec
from jharness.models.openai.responses.profile import OpenAIResponsesProfile
from jharness.models.openai.responses.provider_tools import (
    OpenAIResponsesImageGenerationTool,
    OpenAIResponsesProviderToolRegistry,
    OpenAIResponsesWebSearchTool,
)

OPENAI_RESPONSES_WEB_SEARCH = ProviderToolId(
    "openai.responses",
    "web_search",
)
OPENAI_RESPONSES_IMAGE_GENERATION = ProviderToolId(
    "openai.responses",
    "image_generation",
)


def openai_responses_web_search(
    configuration: Mapping[str, Any] | None = None,
) -> ProviderToolSpec:
    """Declare OpenAI's Responses hosted web-search tool."""

    return ProviderToolSpec(
        OPENAI_RESPONSES_WEB_SEARCH,
        {} if configuration is None else configuration,
    )


def openai_responses_image_generation(
    configuration: Mapping[str, Any] | None = None,
) -> ProviderToolSpec:
    """Declare OpenAI's Responses hosted image-generation tool."""

    return ProviderToolSpec(
        OPENAI_RESPONSES_IMAGE_GENERATION,
        {} if configuration is None else configuration,
    )


def openai_responses_profile() -> OpenAIResponsesProfile:
    """Return the official OpenAI Responses profile with hosted tools installed."""

    base = OpenAIResponsesProfile()
    registry = OpenAIResponsesProviderToolRegistry(
        (
            OpenAIResponsesWebSearchTool(tool=OPENAI_RESPONSES_WEB_SEARCH),
            OpenAIResponsesImageGenerationTool(tool=OPENAI_RESPONSES_IMAGE_GENERATION),
        )
    )
    return replace(
        base,
        capabilities=replace(
            base.capabilities,
            tool_choice_types=base.capabilities.tool_choice_types | {"provider"},
            provider_tools=registry.tools,
        ),
        provider_tool_registry=registry,
    )
