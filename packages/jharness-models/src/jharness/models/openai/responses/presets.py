"""Official OpenAI Responses profile and hosted-tool declarations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from jharness.kernel import ProviderToolSpec
from jharness.models.openai.responses.profile import OpenAIResponsesProfile
from jharness.models.openai.responses.provider_tools import (
    OPENAI_RESPONSES_IMAGE_GENERATION,
    OPENAI_RESPONSES_WEB_SEARCH,
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
    return replace(
        base,
        capabilities=replace(
            base.capabilities,
            tool_choice_types=base.capabilities.tool_choice_types | {"provider"},
            provider_tools=frozenset(
                {OPENAI_RESPONSES_WEB_SEARCH, OPENAI_RESPONSES_IMAGE_GENERATION}
            ),
        ),
    )
