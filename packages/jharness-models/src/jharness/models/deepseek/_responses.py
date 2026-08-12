"""Private DeepSeek Responses wire adaptations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jharness.kernel import ProviderToolId, ProviderToolStatus
from jharness.models.openai.responses_api.provider_tools import (
    ProviderStreamUpdate,
    ResponsesWebSearchTool,
)


class _DeepSeekResponsesWebSearchTool(ResponsesWebSearchTool):
    """Decode DeepSeek search lifecycle events without finalizing the output item."""

    __slots__ = ()

    def stream_event_update(
        self,
        event_type: str,
        value: Mapping[str, Any],
    ) -> ProviderStreamUpdate:
        update = super().stream_event_update(event_type, value)
        return ProviderStreamUpdate(ProviderToolStatus.IN_PROGRESS, update.data)


def deepseek_responses_web_search_codec(tool: ProviderToolId) -> ResponsesWebSearchTool:
    """Build the DeepSeek-specific hosted web-search codec."""

    return _DeepSeekResponsesWebSearchTool(
        tool=tool,
        allowed_variants=frozenset({"web_search", "web_search_2025_08_26"}),
        configuration_fields=frozenset({"variant"}),
    )
