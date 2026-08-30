"""Immutable profile for the Anthropic Messages API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from jharness.kernel import ModelCapabilities, RuntimeToolKind
from jharness.models._profiles import (
    immutable_json_mapping,
    required_string,
    validate_capabilities,
)
from jharness.models.anthropic.messages.server_tools import ANTHROPIC_MESSAGES_WEB_SEARCH

_TOOL_CHOICE_TYPES = frozenset({"auto", "none", "required", "runtime", "provider"})
_DEFAULT_TOOL_CHOICE_TYPES = frozenset({"auto", "none", "required", "runtime"})
_INPUT_MODALITIES = frozenset({"text", "image", "file"})
_OUTPUT_MODALITIES = frozenset({"text", "file"})


def _default_capabilities() -> ModelCapabilities:
    return ModelCapabilities(
        streaming=True,
        runtime_tool_kinds=frozenset({RuntimeToolKind.STRUCTURED}),
        tool_choice_types=_DEFAULT_TOOL_CHOICE_TYPES,
        parallel_runtime_tool_calls=True,
        parallel_runtime_tool_call_control=True,
        input_modalities=_INPUT_MODALITIES,
        output_modalities=_OUTPUT_MODALITIES,
        structured_output=True,
        json_mode=False,
        seed=False,
        usage_reporting=True,
    )


@dataclass(frozen=True, slots=True)
class AnthropicMessagesProfile:
    """Complete Anthropic Messages capability declaration and wire policy."""

    name: str = "anthropic-messages"
    capabilities: ModelCapabilities = field(default_factory=_default_capabilities)
    anthropic_version: str = "2023-06-01"
    default_max_tokens: int = 1024
    json_object_schema: Mapping[str, Any] = field(default_factory=lambda: {"type": "object"})

    def __post_init__(self) -> None:
        required_string(self.name, "profile name")
        required_string(self.anthropic_version, "anthropic_version")
        capabilities = validate_capabilities(
            self.capabilities,
            profile="Anthropic Messages",
            input_modalities=_INPUT_MODALITIES,
            output_modalities=_OUTPUT_MODALITIES,
        )
        unsupported_choices = capabilities.tool_choice_types.difference(_TOOL_CHOICE_TYPES)
        if unsupported_choices:
            choice = min(unsupported_choices)
            raise ValueError(f"unsupported Anthropic Messages tool choice type: {choice}")
        unsupported_provider_tools = capabilities.provider_tools.difference(
            {ANTHROPIC_MESSAGES_WEB_SEARCH}
        )
        if unsupported_provider_tools:
            tool = min(
                unsupported_provider_tools,
                key=lambda item: (item.namespace, item.type),
            )
            raise ValueError(
                f"unsupported Anthropic Messages provider tool: {tool.namespace}/{tool.type}"
            )
        unsupported_runtime_kinds = capabilities.runtime_tool_kinds.difference(
            {RuntimeToolKind.STRUCTURED}
        )
        if unsupported_runtime_kinds:
            kind = min(item.value for item in unsupported_runtime_kinds)
            raise ValueError(f"unsupported Anthropic Messages runtime tool kind: {kind}")
        if capabilities.seed:
            raise ValueError("Anthropic Messages does not support seed")
        if not isinstance(cast(object, self.default_max_tokens), int) or isinstance(
            self.default_max_tokens, bool
        ):
            raise TypeError("default_max_tokens must be an integer")
        if self.default_max_tokens < 1:
            raise ValueError("default_max_tokens must be >= 1")
        object.__setattr__(
            self,
            "json_object_schema",
            immutable_json_mapping(self.json_object_schema, "json_object_schema"),
        )
