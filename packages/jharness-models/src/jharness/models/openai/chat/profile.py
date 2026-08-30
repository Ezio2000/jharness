"""Immutable profile for the OpenAI Chat Completions API."""

from __future__ import annotations

from dataclasses import dataclass, field

from jharness.kernel import ModelCapabilities, RuntimeToolKind
from jharness.models._profiles import (
    required_string,
    validate_capabilities,
)

_TOOL_CHOICE_TYPES = frozenset({"auto", "none", "required", "runtime"})
_INPUT_MODALITIES = frozenset({"text", "image", "audio", "file"})
_OUTPUT_MODALITIES = frozenset({"text"})


def _default_capabilities() -> ModelCapabilities:
    return ModelCapabilities(
        streaming=True,
        runtime_tool_kinds=frozenset({RuntimeToolKind.STRUCTURED}),
        tool_choice_types=_TOOL_CHOICE_TYPES,
        parallel_runtime_tool_calls=True,
        parallel_runtime_tool_call_control=True,
        input_modalities=frozenset({"text", "image"}),
        output_modalities=_OUTPUT_MODALITIES,
        structured_output=False,
        json_mode=True,
        seed=True,
        usage_reporting=True,
    )


@dataclass(frozen=True, slots=True)
class OpenAIChatProfile:
    """OpenAI Chat capability declaration.

    The adapter has exactly one official wire representation.  Profiles declare
    model capabilities only; they do not select provider-specific dialects.
    """

    name: str = "openai-chat"
    capabilities: ModelCapabilities = field(default_factory=_default_capabilities)
    json_schema_name: str = "response"

    def __post_init__(self) -> None:
        required_string(self.name, "profile name")
        capabilities = validate_capabilities(
            self.capabilities,
            profile="OpenAI Chat",
            input_modalities=_INPUT_MODALITIES,
            output_modalities=_OUTPUT_MODALITIES,
        )
        if capabilities.provider_tools:
            raise ValueError("OpenAI Chat profiles cannot declare provider tools")
        unsupported_choices = capabilities.tool_choice_types.difference(_TOOL_CHOICE_TYPES)
        if unsupported_choices:
            choice = min(unsupported_choices)
            raise ValueError(f"unsupported OpenAI Chat tool choice type: {choice}")
        required_string(self.json_schema_name, "json_schema_name")
