"""Immutable profile for the OpenAI Responses API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

from jharness.kernel import ModelCapabilities, RuntimeToolKind
from jharness.models._profiles import required_string, string_set, validate_capabilities
from jharness.models.openai.responses.provider_tools import SUPPORTED_PROVIDER_TOOLS

_TOOL_CHOICE_TYPES = frozenset({"auto", "none", "required", "runtime", "provider"})
_DEFAULT_TOOL_CHOICE_TYPES = frozenset({"auto", "none", "required", "runtime"})
_INPUT_MODALITIES = frozenset({"text", "image", "file"})
_OUTPUT_MODALITIES = frozenset({"text"})


def _default_capabilities() -> ModelCapabilities:
    return ModelCapabilities(
        streaming=True,
        runtime_tool_kinds=frozenset({RuntimeToolKind.STRUCTURED, RuntimeToolKind.FREEFORM}),
        tool_choice_types=_DEFAULT_TOOL_CHOICE_TYPES,
        parallel_runtime_tool_calls=True,
        parallel_runtime_tool_call_control=True,
        input_modalities=frozenset({"text"}),
        output_modalities=_OUTPUT_MODALITIES,
        structured_output=False,
        json_mode=False,
        seed=False,
        usage_reporting=True,
    )


_INCLUDE_VALUES = frozenset(
    {
        "message.input_image.image_url",
        "message.output_text.logprobs",
        "reasoning.encrypted_content",
        "web_search_call.action.sources",
    }
)


def _default_include() -> frozenset[str]:
    return frozenset({"reasoning.encrypted_content"})


@dataclass(frozen=True, slots=True)
class OpenAIResponsesProfile:
    """Complete OpenAI Responses capability declaration and wire policy."""

    name: str = "openai-responses"
    capabilities: ModelCapabilities = field(default_factory=_default_capabilities)
    store: bool = False
    include: frozenset[str] = field(default_factory=_default_include)

    def __post_init__(self) -> None:
        required_string(self.name, "profile name")
        capabilities = validate_capabilities(
            self.capabilities,
            profile="OpenAI Responses",
            input_modalities=_INPUT_MODALITIES,
            output_modalities=_OUTPUT_MODALITIES,
        )
        unsupported_choices = capabilities.tool_choice_types.difference(_TOOL_CHOICE_TYPES)
        if unsupported_choices:
            choice = min(unsupported_choices)
            raise ValueError(f"unsupported OpenAI Responses tool choice type: {choice}")
        if capabilities.seed:
            raise ValueError("OpenAI Responses does not support seed")
        unsupported_tools = capabilities.provider_tools.difference(SUPPORTED_PROVIDER_TOOLS)
        if unsupported_tools:
            tool = min(unsupported_tools, key=lambda item: (item.namespace, item.type))
            raise ValueError(
                f"unsupported OpenAI Responses provider tool: {tool.namespace}/{tool.type}"
            )
        raw_store = cast(object, self.store)
        if not isinstance(raw_store, bool):
            raise TypeError("store must be a bool")
        include = string_set(self.include, "include")
        unsupported_include = include.difference(_INCLUDE_VALUES)
        if unsupported_include:
            raise ValueError(
                "unsupported OpenAI Responses include value: " + min(unsupported_include)
            )
        object.__setattr__(self, "include", include)
