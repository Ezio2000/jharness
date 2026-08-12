"""Immutable wire profile for OpenAI Chat-compatible APIs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from jharness.kernel import ModelCapabilities, RuntimeToolKind
from jharness.models._profiles import (
    immutable_json_mapping,
    immutable_string_mapping,
    required_string,
    validate_capabilities,
    validate_literal,
)

OpenAIChatMaxTokensField = Literal["max_tokens", "max_completion_tokens"]
OpenAIChatReasoningContentMode = Literal["live_only", "round_trip", "required_with_tools"]
OpenAIChatSystemContentMode = Literal["string", "parts"]
OpenAIChatAssistantToolCallContentMode = Literal["nullable", "required"]
OpenAIChatStreamUsageMode = Literal["include", "omit"]
OpenAIChatAutomaticToolChoiceMode = Literal["explicit", "implicit"]

_TOOL_CHOICE_TYPES = frozenset({"auto", "none", "required", "runtime"})
_INPUT_MODALITIES = frozenset({"text", "image", "video", "file"})
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
    """Complete OpenAI Chat capability declaration and wire policy."""

    name: str = "openai-chat"
    capabilities: ModelCapabilities = field(default_factory=_default_capabilities)
    reasoning_content_mode: OpenAIChatReasoningContentMode = "live_only"
    automatic_tool_choice_mode: OpenAIChatAutomaticToolChoiceMode = "explicit"
    assistant_tool_call_content_mode: OpenAIChatAssistantToolCallContentMode = "nullable"
    max_tokens_field: OpenAIChatMaxTokensField = "max_tokens"
    system_content_mode: OpenAIChatSystemContentMode = "string"
    stream_usage_mode: OpenAIChatStreamUsageMode = "include"
    json_schema_name: str = "response"
    extra_request_body: Mapping[str, Any] = field(default_factory=dict[str, Any])
    finish_reason_map: Mapping[str, str] = field(default_factory=dict[str, str])

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
        validate_literal(
            self.reasoning_content_mode,
            "reasoning_content_mode",
            {"live_only", "round_trip", "required_with_tools"},
        )
        validate_literal(
            self.automatic_tool_choice_mode,
            "automatic_tool_choice_mode",
            {"explicit", "implicit"},
        )
        if (
            self.automatic_tool_choice_mode == "implicit"
            and capabilities.tool_choice_types != frozenset({"auto"})
        ):
            raise ValueError("implicit automatic tool choice requires tool_choice_types={'auto'}")
        validate_literal(
            self.assistant_tool_call_content_mode,
            "assistant_tool_call_content_mode",
            {"nullable", "required"},
        )
        validate_literal(
            self.max_tokens_field,
            "max_tokens_field",
            {"max_tokens", "max_completion_tokens"},
        )
        validate_literal(self.system_content_mode, "system_content_mode", {"string", "parts"})
        validate_literal(self.stream_usage_mode, "stream_usage_mode", {"include", "omit"})
        required_string(self.json_schema_name, "json_schema_name")
        object.__setattr__(
            self,
            "extra_request_body",
            immutable_json_mapping(self.extra_request_body, "extra_request_body"),
        )
        object.__setattr__(
            self,
            "finish_reason_map",
            immutable_string_mapping(self.finish_reason_map, "finish_reason_map"),
        )

    def finish_reason(self, raw: str | None) -> str | None:
        if raw is None:
            return None
        return self.finish_reason_map.get(raw, raw)
