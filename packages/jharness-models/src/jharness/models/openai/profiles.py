"""Immutable wire profiles for OpenAI-compatible APIs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from jharness.kernel import ModelCapabilities, ProviderToolId
from jharness.models._profiles import (
    immutable_json_mapping,
    immutable_string_mapping,
    immutable_string_set_mapping,
    required_string,
    string_set,
    validate_capabilities,
)

MaxTokensField = Literal["max_tokens", "max_completion_tokens"]
ReasoningContentMode = Literal["live_only", "round_trip", "required_with_tools"]
SystemContentMode = Literal["string", "parts"]
AssistantToolCallContentMode = Literal["nullable", "required"]
StreamUsageMode = Literal["include", "omit"]
ResponsesReasoningHistoryMode = Literal["summary", "content"]
AutomaticToolChoiceMode = Literal["explicit", "implicit"]

_CHAT_TOOL_CHOICE_TYPES = frozenset({"auto", "none", "required", "runtime"})
_CHAT_INPUT_MODALITIES = frozenset({"text", "image", "video", "file"})
_TEXT_OUTPUT_MODALITIES = frozenset({"text"})
_RESPONSES_TOOL_CHOICE_TYPES = frozenset({"auto", "none", "required", "runtime", "provider"})
_RESPONSES_INPUT_MODALITIES = frozenset({"text", "image", "file"})
_RESPONSES_PROVIDER_TOOL_TYPES = frozenset({"image_generation", "web_search"})
_IMAGE_GENERATION_CONFIGURATION_FIELDS = frozenset(
    {
        "action",
        "background",
        "input_fidelity",
        "input_image_mask",
        "moderation",
        "output_compression",
        "output_format",
        "partial_images",
        "quality",
        "size",
    }
)
_WEB_SEARCH_CONFIGURATION_FIELDS = frozenset(
    {"filters", "search_context_size", "user_location", "variant"}
)


def _openai_chat_capabilities() -> ModelCapabilities:
    return ModelCapabilities(
        streaming=True,
        runtime_tools=True,
        tool_choice_types=_CHAT_TOOL_CHOICE_TYPES,
        parallel_tool_calls=True,
        parallel_tool_call_control=True,
        input_modalities=frozenset({"text", "image"}),
        output_modalities=_TEXT_OUTPUT_MODALITIES,
        structured_output=False,
        json_mode=True,
        seed=True,
        usage_reporting=True,
    )


def _openai_responses_capabilities() -> ModelCapabilities:
    return ModelCapabilities(
        streaming=True,
        runtime_tools=True,
        tool_choice_types=_RESPONSES_TOOL_CHOICE_TYPES,
        parallel_tool_calls=True,
        parallel_tool_call_control=True,
        input_modalities=frozenset({"text", "image", "file"}),
        output_modalities=_TEXT_OUTPUT_MODALITIES,
        provider_tools=frozenset(
            {
                ProviderToolId("openai.responses", "image_generation"),
                ProviderToolId("openai.responses", "web_search"),
            }
        ),
        structured_output=True,
        json_mode=True,
        seed=False,
        usage_reporting=True,
    )


def _openai_responses_tool_configuration() -> Mapping[str, frozenset[str]]:
    return {
        "image_generation": _IMAGE_GENERATION_CONFIGURATION_FIELDS,
        "web_search": _WEB_SEARCH_CONFIGURATION_FIELDS,
    }


@dataclass(frozen=True, slots=True)
class OpenAIChatCompletionsProfile:
    """Complete Chat Completions capability declaration and wire policy."""

    name: str = "openai-chat-completions"
    capabilities: ModelCapabilities = field(default_factory=_openai_chat_capabilities)
    reasoning_content_mode: ReasoningContentMode = "live_only"
    automatic_tool_choice_mode: AutomaticToolChoiceMode = "explicit"
    assistant_tool_call_content_mode: AssistantToolCallContentMode = "nullable"
    max_tokens_field: MaxTokensField = "max_tokens"
    system_content_mode: SystemContentMode = "string"
    stream_usage_mode: StreamUsageMode = "include"
    json_schema_name: str = "response"
    extra_request_body: Mapping[str, Any] = field(default_factory=dict[str, Any])
    finish_reason_map: Mapping[str, str] = field(default_factory=dict[str, str])

    def __post_init__(self) -> None:
        required_string(self.name, "profile name")
        capabilities = validate_capabilities(
            self.capabilities,
            profile="Chat Completions",
            input_modalities=_CHAT_INPUT_MODALITIES,
            output_modalities=_TEXT_OUTPUT_MODALITIES,
        )
        if capabilities.provider_tools:
            raise ValueError("Chat Completions profiles cannot declare provider tools")
        unsupported_choices = capabilities.tool_choice_types.difference(_CHAT_TOOL_CHOICE_TYPES)
        if unsupported_choices:
            choice = min(unsupported_choices)
            raise ValueError(f"unsupported Chat Completions tool choice type: {choice}")
        _validate_literal(
            self.reasoning_content_mode,
            "reasoning_content_mode",
            {"live_only", "round_trip", "required_with_tools"},
        )
        _validate_literal(
            self.automatic_tool_choice_mode,
            "automatic_tool_choice_mode",
            {"explicit", "implicit"},
        )
        if (
            self.automatic_tool_choice_mode == "implicit"
            and capabilities.tool_choice_types != frozenset({"auto"})
        ):
            raise ValueError("implicit automatic tool choice requires tool_choice_types={'auto'}")
        _validate_literal(
            self.assistant_tool_call_content_mode,
            "assistant_tool_call_content_mode",
            {"nullable", "required"},
        )
        _validate_literal(
            self.max_tokens_field,
            "max_tokens_field",
            {"max_tokens", "max_completion_tokens"},
        )
        _validate_literal(
            self.system_content_mode,
            "system_content_mode",
            {"string", "parts"},
        )
        _validate_literal(
            self.stream_usage_mode,
            "stream_usage_mode",
            {"include", "omit"},
        )
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


@dataclass(frozen=True, slots=True)
class OpenAIResponsesProfile:
    """Complete Responses API capability declaration and wire policy."""

    name: str = "openai-responses"
    capabilities: ModelCapabilities = field(default_factory=_openai_responses_capabilities)
    reasoning_history_mode: ResponsesReasoningHistoryMode = "summary"
    provider_tool_configuration_fields: Mapping[str, frozenset[str]] = field(
        default_factory=_openai_responses_tool_configuration
    )
    allowed_models: frozenset[str] = field(default_factory=lambda: frozenset[str]())
    extra_request_body: Mapping[str, Any] = field(default_factory=dict[str, Any])
    finish_reason_map: Mapping[str, str] = field(
        default_factory=lambda: {"max_output_tokens": "length"}
    )

    def __post_init__(self) -> None:
        required_string(self.name, "profile name")
        capabilities = validate_capabilities(
            self.capabilities,
            profile="Responses",
            input_modalities=_RESPONSES_INPUT_MODALITIES,
            output_modalities=_TEXT_OUTPUT_MODALITIES,
            allowed_provider_tools=_RESPONSES_PROVIDER_TOOL_TYPES,
        )
        unsupported_choices = capabilities.tool_choice_types.difference(
            _RESPONSES_TOOL_CHOICE_TYPES
        )
        if unsupported_choices:
            choice = min(unsupported_choices)
            raise ValueError(f"unsupported Responses tool choice type: {choice}")
        _validate_literal(
            self.reasoning_history_mode,
            "reasoning_history_mode",
            {"summary", "content"},
        )
        configuration_fields = immutable_string_set_mapping(
            self.provider_tool_configuration_fields,
            "provider_tool_configuration_fields",
        )
        declared_types = {tool.type for tool in capabilities.provider_tools}
        if set(configuration_fields) != declared_types:
            raise ValueError(
                "provider_tool_configuration_fields must exactly match declared provider tools"
            )
        for tool_type, fields in configuration_fields.items():
            allowed = (
                _IMAGE_GENERATION_CONFIGURATION_FIELDS
                if tool_type == "image_generation"
                else _WEB_SEARCH_CONFIGURATION_FIELDS
            )
            unsupported_fields = fields.difference(allowed)
            if unsupported_fields:
                field_name = min(unsupported_fields)
                raise ValueError(f"unsupported {tool_type} configuration field: {field_name}")
        object.__setattr__(
            self,
            "provider_tool_configuration_fields",
            configuration_fields,
        )
        object.__setattr__(
            self,
            "allowed_models",
            string_set(self.allowed_models, "allowed_models"),
        )
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

    def validate_model(self, model: str) -> None:
        """Reject model identifiers the compatible endpoint does not serve."""

        required_string(model, "model")
        if self.allowed_models and model not in self.allowed_models:
            allowed = ", ".join(sorted(self.allowed_models))
            raise ValueError(f"{self.name} only supports models: {allowed}")

    def provider_tool(self, tool_type: str) -> ProviderToolId:
        """Resolve one Responses wire tool type to its declared provider identity."""

        matches = tuple(tool for tool in self.capabilities.provider_tools if tool.type == tool_type)
        if not matches:
            raise ValueError(f"{self.name} does not declare provider tool: {tool_type}")
        return matches[0]

    def finish_reason(self, raw: str | None) -> str | None:
        if raw is None:
            return None
        return self.finish_reason_map.get(raw, raw)


def _validate_literal(value: object, label: str, allowed: set[str]) -> None:
    if not isinstance(value, str) or value not in allowed:
        expected = ", ".join(sorted(allowed))
        raise ValueError(f"{label} must be one of: {expected}")
