"""Immutable wire profiles for OpenAI-compatible APIs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from jharness.kernel import ModelCapabilities, RuntimeToolKind
from jharness.models._profiles import (
    immutable_json_mapping,
    immutable_string_mapping,
    required_string,
    string_set,
    validate_capabilities,
)

if TYPE_CHECKING:
    from jharness.models.openai.responses_api.provider_tools import (
        ResponsesProviderToolRegistry,
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
_RESPONSES_DEFAULT_TOOL_CHOICE_TYPES = frozenset({"auto", "none", "required", "runtime"})
_RESPONSES_INPUT_MODALITIES = frozenset({"text", "image", "file"})


def _openai_chat_capabilities() -> ModelCapabilities:
    return ModelCapabilities(
        streaming=True,
        runtime_tool_kinds=frozenset({RuntimeToolKind.STRUCTURED}),
        tool_choice_types=_CHAT_TOOL_CHOICE_TYPES,
        parallel_runtime_tool_calls=True,
        parallel_runtime_tool_call_control=True,
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
        runtime_tool_kinds=frozenset({RuntimeToolKind.STRUCTURED, RuntimeToolKind.FREEFORM}),
        tool_choice_types=_RESPONSES_DEFAULT_TOOL_CHOICE_TYPES,
        parallel_runtime_tool_calls=True,
        parallel_runtime_tool_call_control=True,
        input_modalities=frozenset({"text"}),
        output_modalities=_TEXT_OUTPUT_MODALITIES,
        structured_output=False,
        json_mode=False,
        seed=False,
        usage_reporting=True,
    )


def _stateless_responses_include() -> frozenset[str]:
    return frozenset({"reasoning.encrypted_content"})


def _empty_provider_tool_registry() -> ResponsesProviderToolRegistry:
    from jharness.models.openai.responses_api.provider_tools import (
        ResponsesProviderToolRegistry,
    )

    return ResponsesProviderToolRegistry()


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
    store: bool | None = False
    include: frozenset[str] = field(default_factory=_stateless_responses_include)
    provider_tool_registry: ResponsesProviderToolRegistry = field(
        default_factory=_empty_provider_tool_registry
    )
    freeform_runtime_tool_names: frozenset[str] = field(default_factory=lambda: frozenset[str]())
    exact_runtime_tool_choice_kinds: frozenset[RuntimeToolKind] = field(
        default_factory=lambda: frozenset(RuntimeToolKind)
    )
    emit_freeform_runtime_tool_description: bool = True
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
        raw_store = cast(object, self.store)
        if raw_store is not None and not isinstance(raw_store, bool):
            raise TypeError("store must be a bool or None")
        include = string_set(self.include, "include")
        if self.store is False and "reasoning.encrypted_content" not in include:
            raise ValueError(
                "store=False requires reasoning.encrypted_content for stateless reasoning history"
            )
        object.__setattr__(self, "include", include)
        from jharness.models.openai.responses_api.provider_tools import (
            ResponsesProviderToolRegistry,
        )

        raw_registry = cast(object, self.provider_tool_registry)
        if not isinstance(raw_registry, ResponsesProviderToolRegistry):
            raise TypeError("provider_tool_registry must be a ResponsesProviderToolRegistry")
        if self.provider_tool_registry.tools != capabilities.provider_tools:
            raise ValueError(
                "provider_tool_registry must exactly match declared provider tool identities"
            )
        object.__setattr__(
            self,
            "freeform_runtime_tool_names",
            string_set(
                self.freeform_runtime_tool_names,
                "freeform_runtime_tool_names",
            ),
        )
        exact_choice_kinds = frozenset(self.exact_runtime_tool_choice_kinds)
        if not exact_choice_kinds.issubset(capabilities.runtime_tool_kinds):
            raise ValueError(
                "exact_runtime_tool_choice_kinds must be a subset of runtime_tool_kinds"
            )
        object.__setattr__(self, "exact_runtime_tool_choice_kinds", exact_choice_kinds)
        raw_description_policy = cast(object, self.emit_freeform_runtime_tool_description)
        if not isinstance(raw_description_policy, bool):
            raise TypeError("emit_freeform_runtime_tool_description must be a bool")
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

    def allows_freeform_runtime_tool(self, name: str) -> bool:
        """Return whether the Responses dialect accepts this custom-tool name."""

        return not self.freeform_runtime_tool_names or name in self.freeform_runtime_tool_names

    def finish_reason(self, raw: str | None) -> str | None:
        if raw is None:
            return None
        return self.finish_reason_map.get(raw, raw)


def _validate_literal(value: object, label: str, allowed: set[str]) -> None:
    if not isinstance(value, str) or value not in allowed:
        expected = ", ".join(sorted(allowed))
        raise ValueError(f"{label} must be one of: {expected}")
