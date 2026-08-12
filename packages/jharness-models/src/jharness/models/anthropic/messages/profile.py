"""Immutable wire profile for Anthropic-compatible Messages APIs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from jharness.kernel import ModelCapabilities, RuntimeToolKind
from jharness.models._profiles import (
    immutable_json_mapping,
    immutable_string_mapping,
    required_string,
    validate_capabilities,
    validate_literal,
)
from jharness.models.anthropic.messages.server_tools import AnthropicMessagesServerToolRegistry

AnthropicMessagesAuthScheme = Literal["x-api-key", "bearer"]
AnthropicMessagesSystemContentMode = Literal["string", "blocks"]
AnthropicMessagesRedactedThinkingMode = Literal["round_trip", "reject"]
AnthropicMessagesMidConversationSystemMode = Literal["encode", "reject"]
AnthropicMessagesStreamUsageMode = Literal["include", "omit"]
AnthropicMessagesAutomaticToolChoiceMode = Literal["explicit", "implicit"]

_TOOL_CHOICE_TYPES = frozenset({"auto", "none", "required", "runtime", "provider"})
_DEFAULT_TOOL_CHOICE_TYPES = frozenset({"auto", "none", "required", "runtime"})
_INPUT_MODALITIES = frozenset({"text", "image", "file"})
_OUTPUT_MODALITIES = frozenset({"text"})


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
    server_tools: AnthropicMessagesServerToolRegistry = field(
        default_factory=AnthropicMessagesServerToolRegistry
    )
    anthropic_version: str = "2023-06-01"
    auth_scheme: AnthropicMessagesAuthScheme = "x-api-key"
    automatic_tool_choice_mode: AnthropicMessagesAutomaticToolChoiceMode = "explicit"
    redacted_thinking_mode: AnthropicMessagesRedactedThinkingMode = "round_trip"
    stream_usage_mode: AnthropicMessagesStreamUsageMode = "include"
    default_max_tokens: int = 1024
    system_content_mode: AnthropicMessagesSystemContentMode = "string"
    mid_conversation_system_mode: AnthropicMessagesMidConversationSystemMode = "reject"
    seed_field: str | None = None
    file_ref_beta_header: str | None = "files-api-2025-04-14"
    json_object_schema: Mapping[str, Any] = field(default_factory=lambda: {"type": "object"})
    extra_output_config: Mapping[str, Any] = field(default_factory=dict[str, Any])
    extra_request_body: Mapping[str, Any] = field(default_factory=dict[str, Any])
    extra_headers: Mapping[str, str] = field(default_factory=dict[str, str])
    finish_reason_map: Mapping[str, str] = field(default_factory=dict[str, str])

    def __post_init__(self) -> None:
        required_string(self.name, "profile name")
        required_string(self.anthropic_version, "anthropic_version")
        capabilities = validate_capabilities(
            self.capabilities,
            profile="Anthropic Messages",
            input_modalities=_INPUT_MODALITIES,
            output_modalities=_OUTPUT_MODALITIES,
        )
        if not isinstance(cast(object, self.server_tools), AnthropicMessagesServerToolRegistry):
            raise TypeError("server_tools must be an AnthropicMessagesServerToolRegistry")
        if capabilities.provider_tools != self.server_tools.tools:
            raise ValueError(
                "Anthropic Messages capabilities.provider_tools must exactly match the server "
                "tool registry"
            )
        unsupported_choices = capabilities.tool_choice_types.difference(_TOOL_CHOICE_TYPES)
        if unsupported_choices:
            choice = min(unsupported_choices)
            raise ValueError(f"unsupported Anthropic Messages tool choice type: {choice}")
        validate_literal(self.auth_scheme, "auth_scheme", {"x-api-key", "bearer"})
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
            self.redacted_thinking_mode,
            "redacted_thinking_mode",
            {"round_trip", "reject"},
        )
        validate_literal(
            self.stream_usage_mode,
            "stream_usage_mode",
            {"include", "omit"},
        )
        validate_literal(
            self.system_content_mode,
            "system_content_mode",
            {"string", "blocks"},
        )
        validate_literal(
            self.mid_conversation_system_mode,
            "mid_conversation_system_mode",
            {"encode", "reject"},
        )
        if not isinstance(cast(object, self.default_max_tokens), int) or isinstance(
            self.default_max_tokens, bool
        ):
            raise TypeError("default_max_tokens must be an integer")
        if self.default_max_tokens < 1:
            raise ValueError("default_max_tokens must be >= 1")
        _validate_optional_string(self.seed_field, "seed_field")
        if capabilities.seed != (self.seed_field is not None):
            raise ValueError("capabilities.seed must match whether seed_field is declared")
        _validate_optional_string(self.file_ref_beta_header, "file_ref_beta_header")
        for field_name in (
            "json_object_schema",
            "extra_output_config",
            "extra_request_body",
        ):
            object.__setattr__(
                self,
                field_name,
                immutable_json_mapping(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "extra_headers",
            immutable_string_mapping(self.extra_headers, "extra_headers"),
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


def _validate_optional_string(value: object, label: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string when set")
