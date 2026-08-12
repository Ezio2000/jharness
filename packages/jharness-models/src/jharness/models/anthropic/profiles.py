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
)
from jharness.models.anthropic.messages_api.server_tools import AnthropicServerToolRegistry

AnthropicAuthScheme = Literal["x-api-key", "bearer"]
SystemContentMode = Literal["string", "blocks"]
RedactedThinkingMode = Literal["round_trip", "reject"]
MidConversationSystemMode = Literal["encode", "reject"]
StreamUsageMode = Literal["include", "omit"]
AutomaticToolChoiceMode = Literal["explicit", "implicit"]

_ANTHROPIC_TOOL_CHOICE_TYPES = frozenset({"auto", "none", "required", "runtime", "provider"})
_ANTHROPIC_DEFAULT_TOOL_CHOICE_TYPES = frozenset({"auto", "none", "required", "runtime"})
_ANTHROPIC_INPUT_MODALITIES = frozenset({"text", "image", "file"})
_TEXT_OUTPUT_MODALITIES = frozenset({"text"})


def _anthropic_capabilities() -> ModelCapabilities:
    return ModelCapabilities(
        streaming=True,
        runtime_tool_kinds=frozenset({RuntimeToolKind.STRUCTURED}),
        tool_choice_types=_ANTHROPIC_DEFAULT_TOOL_CHOICE_TYPES,
        parallel_runtime_tool_calls=True,
        parallel_runtime_tool_call_control=True,
        input_modalities=_ANTHROPIC_INPUT_MODALITIES,
        output_modalities=_TEXT_OUTPUT_MODALITIES,
        structured_output=True,
        json_mode=False,
        seed=False,
        usage_reporting=True,
    )


@dataclass(frozen=True, slots=True)
class AnthropicProfile:
    """Complete Anthropic Messages capability declaration and wire policy."""

    name: str = "anthropic"
    capabilities: ModelCapabilities = field(default_factory=_anthropic_capabilities)
    server_tools: AnthropicServerToolRegistry = field(default_factory=AnthropicServerToolRegistry)
    anthropic_version: str = "2023-06-01"
    auth_scheme: AnthropicAuthScheme = "x-api-key"
    automatic_tool_choice_mode: AutomaticToolChoiceMode = "explicit"
    redacted_thinking_mode: RedactedThinkingMode = "round_trip"
    stream_usage_mode: StreamUsageMode = "include"
    default_max_tokens: int = 1024
    system_content_mode: SystemContentMode = "string"
    mid_conversation_system_mode: MidConversationSystemMode = "reject"
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
            input_modalities=_ANTHROPIC_INPUT_MODALITIES,
            output_modalities=_TEXT_OUTPUT_MODALITIES,
        )
        if not isinstance(cast(object, self.server_tools), AnthropicServerToolRegistry):
            raise TypeError("server_tools must be an AnthropicServerToolRegistry")
        if capabilities.provider_tools != self.server_tools.tools:
            raise ValueError(
                "Anthropic capabilities.provider_tools must exactly match the server tool registry"
            )
        unsupported_choices = capabilities.tool_choice_types.difference(
            _ANTHROPIC_TOOL_CHOICE_TYPES
        )
        if unsupported_choices:
            choice = min(unsupported_choices)
            raise ValueError(f"unsupported Anthropic tool choice type: {choice}")
        _validate_literal(self.auth_scheme, "auth_scheme", {"x-api-key", "bearer"})
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
            self.redacted_thinking_mode,
            "redacted_thinking_mode",
            {"round_trip", "reject"},
        )
        _validate_literal(
            self.stream_usage_mode,
            "stream_usage_mode",
            {"include", "omit"},
        )
        _validate_literal(
            self.system_content_mode,
            "system_content_mode",
            {"string", "blocks"},
        )
        _validate_literal(
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


def _validate_literal(value: object, label: str, allowed: set[str]) -> None:
    if not isinstance(value, str) or value not in allowed:
        expected = ", ".join(sorted(allowed))
        raise ValueError(f"{label} must be one of: {expected}")


def _validate_optional_string(value: object, label: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string when set")
