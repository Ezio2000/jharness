"""Immutable wire profile for OpenAI Responses-compatible APIs."""

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
    validate_literal,
)

if TYPE_CHECKING:
    from jharness.models.openai.responses.provider_tools import OpenAIResponsesProviderToolRegistry

OpenAIResponsesReasoningHistoryMode = Literal["summary", "content"]

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


def _default_include() -> frozenset[str]:
    return frozenset({"reasoning.encrypted_content"})


def _empty_provider_tool_registry() -> OpenAIResponsesProviderToolRegistry:
    from jharness.models.openai.responses.provider_tools import OpenAIResponsesProviderToolRegistry

    return OpenAIResponsesProviderToolRegistry()


@dataclass(frozen=True, slots=True)
class OpenAIResponsesProfile:
    """Complete OpenAI Responses capability declaration and wire policy."""

    name: str = "openai-responses"
    capabilities: ModelCapabilities = field(default_factory=_default_capabilities)
    reasoning_history_mode: OpenAIResponsesReasoningHistoryMode = "summary"
    store: bool | None = False
    include: frozenset[str] = field(default_factory=_default_include)
    provider_tool_registry: OpenAIResponsesProviderToolRegistry = field(
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
            profile="OpenAI Responses",
            input_modalities=_INPUT_MODALITIES,
            output_modalities=_OUTPUT_MODALITIES,
        )
        unsupported_choices = capabilities.tool_choice_types.difference(_TOOL_CHOICE_TYPES)
        if unsupported_choices:
            choice = min(unsupported_choices)
            raise ValueError(f"unsupported OpenAI Responses tool choice type: {choice}")
        validate_literal(
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
        from jharness.models.openai.responses.provider_tools import (
            OpenAIResponsesProviderToolRegistry,
        )

        raw_registry = cast(object, self.provider_tool_registry)
        if not isinstance(raw_registry, OpenAIResponsesProviderToolRegistry):
            raise TypeError("provider_tool_registry must be an OpenAIResponsesProviderToolRegistry")
        if self.provider_tool_registry.tools != capabilities.provider_tools:
            raise ValueError(
                "provider_tool_registry must exactly match declared provider tool identities"
            )
        object.__setattr__(
            self,
            "freeform_runtime_tool_names",
            string_set(self.freeform_runtime_tool_names, "freeform_runtime_tool_names"),
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
