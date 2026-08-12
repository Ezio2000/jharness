"""Tool conversion for OpenAI-compatible Responses APIs."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from jharness.kernel import (
    ProviderToolSpec,
    RuntimeToolKind,
    RuntimeToolSpec,
    StructuredToolSpec,
    ToolChoice,
    thaw_json_value,
)
from jharness.models.openai.responses.errors import OpenAIResponsesError
from jharness.models.openai.responses.profile import OpenAIResponsesProfile

JsonObject = dict[str, Any]

_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def encode_tools(
    runtime_tools: Sequence[RuntimeToolSpec],
    provider_tools: Sequence[ProviderToolSpec],
    profile: OpenAIResponsesProfile,
) -> list[JsonObject]:
    """Encode runtime- and provider-owned declarations through their wire dialects."""

    encoded = [_encode_runtime_tool(tool, profile) for tool in runtime_tools]
    encoded.extend(
        profile.provider_tool_registry.encode_declaration(tool) for tool in provider_tools
    )
    names = [tool.name for tool in runtime_tools]
    if len(names) != len(set(names)):
        raise OpenAIResponsesError("Responses runtime tool names must be unique")
    return encoded


def encode_tool_choice(
    choice: ToolChoice,
    *,
    runtime_tools: Sequence[RuntimeToolSpec],
    provider_tools: Sequence[ProviderToolSpec],
    profile: OpenAIResponsesProfile,
) -> str | JsonObject | None:
    """Encode one tool-selection policy across both execution owners."""

    if choice.type not in profile.capabilities.tool_choice_types:
        raise OpenAIResponsesError(
            f"{profile.name} does not support tool_choice={choice.type!r} in this mode"
        )
    has_tools = bool(runtime_tools or provider_tools)
    if not has_tools:
        if choice.type in {"required", "runtime", "provider"}:
            raise OpenAIResponsesError(f"tool_choice={choice.type!r} requires tools")
        return None
    if choice.type in {"auto", "none", "required"}:
        return choice.type
    if choice.type == "runtime":
        return _encode_runtime_tool_choice(choice, runtime_tools, profile)
    return _encode_provider_tool_choice(choice, provider_tools, profile)


def _encode_runtime_tool_choice(
    choice: ToolChoice,
    runtime_tools: Sequence[RuntimeToolSpec],
    profile: OpenAIResponsesProfile,
) -> JsonObject:
    selected = next((tool for tool in runtime_tools if tool.name == choice.name), None)
    if selected is None:
        raise OpenAIResponsesError(f"tool_choice names an unavailable runtime tool: {choice.name}")
    kind = (
        RuntimeToolKind.STRUCTURED
        if isinstance(selected, StructuredToolSpec)
        else RuntimeToolKind.FREEFORM
    )
    if kind not in profile.exact_runtime_tool_choice_kinds:
        raise OpenAIResponsesError(
            f"{profile.name} does not support exact {kind.value} runtime tool choice"
        )
    return {
        "type": "function" if isinstance(selected, StructuredToolSpec) else "custom",
        "name": selected.name,
    }


def _encode_provider_tool_choice(
    choice: ToolChoice,
    provider_tools: Sequence[ProviderToolSpec],
    profile: OpenAIResponsesProfile,
) -> JsonObject:
    if choice.provider_tool is None:
        raise OpenAIResponsesError("provider tool choice requires a provider tool")
    selected = next(
        (spec for spec in provider_tools if spec.tool == choice.provider_tool),
        None,
    )
    if selected is None:
        raise OpenAIResponsesError(
            f"tool_choice names an unavailable provider tool: {choice.provider_tool}"
        )
    return profile.provider_tool_registry.encode_choice(selected)


def _encode_runtime_tool(
    tool: RuntimeToolSpec,
    profile: OpenAIResponsesProfile,
) -> JsonObject:
    if _TOOL_NAME.fullmatch(tool.name) is None:
        raise OpenAIResponsesError("Responses runtime tool names must match ^[A-Za-z0-9_-]{1,128}$")
    kind = (
        RuntimeToolKind.STRUCTURED
        if isinstance(tool, StructuredToolSpec)
        else RuntimeToolKind.FREEFORM
    )
    if kind not in profile.capabilities.runtime_tool_kinds:
        raise OpenAIResponsesError(f"{profile.name} does not support {kind.value} runtime tools")
    if isinstance(tool, StructuredToolSpec):
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": thaw_json_value(tool.input_schema),
        }
    if not profile.allows_freeform_runtime_tool(tool.name):
        raise OpenAIResponsesError(
            f"{profile.name} does not support freeform runtime tool: {tool.name}"
        )
    declaration: JsonObject = {"type": "custom", "name": tool.name}
    if profile.emit_freeform_runtime_tool_description:
        declaration["description"] = tool.description
    return declaration
