"""Tool conversion for the OpenAI Responses API."""

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
from jharness.models.openai.responses.provider_tools import (
    encode_provider_choice,
    encode_provider_declaration,
)

JsonObject = dict[str, Any]

_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def encode_tools(
    runtime_tools: Sequence[RuntimeToolSpec],
    provider_tools: Sequence[ProviderToolSpec],
    profile: OpenAIResponsesProfile,
) -> list[JsonObject]:
    """Encode runtime declarations and the fixed standard hosted-tool set."""

    encoded = [_encode_runtime_tool(tool, profile) for tool in runtime_tools]
    for tool in provider_tools:
        if tool.tool not in profile.capabilities.provider_tools:
            raise OpenAIResponsesError(
                "Responses profile does not support provider tool: "
                f"{tool.tool.namespace}/{tool.tool.type}"
            )
        encoded.append(encode_provider_declaration(tool))
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
    return encode_provider_choice(selected)


def _encode_runtime_tool(
    tool: RuntimeToolSpec,
    profile: OpenAIResponsesProfile,
) -> JsonObject:
    kind = (
        RuntimeToolKind.STRUCTURED
        if isinstance(tool, StructuredToolSpec)
        else RuntimeToolKind.FREEFORM
    )
    if kind not in profile.capabilities.runtime_tool_kinds:
        raise OpenAIResponsesError(f"{profile.name} does not support {kind.value} runtime tools")
    if isinstance(tool, StructuredToolSpec):
        if _TOOL_NAME.fullmatch(tool.name) is None:
            raise OpenAIResponsesError("Responses function names must match ^[A-Za-z0-9_-]{1,128}$")
        parameters = thaw_json_value(tool.input_schema)
        if not isinstance(parameters, dict):
            raise OpenAIResponsesError("Responses function parameters must be an object")
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
            "strict": False,
        }
    return {"type": "custom", "name": tool.name, "description": tool.description}
