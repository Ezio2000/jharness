"""Tool conversion for Anthropic Messages."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from typing import Any, cast

from jharness.kernel import (
    ProviderToolSpec,
    RuntimeToolKind,
    RuntimeToolSpec,
    StructuredToolCall,
    StructuredToolSpec,
    ToolChoice,
    thaw_json_value,
)
from jharness.models.anthropic.errors import ANTHROPIC_JSON, AnthropicError
from jharness.models.anthropic.profiles import AnthropicProfile

JsonValue = Any
JsonObject = dict[str, JsonValue]


def encode_tools(
    runtime_tools: Sequence[RuntimeToolSpec],
    provider_tools: Sequence[ProviderToolSpec],
    profile: AnthropicProfile,
) -> list[JsonObject]:
    if not runtime_tools and not provider_tools:
        return []
    encoded: list[JsonObject] = []
    wire_names: set[str] = set()
    if runtime_tools and RuntimeToolKind.STRUCTURED not in profile.capabilities.runtime_tool_kinds:
        raise AnthropicError(f"{profile.name} does not support structured runtime tools")
    for tool in runtime_tools:
        if not isinstance(tool, StructuredToolSpec):
            raise AnthropicError(f"{profile.name} does not support freeform runtime tools")
        if tool.name in wire_names:
            raise AnthropicError(f"duplicate Anthropic tool wire name: {tool.name}")
        wire_names.add(tool.name)
        encoded.append(
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": thaw_json_value(tool.input_schema),
            }
        )
    for spec in provider_tools:
        declaration = profile.server_tools.encode_declaration(spec)
        name = cast(str, declaration["name"])
        if name in wire_names:
            raise AnthropicError(f"duplicate Anthropic tool wire name: {name}")
        wire_names.add(name)
        encoded.append(declaration)
    return encoded


def encode_tool_choice(
    choice: ToolChoice,
    *,
    runtime_tool_names: Collection[str],
    provider_tools: Sequence[ProviderToolSpec],
    may_return_runtime_tool_calls: bool,
    profile: AnthropicProfile,
) -> JsonObject | None:
    if choice.type not in profile.capabilities.tool_choice_types:
        raise AnthropicError(f"{profile.name} does not support tool_choice={choice.type!r}")
    if not runtime_tool_names and not provider_tools:
        if choice.type in {"required", "runtime", "provider"}:
            raise AnthropicError(f"tool_choice={choice.type!r} requires at least one tool")
        return None
    if choice.type == "auto" and profile.automatic_tool_choice_mode == "implicit":
        return None
    if choice.type == "provider":
        spec = next(
            (item for item in provider_tools if item.tool == choice.provider_tool),
            None,
        )
        if spec is None:
            raise AnthropicError("tool_choice names an unavailable provider tool")
        value = profile.server_tools.encode_choice(spec)
    elif choice.type == "runtime":
        if choice.name is None or choice.name not in runtime_tool_names:
            raise AnthropicError(f"tool_choice names an unavailable tool: {choice.name}")
        value: JsonObject = {"type": "tool", "name": choice.name}
    else:
        value = {
            "type": {
                "auto": "auto",
                "none": "none",
                "required": "any",
            }[choice.type]
        }
    if (
        choice.type != "none"
        and may_return_runtime_tool_calls
        and profile.capabilities.parallel_runtime_tool_call_control
    ):
        value["disable_parallel_tool_use"] = not choice.allow_parallel_runtime_tool_calls
    return value


def encode_assistant_tool_uses(calls: Sequence[StructuredToolCall]) -> list[JsonObject]:
    return [
        {
            "type": "tool_use",
            "id": call.id,
            "name": call.name,
            "input": thaw_json_value(call.arguments),
        }
        for call in calls
    ]


def decode_tool_uses(blocks: Sequence[Mapping[str, Any]]) -> list[StructuredToolCall]:
    return [
        StructuredToolCall(
            id=ANTHROPIC_JSON.required_string(block.get("id"), "Anthropic tool_use id"),
            name=ANTHROPIC_JSON.required_string(block.get("name"), "Anthropic tool_use name"),
            arguments=_decode_input(block.get("input")),
        )
        for block in blocks
    ]


def _decode_input(value: object) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    if not isinstance(value, str):
        raise AnthropicError("Anthropic tool_use input must be an object or JSON string")
    try:
        parsed: object = json.loads(value)
    except json.JSONDecodeError as exc:
        raise AnthropicError("Anthropic tool_use input is invalid JSON") from exc
    if not isinstance(parsed, Mapping):
        raise AnthropicError("Anthropic tool_use input must decode to an object")
    return cast(Mapping[str, Any], parsed)
