"""Tool conversion for Anthropic Messages."""

from __future__ import annotations

import re
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
from jharness.models.anthropic.messages.errors import (
    ANTHROPIC_MESSAGES_JSON,
    AnthropicMessagesError,
)
from jharness.models.anthropic.messages.profile import AnthropicMessagesProfile
from jharness.models.anthropic.messages.server_tools import (
    ANTHROPIC_MESSAGES_WEB_SEARCH,
)
from jharness.models.anthropic.messages.server_tools import (
    encode_choice as encode_web_search_choice,
)
from jharness.models.anthropic.messages.server_tools import (
    encode_declaration as encode_web_search_declaration,
)

JsonValue = Any
JsonObject = dict[str, JsonValue]

_TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def validate_tool_name(name: object, label: str = "Anthropic tool name") -> str:
    if not isinstance(name, str) or not _TOOL_NAME_PATTERN.fullmatch(name):
        raise AnthropicMessagesError(f"{label} must match ^[a-zA-Z0-9_-]{{1,64}}$")
    return name


def encode_tools(  # noqa: C901
    runtime_tools: Sequence[RuntimeToolSpec],
    provider_tools: Sequence[ProviderToolSpec],
    profile: AnthropicMessagesProfile,
) -> list[JsonObject]:
    if not runtime_tools and not provider_tools:
        return []
    encoded: list[JsonObject] = []
    wire_names: set[str] = set()
    if runtime_tools and RuntimeToolKind.STRUCTURED not in profile.capabilities.runtime_tool_kinds:
        raise AnthropicMessagesError(f"{profile.name} does not support structured runtime tools")
    for tool in runtime_tools:
        if not isinstance(tool, StructuredToolSpec):
            raise AnthropicMessagesError(f"{profile.name} does not support freeform runtime tools")
        validate_tool_name(tool.name)
        if tool.name in wire_names:
            raise AnthropicMessagesError(f"duplicate Anthropic tool wire name: {tool.name}")
        input_schema = thaw_json_value(tool.input_schema)
        if not isinstance(input_schema, Mapping):
            raise AnthropicMessagesError("Anthropic tool input_schema must be an object")
        wire_names.add(tool.name)
        encoded.append(
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": input_schema,
            }
        )
    for spec in provider_tools:
        if spec.tool not in profile.capabilities.provider_tools:
            raise AnthropicMessagesError(
                f"{profile.name} profile does not support provider tool: "
                f"{spec.tool.namespace}/{spec.tool.type}"
            )
        if spec.tool != ANTHROPIC_MESSAGES_WEB_SEARCH:
            raise AnthropicMessagesError("unsupported Anthropic Messages provider tool")
        declaration = encode_web_search_declaration(spec)
        name = cast(str, declaration["name"])
        if name in wire_names:
            raise AnthropicMessagesError(f"duplicate Anthropic tool wire name: {name}")
        wire_names.add(name)
        encoded.append(declaration)
    return encoded


def encode_tool_choice(
    choice: ToolChoice,
    *,
    runtime_tool_names: Collection[str],
    provider_tools: Sequence[ProviderToolSpec],
    may_return_runtime_tool_calls: bool,
    profile: AnthropicMessagesProfile,
) -> JsonObject | None:
    if choice.type not in profile.capabilities.tool_choice_types:
        raise AnthropicMessagesError(f"{profile.name} does not support tool_choice={choice.type!r}")
    if not runtime_tool_names and not provider_tools:
        if choice.type in {"required", "runtime", "provider"}:
            raise AnthropicMessagesError(f"tool_choice={choice.type!r} requires at least one tool")
        return None
    if choice.type == "provider":
        spec = next(
            (item for item in provider_tools if item.tool == choice.provider_tool),
            None,
        )
        if spec is None:
            raise AnthropicMessagesError("tool_choice names an unavailable provider tool")
        if spec.tool != ANTHROPIC_MESSAGES_WEB_SEARCH:
            raise AnthropicMessagesError("unsupported Anthropic Messages provider tool")
        value = encode_web_search_choice(spec)
    elif choice.type == "runtime":
        if choice.name is None or choice.name not in runtime_tool_names:
            raise AnthropicMessagesError(f"tool_choice names an unavailable tool: {choice.name}")
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
    blocks: list[JsonObject] = []
    for call in calls:
        if call.raw_input is not None:
            raise AnthropicMessagesError(
                "Anthropic tool_use history cannot replay malformed raw input"
            )
        if call.arguments is None:
            raise AnthropicMessagesError("Anthropic tool_use history requires object input")
        blocks.append(
            {
                "type": "tool_use",
                "id": call.id,
                "name": validate_tool_name(call.name, "Anthropic tool_use name"),
                "input": thaw_json_value(call.arguments),
                **_tool_use_metadata(call.metadata, "Anthropic tool_use history"),
            }
        )
    return blocks


def decode_tool_uses(blocks: Sequence[Mapping[str, Any]]) -> list[StructuredToolCall]:
    return [_decode_tool_use(block) for block in blocks]


def _decode_tool_use(block: Mapping[str, Any]) -> StructuredToolCall:
    if block.get("type") != "tool_use":
        raise AnthropicMessagesError("Anthropic tool use requires type='tool_use'")
    if "input" not in block:
        raise AnthropicMessagesError("Anthropic tool_use requires input")
    extras = _tool_use_extras(block, "Anthropic tool_use")
    return StructuredToolCall(
        id=ANTHROPIC_MESSAGES_JSON.required_string(block.get("id"), "Anthropic tool_use id"),
        name=validate_tool_name(
            ANTHROPIC_MESSAGES_JSON.required_string(block.get("name"), "Anthropic tool_use name"),
            "Anthropic tool_use name",
        ),
        arguments=_decode_input(block["input"]),
        metadata={"anthropic": extras} if extras else {},
    )


def _decode_input(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    raise AnthropicMessagesError("Anthropic tool_use input must be an object")


def _tool_use_extras(value: Mapping[str, Any], label: str) -> JsonObject:
    unknown = set(value).difference({"type", "id", "name", "input", "caller", "toolset_name"})
    if unknown:
        raise AnthropicMessagesError(f"{label} has unsupported field: {min(unknown)}")
    extras = {
        key: thaw_json_value(value[key]) for key in ("caller", "toolset_name") if key in value
    }
    _validate_tool_use_extras(extras, label)
    return extras


def _tool_use_metadata(metadata: Mapping[str, Any], label: str) -> JsonObject:
    native = metadata.get("anthropic")
    if native is None:
        return {}
    if not isinstance(native, Mapping):
        raise AnthropicMessagesError(f"{label} metadata must be an object")
    native_mapping = cast(Mapping[str, Any], native)
    extras = {
        key: thaw_json_value(native_mapping[key])
        for key in ("caller", "toolset_name")
        if key in native_mapping
    }
    unknown = set(native_mapping).difference({"caller", "toolset_name"})
    if unknown:
        raise AnthropicMessagesError(f"{label} metadata has unsupported field: {min(unknown)}")
    _validate_tool_use_extras(extras, label)
    return extras


def _validate_tool_use_extras(extras: Mapping[str, Any], label: str) -> None:
    caller = extras.get("caller")
    if caller is not None:
        _validate_caller(caller, f"{label} caller")
    if (
        "toolset_name" in extras
        and extras["toolset_name"] is not None
        and not isinstance(extras["toolset_name"], str)
    ):
        raise AnthropicMessagesError(f"{label} toolset_name must be a string or null")


def _validate_caller(value: object, label: str) -> None:
    if not isinstance(value, Mapping):
        raise AnthropicMessagesError(f"{label} must be an object")
    caller = cast(Mapping[str, object], value)
    caller_type = caller.get("type")
    if caller_type == "direct":
        if set(caller) != {"type"}:
            raise AnthropicMessagesError(f"{label} has unsupported field")
        return
    if caller_type in {
        "code_execution_20250825",
        "code_execution_20260120",
        "code_execution_20260521",
    }:
        if set(caller) != {"type", "tool_id"}:
            raise AnthropicMessagesError(f"{label} has unsupported field")
        tool_id = caller.get("tool_id")
        if not isinstance(tool_id, str) or not tool_id:
            raise AnthropicMessagesError(f"{label}.tool_id must be a non-empty string")
        return
    raise AnthropicMessagesError(f"{label}.type is invalid")
