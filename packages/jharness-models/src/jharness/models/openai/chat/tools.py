"""Tool conversion for OpenAI Chat Completions."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

from jharness.kernel import (
    FreeformToolCall,
    RuntimeToolCall,
    RuntimeToolKind,
    RuntimeToolSpec,
    StructuredToolCall,
    StructuredToolSpec,
    ToolChoice,
    thaw_json_value,
)
from jharness.models.openai.chat.errors import OPENAI_CHAT_JSON, OpenAIChatError
from jharness.models.openai.chat.profile import OpenAIChatProfile

JsonValue = Any
JsonObject = dict[str, JsonValue]
_FUNCTION_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def encode_tools(
    tools: Sequence[RuntimeToolSpec],
    profile: OpenAIChatProfile,
) -> list[JsonObject]:
    if not tools:
        return []
    encoded: list[JsonObject] = []
    for tool in tools:
        if isinstance(tool, StructuredToolSpec):
            if RuntimeToolKind.STRUCTURED not in profile.capabilities.runtime_tool_kinds:
                raise OpenAIChatError(f"{profile.name} does not support structured tools")
            if not isinstance(tool.input_schema, Mapping):
                raise OpenAIChatError("chat completion function parameters must be a JSON object")
            _validate_function_tool_name(tool.name)
            encoded.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": thaw_json_value(tool.input_schema),
                    },
                }
            )
            continue
        if RuntimeToolKind.FREEFORM not in profile.capabilities.runtime_tool_kinds:
            raise OpenAIChatError(f"{profile.name} does not support custom tools")
        encoded.append(
            {
                "type": "custom",
                "custom": {"name": tool.name, "description": tool.description},
            }
        )
    return encoded


def encode_tool_choice(
    choice: ToolChoice,
    *,
    tools_by_name: Mapping[str, RuntimeToolSpec],
    profile: OpenAIChatProfile,
) -> str | JsonObject | None:
    if choice.type not in profile.capabilities.tool_choice_types:
        raise OpenAIChatError(f"{profile.name} does not support tool_choice={choice.type!r}")
    if not tools_by_name:
        if choice.type in {"required", "runtime", "provider"}:
            raise OpenAIChatError(f"tool_choice={choice.type!r} requires at least one tool")
        return None
    if choice.type in {"auto", "none", "required"}:
        return choice.type
    if choice.type == "provider":
        raise OpenAIChatError("Chat Completions does not support provider tool choice")
    if choice.name is None or choice.name not in tools_by_name:
        raise OpenAIChatError(f"tool_choice names an unavailable tool: {choice.name}")
    tool_type = _tool_type_for_spec(tools_by_name[choice.name])
    if tool_type == "function":
        return {"type": "function", "function": {"name": choice.name}}
    return {"type": "custom", "custom": {"name": choice.name}}


def encode_assistant_tool_calls(calls: Sequence[RuntimeToolCall]) -> list[JsonObject]:
    encoded: list[JsonObject] = []
    for call in calls:
        if isinstance(call, StructuredToolCall):
            _validate_function_tool_name(call.name)
            arguments = (
                call.raw_input
                if call.raw_input is not None
                else json.dumps(
                    thaw_json_value(call.arguments),
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            encoded.append(
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": arguments,
                    },
                }
            )
        else:
            encoded.append(
                {
                    "id": call.id,
                    "type": "custom",
                    "custom": {"name": call.name, "input": call.input},
                }
            )
    return encoded


def decode_tool_calls(value: object) -> list[RuntimeToolCall]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise OpenAIChatError("chat completion tool_calls must be an array")
    calls: list[RuntimeToolCall] = []
    for item in cast(Sequence[object], value):
        mapping = OPENAI_CHAT_JSON.mapping(item, "chat completion tool call")
        _require_exact_fields(
            mapping,
            {"id", "type", "function", "custom"},
            "chat completion tool call",
        )
        call_type = mapping.get("type")
        if not isinstance(call_type, str) or not call_type:
            raise OpenAIChatError("chat completion tool call requires non-empty type")
        if call_type == "custom":
            _require_exact_fields(
                mapping, {"id", "type", "custom"}, "chat completion custom tool call"
            )
            custom = OPENAI_CHAT_JSON.mapping(mapping.get("custom"), "chat completion custom tool")
            _require_exact_fields(custom, {"name", "input"}, "chat completion custom tool")
            calls.append(
                FreeformToolCall(
                    id=OPENAI_CHAT_JSON.required_string(
                        mapping.get("id"), "chat completion tool call id"
                    ),
                    name=OPENAI_CHAT_JSON.required_string(
                        custom.get("name"), "chat completion custom tool name"
                    ),
                    input=_custom_input(custom.get("input")),
                )
            )
            continue
        if call_type != "function":
            raise OpenAIChatError(f"unsupported chat completion tool call type: {call_type}")
        _require_exact_fields(
            mapping, {"id", "type", "function"}, "chat completion function tool call"
        )
        function = OPENAI_CHAT_JSON.mapping(
            mapping.get("function"), "chat completion tool function"
        )
        _require_exact_fields(function, {"name", "arguments"}, "chat completion tool function")
        arguments, raw_input = _decode_arguments(function.get("arguments"))
        calls.append(
            StructuredToolCall(
                id=OPENAI_CHAT_JSON.required_string(
                    mapping.get("id"), "chat completion tool call id"
                ),
                name=OPENAI_CHAT_JSON.required_string(
                    function.get("name"),
                    "chat completion tool function name",
                ),
                arguments=arguments,
                raw_input=raw_input,
            )
        )
    return calls


def _tool_type_for_spec(tool: RuntimeToolSpec) -> str:
    if isinstance(tool, StructuredToolSpec):
        return "function"
    return "custom"


def _validate_function_tool_name(name: str) -> None:
    if _FUNCTION_TOOL_NAME.fullmatch(name) is None:
        raise OpenAIChatError(
            "Chat Completions function tool names must match ^[A-Za-z0-9_-]{1,64}$"
        )


def _decode_arguments(value: object) -> tuple[Mapping[str, Any] | None, str | None]:
    if not isinstance(value, str):
        raise OpenAIChatError("chat completion tool function arguments must be a string")
    try:
        parsed: object = json.loads(value)
    except json.JSONDecodeError:
        return None, value
    if not isinstance(parsed, Mapping):
        return None, value
    return cast(Mapping[str, Any], parsed), None


def _require_exact_fields(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unexpected = set(value) - allowed
    if unexpected:
        raise OpenAIChatError(f"{label} has unsupported fields: {', '.join(sorted(unexpected))}")


def _custom_input(value: object) -> str:
    if not isinstance(value, str):
        raise OpenAIChatError("chat completion custom tool input must be a string")
    return value
