"""Tool conversion for OpenAI-compatible Responses APIs."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, cast

from jharness.kernel import (
    ProviderToolId,
    ProviderToolSpec,
    ToolChoice,
    ToolSpec,
    thaw_json_value,
)
from jharness.models.openai.errors import OpenAIResponsesError
from jharness.models.openai.profiles import OpenAIResponsesProfile

JsonObject = dict[str, Any]

_FUNCTION_NAME = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_WEB_SEARCH_VARIANTS = frozenset({"web_search", "web_search_2025_08_26"})


def encode_tools(
    runtime_tools: Sequence[ToolSpec],
    provider_tools: Sequence[ProviderToolSpec],
    profile: OpenAIResponsesProfile,
) -> list[JsonObject]:
    """Encode runtime and provider-owned tools into one Responses list."""

    if runtime_tools and not profile.capabilities.runtime_tools:
        raise OpenAIResponsesError(f"{profile.name} does not support runtime tools")
    encoded = [_encode_runtime_tool(tool) for tool in runtime_tools]
    encoded.extend(_encode_provider_tool(tool, profile) for tool in provider_tools)
    names = [cast(str, tool["name"]) for tool in encoded if tool["type"] == "function"]
    if len(names) != len(set(names)):
        raise OpenAIResponsesError("Responses function tool names must be unique")
    return encoded


def encode_tool_choice(
    choice: ToolChoice,
    *,
    runtime_tools: Sequence[ToolSpec],
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
        return _encode_runtime_tool_choice(choice, runtime_tools)
    return _encode_provider_tool_choice(choice, provider_tools, profile)


def _encode_runtime_tool_choice(
    choice: ToolChoice,
    runtime_tools: Sequence[ToolSpec],
) -> JsonObject:
    names = {tool.name for tool in runtime_tools}
    if choice.name is None or choice.name not in names:
        raise OpenAIResponsesError(f"tool_choice names an unavailable runtime tool: {choice.name}")
    return {"type": "function", "name": choice.name}


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
    return {"type": provider_wire_type(selected, profile)}


def provider_wire_type(
    spec: ProviderToolSpec,
    profile: OpenAIResponsesProfile,
) -> str:
    """Return the validated wire discriminator for one provider tool."""

    _validate_provider_identity(spec.tool, profile)
    if spec.tool.type != "web_search":
        return spec.tool.type
    variant = spec.configuration.get("variant", "web_search")
    if not isinstance(variant, str) or variant not in _WEB_SEARCH_VARIANTS:
        expected = ", ".join(sorted(_WEB_SEARCH_VARIANTS))
        raise OpenAIResponsesError(f"web_search variant must be one of: {expected}")
    return variant


def _encode_runtime_tool(tool: ToolSpec) -> JsonObject:
    if _FUNCTION_NAME.fullmatch(tool.name) is None:
        raise OpenAIResponsesError("Responses function names must match ^[A-Za-z0-9_-]{1,128}$")
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": thaw_json_value(tool.input_schema),
    }


def _encode_provider_tool(
    spec: ProviderToolSpec,
    profile: OpenAIResponsesProfile,
) -> JsonObject:
    wire_type = provider_wire_type(spec, profile)
    configuration = cast(JsonObject, thaw_json_value(spec.configuration))
    allowed = profile.provider_tool_configuration_fields[spec.tool.type]
    unexpected = set(configuration).difference(allowed)
    if unexpected:
        key = min(unexpected)
        raise OpenAIResponsesError(f"unsupported {spec.tool.type} configuration field: {key}")
    configuration.pop("variant", None)
    if spec.tool.type == "image_generation":
        _validate_image_configuration(configuration)
    return {"type": wire_type, **configuration}


def _validate_provider_identity(
    tool: ProviderToolId,
    profile: OpenAIResponsesProfile,
) -> None:
    if tool not in profile.capabilities.provider_tools:
        raise OpenAIResponsesError(f"{profile.name} does not support provider tool: {tool.type}")


def _validate_image_configuration(configuration: Mapping[str, Any]) -> None:
    partial_images = configuration.get("partial_images")
    if partial_images is not None and (
        isinstance(partial_images, bool)
        or not isinstance(partial_images, int)
        or not 0 <= partial_images <= 3
    ):
        raise OpenAIResponsesError("image_generation partial_images must be between 0 and 3")
    output_format = configuration.get("output_format")
    if output_format is not None and output_format not in {"png", "jpeg", "webp"}:
        raise OpenAIResponsesError("image_generation output_format must be png, jpeg, or webp")
    output_compression = configuration.get("output_compression")
    if output_compression is not None and (
        isinstance(output_compression, bool)
        or not isinstance(output_compression, int)
        or not 0 <= output_compression <= 100
    ):
        raise OpenAIResponsesError("image_generation output_compression must be between 0 and 100")
