"""Closed implementation of Anthropic Messages hosted web search.

There is intentionally no provider-extension registry: supported hosted tools
are protocol-owned code with explicit wire validation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from jharness.kernel import (
    ContentPart,
    ErrorInfo,
    ProviderToolCall,
    ProviderToolId,
    ProviderToolSpec,
    ProviderToolStatus,
    thaw_json_value,
)
from jharness.models.anthropic.messages.errors import (
    ANTHROPIC_MESSAGES_JSON,
    AnthropicMessagesError,
)

JsonObject = dict[str, Any]
ANTHROPIC_MESSAGES_WEB_SEARCH = ProviderToolId("anthropic.messages", "web_search")
_RESULT_PART_TYPE = "anthropic_web_search_result"
_VARIANTS = frozenset({"web_search_20250305", "web_search_20260209", "web_search_20260318"})
_CONFIG_FIELDS = frozenset(
    {
        "allowed_domains",
        "blocked_domains",
        "cache_control",
        "max_uses",
        "response_inclusion",
        "strict",
        "user_location",
        "variant",
    }
)


def encode_declaration(spec: ProviderToolSpec) -> JsonObject:
    _spec(spec)
    config = cast(JsonObject, thaw_json_value(spec.configuration))
    unknown = set(config).difference(_CONFIG_FIELDS)
    if unknown:
        raise AnthropicMessagesError(f"unsupported web_search configuration field: {min(unknown)}")
    variant = config.pop("variant", "web_search_20250305")
    if not isinstance(variant, str) or variant not in _VARIANTS:
        raise AnthropicMessagesError("web_search variant must be an official supported version")
    if variant != "web_search_20260318" and "response_inclusion" in config:
        raise AnthropicMessagesError(
            f"unsupported {variant} configuration field: response_inclusion"
        )
    _config(variant, config)
    return {"type": variant, "name": "web_search", **config}


def encode_choice(spec: ProviderToolSpec) -> JsonObject:
    _spec(spec)
    return {"type": "tool", "name": "web_search"}


def is_result_type(block_type: str) -> bool:
    return block_type == "web_search_tool_result"


def decode_call(
    use: Mapping[str, Any] | None, result: Mapping[str, Any] | None
) -> ProviderToolCall:
    if use is None and result is None:
        raise ValueError("Anthropic web_search decoding requires a use or result block")
    use_id = _use_id(use) if use is not None else None
    result_id = _result_id(result) if result is not None else None
    if use_id is not None and result_id is not None and use_id != result_id:
        raise AnthropicMessagesError("Anthropic web_search result references a different use id")
    arguments: Mapping[str, Any] = {}
    metadata: JsonObject = {"server_tool_use": use is not None}
    if use is not None:
        _use(use)
        arguments = cast(Mapping[str, Any], use["input"])
        metadata["name"] = "web_search"
        if "caller" in use:
            metadata["caller"] = thaw_json_value(use["caller"])
    output: tuple[ContentPart, ...] = ()
    error: ErrorInfo | None = None
    status = ProviderToolStatus.IN_PROGRESS
    if result is not None:
        _result(result)
        content = result["content"]
        error = _error(content)
        status = ProviderToolStatus.FAILED if error is not None else ProviderToolStatus.COMPLETED
        native: JsonObject = {"type": "web_search_tool_result", "content": thaw_json_value(content)}
        if "caller" in result:
            native["caller"] = thaw_json_value(result["caller"])
        output = (ContentPart(type=_RESULT_PART_TYPE, data={"anthropic": native}),)
    return ProviderToolCall(
        id=cast(str, use_id or result_id),
        tool=ANTHROPIC_MESSAGES_WEB_SEARCH,
        status=status,
        arguments=arguments,
        output=output,
        error=error,
        metadata={"anthropic": metadata},
    )


def encode_history(call: ProviderToolCall) -> list[JsonObject]:
    if call.tool != ANTHROPIC_MESSAGES_WEB_SEARCH:
        raise AnthropicMessagesError("Anthropic Messages does not support this provider tool")
    native = call.metadata.get("anthropic")
    if not isinstance(native, Mapping):
        raise AnthropicMessagesError("Anthropic web_search history requires native metadata")
    native_mapping = cast(Mapping[str, Any], native)
    blocks: list[JsonObject] = []
    if native_mapping.get("server_tool_use") is True:
        use: JsonObject = {
            "type": "server_tool_use",
            "id": call.id,
            "name": "web_search",
            "input": thaw_json_value(call.arguments),
        }
        if "caller" in native_mapping:
            use["caller"] = thaw_json_value(native_mapping["caller"])
        _use(use)
        blocks.append(use)
    if call.output:
        if len(call.output) != 1 or call.output[0].type != _RESULT_PART_TYPE:
            raise AnthropicMessagesError(
                "Anthropic web_search history requires exactly one native result part"
            )
        raw = call.output[0].data.get("anthropic")
        if not isinstance(raw, Mapping):
            raise AnthropicMessagesError("Anthropic web_search result part requires native data")
        data = cast(JsonObject, thaw_json_value(cast(Mapping[str, Any], raw)))
        result: JsonObject = {
            "type": data.pop("type", None),
            "tool_use_id": call.id,
            "content": data.pop("content", None),
            **data,
        }
        _result(result)
        blocks.append(result)
    if not blocks:
        raise AnthropicMessagesError("Anthropic web_search history contains no native blocks")
    return blocks


def _spec(spec: ProviderToolSpec) -> None:
    if spec.tool != ANTHROPIC_MESSAGES_WEB_SEARCH:
        raise AnthropicMessagesError("Anthropic Messages does not support this provider tool")


def _use_id(value: Mapping[str, Any]) -> str:
    return ANTHROPIC_MESSAGES_JSON.required_string(value.get("id"), "Anthropic web_search use id")


def _result_id(value: Mapping[str, Any]) -> str:
    return ANTHROPIC_MESSAGES_JSON.required_string(
        value.get("tool_use_id"), "Anthropic web_search result tool_use_id"
    )


def _use(value: Mapping[str, Any]) -> None:
    _exact(value, {"type", "id", "name", "input"}, "Anthropic web_search use", {"caller"})
    if value.get("type") != "server_tool_use" or value.get("name") != "web_search":
        raise AnthropicMessagesError("Anthropic server tool use must be web_search")
    _use_id(value)
    if not isinstance(value.get("input"), Mapping):
        raise AnthropicMessagesError("Anthropic server tool input must be an object")
    if "caller" in value:
        _caller(value["caller"])


def _result(value: Mapping[str, Any]) -> None:
    _exact(value, {"type", "tool_use_id", "content"}, "Anthropic web_search result", {"caller"})
    if value.get("type") != "web_search_tool_result":
        raise AnthropicMessagesError("Anthropic server tool result must be web_search_tool_result")
    _result_id(value)
    _result_content(value["content"])
    if "caller" in value:
        _caller(value["caller"])


def _config(variant: str, value: Mapping[str, Any]) -> None:
    allowed, blocked = value.get("allowed_domains"), value.get("blocked_domains")
    if allowed is not None and blocked is not None:
        raise AnthropicMessagesError(
            "web_search allowed_domains and blocked_domains are mutually exclusive"
        )
    for key in ("allowed_domains", "blocked_domains"):
        _strings(key, value.get(key))
    max_uses = value.get("max_uses")
    if max_uses is not None and (
        not isinstance(max_uses, int) or isinstance(max_uses, bool) or max_uses < 1
    ):
        raise AnthropicMessagesError("web_search max_uses must be a positive integer")
    for key in ("strict",):
        if key in value and not isinstance(value[key], bool):
            raise AnthropicMessagesError(f"web_search {key} must be a boolean")
    if value.get("response_inclusion") is not None and (
        variant != "web_search_20260318" or value["response_inclusion"] not in {"full", "excluded"}
    ):
        raise AnthropicMessagesError("web_search response_inclusion must be 'full' or 'excluded'")
    _location(value.get("user_location"))
    _cache(value.get("cache_control"))


def _strings(label: str, value: object) -> None:
    if value is not None and (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes | bytearray)
        or any(not isinstance(x, str) or not x for x in cast(Sequence[object], value))
    ):
        raise AnthropicMessagesError(f"web_search {label} must be an array of non-empty strings")


def _location(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise AnthropicMessagesError("web_search user_location.type must be 'approximate'")
    location = cast(Mapping[str, Any], value)
    if location.get("type") != "approximate":
        raise AnthropicMessagesError("web_search user_location.type must be 'approximate'")
    unknown = set(location).difference({"type", "city", "region", "country", "timezone"})
    if unknown:
        raise AnthropicMessagesError(f"unsupported web_search user_location field: {min(unknown)}")
    fields = set(location).intersection({"city", "region", "country", "timezone"})
    if not fields:
        raise AnthropicMessagesError(
            "web_search user_location requires city, region, country, or timezone"
        )
    if any(not isinstance(location[x], str) or not location[x] for x in fields):
        raise AnthropicMessagesError("web_search user_location values must be non-empty strings")
    country = location.get("country")
    if country is not None and (
        not isinstance(country, str) or len(country) != 2 or not country.isalpha()
    ):
        raise AnthropicMessagesError(
            "web_search user_location.country must be a two-letter ISO country code"
        )


def _cache(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise AnthropicMessagesError("web_search cache_control is invalid")
    cache = cast(Mapping[str, Any], value)
    if (
        set(cache).difference({"type", "ttl"})
        or cache.get("type") != "ephemeral"
        or (cache.get("ttl") is not None and cache.get("ttl") not in {"5m", "1h"})
    ):
        raise AnthropicMessagesError("web_search cache_control is invalid")


def _caller(value: object) -> None:
    if not isinstance(value, Mapping):
        raise AnthropicMessagesError("Anthropic web_search caller must be an object")
    caller = cast(Mapping[str, Any], value)
    _exact(caller, {"type"}, "Anthropic web_search direct caller")
    if caller.get("type") != "direct":
        raise AnthropicMessagesError("Anthropic web_search supports only direct callers")


def _result_content(value: object) -> None:
    if isinstance(value, Mapping):
        if _error(cast(Mapping[str, Any], value)) is None:
            raise AnthropicMessagesError(
                "Anthropic web_search result error must be web_search_tool_result_error"
            )
        return
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise AnthropicMessagesError(
            "Anthropic web_search result content must be an error object or array"
        )
    for item in cast(Sequence[object], value):
        if not isinstance(item, Mapping):
            raise AnthropicMessagesError("Anthropic web_search results must contain objects")
        result = cast(Mapping[str, Any], item)
        _exact(
            result,
            {"type", "encrypted_content", "title", "url"},
            "Anthropic web_search result",
            {"page_age"},
        )
        if result.get("type") != "web_search_result" or any(
            not isinstance(result.get(k), str) or not result[k]
            for k in ("encrypted_content", "title", "url")
        ):
            raise AnthropicMessagesError("Anthropic web_search result is invalid")
        if result.get("page_age") is not None and not isinstance(result["page_age"], str):
            raise AnthropicMessagesError(
                "Anthropic web_search result.page_age must be a string or null"
            )


def _error(value: object) -> ErrorInfo | None:
    if not isinstance(value, Mapping):
        return None
    error = cast(Mapping[str, Any], value)
    if error.get("type") != "web_search_tool_result_error":
        return None
    _exact(error, {"type", "error_code"}, "Anthropic web_search result error")
    code = error.get("error_code")
    if code not in {
        "invalid_tool_input",
        "max_uses_exceeded",
        "query_too_long",
        "request_too_large",
        "too_many_requests",
        "unavailable",
    }:
        raise AnthropicMessagesError("Anthropic web_search error_code is invalid")
    return ErrorInfo(code=f"web_search.{code}", message=cast(str, code))


def _exact(
    value: Mapping[str, Any], required: set[str], label: str, optional: set[str] | None = None
) -> None:
    unknown = set(value).difference(required | (optional or set()))
    if unknown:
        raise AnthropicMessagesError(f"{label} has unsupported field: {min(unknown)}")
    missing = required.difference(value)
    if missing:
        raise AnthropicMessagesError(f"{label} requires field: {min(missing)}")
