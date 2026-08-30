"""Request and response codec for Anthropic Messages."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from typing import Any, cast

from jharness.kernel import ModelRequest, ModelResponse, ModelUsage, ResponseFormat, thaw_json_value
from jharness.models.anthropic.messages.errors import (
    ANTHROPIC_MESSAGES_JSON,
    AnthropicMessagesError,
)
from jharness.models.anthropic.messages.messages import (
    decode_content_blocks,
    encode_messages,
)
from jharness.models.anthropic.messages.profile import AnthropicMessagesProfile
from jharness.models.anthropic.messages.tools import (
    encode_tool_choice,
    encode_tools,
)

JsonValue = Any
JsonObject = dict[str, JsonValue]

_STOP_REASONS = frozenset(
    {
        "end_turn",
        "max_tokens",
        "stop_sequence",
        "tool_use",
        "pause_turn",
        "refusal",
        "model_context_window_exceeded",
    }
)

_MAPPING_SUBSCHEMA_KEYWORDS = (
    "$defs",
    "definitions",
    "dependentSchemas",
    "patternProperties",
    "properties",
)
_SEQUENCE_SUBSCHEMA_KEYWORDS = ("allOf", "anyOf", "oneOf", "prefixItems")
_SINGLE_SUBSCHEMA_KEYWORDS = (
    "additionalItems",
    "additionalProperties",
    "contains",
    "contentSchema",
    "else",
    "if",
    "items",
    "not",
    "propertyNames",
    "then",
    "unevaluatedItems",
    "unevaluatedProperties",
)


class AnthropicMessagesCodec:
    """Translate between kernel model DTOs and Anthropic Messages JSON."""

    def __init__(self, *, model: str, profile: AnthropicMessagesProfile | None = None) -> None:
        if not model:
            raise ValueError("model must not be empty")
        self.model = model
        self.profile = profile or AnthropicMessagesProfile()

    def encode_request(self, request: ModelRequest, *, stream: bool = False) -> JsonObject:
        system, messages = encode_messages(request.messages, self.profile)
        payload: JsonObject = {
            "model": request.options.model or self.model,
            "max_tokens": (
                self.profile.default_max_tokens
                if request.options.max_output_tokens is None
                else request.options.max_output_tokens
            ),
            "messages": messages,
        }
        if system is not None:
            payload["system"] = system
        container_id = _continuation_container_id(request)
        if container_id is not None:
            payload["container"] = {"id": container_id}
        self._add_generation_options(payload, request)
        self._add_tool_options(payload, request)
        self._add_output_options(payload, request)
        self._add_stream_option(payload, stream=stream)
        return payload

    def _add_generation_options(self, payload: JsonObject, request: ModelRequest) -> None:
        for field, value in (
            ("temperature", request.options.temperature),
            ("top_p", request.options.top_p),
        ):
            if value is None:
                continue
            if not 0 <= value <= 1:
                raise AnthropicMessagesError(f"Anthropic Messages {field} must be between 0 and 1")
            payload[field] = value
        if request.options.stop:
            payload["stop_sequences"] = list(request.options.stop)
        if request.options.seed is not None:
            raise AnthropicMessagesError(f"{self.profile.name} does not support seed")

    def _add_tool_options(
        self,
        payload: JsonObject,
        request: ModelRequest,
    ) -> None:
        tools = encode_tools(
            request.runtime_tools,
            request.provider_tools,
            self.profile,
        )
        if tools:
            payload["tools"] = tools
        tool_choice = encode_tool_choice(
            request.tool_choice,
            runtime_tool_names={tool.name for tool in request.runtime_tools},
            provider_tools=request.provider_tools,
            may_return_runtime_tool_calls=request.may_return_runtime_tool_calls,
            profile=self.profile,
        )
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

    def _add_output_options(self, payload: JsonObject, request: ModelRequest) -> None:
        output_config = (
            self._encode_response_format(request.response_format)
            if request.response_format is not None
            else {}
        )
        payload.update(self._merge_output_config(output_config))

    def _add_stream_option(self, payload: JsonObject, *, stream: bool) -> None:
        if stream:
            if not self.profile.capabilities.streaming:
                raise AnthropicMessagesError(f"{self.profile.name} does not support streaming")
            payload["stream"] = True

    def decode_response(
        self,
        value: Mapping[str, Any],
    ) -> ModelResponse:
        if "error" in value:
            raise AnthropicMessagesError("Anthropic response must not contain an error envelope")
        reject_unknown_message_fields(value, stream_start=False)
        response_type = value.get("type")
        if response_type != "message":
            raise AnthropicMessagesError("Anthropic response requires type='message'")
        role = value.get("role")
        if role != "assistant":
            raise AnthropicMessagesError("Anthropic response requires role='assistant'")
        if "content" not in value or value["content"] is None:
            raise AnthropicMessagesError("Anthropic response requires content")
        output = decode_content_blocks(value["content"], self.profile)
        stop_reason = validate_stop_reason(value.get("stop_reason"), "Anthropic stop_reason")
        validate_stop_sequence(stop_reason, value.get("stop_sequence"))
        if "usage" not in value:
            raise AnthropicMessagesError("Anthropic response requires usage")
        usage = decode_usage(value.get("usage"))
        metadata: JsonObject = {"provider": self.profile.name}
        metadata["type"] = response_type
        metadata["role"] = role
        container_id = decode_container_id(value.get("container"), "Anthropic response container")
        if container_id is not None:
            metadata["anthropic"] = {"container_id": container_id}
        stop_details = decode_stop_details(value.get("stop_details"))
        if stop_details is not None:
            metadata["stop_details"] = stop_details
        return ModelResponse(
            output=tuple(output),
            finish_reason=stop_reason,
            usage=usage,
            model_id=ANTHROPIC_MESSAGES_JSON.required_string(value.get("model"), "Anthropic model"),
            response_id=ANTHROPIC_MESSAGES_JSON.required_string(value.get("id"), "Anthropic id"),
            provider_turn_pending=(
                stop_reason == "pause_turn"
                or any(getattr(item, "status", None) == "in_progress" for item in output)
            ),
            metadata=metadata,
        )

    def _encode_response_format(self, response_format: ResponseFormat) -> JsonObject:
        if response_format.type == "text":
            return {}
        if response_format.type == "json_object":
            if not self.profile.capabilities.json_mode:
                raise AnthropicMessagesError(
                    f"{self.profile.name} does not support JSON object mode"
                )
            return {
                "format": {
                    "type": "json_schema",
                    "schema": thaw_json_value(self.profile.json_object_schema),
                }
            }
        if response_format.type == "json_schema":
            if not self.profile.capabilities.structured_output:
                raise AnthropicMessagesError(
                    f"{self.profile.name} does not support JSON schema output"
                )
            if response_format.schema is None:
                raise AnthropicMessagesError("JSON schema response format requires schema")
            schema = thaw_json_value(response_format.schema)
            if not isinstance(schema, Mapping):
                raise AnthropicMessagesError(
                    "Anthropic JSON schema response format requires an object"
                )
            if response_format.strict:
                schema = _strict_json_schema(schema)
            return {
                "format": {
                    "type": "json_schema",
                    "schema": schema,
                }
            }
        raise AnthropicMessagesError(f"unsupported response format type: {response_format.type}")

    def _merge_output_config(
        self,
        output_config: JsonObject,
    ) -> JsonObject:
        if not output_config:
            return {}
        return {"output_config": output_config}


def decode_usage(value: object, *, delta: bool = False) -> ModelUsage:
    usage = ANTHROPIC_MESSAGES_JSON.mapping(value, "Anthropic usage")
    allowed = (
        {
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "input_tokens",
            "output_tokens",
            "output_tokens_details",
            "server_tool_use",
        }
        if delta
        else {
            "cache_creation",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "inference_geo",
            "input_tokens",
            "output_tokens",
            "output_tokens_details",
            "server_tool_use",
            "service_tier",
        }
    )
    unknown = set(usage).difference(allowed)
    if unknown:
        raise AnthropicMessagesError(f"Anthropic usage has unsupported field: {min(unknown)}")
    input_tokens = (
        ANTHROPIC_MESSAGES_JSON.optional_integer(usage.get("input_tokens"))
        if delta
        else _required_integer(usage.get("input_tokens"), "Anthropic usage input_tokens")
    )
    output_tokens = _required_integer(usage.get("output_tokens"), "Anthropic usage output_tokens")
    output_details = usage.get("output_tokens_details")
    reasoning_tokens = None
    if isinstance(output_details, Mapping):
        output_details_mapping = cast(Mapping[str, object], output_details)
        reasoning_tokens = ANTHROPIC_MESSAGES_JSON.optional_integer(
            output_details_mapping.get("thinking_tokens")
        )
    cache_read_tokens = ANTHROPIC_MESSAGES_JSON.optional_integer(
        usage.get("cache_read_input_tokens")
    )
    cache_write_tokens = ANTHROPIC_MESSAGES_JSON.optional_integer(
        usage.get("cache_creation_input_tokens")
    )
    total_tokens = None
    if input_tokens is not None:
        total_tokens = input_tokens + output_tokens
    _validate_usage_extensions(usage, delta=delta)
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )


def _continuation_container_id(request: ModelRequest) -> str | None:
    if not request.messages:
        return None
    last = request.messages[-1]
    if last.role != "assistant":
        return None
    native = last.metadata.get("anthropic")
    if native is None:
        return None
    if not isinstance(native, Mapping):
        raise AnthropicMessagesError("Anthropic continuation metadata must be an object")
    metadata = cast(Mapping[str, object], native)
    unknown = set(metadata).difference({"container_id"})
    if unknown:
        raise AnthropicMessagesError(
            f"Anthropic continuation metadata has unsupported field: {min(unknown)}"
        )
    identifier = metadata.get("container_id")
    if not isinstance(identifier, str) or not identifier:
        raise AnthropicMessagesError(
            "Anthropic continuation metadata container_id must be a non-empty string"
        )
    return identifier


def decode_container_id(value: object, label: str) -> str | None:  # noqa: C901
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise AnthropicMessagesError(f"{label} must be an object")
    container = cast(Mapping[str, object], value)
    unknown = set(container).difference({"id", "expires_at", "skills"})
    if unknown:
        raise AnthropicMessagesError(f"{label} has unsupported field: {min(unknown)}")
    identifier = container.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise AnthropicMessagesError(f"{label} id must be a non-empty string")
    if not isinstance(container.get("expires_at"), str) or not container["expires_at"]:
        raise AnthropicMessagesError(f"{label} expires_at must be a non-empty string")
    skills = container.get("skills")
    if skills is not None:
        if not isinstance(skills, Sequence) or isinstance(skills, str | bytes | bytearray):
            raise AnthropicMessagesError(f"{label} skills must be an array")
        for skill in cast(Sequence[object], skills):
            mapping = ANTHROPIC_MESSAGES_JSON.mapping(skill, f"{label} skill")
            unknown_skill = set(mapping).difference({"skill_id", "type", "version"})
            if unknown_skill:
                raise AnthropicMessagesError(
                    f"{label} skill has unsupported field: {min(unknown_skill)}"
                )
            if mapping.get("type") not in {"anthropic", "custom"}:
                raise AnthropicMessagesError(f"{label} skill type is invalid")
            for field in ("skill_id", "version"):
                ANTHROPIC_MESSAGES_JSON.required_string(
                    mapping.get(field), f"{label} skill {field}"
                )
    return identifier


def _validate_usage_extensions(  # noqa: C901
    usage: Mapping[str, object], *, delta: bool
) -> None:
    details = usage.get("output_tokens_details")
    if details is not None:
        mapping = ANTHROPIC_MESSAGES_JSON.mapping(details, "Anthropic usage output_tokens_details")
        if set(mapping) != {"thinking_tokens"}:
            raise AnthropicMessagesError("Anthropic usage output_tokens_details is invalid")
        _required_integer(mapping.get("thinking_tokens"), "Anthropic usage thinking_tokens")
    server = usage.get("server_tool_use")
    if server is not None:
        mapping = ANTHROPIC_MESSAGES_JSON.mapping(server, "Anthropic usage server_tool_use")
        if set(mapping) != {"web_search_requests"}:
            raise AnthropicMessagesError("Anthropic usage server_tool_use is invalid")
        _required_integer(mapping.get("web_search_requests"), "Anthropic usage web_search_requests")
    if not delta:
        cache_creation = usage.get("cache_creation")
        if cache_creation is not None:
            mapping = ANTHROPIC_MESSAGES_JSON.mapping(
                cache_creation, "Anthropic usage cache_creation"
            )
            if set(mapping) != {"ephemeral_1h_input_tokens", "ephemeral_5m_input_tokens"}:
                raise AnthropicMessagesError("Anthropic usage cache_creation is invalid")
            for field in ("ephemeral_1h_input_tokens", "ephemeral_5m_input_tokens"):
                _required_integer(mapping.get(field), f"Anthropic usage {field}")
        tier = usage.get("service_tier")
        if tier is not None and tier not in {"standard", "priority", "batch"}:
            raise AnthropicMessagesError("Anthropic usage service_tier is invalid")
        geo = usage.get("inference_geo")
        if geo is not None and not isinstance(geo, str):
            raise AnthropicMessagesError("Anthropic usage inference_geo must be a string")


def _required_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AnthropicMessagesError(f"{label} must be a non-negative integer")
    return value


def validate_stop_reason(value: object, label: str) -> str:
    reason = ANTHROPIC_MESSAGES_JSON.required_string(value, label)
    if reason not in _STOP_REASONS:
        raise AnthropicMessagesError(f"{label} is invalid")
    return reason


def validate_stop_sequence(stop_reason: str, value: object) -> None:
    if stop_reason == "stop_sequence":
        if not isinstance(value, str) or not value:
            raise AnthropicMessagesError(
                "Anthropic stop_sequence reason requires a non-empty stop_sequence"
            )
        return
    if value is not None:
        raise AnthropicMessagesError(
            "Anthropic stop_sequence must be null unless stop_reason is stop_sequence"
        )


def reject_unknown_message_fields(value: Mapping[str, object], *, stream_start: bool) -> None:
    allowed = {
        "id",
        "container",
        "content",
        "model",
        "role",
        "stop_details",
        "stop_reason",
        "stop_sequence",
        "type",
        "usage",
    }
    unknown = set(value).difference(allowed)
    if unknown:
        label = "stream message" if stream_start else "response"
        raise AnthropicMessagesError(f"Anthropic {label} has unsupported field: {min(unknown)}")


def decode_stop_details(value: object) -> JsonObject | None:
    if value is None:
        return None
    details = ANTHROPIC_MESSAGES_JSON.mapping(value, "Anthropic stop_details")
    unknown = set(details).difference({"type", "category", "explanation"})
    if unknown or details.get("type") != "refusal":
        raise AnthropicMessagesError("Anthropic stop_details is invalid")
    category = details.get("category")
    if category is not None and category not in {
        "cyber",
        "bio",
        "frontier_llm",
        "reasoning_extraction",
        "general_harms",
    }:
        raise AnthropicMessagesError("Anthropic stop_details category is invalid")
    explanation = details.get("explanation")
    if explanation is not None and not isinstance(explanation, str):
        raise AnthropicMessagesError("Anthropic stop_details explanation must be a string")
    return dict(details)


def _strict_json_schema(schema: Mapping[str, Any]) -> JsonObject:
    value = deepcopy(dict(schema))
    _apply_strict_json_schema(value)
    return value


def _apply_strict_json_schema(value: object) -> None:
    pending = [value]
    while pending:
        current = pending.pop()
        if not isinstance(current, Mapping):
            continue
        mapping = cast(dict[str, Any], current)
        _enforce_strict_object_schema(mapping)
        pending.extend(reversed(tuple(_subschemas(mapping))))


def _subschemas(schema: Mapping[str, object]) -> Iterator[object]:
    for keyword in _MAPPING_SUBSCHEMA_KEYWORDS:
        value = schema.get(keyword)
        if isinstance(value, Mapping):
            yield from cast(Mapping[object, object], value).values()
    dependencies = schema.get("dependencies")
    if isinstance(dependencies, Mapping):
        for child in cast(Mapping[object, object], dependencies).values():
            if isinstance(child, Mapping):
                yield cast(object, child)
    for keyword in _SEQUENCE_SUBSCHEMA_KEYWORDS:
        value = schema.get(keyword)
        if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            yield from cast(Sequence[object], value)
    for keyword in _SINGLE_SUBSCHEMA_KEYWORDS:
        child = schema.get(keyword)
        if isinstance(child, Mapping):
            yield cast(object, child)


def _enforce_strict_object_schema(schema: JsonObject) -> None:
    schema_type = schema.get("type")
    is_object = schema_type == "object" or (
        isinstance(schema_type, list) and "object" in schema_type
    )
    if not is_object and "properties" not in schema:
        return
    additional = schema.get("additionalProperties")
    if additional is not None and additional is not False:
        raise AnthropicMessagesError("strict JSON schema object additionalProperties must be false")
    schema["additionalProperties"] = False
