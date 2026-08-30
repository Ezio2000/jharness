"""Request and response codec for OpenAI Chat Completions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

from jharness.kernel import (
    ContentPart,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ResponseFormat,
    RuntimeToolCall,
    RuntimeToolKind,
    thaw_json_value,
)
from jharness.models.openai.chat.errors import OPENAI_CHAT_JSON, OpenAIChatError
from jharness.models.openai.chat.messages import (
    decode_message_content,
    decode_message_refusal,
    encode_chat_message,
)
from jharness.models.openai.chat.profile import OpenAIChatProfile
from jharness.models.openai.chat.tools import (
    decode_tool_calls,
    encode_tool_choice,
    encode_tools,
)

JsonValue = Any
JsonObject = dict[str, JsonValue]
_FINISH_REASONS = frozenset({"stop", "length", "tool_calls", "content_filter", "function_call"})
_RESPONSE_FIELDS = {
    "id",
    "choices",
    "created",
    "model",
    "object",
    "metadata",
    "moderation",
    "service_tier",
    "system_fingerprint",
    "usage",
}
_SERVICE_TIERS = frozenset({"auto", "default", "fast", "flex", "priority", "scale"})
_USAGE_FIELDS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "prompt_tokens_details",
    "completion_tokens_details",
    "compute_units",
}
_PROMPT_DETAILS_FIELDS = {
    "audio_tokens",
    "cache_write_tokens",
    "cached_tokens",
    "image_tokens",
    "text_tokens",
}
_COMPLETION_DETAILS_FIELDS = {
    "accepted_prediction_tokens",
    "audio_tokens",
    "reasoning_tokens",
    "rejected_prediction_tokens",
    "text_tokens",
}


class OpenAIChatCodec:
    """Translate between kernel model DTOs and Chat Completions JSON."""

    def __init__(
        self,
        *,
        model: str,
        profile: OpenAIChatProfile | None = None,
    ) -> None:
        if not model:
            raise ValueError("model must not be empty")
        self.model = model
        self.profile = profile or OpenAIChatProfile()

    def encode_request(self, request: ModelRequest, *, stream: bool = False) -> JsonObject:
        if request.provider_tools:
            raise OpenAIChatError("Chat Completions does not support provider tool declarations")
        if (
            stream
            and RuntimeToolKind.FREEFORM in request.runtime_tool_kinds
            and request.may_return_runtime_tool_calls
        ):
            raise OpenAIChatError(
                "Chat Completions streaming does not support custom runtime tool calls"
            )
        tools = encode_tools(request.runtime_tools, self.profile)
        payload: JsonObject = {
            "model": request.options.model or self.model,
            "messages": [
                encode_chat_message(message, self.profile) for message in request.messages
            ],
        }
        self._add_model_options(payload, request)
        self._add_tool_options(payload, request, tools)
        self._add_response_format(payload, request)
        self._add_stream_options(payload, stream=stream)
        return payload

    def _add_model_options(self, payload: JsonObject, request: ModelRequest) -> None:
        if request.options.temperature is not None:
            _validate_number_in_range(request.options.temperature, "temperature", 0, 2)
            payload["temperature"] = request.options.temperature
        if request.options.top_p is not None:
            _validate_number_in_range(request.options.top_p, "top_p", 0, 1)
            payload["top_p"] = request.options.top_p
        if request.options.max_output_tokens is not None:
            payload["max_completion_tokens"] = request.options.max_output_tokens
        if request.options.stop:
            if len(request.options.stop) > 4:
                raise OpenAIChatError("Chat Completions stop supports at most 4 sequences")
            payload["stop"] = list(request.options.stop)
        if request.options.seed is not None:
            if not self.profile.capabilities.seed:
                raise OpenAIChatError(f"{self.profile.name} does not support seed")
            payload["seed"] = request.options.seed

    def _add_tool_options(
        self,
        payload: JsonObject,
        request: ModelRequest,
        tools: list[JsonObject],
    ) -> None:
        if tools:
            payload["tools"] = tools
        tool_choice = encode_tool_choice(
            request.tool_choice,
            tools_by_name={tool.name: tool for tool in request.runtime_tools},
            profile=self.profile,
        )
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if (
            tools
            and request.tool_choice.type != "none"
            and request.may_return_runtime_tool_calls
            and self.profile.capabilities.parallel_runtime_tool_call_control
        ):
            payload["parallel_tool_calls"] = request.tool_choice.allow_parallel_runtime_tool_calls

    def _add_response_format(self, payload: JsonObject, request: ModelRequest) -> None:
        if request.response_format is not None:
            payload["response_format"] = self._encode_response_format(request.response_format)

    def _add_stream_options(self, payload: JsonObject, *, stream: bool) -> None:
        if stream:
            if not self.profile.capabilities.streaming:
                raise OpenAIChatError(f"{self.profile.name} does not support streaming")
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}

    def decode_response(
        self,
        value: Mapping[str, Any],
    ) -> ModelResponse:
        _validate_response_envelope(value)
        choice, message = _decode_assistant_choice(value)
        finish_reason = _finish_reason(choice.get("finish_reason"), "chat completion finish_reason")
        reject_logprobs(choice.get("logprobs"), "chat completion choice logprobs")
        parts, tool_calls = _decode_assistant_payload(message)
        metadata = _response_metadata(value, self.profile.name)
        if not parts and not tool_calls:
            metadata["openai_chat"] = {"content_null": True}
        return ModelResponse(
            output=tuple([*parts, *tool_calls]),
            finish_reason=finish_reason,
            usage=decode_usage(value.get("usage")),
            model_id=OPENAI_CHAT_JSON.required_string(value.get("model"), "chat completion model"),
            response_id=OPENAI_CHAT_JSON.required_string(value.get("id"), "chat completion id"),
            metadata=metadata,
        )

    def _encode_response_format(self, response_format: ResponseFormat) -> JsonObject:
        if response_format.type == "text":
            return {"type": "text"}
        if response_format.type == "json_object":
            if not self.profile.capabilities.json_mode:
                raise OpenAIChatError(f"{self.profile.name} does not support JSON object mode")
            return {"type": "json_object"}
        if response_format.type == "json_schema":
            if not self.profile.capabilities.structured_output:
                raise OpenAIChatError(f"{self.profile.name} does not support JSON schema output")
            if response_format.schema is None:
                raise OpenAIChatError("JSON schema response format requires schema")
            if not isinstance(response_format.schema, Mapping):
                raise OpenAIChatError(
                    "Chat Completions JSON schema response schema must be an object"
                )
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": self.profile.json_schema_name,
                    "schema": thaw_json_value(response_format.schema),
                    "strict": response_format.strict,
                },
            }
        raise OpenAIChatError(f"unsupported response format type: {response_format.type}")


def _decode_assistant_choice(
    value: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if "error" in value:
        raise OpenAIChatError("chat completion response must not contain an error envelope")
    raw_choices = value.get("choices")
    if not isinstance(raw_choices, list):
        raise OpenAIChatError("chat completion response requires exactly one choice")
    choices = cast(list[object], raw_choices)
    if len(choices) != 1:
        raise OpenAIChatError("chat completion response requires exactly one choice")
    choice = OPENAI_CHAT_JSON.mapping(choices[0], "chat completion choice")
    _require_fields(
        choice,
        {"index", "message", "finish_reason", "logprobs"},
        "chat completion choice",
    )
    if _choice_index(choice) != 0:
        raise OpenAIChatError("chat completion response choice index must be 0")
    message = OPENAI_CHAT_JSON.mapping(choice.get("message"), "chat completion choice message")
    _require_fields(
        message,
        {
            "role",
            "content",
            "refusal",
            "annotations",
            "tool_calls",
            "audio",
            "function_call",
            "reasoning_content",
        },
        "chat completion choice message",
    )
    if message.get("role") != "assistant":
        raise OpenAIChatError("chat completion response requires role='assistant'")
    return choice, message


def _decode_assistant_payload(
    message: Mapping[str, Any],
) -> tuple[list[ContentPart], list[RuntimeToolCall]]:
    if "reasoning_content" in message:
        raise OpenAIChatError(
            "Chat Completions assistant messages do not support reasoning_content"
        )
    if message.get("audio") is not None or message.get("function_call") is not None:
        raise OpenAIChatError("unsupported standard Chat Completions assistant message field")
    parts = decode_message_content(message.get("content"), message.get("annotations"))
    refusal_parts = decode_message_refusal(message.get("refusal"))
    if refusal_parts and any(part.type == "refusal" for part in parts):
        raise OpenAIChatError("chat completion response must not duplicate refusal content")
    parts.extend(refusal_parts)
    tool_calls = decode_tool_calls(message.get("tool_calls"))
    return parts, tool_calls


def _response_metadata(value: Mapping[str, Any], provider: str) -> JsonObject:
    metadata: JsonObject = {"provider": provider, "choice_count": 1}
    metadata["object"] = "chat.completion"
    created = value["created"]
    if not isinstance(created, int) or isinstance(created, bool):
        raise OpenAIChatError("chat completion created must be an integer")
    metadata["created"] = created
    metadata["service_tier"] = optional_service_tier(value.get("service_tier"), "service_tier")
    metadata["system_fingerprint"] = optional_string(
        value.get("system_fingerprint"), "chat completion system_fingerprint"
    )
    metadata["chat_completion_metadata"] = _optional_string_mapping(
        value.get("metadata"), "chat completion metadata"
    )
    return metadata


def _validate_response_envelope(value: Mapping[str, Any]) -> None:
    _require_exact_fields(value, _RESPONSE_FIELDS, "chat completion response")
    if value.get("object") != "chat.completion":
        raise OpenAIChatError("chat completion object must be 'chat.completion'")
    for field in ("id", "model"):
        OPENAI_CHAT_JSON.required_string(value.get(field), f"chat completion {field}")
    created = value.get("created")
    if isinstance(created, bool) or not isinstance(created, int):
        raise OpenAIChatError("chat completion created must be an integer")
    if "choices" not in value:
        raise OpenAIChatError("chat completion response requires choices")
    if value.get("moderation") is not None:
        raise OpenAIChatError("chat completion moderation is not supported")
    optional_service_tier(value.get("service_tier"), "service_tier")
    optional_string(value.get("system_fingerprint"), "chat completion system_fingerprint")
    _optional_string_mapping(value.get("metadata"), "chat completion metadata")


def _finish_reason(value: object, label: str) -> str:
    finish_reason = OPENAI_CHAT_JSON.required_string(value, label)
    if finish_reason not in _FINISH_REASONS:
        raise OpenAIChatError(f"{label} has unsupported value: {finish_reason}")
    return finish_reason


def _require_fields(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unexpected = set(value) - allowed
    if unexpected:
        raise OpenAIChatError(f"{label} has unsupported fields: {', '.join(sorted(unexpected))}")


def _require_exact_fields(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unexpected = set(value) - allowed
    if unexpected:
        raise OpenAIChatError(f"{label} has unsupported fields: {', '.join(sorted(unexpected))}")


def optional_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise OpenAIChatError(f"{label} must be a string or null")
    return value


def optional_service_tier(value: object, label: str) -> str | None:
    service_tier = optional_string(value, f"chat completion {label}")
    if service_tier is not None and service_tier not in _SERVICE_TIERS:
        raise OpenAIChatError(f"chat completion {label} has unsupported value: {service_tier}")
    return service_tier


def _optional_string_mapping(value: object, label: str) -> JsonObject | None:
    if value is None:
        return None
    mapping = OPENAI_CHAT_JSON.mapping(value, label)
    if len(mapping) > 16:
        raise OpenAIChatError(f"{label} must contain at most 16 entries")
    result: JsonObject = {}
    for key, item in mapping.items():
        if len(key) > 64 or not isinstance(item, str) or len(item) > 512:
            raise OpenAIChatError(
                f"{label} keys must be at most 64 characters and values must be strings "
                "of at most 512 characters"
            )
        result[key] = item
    return result


def reject_logprobs(value: object, label: str) -> None:
    if value is not None:
        raise OpenAIChatError(f"{label} is not supported")


def decode_usage(value: object) -> ModelUsage | None:
    if value is None:
        return None
    usage = OPENAI_CHAT_JSON.mapping(value, "chat completion usage")
    _require_exact_fields(usage, _USAGE_FIELDS, "chat completion usage")
    prompt_tokens = _required_nonnegative_int(usage.get("prompt_tokens"), "prompt_tokens")
    completion_tokens = _required_nonnegative_int(
        usage.get("completion_tokens"), "completion_tokens"
    )
    total_tokens = _required_nonnegative_int(usage.get("total_tokens"), "total_tokens")
    if usage.get("compute_units") is not None:
        _required_nonnegative_int(usage["compute_units"], "compute_units")
    completion_details = _usage_details(
        usage.get("completion_tokens_details"),
        _COMPLETION_DETAILS_FIELDS,
        "completion_tokens_details",
    )
    prompt_details = _usage_details(
        usage.get("prompt_tokens_details"),
        _PROMPT_DETAILS_FIELDS,
        "prompt_tokens_details",
    )
    reasoning_tokens = completion_details.get("reasoning_tokens") if completion_details else None
    cache_read_tokens = prompt_details.get("cached_tokens") if prompt_details else None
    return ModelUsage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=cache_read_tokens,
    )


def _required_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpenAIChatError(f"chat completion usage {label} must be a non-negative integer")
    return value


def _usage_details(
    value: object,
    allowed: set[str],
    label: str,
) -> dict[str, int | None] | None:
    if value is None:
        return None
    details = OPENAI_CHAT_JSON.mapping(value, f"chat completion usage {label}")
    _require_exact_fields(details, allowed, f"chat completion usage {label}")
    result: dict[str, int | None] = {}
    for field, item in details.items():
        if item is not None and (isinstance(item, bool) or not isinstance(item, int) or item < 0):
            raise OpenAIChatError(
                f"chat completion usage {label}.{field} must be a non-negative integer or null"
            )
        result[field] = item
    return result


def _validate_number_in_range(value: object, label: str, minimum: float, maximum: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise OpenAIChatError(
            f"Chat Completions {label} must be a finite number in [{minimum:g}, {maximum:g}]"
        )


def _choice_index(choice: Mapping[str, Any]) -> int:
    if "index" not in choice:
        raise OpenAIChatError("chat completion response choice requires an index")
    index = choice["index"]
    if not isinstance(index, int) or isinstance(index, bool):
        raise OpenAIChatError("chat completion response choice index must be an integer")
    return index
