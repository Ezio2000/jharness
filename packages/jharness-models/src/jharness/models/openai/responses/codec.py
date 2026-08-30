"""Request and terminal-response codec for the OpenAI Responses API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NoReturn, cast

from jharness.kernel import (
    FreeformToolCall,
    ModelError,
    ModelErrorInfo,
    ModelOutputItem,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ProviderToolCall,
    ProviderToolStatus,
    ResponseFormat,
    StructuredToolCall,
    thaw_json_value,
)
from jharness.models.openai.responses.errors import OPENAI_RESPONSES_JSON, OpenAIResponsesError
from jharness.models.openai.responses.messages import (
    decode_output_items,
    encode_responses_input,
)
from jharness.models.openai.responses.profile import OpenAIResponsesProfile
from jharness.models.openai.responses.tools import encode_tool_choice, encode_tools

JsonObject = dict[str, Any]

_RESPONSE_FIELDS = frozenset(
    {
        "id",
        "created_at",
        "error",
        "incomplete_details",
        "instructions",
        "metadata",
        "model",
        "object",
        "output",
        "parallel_tool_calls",
        "temperature",
        "tool_choice",
        "tools",
        "top_p",
        "background",
        "completed_at",
        "conversation",
        "max_output_tokens",
        "max_tool_calls",
        "moderation",
        "previous_response_id",
        "prompt",
        "prompt_cache_key",
        "prompt_cache_options",
        "prompt_cache_retention",
        "reasoning",
        "safety_identifier",
        "service_tier",
        "status",
        "text",
        "top_logprobs",
        "truncation",
        "usage",
        "user",
    }
)


class OpenAIResponsesCodec:
    """Translate kernel model DTOs to and from the Responses wire protocol."""

    def __init__(
        self,
        *,
        model: str,
        profile: OpenAIResponsesProfile | None = None,
    ) -> None:
        self.profile = profile or OpenAIResponsesProfile()
        self.model = model

    def encode_request(self, request: ModelRequest, *, stream: bool = False) -> JsonObject:
        """Encode one complete-history request with ordered protocol items."""

        selected_model = request.options.model or self.model
        tools = encode_tools(request.runtime_tools, request.provider_tools, self.profile)
        payload: JsonObject = {
            "model": selected_model,
            "input": encode_responses_input(request.messages, self.profile),
        }
        self._add_state_options(payload)
        self._add_model_options(payload, request)
        self._add_tool_options(payload, request, tools)
        self._add_response_format(payload, request.response_format)
        self._add_stream_option(payload, stream=stream)
        return payload

    def decode_response(self, value: Mapping[str, Any]) -> ModelResponse:
        """Decode the authoritative full response used by both transports."""

        validate_response_fields(value, terminal=True)
        if value.get("object") != "response":
            raise OpenAIResponsesError("Responses response requires object='response'")
        response_id = OPENAI_RESPONSES_JSON.required_string(
            value.get("id"),
            "Responses response id",
        )
        model = OPENAI_RESPONSES_JSON.required_string(
            value.get("model"),
            "Responses response model",
        )
        created_at = value.get("created_at")
        if isinstance(created_at, bool) or not isinstance(created_at, int | float):
            raise OpenAIResponsesError("Responses created_at must be a number")
        status = OPENAI_RESPONSES_JSON.required_string(
            value.get("status"),
            "Responses response status",
        )
        if status in {"failed", "cancelled"}:
            self._raise_terminal_error(value, response_id=response_id, status=status)
        if status not in {"completed", "incomplete"}:
            raise OpenAIResponsesError(f"Responses terminal response has status={status!r}")
        if value.get("error") is not None:
            raise OpenAIResponsesError("successful Responses response must not contain an error")
        output = decode_output_items(
            value.get("output"),
            self.profile,
            response=value,
        )
        if any(
            isinstance(item, ProviderToolCall) and item.status is ProviderToolStatus.IN_PROGRESS
            for item in output
        ):
            raise OpenAIResponsesError(
                "Responses terminal response cannot contain in-progress provider tools"
            )
        if status == "incomplete" and any(
            isinstance(item, StructuredToolCall | FreeformToolCall) for item in output
        ):
            raise OpenAIResponsesError(
                "incomplete Responses cannot expose runtime tool calls for execution"
            )
        finish_reason = self._finish_reason(status, value, output)
        try:
            return ModelResponse(
                output=tuple(output),
                finish_reason=finish_reason,
                usage=decode_usage(value.get("usage")),
                model_id=model,
                response_id=response_id,
                metadata=_response_metadata(value, self.profile.name),
            )
        except (TypeError, ValueError) as exc:
            raise OpenAIResponsesError(f"Responses terminal response is invalid: {exc}") from exc

    def _add_state_options(self, payload: JsonObject) -> None:
        payload["store"] = self.profile.store
        if self.profile.include:
            payload["include"] = sorted(self.profile.include)

    def _add_model_options(self, payload: JsonObject, request: ModelRequest) -> None:
        options = request.options
        if options.temperature is not None:
            if not 0 <= options.temperature <= 2:
                raise OpenAIResponsesError("Responses temperature must be between 0 and 2")
            payload["temperature"] = options.temperature
        if options.top_p is not None:
            if not 0 <= options.top_p <= 1:
                raise OpenAIResponsesError("Responses top_p must be between 0 and 1")
            payload["top_p"] = options.top_p
        if options.max_output_tokens is not None:
            if options.max_output_tokens < 16:
                raise OpenAIResponsesError("Responses max_output_tokens must be at least 16")
            payload["max_output_tokens"] = options.max_output_tokens
        if options.stop:
            raise OpenAIResponsesError("Responses API does not support stop sequences")
        if options.seed is not None:
            raise OpenAIResponsesError("Responses API does not support seed")

    def _add_tool_options(
        self,
        payload: JsonObject,
        request: ModelRequest,
        tools: list[JsonObject],
    ) -> None:
        if tools:
            payload["tools"] = tools
        choice = encode_tool_choice(
            request.tool_choice,
            runtime_tools=request.runtime_tools,
            provider_tools=request.provider_tools,
            profile=self.profile,
        )
        if choice is not None:
            payload["tool_choice"] = choice
        if not request.may_return_runtime_tool_calls:
            return
        capabilities = self.profile.capabilities
        if not capabilities.parallel_runtime_tool_calls:
            return
        allow_parallel = request.tool_choice.allow_parallel_runtime_tool_calls
        if capabilities.parallel_runtime_tool_call_control:
            payload["parallel_tool_calls"] = allow_parallel
        elif not allow_parallel:
            raise OpenAIResponsesError(
                f"{self.profile.name} cannot disable parallel runtime tool calls"
            )

    def _add_response_format(
        self,
        payload: JsonObject,
        response_format: ResponseFormat | None,
    ) -> None:
        if response_format is None:
            return
        payload["text"] = {"format": self._encode_response_format(response_format)}

    def _encode_response_format(self, response_format: ResponseFormat) -> JsonObject:
        if response_format.type == "text":
            return {"type": "text"}
        if response_format.type == "json_object":
            if not self.profile.capabilities.json_mode:
                raise OpenAIResponsesError(
                    f"{self.profile.name} does not support JSON object output"
                )
            return {"type": "json_object"}
        if response_format.type == "json_schema":
            if not self.profile.capabilities.structured_output:
                raise OpenAIResponsesError(
                    f"{self.profile.name} does not support JSON schema output"
                )
            if response_format.schema is None:
                raise OpenAIResponsesError("JSON schema response format requires schema")
            schema = thaw_json_value(response_format.schema)
            if not isinstance(schema, dict):
                raise OpenAIResponsesError("Responses JSON schema must be an object")
            return {
                "type": "json_schema",
                "name": "response",
                "schema": schema,
                "strict": response_format.strict,
            }
        raise OpenAIResponsesError(f"unsupported Responses output format: {response_format.type}")

    def _add_stream_option(self, payload: JsonObject, *, stream: bool) -> None:
        if not stream:
            return
        if not self.profile.capabilities.streaming:
            raise OpenAIResponsesError(f"{self.profile.name} does not support streaming")
        payload["stream"] = True

    def _finish_reason(
        self,
        status: str,
        value: Mapping[str, Any],
        output: Sequence[ModelOutputItem],
    ) -> str:
        if status == "completed":
            return (
                "tool_calls"
                if any(isinstance(item, StructuredToolCall | FreeformToolCall) for item in output)
                else "stop"
            )
        details_value = value.get("incomplete_details")
        if details_value is None:
            return "incomplete"
        details = OPENAI_RESPONSES_JSON.mapping(
            details_value,
            "Responses incomplete_details",
        )
        unexpected = set(details).difference({"reason"})
        if unexpected:
            raise OpenAIResponsesError(
                "Responses incomplete_details contains unsupported field: " + min(unexpected)
            )
        reason = details.get("reason")
        if reason is None:
            return "incomplete"
        if reason == "max_output_tokens":
            return "length"
        if reason == "content_filter":
            return "content_filter"
        raise OpenAIResponsesError("unsupported Responses incomplete reason")

    def _raise_terminal_error(
        self,
        value: Mapping[str, Any],
        *,
        response_id: str,
        status: str,
    ) -> NoReturn:
        raw_error = value.get("error")
        code = f"response_{status}"
        message = f"provider response {status}"
        if isinstance(raw_error, Mapping):
            error = cast(Mapping[str, object], raw_error)
            unexpected = set(error).difference({"code", "message"})
            if unexpected:
                raise OpenAIResponsesError(
                    "Responses error contains unsupported field: " + min(unexpected)
                )
            raw_code = error.get("code")
            raw_message = error.get("message")
            _response_error_code(raw_code)
            if not isinstance(raw_message, str):
                raise OpenAIResponsesError("Responses error message must be a string")
            code = cast(str, raw_code)
            message = raw_message
        elif raw_error is not None:
            raise OpenAIResponsesError("Responses error must be an object or null")
        raise ModelError(
            ModelErrorInfo(
                code=code,
                message=message,
                provider=self.profile.name,
                retryable=False,
                metadata={"responses": _retained_response_envelope(value, include_output=True)},
            )
        )


def decode_usage(value: object) -> ModelUsage | None:
    """Decode one cumulative Responses usage snapshot."""

    if value is None:
        return None
    usage = OPENAI_RESPONSES_JSON.mapping(value, "Responses usage")
    unexpected = set(usage).difference(
        {
            "input_tokens",
            "input_tokens_details",
            "output_tokens",
            "output_tokens_details",
            "total_tokens",
            "compute_units",
        }
    )
    if unexpected:
        raise OpenAIResponsesError("Responses usage contains unsupported field: " + min(unexpected))
    input_tokens = _usage_counter(usage.get("input_tokens"), "Responses input_tokens")
    output_tokens = _usage_counter(usage.get("output_tokens"), "Responses output_tokens")
    total_tokens = _usage_counter(usage.get("total_tokens"), "Responses total_tokens")
    input_details = OPENAI_RESPONSES_JSON.mapping(
        usage.get("input_tokens_details"),
        "Responses input_tokens_details",
    )
    if set(input_details).difference({"cached_tokens", "cache_write_tokens"}):
        raise OpenAIResponsesError("Responses input_tokens_details contains unsupported field")
    cache_read_tokens = _usage_counter(
        input_details.get("cached_tokens"),
        "Responses input_tokens_details.cached_tokens",
    )
    cache_write_tokens = _usage_counter(
        input_details.get("cache_write_tokens"),
        "Responses input_tokens_details.cache_write_tokens",
    )
    output_details = OPENAI_RESPONSES_JSON.mapping(
        usage.get("output_tokens_details"),
        "Responses output_tokens_details",
    )
    if set(output_details).difference({"reasoning_tokens"}):
        raise OpenAIResponsesError("Responses output_tokens_details contains unsupported field")
    reasoning_tokens = _usage_counter(
        output_details.get("reasoning_tokens"),
        "Responses output_tokens_details.reasoning_tokens",
    )
    if "compute_units" in usage:
        _usage_counter(usage["compute_units"], "Responses compute_units")
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )


def _usage_counter(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpenAIResponsesError(f"{label} must be a non-negative integer")
    return value


def _response_metadata(value: Mapping[str, Any], provider: str) -> JsonObject:
    return {
        "provider": provider,
        "responses": _retained_response_envelope(value, include_output=False),
    }


def _retained_response_envelope(value: Mapping[str, Any], *, include_output: bool) -> JsonObject:
    """Retain validated response fields that have no other lossless representation."""

    # Successful output is represented by ModelResponse.output. Excluding the native
    # duplicate also prevents generated image bytes from bypassing artifact externalization.
    return {
        key: thaw_json_value(raw) for key, raw in value.items() if include_output or key != "output"
    }


def _response_error_code(value: object) -> None:
    codes = {
        "server_error",
        "rate_limit_exceeded",
        "invalid_prompt",
        "data_residency_mismatch",
        "bio_policy",
        "vector_store_timeout",
        "invalid_image",
        "invalid_image_format",
        "invalid_base64_image",
        "invalid_image_url",
        "image_too_large",
        "image_too_small",
        "image_parse_error",
        "image_content_policy_violation",
        "invalid_image_mode",
        "image_file_too_large",
        "unsupported_image_media_type",
        "empty_image_file",
        "failed_to_download_image",
        "image_file_not_found",
    }
    if value not in codes:
        raise OpenAIResponsesError("unsupported Responses error code")


def validate_response_fields(  # noqa: C901
    value: Mapping[str, Any], *, terminal: bool
) -> None:
    """Reject vendor envelope fields and validate standard fields this adapter retains."""

    unexpected = set(value).difference(_RESPONSE_FIELDS)
    if unexpected:
        raise OpenAIResponsesError(
            "Responses response contains unsupported field: " + min(unexpected)
        )
    if value.get("object") != "response":
        raise OpenAIResponsesError("Responses response requires object='response'")
    OPENAI_RESPONSES_JSON.required_string(value.get("id"), "Responses response id")
    if terminal:
        OPENAI_RESPONSES_JSON.required_string(value.get("model"), "Responses response model")
        if not _is_array(value.get("output")):
            raise OpenAIResponsesError("Responses response output must be an array")
        created_at = value.get("created_at")
        if isinstance(created_at, bool) or not isinstance(created_at, int | float):
            raise OpenAIResponsesError("Responses created_at must be a number")
    for field in ("completed_at",):
        if (
            field in value
            and value[field] is not None
            and (isinstance(value[field], bool) or not isinstance(value[field], int | float))
        ):
            raise OpenAIResponsesError(f"Responses {field} must be a number or null")
    status = value.get("status")
    if status is not None and status not in {
        "completed",
        "failed",
        "in_progress",
        "cancelled",
        "queued",
        "incomplete",
    }:
        raise OpenAIResponsesError("Responses status is invalid")
    service_tier = value.get("service_tier")
    if service_tier is not None and service_tier not in {
        "auto",
        "default",
        "flex",
        "scale",
        "priority",
        "fast",
        "ultrafast",
    }:
        raise OpenAIResponsesError("Responses service_tier is invalid")
    truncation = value.get("truncation")
    if truncation is not None and truncation not in {"auto", "disabled"}:
        raise OpenAIResponsesError("Responses truncation is invalid")
    retention = value.get("prompt_cache_retention")
    if retention is not None and retention not in {"in_memory", "24h"}:
        raise OpenAIResponsesError("Responses prompt_cache_retention is invalid")
    for field in ("parallel_tool_calls", "background"):
        if field in value and value[field] is not None and not isinstance(value[field], bool):
            raise OpenAIResponsesError(f"Responses {field} must be a bool or null")
    for field in ("temperature", "top_p"):
        if (
            field in value
            and value[field] is not None
            and (isinstance(value[field], bool) or not isinstance(value[field], int | float))
        ):
            raise OpenAIResponsesError(f"Responses {field} must be a number or null")
    for field in ("max_output_tokens", "max_tool_calls", "top_logprobs"):
        if field in value and value[field] is not None:
            _usage_counter(value[field], f"Responses {field}")
    if isinstance(value.get("top_logprobs"), int) and value["top_logprobs"] > 20:
        raise OpenAIResponsesError("Responses top_logprobs must be at most 20")
    for field in ("tools",):
        if field in value and not _is_array(value[field]):
            raise OpenAIResponsesError(f"Responses {field} must be an array")
    for field in ("previous_response_id", "prompt_cache_key", "safety_identifier", "user"):
        if field in value and value[field] is not None and not isinstance(value[field], str):
            raise OpenAIResponsesError(f"Responses {field} must be a string or null")
    for field in (
        "metadata",
        "conversation",
        "prompt",
        "prompt_cache_options",
        "reasoning",
        "text",
    ):
        if field in value and value[field] is not None and not isinstance(value[field], Mapping):
            raise OpenAIResponsesError(f"Responses {field} must be an object or null")
    if (
        "instructions" in value
        and value["instructions"] is not None
        and not (isinstance(value["instructions"], str) or _is_array(value["instructions"]))
    ):
        raise OpenAIResponsesError("Responses instructions must be a string, array, or null")
    _validate_response_metadata(value.get("metadata"))
    _validate_response_conversation(value.get("conversation"))
    _validate_prompt_cache_options(value.get("prompt_cache_options"))
    _validate_response_reasoning(value.get("reasoning"))
    _validate_response_text(value.get("text"))
    _validate_incomplete_details(value.get("incomplete_details"))
    _validate_response_error(value.get("error"))
    if value.get("moderation") is not None:
        raise OpenAIResponsesError("Responses moderation output is not supported")


def _validate_response_metadata(value: object) -> None:
    if value is None:
        return
    metadata = OPENAI_RESPONSES_JSON.mapping(value, "Responses metadata")
    if len(metadata) > 16:
        raise OpenAIResponsesError("Responses metadata must contain at most 16 entries")
    for key, item in metadata.items():
        if len(key) > 64 or not isinstance(item, str) or len(item) > 512:
            raise OpenAIResponsesError(
                "Responses metadata keys must be at most 64 characters and values must be "
                "strings of at most 512 characters"
            )


def _validate_response_conversation(value: object) -> None:
    if value is None:
        return
    conversation = OPENAI_RESPONSES_JSON.mapping(value, "Responses conversation")
    unexpected = set(conversation).difference({"id"})
    if unexpected:
        raise OpenAIResponsesError(
            "Responses conversation contains unsupported field: " + min(unexpected)
        )
    OPENAI_RESPONSES_JSON.required_string(conversation.get("id"), "Responses conversation id")


def _validate_prompt_cache_options(value: object) -> None:
    if value is None:
        return
    options = OPENAI_RESPONSES_JSON.mapping(value, "Responses prompt_cache_options")
    unexpected = set(options).difference({"mode", "ttl"})
    if unexpected:
        raise OpenAIResponsesError(
            "Responses prompt_cache_options contains unsupported field: " + min(unexpected)
        )
    if options.get("mode") not in {"implicit", "explicit"} or options.get("ttl") != "30m":
        raise OpenAIResponsesError(
            "Responses prompt_cache_options requires mode=implicit|explicit and ttl='30m'"
        )


def _validate_response_reasoning(value: object) -> None:
    if value is None:
        return
    reasoning = OPENAI_RESPONSES_JSON.mapping(value, "Responses reasoning")
    unexpected = set(reasoning).difference(
        {"context", "effort", "generate_summary", "mode", "summary"}
    )
    if unexpected:
        raise OpenAIResponsesError(
            "Responses reasoning contains unsupported field: " + min(unexpected)
        )
    if reasoning.get("context") is not None and reasoning.get("context") not in {
        "auto",
        "current_turn",
        "all_turns",
    }:
        raise OpenAIResponsesError("Responses reasoning.context is invalid")
    if reasoning.get("effort") is not None and reasoning.get("effort") not in {
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    }:
        raise OpenAIResponsesError("Responses reasoning.effort is invalid")
    for field in ("generate_summary", "summary"):
        if reasoning.get(field) is not None and reasoning.get(field) not in {
            "auto",
            "concise",
            "detailed",
        }:
            raise OpenAIResponsesError(f"Responses reasoning.{field} is invalid")
    if (
        "mode" in reasoning
        and reasoning["mode"] is not None
        and not isinstance(reasoning["mode"], str)
    ):
        raise OpenAIResponsesError("Responses reasoning.mode must be a string or null")


def _validate_response_text(value: object) -> None:
    if value is None:
        return
    text = OPENAI_RESPONSES_JSON.mapping(value, "Responses text")
    unexpected = set(text).difference({"format", "verbosity"})
    if unexpected:
        raise OpenAIResponsesError("Responses text contains unsupported field: " + min(unexpected))
    if text.get("verbosity") is not None and text.get("verbosity") not in {
        "low",
        "medium",
        "high",
    }:
        raise OpenAIResponsesError("Responses text.verbosity is invalid")
    response_format = text.get("format")
    if response_format is None:
        return
    format_value = OPENAI_RESPONSES_JSON.mapping(response_format, "Responses text.format")
    format_type = format_value.get("type")
    allowed = {
        "text": {"type"},
        "json_object": {"type"},
        "json_schema": {"type", "name", "schema", "description", "strict"},
    }
    if format_type not in allowed:
        raise OpenAIResponsesError("Responses text.format.type is invalid")
    unexpected_format = set(format_value).difference(allowed[cast(str, format_type)])
    if unexpected_format:
        raise OpenAIResponsesError(
            "Responses text.format contains unsupported field: " + min(unexpected_format)
        )
    if format_type == "json_schema":
        _validate_json_schema_response_format(format_value)


def _validate_json_schema_response_format(value: Mapping[str, Any]) -> None:
    OPENAI_RESPONSES_JSON.required_string(value.get("name"), "Responses text.format.name")
    if not isinstance(value.get("schema"), Mapping):
        raise OpenAIResponsesError("Responses text.format.schema must be an object")
    description = value.get("description")
    if "description" in value and description is not None and not isinstance(description, str):
        raise OpenAIResponsesError("Responses text.format.description must be a string or null")
    strict = value.get("strict")
    if "strict" in value and strict is not None and not isinstance(strict, bool):
        raise OpenAIResponsesError("Responses text.format.strict must be a bool or null")


def _validate_incomplete_details(value: object) -> None:
    if value is None:
        return
    details = OPENAI_RESPONSES_JSON.mapping(value, "Responses incomplete_details")
    unexpected = set(details).difference({"reason"})
    if unexpected:
        raise OpenAIResponsesError(
            "Responses incomplete_details contains unsupported field: " + min(unexpected)
        )
    if details.get("reason") is not None and details.get("reason") not in {
        "max_output_tokens",
        "content_filter",
    }:
        raise OpenAIResponsesError("unsupported Responses incomplete reason")


def _validate_response_error(value: object) -> None:
    if value is None:
        return
    error = OPENAI_RESPONSES_JSON.mapping(value, "Responses error")
    unexpected = set(error).difference({"code", "message"})
    if unexpected:
        raise OpenAIResponsesError("Responses error contains unsupported field: " + min(unexpected))
    _response_error_code(error.get("code"))
    if not isinstance(error.get("message"), str):
        raise OpenAIResponsesError("Responses error message must be a string")


def _is_array(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)
