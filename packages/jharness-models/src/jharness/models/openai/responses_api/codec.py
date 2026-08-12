"""Request and terminal-response codec for compatible Responses APIs."""

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
from jharness.models.openai.errors import OPENAI_RESPONSES_JSON, OpenAIResponsesError
from jharness.models.openai.profiles import OpenAIResponsesProfile
from jharness.models.openai.responses_api.messages import (
    decode_output_items,
    encode_responses_input,
)
from jharness.models.openai.responses_api.tools import encode_tool_choice, encode_tools

JsonObject = dict[str, Any]

_RESERVED_REQUEST_FIELDS = frozenset(
    {
        "include",
        "input",
        "max_output_tokens",
        "model",
        "parallel_tool_calls",
        "previous_response_id",
        "seed",
        "stop",
        "store",
        "stream",
        "temperature",
        "text",
        "tool_choice",
        "tools",
        "top_p",
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
        self.profile.validate_model(model)
        self.model = model

    def encode_request(self, request: ModelRequest, *, stream: bool = False) -> JsonObject:
        """Encode one complete-history request with ordered protocol items."""

        selected_model = request.options.model or self.model
        self.profile.validate_model(selected_model)
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
        self._add_extra_request_body(payload)
        return payload

    def decode_response(self, value: Mapping[str, Any]) -> ModelResponse:
        """Decode the authoritative full response used by both transports."""

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
        status = OPENAI_RESPONSES_JSON.required_string(
            value.get("status"),
            "Responses response status",
        )
        if status == "failed":
            self._raise_failed_response(value, response_id=response_id)
        if status not in {"completed", "incomplete"}:
            raise OpenAIResponsesError(f"Responses terminal response has status={status!r}")
        if value.get("error") is not None:
            raise OpenAIResponsesError("successful Responses response must not contain an error")
        output = decode_output_items(
            value.get("output"),
            self.profile,
            response=value,
        )
        if not output:
            raise OpenAIResponsesError("Responses terminal response requires output")
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
        if self.profile.store is not None:
            payload["store"] = self.profile.store
        if self.profile.include:
            payload["include"] = sorted(self.profile.include)

    def _add_model_options(self, payload: JsonObject, request: ModelRequest) -> None:
        options = request.options
        if options.temperature is not None:
            payload["temperature"] = options.temperature
        if options.top_p is not None:
            payload["top_p"] = options.top_p
        if options.max_output_tokens is not None:
            payload["max_output_tokens"] = options.max_output_tokens
        if options.stop:
            raise OpenAIResponsesError("Responses API does not support stop sequences")
        if options.seed is not None:
            if not self.profile.capabilities.seed:
                raise OpenAIResponsesError(f"{self.profile.name} does not support seed")
            payload["seed"] = options.seed

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
            return {
                "type": "json_schema",
                "name": "response",
                "schema": thaw_json_value(response_format.schema),
                "strict": response_format.strict,
            }
        raise OpenAIResponsesError(f"unsupported Responses output format: {response_format.type}")

    def _add_stream_option(self, payload: JsonObject, *, stream: bool) -> None:
        if not stream:
            return
        if not self.profile.capabilities.streaming:
            raise OpenAIResponsesError(f"{self.profile.name} does not support streaming")
        payload["stream"] = True

    def _add_extra_request_body(self, payload: JsonObject) -> None:
        collision = _RESERVED_REQUEST_FIELDS.intersection(self.profile.extra_request_body)
        if collision:
            key = min(collision)
            raise OpenAIResponsesError(
                f"extra_request_body cannot set reserved request field: {key}"
            )
        payload.update(cast(JsonObject, thaw_json_value(self.profile.extra_request_body)))

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
        reason = OPENAI_RESPONSES_JSON.required_string(
            details.get("reason"),
            "Responses incomplete reason",
        )
        return self.profile.finish_reason(reason) or "incomplete"

    def _raise_failed_response(
        self,
        value: Mapping[str, Any],
        *,
        response_id: str,
    ) -> NoReturn:
        raw_error = value.get("error")
        code = "response_failed"
        message = "provider response failed"
        if isinstance(raw_error, Mapping):
            error = cast(Mapping[str, object], raw_error)
            raw_code = error.get("code")
            raw_message = error.get("message")
            if isinstance(raw_code, str) and raw_code:
                code = raw_code
            if isinstance(raw_message, str) and raw_message:
                message = raw_message
        elif isinstance(raw_error, str) and raw_error:
            message = raw_error
        raise ModelError(
            ModelErrorInfo(
                code=code,
                message=message,
                provider=self.profile.name,
                retryable=False,
                metadata={"response_id": response_id, "status": "failed"},
            )
        )


def decode_usage(value: object) -> ModelUsage | None:
    """Decode one cumulative Responses usage snapshot."""

    if value is None:
        return None
    usage = OPENAI_RESPONSES_JSON.mapping(value, "Responses usage")
    input_tokens = OPENAI_RESPONSES_JSON.optional_integer(usage.get("input_tokens"))
    output_tokens = OPENAI_RESPONSES_JSON.optional_integer(usage.get("output_tokens"))
    total_tokens = OPENAI_RESPONSES_JSON.optional_integer(usage.get("total_tokens"))
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    output_details = usage.get("output_tokens_details")
    if output_details is not None:
        details = OPENAI_RESPONSES_JSON.mapping(
            output_details,
            "Responses output_tokens_details",
        )
        reasoning_tokens = OPENAI_RESPONSES_JSON.optional_integer(details.get("reasoning_tokens"))
    input_details = usage.get("input_tokens_details")
    if input_details is not None:
        details = OPENAI_RESPONSES_JSON.mapping(
            input_details,
            "Responses input_tokens_details",
        )
        cache_read_tokens = OPENAI_RESPONSES_JSON.optional_integer(details.get("cached_tokens"))
        cache_write_tokens = OPENAI_RESPONSES_JSON.optional_integer(
            details.get("cache_write_tokens")
        )
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )


def _response_metadata(value: Mapping[str, Any], provider: str) -> JsonObject:
    metadata: JsonObject = {
        "provider": provider,
        "object": "response",
        "status": cast(str, value["status"]),
    }
    created_at = value.get("created_at")
    if created_at is not None:
        if isinstance(created_at, bool) or not isinstance(created_at, int):
            raise OpenAIResponsesError("Responses created_at must be an integer")
        metadata["created_at"] = created_at
    for key in ("completed_at", "store", "previous_response_id", "incomplete_details"):
        if key in value:
            metadata[key] = value[key]
    return metadata
