"""HTTP client for OpenAI-compatible Responses APIs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypedDict, Unpack

import httpx

from jharness.kernel import (
    DeltaSink,
    ModelCapabilities,
    ModelDelta,
    ModelRequest,
    ModelResponse,
    RunContext,
)
from jharness.models._http import (
    ModelErrorPolicy,
    decode_json_object,
    invoke_json_model,
    invoke_sse_model,
    model_client_config,
    stream_body_error,
)
from jharness.models.openai.errors import OpenAIResponsesError
from jharness.models.openai.profiles import OpenAIResponsesProfile
from jharness.models.openai.responses_api.codec import OpenAIResponsesCodec
from jharness.models.openai.responses_api.stream import OpenAIResponsesStreamDecoder

_DEFAULT_IMAGE_RESPONSE_LIMIT = 64 * 1024 * 1024
_REQUEST_ID_HEADERS = ("x-request-id", "request-id", "x-ds-request-id")


class _OpenAIResponsesModelOptions(TypedDict, total=False):
    profile: OpenAIResponsesProfile | None
    timeout: float | httpx.Timeout | None
    headers: Mapping[str, str] | None
    client: httpx.AsyncClient | None
    max_response_body_bytes: int
    max_sse_line_bytes: int
    max_sse_event_bytes: int


class OpenAIResponsesModel:
    """Model implementation backed by an OpenAI-compatible Responses endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        **options: Unpack[_OpenAIResponsesModelOptions],
    ) -> None:
        transport_options: dict[str, Any] = dict(options)
        transport_options.setdefault(
            "max_response_body_bytes",
            _DEFAULT_IMAGE_RESPONSE_LIMIT,
        )
        transport_options.setdefault("max_sse_line_bytes", _DEFAULT_IMAGE_RESPONSE_LIMIT)
        transport_options.setdefault("max_sse_event_bytes", _DEFAULT_IMAGE_RESPONSE_LIMIT)
        config = model_client_config(
            base_url=base_url,
            api_key=api_key,
            model=model,
            options=transport_options,
            default_profile=OpenAIResponsesProfile(),
            constructor_name="OpenAIResponsesModel.__init__",
        )
        self.base_url = config.base_url
        self._api_key = config.api_key
        self.model = config.model
        self.profile = config.profile
        self.codec = OpenAIResponsesCodec(model=config.model, profile=config.profile)
        self._timeout = config.timeout
        self._max_response_body_bytes = config.max_response_body_bytes
        self._max_sse_line_bytes = config.max_sse_line_bytes
        self._max_sse_event_bytes = config.max_sse_event_bytes
        self._headers = dict(config.headers)
        self._client = config.client
        self._errors = ModelErrorPolicy(
            provider=config.profile.name,
            codec_error=OpenAIResponsesError,
            request_id_headers=_REQUEST_ID_HEADERS,
            error_code_keys=("code", "type"),
            body_request_id_key="request_id",
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        return self.profile.capabilities

    async def invoke(
        self,
        request: ModelRequest,
        context: RunContext,
        *,
        stream: bool,
        emit_delta: DeltaSink | None,
    ) -> ModelResponse:
        if not stream and emit_delta is not None:
            raise ValueError("emit_delta requires stream=True")
        if stream:
            decoder = OpenAIResponsesStreamDecoder(self.codec, self.profile)
            return await invoke_sse_model(
                client=self._client,
                timeout=self._timeout,
                context=context,
                url=self._responses_url(),
                payload=lambda: self.codec.encode_request(request, stream=True),
                headers=lambda _payload: self._request_headers(),
                decode_frame=lambda event, data: self._decode_sse_data(
                    event,
                    data,
                    decoder,
                ),
                completed_response=decoder.completed_response,
                emit_delta=emit_delta,
                errors=self._errors,
                incomplete_error="Responses stream ended before a terminal response event",
                max_response_body_bytes=self._max_response_body_bytes,
                max_sse_line_bytes=self._max_sse_line_bytes,
                max_sse_event_bytes=self._max_sse_event_bytes,
            )
        return await invoke_json_model(
            client=self._client,
            timeout=self._timeout,
            context=context,
            url=self._responses_url(),
            payload=lambda: self.codec.encode_request(request, stream=False),
            headers=lambda _payload: self._request_headers(),
            decode=self.codec.decode_response,
            errors=self._errors,
            response_shape_error="Responses response must be an object",
            body_error_predicate=_is_transport_error_body,
            max_response_body_bytes=self._max_response_body_bytes,
        )

    def _decode_sse_data(
        self,
        event_name: str | None,
        frame_data: str,
        decoder: OpenAIResponsesStreamDecoder,
    ) -> tuple[bool, list[ModelDelta]]:
        data = frame_data.strip()
        if not data:
            return False, []
        if data == "[DONE]":
            raise OpenAIResponsesError(
                "Responses streams terminate with a typed response event, not [DONE]"
            )
        parsed = decode_json_object(
            data,
            OpenAIResponsesError,
            "Responses stream event must be an object",
        )
        if event_name == "error" or parsed.get("type") == "error":
            error_body = parsed if "error" in parsed else {"error": parsed}
            raise stream_body_error(error_body, self._errors)
        if parsed.get("error") is not None and "response" not in parsed:
            raise stream_body_error(parsed, self._errors)
        return decoder.apply_event(event_name, parsed)

    def _responses_url(self) -> str:
        return f"{self.base_url}/responses"

    def _request_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self._headers,
        }


def _is_transport_error_body(value: Mapping[str, object]) -> bool:
    return value.get("object") != "response" and value.get("error") is not None
