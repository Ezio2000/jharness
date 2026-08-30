"""HTTP client for Anthropic Messages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict, Unpack

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
from jharness.models.anthropic.messages.codec import AnthropicMessagesCodec
from jharness.models.anthropic.messages.errors import AnthropicMessagesError
from jharness.models.anthropic.messages.profile import AnthropicMessagesProfile
from jharness.models.anthropic.messages.stream import AnthropicMessagesStreamDecoder

_ADDITIONAL_RETRYABLE_STATUS_CODES = frozenset({529})
_RETRYABLE_ERROR_CODES = frozenset({"overloaded_error"})
_REQUEST_ID_HEADERS = ("request-id", "x-request-id")


class _AnthropicMessagesModelOptions(TypedDict, total=False):
    profile: AnthropicMessagesProfile | None
    timeout: float | httpx.Timeout | None
    headers: Mapping[str, str] | None
    client: httpx.AsyncClient | None
    max_response_body_bytes: int
    max_sse_line_bytes: int
    max_sse_event_bytes: int


class AnthropicMessagesModel:
    """Model implementation backed by Anthropic Messages."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        **options: Unpack[_AnthropicMessagesModelOptions],
    ) -> None:
        config = model_client_config(
            base_url=base_url,
            api_key=api_key,
            model=model,
            options=options,
            default_profile=AnthropicMessagesProfile(),
            constructor_name="AnthropicMessagesModel.__init__",
        )
        self.base_url = config.base_url
        self._api_key = config.api_key
        self.model = config.model
        self.profile = config.profile
        self.codec = AnthropicMessagesCodec(model=config.model, profile=config.profile)
        self._timeout = config.timeout
        self._max_response_body_bytes = config.max_response_body_bytes
        self._max_sse_line_bytes = config.max_sse_line_bytes
        self._max_sse_event_bytes = config.max_sse_event_bytes
        self._headers = dict(config.headers)
        self._client = config.client
        self._errors = ModelErrorPolicy(
            provider=config.profile.name,
            codec_error=AnthropicMessagesError,
            request_id_headers=_REQUEST_ID_HEADERS,
            error_code_keys=("type", "code"),
            additional_retryable_status_codes=_ADDITIONAL_RETRYABLE_STATUS_CODES,
            retryable_error_codes=_RETRYABLE_ERROR_CODES,
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
            decoder = AnthropicMessagesStreamDecoder(self.profile)
            return await invoke_sse_model(
                client=self._client,
                timeout=self._timeout,
                context=context,
                url=self._messages_url(),
                payload=lambda: self.codec.encode_request(request, stream=True),
                headers=self._request_headers,
                decode_frame=lambda event, data: self._decode_sse_data(event, data, decoder),
                completed_response=decoder.completed_response,
                emit_delta=emit_delta,
                errors=self._errors,
                incomplete_error="Anthropic stream ended before message_stop",
                max_response_body_bytes=self._max_response_body_bytes,
                max_sse_line_bytes=self._max_sse_line_bytes,
                max_sse_event_bytes=self._max_sse_event_bytes,
            )
        return await invoke_json_model(
            client=self._client,
            timeout=self._timeout,
            context=context,
            url=self._messages_url(),
            payload=lambda: self.codec.encode_request(request, stream=False),
            headers=self._request_headers,
            decode=self.codec.decode_response,
            errors=self._errors,
            response_shape_error="Anthropic response must be an object",
            max_response_body_bytes=self._max_response_body_bytes,
        )

    def _decode_sse_data(
        self,
        event_name: str | None,
        frame_data: str,
        decoder: AnthropicMessagesStreamDecoder,
    ) -> tuple[bool, list[ModelDelta]]:
        data = frame_data.strip()
        if not data:
            return False, []
        parsed_mapping = decode_json_object(
            data,
            AnthropicMessagesError,
            "Anthropic stream event must be an object",
        )
        if (event_name == "error" or parsed_mapping.get("type") == "error") and (
            "error" in parsed_mapping
        ):
            raise stream_body_error(parsed_mapping, self._errors)
        return decoder.apply_event(event_name, parsed_mapping)

    def _messages_url(self) -> str:
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/messages"
        return f"{self.base_url}/v1/messages"

    def _request_headers(self, payload: Mapping[str, object]) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self.profile.anthropic_version,
            **self._headers,
        }
        headers["x-api-key"] = self._api_key
        return headers
