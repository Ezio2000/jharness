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
from jharness.models.openai.responses.artifacts import OpenAIResponsesArtifactStore
from jharness.models.openai.responses.codec import OpenAIResponsesCodec
from jharness.models.openai.responses.errors import OpenAIResponsesError
from jharness.models.openai.responses.profile import OpenAIResponsesProfile
from jharness.models.openai.responses.stream import OpenAIResponsesStreamDecoder

_DEFAULT_IMAGE_RESPONSE_LIMIT = 64 * 1024 * 1024
_REQUEST_ID_HEADERS = ("x-request-id", "request-id", "x-ds-request-id")


class _OpenAIResponsesModelOptions(TypedDict, total=False):
    profile: OpenAIResponsesProfile | None
    artifact_store: OpenAIResponsesArtifactStore | None
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
        artifact_store = transport_options.pop("artifact_store", None)
        if artifact_store is not None and (
            isinstance(artifact_store, type)
            or not isinstance(artifact_store, OpenAIResponsesArtifactStore)
        ):
            raise TypeError("artifact_store must implement OpenAIResponsesArtifactStore")
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
        self._artifact_store = artifact_store
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
        artifact_store = self._artifact_store
        registry = self.profile.provider_tool_registry
        needs_artifacts = registry.request_requires_artifact_store(
            request
        ) or registry.history_requires_artifact_store(request.messages)
        if artifact_store is None and needs_artifacts:
            raise ValueError("Responses provider artifacts require an OpenAIResponsesArtifactStore")
        wire_request = (
            request
            if artifact_store is None
            else await registry.hydrate_artifact_history(request, artifact_store, context)
        )
        if stream:
            decoder = OpenAIResponsesStreamDecoder(self.codec, self.profile)
            response = await invoke_sse_model(
                client=self._client,
                timeout=self._timeout,
                context=context,
                url=self._responses_url(),
                payload=lambda: self.codec.encode_request(wire_request, stream=True),
                headers=lambda _payload: self._request_headers(),
                decode_frame=lambda event, data: self._decode_sse_data(
                    event,
                    data,
                    decoder,
                ),
                completed_response=lambda: self._require_safe_artifacts(
                    decoder.completed_response()
                ),
                emit_delta=emit_delta,
                errors=self._errors,
                incomplete_error="Responses stream ended before a terminal response event",
                max_response_body_bytes=self._max_response_body_bytes,
                max_sse_line_bytes=self._max_sse_line_bytes,
                max_sse_event_bytes=self._max_sse_event_bytes,
            )
        else:
            response = await invoke_json_model(
                client=self._client,
                timeout=self._timeout,
                context=context,
                url=self._responses_url(),
                payload=lambda: self.codec.encode_request(wire_request, stream=False),
                headers=lambda _payload: self._request_headers(),
                decode=self._decode_response,
                errors=self._errors,
                response_shape_error="Responses response must be an object",
                body_error_predicate=_is_transport_error_body,
                max_response_body_bytes=self._max_response_body_bytes,
            )
        if artifact_store is None:
            return response
        return await registry.externalize_artifacts(response, artifact_store, context)

    def _decode_response(self, value: Mapping[str, Any]) -> ModelResponse:
        return self._require_safe_artifacts(self.codec.decode_response(value))

    def _require_safe_artifacts(self, response: ModelResponse) -> ModelResponse:
        if (
            self._artifact_store is None
            and self.profile.provider_tool_registry.response_requires_artifact_store(response)
        ):
            raise OpenAIResponsesError(
                "Responses returned inline provider artifact data without an "
                "OpenAIResponsesArtifactStore"
            )
        return response

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
