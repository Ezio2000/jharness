"""HTTP client for OpenAI Chat Completions."""

from __future__ import annotations

from typing import Unpack

from jharness.kernel import (
    DeltaSink,
    ModelCapabilities,
    ModelDelta,
    ModelRequest,
    ModelResponse,
    RunContext,
)
from jharness.models._http import (
    ModelClientOptions,
    ModelErrorPolicy,
    decode_json_object,
    invoke_json_model,
    invoke_sse_model,
    model_client_config,
    stream_body_error,
)
from jharness.models.openai.chat.codec import OpenAIChatCodec
from jharness.models.openai.chat.errors import OpenAIChatError
from jharness.models.openai.chat.profile import OpenAIChatProfile
from jharness.models.openai.chat.stream import OpenAIChatStreamDecoder

_REQUEST_ID_HEADERS = ("x-request-id",)


class _OpenAIChatModelOptions(ModelClientOptions, total=False):
    profile: OpenAIChatProfile | None


class OpenAIChatModel:
    """Model implementation backed by OpenAI Chat Completions."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        **options: Unpack[_OpenAIChatModelOptions],
    ) -> None:
        config = model_client_config(
            base_url=base_url,
            api_key=api_key,
            model=model,
            options=options,
            default_profile=OpenAIChatProfile(),
            constructor_name="OpenAIChatModel.__init__",
        )
        self.base_url = config.base_url
        self._api_key = config.api_key
        self.model = config.model
        self.profile = config.profile
        self.codec = OpenAIChatCodec(model=config.model, profile=config.profile)
        self._transport = config.transport
        self._headers = dict(config.headers)
        self._errors = ModelErrorPolicy(
            provider=config.profile.name,
            codec_error=OpenAIChatError,
            request_id_headers=_REQUEST_ID_HEADERS,
            error_code_keys=("code", "type"),
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
            decoder = OpenAIChatStreamDecoder(self.profile)
            return await invoke_sse_model(
                transport=self._transport,
                context=context,
                url=self._chat_completions_url(),
                payload=lambda: self.codec.encode_request(request, stream=True),
                headers=lambda _payload: self._request_headers(),
                decode_frame=lambda _event, data: self._decode_sse_data(data, decoder),
                completed_response=decoder.completed_response,
                emit_delta=emit_delta,
                errors=self._errors,
                incomplete_error="chat completion stream ended before [DONE]",
            )
        return await invoke_json_model(
            transport=self._transport,
            context=context,
            url=self._chat_completions_url(),
            payload=lambda: self.codec.encode_request(request, stream=False),
            headers=lambda _payload: self._request_headers(),
            decode=self.codec.decode_response,
            errors=self._errors,
            response_shape_error="chat completion response must be an object",
        )

    def _decode_sse_data(
        self,
        frame_data: str,
        decoder: OpenAIChatStreamDecoder,
    ) -> tuple[bool, list[ModelDelta]]:
        data = frame_data.strip()
        if not data:
            return False, []
        if data == "[DONE]":
            return True, []
        parsed_mapping = decode_json_object(
            data,
            OpenAIChatError,
            "chat completion stream chunk must be an object",
        )
        if "error" in parsed_mapping:
            raise stream_body_error(parsed_mapping, self._errors)
        return False, decoder.apply_chunk(parsed_mapping)

    def _chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _request_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self._headers,
        }
