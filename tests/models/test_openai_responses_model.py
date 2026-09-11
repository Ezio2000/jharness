from __future__ import annotations

import base64
import json
from dataclasses import replace
from hashlib import sha256
from time import time
from typing import Any, cast

import httpx
import pytest

from jharness.kernel import (
    ArtifactRef,
    ContentPart,
    FreeformToolCall,
    FreeformToolSpec,
    Message,
    ModelContentDelta,
    ModelDelta,
    ModelError,
    ModelErrorInfo,
    ModelOptions,
    ModelProviderToolCallDelta,
    ModelReasoningDelta,
    ModelRequest,
    ModelRuntimeToolCallDelta,
    ModelUsageDelta,
    ProviderToolCall,
    ProviderToolId,
    ProviderToolSpec,
    ProviderToolStatus,
    ResponseFormat,
    RunContext,
    Runtime,
    RuntimeToolKind,
    StructuredToolCall,
    StructuredToolSpec,
    ToolChoice,
    ToolSuccess,
)
from jharness.models.openai import (
    OpenAIResponsesArtifactStore,
    OpenAIResponsesCodec,
    OpenAIResponsesError,
    OpenAIResponsesModel,
    OpenAIResponsesProfile,
)
from jharness.models.openai.responses.stream import OpenAIResponsesStreamDecoder
from tests.models.support import terminal_response

_OPENAI_WEB = ProviderToolId("openai.responses", "web_search")
_OPENAI_IMAGE = ProviderToolId("openai.responses", "image_generation")
_JPEG_BYTES = b"\xff\xd8\xffjpeg-payload"
_JPEG_BASE64 = base64.b64encode(_JPEG_BYTES).decode("ascii")


def _openai_feature_profile(
    *,
    image_input: bool = False,
    image_generation: bool = False,
) -> OpenAIResponsesProfile:
    default = OpenAIResponsesProfile()
    provider_tools = frozenset({_OPENAI_IMAGE}) if image_generation else frozenset[ProviderToolId]()
    return OpenAIResponsesProfile(
        capabilities=replace(
            default.capabilities,
            tool_choice_types=(
                default.capabilities.tool_choice_types | {"provider"}
                if provider_tools
                else default.capabilities.tool_choice_types
            ),
            input_modalities=(
                frozenset({"text", "image", "file"})
                if image_input
                else default.capabilities.input_modalities
            ),
            provider_tools=provider_tools,
        ),
    )


def _openai_web_profile() -> OpenAIResponsesProfile:
    default = OpenAIResponsesProfile()
    return OpenAIResponsesProfile(
        capabilities=replace(
            default.capabilities,
            tool_choice_types=default.capabilities.tool_choice_types | {"provider"},
            provider_tools=frozenset({_OPENAI_WEB}),
        ),
    )


class _MemoryOpenAIResponsesArtifactStore:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.saved: list[str] = []
        self.loaded: list[str] = []

    async def save_image(
        self,
        data: bytes,
        *,
        media_type: str,
        call_id: str,
        context: RunContext,
    ) -> ArtifactRef:
        del call_id, context
        digest = sha256(data).hexdigest()
        ref = f"artifact:sha256:{digest}"
        self.values[ref] = data
        self.saved.append(ref)
        return ArtifactRef(
            ref,
            media_type=media_type,
            size_bytes=len(data),
            sha256=digest,
        )

    async def load_image(
        self,
        artifact: ArtifactRef,
        *,
        call_id: str,
        context: RunContext,
    ) -> bytes:
        del call_id, context
        self.loaded.append(artifact.ref)
        return self.values[artifact.ref]


class _StaticArtifactStore(_MemoryOpenAIResponsesArtifactStore):
    def __init__(self, artifact: object) -> None:
        super().__init__()
        self.artifact = artifact

    async def save_image(
        self,
        data: bytes,
        *,
        media_type: str,
        call_id: str,
        context: RunContext,
    ) -> ArtifactRef:
        del data, media_type, call_id, context
        return cast(ArtifactRef, self.artifact)


class _FailingArtifactStore(_MemoryOpenAIResponsesArtifactStore):
    async def save_image(
        self,
        data: bytes,
        *,
        media_type: str,
        call_id: str,
        context: RunContext,
    ) -> ArtifactRef:
        del data, media_type, call_id, context
        raise OSError("artifact save failed")

    async def load_image(
        self,
        artifact: ArtifactRef,
        *,
        call_id: str,
        context: RunContext,
    ) -> bytes:
        del artifact, call_id, context
        raise OSError("artifact load failed")


def _message(message_id: str, content: list[dict[str, Any]], *, status: str) -> dict[str, Any]:
    return {
        "id": message_id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": content,
    }


def _stream_event(
    decoder: OpenAIResponsesStreamDecoder,
    event_type: str,
    sequence_number: int,
    **fields: Any,
) -> tuple[bool, list[ModelDelta]]:
    event = {"type": event_type, "sequence_number": sequence_number, **fields}
    return decoder.apply_event(event_type, event)


def test_openai_responses_default_profile_is_conservative_and_stateless() -> None:
    profile = OpenAIResponsesProfile()
    codec = OpenAIResponsesCodec(model="gpt-test", profile=profile)
    payload = codec.encode_request(ModelRequest(messages=(Message.user("hello"),)))

    assert profile.capabilities.input_modalities == frozenset({"text"})
    assert profile.capabilities.provider_tools == frozenset()
    assert profile.capabilities.structured_output is False
    assert profile.capabilities.json_mode is False
    assert payload["store"] is False
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert "previous_response_id" not in payload

    stored = OpenAIResponsesProfile(store=True, include=frozenset())
    stored_payload = OpenAIResponsesCodec(model="gpt-test", profile=stored).encode_request(
        ModelRequest(messages=(Message.user("hello"),))
    )
    assert stored_payload["store"] is True
    assert "include" not in stored_payload

    reasoning = {
        "id": "reasoning-1",
        "type": "reasoning",
        "status": "completed",
        "summary": [{"type": "summary_text", "text": "summary"}],
    }
    with pytest.raises(OpenAIResponsesError, match="encrypted_content"):
        codec.decode_response(terminal_response([reasoning]))
    reasoning["encrypted_content"] = "encrypted-state"
    response = codec.decode_response(terminal_response([reasoning]))
    assert response.metadata["provider"] == "openai-responses"
    replay = codec.encode_request(
        ModelRequest(messages=(Message.user("hello"), response.to_assistant_message()))
    )
    assert cast(list[dict[str, Any]], replay["input"])[1]["encrypted_content"] == (
        "encrypted-state"
    )


def test_openai_responses_retains_validated_standard_response_envelope() -> None:
    wire = terminal_response([])
    wire.update(
        {
            "service_tier": "priority",
            "prompt_cache_key": "cache-key",
            "temperature": 0.2,
            "top_p": 0.9,
            "metadata": {"trace": "abc"},
            "text": {"format": {"type": "text"}, "verbosity": "low"},
        }
    )

    response = OpenAIResponsesCodec(model="gpt-test").decode_response(wire)

    retained = cast(dict[str, Any], response.metadata["responses"])
    assert "output" not in retained
    assert retained == {key: value for key, value in wire.items() if key != "output"}


@pytest.mark.parametrize(
    ("reason", "finish_reason"),
    ((None, "incomplete"), ("max_output_tokens", "length"), ("content_filter", "content_filter")),
)
def test_openai_responses_incomplete_details_are_closed_schema(
    reason: str | None, finish_reason: str
) -> None:
    response = terminal_response(
        [
            _message(
                "msg-incomplete",
                [{"type": "output_text", "text": "partial", "annotations": []}],
                status="incomplete",
            )
        ],
        status="incomplete",
    )
    response["incomplete_details"] = {"reason": reason}
    assert (
        OpenAIResponsesCodec(model="gpt-test").decode_response(response).finish_reason
        == finish_reason
    )

    response["incomplete_details"] = {"reason": "vendor"}
    with pytest.raises(OpenAIResponsesError, match="incomplete reason"):
        OpenAIResponsesCodec(model="gpt-test").decode_response(response)


def test_openai_responses_profile_and_request_encode_native_responses() -> None:
    profile = _openai_web_profile()
    codec = OpenAIResponsesCodec(model="gpt-test", profile=profile)
    request = ModelRequest(
        messages=(Message.system("policy"), Message.user("question")),
        runtime_tools=(StructuredToolSpec("lookup", "lookup", {"type": "object"}),),
        provider_tools=(ProviderToolSpec(_OPENAI_WEB),),
        tool_choice=ToolChoice(
            type="provider",
            provider_tool=_OPENAI_WEB,
            allow_parallel_runtime_tool_calls=False,
        ),
    )

    payload = codec.encode_request(request)

    assert profile.capabilities.input_modalities == frozenset({"text"})
    assert profile.capabilities.output_modalities == frozenset({"text"})
    assert profile.capabilities.provider_tools == frozenset({_OPENAI_WEB})
    assert payload["model"] == "gpt-test"
    assert payload["tool_choice"] == {"type": "web_search"}
    assert payload["tools"] == [
        {
            "type": "function",
            "name": "lookup",
            "description": "lookup",
            "parameters": {"type": "object"},
            "strict": False,
        },
        {"type": "web_search"},
    ]
    assert [item["role"] for item in cast(list[dict[str, Any]], payload["input"])] == [
        "system",
        "user",
    ]
    assert payload["store"] is False
    assert "previous_response_id" not in payload
    assert "parallel_tool_calls" not in payload

    provider_only_payload = codec.encode_request(
        ModelRequest(
            messages=(Message.user("search"),),
            provider_tools=(ProviderToolSpec(_OPENAI_WEB),),
            tool_choice=ToolChoice(allow_parallel_runtime_tool_calls=False),
        )
    )
    assert "parallel_tool_calls" not in provider_only_payload

    custom = codec.encode_request(
        ModelRequest(
            messages=(Message.user("patch"),),
            runtime_tools=(FreeformToolSpec("apply_patch", "must not reach wire"),),
            tool_choice=ToolChoice(type="required"),
        )
    )
    assert custom["tools"] == [
        {"type": "custom", "name": "apply_patch", "description": "must not reach wire"}
    ]
    with pytest.raises(ValueError, match="input modality"):
        OpenAIResponsesProfile(
            capabilities=replace(
                OpenAIResponsesProfile().capabilities,
                input_modalities=frozenset({"audio"}),
            )
        )
    with pytest.raises(ValueError, match="output modality"):
        OpenAIResponsesProfile(
            capabilities=replace(
                OpenAIResponsesProfile().capabilities,
                output_modalities=frozenset({"image"}),
            )
        )
    with pytest.raises(ValueError, match="does not support seed"):
        OpenAIResponsesProfile(
            capabilities=replace(
                OpenAIResponsesProfile().capabilities,
                seed=True,
            )
        )
    with pytest.raises(ValueError, match="unsupported OpenAI Responses provider tool"):
        OpenAIResponsesProfile(
            capabilities=replace(
                OpenAIResponsesProfile().capabilities,
                provider_tools=frozenset({ProviderToolId("test", "computer")}),
                tool_choice_types=frozenset({"auto", "none", "required", "runtime", "provider"}),
            ),
        )


def test_openai_responses_rejects_boolean_function_and_response_schemas() -> None:
    with pytest.raises(OpenAIResponsesError, match="function parameters must be an object"):
        OpenAIResponsesCodec(model="gpt-test").encode_request(
            ModelRequest(
                messages=(Message.user("question"),),
                runtime_tools=(StructuredToolSpec("lookup", "lookup", True),),
            )
        )


@pytest.mark.parametrize(
    ("options", "message"),
    ((ModelOptions(temperature=2.1), "temperature"), (ModelOptions(top_p=-0.1), "top_p")),
)
def test_openai_responses_rejects_out_of_range_sampling_options(
    options: ModelOptions, message: str
) -> None:
    with pytest.raises(OpenAIResponsesError, match=message):
        OpenAIResponsesCodec(model="gpt-test").encode_request(
            ModelRequest(messages=(Message.user("question"),), options=options)
        )


def test_openai_responses_requires_its_protocol_minimum_output_token_limit() -> None:
    with pytest.raises(OpenAIResponsesError, match="at least 16"):
        OpenAIResponsesCodec(model="gpt-test").encode_request(
            ModelRequest(
                messages=(Message.user("question"),),
                options=ModelOptions(max_output_tokens=15),
            )
        )


def test_openai_responses_strict_envelope_usage_and_empty_output() -> None:
    codec = OpenAIResponsesCodec(model="gpt-test")
    empty = terminal_response([])
    response = codec.decode_response(empty)
    assert response.output == ()
    with pytest.raises(OpenAIResponsesError, match="unsupported field"):
        codec.decode_response({**empty, "vendor": True})
    for sdk_or_request_only_field in ("output_text", "store"):
        with pytest.raises(OpenAIResponsesError, match="unsupported field"):
            codec.decode_response({**empty, sdk_or_request_only_field: False})
    with pytest.raises(OpenAIResponsesError, match="cache_write_tokens"):
        codec.decode_response(
            {
                **empty,
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 2,
                },
            }
        )
    function_item = {
        "id": "fc-empty-arguments",
        "type": "function_call",
        "call_id": "call-empty-arguments",
        "name": "lookup",
        "arguments": "",
    }
    call = cast(
        StructuredToolCall, codec.decode_response(terminal_response([function_item])).output[0]
    )
    assert call.arguments is None and call.raw_input == ""


def test_openai_responses_stream_requires_sequence_and_rejects_vendor_event_fields() -> None:
    codec = OpenAIResponsesCodec(model="gpt-test")
    decoder = OpenAIResponsesStreamDecoder(codec, codec.profile)
    with pytest.raises(OpenAIResponsesError, match="sequence_number"):
        decoder.apply_event(
            "response.created",
            {
                "type": "response.created",
                "response": {"id": "resp-1", "object": "response", "status": "in_progress"},
            },
        )
    with pytest.raises(OpenAIResponsesError, match="unsupported field"):
        decoder.apply_event(
            "response.created",
            {
                "type": "response.created",
                "sequence_number": 0,
                "vendor": True,
                "response": {"id": "resp-1", "object": "response", "status": "in_progress"},
            },
        )


def test_openai_responses_provider_calls_reject_nonstandard_items_and_actions() -> None:
    codec = OpenAIResponsesCodec(model="gpt-test", profile=_openai_web_profile())
    base_web = {"id": "ws-1", "type": "web_search_call", "status": "completed"}
    with pytest.raises(OpenAIResponsesError, match="action"):
        codec.decode_response(terminal_response([base_web]))
    with pytest.raises(OpenAIResponsesError, match="action type"):
        codec.decode_response(terminal_response([{**base_web, "action": {"type": "other"}}]))
    with pytest.raises(OpenAIResponsesError, match="unsupported field"):
        codec.decode_response(
            terminal_response(
                [{**base_web, "action": {"type": "search", "query": "q"}, "vendor": True}]
            )
        )
    decoded = codec.decode_response(
        terminal_response(
            [
                {
                    **base_web,
                    "action": {
                        "type": "search",
                        "queries": ["q1", "q2"],
                        "sources": [{"type": "url", "url": "https://example.test"}],
                    },
                }
            ]
        )
    )
    assert decoded.provider_tool_calls()[0].arguments["queries"] == ["q1", "q2"]
    assert codec.decode_response(
        terminal_response([{**base_web, "action": {"type": "open_page", "url": None}}])
    ).provider_tool_calls()
    with pytest.raises(OpenAIResponsesError, match="source url"):
        codec.decode_response(
            terminal_response(
                [
                    {
                        **base_web,
                        "action": {"type": "search", "sources": [{"type": "url", "url": None}]},
                    }
                ]
            )
        )


def test_openai_responses_web_search_uses_only_stable_configuration_and_call_fields() -> None:
    codec = OpenAIResponsesCodec(model="gpt-test", profile=_openai_web_profile())
    payload = codec.encode_request(
        ModelRequest(
            messages=(Message.user("find a bridge"),),
            provider_tools=(
                ProviderToolSpec(
                    _OPENAI_WEB,
                    configuration={
                        "external_web_access": True,
                        "filters": {"allowed_domains": ["openai.com"]},
                        "search_context_size": "high",
                        "user_location": {
                            "country": None,
                            "city": None,
                            "region": None,
                            "timezone": None,
                        },
                    },
                ),
            ),
        )
    )
    assert payload["tools"][0]["filters"] == {"allowed_domains": ["openai.com"]}
    item = {
        "id": "ws-image-1",
        "type": "web_search_call",
        "status": "completed",
        "action": {"type": "search", "queries": ["bridge"]},
    }
    call = codec.decode_response(terminal_response([item])).provider_tool_calls()[0]
    assert call.arguments["queries"] == ["bridge"]
    replay = codec.encode_request(
        ModelRequest(messages=(Message.user("find"), Message.assistant((call,))))
    )
    assert cast(list[dict[str, Any]], replay["input"])[1] == item
    searching_action: dict[str, Any] = {
        "type": "find_in_page",
        "url": "",
        "pattern": "",
    }
    searching_item: dict[str, Any] = {
        **item,
        "status": "searching",
        "action": searching_action,
    }
    searching_call = ProviderToolCall(
        id="ws-image-1",
        tool=_OPENAI_WEB,
        status=ProviderToolStatus.IN_PROGRESS,
        arguments=searching_action,
        metadata={"responses": {"item": searching_item}},
    )
    searching_replay = codec.encode_request(
        ModelRequest(messages=(Message.user("find"), Message.assistant((searching_call,))))
    )
    assert cast(list[dict[str, Any]], searching_replay["input"])[1] == searching_item
    profile = _openai_web_profile()
    decoder = OpenAIResponsesStreamDecoder(
        OpenAIResponsesCodec(model="gpt-test", profile=profile),
        profile,
    )
    _stream_event(
        decoder,
        "response.created",
        0,
        response={"id": "resp-1", "object": "response", "status": "in_progress"},
    )
    _stream_event(
        decoder,
        "response.output_item.added",
        1,
        output_index=0,
        item={
            "id": "ws-image-1",
            "type": "web_search_call",
            "status": "in_progress",
            "action": {"type": "search", "query": "bridge"},
        },
    )
    _stream_event(
        decoder,
        "response.output_item.done",
        2,
        output_index=0,
        item=item,
    )
    terminal, _ = _stream_event(
        decoder,
        "response.completed",
        3,
        response=terminal_response([item]),
    )
    assert terminal is True
    assert decoder.completed_response().output[0] == call
    for configuration in (
        {"search_content_types": ["image"]},
        {"image_settings": {"max_results": 1}},
        {"return_token_budget": "unlimited"},
        {"filters": {"blocked_domains": ["x"]}},
        {"user_location": {"country": "USA"}},
        {"user_location": {"type": None}},
    ):
        with pytest.raises(OpenAIResponsesError):
            codec.encode_request(
                ModelRequest(
                    messages=(Message.user("find"),),
                    provider_tools=(ProviderToolSpec(_OPENAI_WEB, configuration=configuration),),
                )
            )
    with pytest.raises(OpenAIResponsesError, match="unsupported field"):
        codec.decode_response(terminal_response([{**item, "results": [{"type": "image_result"}]}]))

    image_codec = OpenAIResponsesCodec(
        model="gpt-test", profile=_openai_feature_profile(image_generation=True)
    )
    base_image = {"id": "ig-1", "type": "image_generation_call", "status": "failed"}
    assert image_codec.decode_response(terminal_response([base_image])).provider_tool_calls()
    with pytest.raises(OpenAIResponsesError, match="unsupported field"):
        image_codec.decode_response(
            terminal_response([{**base_image, "result": None, "vendor": True}])
        )
    with pytest.raises(OpenAIResponsesError, match="incomplete"):
        image_codec.decode_response(terminal_response([{**base_image, "status": "incomplete"}]))
    assert image_codec.encode_request(
        ModelRequest(
            messages=(Message.user("draw"),),
            provider_tools=(
                ProviderToolSpec(_OPENAI_IMAGE, {"model": "gpt-image-2", "size": "1536x864"}),
            ),
        )
    )["tools"] == [{"type": "image_generation", "model": "gpt-image-2", "size": "1536x864"}]
    assert image_codec.encode_request(
        ModelRequest(
            messages=(Message.user("draw"),),
            provider_tools=(ProviderToolSpec(_OPENAI_IMAGE, {"input_fidelity": None}),),
        )
    )["tools"] == [{"type": "image_generation", "input_fidelity": None}]
    with pytest.raises(OpenAIResponsesError, match="input_image_mask"):
        image_codec.encode_request(
            ModelRequest(
                messages=(Message.user("draw"),),
                provider_tools=(ProviderToolSpec(_OPENAI_IMAGE, {"input_image_mask": None}),),
            )
        )

    default = OpenAIResponsesProfile()
    codec = OpenAIResponsesCodec(
        model="gpt-test",
        profile=replace(
            default,
            capabilities=replace(default.capabilities, structured_output=True),
        ),
    )
    with pytest.raises(OpenAIResponsesError, match="JSON schema must be an object"):
        codec.encode_request(
            ModelRequest(
                messages=(Message.user("question"),),
                response_format=ResponseFormat("json_schema", True),
            )
        )


def test_openai_responses_custom_tool_terminal_history_and_output_round_trip() -> None:
    profile = _openai_web_profile()
    codec = OpenAIResponsesCodec(model="gpt-test", profile=profile)
    wire_item = {
        "id": "ct-item-1",
        "type": "custom_tool_call",
        "call_id": "ct-call-1",
        "name": "apply_patch",
        "input": "*** Begin Patch\n*** End Patch",
    }

    response = codec.decode_response(terminal_response([wire_item]))

    call = cast(FreeformToolCall, response.runtime_tool_calls()[0])
    assert (call.id, call.name, call.input) == (
        "ct-call-1",
        "apply_patch",
        "*** Begin Patch\n*** End Patch",
    )
    payload = codec.encode_request(
        ModelRequest(
            messages=(
                Message.user("patch"),
                response.to_assistant_message(),
                Message.tool(
                    "ct-call-1",
                    ToolSuccess((ContentPart.text_part("Done!"),)),
                ),
            ),
            runtime_tools=(FreeformToolSpec("apply_patch", "not emitted"),),
            tool_choice=ToolChoice(type="none"),
        )
    )

    assert payload["tools"] == [
        {"type": "custom", "name": "apply_patch", "description": "not emitted"}
    ]
    assert cast(list[dict[str, Any]], payload["input"])[1:] == [
        {
            "id": "ct-item-1",
            "type": "custom_tool_call",
            "call_id": "ct-call-1",
            "name": "apply_patch",
            "input": "*** Begin Patch\n*** End Patch",
        },
        {
            "type": "custom_tool_call_output",
            "call_id": "ct-call-1",
            "output": "Done!",
        },
    ]


def test_openai_responses_custom_tool_stream_round_trip() -> None:
    profile = _openai_web_profile()
    decoder = OpenAIResponsesStreamDecoder(
        OpenAIResponsesCodec(model="gpt-test", profile=profile),
        profile,
    )
    final_item = {
        "id": "ct-item-1",
        "type": "custom_tool_call",
        "call_id": "ct-call-1",
        "name": "apply_patch",
        "input": "*** Begin Patch\n*** End Patch",
    }
    _stream_event(
        decoder,
        "response.created",
        0,
        response={"id": "resp-1", "object": "response", "status": "in_progress"},
    )
    _, added = _stream_event(
        decoder,
        "response.output_item.added",
        1,
        output_index=0,
        item={**final_item, "input": ""},
    )
    _, streamed = _stream_event(
        decoder,
        "response.custom_tool_call_input.delta",
        2,
        item_id="ct-item-1",
        output_index=0,
        delta="*** Begin Patch\n*** End Patch",
    )
    _stream_event(
        decoder,
        "response.custom_tool_call_input.done",
        3,
        item_id="ct-item-1",
        output_index=0,
        input="*** Begin Patch\n*** End Patch",
    )
    _stream_event(
        decoder,
        "response.output_item.done",
        4,
        output_index=0,
        item=final_item,
    )
    terminal, _ = _stream_event(
        decoder,
        "response.completed",
        5,
        response=terminal_response([final_item]),
    )

    assert terminal is True
    assert added == [
        ModelRuntimeToolCallDelta(
            output_index=0,
            input_kind=RuntimeToolKind.FREEFORM,
            input_delta="",
            id="ct-call-1",
            name="apply_patch",
            metadata={
                "responses": {
                    "item": {
                        "id": "ct-item-1",
                        "type": "custom_tool_call",
                        "call_id": "ct-call-1",
                        "name": "apply_patch",
                        "input": "",
                    }
                }
            },
        )
    ]
    assert streamed == [
        ModelRuntimeToolCallDelta(
            output_index=0,
            input_kind=RuntimeToolKind.FREEFORM,
            input_delta="*** Begin Patch\n*** End Patch",
        )
    ]
    call = cast(FreeformToolCall, decoder.completed_response().runtime_tool_calls()[0])
    assert (call.id, call.name, call.input) == (
        "ct-call-1",
        "apply_patch",
        "*** Begin Patch\n*** End Patch",
    )


def test_openai_responses_parallel_control_applies_only_when_runtime_calls_can_be_parallel() -> (
    None
):
    runtime_tool = StructuredToolSpec("lookup", "lookup", {"type": "object"})
    request = ModelRequest(
        messages=(Message.user("question"),),
        runtime_tools=(runtime_tool,),
        tool_choice=ToolChoice(allow_parallel_runtime_tool_calls=False),
    )

    controllable = OpenAIResponsesCodec(model="gpt-test").encode_request(request)
    assert controllable["parallel_tool_calls"] is False

    default_profile = OpenAIResponsesProfile()
    serial_profile = replace(
        default_profile,
        capabilities=replace(
            default_profile.capabilities,
            parallel_runtime_tool_calls=False,
            parallel_runtime_tool_call_control=False,
        ),
    )
    inherently_serial = OpenAIResponsesCodec(
        model="gpt-test",
        profile=serial_profile,
    ).encode_request(request)
    assert "parallel_tool_calls" not in inherently_serial

    uncontrollable_profile = replace(
        default_profile,
        capabilities=replace(
            default_profile.capabilities,
            parallel_runtime_tool_call_control=False,
        ),
    )
    with pytest.raises(OpenAIResponsesError, match="parallel runtime tool calls"):
        OpenAIResponsesCodec(
            model="gpt-test",
            profile=uncontrollable_profile,
        ).encode_request(request)


def test_openai_responses_rejects_non_native_assistant_history() -> None:
    with pytest.raises(OpenAIResponsesError, match="native response output message"):
        OpenAIResponsesCodec(model="gpt-test").encode_request(
            ModelRequest(
                messages=(
                    Message.user("question"),
                    Message.assistant((ContentPart.text_part("answer"),)),
                ),
            )
        )


def test_openai_responses_replays_standard_message_phase_and_incomplete_status() -> None:
    codec = OpenAIResponsesCodec(model="gpt-test")
    response = codec.decode_response(
        terminal_response(
            [
                {
                    "id": "msg-1",
                    "type": "message",
                    "status": "incomplete",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": [{"type": "output_text", "text": "partial", "annotations": []}],
                }
            ],
            status="incomplete",
        )
    )

    replay = codec.encode_request(
        ModelRequest(messages=(Message.user("question"), response.to_assistant_message()))
    )
    assert cast(list[dict[str, Any]], replay["input"])[1] == {
        "id": "msg-1",
        "type": "message",
        "status": "incomplete",
        "role": "assistant",
        "phase": "commentary",
        "content": [{"type": "output_text", "text": "partial", "annotations": []}],
    }


@pytest.mark.parametrize(
    ("item_type", "missing_field"),
    (("function_call", "arguments"), ("custom_tool_call", "input")),
)
def test_openai_responses_stream_requires_runtime_call_input_fields(
    item_type: str,
    missing_field: str,
) -> None:
    codec = OpenAIResponsesCodec(model="gpt-test")
    decoder = OpenAIResponsesStreamDecoder(codec, codec.profile)
    _stream_event(
        decoder,
        "response.created",
        0,
        response={"id": "resp-1", "object": "response", "status": "in_progress"},
    )
    item = {
        "id": "call-item-1",
        "type": item_type,
        "call_id": "call-1",
        "name": "tool",
    }
    if item_type == "function_call":
        item["status"] = "in_progress"
    with pytest.raises(OpenAIResponsesError, match=missing_field):
        _stream_event(decoder, "response.output_item.added", 1, output_index=0, item=item)


def test_openai_responses_accepts_exact_custom_tool_choice() -> None:
    codec = OpenAIResponsesCodec(
        model="gpt-test",
        profile=_openai_web_profile(),
    )

    payload = codec.encode_request(
        ModelRequest(
            messages=(Message.user("patch"),),
            runtime_tools=(FreeformToolSpec("apply_patch", "apply a patch"),),
            tool_choice=ToolChoice(type="runtime", name="apply_patch"),
        )
    )
    assert payload["tool_choice"] == {"type": "custom", "name": "apply_patch"}


async def test_openai_responses_nonstream_client_preserves_interleaved_output_order() -> None:
    captured: dict[str, object] = {}
    wire_response = terminal_response(
        [
            {
                "id": "ws-1",
                "type": "web_search_call",
                "status": "completed",
                "action": {"type": "search", "query": "first"},
            },
            _message(
                "msg-1",
                [{"type": "output_text", "text": "one", "annotations": []}],
                status="completed",
            ),
            {
                "id": "ws-2",
                "type": "web_search_call",
                "status": "failed",
                "action": {"type": "search", "query": "second"},
            },
            _message(
                "msg-2",
                [{"type": "output_text", "text": "two", "annotations": []}],
                status="completed",
            ),
        ],
        model="gpt-test",
    )

    async def handler(raw: httpx.Request) -> httpx.Response:
        captured["url"] = str(raw.url)
        captured["body"] = json.loads(raw.content)
        return httpx.Response(200, json=wire_response, request=raw)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAIResponsesModel(
            base_url="https://api.openai.test/v1",
            api_key="secret",
            model="gpt-test",
            profile=_openai_web_profile(),
            client=client,
        )
        response = await model.invoke(
            ModelRequest(
                messages=(Message.user("question"),),
                provider_tools=(ProviderToolSpec(_OPENAI_WEB),),
            ),
            RunContext("run-1", time()),
            stream=False,
            emit_delta=None,
        )

    assert captured["url"] == "https://api.openai.test/v1/responses"
    assert cast(dict[str, object], captured["body"])["store"] is False
    assert [type(item) for item in response.output] == [
        ProviderToolCall,
        ContentPart,
        ProviderToolCall,
        ContentPart,
    ]
    assert [part.text for part in response.visible_parts()] == ["one", "two"]
    first, _, failed, _ = response.output
    assert isinstance(first, ProviderToolCall)
    assert first.status is ProviderToolStatus.COMPLETED
    assert isinstance(failed, ProviderToolCall)
    assert failed.status is ProviderToolStatus.FAILED
    assert failed.error is not None and failed.error.code == "web_search_failed"


def test_openai_responses_terminal_function_call_preserves_optional_status() -> None:
    codec = OpenAIResponsesCodec(model="gpt-test")

    for status in ("in_progress", "completed", "incomplete"):
        response = codec.decode_response(
            terminal_response(
                [
                    {
                        "id": "fc-1",
                        "type": "function_call",
                        "status": status,
                        "call_id": "call-1",
                        "name": "lookup",
                        "arguments": "{}",
                    }
                ]
            )
        )
        call = cast(StructuredToolCall, response.output[0])
        assert call.metadata["responses"]["item"]["status"] == status

    response = codec.decode_response(
        terminal_response(
            [
                {
                    "id": "fc-1",
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "lookup",
                    "arguments": "{}",
                }
            ]
        )
    )
    call = cast(StructuredToolCall, response.runtime_tool_calls()[0])
    assert (call.id, call.name, call.arguments, call.raw_input) == (
        "call-1",
        "lookup",
        {},
        None,
    )

    with pytest.raises(OpenAIResponsesError, match="incomplete Responses"):
        codec.decode_response(
            terminal_response(
                [
                    {
                        "id": "fc-1",
                        "type": "function_call",
                        "status": "completed",
                        "call_id": "call-1",
                        "name": "lookup",
                        "arguments": "{}",
                    }
                ],
                status="incomplete",
            )
        )


def test_openai_responses_runtime_calls_round_trip_standard_metadata_and_raw_arguments() -> None:
    codec = OpenAIResponsesCodec(model="gpt-test")
    item = {
        "id": "fc-raw-1",
        "type": "function_call",
        "status": "in_progress",
        "call_id": "call-raw-1",
        "name": "lookup",
        "arguments": "{",
        "caller": {"type": "program", "caller_id": "program-1"},
        "namespace": "tools",
    }
    response = codec.decode_response(terminal_response([item]))
    call = cast(StructuredToolCall, response.runtime_tool_calls()[0])
    assert call.arguments is None and call.raw_input == "{"
    payload = codec.encode_request(
        ModelRequest(messages=(Message.user("q"), response.to_assistant_message()))
    )
    assert cast(list[dict[str, Any]], payload["input"])[1] == item
    with pytest.raises(OpenAIResponsesError, match="unsupported field"):
        codec.decode_response(terminal_response([{**item, "reasoning_content": "vendor"}]))
    with pytest.raises(OpenAIResponsesError, match="custom_tool_call contains unsupported field"):
        codec.decode_response(
            terminal_response(
                [
                    {
                        "id": "ct-1",
                        "type": "custom_tool_call",
                        "call_id": "ct-call-1",
                        "name": "custom",
                        "input": "",
                        "status": "completed",
                    }
                ]
            )
        )


def test_openai_responses_reasoning_preserves_an_absent_status() -> None:
    profile = OpenAIResponsesProfile(store=True, include=frozenset())
    codec = OpenAIResponsesCodec(model="gpt-test", profile=profile)
    reasoning = {
        "id": "rs-no-status",
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "summary"}],
    }
    response = codec.decode_response(terminal_response([reasoning]))
    payload = codec.encode_request(
        ModelRequest(messages=(Message.user("q"), response.to_assistant_message()))
    )
    assert cast(list[dict[str, Any]], payload["input"])[1] == reasoning


def test_openai_responses_reasoning_sse_tracks_open_part_and_uses_terminal_response() -> None:
    profile = OpenAIResponsesProfile(include=frozenset({"reasoning.encrypted_content"}))
    codec = OpenAIResponsesCodec(model="gpt-test", profile=profile)
    decoder = OpenAIResponsesStreamDecoder(codec, profile)
    reasoning_item = {
        "id": "rs-1",
        "type": "reasoning",
        "status": "completed",
        "content": [{"type": "reasoning_text", "text": "分析"}],
        "summary": [],
        "encrypted_content": "encrypted-state",
    }

    _stream_event(
        decoder,
        "response.created",
        0,
        response={"id": "resp-1", "object": "response", "status": "in_progress"},
    )
    _stream_event(
        decoder,
        "response.output_item.added",
        1,
        output_index=0,
        item={**reasoning_item, "status": "in_progress", "content": []},
    )
    _stream_event(
        decoder,
        "response.content_part.added",
        2,
        item_id="rs-1",
        output_index=0,
        content_index=0,
        part={"type": "reasoning_text", "text": ""},
    )
    _, reasoning_deltas = _stream_event(
        decoder,
        "response.reasoning_text.delta",
        3,
        item_id="rs-1",
        output_index=0,
        content_index=0,
        delta="分析",
    )
    _stream_event(
        decoder,
        "response.reasoning_text.done",
        4,
        item_id="rs-1",
        output_index=0,
        content_index=0,
        text="分析",
    )
    with pytest.raises(OpenAIResponsesError, match="emitted twice"):
        _stream_event(
            decoder,
            "response.reasoning_text.done",
            5,
            item_id="rs-1",
            output_index=0,
            content_index=0,
            text="分析",
        )
    _stream_event(
        decoder,
        "response.content_part.done",
        7,
        item_id="rs-1",
        output_index=0,
        content_index=0,
        part={"type": "reasoning_text", "text": "分析"},
    )
    _stream_event(
        decoder,
        "response.output_item.done",
        8,
        output_index=0,
        item=reasoning_item,
    )
    terminal, usage_deltas = _stream_event(
        decoder,
        "response.completed",
        9,
        response=terminal_response([reasoning_item]),
    )

    assert len(reasoning_deltas) == 1
    assert isinstance(reasoning_deltas[0], ModelReasoningDelta)
    assert reasoning_deltas[0].text_delta == "分析"
    assert terminal is True
    assert len(usage_deltas) == 1 and isinstance(usage_deltas[0], ModelUsageDelta)
    completed = decoder.completed_response()
    reasoning = completed.output[0]
    assert isinstance(reasoning, ContentPart)
    assert reasoning.type == "reasoning"
    assert reasoning.text == "分析"


def test_openai_responses_web_search_completed_lifecycle_is_terminal() -> None:
    profile = _openai_web_profile()
    decoder = OpenAIResponsesStreamDecoder(
        OpenAIResponsesCodec(model="gpt-test", profile=profile),
        profile,
    )
    action = {"type": "search", "query": "JHarness"}
    final_item = {
        "id": "ws-1",
        "type": "web_search_call",
        "status": "completed",
        "action": action,
    }
    _stream_event(
        decoder,
        "response.created",
        0,
        response={"id": "resp-1", "object": "response", "status": "in_progress"},
    )
    _, added = _stream_event(
        decoder,
        "response.output_item.added",
        1,
        output_index=0,
        item={
            "id": "ws-1",
            "type": "web_search_call",
            "status": "searching",
            "action": action,
        },
    )
    _, lifecycle = _stream_event(
        decoder,
        "response.web_search_call.completed",
        2,
        item_id="ws-1",
        output_index=0,
    )
    _, done = _stream_event(
        decoder,
        "response.output_item.done",
        3,
        output_index=0,
        item=final_item,
    )
    terminal, _ = _stream_event(
        decoder,
        "response.completed",
        4,
        response=terminal_response([final_item]),
    )

    deltas = [added[0], lifecycle[0]]
    assert all(isinstance(delta, ModelProviderToolCallDelta) for delta in deltas)
    assert [cast(ModelProviderToolCallDelta, delta).status for delta in deltas] == [
        ProviderToolStatus.IN_PROGRESS,
        ProviderToolStatus.COMPLETED,
    ]
    assert cast(ModelProviderToolCallDelta, lifecycle[0]).event == (
        "response.web_search_call.completed"
    )
    assert cast(ModelProviderToolCallDelta, lifecycle[0]).data == {}
    assert terminal is True
    completed_call = decoder.completed_response().output[0]
    assert isinstance(completed_call, ProviderToolCall)
    assert done == []
    assert completed_call.status is ProviderToolStatus.COMPLETED


def _generic_web_search_stream_decoder() -> OpenAIResponsesStreamDecoder:
    profile = _openai_web_profile()
    return OpenAIResponsesStreamDecoder(
        OpenAIResponsesCodec(model="gpt-test", profile=profile),
        profile,
    )


def test_openai_responses_provider_lifecycle_rejects_conflicting_terminal_statuses() -> None:
    decoder = _generic_web_search_stream_decoder()
    _stream_event(
        decoder,
        "response.created",
        0,
        response={"id": "resp-1", "object": "response", "status": "in_progress"},
    )
    _stream_event(
        decoder,
        "response.output_item.added",
        1,
        output_index=0,
        item={
            "id": "ws-1",
            "type": "web_search_call",
            "status": "searching",
            "action": {"type": "search", "query": "JHarness"},
        },
    )
    _stream_event(
        decoder,
        "response.web_search_call.completed",
        2,
        item_id="ws-1",
        output_index=0,
    )
    with pytest.raises(OpenAIResponsesError, match="cannot return to in_progress"):
        _stream_event(
            decoder,
            "response.web_search_call.in_progress",
            3,
            item_id="ws-1",
            output_index=0,
        )
    with pytest.raises(OpenAIResponsesError, match="conflicting terminal statuses"):
        _stream_event(
            decoder,
            "response.output_item.done",
            4,
            output_index=0,
            item={
                "id": "ws-1",
                "type": "web_search_call",
                "status": "failed",
                "action": {"type": "search", "query": "JHarness"},
            },
        )


def test_openai_responses_provider_output_item_done_requires_a_terminal_status() -> None:
    decoder = _generic_web_search_stream_decoder()
    _stream_event(
        decoder,
        "response.created",
        0,
        response={"id": "resp-1", "object": "response", "status": "in_progress"},
    )
    _stream_event(
        decoder,
        "response.output_item.added",
        1,
        output_index=0,
        item={
            "id": "ws-1",
            "type": "web_search_call",
            "status": "searching",
            "action": {"type": "search", "query": "JHarness"},
        },
    )

    with pytest.raises(OpenAIResponsesError, match="requires a terminal status"):
        _stream_event(
            decoder,
            "response.output_item.done",
            2,
            output_index=0,
            item={
                "id": "ws-1",
                "type": "web_search_call",
                "status": "searching",
                "action": {"type": "search", "query": "JHarness"},
            },
        )


def test_openai_responses_partial_image_accepts_nullable_rendering_fields() -> None:
    profile = _openai_feature_profile(image_generation=True)
    decoder = OpenAIResponsesStreamDecoder(
        OpenAIResponsesCodec(model="gpt-test", profile=profile), profile
    )
    _stream_event(
        decoder,
        "response.created",
        0,
        response={"id": "resp-1", "object": "response", "status": "in_progress"},
    )
    _stream_event(
        decoder,
        "response.output_item.added",
        1,
        output_index=0,
        item={"id": "ig-1", "type": "image_generation_call", "status": "generating"},
    )
    _, deltas = _stream_event(
        decoder,
        "response.image_generation_call.partial_image",
        2,
        item_id="ig-1",
        output_index=0,
        partial_image_index=0,
        partial_image_b64=_JPEG_BASE64,
        background=None,
        output_format=None,
        quality=None,
        size=None,
    )
    assert cast(ModelProviderToolCallDelta, deltas[0]).data == {
        "base64": _JPEG_BASE64,
        "partial_image_index": 0,
    }


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_openai_responses_terminal_response_accepts_provider_status_matching_output_item_done(
    status: str,
) -> None:
    decoder = _generic_web_search_stream_decoder()
    final_item = {
        "id": "ws-1",
        "type": "web_search_call",
        "status": status,
        "action": {"type": "search", "query": "JHarness"},
    }
    _stream_event(
        decoder,
        "response.created",
        0,
        response={"id": "resp-1", "object": "response", "status": "in_progress"},
    )
    _stream_event(
        decoder,
        "response.output_item.added",
        1,
        output_index=0,
        item={
            "id": "ws-1",
            "type": "web_search_call",
            "status": "searching",
            "action": {"type": "search", "query": "JHarness"},
        },
    )
    _stream_event(
        decoder,
        "response.output_item.done",
        2,
        output_index=0,
        item=final_item,
    )
    terminal, _ = _stream_event(
        decoder,
        "response.completed",
        3,
        response=terminal_response([final_item], model="gpt-test"),
    )

    assert terminal is True
    call = decoder.completed_response().output[0]
    assert isinstance(call, ProviderToolCall)
    assert call.status is ProviderToolStatus(status)


@pytest.mark.parametrize(
    ("done_status", "terminal_status"),
    [("failed", "completed"), ("completed", "failed")],
)
def test_openai_responses_terminal_response_rejects_provider_status_mismatching_output_item_done(
    done_status: str,
    terminal_status: str,
) -> None:
    decoder = _generic_web_search_stream_decoder()
    done_item = {
        "id": "ws-1",
        "type": "web_search_call",
        "status": done_status,
        "action": {"type": "search", "query": "JHarness"},
    }
    terminal_item = {**done_item, "status": terminal_status}
    _stream_event(
        decoder,
        "response.created",
        0,
        response={"id": "resp-1", "object": "response", "status": "in_progress"},
    )
    _stream_event(
        decoder,
        "response.output_item.added",
        1,
        output_index=0,
        item={
            "id": "ws-1",
            "type": "web_search_call",
            "status": "searching",
            "action": {"type": "search", "query": "JHarness"},
        },
    )
    _stream_event(
        decoder,
        "response.output_item.done",
        2,
        output_index=0,
        item=done_item,
    )

    with pytest.raises(OpenAIResponsesError, match=r"does not match output_item\.done"):
        _stream_event(
            decoder,
            "response.completed",
            3,
            response=terminal_response([terminal_item], model="gpt-test"),
        )


def test_openai_responses_output_text_annotation_event_validates_the_open_message_part() -> None:
    codec = OpenAIResponsesCodec(model="gpt-test")
    decoder = OpenAIResponsesStreamDecoder(codec, codec.profile)
    _stream_event(
        decoder,
        "response.created",
        0,
        response={"id": "resp-1", "object": "response", "status": "in_progress"},
    )
    _stream_event(
        decoder,
        "response.output_item.added",
        1,
        output_index=0,
        item=_message("msg-1", [], status="in_progress"),
    )
    _stream_event(
        decoder,
        "response.content_part.added",
        2,
        item_id="msg-1",
        output_index=0,
        content_index=0,
        part={"type": "output_text", "text": "", "annotations": []},
    )
    _, deltas = _stream_event(
        decoder,
        "response.output_text.annotation.added",
        7,
        item_id="msg-1",
        output_index=0,
        content_index=0,
        annotation_index=0,
        annotation={
            "type": "url_citation",
            "start_index": 0,
            "end_index": 4,
            "url": "https://example.test/source",
            "title": "source",
        },
    )

    assert deltas == []
    _, null_deltas = _stream_event(
        decoder,
        "response.output_text.annotation.added",
        8,
        item_id="msg-1",
        output_index=0,
        content_index=0,
        annotation_index=1,
        annotation=None,
    )
    assert null_deltas == []


def test_openai_responses_text_stream_validates_nested_standard_shapes() -> None:
    codec = OpenAIResponsesCodec(model="gpt-test")
    decoder = OpenAIResponsesStreamDecoder(codec, codec.profile)
    initial_item: dict[str, Any] = _message("msg-stream", [], status="in_progress")
    final_part: dict[str, Any] = {
        "type": "output_text",
        "text": "hello",
        "annotations": [],
        "logprobs": [],
    }
    final_item: dict[str, Any] = {
        **initial_item,
        "status": "completed",
        "content": [final_part],
    }

    _stream_event(
        decoder,
        "response.created",
        0,
        response={"id": "resp-1", "object": "response", "status": "in_progress"},
    )
    _stream_event(
        decoder,
        "response.output_item.added",
        1,
        output_index=0,
        item=initial_item,
    )
    _stream_event(
        decoder,
        "response.content_part.added",
        2,
        item_id="msg-stream",
        output_index=0,
        content_index=0,
        part={"type": "output_text", "text": "", "annotations": []},
    )
    _, deltas = _stream_event(
        decoder,
        "response.output_text.delta",
        3,
        item_id="msg-stream",
        output_index=0,
        content_index=0,
        delta="hello",
        logprobs=[
            {
                "token": "hello",
                "logprob": -0.1,
                "top_logprobs": [
                    {"token": "hello", "logprob": -0.1},
                    {"token": None, "logprob": None},
                ],
            }
        ],
    )
    assert len(deltas) == 1 and isinstance(deltas[0], ModelContentDelta)
    _stream_event(
        decoder,
        "response.output_text.done",
        4,
        item_id="msg-stream",
        output_index=0,
        content_index=0,
        text="hello",
        logprobs=[],
    )
    _stream_event(
        decoder,
        "response.content_part.done",
        5,
        item_id="msg-stream",
        output_index=0,
        content_index=0,
        part=final_part,
    )
    _stream_event(
        decoder,
        "response.output_item.done",
        6,
        output_index=0,
        item=final_item,
    )
    terminal, _ = _stream_event(
        decoder,
        "response.completed",
        7,
        response=terminal_response([final_item]),
    )
    assert terminal is True


def test_openai_responses_summary_done_and_error_events_use_exact_schemas() -> None:
    profile = OpenAIResponsesProfile(store=True, include=frozenset())
    codec = OpenAIResponsesCodec(model="gpt-test", profile=profile)
    decoder = OpenAIResponsesStreamDecoder(codec, profile)
    _stream_event(
        decoder,
        "response.created",
        0,
        response={"id": "resp-1", "object": "response", "status": "in_progress"},
    )
    _stream_event(
        decoder,
        "response.output_item.added",
        1,
        output_index=0,
        item={"id": "rs-1", "type": "reasoning", "summary": []},
    )
    summary = {"type": "summary_text", "text": ""}
    _stream_event(
        decoder,
        "response.reasoning_summary_part.added",
        2,
        item_id="rs-1",
        output_index=0,
        summary_index=0,
        part=summary,
    )
    _stream_event(
        decoder,
        "response.reasoning_summary_part.done",
        3,
        item_id="rs-1",
        output_index=0,
        summary_index=0,
        part=summary,
        status="incomplete",
    )

    error_decoder = OpenAIResponsesStreamDecoder(codec, profile)
    with pytest.raises(OpenAIResponsesError, match=r"E_TEST.*failed"):
        _stream_event(
            error_decoder,
            "error",
            0,
            code="E_TEST",
            message="failed",
            param=None,
        )
    invalid_error_decoder = OpenAIResponsesStreamDecoder(codec, profile)
    with pytest.raises(OpenAIResponsesError, match="unsupported field"):
        _stream_event(
            invalid_error_decoder,
            "error",
            0,
            code="E_TEST",
            message="failed",
            vendor=True,
        )


def test_openai_responses_provider_only_terminal_response_is_valid() -> None:
    profile = _openai_web_profile()
    response = OpenAIResponsesCodec(
        model="gpt-test",
        profile=profile,
    ).decode_response(
        terminal_response(
            [
                {
                    "id": "ws-1",
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {"type": "search", "query": "only"},
                }
            ],
            model="gpt-test",
        )
    )

    assert len(response.output) == 1
    assert isinstance(response.output[0], ProviderToolCall)
    assert response.visible_parts() == ()
    assert response.finish_reason == "stop"


def test_openai_responses_validates_and_replays_output_text_metadata() -> None:
    codec = OpenAIResponsesCodec(model="gpt-test")
    item = _message(
        "msg-metadata",
        [
            {
                "type": "output_text",
                "text": "source",
                "annotations": [
                    {
                        "type": "url_citation",
                        "start_index": 0,
                        "end_index": 6,
                        "url": "https://example.test",
                        "title": "Example",
                    },
                    {"type": "file_path", "file_id": "file-1", "index": 0},
                    {
                        "type": "container_file_citation",
                        "container_id": "cntr-1",
                        "file_id": "file-2",
                        "filename": "source.txt",
                        "start_index": 0,
                        "end_index": 6,
                    },
                ],
                "logprobs": [
                    {
                        "token": "source",
                        "logprob": -0.1,
                        "bytes": [115],
                        "top_logprobs": [{"token": "source", "logprob": -0.1, "bytes": [115]}],
                    }
                ],
            }
        ],
        status="completed",
    )
    response = codec.decode_response(terminal_response([item]))
    replay = codec.encode_request(
        ModelRequest(messages=(Message.user("question"), response.to_assistant_message()))
    )
    assert cast(list[dict[str, Any]], replay["input"])[1] == item


@pytest.mark.parametrize(
    "content",
    (
        {"type": "output_text", "text": "x", "vendor": True},
        {"type": "refusal", "refusal": "no", "vendor": True},
        {"type": "output_text", "text": "x", "annotations": [{"type": "unknown"}]},
    ),
)
def test_openai_responses_rejects_nonstandard_output_metadata(content: dict[str, Any]) -> None:
    codec = OpenAIResponsesCodec(model="gpt-test")
    with pytest.raises(OpenAIResponsesError, match="unsupported"):
        codec.decode_response(
            terminal_response([_message("msg-invalid", [content], status="completed")])
        )


def test_openai_responses_reasoning_requires_summary_and_closed_blocks() -> None:
    codec = OpenAIResponsesCodec(model="gpt-test")
    invalid: dict[str, object] = {
        "id": "reasoning-invalid",
        "type": "reasoning",
        "status": "completed",
        "encrypted_content": "encrypted",
    }
    with pytest.raises(OpenAIResponsesError, match="requires summary"):
        codec.decode_response(terminal_response([invalid]))
    invalid["summary"] = [{"type": "summary_text", "text": "x", "vendor": True}]
    with pytest.raises(OpenAIResponsesError, match="unsupported field"):
        codec.decode_response(terminal_response([invalid]))


def test_openai_responses_replays_stored_reasoning_without_encrypted_content() -> None:
    profile = OpenAIResponsesProfile(store=True, include=frozenset())
    codec = OpenAIResponsesCodec(model="gpt-test", profile=profile)
    response = codec.decode_response(
        terminal_response(
            [
                {
                    "id": "reasoning-stored",
                    "type": "reasoning",
                    "status": "completed",
                    "summary": [{"type": "summary_text", "text": "summary"}],
                }
            ]
        )
    )
    replay = codec.encode_request(
        ModelRequest(messages=(Message.user("question"), response.to_assistant_message()))
    )
    assert "encrypted_content" not in cast(list[dict[str, Any]], replay["input"])[1]

    with pytest.raises(OpenAIResponsesError, match="in-progress provider tools"):
        OpenAIResponsesCodec(
            model="gpt-test",
            profile=_openai_web_profile(),
        ).decode_response(
            terminal_response(
                [
                    {
                        "id": "ws-2",
                        "type": "web_search_call",
                        "status": "searching",
                        "action": {"type": "search", "query": "pending"},
                    }
                ],
                model="gpt-test",
            )
        )


def test_openai_responses_vision_inputs_encode_url_base64_and_artifact_with_media_validation() -> (
    None
):
    codec = OpenAIResponsesCodec(
        model="gpt-test",
        profile=_openai_feature_profile(image_input=True),
    )
    request = ModelRequest(
        messages=(
            Message(
                "user",
                (
                    ContentPart(type="image", uri="https://images.test/cat.png"),
                    ContentPart(type="image", data={"base64": _JPEG_BASE64}),
                    ContentPart(
                        type="image",
                        media_type="IMAGE/JPEG",
                        data={"base64": _JPEG_BASE64},
                    ),
                    ContentPart(
                        type="file",
                        uri=f"DATA:image/jpeg;base64,{_JPEG_BASE64}",
                    ),
                    ContentPart.artifact_part(ArtifactRef("file-image", media_type="image/png")),
                ),
            ),
        )
    )

    content = cast(list[dict[str, Any]], codec.encode_request(request)["input"])[0]["content"]
    assert content == [
        {"type": "input_image", "image_url": "https://images.test/cat.png", "detail": "auto"},
        {
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{_JPEG_BASE64}",
            "detail": "auto",
        },
        {
            "type": "input_image",
            "image_url": f"data:IMAGE/JPEG;base64,{_JPEG_BASE64}",
            "detail": "auto",
        },
        {
            "type": "input_image",
            "image_url": f"DATA:image/jpeg;base64,{_JPEG_BASE64}",
            "detail": "auto",
        },
        {"type": "input_image", "file_id": "file-image", "detail": "auto"},
    ]

    with pytest.raises(OpenAIResponsesError, match="does not match"):
        codec.encode_request(
            ModelRequest(
                messages=(
                    Message(
                        "user",
                        (
                            ContentPart(
                                type="image",
                                media_type="image/png",
                                data={"base64": _JPEG_BASE64},
                            ),
                        ),
                    ),
                )
            )
        )
    with pytest.raises(OpenAIResponsesError, match="valid base64"):
        codec.encode_request(
            ModelRequest(
                messages=(
                    Message(
                        "user",
                        (
                            ContentPart(
                                type="image",
                                media_type="image/jpeg",
                                data={"base64": f"{_JPEG_BASE64}!"},
                            ),
                        ),
                    ),
                )
            )
        )


async def test_runtime_sends_file_typed_image_data_url_as_responses_image() -> None:
    captured: dict[str, object] = {}
    default = OpenAIResponsesProfile()
    image_only = OpenAIResponsesProfile(
        capabilities=replace(
            default.capabilities,
            input_modalities=frozenset({"text", "image"}),
        )
    )

    async def handler(raw: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(raw.content)
        return httpx.Response(
            200,
            json=terminal_response(
                [
                    _message(
                        "msg-1",
                        [{"type": "output_text", "text": "done", "annotations": []}],
                        status="completed",
                    )
                ],
                model="vision-test",
            ),
            request=raw,
        )

    image_uri = f"DATA:image/jpeg;base64,{_JPEG_BASE64}"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAIResponsesModel(
            base_url="https://provider.test/v1",
            api_key="secret",
            model="vision-test",
            profile=image_only,
            client=client,
        )
        checkpoint = (
            await Runtime(model=model)
            .start((Message("user", (ContentPart("file", uri=image_uri),)),))
            .result()
        )

    assert checkpoint.snapshot.status == "completed"
    body = cast(dict[str, Any], captured["body"])
    assert body["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_image", "image_url": image_uri, "detail": "auto"}],
        }
    ]


def test_openai_responses_image_generation_decodes_media_type_and_replays_base64_history() -> None:
    codec = OpenAIResponsesCodec(
        model="gpt-test",
        profile=_openai_feature_profile(image_generation=True),
    )
    wire = terminal_response(
        [
            {
                "id": "ig-1",
                "type": "image_generation_call",
                "status": "completed",
                "result": _JPEG_BASE64,
            }
        ],
        tools=[{"type": "image_generation", "output_format": "jpeg"}],
    )

    response = codec.decode_response(wire)
    call = response.provider_tool_calls()[0]
    assert call.tool == _OPENAI_IMAGE
    assert call.output[0].type == "image"
    assert call.output[0].media_type == "image/jpeg"
    assert call.output[0].data["base64"] == _JPEG_BASE64
    native_item = cast(dict[str, Any], call.metadata["responses"])["item"]
    assert "result" not in native_item

    payload = codec.encode_request(
        ModelRequest(
            messages=(Message.user("draw"), response.to_assistant_message()),
            provider_tools=(ProviderToolSpec(_OPENAI_IMAGE, {"output_format": "jpeg"}),),
        )
    )
    replay = cast(list[dict[str, Any]], payload["input"])[1]
    assert replay == {
        "id": "ig-1",
        "type": "image_generation_call",
        "status": "completed",
        "result": _JPEG_BASE64,
    }

    mismatched = terminal_response(
        cast(list[dict[str, Any]], wire["output"]),
        tools=[{"type": "image_generation", "output_format": "png"}],
    )
    with pytest.raises(OpenAIResponsesError, match="does not match"):
        codec.decode_response(mismatched)


async def test_openai_responses_image_generation_externalizes_and_hydrates_artifact_history() -> (
    None
):
    profile = _openai_feature_profile(image_generation=True)
    store = _MemoryOpenAIResponsesArtifactStore()
    captured: list[dict[str, Any]] = []
    responses = [
        terminal_response(
            [
                {
                    "id": "ig-1",
                    "type": "image_generation_call",
                    "status": "completed",
                    "result": _JPEG_BASE64,
                }
            ],
            tools=[{"type": "image_generation", "output_format": "jpeg"}],
        ),
        terminal_response(
            [
                _message(
                    "msg-2",
                    [{"type": "output_text", "text": "saved", "annotations": []}],
                    status="completed",
                )
            ]
        ),
    ]

    async def handler(raw: httpx.Request) -> httpx.Response:
        captured.append(json.loads(raw.content))
        return httpx.Response(200, json=responses.pop(0), request=raw)

    assert isinstance(store, OpenAIResponsesArtifactStore)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        without_store = OpenAIResponsesModel(
            base_url="https://provider.test/v1",
            api_key="secret",
            model="gpt-test",
            profile=profile,
            client=client,
        )
        image_request = ModelRequest(
            messages=(Message.user("draw"),),
            provider_tools=(ProviderToolSpec(_OPENAI_IMAGE, {"output_format": "jpeg"}),),
        )
        with pytest.raises(ValueError, match="OpenAIResponsesArtifactStore"):
            await without_store.invoke(
                image_request,
                RunContext("run-unsafe", time()),
                stream=False,
                emit_delta=None,
            )

        model = OpenAIResponsesModel(
            base_url="https://provider.test/v1",
            api_key="secret",
            model="gpt-test",
            profile=profile,
            artifact_store=store,
            client=client,
        )
        context = RunContext("run-1", time())
        generated = await model.invoke(
            image_request,
            context,
            stream=False,
            emit_delta=None,
        )
        call = generated.provider_tool_calls()[0]
        assert call.output[0].type == "artifact"
        digest = sha256(_JPEG_BYTES).hexdigest()
        artifact_ref = f"artifact:sha256:{digest}"
        assert call.output[0].artifact == ArtifactRef(
            artifact_ref,
            media_type="image/jpeg",
            size_bytes=len(_JPEG_BYTES),
            sha256=digest,
        )
        assert "base64" not in call.output[0].data

        completed = await model.invoke(
            ModelRequest(
                messages=(
                    Message.user("draw"),
                    generated.to_assistant_message(),
                    Message.user("confirm"),
                ),
                provider_tools=(ProviderToolSpec(_OPENAI_IMAGE, {"output_format": "jpeg"}),),
            ),
            context,
            stream=False,
            emit_delta=None,
        )

    replay = cast(list[dict[str, Any]], captured[1]["input"])[1]
    assert replay["result"] == _JPEG_BASE64
    assert store.saved == [artifact_ref]
    assert store.loaded == [artifact_ref]
    assert completed.visible_parts()[0].text == "saved"


async def test_openai_responses_unrequested_inline_image_result_requires_artifact_store() -> None:
    profile = _openai_feature_profile(image_generation=True)
    wire = terminal_response(
        [
            {
                "id": "ig-unrequested",
                "type": "image_generation_call",
                "status": "completed",
                "result": _JPEG_BASE64,
            }
        ],
        tools=[{"type": "image_generation", "output_format": "jpeg"}],
    )

    async def handler(raw: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=wire, request=raw)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAIResponsesModel(
            base_url="https://provider.test/v1",
            api_key="secret",
            model="gpt-test",
            profile=profile,
            client=client,
        )
        with pytest.raises(ModelError, match="without an OpenAIResponsesArtifactStore") as caught:
            await model.invoke(
                ModelRequest(messages=(Message.user("text only"),)),
                RunContext("run-unrequested", time()),
                stream=False,
                emit_delta=None,
            )
    assert caught.value.info.code == "codec_error"


@pytest.mark.parametrize(
    ("artifact", "error_type", "message"),
    (
        ("not-an-artifact", TypeError, "must return ArtifactRef"),
        (
            ArtifactRef("artifact:missing-size", media_type="image/jpeg"),
            ValueError,
            "requires size_bytes",
        ),
        (
            ArtifactRef(
                "artifact:missing-sha",
                media_type="image/jpeg",
                size_bytes=len(_JPEG_BYTES),
            ),
            ValueError,
            "requires sha256",
        ),
        (
            ArtifactRef(
                "artifact:wrong-media",
                media_type="image/png",
                size_bytes=len(_JPEG_BYTES),
                sha256=sha256(_JPEG_BYTES).hexdigest(),
            ),
            ValueError,
            "media_type must match",
        ),
        (
            ArtifactRef(
                "artifact:wrong-size",
                media_type="image/jpeg",
                size_bytes=len(_JPEG_BYTES) + 1,
                sha256=sha256(_JPEG_BYTES).hexdigest(),
            ),
            ValueError,
            "size_bytes does not match",
        ),
        (
            ArtifactRef(
                "artifact:wrong-sha",
                media_type="image/jpeg",
                size_bytes=len(_JPEG_BYTES),
                sha256="0" * 64,
            ),
            ValueError,
            "sha256 does not match",
        ),
    ),
)
async def test_openai_responses_image_artifact_store_return_is_fully_validated(
    artifact: object,
    error_type: type[Exception],
    message: str,
) -> None:
    profile = _openai_feature_profile(image_generation=True)
    wire = terminal_response(
        [
            {
                "id": "ig-invalid-artifact",
                "type": "image_generation_call",
                "status": "completed",
                "result": _JPEG_BASE64,
            }
        ],
        tools=[{"type": "image_generation", "output_format": "jpeg"}],
    )

    async def handler(raw: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=wire, request=raw)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAIResponsesModel(
            base_url="https://provider.test/v1",
            api_key="secret",
            model="gpt-test",
            profile=profile,
            artifact_store=_StaticArtifactStore(artifact),
            client=client,
        )
        request = ModelRequest(
            messages=(Message.user("draw"),),
            provider_tools=(ProviderToolSpec(_OPENAI_IMAGE, {"output_format": "jpeg"}),),
        )
        with pytest.raises(error_type, match=message):
            await model.invoke(
                request,
                RunContext("run-invalid-artifact", time()),
                stream=False,
                emit_delta=None,
            )


async def test_openai_responses_image_artifact_save_failure_aborts_the_model_response() -> None:
    profile = _openai_feature_profile(image_generation=True)
    wire = terminal_response(
        [
            {
                "id": "ig-save-failure",
                "type": "image_generation_call",
                "status": "completed",
                "result": _JPEG_BASE64,
            }
        ],
        tools=[{"type": "image_generation", "output_format": "jpeg"}],
    )

    async def handler(raw: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=wire, request=raw)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAIResponsesModel(
            base_url="https://provider.test/v1",
            api_key="secret",
            model="gpt-test",
            profile=profile,
            artifact_store=_FailingArtifactStore(),
            client=client,
        )
        with pytest.raises(OSError, match="artifact save failed"):
            await model.invoke(
                ModelRequest(
                    messages=(Message.user("draw"),),
                    provider_tools=(ProviderToolSpec(_OPENAI_IMAGE, {"output_format": "jpeg"}),),
                ),
                RunContext("run-save-failure", time()),
                stream=False,
                emit_delta=None,
            )


async def test_openai_responses_image_artifact_hydration_rejects_corrupt_stored_bytes() -> None:
    profile = _openai_feature_profile(image_generation=True)
    digest = sha256(_JPEG_BYTES).hexdigest()
    artifact = ArtifactRef(
        f"artifact:sha256:{digest}",
        media_type="image/jpeg",
        size_bytes=len(_JPEG_BYTES),
        sha256=digest,
    )
    store = _MemoryOpenAIResponsesArtifactStore()
    store.values[artifact.ref] = b"x" * len(_JPEG_BYTES)
    history = ProviderToolCall(
        "ig-corrupt",
        _OPENAI_IMAGE,
        ProviderToolStatus.COMPLETED,
        output=(ContentPart.artifact_part(artifact),),
    )

    async def unexpected_handler(raw: httpx.Request) -> httpx.Response:
        raise AssertionError(f"corrupt artifact must fail before HTTP: {raw.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected_handler)) as client:
        model = OpenAIResponsesModel(
            base_url="https://provider.test/v1",
            api_key="secret",
            model="gpt-test",
            profile=profile,
            artifact_store=store,
            client=client,
        )
        request = ModelRequest(
            messages=(
                Message.user("draw"),
                Message.assistant((history,)),
                Message.user("continue"),
            ),
            provider_tools=(ProviderToolSpec(_OPENAI_IMAGE, {"output_format": "jpeg"}),),
        )
        with pytest.raises(ValueError, match="sha256 does not match"):
            await model.invoke(
                request,
                RunContext("run-corrupt", time()),
                stream=False,
                emit_delta=None,
            )


async def test_openai_responses_image_artifact_load_failure_aborts_before_http() -> None:
    profile = _openai_feature_profile(image_generation=True)
    digest = sha256(_JPEG_BYTES).hexdigest()
    artifact = ArtifactRef(
        f"artifact:sha256:{digest}",
        media_type="image/jpeg",
        size_bytes=len(_JPEG_BYTES),
        sha256=digest,
    )
    history = ProviderToolCall(
        "ig-load-failure",
        _OPENAI_IMAGE,
        ProviderToolStatus.COMPLETED,
        output=(ContentPart.artifact_part(artifact),),
    )

    async def unexpected_handler(raw: httpx.Request) -> httpx.Response:
        raise AssertionError(f"load failure must abort before HTTP: {raw.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected_handler)) as client:
        model = OpenAIResponsesModel(
            base_url="https://provider.test/v1",
            api_key="secret",
            model="gpt-test",
            profile=profile,
            artifact_store=_FailingArtifactStore(),
            client=client,
        )
        with pytest.raises(OSError, match="artifact load failed"):
            await model.invoke(
                ModelRequest(
                    messages=(
                        Message.user("draw"),
                        Message.assistant((history,)),
                        Message.user("continue"),
                    ),
                    provider_tools=(ProviderToolSpec(_OPENAI_IMAGE, {"output_format": "jpeg"}),),
                ),
                RunContext("run-load-failure", time()),
                stream=False,
                emit_delta=None,
            )


@pytest.mark.parametrize("status", ["failed"])
async def test_openai_responses_terminal_partial_or_failed_image_results_are_externalized(
    status: str,
) -> None:
    profile = _openai_feature_profile(image_generation=True)
    item: dict[str, Any] = {
        "id": f"ig-{status}",
        "type": "image_generation_call",
        "status": status,
        "result": _JPEG_BASE64,
    }
    wire = terminal_response(
        [item],
        tools=[{"type": "image_generation", "output_format": "jpeg"}],
    )

    async def handler(raw: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=wire, request=raw)

    store = _MemoryOpenAIResponsesArtifactStore()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAIResponsesModel(
            base_url="https://provider.test/v1",
            api_key="secret",
            model="gpt-test",
            profile=profile,
            artifact_store=store,
            client=client,
        )
        response = await model.invoke(
            ModelRequest(
                messages=(Message.user("draw"),),
                provider_tools=(ProviderToolSpec(_OPENAI_IMAGE, {"output_format": "jpeg"}),),
            ),
            RunContext(f"run-{status}", time()),
            stream=False,
            emit_delta=None,
        )

    call = response.provider_tool_calls()[0]
    assert call.status.value == status
    assert call.output[0].type == "artifact"


async def test_openai_responses_streamed_image_result_is_externalized_after_live_partial_data() -> (
    None
):
    profile = _openai_feature_profile(image_generation=True)
    final_item = {
        "id": "ig-stream",
        "type": "image_generation_call",
        "status": "completed",
        "result": _JPEG_BASE64,
    }
    terminal = terminal_response(
        [final_item],
        tools=[{"type": "image_generation", "output_format": "jpeg"}],
    )
    events: list[dict[str, Any]] = [
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {"id": "resp-1", "object": "response", "status": "in_progress"},
        },
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": {
                "id": "ig-stream",
                "type": "image_generation_call",
                "status": "generating",
            },
        },
        {
            "type": "response.image_generation_call.partial_image",
            "sequence_number": 2,
            "item_id": "ig-stream",
            "output_index": 0,
            "partial_image_index": 0,
            "partial_image_b64": _JPEG_BASE64,
        },
        {
            "type": "response.image_generation_call.completed",
            "sequence_number": 3,
            "item_id": "ig-stream",
            "output_index": 0,
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 4,
            "output_index": 0,
            "item": final_item,
        },
        {
            "type": "response.completed",
            "sequence_number": 5,
            "response": terminal,
        },
    ]
    body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)

    async def handler(raw: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body,
            request=raw,
        )

    deltas: list[ModelDelta] = []

    async def emit_delta(delta: ModelDelta) -> None:
        deltas.append(delta)

    store = _MemoryOpenAIResponsesArtifactStore()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAIResponsesModel(
            base_url="https://provider.test/v1",
            api_key="secret",
            model="gpt-test",
            profile=profile,
            artifact_store=store,
            client=client,
        )
        response = await model.invoke(
            ModelRequest(
                messages=(Message.user("draw"),),
                provider_tools=(ProviderToolSpec(_OPENAI_IMAGE, {"output_format": "jpeg"}),),
            ),
            RunContext("run-stream-image", time()),
            stream=True,
            emit_delta=emit_delta,
        )
        unsafe_model = OpenAIResponsesModel(
            base_url="https://provider.test/v1",
            api_key="secret",
            model="gpt-test",
            profile=profile,
            client=client,
        )
        with pytest.raises(ModelError, match="without an OpenAIResponsesArtifactStore") as caught:
            await unsafe_model.invoke(
                ModelRequest(messages=(Message.user("text only"),)),
                RunContext("run-stream-unrequested", time()),
                stream=True,
                emit_delta=None,
            )

    assert caught.value.info.code == "codec_error"
    call = response.provider_tool_calls()[0]
    assert call.output[0].type == "artifact"
    assert any(
        isinstance(delta, ModelProviderToolCallDelta) and delta.data.get("base64") == _JPEG_BASE64
        for delta in deltas
    )


async def test_openai_responses_stream_rejects_done_sentinel() -> None:
    async def handler(raw: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="data: [DONE]\n\n",
            request=raw,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAIResponsesModel(
            base_url="https://provider.test/v1",
            api_key="secret",
            model="gpt-test",
            client=client,
        )
        with pytest.raises(ModelError, match="typed response event") as caught:
            await model.invoke(
                ModelRequest(messages=(Message.user("question"),)),
                RunContext("run-1", time()),
                stream=True,
                emit_delta=None,
            )

    assert caught.value.info.code == "codec_error"


async def test_openai_responses_failed_response_has_same_nonstream_and_stream_semantics() -> None:
    failed_response: dict[str, Any] = {
        "id": "resp-failed",
        "object": "response",
        "model": "gpt-test",
        "created_at": 1,
        "status": "failed",
        "error": {"code": "server_error", "message": "generation failed"},
        "output": [],
    }

    async def invoke_failed(*, stream: bool) -> ModelErrorInfo:
        async def handler(raw: httpx.Request) -> httpx.Response:
            if not stream:
                return httpx.Response(
                    200,
                    headers={"x-request-id": "request-1"},
                    json=failed_response,
                    request=raw,
                )
            created = {
                "type": "response.created",
                "sequence_number": 0,
                "response": {
                    "id": "resp-failed",
                    "object": "response",
                    "status": "in_progress",
                },
            }
            failed: dict[str, Any] = {
                "type": "response.failed",
                "sequence_number": 1,
                "response": failed_response,
            }
            body = (
                f"event: response.created\ndata: {json.dumps(created)}\n\n"
                f"event: response.failed\ndata: {json.dumps(failed)}\n\n"
            )
            return httpx.Response(
                200,
                headers={
                    "content-type": "text/event-stream",
                    "x-request-id": "request-1",
                },
                content=body,
                request=raw,
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            model = OpenAIResponsesModel(
                base_url="https://provider.test/v1",
                api_key="secret",
                model="gpt-test",
                client=client,
            )
            with pytest.raises(ModelError) as caught:
                await model.invoke(
                    ModelRequest(messages=(Message.user("question"),)),
                    RunContext("run-failed", time()),
                    stream=stream,
                    emit_delta=None,
                )
        return caught.value.info

    nonstream = await invoke_failed(stream=False)
    streamed = await invoke_failed(stream=True)

    assert nonstream == streamed
    assert nonstream.code == "server_error"
    assert nonstream.provider == "openai-responses"
    assert nonstream.status_code is None
    assert nonstream.request_id == "request-1"
    retained = cast(dict[str, Any], nonstream.metadata["responses"])
    assert retained["id"] == "resp-failed"
    assert retained["status"] == "failed"
    assert retained["output"] == []

    async def envelope_handler(raw: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"error": {"code": "plain_error", "message": "plain envelope"}},
            request=raw,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(envelope_handler)) as client:
        model = OpenAIResponsesModel(
            base_url="https://provider.test/v1",
            api_key="secret",
            model="gpt-test",
            client=client,
        )
        with pytest.raises(ModelError) as caught:
            await model.invoke(
                ModelRequest(messages=(Message.user("question"),)),
                RunContext("run-envelope", time()),
                stream=False,
                emit_delta=None,
            )
    assert caught.value.info.code == "plain_error"


def test_openai_responses_cancelled_is_a_terminal_error_but_not_an_sse_event() -> None:
    cancelled: dict[str, object] = {
        "id": "resp-cancelled",
        "object": "response",
        "model": "gpt-test",
        "created_at": 1,
        "status": "cancelled",
        "error": None,
        "output": [],
    }
    codec = OpenAIResponsesCodec(model="gpt-test")
    with pytest.raises(ModelError, match="cancelled") as full:
        codec.decode_response(cancelled)
    retained = cast(dict[str, Any], full.value.info.metadata["responses"])
    assert retained["id"] == "resp-cancelled"
    assert retained["status"] == "cancelled"
    assert retained["output"] == []

    decoder = OpenAIResponsesStreamDecoder(codec, codec.profile)
    _stream_event(
        decoder,
        "response.created",
        0,
        response={"id": "resp-cancelled", "object": "response", "status": "in_progress"},
    )
    with pytest.raises(OpenAIResponsesError, match="unsupported Responses stream event type"):
        _stream_event(decoder, "response.cancelled", 1, response=cancelled)
