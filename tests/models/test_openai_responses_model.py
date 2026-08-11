from __future__ import annotations

import base64
import json
from dataclasses import replace
from time import time
from typing import Any, cast

import httpx
import pytest

from jharness.kernel import (
    ArtifactRef,
    ContentPart,
    Message,
    ModelCapabilities,
    ModelDelta,
    ModelError,
    ModelErrorInfo,
    ModelProviderToolCallDelta,
    ModelReasoningDelta,
    ModelRequest,
    ModelUsageDelta,
    ProviderToolCall,
    ProviderToolId,
    ProviderToolSpec,
    ProviderToolStatus,
    RunContext,
    ToolCall,
    ToolChoice,
    ToolSpec,
)
from jharness.models.deepseek import deepseek_openai_responses_profile
from jharness.models.openai import (
    OpenAIResponsesCodec,
    OpenAIResponsesError,
    OpenAIResponsesModel,
    OpenAIResponsesProfile,
)
from jharness.models.openai.responses_api.stream import OpenAIResponsesStreamDecoder

_DEEPSEEK_WEB = ProviderToolId("deepseek.responses", "web_search")
_OPENAI_IMAGE = ProviderToolId("openai.responses", "image_generation")
_JPEG_BASE64 = base64.b64encode(b"\xff\xd8\xffjpeg-payload").decode("ascii")


def _terminal_response(
    output: list[dict[str, Any]],
    *,
    model: str = "gpt-test",
    status: str = "completed",
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": "resp-1",
        "object": "response",
        "created_at": 1,
        "completed_at": 2,
        "status": status,
        "error": None,
        "incomplete_details": ({"reason": "max_output_tokens"} if status == "incomplete" else None),
        "model": model,
        "output": output,
        "previous_response_id": None,
        "store": False,
        "tools": [] if tools is None else tools,
        "usage": {
            "input_tokens": 3,
            "output_tokens": 2,
            "total_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 1},
        },
    }


def _stream_event(
    decoder: OpenAIResponsesStreamDecoder,
    event_type: str,
    sequence_number: int,
    **fields: Any,
) -> tuple[bool, list[ModelDelta]]:
    event = {"type": event_type, "sequence_number": sequence_number, **fields}
    return decoder.apply_event(event_type, event)


def test_deepseek_profile_and_request_encode_native_responses() -> None:
    profile = deepseek_openai_responses_profile(effort="none")
    codec = OpenAIResponsesCodec(model="deepseek-v4-flash", profile=profile)
    request = ModelRequest(
        messages=(Message.system("policy"), Message.user("question")),
        runtime_tools=(ToolSpec("lookup", "lookup", {"type": "object"}),),
        provider_tools=(ProviderToolSpec(_DEEPSEEK_WEB),),
        tool_choice=ToolChoice(type="provider", provider_tool=_DEEPSEEK_WEB),
    )

    payload = codec.encode_request(request)

    assert profile.capabilities.input_modalities == frozenset({"text"})
    assert profile.capabilities.output_modalities == frozenset({"text"})
    assert profile.capabilities.provider_tools == frozenset({_DEEPSEEK_WEB})
    assert payload["model"] == "deepseek-v4-flash"
    assert payload["reasoning"] == {"effort": "none"}
    assert payload["tool_choice"] == {"type": "web_search"}
    assert payload["tools"] == [
        {
            "type": "function",
            "name": "lookup",
            "description": "lookup",
            "parameters": {"type": "object"},
        },
        {"type": "web_search"},
    ]
    assert [item["role"] for item in cast(list[dict[str, Any]], payload["input"])] == [
        "system",
        "user",
    ]
    assert "store" not in payload
    assert "previous_response_id" not in payload
    assert "parallel_tool_calls" not in payload

    thinking_profile = deepseek_openai_responses_profile()
    thinking_codec = OpenAIResponsesCodec(
        model="deepseek-v4-flash",
        profile=thinking_profile,
    )
    with pytest.raises(OpenAIResponsesError, match="does not support tool_choice='required'"):
        thinking_codec.encode_request(
            ModelRequest(
                messages=(Message.user("question"),),
                provider_tools=(ProviderToolSpec(_DEEPSEEK_WEB),),
                tool_choice=ToolChoice(type="required"),
            )
        )
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
    with pytest.raises(ValueError, match="provider tool type"):
        OpenAIResponsesProfile(
            capabilities=ModelCapabilities(
                provider_tools=frozenset({ProviderToolId("test", "computer")}),
                tool_choice_types=frozenset({"auto", "none", "required", "runtime", "provider"}),
            ),
            provider_tool_configuration_fields={"computer": frozenset()},
        )


async def test_deepseek_nonstream_client_preserves_interleaved_output_order() -> None:
    captured: dict[str, object] = {}
    wire_response = _terminal_response(
        [
            {
                "id": "ws-1",
                "type": "web_search_call",
                "status": "completed",
                "action": {"type": "search", "query": "first"},
            },
            {
                "id": "msg-1",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "one", "annotations": []}],
            },
            {
                "id": "ws-2",
                "type": "web_search_call",
                "status": "failed",
                "action": {"type": "search", "query": "second"},
            },
            {
                "id": "msg-2",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "two", "annotations": []}],
            },
        ],
        model="deepseek-v4-flash",
    )

    async def handler(raw: httpx.Request) -> httpx.Response:
        captured["url"] = str(raw.url)
        captured["body"] = json.loads(raw.content)
        return httpx.Response(200, json=wire_response, request=raw)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAIResponsesModel(
            base_url="https://api.deepseek.test/v1",
            api_key="secret",
            model="deepseek-v4-flash",
            profile=deepseek_openai_responses_profile(effort="none"),
            client=client,
        )
        response = await model.invoke(
            ModelRequest(
                messages=(Message.user("question"),),
                provider_tools=(ProviderToolSpec(_DEEPSEEK_WEB),),
            ),
            RunContext("run-1", time()),
            stream=False,
            emit_delta=None,
        )

    assert captured["url"] == "https://api.deepseek.test/v1/responses"
    assert "store" not in cast(dict[str, object], captured["body"])
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


def test_terminal_function_call_rejects_incomplete_execution() -> None:
    codec = OpenAIResponsesCodec(model="gpt-test")

    for status in ("in_progress", "incomplete"):
        with pytest.raises(OpenAIResponsesError, match="must be completed"):
            codec.decode_response(
                _terminal_response(
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

    response = codec.decode_response(
        _terminal_response(
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
    assert response.runtime_tool_calls() == (ToolCall("call-1", "lookup", {}),)

    with pytest.raises(OpenAIResponsesError, match="incomplete Responses"):
        codec.decode_response(
            _terminal_response(
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


def test_deepseek_reasoning_sse_tracks_open_part_and_uses_terminal_response() -> None:
    profile = deepseek_openai_responses_profile()
    codec = OpenAIResponsesCodec(model="deepseek-v4-flash", profile=profile)
    decoder = OpenAIResponsesStreamDecoder(codec, profile)
    reasoning_item = {
        "id": "rs-1",
        "type": "reasoning",
        "status": "completed",
        "content": [{"type": "reasoning_text", "text": "分析"}],
        "summary": [],
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
        response=_terminal_response([reasoning_item], model="deepseek-v4-flash"),
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


def test_web_search_stream_can_finish_failed_after_completed_progress() -> None:
    profile = deepseek_openai_responses_profile(effort="none")
    decoder = OpenAIResponsesStreamDecoder(
        OpenAIResponsesCodec(model="deepseek-v4-flash", profile=profile),
        profile,
    )
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
        },
    )
    _, completed = _stream_event(
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
    _, failed = _stream_event(
        decoder,
        "response.output_item.done",
        4,
        output_index=0,
        item={
            "id": "ws-1",
            "type": "web_search_call",
            "status": "failed",
            "error": {"code": "search_failed", "message": "failed"},
        },
    )

    deltas = [added[0], completed[0], failed[0]]
    assert all(isinstance(delta, ModelProviderToolCallDelta) for delta in deltas)
    assert [cast(ModelProviderToolCallDelta, delta).status for delta in deltas] == [
        ProviderToolStatus.IN_PROGRESS,
        ProviderToolStatus.COMPLETED,
        ProviderToolStatus.FAILED,
    ]


def test_output_text_annotation_event_validates_the_open_message_part() -> None:
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
        item={
            "id": "msg-1",
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        },
    )
    _stream_event(
        decoder,
        "response.content_part.added",
        2,
        item_id="msg-1",
        output_index=0,
        content_index=0,
        part={"type": "output_text", "text": ""},
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
    with pytest.raises(OpenAIResponsesError, match="annotation"):
        _stream_event(
            decoder,
            "response.output_text.annotation.added",
            8,
            item_id="msg-1",
            output_index=0,
            content_index=0,
            annotation_index=1,
            annotation=None,
        )


def test_provider_only_terminal_response_is_valid() -> None:
    profile = deepseek_openai_responses_profile()
    response = OpenAIResponsesCodec(
        model="deepseek-v4-flash",
        profile=profile,
    ).decode_response(
        _terminal_response(
            [
                {
                    "id": "ws-1",
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {"type": "search", "query": "only"},
                }
            ],
            model="deepseek-v4-flash",
        )
    )

    assert len(response.output) == 1
    assert isinstance(response.output[0], ProviderToolCall)
    assert response.visible_parts() == ()
    assert response.finish_reason == "stop"

    with pytest.raises(OpenAIResponsesError, match="in-progress provider tools"):
        OpenAIResponsesCodec(
            model="deepseek-v4-flash",
            profile=profile,
        ).decode_response(
            _terminal_response(
                [
                    {
                        "id": "ws-2",
                        "type": "web_search_call",
                        "status": "searching",
                        "action": {"type": "search", "query": "pending"},
                    }
                ],
                model="deepseek-v4-flash",
            )
        )


def test_vision_inputs_encode_url_base64_and_artifact_with_media_validation() -> None:
    codec = OpenAIResponsesCodec(model="gpt-test")
    request = ModelRequest(
        messages=(
            Message(
                "user",
                (
                    ContentPart(type="image", uri="https://images.test/cat.png"),
                    ContentPart(type="image", data={"base64": _JPEG_BASE64}),
                    ContentPart.artifact_part(ArtifactRef("file-image", media_type="image/png")),
                ),
            ),
        )
    )

    content = cast(list[dict[str, Any]], codec.encode_request(request)["input"])[0]["content"]
    assert content == [
        {"type": "input_image", "image_url": "https://images.test/cat.png"},
        {
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{_JPEG_BASE64}",
        },
        {"type": "input_image", "file_id": "file-image"},
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


def test_image_generation_decodes_media_type_and_replays_base64_history() -> None:
    codec = OpenAIResponsesCodec(model="gpt-test")
    wire = _terminal_response(
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

    mismatched = _terminal_response(
        cast(list[dict[str, Any]], wire["output"]),
        tools=[{"type": "image_generation", "output_format": "png"}],
    )
    with pytest.raises(OpenAIResponsesError, match="does not match"):
        codec.decode_response(mismatched)


async def test_responses_stream_rejects_done_sentinel() -> None:
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


async def test_failed_responses_have_the_same_nonstream_and_stream_semantics() -> None:
    failed_response: dict[str, Any] = {
        "id": "resp-failed",
        "object": "response",
        "model": "gpt-test",
        "status": "failed",
        "error": {"code": "generation_failed", "message": "generation failed"},
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
    assert nonstream.code == "generation_failed"
    assert nonstream.status_code is None
    assert nonstream.request_id == "request-1"
    assert nonstream.metadata == {"response_id": "resp-failed", "status": "failed"}

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
