from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, cast

import httpx
import pytest

from jharness.kernel import (
    ArtifactRef,
    ContentPart,
    FreeformToolCall,
    FreeformToolSpec,
    Message,
    Model,
    ModelContentDelta,
    ModelDelta,
    ModelError,
    ModelOptions,
    ModelRequest,
    ResponseFormat,
    RunContext,
    RuntimeToolKind,
    StructuredToolCall,
    StructuredToolSpec,
    ToolChoice,
)
from jharness.models.openai import (
    OpenAIChatCodec,
    OpenAIChatError,
    OpenAIChatModel,
    OpenAIChatProfile,
)
from jharness.models.openai.chat.stream import OpenAIChatStreamDecoder


def request() -> ModelRequest:
    return ModelRequest(
        messages=(Message.system("policy"), Message.user("hello")),
        runtime_tools=(StructuredToolSpec("search", "search", {"type": "object"}),),
        options=ModelOptions(temperature=0.2, max_output_tokens=50),
        tool_choice=ToolChoice(
            type="runtime",
            name="search",
            allow_parallel_runtime_tool_calls=False,
        ),
        response_format=ResponseFormat("json_schema", {"type": "object"}, True),
    )


def profile() -> OpenAIChatProfile:
    default = OpenAIChatProfile()
    return OpenAIChatProfile(capabilities=replace(default.capabilities, structured_output=True))


def http_model(client: httpx.AsyncClient) -> OpenAIChatModel:
    return OpenAIChatModel(
        base_url="https://provider.test/v1",
        api_key="secret",
        model="gpt-test",
        client=client,
    )


def test_openai_chat_codec_encodes_direct_tool_identity_and_decodes_response() -> None:
    codec = OpenAIChatCodec(model="gpt-test", profile=profile())
    payload = codec.encode_request(request())
    tool = cast(dict[str, Any], cast(list[object], payload["tools"])[0])
    function = cast(dict[str, Any], tool["function"])

    assert payload["model"] == "gpt-test"
    assert function["name"] == "search"
    assert "mode" not in function
    assert payload["parallel_tool_calls"] is False
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "schema": {"type": "object"},
            "strict": True,
        },
    }

    response = codec.decode_response(
        {
            "id": "resp-1",
            "model": "gpt-test",
            "object": "chat.completion",
            "created": 1,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "checking",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"q":"x"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }
    )
    assert response.visible_parts()[0].text == "checking"
    assert response.runtime_tool_calls() == (StructuredToolCall("call-1", "search", {"q": "x"}),)
    assert response.usage is not None and response.usage.total_tokens == 5
    assert response.metadata["provider"] == "openai-chat"


def test_openai_chat_stream_decoder_builds_complete_response() -> None:
    decoder = OpenAIChatStreamDecoder(profile())
    first = decoder.apply_chunk(
        {
            "id": "resp-1",
            "model": "gpt-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "hel"},
                    "finish_reason": None,
                }
            ],
        }
    )
    second = decoder.apply_chunk(
        {
            "id": "resp-1",
            "model": "gpt-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}],
        }
    )
    completed = decoder.completed_response()

    assert len(first) == 1 and len(second) == 1
    assert completed.visible_parts()[0].text == "hello"
    assert completed.finish_reason == "stop"
    assert completed.response_id == "resp-1"
    assert completed.metadata["provider"] == "openai-chat"


def test_openai_chat_stream_decoder_accumulates_tool_call_arguments() -> None:
    decoder = OpenAIChatStreamDecoder(profile())
    decoder.apply_chunk(
        {
            "id": "resp-1",
            "model": "gpt-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "search", "arguments": '{"q":'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
    )
    decoder.apply_chunk(
        {
            "id": "resp-1",
            "model": "gpt-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"x"}'}}]},
                    "finish_reason": "tool_calls",
                }
            ],
        }
    )

    assert decoder.completed_response().runtime_tool_calls() == (
        StructuredToolCall("call-1", "search", {"q": "x"}),
    )


async def test_openai_chat_client_uses_http_transport_and_maps_http_errors() -> None:
    captured: dict[str, object] = {}

    async def success_handler(raw: httpx.Request) -> httpx.Response:
        captured["authorization"] = raw.headers["authorization"]
        captured["body"] = json.loads(raw.content)
        return httpx.Response(
            200,
            json={
                "id": "resp-1",
                "model": "gpt-test",
                "object": "chat.completion",
                "created": 1,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "done"},
                    }
                ],
            },
            request=raw,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(success_handler)) as client:
        model = http_model(client)
        result = await model.invoke(
            ModelRequest(messages=(Message.user("hello"),)),
            RunContext("run-1", 1.0),
            stream=False,
            emit_delta=None,
        )
    assert result.visible_parts()[0].text == "done"
    assert captured["authorization"] == "Bearer secret"
    assert isinstance(model, Model)
    assert not hasattr(model, "complete")
    assert not hasattr(model, "stream")

    async def error_handler(raw: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"x-request-id": "req-1"},
            json={"error": {"message": "rate limited", "code": "rate_limit"}},
            request=raw,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(error_handler)) as client:
        model = http_model(client)
        with pytest.raises(ModelError) as caught:
            await model.invoke(
                ModelRequest(messages=(Message.user("hello"),)),
                RunContext("run-1", 1.0),
                stream=False,
                emit_delta=None,
            )
    assert caught.value.info.code == "rate_limit"
    assert caught.value.info.provider == "openai-chat"
    assert caught.value.info.retryable is True
    assert caught.value.info.request_id == "req-1"


async def test_openai_chat_client_sends_nested_file_reference() -> None:
    captured: dict[str, object] = {}

    async def handler(raw: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(raw.content)
        return httpx.Response(
            200,
            json={
                "id": "resp-1",
                "model": "vision-test",
                "object": "chat.completion",
                "created": 1,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "done"},
                    }
                ],
            },
            request=raw,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAIChatModel(
            base_url="https://provider.test/v1",
            api_key="secret",
            model="vision-test",
            profile=OpenAIChatProfile(
                capabilities=replace(
                    OpenAIChatProfile().capabilities,
                    input_modalities=frozenset({"text", "image", "file"}),
                )
            ),
            client=client,
        )
        await model.invoke(
            ModelRequest(
                messages=(
                    Message(
                        "user",
                        (
                            ContentPart.artifact_part(
                                ArtifactRef("image-file", media_type="IMAGE/PNG")
                            ),
                        ),
                    ),
                )
            ),
            RunContext("run-1", 1.0),
            stream=False,
            emit_delta=None,
        )

    body = cast(dict[str, Any], captured["body"])
    assert body["messages"] == [
        {
            "role": "user",
            "content": [{"type": "file", "file": {"file_id": "image-file"}}],
        }
    ]


def test_openai_chat_codec_rejects_invalid_choice_shape() -> None:
    codec = OpenAIChatCodec(model="gpt-test")
    with pytest.raises(OpenAIChatError, match="exactly one choice"):
        codec.decode_response(
            {
                "id": "resp",
                "model": "model",
                "object": "chat.completion",
                "created": 1,
                "choices": [],
            }
        )


def test_openai_chat_rejects_boolean_json_schema() -> None:
    codec = OpenAIChatCodec(model="gpt-test", profile=profile())
    with pytest.raises(OpenAIChatError, match="response schema must be an object"):
        codec.encode_request(
            ModelRequest(
                messages=(Message.user("hello"),),
                response_format=ResponseFormat("json_schema", True, True),
            )
        )


@pytest.mark.parametrize(
    ("options", "pattern"),
    (
        (ModelOptions(temperature=-0.1), "temperature"),
        (ModelOptions(temperature=2.1), "temperature"),
        (ModelOptions(top_p=-0.1), "top_p"),
        (ModelOptions(top_p=1.1), "top_p"),
        (ModelOptions(stop=("a", "b", "c", "d", "e")), "at most 4"),
    ),
)
def test_openai_chat_rejects_out_of_range_request_options(
    options: ModelOptions,
    pattern: str,
) -> None:
    with pytest.raises(OpenAIChatError, match=pattern):
        OpenAIChatCodec(model="gpt-test").encode_request(
            ModelRequest(messages=(Message.user("hello"),), options=options)
        )


def test_openai_chat_rejects_nonstandard_reasoning_content() -> None:
    codec = OpenAIChatCodec(model="gpt-test")
    with pytest.raises(OpenAIChatError, match="do not support reasoning_content"):
        codec.decode_response(
            {
                "id": "resp",
                "model": "model",
                "object": "chat.completion",
                "created": 1,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "reasoning_content": "x", "content": "a"},
                    }
                ],
            }
        )
    with pytest.raises(OpenAIChatError, match="does not support assistant reasoning"):
        codec.encode_request(
            ModelRequest(messages=(Message.assistant((ContentPart(type="reasoning", text="x"),)),))
        )


def test_openai_chat_encodes_max_completion_tokens_and_standard_file_data() -> None:
    codec = OpenAIChatCodec(
        model="gpt-test",
        profile=OpenAIChatProfile(
            capabilities=replace(
                OpenAIChatProfile().capabilities,
                input_modalities=frozenset({"text", "image", "file"}),
            )
        ),
    )
    payload = codec.encode_request(
        ModelRequest(
            messages=(
                Message(
                    "user",
                    (
                        ContentPart(
                            type="file", uri="data:text/plain;base64,aGVsbG8=", name="a.txt"
                        ),
                    ),
                ),
            ),
            options=ModelOptions(max_output_tokens=7),
        )
    )
    assert payload["max_completion_tokens"] == 7
    assert "max_tokens" not in payload
    assert payload["messages"][0]["content"] == [
        {"type": "file", "file": {"file_data": "aGVsbG8=", "filename": "a.txt"}}
    ]
    with pytest.raises(OpenAIChatError, match="base64 data, not a URL"):
        codec.encode_request(
            ModelRequest(messages=(Message("user", (ContentPart(type="file", uri="https://x"),)),))
        )


def test_openai_chat_encodes_audio_and_custom_tools_with_history() -> None:
    codec = OpenAIChatCodec(
        model="gpt-test",
        profile=OpenAIChatProfile(
            capabilities=replace(
                OpenAIChatProfile().capabilities,
                runtime_tool_kinds=frozenset(
                    {RuntimeToolKind.STRUCTURED, RuntimeToolKind.FREEFORM}
                ),
                input_modalities=frozenset({"text", "image", "audio"}),
            )
        ),
    )
    custom = FreeformToolSpec("patch", "apply a patch")
    call = FreeformToolCall("call-1", "patch", "*** Begin Patch")
    payload = codec.encode_request(
        ModelRequest(
            messages=(
                Message(
                    "user",
                    (ContentPart(type="audio", uri="data:audio/wav;base64,aGk="),),
                ),
                Message.assistant((call,)),
            ),
            runtime_tools=(custom,),
            tool_choice=ToolChoice(type="runtime", name="patch"),
        )
    )
    assert payload["messages"][0]["content"] == [
        {"type": "input_audio", "input_audio": {"data": "aGk=", "format": "wav"}}
    ]
    assert payload["tools"] == [
        {"type": "custom", "custom": {"name": "patch", "description": "apply a patch"}}
    ]
    assert payload["tool_choice"] == {"type": "custom", "custom": {"name": "patch"}}
    assert payload["messages"][1] == {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "custom",
                "custom": {"name": "patch", "input": "*** Begin Patch"},
            }
        ],
    }

    decoded = codec.decode_response(
        {
            "id": "resp",
            "model": "model",
            "object": "chat.completion",
            "created": 1,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-2",
                                "type": "custom",
                                "custom": {"name": "patch", "input": "*** End Patch"},
                            }
                        ],
                    },
                }
            ],
        }
    )
    assert decoded.runtime_tool_calls() == (FreeformToolCall("call-2", "patch", "*** End Patch"),)


def test_openai_chat_stream_rejects_custom_tool_calls_and_decoder_custom_deltas() -> None:
    profile = OpenAIChatProfile(
        capabilities=replace(
            OpenAIChatProfile().capabilities,
            runtime_tool_kinds=frozenset({RuntimeToolKind.STRUCTURED, RuntimeToolKind.FREEFORM}),
        )
    )
    codec = OpenAIChatCodec(model="gpt-test", profile=profile)
    custom = FreeformToolSpec("patch", "apply a patch")
    with pytest.raises(OpenAIChatError, match="streaming does not support custom"):
        codec.encode_request(
            ModelRequest(
                messages=(Message.user("hello"),),
                runtime_tools=(custom,),
                tool_choice=ToolChoice(type="runtime", name="patch"),
            ),
            stream=True,
        )
    payload = codec.encode_request(
        ModelRequest(
            messages=(Message.user("hello"),),
            runtime_tools=(custom,),
            tool_choice=ToolChoice(type="none"),
        ),
        stream=True,
    )
    assert payload["stream"] is True
    with pytest.raises(
        OpenAIChatError, match="unsupported chat completion stream tool call type: custom"
    ):
        OpenAIChatStreamDecoder(profile).apply_chunk(
            {
                "id": "resp",
                "model": "model",
                "object": "chat.completion.chunk",
                "created": 1,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "type": "custom"}]},
                        "finish_reason": None,
                    }
                ],
            }
        )


def test_openai_chat_seed_validate_wire_values() -> None:
    codec = OpenAIChatCodec(model="gpt-test")
    seeded_request = ModelRequest(
        messages=(Message.user("question"),),
        options=ModelOptions(seed=7),
    )
    assert codec.encode_request(seeded_request)["seed"] == 7
    no_seed = OpenAIChatCodec(
        model="gpt-test",
        profile=OpenAIChatProfile(
            capabilities=replace(
                OpenAIChatProfile().capabilities,
                seed=False,
            )
        ),
    )
    with pytest.raises(OpenAIChatError, match="does not support seed"):
        no_seed.encode_request(seeded_request)


def test_openai_chat_usage_reads_standard_cached_tokens() -> None:
    codec = OpenAIChatCodec(model="gpt-test")

    def decode_cache(usage: dict[str, object]) -> int | None:
        response = codec.decode_response(
            {
                "id": "resp",
                "model": "model",
                "object": "chat.completion",
                "created": 1,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "answer"},
                    }
                ],
                "usage": usage,
            }
        )
        assert response.usage is not None
        return response.usage.cache_read_tokens

    assert (
        decode_cache(
            {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
                "prompt_tokens_details": {
                    "audio_tokens": 0,
                    "cache_write_tokens": 1,
                    "cached_tokens": 7,
                    "image_tokens": 0,
                    "text_tokens": 1,
                },
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 0,
                    "audio_tokens": 0,
                    "reasoning_tokens": 0,
                    "rejected_prediction_tokens": 0,
                    "text_tokens": 2,
                },
            }
        )
        == 7
    )


def test_openai_chat_content_filter_preserves_null_history_content() -> None:
    codec = OpenAIChatCodec(model="gpt-test")
    response = codec.decode_response(
        {
            "id": "resp",
            "model": "model",
            "object": "chat.completion",
            "created": 1,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "content_filter",
                    "message": {"role": "assistant", "content": None},
                }
            ],
            "service_tier": "fast",
            "system_fingerprint": "fp_123",
        }
    )

    assert response.output == ()
    assert response.metadata["service_tier"] == "fast"
    assert response.metadata["openai_chat"] == {"content_null": True}
    payload = codec.encode_request(ModelRequest(messages=(response.to_assistant_message(),)))
    assert payload["messages"] == [{"role": "assistant", "content": None}]

    stopped = codec.decode_response(
        {
            "id": "resp-empty",
            "model": "model",
            "object": "chat.completion",
            "created": 1,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": None},
                }
            ],
        }
    )
    assert stopped.output == ()


def test_openai_chat_rejects_unsupported_response_observability_and_invalid_usage() -> None:
    codec = OpenAIChatCodec(model="gpt-test")
    response: dict[str, Any] = {
        "id": "resp",
        "model": "model",
        "object": "chat.completion",
        "created": 1,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
                "logprobs": {"content": []},
            }
        ],
    }
    with pytest.raises(OpenAIChatError, match="logprobs is not supported"):
        codec.decode_response(response)

    response["choices"] = [
        {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}
    ]
    response["usage"] = {"prompt_tokens": 1, "completion_tokens": 1}
    with pytest.raises(OpenAIChatError, match="total_tokens"):
        codec.decode_response(response)


def test_openai_chat_stream_rejects_null_choices_even_with_usage() -> None:
    decoder = OpenAIChatStreamDecoder(profile())
    with pytest.raises(OpenAIChatError, match="choices must be an array"):
        decoder.apply_chunk(
            {
                "id": "resp",
                "model": "model",
                "object": "chat.completion.chunk",
                "created": 1,
                "choices": None,
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )


async def test_openai_chat_client_decodes_sse_stream() -> None:
    body = (
        'data: {"id":"resp-1","model":"gpt-test",'
        '"object":"chat.completion.chunk","created":1,"choices":['
        '{"index":0,"delta":{"role":"assistant","content":"hello"},'
        '"finish_reason":"stop"}]}\n\n'
        "data: [DONE]\n\n"
    )

    async def handler(raw: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body,
            request=raw,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = http_model(client)
        deltas: list[ModelDelta] = []

        async def emit_delta(delta: ModelDelta, /) -> None:
            deltas.append(delta)

        result = await model.invoke(
            ModelRequest(messages=(Message.user("hello"),)),
            RunContext("run-1", 1.0),
            stream=True,
            emit_delta=emit_delta,
        )
        unobserved = await model.invoke(
            ModelRequest(messages=(Message.user("hello"),)),
            RunContext("run-2", 1.0),
            stream=True,
            emit_delta=None,
        )

    assert len(deltas) == 1
    assert isinstance(deltas[0], ModelContentDelta)
    assert result.visible_parts()[0].text == "hello"
    assert unobserved.visible_parts()[0].text == "hello"
