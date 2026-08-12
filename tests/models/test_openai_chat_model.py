from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, cast

import httpx
import pytest

from jharness.kernel import (
    ContentPart,
    Message,
    Model,
    ModelContentDelta,
    ModelDelta,
    ModelError,
    ModelOptions,
    ModelRequest,
    ResponseFormat,
    RunContext,
    StructuredToolCall,
    StructuredToolSpec,
    ToolChoice,
)
from jharness.models.deepseek import deepseek_chat_profile
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


def test_openai_chat_codec_thaws_nested_profile_json_at_wire_boundary() -> None:
    deepseek_profile = deepseek_chat_profile(thinking=True)
    codec = OpenAIChatCodec(
        model="deepseek-v4-pro",
        profile=deepseek_profile,
    )
    model_request = ModelRequest(messages=(Message.user("hello"),))

    payload = codec.encode_request(model_request)
    assert json.loads(json.dumps(payload))["thinking"] == {"type": "enabled"}

    thinking = cast(dict[str, Any], payload["thinking"])
    thinking["type"] = "disabled"
    assert codec.encode_request(model_request)["thinking"] == {"type": "enabled"}
    assert deepseek_profile.extra_request_body["thinking"] == {"type": "enabled"}
    with pytest.raises(TypeError, match="extra_request_body is immutable"):
        cast(dict[str, Any], deepseek_profile.extra_request_body["thinking"])["type"] = "disabled"


def test_openai_chat_stream_decoder_builds_complete_response() -> None:
    decoder = OpenAIChatStreamDecoder(profile())
    first = decoder.apply_chunk(
        {
            "id": "resp-1",
            "model": "gpt-test",
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
            ]
        }
    )
    decoder.apply_chunk(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"x"}'}}]},
                    "finish_reason": "tool_calls",
                }
            ]
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


def test_openai_chat_codec_rejects_invalid_choice_shape() -> None:
    codec = OpenAIChatCodec(model="gpt-test")
    with pytest.raises(OpenAIChatError, match="exactly one choice"):
        codec.decode_response({"choices": []})


def test_openai_chat_reasoning_content_round_trips_through_history() -> None:
    profile = OpenAIChatProfile(reasoning_content_mode="round_trip")
    codec = OpenAIChatCodec(model="gpt-test", profile=profile)
    response = codec.decode_response(
        {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "reasoning_content": "think",
                        "content": "answer",
                    },
                }
            ]
        }
    )

    assert [(part.type, part.text) for part in response.visible_parts()] == [
        ("reasoning", "think"),
        ("text", "answer"),
    ]
    encoded = codec.encode_request(
        ModelRequest(
            messages=(
                Message.user("question"),
                Message.assistant(
                    (
                        ContentPart(type="reasoning", text="one"),
                        ContentPart.text_part("answer"),
                        ContentPart(type="reasoning", text="two"),
                    )
                ),
            )
        )
    )
    assert encoded["messages"][1] == {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "onetwo",
    }


def test_openai_chat_reasoning_content_modes_enforce_tool_round_trip() -> None:
    call = StructuredToolCall("call-1", "search", {})
    required = OpenAIChatCodec(
        model="gpt-test",
        profile=OpenAIChatProfile(reasoning_content_mode="required_with_tools"),
    )
    with pytest.raises(OpenAIChatError, match="requires non-empty reasoning"):
        required.decode_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "search",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                    }
                ]
            }
        )
    with pytest.raises(OpenAIChatError, match="requires non-empty reasoning"):
        required.encode_request(
            ModelRequest(
                messages=(
                    Message.user("question"),
                    Message.assistant((call,)),
                )
            )
        )

    payload = required.encode_request(
        ModelRequest(
            messages=(
                Message.user("question"),
                Message.assistant(
                    (ContentPart(type="reasoning", text="why"), call),
                ),
            )
        )
    )
    assert payload["messages"][1]["reasoning_content"] == "why"
    with pytest.raises(OpenAIChatError, match="reasoning content"):
        OpenAIChatCodec(model="gpt-test").encode_request(
            ModelRequest(
                messages=(
                    Message.user("question"),
                    Message.assistant((ContentPart(type="reasoning", text="why"),)),
                )
            )
        )


def test_deepseek_chat_thinking_tool_replay_omits_tool_choice_and_keeps_content_non_null() -> None:
    codec = OpenAIChatCodec(
        model="deepseek-v4-pro",
        profile=deepseek_chat_profile(thinking=True),
    )
    call = StructuredToolCall("call-1", "search", {})
    payload = codec.encode_request(
        ModelRequest(
            messages=(
                Message.user("question"),
                Message.assistant(
                    (ContentPart(type="reasoning", text="why"), call),
                ),
            ),
            runtime_tools=(StructuredToolSpec("search", "search", {"type": "object"}),),
        )
    )

    assert "tool_choice" not in payload
    assert payload["messages"][1]["content"] == ""
    assert payload["messages"][1]["reasoning_content"] == "why"


def test_openai_chat_reasoning_content_and_seed_validate_wire_values() -> None:
    round_trip = OpenAIChatCodec(
        model="gpt-test",
        profile=OpenAIChatProfile(reasoning_content_mode="round_trip"),
    )
    with pytest.raises(OpenAIChatError, match="reasoning_content must be"):
        round_trip.decode_response(
            {
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "reasoning_content": 1,
                            "content": "answer",
                        },
                    }
                ]
            }
        )

    seeded_request = ModelRequest(
        messages=(Message.user("question"),),
        options=ModelOptions(seed=7),
    )
    assert round_trip.encode_request(seeded_request)["seed"] == 7
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


def test_openai_chat_usage_supports_deepseek_cache_fields_with_nested_precedence() -> None:
    codec = OpenAIChatCodec(model="gpt-test")

    def decode_cache(usage: dict[str, object]) -> int | None:
        response = codec.decode_response(
            {
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

    assert decode_cache({"prompt_cache_hit_tokens": 11}) == 11
    assert (
        decode_cache(
            {
                "prompt_cache_hit_tokens": 11,
                "prompt_tokens_details": {"cached_tokens": 7},
            }
        )
        == 7
    )


async def test_openai_chat_client_decodes_sse_stream() -> None:
    body = (
        'data: {"id":"resp-1","model":"gpt-test","choices":['
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
