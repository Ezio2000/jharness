from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, cast

import httpx
import pytest

from jharness.kernel import (
    ArtifactRef,
    ContentPart,
    Message,
    Model,
    ModelContentDelta,
    ModelDelta,
    ModelError,
    ModelOptions,
    ModelReasoningDelta,
    ModelRequest,
    ModelUsageDelta,
    ResponseFormat,
    RunContext,
    StructuredToolCall,
    StructuredToolSpec,
    ToolChoice,
)
from jharness.models.anthropic import (
    AnthropicMessagesCodec,
    AnthropicMessagesError,
    AnthropicMessagesModel,
    AnthropicMessagesProfile,
)
from jharness.models.anthropic.messages.stream import AnthropicMessagesStreamDecoder


def request() -> ModelRequest:
    return ModelRequest(
        messages=(Message.system("policy"), Message.user("hello")),
        runtime_tools=(StructuredToolSpec("search", "search", {"type": "object"}),),
        options=ModelOptions(max_output_tokens=50),
        tool_choice=ToolChoice(
            type="runtime",
            name="search",
            allow_parallel_runtime_tool_calls=False,
        ),
        response_format=ResponseFormat(
            "json_schema",
            {
                "type": "object",
                "properties": {"x": {"type": "object"}},
                "dependencies": {"x": {"type": "object"}, "property": ["x"]},
                "allOf": [{"type": "object"}],
                "items": {"type": "object"},
                "examples": [{"type": "object"}],
            },
            True,
        ),
    )


def http_model(client: httpx.AsyncClient) -> AnthropicMessagesModel:
    return AnthropicMessagesModel(
        base_url="https://provider.test",
        api_key="secret",
        model="claude-test",
        client=client,
    )


def test_anthropic_messages_codec_encodes_tools_and_decodes_blocks() -> None:
    codec = AnthropicMessagesCodec(model="claude-test")
    payload = codec.encode_request(request())
    tool = cast(dict[str, Any], cast(list[object], payload["tools"])[0])

    assert payload["system"] == [{"type": "text", "text": "policy"}]
    assert tool["name"] == "search"
    assert "mode" not in tool
    assert payload["tool_choice"] == {
        "type": "tool",
        "name": "search",
        "disable_parallel_tool_use": True,
    }
    output = cast(dict[str, Any], payload["output_config"])
    schema = cast(dict[str, Any], cast(dict[str, Any], output["format"])["schema"])
    assert schema["additionalProperties"] is False
    assert schema["properties"]["x"]["additionalProperties"] is False
    assert schema["dependencies"]["x"]["additionalProperties"] is False
    assert schema["allOf"][0]["additionalProperties"] is False
    assert schema["items"]["additionalProperties"] is False
    assert "additionalProperties" not in schema["examples"][0]

    response = codec.decode_response(
        {
            "type": "message",
            "role": "assistant",
            "id": "msg-1",
            "model": "claude-test",
            "stop_reason": "tool_use",
            "container": {"id": "container-1", "expires_at": "2026-01-01T00:00:00Z"},
            "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "call-1", "name": "search", "input": {"q": "x"}},
            ],
            "usage": {"input_tokens": 2, "output_tokens": 3},
        }
    )
    assert response.visible_parts()[0].text == "checking"
    assert response.runtime_tool_calls() == (StructuredToolCall("call-1", "search", {"q": "x"}),)
    assert response.usage is not None and response.usage.total_tokens == 5
    assert response.metadata["provider"] == "anthropic-messages"


def test_anthropic_messages_allows_zero_max_tokens_for_cache_prewarming() -> None:
    payload = AnthropicMessagesCodec(model="claude-test").encode_request(
        ModelRequest(
            messages=(Message.user("warm cache"),),
            options=ModelOptions(max_output_tokens=0),
        )
    )

    assert payload["max_tokens"] == 0


def test_anthropic_messages_decodes_empty_terminal_response_for_zero_tokens() -> None:
    response = AnthropicMessagesCodec(model="claude-test").decode_response(
        {
            "type": "message",
            "role": "assistant",
            "id": "msg-empty",
            "model": "claude-test",
            "stop_reason": "max_tokens",
            "content": [],
            "usage": {"input_tokens": 2, "output_tokens": 0},
        }
    )

    assert response.output == ()


def test_anthropic_messages_usage_accepts_only_web_search_server_tool_counter() -> None:
    response = AnthropicMessagesCodec(model="claude-test").decode_response(
        {
            "type": "message",
            "role": "assistant",
            "id": "msg-usage",
            "model": "claude-test",
            "stop_reason": "end_turn",
            "content": [],
            "usage": {
                "input_tokens": 2,
                "output_tokens": 0,
                "server_tool_use": {"web_search_requests": 1},
            },
        }
    )
    assert response.usage is not None and response.usage.total_tokens == 2

    with pytest.raises(AnthropicMessagesError, match="server_tool_use is invalid"):
        AnthropicMessagesCodec(model="claude-test").decode_response(
            {
                "type": "message",
                "role": "assistant",
                "id": "msg-usage-invalid",
                "model": "claude-test",
                "stop_reason": "end_turn",
                "content": [],
                "usage": {
                    "input_tokens": 2,
                    "output_tokens": 0,
                    "server_tool_use": {"web_search_requests": 1, "vendor_requests": 1},
                },
            }
        )


def test_anthropic_messages_codec_rejects_non_object_schemas() -> None:
    codec = AnthropicMessagesCodec(model="claude-test")
    with pytest.raises(AnthropicMessagesError, match="input_schema must be an object"):
        codec.encode_request(
            ModelRequest(
                messages=(Message.user("hello"),),
                runtime_tools=(StructuredToolSpec("search", "search", True),),
            )
        )
    with pytest.raises(
        AnthropicMessagesError, match="JSON schema response format requires an object"
    ):
        codec.encode_request(
            ModelRequest(
                messages=(Message.user("hello"),),
                response_format=ResponseFormat("json_schema", True),
            )
        )


def test_anthropic_messages_codec_encodes_standard_sampling_options() -> None:
    payload = AnthropicMessagesCodec(model="claude-test").encode_request(
        ModelRequest(
            messages=(Message.user("hello"),),
            options=ModelOptions(temperature=0.0, top_p=1.0),
        )
    )

    assert payload["temperature"] == 0.0
    assert payload["top_p"] == 1.0


@pytest.mark.parametrize(
    "options, field",
    ((ModelOptions(temperature=-0.1), "temperature"), (ModelOptions(top_p=1.1), "top_p")),
)
def test_anthropic_messages_codec_rejects_sampling_options_outside_standard_range(
    options: ModelOptions, field: str
) -> None:
    with pytest.raises(AnthropicMessagesError, match=field):
        AnthropicMessagesCodec(model="claude-test").encode_request(
            ModelRequest(messages=(Message.user("hello"),), options=options)
        )


@pytest.mark.parametrize("name", ("invalid.name", "x" * 65))
def test_anthropic_messages_codec_rejects_invalid_official_tool_names(name: str) -> None:
    with pytest.raises(AnthropicMessagesError, match="must match"):
        AnthropicMessagesCodec(model="claude-test").encode_request(
            ModelRequest(
                messages=(Message.user("hello"),),
                runtime_tools=(StructuredToolSpec(name, "test", {"type": "object"}),),
            )
        )


def test_anthropic_messages_codec_encodes_image_artifacts_as_image_sources() -> None:
    codec = AnthropicMessagesCodec(model="claude-test")
    payload = codec.encode_request(
        ModelRequest(
            messages=(
                Message(
                    "user",
                    (
                        ContentPart.artifact_part(
                            ArtifactRef("image-file", media_type="IMAGE/PNG")
                        ),
                        ContentPart(
                            "file",
                            uri="https://example.test/image.webp",
                            media_type="image/webp",
                        ),
                        ContentPart(
                            "file",
                            uri="DATA:image/gif;BASE64,aGVsbG8=",
                        ),
                        ContentPart.artifact_part(
                            ArtifactRef("pdf-file", media_type="application/pdf", name="report.pdf")
                        ),
                        ContentPart.artifact_part(
                            ArtifactRef("dataset-file", media_type="text/csv", name="data.csv")
                        ),
                    ),
                ),
            )
        )
    )

    assert payload["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "file", "file_id": "image-file"}},
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://example.test/image.webp"},
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/gif",
                        "data": "aGVsbG8=",
                    },
                },
                {
                    "type": "document",
                    "source": {"type": "file", "file_id": "pdf-file"},
                    "title": "report.pdf",
                },
                {"type": "container_upload", "file_id": "dataset-file"},
            ],
        }
    ]


def test_anthropic_messages_replays_official_native_blocks_and_container_continuation() -> None:
    codec = AnthropicMessagesCodec(model="claude-test")
    payload = codec.encode_request(
        ModelRequest(
            messages=(
                Message(
                    "user",
                    (
                        ContentPart(
                            "opaque",
                            data={
                                "anthropic": {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "aGVsbG8=",
                                    },
                                    "transformations": {"oversized_image": "error"},
                                }
                            },
                        ),
                        ContentPart(
                            "opaque",
                            data={
                                "anthropic": {
                                    "type": "search_result",
                                    "source": "https://example.test",
                                    "title": "Example",
                                    "content": [{"type": "text", "text": "result"}],
                                }
                            },
                        ),
                    ),
                ),
                Message.assistant(
                    (ContentPart.text_part("previous"),),
                    metadata={"anthropic": {"container_id": "container-1"}},
                ),
            )
        )
    )
    assert payload["container"] == {"id": "container-1"}
    assert cast(list[dict[str, Any]], payload["messages"])[0]["content"][0]["transformations"] == {
        "oversized_image": "error"
    }


def test_anthropic_messages_response_container_and_assistant_upload_round_trip() -> None:
    codec = AnthropicMessagesCodec(model="claude-test")
    response = codec.decode_response(
        {
            "type": "message",
            "role": "assistant",
            "id": "msg-1",
            "model": "claude-test",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "container": {
                "id": "container-1",
                "expires_at": "2026-01-01T00:00:00Z",
                "skills": [{"skill_id": "skill-1", "type": "custom", "version": "v1"}],
            },
            "content": [{"type": "container_upload", "file_id": "file-1"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    payload = codec.encode_request(
        ModelRequest(messages=(Message.user("x"), response.to_assistant_message()))
    )
    assert payload["container"] == {"id": "container-1"}
    assert cast(list[dict[str, Any]], payload["messages"])[1]["content"] == [
        {"type": "container_upload", "file_id": "file-1"}
    ]

    with pytest.raises(AnthropicMessagesError, match="expires_at"):
        codec.decode_response(
            {
                "type": "message",
                "role": "assistant",
                "id": "msg-2",
                "model": "claude-test",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "x"}],
                "container": {"id": "container-2"},
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )


def test_anthropic_messages_text_citations_are_not_compressed_or_lost() -> None:
    codec = AnthropicMessagesCodec(model="claude-test")
    response = codec.decode_response(
        {
            "type": "message",
            "role": "assistant",
            "id": "msg-1",
            "model": "claude-test",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "quoted", "citations": []}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    payload = codec.encode_request(
        ModelRequest(messages=(Message.user("x"), response.to_assistant_message()))
    )
    assert cast(list[dict[str, Any]], payload["messages"])[1]["content"] == [
        {"type": "text", "text": "quoted", "citations": []}
    ]

    with pytest.raises(AnthropicMessagesError, match="do not support cache_control"):
        codec.decode_response(
            {
                "type": "message",
                "role": "assistant",
                "id": "msg-2",
                "model": "claude-test",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )


def test_anthropic_messages_tool_use_metadata_round_trips() -> None:
    codec = AnthropicMessagesCodec(model="claude-test")
    response = codec.decode_response(
        {
            "type": "message",
            "role": "assistant",
            "id": "msg-1",
            "model": "claude-test",
            "stop_reason": "tool_use",
            "container": {"id": "container-1", "expires_at": "2026-01-01T00:00:00Z"},
            "content": [
                {
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "search",
                    "input": {},
                    "caller": {"type": "code_execution_20260521", "tool_id": "tool-1"},
                    "toolset_name": None,
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    call = response.runtime_tool_calls()[0]
    assert response.to_assistant_message().metadata == response.metadata
    assert call.metadata == {
        "anthropic": {
            "caller": {"type": "code_execution_20260521", "tool_id": "tool-1"},
            "toolset_name": None,
        }
    }
    payload = codec.encode_request(
        ModelRequest(messages=(Message.user("x"), response.to_assistant_message()))
    )
    assert payload["container"] == {"id": "container-1"}
    assert cast(list[dict[str, Any]], payload["messages"])[1]["content"] == [
        {
            "type": "tool_use",
            "id": "call-1",
            "name": "search",
            "input": {},
            "caller": {"type": "code_execution_20260521", "tool_id": "tool-1"},
            "toolset_name": None,
        }
    ]


def test_anthropic_messages_codec_rejects_nonstandard_message_role_system() -> None:
    codec = AnthropicMessagesCodec(model="claude-test")
    with pytest.raises(AnthropicMessagesError, match="system content before messages"):
        codec.encode_request(
            ModelRequest(
                messages=(
                    Message.user("first"),
                    Message.system("late policy"),
                )
            )
        )


def test_anthropic_messages_image_artifacts_require_image_capability() -> None:
    default_profile = AnthropicMessagesProfile()
    codec = AnthropicMessagesCodec(
        model="claude-test",
        profile=AnthropicMessagesProfile(
            capabilities=replace(
                default_profile.capabilities,
                input_modalities=frozenset({"text", "file"}),
            )
        ),
    )

    with pytest.raises(AnthropicMessagesError, match="image input"):
        codec.encode_request(
            ModelRequest(
                messages=(
                    Message(
                        "user",
                        (ContentPart.artifact_part(ArtifactRef("image-file", "image/png")),),
                    ),
                )
            )
        )

    with pytest.raises(AnthropicMessagesError, match="must use image/jpeg"):
        AnthropicMessagesCodec(model="claude-test").encode_request(
            ModelRequest(
                messages=(
                    Message(
                        "user",
                        (ContentPart.artifact_part(ArtifactRef("bmp-file", "image/bmp")),),
                    ),
                )
            )
        )


@pytest.mark.parametrize(
    "response_format",
    (ResponseFormat("json_object"),),
    ids=("json-object",),
)
def test_anthropic_messages_codec_thaws_json_object_schema_at_wire_boundary(
    response_format: ResponseFormat | None,
) -> None:
    default_profile = AnthropicMessagesProfile()
    messages_profile = AnthropicMessagesProfile(
        capabilities=replace(default_profile.capabilities, json_mode=True),
        json_object_schema={
            "type": "object",
            "properties": {"values": {"type": "array", "items": {"type": "string"}}},
        },
    )
    codec = AnthropicMessagesCodec(model="claude-test", profile=messages_profile)
    model_request = ModelRequest(
        messages=(Message.user("hello"),),
        response_format=response_format,
    )

    payload = codec.encode_request(model_request)
    serialized = json.loads(json.dumps(payload))
    assert serialized["output_config"]["format"]["schema"] == {
        "type": "object",
        "properties": {"values": {"type": "array", "items": {"type": "string"}}},
    }

    schema = cast(dict[str, Any], payload["output_config"])["format"]["schema"]
    cast(dict[str, Any], schema)["properties"]["changed"] = {"type": "string"}
    fresh_payload = codec.encode_request(model_request)
    assert (
        "changed"
        not in cast(dict[str, Any], fresh_payload["output_config"])["format"]["schema"][
            "properties"
        ]
    )


def test_anthropic_messages_stream_decoder_builds_complete_response() -> None:
    decoder = AnthropicMessagesStreamDecoder(AnthropicMessagesProfile())
    decoder.apply_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "type": "message",
                "role": "assistant",
                "id": "msg-1",
                "model": "claude-test",
                "content": [],
                "usage": {"input_tokens": 2, "output_tokens": 0},
            },
        },
    )
    decoder.apply_event(
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "h"}},
    )
    decoder.apply_event(
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "i"}},
    )
    decoder.apply_event("content_block_stop", {"type": "content_block_stop", "index": 0})
    decoder.apply_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 3},
        },
    )
    done, _ = decoder.apply_event("message_stop", {"type": "message_stop"})
    completed = decoder.completed_response()

    assert done is True
    assert completed.visible_parts()[0].text == "hi"
    assert completed.usage is not None and completed.usage.total_tokens == 5
    assert completed.metadata["provider"] == "anthropic-messages"


def test_anthropic_messages_stream_preserves_container_upload_and_citations() -> None:
    decoder = AnthropicMessagesStreamDecoder(AnthropicMessagesProfile())
    decoder.apply_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "type": "message",
                "role": "assistant",
                "id": "msg-1",
                "model": "claude-test",
                "content": [],
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
    )
    decoder.apply_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": "x", "citations": []},
        },
    )
    decoder.apply_event("content_block_stop", {"type": "content_block_stop", "index": 0})
    decoder.apply_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "container_upload", "file_id": "file-1"},
        },
    )
    decoder.apply_event("content_block_stop", {"type": "content_block_stop", "index": 1})
    decoder.apply_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 1},
        },
    )
    decoder.apply_event("message_stop", {"type": "message_stop"})
    response = decoder.completed_response()
    assert response.visible_parts()[0].metadata == {"anthropic": {"extra": {"citations": ()}}}
    assert response.visible_parts()[1].artifact == ArtifactRef(
        "file-1", metadata={"anthropic": {"type": "container_upload", "file_id": "file-1"}}
    )


def test_anthropic_messages_stream_requires_delta_output_usage() -> None:
    decoder = AnthropicMessagesStreamDecoder(AnthropicMessagesProfile())
    decoder.apply_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "type": "message",
                "role": "assistant",
                "id": "msg-1",
                "model": "claude-test",
                "content": [],
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
    )
    with pytest.raises(AnthropicMessagesError, match="unsupported fields"):
        decoder.apply_event(
            "message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}
        )


def test_anthropic_messages_stream_allows_empty_terminal_output() -> None:
    decoder = AnthropicMessagesStreamDecoder(AnthropicMessagesProfile())
    decoder.apply_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "type": "message",
                "role": "assistant",
                "id": "msg-empty",
                "model": "claude-test",
                "content": [],
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
    )
    decoder.apply_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "max_tokens"},
            "usage": {"output_tokens": 0},
        },
    )
    done, _ = decoder.apply_event("message_stop", {"type": "message_stop"})

    assert done is True
    assert decoder.completed_response().output == ()


def test_anthropic_messages_stream_rejects_nonstandard_event_and_delta_fields() -> None:
    decoder = AnthropicMessagesStreamDecoder(AnthropicMessagesProfile())
    decoder.apply_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "type": "message",
                "role": "assistant",
                "id": "msg-1",
                "model": "claude-test",
                "content": [],
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
    )
    with pytest.raises(AnthropicMessagesError, match="unsupported Anthropic stream event type"):
        decoder.apply_event("vendor_event", {"type": "vendor_event"})
    with pytest.raises(AnthropicMessagesError, match="content_block_start has unsupported"):
        decoder.apply_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
                "vendor": True,
            },
        )
    decoder.apply_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    )
    with pytest.raises(AnthropicMessagesError, match="text_delta has unsupported"):
        decoder.apply_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "", "vendor": True},
            },
        )
    decoder.apply_event("content_block_stop", {"type": "content_block_stop", "index": 0})
    decoder.apply_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 0},
        },
    )
    decoder.apply_event("message_stop", {"type": "message_stop"})
    assert decoder.completed_response().visible_parts()[0].text == ""


def test_anthropic_messages_thinking_deltas_stay_incremental_and_finalize_once() -> None:
    decoder = AnthropicMessagesStreamDecoder(AnthropicMessagesProfile())
    decoder.apply_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "type": "message",
                "role": "assistant",
                "id": "msg-1",
                "model": "claude-test",
                "content": [],
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
    )
    decoder.apply_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        },
    )
    for _ in range(4096):
        _, deltas = decoder.apply_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "x"},
            },
        )
        assert isinstance(deltas[0], ModelReasoningDelta)
        content_delta = next(delta for delta in deltas if isinstance(delta, ModelContentDelta))
        assert content_delta.text_delta == "x"
        assert content_delta.data == {}
    _, signature_deltas = decoder.apply_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig"},
        },
    )
    signature_delta = cast(ModelContentDelta, signature_deltas[0])
    assert signature_delta.data == {"anthropic": {"type": "thinking", "signature": "sig"}}
    decoder.apply_event("content_block_stop", {"type": "content_block_stop", "index": 0})
    decoder.apply_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 4096},
        },
    )
    decoder.apply_event("message_stop", {"type": "message_stop"})

    part = decoder.completed_response().visible_parts()[0]
    assert part.text == "x" * 4096
    assert part.data == {
        "anthropic": {
            "type": "thinking",
            "thinking": "x" * 4096,
            "signature": "sig",
        }
    }


def test_anthropic_messages_stream_rejects_unsigned_thinking_and_fallback() -> None:
    decoder = AnthropicMessagesStreamDecoder(AnthropicMessagesProfile())
    decoder.apply_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "type": "message",
                "role": "assistant",
                "id": "msg-1",
                "model": "claude-test",
                "content": [],
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
    )
    decoder.apply_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        },
    )
    with pytest.raises(AnthropicMessagesError, match="requires a signature"):
        decoder.apply_event("content_block_stop", {"type": "content_block_stop", "index": 0})

    fallback = AnthropicMessagesStreamDecoder(AnthropicMessagesProfile())
    fallback.apply_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "type": "message",
                "role": "assistant",
                "id": "msg-2",
                "model": "claude-test",
                "content": [],
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
    )
    with pytest.raises(AnthropicMessagesError, match="unenabled beta"):
        fallback.apply_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "fallback"},
            },
        )


def test_anthropic_messages_stream_decoder_accumulates_tool_input() -> None:
    decoder = AnthropicMessagesStreamDecoder(AnthropicMessagesProfile())
    decoder.apply_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "type": "message",
                "role": "assistant",
                "id": "msg-1",
                "model": "claude-test",
                "content": [],
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
    )
    decoder.apply_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "call-1",
                "name": "search",
                "input": {},
            },
        },
    )
    decoder.apply_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"q":"x"}'},
        },
    )
    decoder.apply_event("content_block_stop", {"type": "content_block_stop", "index": 0})
    decoder.apply_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 1},
        },
    )
    decoder.apply_event("message_stop", {"type": "message_stop"})

    assert decoder.completed_response().runtime_tool_calls() == (
        StructuredToolCall("call-1", "search", {"q": "x"}),
    )


async def test_anthropic_messages_client_uses_http_transport_and_maps_errors() -> None:
    captured: dict[str, object] = {}

    async def success_handler(raw: httpx.Request) -> httpx.Response:
        captured["api_key"] = raw.headers["x-api-key"]
        captured["body"] = json.loads(raw.content)
        return httpx.Response(
            200,
            json={
                "type": "message",
                "role": "assistant",
                "id": "msg-1",
                "model": "claude-test",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "done"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
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
    assert captured["api_key"] == "secret"
    assert isinstance(model, Model)
    assert not hasattr(model, "complete")
    assert not hasattr(model, "stream")

    async def error_handler(raw: httpx.Request) -> httpx.Response:
        return httpx.Response(
            529,
            json={"error": {"type": "overloaded_error", "message": "busy"}},
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
    assert caught.value.info.code == "overloaded_error"
    assert caught.value.info.provider == "anthropic-messages"
    assert caught.value.info.retryable is True


async def test_anthropic_messages_client_sends_image_file_reference_without_beta_header() -> None:
    captured: dict[str, object] = {}

    async def handler(raw: httpx.Request) -> httpx.Response:
        captured["beta"] = raw.headers.get("anthropic-beta")
        captured["body"] = json.loads(raw.content)
        return httpx.Response(
            200,
            json={
                "type": "message",
                "role": "assistant",
                "id": "msg-1",
                "model": "vision-test",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "done"}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
            request=raw,
        )

    default_profile = AnthropicMessagesProfile()
    image_profile = AnthropicMessagesProfile(
        capabilities=replace(
            default_profile.capabilities,
            input_modalities=frozenset({"text", "image"}),
        )
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = AnthropicMessagesModel(
            base_url="https://provider.test",
            api_key="secret",
            model="vision-test",
            profile=image_profile,
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
            "content": [
                {
                    "type": "image",
                    "source": {"type": "file", "file_id": "image-file"},
                }
            ],
        }
    ]
    assert captured["beta"] is None


async def test_anthropic_messages_stream_overload_keeps_semantic_status_and_retryability() -> None:
    body = (
        "event: error\n"
        'data: {"type":"error","error":'
        '{"type":"overloaded_error","message":"busy"}}\n\n'
    )

    def handler(raw: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, request=raw)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ModelError) as caught:
            await http_model(client).invoke(
                ModelRequest(messages=(Message.user("hello"),)),
                RunContext("run-1", 1.0),
                stream=True,
                emit_delta=None,
            )

    assert caught.value.info.code == "overloaded_error"
    assert caught.value.info.status_code is None
    assert caught.value.info.retryable is True


def test_anthropic_messages_codec_rejects_invalid_envelope() -> None:
    with pytest.raises(AnthropicMessagesError, match="type='message'"):
        AnthropicMessagesCodec(model="claude-test").decode_response({"type": "error"})


async def test_anthropic_messages_client_decodes_named_sse_stream() -> None:
    body = "".join(
        (
            "event: message_start\n",
            'data: {"type":"message_start","message":{"type":"message",'
            '"role":"assistant","id":"msg-1","model":"claude-test",'
            '"content":[],"usage":{"input_tokens":2,"output_tokens":0}}}\n\n',
            "event: content_block_start\n",
            'data: {"type":"content_block_start","index":0,'
            '"content_block":{"type":"text","text":"hello"}}\n\n',
            "event: content_block_stop\n",
            'data: {"type":"content_block_stop","index":0}\n\n',
            "event: message_delta\n",
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
            '"usage":{"output_tokens":3}}\n\n',
            "event: message_stop\n",
            'data: {"type":"message_stop"}\n\n',
        )
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

    assert any(isinstance(delta, ModelContentDelta) for delta in deltas)
    assert any(isinstance(delta, ModelUsageDelta) for delta in deltas)
    assert result.visible_parts()[0].text == "hello"
    assert result.usage is not None
    assert result.usage.total_tokens == 5
    assert unobserved.visible_parts()[0].text == "hello"
