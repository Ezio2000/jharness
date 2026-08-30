from __future__ import annotations

import base64
from dataclasses import replace

import pytest

from jharness.kernel import (
    ArtifactRef,
    ContentPart,
    Message,
    StructuredToolCall,
    StructuredToolSpec,
    ToolChoice,
    ToolFailure,
    ToolSuccess,
)
from jharness.models.anthropic import AnthropicMessagesError, AnthropicMessagesProfile
from jharness.models.anthropic.messages.messages import (
    decode_content_blocks,
    encode_message,
    encode_messages,
    encode_user_content_part,
)
from jharness.models.anthropic.messages.tools import (
    decode_tool_uses,
)
from jharness.models.anthropic.messages.tools import (
    encode_tool_choice as encode_anthropic_choice,
)
from jharness.models.anthropic.messages.tools import (
    encode_tools as encode_anthropic_tools,
)
from jharness.models.openai import OpenAIChatError, OpenAIChatProfile
from jharness.models.openai.chat.messages import (
    decode_message_content,
    decode_message_refusal,
    encode_chat_message,
    encode_content_part,
)
from jharness.models.openai.chat.tools import (
    decode_tool_calls,
)
from jharness.models.openai.chat.tools import (
    encode_tool_choice as encode_openai_choice,
)
from jharness.models.openai.chat.tools import (
    encode_tools as encode_openai_tools,
)


def tool_message(call_id: str, *, failure: bool = False) -> Message:
    parts = (ContentPart.text_part("result"),)
    outcome = (
        ToolFailure.from_error("failed", "tool failed")
        if failure
        else ToolSuccess(parts, {"value": 1})
    )
    return Message.tool(call_id, outcome)


def test_openai_chat_message_codec_covers_roles_multimodal_and_native_parts() -> None:
    profile = OpenAIChatProfile()
    profile = OpenAIChatProfile(
        capabilities=replace(
            profile.capabilities,
            input_modalities=frozenset({"text", "image", "audio", "file"}),
        )
    )
    call = StructuredToolCall("call-1", "search", {"q": "x"})

    assert encode_chat_message(Message.external("callback"), profile) == {
        "role": "user",
        "content": "callback",
    }
    assert encode_chat_message(Message.system("policy"), profile)["content"] == "policy"
    assistant = Message.assistant(
        (
            ContentPart.text_part("cannot"),
            call,
        ),
    )
    encoded_assistant = encode_chat_message(assistant, profile)
    assert encoded_assistant["tool_calls"][0]["id"] == "call-1"
    assert encode_chat_message(tool_message("call-1"), profile) == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "result",
    }

    assert (
        encode_content_part(ContentPart("image", uri="https://x/image.png"), profile)["type"]
        == "image_url"
    )
    assert encode_content_part(ContentPart.artifact_part(ArtifactRef("file-1")), profile) == {
        "type": "file",
        "file": {"file_id": "file-1"},
    }
    assert (
        encode_content_part(
            ContentPart("file", uri="data:text/plain;base64,SGk=", name="note.txt"), profile
        )["file"]["filename"]
        == "note.txt"
    )

    decoded = decode_message_content(
        "hello",
    )
    assert [part.type for part in decoded] == ["text"]
    assert decode_message_refusal("blocked")[0].type == "refusal"


def test_openai_chat_files_use_one_nested_shape_and_require_file_capability() -> None:
    default = OpenAIChatProfile()
    file_profile = OpenAIChatProfile(
        capabilities=replace(default.capabilities, input_modalities=frozenset({"text", "file"})),
    )
    image_artifact = ContentPart.artifact_part(
        ArtifactRef("file-image", media_type="IMAGE/PNG", name="chart.png")
    )

    assert encode_content_part(image_artifact, file_profile) == {
        "type": "file",
        "file": {"file_id": "file-image"},
    }
    assert encode_content_part(
        ContentPart("file", uri="data:image/png;base64,AA=="),
        file_profile,
    ) == {
        "type": "file",
        "file": {"file_data": "AA==", "filename": "file"},
    }
    with pytest.raises(OpenAIChatError, match="does not support file input"):
        encode_content_part(
            ContentPart.artifact_part(ArtifactRef("file-text")),
            OpenAIChatProfile(
                capabilities=replace(
                    default.capabilities,
                    input_modalities=frozenset({"text", "image", "audio"}),
                )
            ),
        )


def test_openai_chat_tool_codec_validates_choices_and_arguments() -> None:
    spec = StructuredToolSpec("search", "search", {"type": "object"})
    profile = OpenAIChatProfile()

    assert encode_openai_tools((spec,), profile)[0]["function"]["name"] == "search"
    assert (
        encode_openai_choice(ToolChoice(), tools_by_name={"search": spec}, profile=profile)
        == "auto"
    )
    assert encode_openai_choice(
        ToolChoice(type="runtime", name="search"),
        tools_by_name={"search": spec},
        profile=profile,
    ) == {"type": "function", "function": {"name": "search"}}
    calls = decode_tool_calls(
        [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q":"x"}'},
            },
            {
                "id": "call-2",
                "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            },
        ]
    )
    assert calls == [
        StructuredToolCall("call-1", "search", {"q": "x"}),
        StructuredToolCall("call-2", "search", {}),
    ]

    with pytest.raises(OpenAIChatError, match="requires at least one"):
        encode_openai_choice(ToolChoice("required"), tools_by_name={}, profile=profile)
    with pytest.raises(OpenAIChatError, match="unavailable"):
        encode_openai_choice(
            ToolChoice(type="runtime", name="other"),
            tools_by_name={"search": spec},
            profile=profile,
        )
    raw_call = decode_tool_calls(
        [
            {
                "id": "call",
                "type": "function",
                "function": {"name": "search", "arguments": "{"},
            }
        ]
    )[0]
    assert isinstance(raw_call, StructuredToolCall)
    assert raw_call.arguments is None
    assert raw_call.raw_input == "{"


def test_openai_chat_message_codec_rejects_unsupported_content() -> None:
    default = OpenAIChatProfile()
    profile = OpenAIChatProfile(
        capabilities=replace(
            default.capabilities,
            input_modalities=frozenset({"text"}),
        ),
    )
    with pytest.raises(OpenAIChatError, match="image input"):
        encode_content_part(ContentPart("image", uri="https://x/image"), profile)
    with pytest.raises(OpenAIChatError, match="audio input"):
        encode_content_part(ContentPart("audio", uri="https://x/audio"), profile)
    with pytest.raises(OpenAIChatError, match="string or null"):
        decode_message_content(3)
    refusal = decode_message_refusal("")
    assert len(refusal) == 1
    assert refusal[0].type == "refusal"
    assert refusal[0].text == ""


def test_anthropic_messages_message_codec_covers_system_tools_and_native_parts() -> None:
    profile = AnthropicMessagesProfile()
    call = StructuredToolCall("call-1", "search", {"q": "x"})
    thinking = ContentPart(
        type="thinking",
        text="reason",
        data={"anthropic": {"type": "thinking", "thinking": "reason", "signature": "sig"}},
    )
    redacted = ContentPart(
        type="redacted_thinking",
        data={"anthropic": {"type": "redacted_thinking", "data": "secret"}},
    )
    system, messages = encode_messages(
        (
            Message.system("policy"),
            Message.user("hello"),
            Message.assistant((thinking, redacted, call)),
            tool_message("call-1", failure=True),
            Message.external("callback"),
        ),
        profile,
    )

    assert system == [{"type": "text", "text": "policy"}]
    assert messages[1]["content"][-1] == {
        "type": "tool_use",
        "id": "call-1",
        "name": "search",
        "input": {"q": "x"},
    }
    assert messages[2]["content"][0]["is_error"] is True
    assert messages[-1] == {"role": "user", "content": "callback"}

    blocks: list[dict[str, object]] = [
        {"type": "text", "text": "hello"},
        {"type": "thinking", "thinking": "why", "signature": "sig"},
        {"type": "redacted_thinking", "data": "secret"},
        {"type": "tool_use", "id": "call-2", "name": "search", "input": {}},
    ]
    decoded = Message.assistant(decode_content_blocks(blocks, profile))
    assert [part.type for part in decoded.visible_parts()] == [
        "text",
        "thinking",
        "redacted_thinking",
    ]
    assert decoded.runtime_tool_calls() == (StructuredToolCall("call-2", "search", {}),)
    assert encode_message(Message.external("callback"), profile)["role"] == "user"


def test_anthropic_messages_message_codec_round_trips_redacted_thinking() -> None:
    default_profile = AnthropicMessagesProfile()
    metadata_redacted = ContentPart(
        type="redacted_thinking",
        metadata={"anthropic": {"data": "secret"}},
    )
    assert encode_message(
        Message.assistant((metadata_redacted,)),
        default_profile,
    )["content"] == [{"type": "redacted_thinking", "data": "secret"}]

    native_redacted = ContentPart(
        type="redacted_thinking",
        data={"anthropic": {"type": "redacted_thinking", "data": "secret"}},
    )
    assert encode_message(Message.assistant((native_redacted,)), default_profile)["content"] == [
        {"type": "redacted_thinking", "data": "secret"}
    ]

    thinking = ContentPart(
        type="thinking",
        text="reason",
        metadata={"anthropic": {"signature": "sig"}},
    )
    assert encode_message(Message.assistant((thinking,)), default_profile)["content"] == [
        {"type": "thinking", "thinking": "reason", "signature": "sig"}
    ]


def test_anthropic_messages_native_blocks_reject_unknown_metadata_and_require_signature() -> None:
    profile = AnthropicMessagesProfile()
    unknown_text = ContentPart.text_part(
        "hello",
        metadata={"anthropic": {"extra": {"vendor_extra": True}}},
    )
    unsigned_thinking = ContentPart(type="thinking", text="reason")

    with pytest.raises(AnthropicMessagesError, match="unsupported field: vendor_extra"):
        encode_message(Message.assistant((unknown_text,)), profile)
    with pytest.raises(AnthropicMessagesError, match="metadata signature"):
        encode_message(Message.assistant((unsigned_thinking,)), profile)


@pytest.mark.parametrize(
    "citation",
    (
        {
            "type": "char_location",
            "cited_text": "text",
            "document_index": 0,
            "start_char_index": 0,
            "end_char_index": 4,
            "document_title": None,
            "file_id": None,
        },
        {
            "type": "page_location",
            "cited_text": "text",
            "document_index": 0,
            "start_page_number": 1,
            "end_page_number": 1,
        },
        {
            "type": "content_block_location",
            "cited_text": "text",
            "document_index": 0,
            "start_block_index": 0,
            "end_block_index": 1,
        },
        {
            "type": "web_search_result_location",
            "cited_text": "text",
            "encrypted_index": "encrypted",
            "url": "https://example.com",
            "title": None,
        },
        {
            "type": "search_result_location",
            "cited_text": "text",
            "search_result_index": 0,
            "source": "source",
            "start_block_index": 0,
            "end_block_index": 1,
            "title": None,
        },
    ),
)
def test_anthropic_messages_accepts_all_official_citation_variants(
    citation: dict[str, object],
) -> None:
    parts = decode_content_blocks(
        [{"type": "text", "text": "answer", "citations": [citation]}], AnthropicMessagesProfile()
    )
    assert parts[0].metadata["anthropic"]["extra"]["citations"] == [citation]


def test_anthropic_messages_rejects_unknown_citation_fields() -> None:
    with pytest.raises(AnthropicMessagesError, match="unsupported field: vendor_extra"):
        decode_content_blocks(
            [
                {
                    "type": "text",
                    "text": "answer",
                    "citations": [
                        {
                            "type": "page_location",
                            "cited_text": "text",
                            "document_index": 0,
                            "start_page_number": 1,
                            "end_page_number": 1,
                            "vendor_extra": True,
                        }
                    ],
                }
            ],
            AnthropicMessagesProfile(),
        )


def test_anthropic_messages_native_document_citations_use_config_not_text_citations() -> None:
    profile = AnthropicMessagesProfile()
    document = ContentPart(
        "opaque",
        data={
            "anthropic": {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "aGVsbG8=",
                },
                "citations": {"enabled": True},
            }
        },
    )
    assert encode_user_content_part(document, profile)["citations"] == {"enabled": True}

    invalid = ContentPart(
        "opaque",
        data={
            "anthropic": {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "aGVsbG8=",
                },
                "citations": [],
            }
        },
    )
    with pytest.raises(AnthropicMessagesError, match="document citations must be an object"):
        encode_user_content_part(invalid, profile)


@pytest.mark.parametrize(
    "block, match",
    (
        (
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/svg+xml", "data": "aGVsbG8="},
            },
            "unsupported media_type",
        ),
        (
            {
                "type": "document",
                "source": {"type": "base64", "media_type": "text/plain", "data": "aGVsbG8="},
            },
            "unsupported media_type",
        ),
        (
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/html", "data": "text"},
            },
            "must be text/plain",
        ),
        (
            {"type": "document", "source": {"type": "container_upload", "file_id": "file"}},
            "unsupported Anthropic-native document source type",
        ),
    ),
)
def test_anthropic_messages_native_sources_reject_unsupported_shapes(
    block: dict[str, object], match: str
) -> None:
    with pytest.raises(AnthropicMessagesError, match=match):
        encode_user_content_part(
            ContentPart("opaque", data={"anthropic": block}), AnthropicMessagesProfile()
        )


def test_anthropic_messages_document_content_source_validates_nested_blocks() -> None:
    profile = AnthropicMessagesProfile()
    document = ContentPart(
        "opaque",
        data={
            "anthropic": {
                "type": "document",
                "source": {
                    "type": "content",
                    "content": [
                        {"type": "text", "text": "source text"},
                        {"type": "image", "source": {"type": "file", "file_id": "image"}},
                    ],
                },
            }
        },
    )
    assert encode_user_content_part(document, profile)["source"]["type"] == "content"
    bad_document = ContentPart(
        "opaque",
        data={
            "anthropic": {
                "type": "document",
                "source": {"type": "content", "content": [{"type": "vendor"}]},
            }
        },
    )
    with pytest.raises(AnthropicMessagesError, match="must be text or image"):
        encode_user_content_part(bad_document, profile)


def test_anthropic_messages_media_and_tool_choice_codec() -> None:
    profile = AnthropicMessagesProfile()
    image = ContentPart(
        "image",
        uri="data:image/png;base64,aGVsbG8=",
        media_type="image/png",
    )
    pdf = ContentPart(
        "file",
        uri="data:application/pdf;base64,aGVsbG8=",
        media_type="application/pdf",
    )
    text = ContentPart(
        "file",
        uri=f"data:text/plain;base64,{base64.b64encode(b'hello').decode()}",
        media_type="text/plain",
    )

    assert encode_user_content_part(image, profile)["source"]["type"] == "base64"
    with pytest.raises(AnthropicMessagesError, match="image data URLs"):
        encode_user_content_part(ContentPart("image", uri="data:;base64,aGVsbG8="), profile)
    assert encode_user_content_part(pdf, profile)["source"]["media_type"] == "application/pdf"
    assert encode_user_content_part(text, profile)["source"]["data"] == "hello"
    assert encode_user_content_part(
        ContentPart.artifact_part(ArtifactRef("file-1", media_type="application/pdf")),
        profile,
    )["source"] == {
        "type": "file",
        "file_id": "file-1",
    }
    assert encode_user_content_part(
        ContentPart.artifact_part(ArtifactRef("dataset-1", media_type="text/csv")),
        profile,
    ) == {"type": "container_upload", "file_id": "dataset-1"}
    spec = StructuredToolSpec("search", "search", {"type": "object"})
    assert encode_anthropic_tools((spec,), (), profile)[0]["name"] == "search"
    assert encode_anthropic_choice(
        ToolChoice("required", allow_parallel_runtime_tool_calls=False),
        runtime_tool_names={"search"},
        provider_tools=(),
        may_return_runtime_tool_calls=True,
        profile=profile,
    ) == {"type": "any", "disable_parallel_tool_use": True}


def test_anthropic_messages_codec_rejects_invalid_roles_media_and_blocks() -> None:
    default = AnthropicMessagesProfile()
    profile = AnthropicMessagesProfile(
        capabilities=replace(
            default.capabilities,
            input_modalities=frozenset({"text"}),
        )
    )
    with pytest.raises(AnthropicMessagesError, match="all system content before messages"):
        encode_messages(
            (Message.user("hello"), Message.system("late")),
            profile,
        )
    with pytest.raises(AnthropicMessagesError, match="image input"):
        encode_user_content_part(ContentPart("image", uri="https://x/image"), profile)
    with pytest.raises(AnthropicMessagesError, match="video input"):
        encode_user_content_part(ContentPart("video", uri="https://x/video"), profile)
    empty = decode_content_blocks([{"type": "text", "text": ""}], profile)
    assert empty == [ContentPart.text_part("")]
    with pytest.raises(AnthropicMessagesError, match="input must be an object"):
        decode_tool_uses([{"type": "tool_use", "id": "call", "name": "tool", "input": "{"}])
