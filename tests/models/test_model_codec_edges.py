from __future__ import annotations

import base64
from dataclasses import replace
from typing import Any

import pytest

from jharness.kernel import (
    ArtifactRef,
    ContentPart,
    FreeformToolSpec,
    Message,
    RuntimeToolKind,
    StructuredToolCall,
    StructuredToolSpec,
    ToolChoice,
)
from jharness.models.anthropic import AnthropicMessagesError, AnthropicMessagesProfile
from jharness.models.anthropic.messages.messages import (
    decode_content_blocks,
    encode_tool_result_content,
    encode_user_content_part,
)
from jharness.models.anthropic.messages.messages import (
    encode_message as encode_anthropic_message,
)
from jharness.models.anthropic.messages.messages import (
    encode_message_content as encode_anthropic_content,
)
from jharness.models.anthropic.messages.messages import (
    encode_messages as encode_anthropic_messages,
)
from jharness.models.anthropic.messages.tools import (
    decode_tool_uses,
    encode_assistant_tool_uses,
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
from jharness.models.openai.chat.messages import (
    encode_message_content as encode_openai_content,
)
from jharness.models.openai.chat.tools import (
    decode_tool_calls,
    encode_assistant_tool_calls,
)
from jharness.models.openai.chat.tools import (
    encode_tool_choice as encode_openai_choice,
)
from jharness.models.openai.chat.tools import (
    encode_tools as encode_openai_tools,
)


def test_openai_chat_content_edges_and_incremental_native_parts() -> None:
    default = OpenAIChatProfile()
    profile = OpenAIChatProfile(
        capabilities=replace(
            default.capabilities,
            input_modalities=frozenset({"text", "image", "audio", "file"}),
        ),
    )
    call = StructuredToolCall("call", "lookup")
    assert "content" not in encode_chat_message(Message.assistant((call,)), profile)
    assert encode_openai_content((), "user", profile) == ""
    assert encode_openai_content((ContentPart.text_part("ok"),), "tool", profile) == "ok"
    assert decode_message_content(None) == []
    empty = decode_message_content("")
    assert len(empty) == 1 and empty[0].type == "text" and empty[0].text == ""
    with pytest.raises(OpenAIChatError, match="string or null"):
        decode_message_content([{"type": "text", "text": ""}])
    assert decode_message_refusal(None) == []
    assert (
        encode_content_part(ContentPart("file", uri="data:text/plain;base64,eA=="), profile)[
            "file"
        ]["filename"]
        == "file"
    )
    assert encode_assistant_tool_calls((call,))[0]["function"]["arguments"] == "{}"

    bare_refusal = ContentPart("refusal", text="no")
    assert encode_chat_message(Message.assistant((bare_refusal,)), profile)["content"] == [
        {"type": "refusal", "refusal": "no"}
    ]
    native_refusal = ContentPart(
        "refusal",
        text="no",
        data={"openai": {"type": "refusal", "refusal": "native", "unknown": True}},
    )
    with pytest.raises(OpenAIChatError, match="unsupported field"):
        encode_chat_message(Message.assistant((native_refusal,)), profile)


@pytest.mark.parametrize(
    "part,pattern",
    [
        (ContentPart("image"), "requires a uri"),
        (ContentPart("file"), "requires a uri"),
        (ContentPart("audio", uri="https://x/audio"), "media_type or a data URI"),
    ],
)
def test_openai_chat_content_rejects_missing_or_unsupported_sources(
    part: ContentPart, pattern: str
) -> None:
    default = OpenAIChatProfile()
    profile = OpenAIChatProfile(
        capabilities=replace(
            default.capabilities,
            input_modalities=frozenset({"text", "image", "audio", "file"}),
        ),
    )
    with pytest.raises(OpenAIChatError, match=pattern):
        encode_content_part(part, profile)


def test_openai_chat_content_rejects_disabled_capabilities_and_bad_native_data() -> None:
    default = OpenAIChatProfile()
    disabled = OpenAIChatProfile(
        capabilities=replace(
            default.capabilities,
            input_modalities=frozenset({"text"}),
        ),
    )
    for part, pattern in (
        (ContentPart("image", uri="https://x/image"), "image input"),
        (ContentPart("file", uri="https://x/file"), "file input"),
    ):
        with pytest.raises(OpenAIChatError, match=pattern):
            encode_content_part(part, disabled)

    profile = OpenAIChatProfile()
    with pytest.raises(OpenAIChatError, match="only support text"):
        encode_openai_content((ContentPart("image", uri="https://x/image"),), "system", profile)
    for part, pattern in (
        (ContentPart("audio", uri="x"), "assistant history"),
        (ContentPart("refusal", text="no", data={"openai": "bad"}), "must be an object"),
        (
            ContentPart(
                "refusal",
                text="no",
                data={"openai": {"type": "other", "refusal": "no"}},
            ),
            "unsupported OpenAI-native",
        ),
    ):
        with pytest.raises(OpenAIChatError, match=pattern):
            encode_chat_message(Message.assistant((part,)), profile)
    empty_refusal = ContentPart(
        "refusal",
        text="ignored",
        data={"openai": {"type": "refusal", "refusal": ""}},
    )
    assert encode_chat_message(Message.assistant((empty_refusal,)), profile)["content"] == [
        {"type": "refusal", "refusal": ""}
    ]
    with pytest.raises(OpenAIChatError, match="text parts only or one refusal"):
        encode_chat_message(
            Message.assistant((ContentPart.text_part("answer"), ContentPart("refusal", text="no"))),
            profile,
        )


@pytest.mark.parametrize("value", ([1], [{"type": "text", "text": ""}], 1))
def test_openai_chat_content_decoder_rejects_non_string_content(value: object) -> None:
    with pytest.raises(OpenAIChatError, match="string or null"):
        decode_message_content(value)


def test_openai_chat_tool_codec_edges() -> None:
    spec = StructuredToolSpec("lookup", "lookup", {"type": "object"})
    profile = OpenAIChatProfile()
    assert encode_openai_tools((), profile) == []
    disabled_capabilities = replace(
        profile.capabilities,
        runtime_tool_kinds=frozenset(),
        tool_choice_types=frozenset({"auto", "none"}),
    )
    with pytest.raises(OpenAIChatError, match="does not support structured tools"):
        encode_openai_tools(
            (spec,),
            OpenAIChatProfile(capabilities=disabled_capabilities),
        )
    with pytest.raises(OpenAIChatError, match="parameters must be a JSON object"):
        encode_openai_tools(
            (StructuredToolSpec("boolean", "boolean", True),),
            profile,
        )
    with pytest.raises(OpenAIChatError, match="function tool names must match"):
        encode_openai_tools(
            (StructuredToolSpec("not valid", "invalid", {"type": "object"}),),
            profile,
        )
    custom_profile = OpenAIChatProfile(
        capabilities=replace(
            profile.capabilities,
            runtime_tool_kinds=frozenset({RuntimeToolKind.STRUCTURED, RuntimeToolKind.FREEFORM}),
        )
    )
    assert (
        encode_openai_tools((FreeformToolSpec("not valid", "custom"),), custom_profile)[0][
            "custom"
        ]["name"]
        == "not valid"
    )
    assert encode_openai_choice(ToolChoice(), tools_by_name={}, profile=profile) is None
    auto_only = OpenAIChatProfile(
        capabilities=replace(
            profile.capabilities,
            tool_choice_types=frozenset({"auto"}),
        ),
    )
    assert (
        encode_openai_choice(ToolChoice(), tools_by_name={"lookup": spec}, profile=auto_only)
        == "auto"
    )
    with pytest.raises(OpenAIChatError, match="does not support tool_choice"):
        encode_openai_choice(ToolChoice("none"), tools_by_name={"lookup": spec}, profile=auto_only)
    assert (
        encode_openai_choice(
            ToolChoice("required"), tools_by_name={"lookup": spec}, profile=profile
        )
        == "required"
    )
    assert decode_tool_calls(None) == []


@pytest.mark.parametrize(
    "value,pattern",
    [
        (1, "must be an array"),
        ([1], "must be an object"),
        ([{"type": "other"}], "unsupported"),
        (
            [{"id": "call", "function": {"name": "tool", "arguments": ""}}],
            "requires non-empty type",
        ),
        ([{"id": "call", "type": "function", "function": 1}], "function must be an object"),
        (
            [{"id": 1, "type": "function", "function": {"name": "tool", "arguments": ""}}],
            "id must be a string",
        ),
        (
            [{"id": "", "type": "function", "function": {"name": "tool", "arguments": ""}}],
            "id must not be empty",
        ),
        (
            [{"id": "call", "type": "function", "function": {"name": "", "arguments": ""}}],
            "name must not be empty",
        ),
        (
            [{"id": "call", "type": "function", "function": {"name": "tool", "arguments": 1}}],
            "arguments must be a string",
        ),
    ],
)
def test_openai_chat_tool_decoder_rejects_invalid_values(value: object, pattern: str) -> None:
    with pytest.raises(OpenAIChatError, match=pattern):
        decode_tool_calls(value)


def test_openai_chat_tool_decoder_preserves_nonobject_json_as_raw_input() -> None:
    call = decode_tool_calls(
        [{"id": "call", "type": "function", "function": {"name": "tool", "arguments": "[]"}}]
    )[0]
    assert isinstance(call, StructuredToolCall)
    assert call.arguments is None
    assert call.raw_input == "[]"


def test_anthropic_messages_message_grouping_rejects_mid_conversation_system() -> None:
    profile = AnthropicMessagesProfile()
    system, messages = encode_anthropic_messages((Message.system("policy"),), profile)
    assert system == [{"type": "text", "text": "policy"}]
    assert messages == []
    with pytest.raises(AnthropicMessagesError, match="unsupported Anthropic message role"):
        encode_anthropic_message(Message.system("policy"), profile)
    with pytest.raises(AnthropicMessagesError, match="system content before messages"):
        encode_anthropic_messages(
            (Message.user("one"), Message.system("instruction")),
            profile,
        )


def test_anthropic_messages_content_shapes_and_native_metadata() -> None:
    profile = AnthropicMessagesProfile()
    call = StructuredToolCall("call", "lookup")
    assert encode_anthropic_content((), "user", profile) == ""
    assert encode_anthropic_content((ContentPart.text_part("a"),), "assistant", profile) == "a"
    assert encode_anthropic_message(Message.assistant((call,)), profile)["content"] == [
        {"type": "tool_use", "id": "call", "name": "lookup", "input": {}}
    ]
    assert encode_tool_result_content((), profile) == ""
    assert encode_tool_result_content((ContentPart.text_part("x"),), profile) == "x"
    assert (
        encode_user_content_part(
            ContentPart.artifact_part(
                ArtifactRef("file", media_type="application/pdf", name="report.pdf")
            ),
            profile,
        )["title"]
        == "report.pdf"
    )
    assert encode_user_content_part(ContentPart("image", uri="https://x/image"), profile)[
        "source"
    ] == {"type": "url", "url": "https://x/image"}

    thinking = ContentPart(
        "thinking",
        text="reason",
        metadata={"anthropic": {"signature": "sig"}},
    )
    redacted = ContentPart(
        "redacted_thinking",
        metadata={"anthropic": {"data": "secret"}},
    )
    encoded = encode_anthropic_message(Message.assistant((thinking, redacted)), profile)
    assert encoded["content"] == [
        {"type": "thinking", "thinking": "reason", "signature": "sig"},
        {"type": "redacted_thinking", "data": "secret"},
    ]
    with pytest.raises(AnthropicMessagesError, match="require anthropic metadata"):
        encode_anthropic_message(Message.assistant((ContentPart("redacted_thinking"),)), profile)
    with pytest.raises(AnthropicMessagesError, match="metadata signature must be a string"):
        encode_anthropic_message(
            Message.assistant(
                (
                    ContentPart(
                        "thinking",
                        text="reason",
                        metadata={"anthropic": {"signature": 1}},
                    ),
                )
            ),
            profile,
        )


@pytest.mark.parametrize(
    "block,pattern",
    [
        (1, "must be an object"),
        ({}, "non-empty type"),
        ({"type": "other"}, "unsupported"),
        ({"type": "thinking", "thinking": ""}, "require a signature"),
        ({"type": "thinking", "thinking": "ok", "signature": 1}, "require a signature"),
    ],
)
def test_anthropic_messages_content_decoder_rejects_invalid_blocks(
    block: object, pattern: str
) -> None:
    with pytest.raises(AnthropicMessagesError, match=pattern):
        decode_content_blocks([block], AnthropicMessagesProfile())


def test_anthropic_messages_content_decoder_preserves_empty_standard_strings() -> None:
    parts = decode_content_blocks(
        [
            {"type": "text", "text": ""},
            {"type": "thinking", "thinking": "", "signature": ""},
            {"type": "redacted_thinking", "data": ""},
        ],
        AnthropicMessagesProfile(),
    )
    assert all(isinstance(part, ContentPart) for part in parts)
    content_parts = tuple(part for part in parts if isinstance(part, ContentPart))
    assert [(part.type, part.text) for part in content_parts] == [
        ("text", ""),
        ("thinking", ""),
        ("redacted_thinking", None),
    ]


def test_anthropic_messages_native_blocks_validate_role_shape_and_capabilities() -> None:
    profile = AnthropicMessagesProfile()
    text_only = replace(
        profile.capabilities,
        input_modalities=frozenset({"text"}),
    )
    without_image = replace(
        profile.capabilities,
        input_modalities=frozenset({"text", "file"}),
    )
    native_text = ContentPart(
        "opaque",
        data={"anthropic": {"type": "text", "text": "policy"}},
    )
    system, _ = encode_anthropic_messages((Message("system", (native_text,)),), profile)
    assert system == [{"type": "text", "text": "policy"}]
    cases = (
        (
            ContentPart("opaque", data={"anthropic": {}}),
            profile,
            "non-empty type",
        ),
        (
            ContentPart("opaque", data={"anthropic": {"type": "thinking", "thinking": "x"}}),
            profile,
            "not allowed for user",
        ),
        (
            ContentPart("opaque", data={"anthropic": {"type": "image", "source": {}}}),
            AnthropicMessagesProfile(capabilities=without_image),
            "does not support image",
        ),
        (
            ContentPart("opaque", data={"anthropic": {"type": "document", "source": {}}}),
            AnthropicMessagesProfile(capabilities=text_only),
            "does not support file",
        ),
        (
            ContentPart("opaque", data={"anthropic": {"type": "image", "source": 1}}),
            profile,
            "source must be an object",
        ),
    )
    for part, selected_profile, pattern in cases:
        with pytest.raises(AnthropicMessagesError, match=pattern):
            encode_user_content_part(part, selected_profile)


@pytest.mark.parametrize(
    "part,pattern",
    [
        (ContentPart("file"), "requires a uri or artifact"),
        (
            ContentPart("file", uri="data:application/json;base64,e30="),
            "requires an artifact reference",
        ),
        (ContentPart("file", uri="https://x/file", media_type="text/plain"), "must be PDFs"),
        (ContentPart("file", uri="data:text/plain,abc"), "must use base64"),
        (ContentPart("file", uri="data:text/plain;base64,"), "requires base64 data"),
        (ContentPart("file", uri="data:text/plain;base64,***"), "invalid base64"),
        (
            ContentPart(
                "file",
                uri="data:text/plain;base64," + base64.b64encode(b"\xff").decode(),
            ),
            "decode as UTF-8",
        ),
    ],
)
def test_anthropic_messages_media_rejects_invalid_sources(part: ContentPart, pattern: str) -> None:
    with pytest.raises(AnthropicMessagesError, match=pattern):
        encode_user_content_part(part, AnthropicMessagesProfile())


def test_anthropic_messages_tool_codec_edges() -> None:
    spec = StructuredToolSpec("lookup", "lookup", {"type": "object"})
    profile = AnthropicMessagesProfile()
    call = StructuredToolCall("call", "lookup", {"x": 1})
    assert encode_anthropic_tools((), (), profile) == []
    disabled_capabilities = replace(
        profile.capabilities,
        runtime_tool_kinds=frozenset(),
        tool_choice_types=frozenset({"auto", "none"}),
    )
    with pytest.raises(AnthropicMessagesError, match="does not support structured runtime tools"):
        encode_anthropic_tools(
            (spec,),
            (),
            AnthropicMessagesProfile(capabilities=disabled_capabilities),
        )
    assert (
        encode_anthropic_choice(
            ToolChoice(),
            runtime_tool_names=set(),
            provider_tools=(),
            may_return_runtime_tool_calls=False,
            profile=profile,
        )
        is None
    )
    auto_only = AnthropicMessagesProfile(
        capabilities=replace(profile.capabilities, tool_choice_types=frozenset({"auto"})),
    )
    assert encode_anthropic_choice(
        ToolChoice(),
        runtime_tool_names={"lookup"},
        provider_tools=(),
        may_return_runtime_tool_calls=True,
        profile=auto_only,
    ) == {"type": "auto", "disable_parallel_tool_use": False}
    with pytest.raises(AnthropicMessagesError, match="does not support tool_choice"):
        encode_anthropic_choice(
            ToolChoice("none"),
            runtime_tool_names={"lookup"},
            provider_tools=(),
            may_return_runtime_tool_calls=True,
            profile=auto_only,
        )
    assert encode_anthropic_choice(
        ToolChoice("none"),
        runtime_tool_names={"lookup"},
        provider_tools=(),
        may_return_runtime_tool_calls=True,
        profile=profile,
    ) == {"type": "none"}
    assert encode_assistant_tool_uses((call,))[0]["input"] == {"x": 1}
    assert decode_tool_uses(
        [{"type": "tool_use", "id": "call", "name": "lookup", "input": {}}]
    ) == [StructuredToolCall("call", "lookup")]
    assert decode_tool_uses(
        [{"type": "tool_use", "id": "call", "name": "lookup", "input": {"x": 1}}]
    ) == [call]


@pytest.mark.parametrize(
    "block,pattern",
    [
        ({"type": "tool_use", "id": 1, "name": "tool", "input": {}}, "id must be a string"),
        ({"type": "tool_use", "id": "", "name": "tool", "input": {}}, "id must not be empty"),
        ({"type": "tool_use", "id": "call", "name": "", "input": {}}, "name must not be empty"),
        ({"type": "tool_use", "id": "call", "name": "tool", "input": 1}, "must be an object"),
        ({"type": "tool_use", "id": "call", "name": "tool", "input": "{"}, "must be an object"),
        ({"type": "tool_use", "id": "call", "name": "tool", "input": "[]"}, "must be an object"),
    ],
)
def test_anthropic_messages_tool_decoder_rejects_invalid_values(
    block: dict[str, Any], pattern: str
) -> None:
    with pytest.raises(AnthropicMessagesError, match=pattern):
        decode_tool_uses([block])
