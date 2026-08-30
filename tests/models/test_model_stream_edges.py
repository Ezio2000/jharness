from __future__ import annotations

from typing import Any

import pytest

from jharness.kernel import (
    ContentPart,
    ModelContentDelta,
    ModelRuntimeToolCallDelta,
    ModelUsageDelta,
    StructuredToolCall,
)
from jharness.models.anthropic import AnthropicMessagesError, AnthropicMessagesProfile
from jharness.models.anthropic.messages.stream import AnthropicMessagesStreamDecoder
from jharness.models.openai import OpenAIChatError, OpenAIChatProfile
from jharness.models.openai.chat.stream import OpenAIChatStreamDecoder


def openai_choice(
    delta: object,
    *,
    finish_reason: object = None,
    index: object = 0,
    **metadata: object,
) -> dict[str, Any]:
    return {
        "id": "response",
        "model": "model",
        "object": "chat.completion.chunk",
        "created": 1,
        **metadata,
        "choices": [{"index": index, "delta": delta, "finish_reason": finish_reason}],
    }


def openai_tool_call(index: int, call_id: str, name: str) -> dict[str, Any]:
    return {
        "index": index,
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


def test_openai_chat_stream_completion_guards_usage_and_metadata() -> None:
    decoder = OpenAIChatStreamDecoder(OpenAIChatProfile())
    with pytest.raises(OpenAIChatError, match="without a choice"):
        decoder.completed_response()
    usage = decoder.apply_chunk(
        {
            "id": "response",
            "model": "model",
            "object": "chat.completion.chunk",
            "created": 1,
            "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
        }
    )
    assert isinstance(usage[0], ModelUsageDelta)
    decoder.apply_chunk(openai_choice({}, finish_reason="stop"))
    empty = decoder.completed_response()
    assert empty.output == ()
    assert empty.metadata["openai_chat"] == {"content_null": True}

    unfinished = OpenAIChatStreamDecoder(OpenAIChatProfile())
    unfinished.apply_chunk(openai_choice({"content": "x"}))
    with pytest.raises(OpenAIChatError, match="before finish_reason"):
        unfinished.completed_response()

    complete = OpenAIChatStreamDecoder(OpenAIChatProfile())
    deltas = complete.apply_chunk(
        openai_choice(
            {"role": "assistant", "refusal": "no"},
            finish_reason="stop",
            id="response",
            model="model",
            object="chat.completion.chunk",
            created=7,
        )
    )
    assert any(isinstance(delta, ModelContentDelta) for delta in deltas)
    response = complete.completed_response()
    assert response.visible_parts()[0].type == "refusal"
    assert response.metadata["object"] == "chat.completion.chunk"
    assert response.metadata["created"] == 7


def test_openai_chat_stream_content_filter_preserves_chunk_obfuscation_and_null_history() -> None:
    decoder = OpenAIChatStreamDecoder(OpenAIChatProfile())
    decoder.apply_chunk(
        {
            "id": "response",
            "model": "model",
            "object": "chat.completion.chunk",
            "created": 1,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "content_filter"}],
            "service_tier": "priority",
            "system_fingerprint": "fp_123",
            "obfuscation": "first",
        }
    )
    response = decoder.completed_response()

    assert response.output == ()
    assert response.metadata["obfuscation"] == ["first"]
    assert response.metadata["openai_chat"] == {"content_null": True}


@pytest.mark.parametrize(
    "chunk,pattern",
    [
        ({"unused": None}, "stream chunk has unsupported fields"),
        ({"choices": 1}, "choices must be an array"),
        ({"choices": list[object]()}, "empty choices require usage"),
        (
            {"choices": [{"unused": None}, {"unused": None}]},
            "exactly one choice",
        ),
        ({"choices": [1]}, "choice must be an object"),
        (openai_choice({"unused": None}, index=True), "index must be an integer"),
        (openai_choice({"unused": None}, index=1), "index must be 0"),
        (openai_choice(1), "delta must be an object"),
        (openai_choice({}, finish_reason=""), "finish_reason has an unsupported"),
        (openai_choice({"role": "user"}), "role must be 'assistant'"),
        (openai_choice({"content": 1}), "content delta must be"),
        (openai_choice({"reasoning_content": 1}), "do not support reasoning_content"),
        (openai_choice({"refusal": 1}), "refusal delta must be"),
        (openai_choice({"tool_calls": 1}), "tool_calls must be an array"),
        (openai_choice({"tool_calls": [1]}), "tool call must be an object"),
        (
            openai_choice({"tool_calls": [{"type": "other"}]}),
            "unsupported chat completion stream tool call type",
        ),
        (
            openai_choice({"tool_calls": [{"function": 1}]}),
            "tool function must be an object",
        ),
        (
            openai_choice({"tool_calls": [{"id": 1}]}),
            "expected string or null",
        ),
        (
            openai_choice({"tool_calls": [{"index": True, "id": "call"}]}),
            "tool call index must be an integer",
        ),
        (
            openai_choice({"tool_calls": [{"index": -1, "id": "call"}]}),
            "tool call index must be >= 0",
        ),
        (openai_choice({"unused": None}, id=1), "id must be a string or null"),
        (openai_choice({"unused": None}, id=""), "id must not be empty"),
        (
            openai_choice({"unused": None}, created=True),
            "created must be an integer or null",
        ),
    ],
)
def test_openai_chat_stream_rejects_invalid_chunks(chunk: dict[str, Any], pattern: str) -> None:
    with pytest.raises(OpenAIChatError, match=pattern):
        OpenAIChatStreamDecoder(OpenAIChatProfile()).apply_chunk(
            {
                "id": "response",
                "model": "model",
                "object": "chat.completion.chunk",
                "created": 1,
                **chunk,
            }
        )


def test_openai_chat_stream_rejects_metadata_changes_and_post_finish_choices() -> None:
    changed = OpenAIChatStreamDecoder(OpenAIChatProfile())
    changed.apply_chunk(openai_choice({"content": "a"}, id="one"))
    with pytest.raises(OpenAIChatError, match="id changed"):
        changed.apply_chunk(openai_choice({"content": "b"}, id="two"))

    finished = OpenAIChatStreamDecoder(OpenAIChatProfile())
    finished.apply_chunk(openai_choice({"content": "a"}, finish_reason="stop"))
    with pytest.raises(OpenAIChatError, match="after finish_reason"):
        finished.apply_chunk(openai_choice({"content": "b"}))

    empty_call = OpenAIChatStreamDecoder(OpenAIChatProfile())
    with pytest.raises(OpenAIChatError, match="tool call has unsupported fields"):
        empty_call.apply_chunk(openai_choice({"tool_calls": [{"unused": None}]}))


@pytest.mark.parametrize(
    ("wire_field", "part_type"),
    (("content", "text"), ("refusal", "refusal")),
)
def test_openai_chat_stream_reserves_content_before_tool_calls(
    wire_field: str,
    part_type: str,
) -> None:
    decoder = OpenAIChatStreamDecoder(OpenAIChatProfile())
    content_deltas = decoder.apply_chunk(openai_choice({wire_field: "answer"}))
    tool_deltas = decoder.apply_chunk(
        openai_choice(
            {"tool_calls": [openai_tool_call(0, "call-1", "search")]},
            finish_reason="tool_calls",
        )
    )

    assert [(type(delta), getattr(delta, "output_index", None)) for delta in content_deltas] == [
        (ModelContentDelta, 0)
    ]
    assert [(type(delta), getattr(delta, "output_index", None)) for delta in tool_deltas] == [
        (ModelRuntimeToolCallDelta, 1)
    ]
    assert decoder.completed_response().output == (
        ContentPart(part_type, "answer"),
        StructuredToolCall("call-1", "search", {}),
    )


def test_openai_chat_stream_reserves_same_chunk_prefix_for_multiple_tools() -> None:
    decoder = OpenAIChatStreamDecoder(OpenAIChatProfile())
    deltas = decoder.apply_chunk(
        openai_choice(
            {
                "content": "answer",
                "refusal": "no",
                "tool_calls": [
                    openai_tool_call(0, "call-1", "first"),
                    openai_tool_call(1, "call-2", "second"),
                ],
            },
            finish_reason="tool_calls",
        )
    )

    assert [(type(delta), getattr(delta, "output_index", None)) for delta in deltas] == [
        (ModelContentDelta, 0),
        (ModelContentDelta, 1),
        (ModelRuntimeToolCallDelta, 2),
        (ModelRuntimeToolCallDelta, 3),
    ]
    assert decoder.completed_response().output == (
        ContentPart("text", "answer"),
        ContentPart("refusal", "no"),
        StructuredToolCall("call-1", "first", {}),
        StructuredToolCall("call-2", "second", {}),
    )


def test_openai_chat_empty_tool_fragment_does_not_freeze_output_offset() -> None:
    decoder = OpenAIChatStreamDecoder(OpenAIChatProfile())
    empty_tool_call: dict[str, Any] = {"index": 0, "type": "function", "function": {}}
    assert decoder.apply_chunk(openai_choice({"tool_calls": [empty_tool_call]})) == []

    content_deltas = decoder.apply_chunk(openai_choice({"content": "answer"}))
    tool_deltas = decoder.apply_chunk(
        openai_choice(
            {"tool_calls": [openai_tool_call(0, "call-1", "search")]},
            finish_reason="tool_calls",
        )
    )

    assert [(type(delta), getattr(delta, "output_index", None)) for delta in content_deltas] == [
        (ModelContentDelta, 0)
    ]
    assert [(type(delta), getattr(delta, "output_index", None)) for delta in tool_deltas] == [
        (ModelRuntimeToolCallDelta, 1)
    ]
    assert decoder.completed_response().output == (
        ContentPart("text", "answer"),
        StructuredToolCall("call-1", "search", {}),
    )


def anthropic_started(
    *, profile: AnthropicMessagesProfile | None = None
) -> AnthropicMessagesStreamDecoder:
    decoder = AnthropicMessagesStreamDecoder(
        AnthropicMessagesProfile() if profile is None else profile
    )
    decoder.apply_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "type": "message",
                "role": "assistant",
                "content": [],
                "id": "message",
                "model": "model",
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )
    return decoder


def anthropic_start_block(
    decoder: AnthropicMessagesStreamDecoder,
    block: dict[str, Any],
    *,
    index: object = 0,
) -> None:
    decoder.apply_event(
        "content_block_start",
        {"type": "content_block_start", "index": index, "content_block": block},
    )


def anthropic_delta(
    decoder: AnthropicMessagesStreamDecoder,
    delta: dict[str, Any],
    *,
    index: int = 0,
) -> None:
    decoder.apply_event(
        "content_block_delta",
        {"type": "content_block_delta", "index": index, "delta": delta},
    )


def anthropic_stop(decoder: AnthropicMessagesStreamDecoder, *, index: int = 0) -> None:
    decoder.apply_event("content_block_stop", {"type": "content_block_stop", "index": index})


def test_anthropic_messages_stream_event_envelope_and_start_guards() -> None:
    decoder = AnthropicMessagesStreamDecoder(AnthropicMessagesProfile())
    assert decoder.apply_event("ping", {"type": "ping"}) == (False, [])
    with pytest.raises(AnthropicMessagesError, match="before message_stop"):
        decoder.completed_response()
    for event_name, value, pattern in (
        (None, {"unused": None}, "requires a type"),
        ("ping", {"type": "message_start"}, "name must match"),
        ("error", {"type": "error"}, "stream error event"),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"unused": None},
            },
            "requires message_start",
        ),
        ("message_stop", {"type": "message_stop"}, "requires message_start"),
    ):
        with pytest.raises(AnthropicMessagesError, match=pattern):
            AnthropicMessagesStreamDecoder(AnthropicMessagesProfile()).apply_event(
                event_name, value
            )
    with pytest.raises(AnthropicMessagesError, match="unsupported Anthropic stream event"):
        decoder.apply_event("other", {"type": "other"})

    started = anthropic_started()
    with pytest.raises(AnthropicMessagesError, match="more than once"):
        started.apply_event(
            "message_start",
            {
                "type": "message_start",
                "message": {"type": "message", "role": "assistant", "content": []},
            },
        )


@pytest.mark.parametrize(
    "message,pattern",
    [
        (1, "message must be an object"),
        ({"type": "other", "role": "assistant", "content": []}, "type='message'"),
        ({"type": "message", "role": "user", "content": []}, "role='assistant'"),
        ({"type": "message", "role": "assistant", "content": ""}, "content must be an array"),
        (
            {"type": "message", "role": "assistant", "content": [{"unused": None}]},
            "content must be empty",
        ),
        ({"type": "message", "role": "assistant", "content": [], "id": ""}, "must not be empty"),
    ],
)
def test_anthropic_messages_stream_rejects_invalid_message_start(
    message: object, pattern: str
) -> None:
    with pytest.raises(AnthropicMessagesError, match=pattern):
        AnthropicMessagesStreamDecoder(AnthropicMessagesProfile()).apply_event(
            "message_start", {"type": "message_start", "message": message}
        )


@pytest.mark.parametrize(
    "block,index,pattern",
    [
        ({"type": "text", "text": "x"}, None, "requires field: index"),
        ({"type": "text", "text": "x"}, True, "index must be an integer"),
        ({"type": "text", "text": "x"}, -1, "index must be >= 0"),
        ({"unused": None}, 0, "requires non-empty type"),
        ({"type": "other"}, 0, "unsupported Anthropic stream content block"),
        ({"type": "text", "text": 1}, 0, "text block requires text"),
        ({"type": "thinking", "thinking": 1}, 0, "thinking block requires thinking"),
        ({"type": "tool_use", "id": "", "name": "tool"}, 0, "id must not be empty"),
        ({"type": "tool_use", "id": "call", "name": ""}, 0, "name must not be empty"),
        (
            {"type": "tool_use", "id": "call", "name": "tool", "input": 1},
            0,
            "input must be an object",
        ),
    ],
)
def test_anthropic_messages_stream_rejects_invalid_block_starts(
    block: dict[str, Any], index: object, pattern: str
) -> None:
    decoder = anthropic_started()
    value: dict[str, Any] = {"type": "content_block_start", "content_block": block}
    if index is not None:
        value["index"] = index
    with pytest.raises(AnthropicMessagesError, match=pattern):
        decoder.apply_event("content_block_start", value)


def test_anthropic_messages_stream_rejects_duplicate_and_empty_blocks() -> None:
    duplicate = anthropic_started()
    anthropic_start_block(duplicate, {"type": "text", "text": "x"})
    with pytest.raises(AnthropicMessagesError, match="started more than once"):
        anthropic_start_block(duplicate, {"type": "text", "text": "y"})

    empty_text = anthropic_started()
    anthropic_start_block(empty_text, {"type": "text", "text": ""})
    anthropic_stop(empty_text)

    unsigned_thinking = anthropic_started()
    anthropic_start_block(unsigned_thinking, {"type": "thinking", "thinking": ""})
    with pytest.raises(AnthropicMessagesError, match="requires a signature"):
        anthropic_stop(unsigned_thinking)


def test_anthropic_messages_stream_rejects_events_for_a_closed_block() -> None:
    decoder = anthropic_started()
    anthropic_start_block(decoder, {"type": "text", "text": "x"})
    anthropic_stop(decoder)

    with pytest.raises(AnthropicMessagesError, match="requires an open index"):
        anthropic_delta(decoder, {"type": "text_delta", "text": "y"})
    with pytest.raises(AnthropicMessagesError, match="requires an open index"):
        anthropic_stop(decoder)


def test_anthropic_messages_stream_interleaves_blocks_and_keeps_monotonic_tool_order() -> None:
    decoder = anthropic_started()
    anthropic_start_block(
        decoder,
        {"type": "tool_use", "id": "call-1", "name": "first", "input": {}},
        index=4,
    )
    anthropic_start_block(decoder, {"type": "text", "text": "a"}, index=1)
    anthropic_delta(decoder, {"type": "input_json_delta", "partial_json": '{"x":1}'}, index=4)
    anthropic_delta(decoder, {"type": "text_delta", "text": "b"}, index=1)
    anthropic_stop(decoder, index=1)
    anthropic_stop(decoder, index=4)
    anthropic_start_block(
        decoder,
        {"type": "tool_use", "id": "call-2", "name": "second", "input": {"y": 2}},
        index=9,
    )
    anthropic_stop(decoder, index=9)
    decoder.apply_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 0},
        },
    )
    decoder.apply_event("message_stop", {"type": "message_stop"})

    response = decoder.completed_response()
    assert response.visible_parts()[0].text == "ab"
    calls = response.runtime_tool_calls()
    assert all(isinstance(call, StructuredToolCall) for call in calls)
    structured_calls = [call for call in calls if isinstance(call, StructuredToolCall)]
    assert [(call.id, call.name, call.arguments) for call in structured_calls] == [
        ("call-1", "first", {"x": 1}),
        ("call-2", "second", {"y": 2}),
    ]


@pytest.mark.parametrize(
    "block,delta,pattern",
    [
        ({"type": "text", "text": "x"}, {"type": "other"}, "unsupported"),
        (
            {"type": "text", "text": "x"},
            {"type": "thinking_delta", "thinking": "x"},
            "does not match text",
        ),
        ({"type": "text", "text": "x"}, {"type": "text_delta", "text": 1}, "requires text"),
        (
            {"type": "thinking", "thinking": "x"},
            {"type": "thinking_delta", "thinking": 1},
            "requires thinking",
        ),
        (
            {"type": "thinking", "thinking": "x"},
            {"type": "signature_delta", "signature": 1},
            "requires signature",
        ),
        (
            {"type": "tool_use", "id": "call", "name": "tool", "input": {}},
            {"type": "input_json_delta", "partial_json": 1},
            "requires partial_json",
        ),
    ],
)
def test_anthropic_messages_stream_rejects_invalid_block_deltas(
    block: dict[str, Any], delta: dict[str, Any], pattern: str
) -> None:
    decoder = anthropic_started()
    anthropic_start_block(decoder, block)
    with pytest.raises(AnthropicMessagesError, match=pattern):
        anthropic_delta(decoder, delta)


def test_anthropic_messages_stream_terminal_guards_and_usage() -> None:
    no_open = anthropic_started()
    with pytest.raises(AnthropicMessagesError, match="requires an open index"):
        anthropic_delta(no_open, {"type": "text_delta", "text": "x"})
    with pytest.raises(AnthropicMessagesError, match="requires an open index"):
        anthropic_stop(no_open)
    with pytest.raises(AnthropicMessagesError, match="requires a terminal message_delta"):
        no_open.apply_event("message_stop", {"type": "message_stop"})

    open_block = anthropic_started()
    anthropic_start_block(open_block, {"type": "text", "text": "x"})
    with pytest.raises(AnthropicMessagesError, match="all content blocks to stop"):
        open_block.apply_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 0},
            },
        )
    with pytest.raises(AnthropicMessagesError, match="requires a terminal message_delta"):
        open_block.apply_event("message_stop", {"type": "message_stop"})

    no_data = anthropic_started()
    no_data.apply_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 0},
        },
    )
    assert no_data.apply_event("message_stop", {"type": "message_stop"}) == (True, [])
    assert no_data.completed_response().output == ()
    with pytest.raises(AnthropicMessagesError, match="after message_stop"):
        no_data.apply_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 0},
            },
        )

    with_usage = anthropic_started()
    anthropic_start_block(with_usage, {"type": "text", "text": "x"})
    anthropic_stop(with_usage)
    _, usage = with_usage.apply_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 1},
        },
    )
    assert len(usage) == 1
    assert isinstance(usage[0], ModelUsageDelta)
    assert usage[0].usage.output_tokens == 1
    with_usage.apply_event("message_stop", {"type": "message_stop"})
    with_usage.completed_response()
    with pytest.raises(AnthropicMessagesError, match="after message_stop"):
        with_usage.apply_event("ping", {"type": "ping"})
