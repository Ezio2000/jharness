from __future__ import annotations

from typing import Any, cast

import pytest

from jharness.kernel import (
    ContentPart,
    Message,
    ModelContentDelta,
    ModelProviderToolCallDelta,
    ModelRequest,
    ProviderToolCall,
    ProviderToolId,
    ProviderToolSpec,
    ProviderToolStatus,
    ToolChoice,
)
from jharness.models.anthropic import (
    AnthropicMessagesCodec,
    AnthropicMessagesProfile,
    anthropic_messages_profile,
)
from jharness.models.anthropic.messages.errors import AnthropicMessagesError
from jharness.models.anthropic.messages.stream import AnthropicMessagesStreamDecoder

_WEB_SEARCH = ProviderToolId("anthropic.messages", "web_search")
_SERVER_USE: dict[str, Any] = {
    "type": "server_tool_use",
    "id": "server-1",
    "name": "web_search",
    "input": {"query": "JHarness"},
    "caller": {"type": "direct"},
}
_SERVER_RESULT: dict[str, Any] = {
    "type": "web_search_tool_result",
    "tool_use_id": "server-1",
    "content": [
        {
            "type": "web_search_result",
            "url": "https://example.com/jharness",
            "title": "JHarness",
            "encrypted_content": "opaque",
        }
    ],
}


def _profile() -> AnthropicMessagesProfile:
    return anthropic_messages_profile()


def _codec() -> AnthropicMessagesCodec:
    return AnthropicMessagesCodec(model="claude-opus-4-6", profile=_profile())


def _response_content(*blocks: dict[str, Any], stop_reason: str = "end_turn") -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "id": "message-1",
        "model": "claude-opus-4-6",
        "stop_reason": stop_reason,
        "content": list(blocks),
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def test_anthropic_messages_encodes_web_search_and_exact_provider_choice() -> None:
    profile = _profile()
    spec = ProviderToolSpec(
        _WEB_SEARCH,
        {
            "allowed_domains": ["example.com"],
            "max_uses": 2,
        },
    )

    payload = AnthropicMessagesCodec(
        model="claude-opus-4-6",
        profile=profile,
    ).encode_request(
        ModelRequest(
            messages=(Message.user("search"),),
            provider_tools=(spec,),
            tool_choice=ToolChoice(
                type="provider",
                provider_tool=_WEB_SEARCH,
                allow_parallel_runtime_tool_calls=False,
            ),
        )
    )

    assert profile.capabilities.provider_tools == frozenset({_WEB_SEARCH})
    assert payload["tools"] == [
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "allowed_domains": ["example.com"],
            "max_uses": 2,
        }
    ]
    assert payload["tool_choice"] == {"type": "tool", "name": "web_search"}


def _declaration(configuration: dict[str, object]) -> dict[str, object]:
    request = ModelRequest(
        messages=(Message.user("search"),),
        provider_tools=(ProviderToolSpec(_WEB_SEARCH, configuration),),
    )
    return cast(dict[str, object], _codec().encode_request(request)["tools"][0])


def test_anthropic_messages_web_search_limits_response_inclusion_by_variant() -> None:
    with pytest.raises(AnthropicMessagesError, match="unsupported web_search_20250305"):
        _declaration({"response_inclusion": "all"})
    assert (
        _declaration({"variant": "web_search_20260318", "response_inclusion": "full"})[
            "response_inclusion"
        ]
        == "full"
    )


@pytest.mark.parametrize(
    "configuration, match",
    (
        (
            {"allowed_domains": ["example.com"], "blocked_domains": ["blocked.example"]},
            "mutually exclusive",
        ),
        ({"max_uses": 0}, "positive integer"),
        ({"allowed_callers": ["direct"]}, "unsupported web_search configuration field"),
        ({"defer_loading": True}, "unsupported web_search configuration field"),
        ({"user_location": {"type": "exact"}}, "approximate"),
        ({"user_location": {"type": "approximate"}}, "requires city"),
        ({"user_location": {"type": "approximate", "country": "USA"}}, "two-letter ISO"),
        ({"variant": "web_search_20260318", "response_inclusion": "all"}, "full"),
    ),
)
def test_anthropic_messages_web_search_validates_official_configuration(
    configuration: dict[str, object], match: str
) -> None:
    with pytest.raises(AnthropicMessagesError, match=match):
        _declaration(configuration)


def test_anthropic_messages_web_search_accepts_direct_configuration_only() -> None:
    declaration = _declaration(
        {
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
            "strict": False,
        }
    )
    assert declaration["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert declaration["strict"] is False


def test_anthropic_messages_web_search_rejects_unknown_error_code() -> None:
    result = {
        "type": "web_search_tool_result",
        "tool_use_id": "server-1",
        "content": {
            "type": "web_search_tool_result_error",
            "error_code": "unofficial",
        },
    }

    with pytest.raises(AnthropicMessagesError, match="error_code"):
        _codec().decode_response(_response_content(_SERVER_USE, result, stop_reason="tool_use"))


def test_anthropic_messages_server_tool_use_requires_object_input() -> None:
    use = {
        "type": "server_tool_use",
        "id": "server-1",
        "name": "web_search",
        "input": "{}",
    }

    with pytest.raises(AnthropicMessagesError, match="input must be an object"):
        _codec().decode_response(_response_content(use, stop_reason="tool_use"))


def test_anthropic_messages_rejects_code_execution_web_search_caller() -> None:
    use = {
        **_SERVER_USE,
        "caller": {"type": "code_execution_20260521", "tool_id": "exec-1"},
    }

    with pytest.raises(AnthropicMessagesError, match="direct caller has unsupported field"):
        _codec().decode_response(_response_content(use, stop_reason="tool_use"))


@pytest.mark.parametrize(
    "block, match",
    (
        ({**_SERVER_USE, "vendor_extra": True}, "unsupported field: vendor_extra"),
        ({**_SERVER_USE, "caller": {"type": "direct", "tool_id": "bad"}}, "unsupported field"),
        ({**_SERVER_RESULT, "vendor_extra": True}, "unsupported field: vendor_extra"),
    ),
)
def test_anthropic_messages_web_search_rejects_unknown_or_invalid_extras(
    block: dict[str, object], match: str
) -> None:
    content = (block,) if block.get("type") == "server_tool_use" else (_SERVER_USE, block)
    with pytest.raises(AnthropicMessagesError, match=match):
        _codec().decode_response(_response_content(*content, stop_reason="tool_use"))


_INVALID_WEB_SEARCH_CONTENT: tuple[tuple[object, str], ...] = (
    (
        [{"type": "web_search_result", "title": "title", "url": "https://example.com"}],
        "encrypted_content",
    ),
    (
        [{**_SERVER_RESULT["content"][0], "vendor_extra": True}],
        "unsupported field: vendor_extra",
    ),
    (
        {"type": "web_search_tool_result_error", "error_code": "invalid_input"},
        "error_code",
    ),
    (dict[str, object](), "must be web_search_tool_result_error"),
)


@pytest.mark.parametrize(
    "content, match",
    _INVALID_WEB_SEARCH_CONTENT,
)
def test_anthropic_messages_web_search_validates_result_content(
    content: object, match: str
) -> None:
    result = {**_SERVER_RESULT, "content": content}
    with pytest.raises(AnthropicMessagesError, match=match):
        _codec().decode_response(_response_content(_SERVER_USE, result, stop_reason="tool_use"))


def test_anthropic_messages_server_tool_pair_terminal_and_provider_stop_not_pending() -> None:
    response = _codec().decode_response(
        _response_content(_SERVER_USE, _SERVER_RESULT, stop_reason="tool_use")
    )

    assert len(response.output) == 1
    call = cast(ProviderToolCall, response.output[0])
    assert call.id == "server-1"
    assert call.tool == _WEB_SEARCH
    assert call.status is ProviderToolStatus.COMPLETED
    assert call.arguments == {"query": "JHarness"}
    assert call.error is None
    assert call.output[0].data == {
        "anthropic": {
            "type": "web_search_tool_result",
            "content": _SERVER_RESULT["content"],
        }
    }
    assert response.provider_turn_pending is False


def test_anthropic_messages_server_tool_result_error_is_terminal_failure() -> None:
    result_error = {
        "type": "web_search_tool_result",
        "tool_use_id": "server-1",
        "content": {
            "type": "web_search_tool_result_error",
            "error_code": "unavailable",
        },
    }

    response = _codec().decode_response(
        _response_content(_SERVER_USE, result_error, stop_reason="tool_use")
    )
    call = response.provider_tool_calls()[0]

    assert call.status is ProviderToolStatus.FAILED
    assert call.error is not None
    assert call.error.code == "web_search.unavailable"
    assert call.error.message == "unavailable"
    assert response.provider_turn_pending is False


def test_anthropic_messages_server_tool_history_replays_exact_native_blocks() -> None:
    codec = _codec()
    response = codec.decode_response(_response_content(_SERVER_USE, _SERVER_RESULT))

    payload = codec.encode_request(
        ModelRequest(
            messages=(Message.user("search"), response.to_assistant_message()),
            provider_tools=(ProviderToolSpec(_WEB_SEARCH),),
        )
    )

    messages = cast(list[dict[str, Any]], payload["messages"])
    assert messages[1] == {
        "role": "assistant",
        "content": [_SERVER_USE, _SERVER_RESULT],
    }


def test_anthropic_messages_server_tool_stream_pairs_result_at_use_position() -> None:
    decoder = AnthropicMessagesStreamDecoder(_profile())
    decoder.apply_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "type": "message",
                "role": "assistant",
                "id": "message-1",
                "model": "claude-opus-4-6",
                "content": [],
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
    )
    _, started = decoder.apply_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 4,
            "content_block": {
                "type": "server_tool_use",
                "id": "server-1",
                "name": "web_search",
                "input": {},
            },
        },
    )
    _, input_delta = decoder.apply_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 4,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '{"query":"JHarness"}',
            },
        },
    )
    decoder.apply_event(
        "content_block_stop",
        {"type": "content_block_stop", "index": 4},
    )
    _, result = decoder.apply_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 9,
            "content_block": _SERVER_RESULT,
        },
    )
    decoder.apply_event(
        "content_block_stop",
        {"type": "content_block_stop", "index": 9},
    )
    _, text = decoder.apply_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 10,
            "content_block": {"type": "text", "text": "done"},
        },
    )
    decoder.apply_event(
        "content_block_stop",
        {"type": "content_block_stop", "index": 10},
    )
    decoder.apply_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 1},
        },
    )
    decoder.apply_event("message_stop", {"type": "message_stop"})

    start_delta = cast(ModelProviderToolCallDelta, started[0])
    next_delta = cast(ModelProviderToolCallDelta, input_delta[0])
    result_delta = cast(ModelProviderToolCallDelta, result[0])
    text_delta = cast(ModelContentDelta, text[0])
    assert (start_delta.output_index, start_delta.status) == (
        0,
        ProviderToolStatus.IN_PROGRESS,
    )
    assert next_delta.output_index == 0
    assert (result_delta.output_index, result_delta.status) == (
        0,
        ProviderToolStatus.COMPLETED,
    )
    assert text_delta.output_index == 1

    response = decoder.completed_response()
    assert [type(item) for item in response.output] == [ProviderToolCall, ContentPart]
    call = cast(ProviderToolCall, response.output[0])
    assert call.status is ProviderToolStatus.COMPLETED
    assert call.arguments == {"query": "JHarness"}
    assert cast(ContentPart, response.output[1]).text == "done"
    assert response.provider_turn_pending is False


def test_anthropic_messages_unmatched_streamed_server_use_keeps_provider_turn_pending() -> None:
    decoder = AnthropicMessagesStreamDecoder(_profile())
    decoder.apply_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "type": "message",
                "role": "assistant",
                "id": "message-2",
                "model": "claude-opus-4-6",
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
                "type": "server_tool_use",
                "id": "server-1",
                "name": "web_search",
                "input": {"query": "JHarness"},
            },
        },
    )
    decoder.apply_event(
        "content_block_stop",
        {"type": "content_block_stop", "index": 0},
    )
    decoder.apply_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 1},
        },
    )
    decoder.apply_event("message_stop", {"type": "message_stop"})

    response = decoder.completed_response()
    call = response.provider_tool_calls()[0]
    assert call.status is ProviderToolStatus.IN_PROGRESS
    assert response.provider_turn_pending is True
