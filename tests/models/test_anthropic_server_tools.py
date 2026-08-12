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
from jharness.models.anthropic import AnthropicCodec, AnthropicProfile
from jharness.models.anthropic.errors import AnthropicError
from jharness.models.anthropic.messages_api.server_tools import anthropic_web_search_codec
from jharness.models.anthropic.messages_api.stream import AnthropicStreamDecoder
from jharness.models.deepseek import deepseek_anthropic_profile

_WEB_SEARCH = ProviderToolId("deepseek.anthropic", "web_search")
_SERVER_USE = {
    "type": "server_tool_use",
    "id": "server-1",
    "name": "web_search",
    "input": {"query": "JHarness"},
    "caller": {"type": "direct"},
}
_SERVER_RESULT = {
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


def _profile() -> AnthropicProfile:
    return deepseek_anthropic_profile(thinking=False)


def _codec() -> AnthropicCodec:
    return AnthropicCodec(model="deepseek-v4-flash", profile=_profile())


def _response_content(*blocks: dict[str, Any], stop_reason: str = "end_turn") -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "id": "message-1",
        "model": "deepseek-v4-flash",
        "stop_reason": stop_reason,
        "content": list(blocks),
    }


def test_deepseek_anthropic_encodes_web_search_and_exact_provider_choice() -> None:
    profile = _profile()
    spec = ProviderToolSpec(
        _WEB_SEARCH,
        {
            "allowed_domains": ["example.com"],
            "max_uses": 2,
        },
    )

    payload = AnthropicCodec(
        model="deepseek-v4-flash",
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
    assert profile.server_tools.tools == frozenset({_WEB_SEARCH})
    assert payload["tools"] == [
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "allowed_domains": ["example.com"],
            "max_uses": 2,
        }
    ]
    assert payload["tool_choice"] == {"type": "tool", "name": "web_search"}


def test_anthropic_web_search_limits_response_inclusion_by_variant() -> None:
    codec = anthropic_web_search_codec(
        _WEB_SEARCH,
        variants=frozenset({"web_search_20250305", "web_search_20260318"}),
    )

    with pytest.raises(AnthropicError, match="unsupported web_search_20250305"):
        codec.encode_declaration(ProviderToolSpec(_WEB_SEARCH, {"response_inclusion": "all"}))

    declaration = codec.encode_declaration(
        ProviderToolSpec(
            _WEB_SEARCH,
            {"variant": "web_search_20260318", "response_inclusion": "all"},
        )
    )
    assert declaration["response_inclusion"] == "all"


def test_anthropic_server_tool_pair_is_terminal_and_provider_only_stop_is_not_pending() -> None:
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


def test_anthropic_server_tool_result_error_is_terminal_failure() -> None:
    result_error = {
        "type": "web_search_tool_result",
        "tool_use_id": "server-1",
        "content": {
            "type": "web_search_tool_result_error",
            "error_code": "unavailable",
            "error_message": "search temporarily unavailable",
        },
    }

    response = _codec().decode_response(
        _response_content(_SERVER_USE, result_error, stop_reason="tool_use")
    )
    call = response.provider_tool_calls()[0]

    assert call.status is ProviderToolStatus.FAILED
    assert call.error is not None
    assert call.error.code == "web_search.unavailable"
    assert call.error.message == "search temporarily unavailable"
    assert response.provider_turn_pending is False


def test_anthropic_server_tool_history_replays_exact_native_blocks() -> None:
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


def test_anthropic_server_tool_stream_pairs_result_at_use_position() -> None:
    decoder = AnthropicStreamDecoder(_profile())
    decoder.apply_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "type": "message",
                "role": "assistant",
                "id": "message-1",
                "model": "deepseek-v4-flash",
                "content": [],
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


def test_anthropic_unmatched_streamed_server_use_keeps_provider_turn_pending() -> None:
    decoder = AnthropicStreamDecoder(_profile())
    decoder.apply_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "type": "message",
                "role": "assistant",
                "content": [],
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
        },
    )
    decoder.apply_event("message_stop", {"type": "message_stop"})

    response = decoder.completed_response()
    call = response.provider_tool_calls()[0]
    assert call.status is ProviderToolStatus.IN_PROGRESS
    assert response.provider_turn_pending is True
