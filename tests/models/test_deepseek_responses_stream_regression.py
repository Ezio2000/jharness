from __future__ import annotations

import json
from time import time
from typing import Any

import httpx

from jharness.kernel import (
    Message,
    ModelDelta,
    ModelProviderToolCallDelta,
    ModelRequest,
    ProviderToolStatus,
    RunContext,
)
from jharness.models.deepseek import (
    deepseek_openai_responses_profile,
    deepseek_responses_web_search,
)
from jharness.models.openai import OpenAIResponsesModel


def _terminal_response(output: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "resp-deepseek-search",
        "object": "response",
        "created_at": 1,
        "completed_at": 2,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "model": "deepseek-v4-flash",
        "output": output,
        "previous_response_id": None,
        "store": False,
        "tools": [{"type": "web_search"}],
        "usage": {
            "input_tokens": 12,
            "output_tokens": 8,
            "total_tokens": 20,
        },
    }


async def test_deepseek_raw_sse_preserves_mixed_web_search_actions_in_order() -> None:
    actions = [
        {"type": "search", "query": "JHarness Responses protocol"},
        {"type": "open_page", "url": "https://example.invalid/jharness"},
        {"type": "find_in_page", "pattern": "Responses"},
    ]
    final_items = [
        {
            "id": "ws-search",
            "type": "web_search_call",
            "status": "completed",
            "action": actions[0],
        },
        {
            "id": "ws-open-page",
            "type": "web_search_call",
            "status": "failed",
            "action": actions[1],
            "error": {"code": "open_page_failed", "message": "page unavailable"},
        },
        {
            "id": "ws-find-in-page",
            "type": "web_search_call",
            "status": "failed",
            "action": actions[2],
            "error": {"code": "find_in_page_failed", "message": "pattern unavailable"},
        },
    ]
    events: list[dict[str, Any]] = [
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {
                "id": "resp-deepseek-search",
                "object": "response",
                "status": "in_progress",
            },
        }
    ]
    sequence_number = 1
    for output_index, (action, final_item) in enumerate(zip(actions, final_items, strict=True)):
        events.extend(
            [
                {
                    "type": "response.output_item.added",
                    "sequence_number": sequence_number,
                    "output_index": output_index,
                    "item": {
                        "id": final_item["id"],
                        "type": "web_search_call",
                        "status": "searching",
                    },
                },
                {
                    "type": "response.web_search_call.completed",
                    "sequence_number": sequence_number + 1,
                    "item_id": final_item["id"],
                    "output_index": output_index,
                    "action": action,
                },
                {
                    "type": "response.output_item.done",
                    "sequence_number": sequence_number + 2,
                    "output_index": output_index,
                    "item": final_item,
                },
            ]
        )
        sequence_number += 3
    events.append(
        {
            "type": "response.completed",
            "sequence_number": sequence_number,
            "response": _terminal_response(final_items),
        }
    )
    raw_sse = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.deepseek.invalid/responses"
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=raw_sse,
            request=request,
        )

    deltas: list[ModelDelta] = []

    async def emit_delta(delta: ModelDelta) -> None:
        deltas.append(delta)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await OpenAIResponsesModel(
            base_url="https://api.deepseek.invalid",
            api_key="test-only",
            model="deepseek-v4-flash",
            profile=deepseek_openai_responses_profile(effort="none"),
            client=client,
        ).invoke(
            ModelRequest(
                messages=(Message.user("Search, open, and inspect the result."),),
                provider_tools=(deepseek_responses_web_search(),),
            ),
            RunContext("run-deepseek-search-regression", time()),
            stream=True,
            emit_delta=emit_delta,
        )

    provider_deltas = [delta for delta in deltas if isinstance(delta, ModelProviderToolCallDelta)]
    assert [
        (delta.output_index, delta.id, delta.status, delta.event) for delta in provider_deltas
    ] == [
        (0, "ws-search", ProviderToolStatus.IN_PROGRESS, "response.output_item.added"),
        (
            0,
            "ws-search",
            ProviderToolStatus.IN_PROGRESS,
            "response.web_search_call.completed",
        ),
        (0, "ws-search", ProviderToolStatus.COMPLETED, "response.output_item.done"),
        (1, "ws-open-page", ProviderToolStatus.IN_PROGRESS, "response.output_item.added"),
        (
            1,
            "ws-open-page",
            ProviderToolStatus.IN_PROGRESS,
            "response.web_search_call.completed",
        ),
        (1, "ws-open-page", ProviderToolStatus.FAILED, "response.output_item.done"),
        (
            2,
            "ws-find-in-page",
            ProviderToolStatus.IN_PROGRESS,
            "response.output_item.added",
        ),
        (
            2,
            "ws-find-in-page",
            ProviderToolStatus.IN_PROGRESS,
            "response.web_search_call.completed",
        ),
        (2, "ws-find-in-page", ProviderToolStatus.FAILED, "response.output_item.done"),
    ]
    lifecycle_deltas = [
        delta for delta in provider_deltas if delta.event == "response.web_search_call.completed"
    ]
    assert [delta.data["action"] for delta in lifecycle_deltas] == actions

    calls = response.provider_tool_calls()
    assert [call.id for call in calls] == [
        "ws-search",
        "ws-open-page",
        "ws-find-in-page",
    ]
    assert [call.status for call in calls] == [
        ProviderToolStatus.COMPLETED,
        ProviderToolStatus.FAILED,
        ProviderToolStatus.FAILED,
    ]
    assert [call.arguments["type"] for call in calls] == [
        "search",
        "open_page",
        "find_in_page",
    ]
