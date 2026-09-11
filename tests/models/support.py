"""Explicit provider response inputs shared by model adapter tests."""

from __future__ import annotations

from typing import Any


def terminal_response(
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
        "tools": [] if tools is None else tools,
        "usage": {
            "input_tokens": 3,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 2,
            "total_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 1},
        },
    }
