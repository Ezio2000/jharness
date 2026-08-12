"""Explicit codecs for complete model responses and usage."""

from __future__ import annotations

from typing import Any

from jharness.kernel.models import ModelResponse, ModelUsage
from jharness.kernel.wire._helpers import (
    array,
    boolean,
    decode_document,
    json_object,
    object_fields,
    optional_integer,
    optional_string,
    thaw_object,
)
from jharness.kernel.wire.messages import (
    decode_model_output_item_value,
    encode_model_output_item,
)

__all__ = [
    "decode_model_response",
    "decode_model_usage",
    "encode_model_response",
    "encode_model_usage",
]

_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
)


def encode_model_usage(usage: ModelUsage) -> dict[str, int | None]:
    """Encode one complete portable usage value."""

    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
    }


def decode_model_usage(value: object) -> ModelUsage:
    """Decode one complete portable usage value."""

    return decode_document(value, "model usage", decode_model_usage_value)


def decode_model_usage_value(value: object) -> ModelUsage:
    fields = object_fields(value, "model usage", set(_USAGE_FIELDS))
    return ModelUsage(
        input_tokens=optional_integer(fields["input_tokens"], "input_tokens", minimum=0),
        output_tokens=optional_integer(fields["output_tokens"], "output_tokens", minimum=0),
        total_tokens=optional_integer(fields["total_tokens"], "total_tokens", minimum=0),
        reasoning_tokens=optional_integer(
            fields["reasoning_tokens"],
            "reasoning_tokens",
            minimum=0,
        ),
        cache_read_tokens=optional_integer(
            fields["cache_read_tokens"],
            "cache_read_tokens",
            minimum=0,
        ),
        cache_write_tokens=optional_integer(
            fields["cache_write_tokens"],
            "cache_write_tokens",
            minimum=0,
        ),
    )


def encode_model_response(response: ModelResponse) -> dict[str, Any]:
    """Encode the sole complete provider-neutral model result."""

    return {
        "output": [encode_model_output_item(item) for item in response.output],
        "finish_reason": response.finish_reason,
        "usage": None if response.usage is None else encode_model_usage(response.usage),
        "model_id": response.model_id,
        "response_id": response.response_id,
        "provider_turn_pending": response.provider_turn_pending,
        "metadata": thaw_object(response.metadata),
    }


def decode_model_response(value: object) -> ModelResponse:
    """Decode the sole complete provider-neutral model result."""

    return decode_document(value, "model response", decode_model_response_value)


def decode_model_response_value(value: object) -> ModelResponse:
    fields = object_fields(
        value,
        "model response",
        {
            "output",
            "finish_reason",
            "usage",
            "model_id",
            "response_id",
            "provider_turn_pending",
            "metadata",
        },
    )
    raw_usage = fields["usage"]
    return ModelResponse(
        output=tuple(
            decode_model_output_item_value(item)
            for item in array(fields["output"], "response output")
        ),
        finish_reason=optional_string(fields["finish_reason"], "finish_reason"),
        usage=None if raw_usage is None else decode_model_usage_value(raw_usage),
        model_id=optional_string(fields["model_id"], "model_id"),
        response_id=optional_string(fields["response_id"], "response_id"),
        provider_turn_pending=boolean(
            fields["provider_turn_pending"],
            "provider_turn_pending",
        ),
        metadata=json_object(fields["metadata"], "model response metadata"),
    )
