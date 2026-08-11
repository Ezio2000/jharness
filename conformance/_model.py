"""Fixture-driven implementation of the single model operation."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from conformance._values import boolean, integer, mapping, number, sequence, string
from jharness.kernel import (
    DeltaSink,
    ModelCapabilities,
    ModelContentDelta,
    ModelDelta,
    ModelError,
    ModelErrorInfo,
    ModelProviderToolCallDelta,
    ModelReasoningDelta,
    ModelRequest,
    ModelResponse,
    ModelToolCallDelta,
    ModelUsageDelta,
    ProtocolError,
    ProviderToolId,
    ProviderToolStatus,
    RunContext,
)
from jharness.kernel.wire import decode_model_response, decode_model_usage


class CaseModel:
    """Consume exactly one fixture step per logical model invocation."""

    def __init__(
        self,
        steps: Sequence[object],
        *,
        provider_tools: Sequence[ProviderToolId] = (),
    ) -> None:
        self._steps = tuple(mapping(step, "model step") for step in steps)
        self._cursor = 0
        self.streaming = any(
            sequence(step.get("deltas", ()), "model deltas") for step in self._steps
        )
        modalities = frozenset({"text", "image", "audio", "video", "file"})
        self.capabilities = ModelCapabilities(
            streaming=self.streaming,
            input_modalities=modalities,
            output_modalities=modalities,
            provider_tools=frozenset(provider_tools),
        )
        self.requests: list[ModelRequest] = []

    async def invoke(
        self,
        request: ModelRequest,
        context: RunContext,
        *,
        stream: bool,
        emit_delta: DeltaSink | None,
    ) -> ModelResponse:
        self.requests.append(request)
        step = self._next()
        deltas = sequence(step.get("deltas", ()), "model deltas")
        if deltas and (not stream or emit_delta is None):
            raise AssertionError("model fixture deltas require streaming")
        if emit_delta is not None:
            for raw_delta in deltas:
                await emit_delta(_delta(mapping(raw_delta, "model delta")))

        outcome = mapping(step["outcome"], "model outcome")
        kind = string(outcome["kind"], "model outcome kind")
        if kind == "block":
            await _block_forever()
        await _delay(step)
        if kind == "error":
            error = _model_error(mapping(outcome["error"], "model error"))
            if error.code == "model_protocol_error":
                raise ProtocolError(error.message)
            raise ModelError(error)
        if kind != "response":
            raise AssertionError(f"unsupported model outcome: {kind!r}")
        return decode_model_response(outcome["response"])

    def assert_consumed(self) -> None:
        if self._cursor != len(self._steps):
            remaining = len(self._steps) - self._cursor
            raise AssertionError(f"model fixture left {remaining} unconsumed step(s)")

    def _next(self) -> Mapping[str, Any]:
        if self._cursor >= len(self._steps):
            raise AssertionError("model fixture exhausted")
        step = self._steps[self._cursor]
        self._cursor += 1
        return step


async def _delay(step: Mapping[str, Any]) -> None:
    delay = number(step.get("delay_seconds", 0), "model delay_seconds")
    if delay:
        await asyncio.sleep(delay)


async def _block_forever() -> None:
    await asyncio.Future[None]()


def _model_error(value: Mapping[str, Any]) -> ModelErrorInfo:
    status_code = value["status_code"]
    return ModelErrorInfo(
        code=string(value["code"], "model error code"),
        message=string(value["message"], "model error message"),
        provider=_optional_string(value["provider"], "model error provider"),
        status_code=(
            None if status_code is None else integer(status_code, "model error status_code")
        ),
        retryable=boolean(value["retryable"], "model error retryable"),
        request_id=_optional_string(value["request_id"], "model error request_id"),
        metadata=mapping(value["metadata"], "model error metadata"),
    )


def _optional_string(value: object, label: str) -> str | None:
    return None if value is None else string(value, label)


def _delta(value: Mapping[str, Any]) -> ModelDelta:
    kind = string(value["kind"], "model delta kind")
    if kind == "content":
        return ModelContentDelta(
            output_index=integer(value["output_index"], "content delta output_index"),
            text_delta=string(value["text_delta"], "content delta text"),
            part_type=string(value["part_type"], "content delta part_type"),
            content_index=integer(value["content_index"], "content delta content_index"),
            data=mapping(value["data"], "content delta data"),
        )
    if kind == "tool_call":
        return ModelToolCallDelta(
            output_index=integer(value["output_index"], "tool call delta output_index"),
            arguments_delta=string(value["arguments_delta"], "tool call arguments delta"),
            id=_optional_string(value["id"], "tool call delta id"),
            name=_optional_string(value["name"], "tool call delta name"),
        )
    if kind == "reasoning":
        return ModelReasoningDelta(
            output_index=integer(value["output_index"], "reasoning delta output_index"),
            text_delta=string(value["text_delta"], "reasoning delta text"),
            content_index=integer(value["content_index"], "reasoning delta content_index"),
        )
    if kind == "provider_tool_call":
        tool = mapping(value["tool"], "provider tool delta tool")
        raw_status = value["status"]
        return ModelProviderToolCallDelta(
            output_index=integer(
                value["output_index"],
                "provider tool delta output_index",
            ),
            id=string(value["id"], "provider tool delta id"),
            tool=ProviderToolId(
                namespace=string(tool["namespace"], "provider tool delta namespace"),
                type=string(tool["type"], "provider tool delta type"),
            ),
            status=(
                None
                if raw_status is None
                else ProviderToolStatus(string(raw_status, "provider tool delta status"))
            ),
            event=_optional_string(value["event"], "provider tool delta event"),
            data=mapping(value["data"], "provider tool delta data"),
        )
    if kind == "usage":
        return ModelUsageDelta(decode_model_usage(value["usage"]))
    raise ValueError(f"unsupported model delta: {kind!r}")
