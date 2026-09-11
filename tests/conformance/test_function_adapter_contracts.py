"""Verify function adapter results against portable schemas and runtime semantics."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from conformance._schemas import SchemaSuite
from jharness.kernel import (
    Completed,
    ContentPart,
    DeltaSink,
    EventKind,
    FreeformToolCall,
    FreeformToolSpec,
    Message,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    RunContext,
    Runtime,
    RuntimeToolCall,
    RuntimeToolKind,
    SettledResult,
    StructuredToolCall,
    StructuredToolSpec,
    Suspended,
    Suspension,
    SuspensionSelector,
    ToolContext,
    ToolFailure,
    ToolResult,
    ToolSuccess,
    ToolWaiting,
    WaitingResult,
)
from jharness.kernel.diagnostics import build_trace, verify_trace
from jharness.kernel.wire import (
    decode_checkpoint,
    encode_checkpoint,
    encode_tool_result,
    encode_trace,
)
from jharness.repository import MemoryRunRepository
from jharness.toolkit import FreeformFunctionTool, FunctionTool, ToolRegistry, function_tool

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def schemas() -> SchemaSuite:
    return SchemaSuite(ROOT / "contracts" / "v0", ROOT / "conformance" / "case.schema.json")


class _Model:
    def __init__(self, call: RuntimeToolCall) -> None:
        self.call = call

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            runtime_tool_kinds=frozenset({RuntimeToolKind.STRUCTURED, RuntimeToolKind.FREEFORM})
        )

    async def invoke(
        self,
        request: ModelRequest,
        context: RunContext,
        *,
        stream: bool,
        emit_delta: DeltaSink | None,
    ) -> ModelResponse:
        del context, stream, emit_delta
        if any(message.role == "tool" for message in request.messages):
            return ModelResponse((ContentPart.text_part("done"),))
        return ModelResponse((self.call,))


@pytest.mark.parametrize("freeform", [False, True])
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("你好", ToolSuccess((ContentPart.text_part("你好"),), "你好")),
        ({"ok": True}, ToolSuccess((ContentPart.text_part('{"ok":true}'),), {"ok": True})),
        (None, ToolSuccess((ContentPart.text_part("null"),), None)),
    ],
)
async def test_normalized_success_uses_existing_result_and_checkpoint_schemas(
    value: object, expected: ToolSuccess, freeform: bool, schemas: SchemaSuite
) -> None:
    async def business(source: str = "") -> object:
        del source
        return value

    if freeform:
        tool = FreeformFunctionTool(FreeformToolSpec("business", "Business"), business)
        call = FreeformToolCall("call", "business", "raw")
    else:
        tool = FunctionTool(
            StructuredToolSpec("business", "Business", {"type": "object"}), business
        )
        call = StructuredToolCall("call", "business")
    repository = MemoryRunRepository()
    runtime = Runtime(model=_Model(call), tools=ToolRegistry((tool,)), repository=repository)
    invocation = runtime.start((Message.user("run business"),))
    events = tuple([event async for event in invocation.events()])
    checkpoint = await invocation.result()
    assert isinstance(checkpoint.snapshot.state, Completed)
    assert checkpoint.snapshot.metrics.tool_calls == 1
    assert checkpoint.snapshot.history[2].outcome == expected
    assert await repository.get_head(checkpoint.snapshot.context.run_id) == checkpoint

    wire = encode_checkpoint(checkpoint)
    schemas.validate("checkpoint.schema.json", wire)
    restored = decode_checkpoint(json.loads(json.dumps(wire)))
    assert restored == checkpoint
    schemas.validate("tool-result.schema.json", encode_tool_result(SettledResult(expected)))
    trace = build_trace(events, "start")
    schemas.validate("run-trace.schema.json", encode_trace(trace))
    verify_trace(trace)


async def test_native_waiting_result_resumes_from_wire_with_a_fresh_runtime(
    schemas: SchemaSuite,
) -> None:
    calls: list[str] = []

    @function_tool(
        name="business",
        description="Wait for an answer",
        input_schema={
            "type": "object",
            "properties": {"ticket": {"type": "string"}},
            "required": ["ticket"],
            "additionalProperties": False,
        },
        output_schema={"type": "string"},
        context_parameter="ctx",
    )
    async def business(ticket: str, *, ctx: ToolContext) -> ToolResult:
        calls.append(ticket)
        await ctx.emit_progress({"ticket": ticket})
        return WaitingResult(
            ToolWaiting((ContentPart.text_part("waiting"),), structured_content=ticket),
            Suspension("external", "business", ticket),
        )

    call = StructuredToolCall("call", "business", {"ticket": "ticket-1"})
    runtime = Runtime(model=_Model(call), tools=ToolRegistry((business,)))
    invocation = runtime.start((Message.user("wait"),))
    events = tuple([event async for event in invocation.events()])
    paused = await invocation.result()
    assert isinstance(paused.snapshot.state, Suspended)
    assert any(event.kind is EventKind.TOOL_PROGRESS for event in events)
    schemas.validate("checkpoint.schema.json", encode_checkpoint(paused))
    restored = decode_checkpoint(json.loads(json.dumps(encode_checkpoint(paused))))
    fresh = Runtime(model=_Model(call), tools=ToolRegistry((business,)))
    resumed = fresh.resume(
        restored,
        selector=SuspensionSelector(wait_id="ticket-1"),
        append_messages=(Message.external("confirmed"),),
    )
    resumed_events = tuple([event async for event in resumed.events()])
    completed = await resumed.result()
    assert isinstance(completed.snapshot.state, Completed)
    assert calls == ["ticket-1"]
    schemas.validate("checkpoint.schema.json", encode_checkpoint(completed))
    verify_trace(build_trace(events, "start"))
    verify_trace(build_trace(resumed_events, "resume"))


@pytest.mark.parametrize(
    ("arguments", "schema", "error_code"),
    [
        (
            {"value": "invalid"},
            {"type": "object", "properties": {"value": {"type": "integer"}}},
            "invalid_tool_call",
        ),
        ({}, {"type": "object"}, "tool_error"),
        ({"value": 1}, {"type": "object"}, "tool_error"),
    ],
)
async def test_adapter_failures_follow_existing_runtime_failure_semantics(
    arguments: Mapping[str, Any], schema: Mapping[str, Any], error_code: str, schemas: SchemaSuite
) -> None:
    async def business(value: int) -> int:
        raise ValueError(f"business rejected {value}")

    tool = FunctionTool(StructuredToolSpec("business", "Business", schema), business)
    call = StructuredToolCall("call", "business", arguments)
    checkpoint = (
        await Runtime(model=_Model(call), tools=ToolRegistry((tool,)))
        .start((Message.user("run"),))
        .result()
    )
    assert isinstance(checkpoint.snapshot.state, Completed)
    outcome = checkpoint.snapshot.history[2].outcome
    assert isinstance(outcome, ToolFailure)
    assert outcome.error.code == error_code
    schemas.validate("checkpoint.schema.json", encode_checkpoint(checkpoint))
