from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from jharness.kernel import (
    ArtifactRef,
    ContentPart,
    FreeformToolCall,
    FreeformToolSpec,
    RunContext,
    SettledResult,
    StructuredToolCall,
    StructuredToolSpec,
    Suspension,
    ToolAccepted,
    ToolContext,
    ToolError,
    ToolExecution,
    ToolFailure,
    ToolResult,
    ToolSuccess,
    ToolWaiting,
    WaitingResult,
)
from jharness.toolkit import (
    FreeformFunctionTool,
    FunctionTool,
    RetryingTool,
    ToolFunction,
    ToolRegistry,
    freeform_tool,
    function_tool,
)


async def _ignore_progress(value: Mapping[str, Any]) -> None:
    del value


def _context(run_id: str = "run") -> ToolContext:
    return ToolContext(RunContext(run_id, 1.0), _ignore_progress, lambda: False)


def _spec(**options: Any) -> StructuredToolSpec:
    return StructuredToolSpec("business", "A business function", {"type": "object"}, **options)


async def test_named_arguments_defaults_and_mutable_json_are_detached_from_call() -> None:
    @function_tool(
        name="business",
        description="A business function",
        input_schema={
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}},
                "options": {"type": "object"},
                "suffix": {"type": "string", "default": "schema-default"},
            },
            "required": ["items", "options"],
            "additionalProperties": False,
        },
    )
    async def business(
        items: list[str], options: dict[str, str], *, suffix: str = "python-default"
    ) -> dict[str, object]:
        items.append(suffix)
        options["updated"] = "yes"
        return {"items": items, "options": options}

    call = StructuredToolCall("call", "business", {"items": ["first"], "options": {}})
    catalog = await ToolRegistry((business,)).open_catalog()
    binding = catalog.bind(call)
    first = await binding.invoke(_context())
    second = await binding.invoke(_context())
    assert first == second
    assert call.arguments == {"items": ["first"], "options": {}}
    assert first.outcome.structured_content == {
        "items": ["first", "python-default"],
        "options": {"updated": "yes"},
    }
    with pytest.raises(FrozenInstanceError):
        business.context_parameter = "ctx"  # type: ignore[misc]


@pytest.mark.parametrize("arguments", [{}, {"value": 1, "extra": 2}])
async def test_signature_binding_rejects_missing_or_unknown_arguments_before_business(
    arguments: dict[str, int],
) -> None:
    entered = False

    async def business(value: int) -> int:
        nonlocal entered
        entered = True
        return value

    tool = FunctionTool(_spec(), business)
    with pytest.raises(ToolError, match="function signature"):
        await tool.invoke(StructuredToolCall("call", "business", arguments), _context())
    assert not entered


async def test_schema_validation_precedes_business_and_does_not_coerce_annotations() -> None:
    entered = False

    @function_tool(
        name="business",
        description="A business function",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )
    async def business(value: int) -> str:
        nonlocal entered
        entered = True
        return type(value).__name__

    catalog = await ToolRegistry((business,)).open_catalog()
    with pytest.raises(ToolError, match="input_schema"):
        catalog.bind(StructuredToolCall("bad", "business", {"value": 1}))
    assert not entered
    result = await catalog.bind(StructuredToolCall("good", "business", {"value": "1"})).invoke(
        _context()
    )
    assert result.outcome.structured_content == "str"


async def test_context_is_explicit_keyword_only_and_local_to_each_invocation() -> None:
    progress: list[tuple[str, object]] = []

    def context(run_id: str) -> ToolContext:
        async def emit(value: Mapping[str, Any]) -> None:
            progress.append((run_id, value["value"]))

        return ToolContext(RunContext(run_id, 1.0), emit, lambda: run_id == "cancelled")

    @function_tool(
        name="business",
        description="Report progress",
        input_schema={"type": "object"},
        context_parameter="ctx",
    )
    async def business(value: str, *, ctx: ToolContext) -> dict[str, object]:
        await ctx.emit_progress({"value": value})
        await asyncio.sleep(0)
        return {"run": ctx.run.run_id, "cancelled": ctx.cancel_requested}

    results = await asyncio.gather(
        business.invoke(StructuredToolCall("a", "business", {"value": "A"}), context("one")),
        business.invoke(StructuredToolCall("b", "business", {"value": "B"}), context("cancelled")),
    )
    assert [result.outcome.structured_content for result in results] == [
        {"run": "one", "cancelled": False},
        {"run": "cancelled", "cancelled": True},
    ]
    assert progress == [("one", "A"), ("cancelled", "B")]


async def test_context_name_is_an_ordinary_business_parameter_without_opt_in() -> None:
    async def business(context: str) -> str:
        return context

    tool = FunctionTool(_spec(), business)
    result = await tool.invoke(
        StructuredToolCall("call", "business", {"context": "from-model"}), _context()
    )
    assert result.outcome.structured_content == "from-model"


async def test_context_annotation_does_not_enable_injection() -> None:
    async def business(*, ctx: ToolContext) -> str:
        return ctx.run.run_id

    with pytest.raises(ToolError, match="function signature"):
        await FunctionTool(_spec(), business).invoke(
            StructuredToolCall("call", "business"), _context()
        )


async def test_model_cannot_supply_context_even_through_an_open_schema() -> None:
    entered = False

    async def business(*, ctx: ToolContext) -> str:
        nonlocal entered
        entered = True
        return ctx.run.run_id

    tool = FunctionTool(_spec(), business, context_parameter="ctx")
    catalog = await ToolRegistry((tool,)).open_catalog()
    with pytest.raises(ToolError, match="must not supply the context"):
        await catalog.bind(
            StructuredToolCall("call", "business", {"ctx": {"run_id": "spoof"}})
        ).invoke(_context())
    assert not entered


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": {"ctx": {"type": "string"}}},
        {"type": "object", "required": ["ctx"]},
    ],
)
def test_context_must_not_be_declared_as_model_input(schema: dict[str, object]) -> None:
    async def business(*, ctx: ToolContext) -> str:
        return ctx.run.run_id

    with pytest.raises(ValueError, match="must not declare the context"):
        FunctionTool(
            StructuredToolSpec("business", "Business", schema), business, context_parameter="ctx"
        )


@pytest.mark.parametrize("name", ["", "missing", "value", 1])
def test_context_configuration_is_checked_at_construction(name: object) -> None:
    async def business(value: str, *, ctx: ToolContext) -> str:
        return value + ctx.run.run_id

    with pytest.raises((TypeError, ValueError), match="context_parameter"):
        FunctionTool(_spec(), business, context_parameter=cast(Any, name))
    with pytest.raises((TypeError, ValueError), match="context_parameter"):
        FreeformFunctionTool(
            FreeformToolSpec("business", "Business"), business, context_parameter=cast(Any, name)
        )


def test_unsupported_function_parameter_kinds_are_rejected() -> None:
    async def positional(value: str, /) -> str:
        return value

    async def variadic(*values: str) -> str:
        return "".join(values)

    async def arbitrary(**values: str) -> dict[str, str]:
        return values

    for function in (positional, variadic, arbitrary):
        with pytest.raises(TypeError, match="named arguments"):
            FunctionTool(_spec(), function)
        with pytest.raises(TypeError, match="named arguments"):
            FreeformFunctionTool(FreeformToolSpec("business", "Business"), function)


async def test_bound_async_methods_use_their_business_signature() -> None:
    class Business:
        async def lookup(self, name: str) -> str:
            return f"hello {name}"

    tool = FunctionTool(_spec(), Business().lookup)
    result = await tool.invoke(
        StructuredToolCall("call", "business", {"name": "world"}), _context()
    )
    assert result.outcome.structured_content == "hello world"


async def test_call_objects_are_never_forwarded_to_business_functions() -> None:
    entered = False

    async def business(call: StructuredToolCall, context: ToolContext) -> ToolResult:
        nonlocal entered
        entered = True
        return SettledResult(ToolSuccess((ContentPart.text_part(call.id + context.run.run_id),)))

    tool = FunctionTool(_spec(), business)
    with pytest.raises(ToolError, match="function signature"):
        await tool.invoke(StructuredToolCall("call", "business", {"order_id": "A1001"}), _context())
    assert not entered
    with pytest.raises(ValueError, match="exactly one input parameter"):
        FreeformFunctionTool(FreeformToolSpec("business", "Business"), business)


@pytest.mark.parametrize(
    "text", ["", '  {"raw":true}\n\t文本\x00\r\n', "*** Begin Patch\n*** End Patch"]
)
async def test_freeform_preserves_exact_text_with_explicit_context(text: str) -> None:
    @freeform_tool(name="business", description="Raw text", context_parameter="ctx")
    async def business(*, source: str, ctx: ToolContext) -> dict[str, str]:
        return {"text": source, "run": ctx.run.run_id}

    catalog = await ToolRegistry((business,)).open_catalog()
    result = await catalog.bind(FreeformToolCall("call", "business", text)).invoke(_context())
    assert result.outcome.structured_content == {"text": text, "run": "run"}


def test_freeform_requires_exactly_one_business_parameter() -> None:
    async def no_input() -> str:
        return "none"

    async def extra_input(text: str, extra: str = "default") -> str:
        return text + extra

    for function in (no_input, extra_input):
        with pytest.raises(ValueError, match="exactly one input parameter"):
            FreeformFunctionTool(FreeformToolSpec("business", "Business"), function)


@pytest.mark.parametrize(
    ("value", "text"),
    [
        ("hello", "hello"),
        ("", ""),
        ("你好\nworld", "你好\nworld"),
        (None, "null"),
        (False, "false"),
        (0, "0"),
        (1.5, "1.5"),
        ({"z": [True, None], "a": "你好"}, '{"a":"你好","z":[true,null]}'),
        ([1, "two"], '[1,"two"]'),
    ],
)
@pytest.mark.parametrize("freeform", [False, True])
async def test_json_return_values_have_text_and_structured_content(
    value: object, text: str, freeform: bool
) -> None:
    async def business(source: str = "") -> object:
        del source
        return value

    if freeform:
        result = await FreeformFunctionTool(
            FreeformToolSpec("business", "Business"), business
        ).invoke(FreeformToolCall("call", "business", "raw"), _context())
    else:
        result = await FunctionTool(_spec(), business).invoke(
            StructuredToolCall("call", "business"), _context()
        )
    assert isinstance(result, SettledResult)
    assert isinstance(result.outcome, ToolSuccess)
    assert result.outcome.parts == (ContentPart.text_part(text),)
    assert result.outcome.structured_content == value


@pytest.mark.parametrize(
    "native",
    [
        SettledResult(
            ToolSuccess(
                (
                    ContentPart.artifact_part(
                        ArtifactRef("host:report", media_type="application/pdf")
                    ),
                ),
                {"report": "ready"},
            )
        ),
        SettledResult(ToolFailure.from_error("business_error", "not available")),
        SettledResult(ToolAccepted((ContentPart.text_part("accepted"),), "job-1")),
        WaitingResult(
            ToolWaiting(
                (ContentPart.text_part("waiting"),), structured_content={"ticket": "job-1"}
            ),
            Suspension("external", "business", "job-1"),
        ),
    ],
)
async def test_native_results_are_passed_through_without_reconstruction(native: ToolResult) -> None:
    async def business(text: str = "") -> ToolResult:
        del text
        return native

    structured = FunctionTool(_spec(), business)
    freeform = FreeformFunctionTool(FreeformToolSpec("raw", "Raw"), business)
    catalog = await ToolRegistry((structured, freeform)).open_catalog()
    assert await catalog.bind(StructuredToolCall("s", "business")).invoke(_context()) is native
    assert await catalog.bind(FreeformToolCall("f", "raw", "input")).invoke(_context()) is native


@pytest.mark.parametrize(
    "value", [b"bytes", {1, 2}, object(), float("nan"), float("inf"), {1: "bad-key"}]
)
async def test_unsupported_return_values_are_not_stringified(value: object) -> None:
    async def business() -> object:
        return value

    with pytest.raises(ToolError, match="JSON-compatible or ToolResult"):
        await FunctionTool(_spec(), business).invoke(
            StructuredToolCall("call", "business"), _context()
        )


async def test_cyclic_results_and_unwrapped_outcomes_are_rejected() -> None:
    cycle: list[object] = []
    cycle.append(cycle)
    values: tuple[object, ...] = (cycle, ToolSuccess((ContentPart.text_part("unwrapped"),)))
    for value in values:

        async def business(result: object = value) -> object:
            return result

        with pytest.raises(ToolError, match="JSON-compatible or ToolResult"):
            await FunctionTool(_spec(), business).invoke(
                StructuredToolCall("call", "business"), _context()
            )


@pytest.mark.parametrize("freeform", [False, True])
async def test_output_schema_checks_normalized_and_native_results(freeform: bool) -> None:
    value: object = "valid"

    async def business(text: str = "") -> object:
        del text
        return value

    if freeform:
        tool = FreeformFunctionTool(
            FreeformToolSpec("business", "Business", output_schema={"type": "string"}), business
        )
        call = FreeformToolCall("call", "business", "raw")
    else:
        tool = FunctionTool(_spec(output_schema={"type": "string"}), business)
        call = StructuredToolCall("call", "business")
    catalog = await ToolRegistry((tool,)).open_catalog()
    binding = catalog.bind(call)
    assert (await binding.invoke(_context())).outcome.structured_content == "valid"
    value = {"wrong": "type"}
    with pytest.raises(ToolError, match="output_schema"):
        await binding.invoke(_context())
    value = SettledResult(ToolFailure.from_error("missing", "not found"))
    assert await binding.invoke(_context()) is value
    value = WaitingResult(
        ToolWaiting((ContentPart.text_part("waiting"),), structured_content=1),
        Suspension("external", "business"),
    )
    with pytest.raises(ToolError, match="output_schema"):
        await binding.invoke(_context())


async def test_retry_receives_original_exception_and_fresh_arguments() -> None:
    error = ConnectionError("transient")
    attempts: list[list[int]] = []

    async def business(values: list[int]) -> int:
        attempts.append(values.copy())
        values.append(2)
        if len(attempts) == 1:
            raise error
        return sum(values)

    function = FunctionTool(_spec(execution=ToolExecution(idempotent=True)), business)
    call = StructuredToolCall("call", "business", {"values": [1]})
    with pytest.raises(ConnectionError) as caught:
        await function.invoke(call, _context())
    assert caught.value is error
    attempts.clear()
    retry = RetryingTool(function, backoff_initial_seconds=0.001, jitter_ratio=0)
    result = await retry.invoke(call, _context())
    assert attempts == [[1], [1]]
    assert result.outcome.structured_content == 3
    assert call.arguments == {"values": [1]}


@pytest.mark.parametrize("freeform", [False, True])
async def test_cancellation_is_control_flow(freeform: bool) -> None:
    entered = asyncio.Event()

    async def business(text: str = "") -> str:
        entered.set()
        await asyncio.Event().wait()
        return text

    if freeform:
        invocation = FreeformFunctionTool(
            FreeformToolSpec("business", "Business"), business
        ).invoke(FreeformToolCall("call", "business", "text"), _context())
    else:
        invocation = FunctionTool(_spec(), business).invoke(
            StructuredToolCall("call", "business"), _context()
        )
    task = asyncio.create_task(invocation)
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_direct_invocation_rejects_wrong_call_kinds_and_unparsed_input() -> None:
    async def business(text: str = "") -> str:
        return text

    structured = FunctionTool(_spec(), business)
    freeform = FreeformFunctionTool(FreeformToolSpec("raw", "Raw"), business)
    with pytest.raises(TypeError, match="StructuredToolCall"):
        await structured.invoke(cast(Any, FreeformToolCall("call", "business", "raw")), _context())
    with pytest.raises(TypeError, match="FreeformToolCall"):
        await freeform.invoke(cast(Any, StructuredToolCall("call", "raw")), _context())
    with pytest.raises(ToolError, match="JSON object arguments"):
        await structured.invoke(
            StructuredToolCall("call", "business", arguments=None, raw_input="[1]"), _context()
        )


def test_both_adapters_reject_non_async_functions() -> None:
    def business(text: str) -> str:
        return text

    for construct in (
        lambda: FunctionTool(_spec(), cast(ToolFunction, business)),
        lambda: FreeformFunctionTool(FreeformToolSpec("raw", "Raw"), cast(ToolFunction, business)),
    ):
        with pytest.raises(TypeError, match="must be async"):
            construct()
