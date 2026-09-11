"""Concrete Python tool protocol and function adapter."""

from __future__ import annotations

import json
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from inspect import Parameter, Signature, iscoroutinefunction, signature
from typing import Any, Protocol, cast, runtime_checkable

from jharness.kernel import (
    ContentPart,
    FreeformToolCall,
    FreeformToolSpec,
    SettledResult,
    StructuredToolCall,
    StructuredToolSpec,
    ToolContext,
    ToolError,
    ToolExecution,
    ToolResult,
    ToolRisk,
    ToolSuccess,
    WaitingResult,
    thaw_json_value,
)


@runtime_checkable
class Tool(Protocol):
    """One structured-input async tool implementation."""

    @property
    def spec(self) -> StructuredToolSpec: ...

    async def invoke(self, call: StructuredToolCall, context: ToolContext) -> ToolResult: ...


ToolFunction = Callable[..., Coroutine[Any, Any, object]]


@runtime_checkable
class FreeformTool(Protocol):
    """One freeform-input async tool implementation."""

    @property
    def spec(self) -> FreeformToolSpec: ...

    async def invoke(self, call: FreeformToolCall, context: ToolContext) -> ToolResult: ...


@dataclass(frozen=True, slots=True)
class FunctionTool:
    """Bind JSON object arguments to an async business function by name."""

    spec: StructuredToolSpec
    function: ToolFunction
    context_parameter: str | None = field(default=None, kw_only=True)
    _signature: Signature = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.spec), StructuredToolSpec):
            raise TypeError("function tool spec must be StructuredToolSpec")
        function_signature = _function_signature(self.function, self.context_parameter)
        _validate_context_schema(self.spec.input_schema, self.context_parameter)
        object.__setattr__(self, "_signature", function_signature)

    async def invoke(self, call: StructuredToolCall, context: ToolContext) -> ToolResult:
        if not isinstance(cast(object, call), StructuredToolCall):
            raise TypeError("function tool requires StructuredToolCall")
        if call.arguments is None or call.raw_input is not None:
            raise ToolError("function tool requires JSON object arguments")
        arguments = cast(dict[str, Any], thaw_json_value(call.arguments))
        return await _invoke_function(
            self.function, self._signature, arguments, self.context_parameter, context
        )


@dataclass(frozen=True, slots=True)
class FreeformFunctionTool:
    """Pass an unmodified input string to one async business parameter."""

    spec: FreeformToolSpec
    function: ToolFunction
    context_parameter: str | None = field(default=None, kw_only=True)
    _signature: Signature = field(init=False, repr=False, compare=False)
    _input_parameter: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.spec), FreeformToolSpec):
            raise TypeError("freeform function tool spec must be FreeformToolSpec")
        function_signature = _function_signature(self.function, self.context_parameter)
        parameters = tuple(
            name for name in function_signature.parameters if name != self.context_parameter
        )
        if len(parameters) != 1:
            raise ValueError("freeform function requires exactly one input parameter")
        object.__setattr__(self, "_signature", function_signature)
        object.__setattr__(self, "_input_parameter", parameters[0])

    async def invoke(self, call: FreeformToolCall, context: ToolContext) -> ToolResult:
        if not isinstance(cast(object, call), FreeformToolCall):
            raise TypeError("freeform function tool requires FreeformToolCall")
        return await _invoke_function(
            self.function,
            self._signature,
            {self._input_parameter: call.input},
            self.context_parameter,
            context,
        )


def function_tool(
    *,
    name: str,
    description: str,
    input_schema: Mapping[str, Any] | bool,
    output_schema: Mapping[str, Any] | bool | None = None,
    execution: ToolExecution | None = None,
    risk: ToolRisk | None = None,
    context_parameter: str | None = None,
) -> Callable[[ToolFunction], FunctionTool]:
    """Adapt named business parameters; schemas and context injection stay explicit."""

    tool_execution = ToolExecution() if execution is None else execution
    tool_risk = ToolRisk() if risk is None else risk
    spec = StructuredToolSpec(
        name,
        description,
        input_schema,
        output_schema,
        tool_execution,
        tool_risk,
    )

    def decorate(function: ToolFunction) -> FunctionTool:
        return FunctionTool(spec, function, context_parameter=context_parameter)

    return decorate


def freeform_tool(
    *,
    name: str,
    description: str,
    output_schema: Mapping[str, Any] | bool | None = None,
    execution: ToolExecution | None = None,
    risk: ToolRisk | None = None,
    context_parameter: str | None = None,
) -> Callable[[ToolFunction], FreeformFunctionTool]:
    """Adapt one text parameter with the same result and context rules as function_tool."""

    spec = FreeformToolSpec(
        name,
        description,
        output_schema,
        ToolExecution() if execution is None else execution,
        ToolRisk() if risk is None else risk,
    )

    def decorate(function: ToolFunction) -> FreeformFunctionTool:
        return FreeformFunctionTool(spec, function, context_parameter=context_parameter)

    return decorate


def _function_signature(function: ToolFunction, context_parameter: str | None) -> Signature:
    if not iscoroutinefunction(function):
        raise TypeError("tool function must be async")
    function_signature = signature(function)
    for parameter in function_signature.parameters.values():
        if parameter.kind not in {Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY}:
            raise TypeError("tool function parameters must accept named arguments")
    if context_parameter is not None:
        if not isinstance(cast(object, context_parameter), str):
            raise TypeError("context_parameter must be a string or None")
        if not context_parameter:
            raise ValueError("context_parameter must not be empty")
        parameter = function_signature.parameters.get(context_parameter)
        if parameter is None or parameter.kind != Parameter.KEYWORD_ONLY:
            raise ValueError("context_parameter must name a keyword-only function parameter")
    return function_signature


def _validate_context_schema(schema: Mapping[str, Any] | bool, name: str | None) -> None:
    if name is None or isinstance(schema, bool):
        return
    properties = schema.get("properties", {})
    required = schema.get("required", ())
    if (isinstance(properties, Mapping) and name in properties) or (
        isinstance(required, Sequence) and not isinstance(required, str) and name in required
    ):
        raise ValueError("input_schema must not declare the context parameter")


async def _invoke_function(
    function: ToolFunction,
    function_signature: Signature,
    arguments: dict[str, Any],
    context_parameter: str | None,
    context: ToolContext,
) -> ToolResult:
    if context_parameter is not None:
        if context_parameter in arguments:
            raise ToolError("tool arguments must not supply the context parameter")
        arguments[context_parameter] = context
    try:
        bound = function_signature.bind(**arguments)
    except TypeError as exc:
        raise ToolError(f"tool arguments do not match function signature: {exc}") from exc
    value = await function(*bound.args, **bound.kwargs)
    return _function_result(value)


def _function_result(value: object) -> ToolResult:
    if isinstance(value, SettledResult | WaitingResult):
        return value
    try:
        plain = thaw_json_value(value, label="function result")
        text = (
            plain
            if isinstance(plain, str)
            else json.dumps(
                plain, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        )
    except (TypeError, ValueError) as exc:
        raise ToolError(f"function result must be JSON-compatible or ToolResult: {exc}") from exc
    return SettledResult(ToolSuccess((ContentPart.text_part(text),), structured_content=plain))
