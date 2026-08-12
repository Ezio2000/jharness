"""Concrete Python tool protocol and function adapter."""

from __future__ import annotations

from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
from inspect import iscoroutinefunction
from typing import Any, Protocol, cast, runtime_checkable

from jharness.kernel import (
    FreeformToolCall,
    FreeformToolSpec,
    StructuredToolCall,
    StructuredToolSpec,
    ToolContext,
    ToolExecution,
    ToolResult,
    ToolRisk,
)


@runtime_checkable
class Tool(Protocol):
    """One structured-input async tool implementation."""

    @property
    def spec(self) -> StructuredToolSpec: ...

    async def invoke(self, call: StructuredToolCall, context: ToolContext) -> ToolResult: ...


ToolFunction = Callable[[StructuredToolCall, ToolContext], Coroutine[Any, Any, ToolResult]]
FreeformToolFunction = Callable[[FreeformToolCall, ToolContext], Coroutine[Any, Any, ToolResult]]


@runtime_checkable
class FreeformTool(Protocol):
    """One freeform-input async tool implementation."""

    @property
    def spec(self) -> FreeformToolSpec: ...

    async def invoke(self, call: FreeformToolCall, context: ToolContext) -> ToolResult: ...


@dataclass(frozen=True, slots=True)
class FunctionTool:
    """Adapt one async function to the concrete tool protocol."""

    spec: StructuredToolSpec
    function: ToolFunction

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.spec), StructuredToolSpec):
            raise TypeError("function tool spec must be StructuredToolSpec")
        if not iscoroutinefunction(self.function):
            raise TypeError("function tool must be async")

    async def invoke(self, call: StructuredToolCall, context: ToolContext) -> ToolResult:
        return await self.function(call, context)


@dataclass(frozen=True, slots=True)
class FreeformFunctionTool:
    """Adapt one async string-input function to the freeform tool protocol."""

    spec: FreeformToolSpec
    function: FreeformToolFunction

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.spec), FreeformToolSpec):
            raise TypeError("freeform function tool spec must be FreeformToolSpec")
        if not iscoroutinefunction(self.function):
            raise TypeError("freeform function tool must be async")

    async def invoke(self, call: FreeformToolCall, context: ToolContext) -> ToolResult:
        return await self.function(call, context)


def function_tool(
    *,
    name: str,
    description: str,
    input_schema: Mapping[str, Any] | bool,
    output_schema: Mapping[str, Any] | bool | None = None,
    execution: ToolExecution | None = None,
    risk: ToolRisk | None = None,
) -> Callable[[ToolFunction], FunctionTool]:
    """Create a `FunctionTool` while keeping schemas and policies explicit."""

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
        return FunctionTool(spec, function)

    return decorate


def freeform_tool(
    *,
    name: str,
    description: str,
    output_schema: Mapping[str, Any] | bool | None = None,
    execution: ToolExecution | None = None,
    risk: ToolRisk | None = None,
) -> Callable[[FreeformToolFunction], FreeformFunctionTool]:
    """Create a `FreeformFunctionTool` with explicit output and scheduling policy."""

    spec = FreeformToolSpec(
        name,
        description,
        output_schema,
        ToolExecution() if execution is None else execution,
        ToolRisk() if risk is None else risk,
    )

    def decorate(function: FreeformToolFunction) -> FreeformFunctionTool:
        return FreeformFunctionTool(spec, function)

    return decorate
