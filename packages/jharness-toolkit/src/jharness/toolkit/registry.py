"""Immutable invocation catalogs with compiled JSON Schema validation."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from inspect import iscoroutinefunction
from threading import Lock
from types import MappingProxyType
from typing import Any, cast
from uuid import uuid4

from jsonschema import Draft202012Validator, validators
from jsonschema.exceptions import SchemaError, ValidationError
from jsonschema.protocols import Validator
from referencing import Registry
from referencing.exceptions import Unresolvable
from referencing.jsonschema import DRAFT202012, Schema, SchemaRegistry, SchemaResource

from jharness.kernel import (
    FreeformToolCall,
    FreeformToolSpec,
    RuntimeToolCall,
    RuntimeToolSpec,
    SettledResult,
    StructuredToolCall,
    StructuredToolSpec,
    ToolBinding,
    ToolCatalog,
    ToolContext,
    ToolError,
    ToolFailure,
    ToolResult,
    WaitingResult,
    thaw_json_value,
)
from jharness.toolkit.tool import FreeformTool, Tool

RegisteredTool = Tool | FreeformTool


def _is_lexical_integer(_checker: object, value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


_extend_validator = cast(
    Callable[..., type[Draft202012Validator]],
    validators.extend,  # pyright: ignore[reportUnknownMemberType]
)
StrictDraft202012Validator: type[Draft202012Validator] = _extend_validator(
    Draft202012Validator,
    type_checker=Draft202012Validator.TYPE_CHECKER.redefine(
        "integer",
        _is_lexical_integer,
    ),
)


@dataclass(frozen=True, slots=True)
class _Registered:
    tool: RegisteredTool
    spec: RuntimeToolSpec
    input_validator: Validator | None
    output_validator: Validator | None


class ToolRegistry:
    """Thread-safe registry that opens immutable invocation catalogs."""

    __slots__ = ("_entries", "_lock")

    def __init__(self, tools: Sequence[RegisteredTool] = ()) -> None:
        self._lock = Lock()
        self._entries: dict[str, _Registered] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: RegisteredTool) -> None:
        _validate_tool(tool)
        spec = tool.spec
        registered = _Registered(
            tool,
            spec,
            (
                _compile_schema(spec.input_schema, f"tool {spec.name} input_schema")
                if isinstance(spec, StructuredToolSpec)
                else None
            ),
            (
                None
                if spec.output_schema is None
                else _compile_schema(
                    spec.output_schema,
                    f"tool {spec.name} output_schema",
                )
            ),
        )
        with self._lock:
            if spec.name in self._entries:
                raise ValueError(f"duplicate tool name: {spec.name}")
            self._entries[spec.name] = registered

    async def open_catalog(self) -> ToolCatalog:
        with self._lock:
            return _Catalog(self._entries)


class _Catalog:
    __slots__ = ("_entries", "_specs")

    def __init__(self, entries: Mapping[str, _Registered]) -> None:
        self._entries = MappingProxyType(dict(entries))
        self._specs = tuple(entry.spec for entry in self._entries.values())

    def specs(self) -> tuple[RuntimeToolSpec, ...]:
        return self._specs

    def spec(self, name: str) -> RuntimeToolSpec | None:
        entry = self._entries.get(name)
        return None if entry is None else entry.spec

    def bind(self, call: RuntimeToolCall) -> ToolBinding:
        entry = self._entries.get(call.name)
        if entry is None:
            raise ToolError(f"unknown tool: {call.name}")
        if isinstance(call, StructuredToolCall) and isinstance(entry.spec, StructuredToolSpec):
            validator = cast(Validator, entry.input_validator)
            try:
                validator.validate(thaw_json_value(call.arguments))
            except ValidationError as exc:
                raise ToolError(
                    f"tool {call.name} arguments do not match input_schema: {exc.message}"
                ) from exc
            except Unresolvable as exc:
                raise ToolError(
                    f"tool {call.name} input_schema reference cannot be resolved"
                ) from exc
        elif not isinstance(call, FreeformToolCall) or not isinstance(entry.spec, FreeformToolSpec):
            raise ToolError(f"tool {call.name} does not accept this runtime input kind")
        return _Binding(call, entry)


@dataclass(frozen=True, slots=True)
class _Binding:
    call: RuntimeToolCall
    _entry: _Registered

    @property
    def spec(self) -> RuntimeToolSpec:
        return self._entry.spec

    async def invoke(self, context: ToolContext) -> ToolResult:
        if isinstance(self.call, StructuredToolCall):
            tool = cast(Tool, self._entry.tool)
            result = _ensure_result(await tool.invoke(self.call, context))
        else:
            tool = cast(FreeformTool, self._entry.tool)
            result = _ensure_result(await tool.invoke(self.call, context))
        validator = self._entry.output_validator
        if validator is not None and not isinstance(result.outcome, ToolFailure):
            try:
                validator.validate(thaw_json_value(result.outcome.structured_content))
            except ValidationError as exc:
                raise ToolError(
                    f"tool {self.call.name} structured_content does not match "
                    f"output_schema: {exc.message}"
                ) from exc
            except Unresolvable as exc:
                raise ToolError(
                    f"tool {self.call.name} output_schema reference cannot be resolved"
                ) from exc
        return result


def _compile_schema(
    schema: Mapping[str, Any] | bool,
    label: str,
) -> Validator:
    plain: Schema = (
        schema if isinstance(schema, bool) else cast(dict[str, Any], thaw_json_value(schema))
    )
    try:
        Draft202012Validator.check_schema(plain)
    except SchemaError as exc:
        raise ValueError(f"{label} must be a valid JSON Schema: {exc.message}") from exc
    base = (
        cast(str, plain.get("$id"))
        if isinstance(plain, dict) and isinstance(plain.get("$id"), str)
        else f"urn:jharness:toolkit:schema:{uuid4()}"
    )
    resource: SchemaResource = DRAFT202012.create_resource(plain)
    registry: SchemaRegistry = Registry[Schema]().with_resource(base, resource).crawl()
    resolver = registry.resolver(base)
    try:
        for reference in _references(plain):
            resolver.lookup(reference)
    except Exception as exc:
        raise ValueError(f"{label} contains an unresolvable reference") from exc
    return StrictDraft202012Validator(plain, registry=registry)


def _ensure_result(value: object) -> ToolResult:
    if not isinstance(value, SettledResult | WaitingResult):
        raise ToolError("tool invoke must return SettledResult or WaitingResult")
    return value


def _validate_tool(value: object) -> None:
    if not isinstance(value, Tool | FreeformTool):
        raise TypeError("registered tool must implement Tool")
    if not isinstance(cast(object, value.spec), RuntimeToolSpec):
        raise TypeError("registered tool spec must be RuntimeToolSpec")
    if not iscoroutinefunction(value.invoke):
        raise TypeError("registered tool invoke must be async")


def _references(value: object) -> Iterator[str]:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        for keyword in ("$ref", "$dynamicRef"):
            reference = mapping.get(keyword)
            if isinstance(reference, str):
                yield reference
        for item in mapping.values():
            yield from _references(item)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for item in cast(Sequence[object], value):
            yield from _references(item)
