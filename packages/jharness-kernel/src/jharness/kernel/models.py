"""Immutable provider-neutral model protocol values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, cast, runtime_checkable

from jharness.kernel._validation import (
    expect_bool,
    expect_instance,
    expect_instance_tuple,
    expect_int,
    expect_non_empty_str,
    expect_nonnegative_int,
    expect_number,
    expect_optional_int,
    expect_str,
    freeze_mapping,
)
from jharness.kernel.context import RunContext
from jharness.kernel.messages import (
    ContentPart,
    Message,
    ModelOutputItem,
    ProviderToolCall,
    ProviderToolId,
    ProviderToolStatus,
    RuntimeToolCall,
    RuntimeToolKind,
)

if TYPE_CHECKING:
    from jharness.kernel.tools import RuntimeToolSpec


@dataclass(frozen=True, slots=True)
class ModelOptions:
    """Portable model sampling and output options."""

    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    stop: tuple[str, ...] | None = None
    seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        if self.model is not None:
            expect_str(self.model, "model")
        if self.temperature is not None:
            expect_number(self.temperature, "temperature")
        if self.top_p is not None:
            expect_number(self.top_p, "top_p")
        if (
            self.max_output_tokens is not None
            and expect_int(self.max_output_tokens, "max_output_tokens") < 1
        ):
            raise ValueError("max_output_tokens must be >= 1")
        if self.stop is not None:
            object.__setattr__(self, "stop", expect_instance_tuple(self.stop, str, "stop"))
        expect_optional_int(self.seed, "seed")
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata, "model metadata"))


@dataclass(frozen=True, slots=True)
class ToolChoice:
    """Portable model-side tool selection intent."""

    type: str = "auto"
    name: str | None = None
    provider_tool: ProviderToolId | None = None
    allow_parallel_runtime_tool_calls: bool = True

    def __post_init__(self) -> None:
        choice_type = expect_str(self.type, "tool choice type")
        if choice_type not in {"auto", "none", "required", "runtime", "provider"}:
            raise ValueError(f"unsupported tool choice: {choice_type}")
        if choice_type == "runtime":
            if self.name is None or not expect_str(self.name, "tool choice name"):
                raise ValueError("runtime tool choice requires name")
            if self.provider_tool is not None:
                raise ValueError("runtime tool choice cannot carry provider_tool")
        elif choice_type == "provider":
            if self.name is not None:
                raise ValueError("provider tool choice cannot carry name")
            if self.provider_tool is None:
                raise ValueError("provider tool choice requires provider_tool")
            expect_instance(self.provider_tool, ProviderToolId, "tool choice provider_tool")
        elif self.name is not None or self.provider_tool is not None:
            raise ValueError("only targeted tool choice may carry a target")
        expect_bool(self.allow_parallel_runtime_tool_calls, "allow_parallel_runtime_tool_calls")


@dataclass(frozen=True, slots=True)
class ResponseFormat:
    """Portable final response-format request."""

    type: str
    schema: Mapping[str, Any] | bool | None = None
    strict: bool = False

    def __post_init__(self) -> None:
        format_type = expect_str(self.type, "response format type")
        expect_bool(self.strict, "response format strict")
        if format_type not in {"text", "json_object", "json_schema"}:
            raise ValueError(f"unsupported response format: {format_type}")
        if format_type == "json_schema":
            if not isinstance(self.schema, Mapping | bool):
                raise TypeError("json_schema response format requires schema")
            if isinstance(self.schema, Mapping):
                object.__setattr__(self, "schema", freeze_mapping(self.schema, "response schema"))
        elif self.schema is not None:
            raise ValueError("only json_schema response format may carry schema")
        elif self.strict:
            raise ValueError("only json_schema response format may be strict")


_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
)


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """Standard cumulative model usage fields."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in _USAGE_FIELDS:
            value = cast(int | None, getattr(self, name))
            if value is not None:
                expect_nonnegative_int(value, name)

    def add(self, other: ModelUsage | None) -> ModelUsage:
        """Add independently reported usage counters field by field."""

        if other is None:
            return self
        expect_instance(other, ModelUsage, "model usage")
        values: dict[str, int | None] = {}
        for name in _USAGE_FIELDS:
            current = cast(int | None, getattr(self, name))
            added = cast(int | None, getattr(other, name))
            values[name] = current if added is None else (current or 0) + added
        return ModelUsage(**values)

    def merge_snapshot(self, other: ModelUsage) -> ModelUsage:
        """Replace fields reported by a later cumulative usage snapshot."""

        expect_instance(other, ModelUsage, "model usage snapshot")
        return ModelUsage(
            **{
                name: (
                    getattr(other, name)
                    if getattr(other, name) is not None
                    else getattr(self, name)
                )
                for name in _USAGE_FIELDS
            }
        )

    def with_fallback(self, fallback: ModelUsage | None) -> ModelUsage:
        """Fill fields omitted by this usage value from a fallback value."""

        if fallback is None:
            return self
        expect_instance(fallback, ModelUsage, "fallback model usage")
        return ModelUsage(
            **{
                name: (
                    getattr(self, name)
                    if getattr(self, name) is not None
                    else getattr(fallback, name)
                )
                for name in _USAGE_FIELDS
            }
        )


_BOOLEAN_CAPABILITY_FIELDS = (
    "streaming",
    "parallel_runtime_tool_calls",
    "parallel_runtime_tool_call_control",
    "structured_output",
    "json_mode",
    "seed",
    "usage_reporting",
)

_TOOL_CHOICE_TYPES = frozenset({"auto", "none", "required", "runtime", "provider"})


def _modality_set(value: object, label: str) -> frozenset[str]:
    if not isinstance(value, frozenset):
        raise TypeError(f"{label} must be a frozenset")
    modalities = cast(frozenset[object], value)
    if not modalities:
        raise ValueError(f"{label} must not be empty")
    if any(not isinstance(item, str) or not item for item in modalities):
        raise ValueError(f"{label} must contain non-empty strings")
    return cast(frozenset[str], modalities)


def _runtime_tool_kind_set(value: object) -> frozenset[RuntimeToolKind]:
    if not isinstance(value, frozenset):
        raise TypeError("model runtime_tool_kinds must be a frozenset")
    kinds = cast(frozenset[object], value)
    if any(not isinstance(item, RuntimeToolKind) for item in kinds):
        raise TypeError("model runtime_tool_kinds must contain RuntimeToolKind values")
    return cast(frozenset[RuntimeToolKind], kinds)


def _tool_choice_type_set(value: object) -> frozenset[str]:
    if not isinstance(value, frozenset):
        raise TypeError("model tool_choice_types must be a frozenset")
    choices = cast(frozenset[object], value)
    if any(not isinstance(item, str) or not item for item in choices):
        raise ValueError("model tool_choice_types must contain non-empty strings")
    unsupported = choices.difference(_TOOL_CHOICE_TYPES)
    if unsupported:
        choice = min(cast(frozenset[str], unsupported))
        raise ValueError(f"unsupported model tool choice type: {choice}")
    typed_choices = cast(frozenset[str], choices)
    if "auto" not in typed_choices:
        raise ValueError("model tool_choice_types must include auto")
    return typed_choices


def _provider_tool_set(value: object) -> frozenset[ProviderToolId]:
    if not isinstance(value, frozenset):
        raise TypeError("model provider_tools must be a frozenset")
    tools = cast(frozenset[object], value)
    if any(not isinstance(item, ProviderToolId) for item in tools):
        raise TypeError("model provider_tools must contain ProviderToolId values")
    return cast(frozenset[ProviderToolId], tools)


def _validate_tool_choice_capabilities(
    choices: frozenset[str],
    runtime_kinds: frozenset[RuntimeToolKind],
    provider_tools: frozenset[ProviderToolId],
) -> None:
    if not runtime_kinds and "runtime" in choices:
        raise ValueError(
            "model tool_choice_types cannot include runtime without runtime tool kinds"
        )
    if not provider_tools and "provider" in choices:
        raise ValueError("model tool_choice_types cannot include provider without provider_tools")


def _model_output(value: object, label: str) -> tuple[ModelOutputItem, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{label} must be a tuple")
    items = cast(tuple[object, ...], value)
    if any(
        not isinstance(item, ContentPart | RuntimeToolCall | ProviderToolCall) for item in items
    ):
        raise TypeError(f"{label} contains an unsupported item")
    return cast(tuple[ModelOutputItem, ...], items)


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Immutable advertised model capabilities."""

    streaming: bool = False
    runtime_tool_kinds: frozenset[RuntimeToolKind] = field(
        default_factory=lambda: frozenset({RuntimeToolKind.STRUCTURED})
    )
    tool_choice_types: frozenset[str] = field(
        default_factory=lambda: frozenset({"auto", "none", "required", "runtime"})
    )
    parallel_runtime_tool_calls: bool = True
    parallel_runtime_tool_call_control: bool = True
    input_modalities: frozenset[str] = field(default_factory=lambda: frozenset({"text"}))
    output_modalities: frozenset[str] = field(default_factory=lambda: frozenset({"text"}))
    provider_tools: frozenset[ProviderToolId] = field(
        default_factory=lambda: frozenset[ProviderToolId]()
    )
    structured_output: bool = True
    json_mode: bool = True
    seed: bool = True
    usage_reporting: bool = True

    def __post_init__(self) -> None:
        for name in _BOOLEAN_CAPABILITY_FIELDS:
            expect_bool(getattr(self, name), f"model capability {name}")
        object.__setattr__(
            self,
            "input_modalities",
            _modality_set(self.input_modalities, "model input modalities"),
        )
        object.__setattr__(
            self,
            "output_modalities",
            _modality_set(self.output_modalities, "model output modalities"),
        )
        runtime_tool_kinds = _runtime_tool_kind_set(self.runtime_tool_kinds)
        tool_choice_types = _tool_choice_type_set(self.tool_choice_types)
        provider_tools = _provider_tool_set(self.provider_tools)
        _validate_tool_choice_capabilities(
            tool_choice_types,
            runtime_tool_kinds,
            provider_tools,
        )
        object.__setattr__(self, "runtime_tool_kinds", runtime_tool_kinds)
        object.__setattr__(self, "tool_choice_types", tool_choice_types)
        object.__setattr__(self, "provider_tools", provider_tools)


@dataclass(frozen=True, slots=True)
class ProviderToolSpec:
    """Provider-specific tool configuration carried through the neutral model port."""

    tool: ProviderToolId
    configuration: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        expect_instance(self.tool, ProviderToolId, "provider tool spec tool")
        object.__setattr__(
            self,
            "configuration",
            freeze_mapping(self.configuration, "provider tool configuration"),
        )


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """Complete provider-neutral model input."""

    messages: tuple[Message, ...]
    runtime_tools: tuple[RuntimeToolSpec, ...] = ()
    provider_tools: tuple[ProviderToolSpec, ...] = ()
    options: ModelOptions = field(default_factory=ModelOptions)
    tool_choice: ToolChoice = field(default_factory=ToolChoice)
    response_format: ResponseFormat | None = None

    def __post_init__(self) -> None:
        from jharness.kernel.tools import RuntimeToolSpec

        messages = expect_instance_tuple(self.messages, Message, "model request messages")
        raw_tools = cast(object, self.runtime_tools)
        if not isinstance(raw_tools, tuple):
            raise TypeError("model request runtime_tools must contain RuntimeToolSpec values")
        tool_values = cast(tuple[object, ...], raw_tools)
        if any(not isinstance(item, RuntimeToolSpec) for item in tool_values):
            raise TypeError("model request runtime_tools must contain RuntimeToolSpec values")
        tools = cast(tuple[RuntimeToolSpec, ...], raw_tools)
        provider_tools = expect_instance_tuple(
            self.provider_tools,
            ProviderToolSpec,
            "model request provider_tools",
        )
        if not messages:
            raise ValueError("model request requires messages")
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError("model request tool names must be unique")
        expect_instance(self.options, ModelOptions, "model request options")
        provider_ids = [spec.tool for spec in provider_tools]
        if len(provider_ids) != len(set(provider_ids)):
            raise ValueError("model request provider tools must be unique")
        expect_instance(
            self.tool_choice,
            ToolChoice,
            "model request tool_choice",
        )
        if self.tool_choice.type == "runtime" and self.tool_choice.name not in names:
            raise ValueError("model request tool choice names an unavailable runtime tool")
        if (
            self.tool_choice.type == "provider"
            and self.tool_choice.provider_tool not in provider_ids
        ):
            raise ValueError("model request tool choice names an unavailable provider tool")
        if self.tool_choice.type == "required" and not tools and not provider_tools:
            raise ValueError("required tool choice requires at least one tool")
        if self.response_format is not None:
            expect_instance(self.response_format, ResponseFormat, "model request response_format")
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "runtime_tools", tools)
        object.__setattr__(self, "provider_tools", provider_tools)

    @property
    def may_return_runtime_tool_calls(self) -> bool:
        """Whether this request permits the model to select a runtime-owned tool."""

        return bool(self.runtime_tools) and self.tool_choice.type not in {"none", "provider"}

    @property
    def runtime_tool_kinds(self) -> frozenset[RuntimeToolKind]:
        """Return the input kinds declared by this request's runtime tools."""

        from jharness.kernel.tools import StructuredToolSpec

        return frozenset(
            RuntimeToolKind.STRUCTURED
            if isinstance(spec, StructuredToolSpec)
            else RuntimeToolKind.FREEFORM
            for spec in self.runtime_tools
        )


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """The sole complete provider-neutral model result."""

    output: tuple[ModelOutputItem, ...]
    finish_reason: str | None = None
    usage: ModelUsage | None = None
    model_id: str | None = None
    response_id: str | None = None
    provider_turn_pending: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        output = _model_output(self.output, "model response output")
        if not output:
            raise ValueError("model response requires output")
        ids = [item.id for item in output if isinstance(item, RuntimeToolCall | ProviderToolCall)]
        if len(ids) != len(set(ids)):
            raise ValueError("model response tool call ids must be unique")
        if self.finish_reason is not None:
            expect_str(self.finish_reason, "finish_reason")
        if self.usage is not None:
            expect_instance(self.usage, ModelUsage, "model response usage")
        if self.model_id is not None:
            expect_str(self.model_id, "model_id")
        if self.response_id is not None:
            expect_str(self.response_id, "response_id")
        expect_bool(self.provider_turn_pending, "provider_turn_pending")
        if (
            any(
                isinstance(item, ProviderToolCall) and item.status is ProviderToolStatus.IN_PROGRESS
                for item in output
            )
            and not self.provider_turn_pending
        ):
            raise ValueError("in-progress provider tool call requires a pending provider turn")
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata, "model metadata"))

    def to_assistant_message(self) -> Message:
        """Project the complete response into durable conversation history."""

        return Message.assistant(self.output)

    def runtime_tool_calls(self) -> tuple[RuntimeToolCall, ...]:
        """Return calls that the JHarness runtime must execute."""

        return tuple(item for item in self.output if isinstance(item, RuntimeToolCall))

    def provider_tool_calls(self) -> tuple[ProviderToolCall, ...]:
        """Return calls already owned by the remote provider."""

        return tuple(item for item in self.output if isinstance(item, ProviderToolCall))

    def visible_parts(self) -> tuple[ContentPart, ...]:
        """Project final user-visible content while preserving output order."""

        return self.to_assistant_message().visible_parts()


@dataclass(frozen=True, slots=True)
class ModelContentDelta:
    """Incremental content for one zero-based response-part position."""

    output_index: int
    text_delta: str
    part_type: str = "text"
    content_index: int = 0
    data: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        expect_nonnegative_int(self.output_index, "content delta output_index")
        expect_nonnegative_int(self.content_index, "content delta content_index")
        expect_str(self.text_delta, "content delta text")
        expect_non_empty_str(self.part_type, "content delta part_type")
        object.__setattr__(self, "data", freeze_mapping(self.data, "content delta data"))


@dataclass(frozen=True, slots=True)
class ModelRuntimeToolCallDelta:
    """Incremental input and identity for one ordered runtime tool call."""

    output_index: int
    input_kind: RuntimeToolKind
    input_delta: str
    id: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        expect_nonnegative_int(self.output_index, "tool call delta output_index")
        expect_instance(self.input_kind, RuntimeToolKind, "tool call delta input_kind")
        expect_str(self.input_delta, "tool call input_delta")
        if self.id is not None:
            expect_str(self.id, "tool call delta id")
        if self.name is not None:
            expect_str(self.name, "tool call delta name")
        if self.id == "" or self.name == "":
            raise ValueError("tool call delta id and name must not be empty")


@dataclass(frozen=True, slots=True)
class ModelReasoningDelta:
    """Incremental reasoning text for one zero-based response position."""

    output_index: int
    text_delta: str
    content_index: int = 0

    def __post_init__(self) -> None:
        expect_nonnegative_int(self.output_index, "reasoning delta output_index")
        expect_nonnegative_int(self.content_index, "reasoning delta content_index")
        expect_str(self.text_delta, "reasoning delta text")


@dataclass(frozen=True, slots=True)
class ModelProviderToolCallDelta:
    """Live-only progress from one provider-executed tool call."""

    output_index: int
    id: str
    tool: ProviderToolId
    status: ProviderToolStatus | None = None
    event: str | None = None
    data: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def __post_init__(self) -> None:
        expect_nonnegative_int(self.output_index, "provider tool delta output_index")
        expect_non_empty_str(self.id, "provider tool delta id")
        expect_instance(self.tool, ProviderToolId, "provider tool delta tool")
        if self.status is not None:
            expect_instance(self.status, ProviderToolStatus, "provider tool delta status")
        if self.event is not None:
            expect_non_empty_str(self.event, "provider tool delta event")
        object.__setattr__(self, "data", freeze_mapping(self.data, "provider tool delta data"))


@dataclass(frozen=True, slots=True)
class ModelUsageDelta:
    """A cumulative provider usage snapshot."""

    usage: ModelUsage

    def __post_init__(self) -> None:
        expect_instance(self.usage, ModelUsage, "usage delta")


ModelDelta: TypeAlias = (
    ModelContentDelta
    | ModelRuntimeToolCallDelta
    | ModelReasoningDelta
    | ModelProviderToolCallDelta
    | ModelUsageDelta
)


class DeltaSink(Protocol):
    """Ordered async observer for live-only model deltas."""

    async def __call__(self, delta: ModelDelta, /) -> None: ...


@runtime_checkable
class Model(Protocol):
    """One provider-neutral model operation with optional live deltas."""

    @property
    def capabilities(self) -> ModelCapabilities: ...

    async def invoke(
        self,
        request: ModelRequest,
        context: RunContext,
        *,
        stream: bool,
        emit_delta: DeltaSink | None,
    ) -> ModelResponse: ...
