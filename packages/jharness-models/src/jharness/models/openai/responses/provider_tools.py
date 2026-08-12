"""Provider-owned tool dialects for OpenAI-compatible Responses APIs."""

from __future__ import annotations

import base64
import binascii
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, cast

from jharness.kernel import (
    ContentPart,
    ErrorInfo,
    Message,
    ModelOutputItem,
    ModelRequest,
    ModelResponse,
    ProviderToolCall,
    ProviderToolId,
    ProviderToolSpec,
    ProviderToolStatus,
    RunContext,
    thaw_json_value,
)
from jharness.models.openai.responses.artifacts import (
    OpenAIResponsesArtifactStore,
    externalize_image_call,
    hydrate_image_call,
    image_call_has_artifact,
    image_call_has_inline_result,
)
from jharness.models.openai.responses.errors import OPENAI_RESPONSES_JSON, OpenAIResponsesError

JsonObject = dict[str, Any]

_IN_PROGRESS_STATUSES = frozenset({"generating", "in_progress", "searching"})
_PROVIDER_TERMINAL_STATUSES = frozenset(
    {
        ProviderToolStatus.COMPLETED,
        ProviderToolStatus.INCOMPLETE,
        ProviderToolStatus.FAILED,
    }
)


@dataclass(frozen=True, slots=True)
class OpenAIResponsesProviderToolStreamUpdate:
    """One normalized status/data update produced by a provider-tool dialect."""

    status: ProviderToolStatus
    data: Mapping[str, object] = field(default_factory=dict[str, object])


class OpenAIResponsesProviderToolCodec(ABC):
    """One immutable provider-tool wire family installed into a profile registry."""

    tool: ProviderToolId
    output_item_type: str
    event_prefix: str

    @property
    @abstractmethod
    def declaration_types(self) -> frozenset[str]:
        """Return every request-side wire discriminator accepted by this codec."""

    @abstractmethod
    def encode_declaration(self, spec: ProviderToolSpec) -> JsonObject:
        """Encode one requested declaration after validating its configuration."""

    def encode_choice(self, spec: ProviderToolSpec) -> JsonObject:
        """Encode an exact provider-tool choice using the declaration discriminator."""

        return {"type": cast(str, self.encode_declaration(spec)["type"])}

    @abstractmethod
    def decode_call(
        self,
        item: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> ProviderToolCall:
        """Decode one terminal output item."""

    @abstractmethod
    def encode_history(self, call: ProviderToolCall) -> JsonObject:
        """Encode one durable provider call for a later stateless request."""

    def stream_item_update(
        self, item: Mapping[str, Any]
    ) -> OpenAIResponsesProviderToolStreamUpdate:
        """Decode the provider status carried by output_item.added/done."""

        return OpenAIResponsesProviderToolStreamUpdate(
            provider_status(
                item.get("status"),
                label=f"Responses {self.output_item_type} status",
            )
        )

    @abstractmethod
    def stream_event_update(
        self,
        event_type: str,
        value: Mapping[str, Any],
    ) -> OpenAIResponsesProviderToolStreamUpdate:
        """Decode a tool-specific lifecycle/progress event."""

    def request_requires_artifact_store(self, spec: ProviderToolSpec) -> bool:
        """Return whether invoking this tool requires host artifact persistence."""

        del spec
        return False

    def history_requires_artifact_store(self, call: ProviderToolCall) -> bool:
        """Return whether replaying this call requires artifact hydration."""

        del call
        return False

    def response_requires_artifact_store(self, call: ProviderToolCall) -> bool:
        """Return whether this decoded call still carries inline artifact bytes."""

        del call
        return False

    async def hydrate_call(
        self,
        call: ProviderToolCall,
        store: OpenAIResponsesArtifactStore,
        context: RunContext,
    ) -> ProviderToolCall:
        """Hydrate invocation-local wire data for one durable call."""

        del store, context
        return call

    async def externalize_call(
        self,
        call: ProviderToolCall,
        store: OpenAIResponsesArtifactStore,
        context: RunContext,
    ) -> ProviderToolCall:
        """Externalize provider bytes before checkpoint persistence."""

        del store, context
        return call


@dataclass(frozen=True, slots=True)
class OpenAIResponsesWebSearchTool(OpenAIResponsesProviderToolCodec):
    """Responses web-search request, output, history, and SSE dialect."""

    tool: ProviderToolId
    allowed_variants: frozenset[str] = field(default_factory=lambda: frozenset({"web_search"}))
    default_variant: str = "web_search"
    configuration_fields: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"filters", "search_context_size", "user_location", "variant"}
        )
    )
    output_item_type: str = field(default="web_search_call", init=False)
    event_prefix: str = field(default="response.web_search_call.", init=False)

    def __post_init__(self) -> None:
        _validate_tool_identity(self.tool, "web_search")
        variants = _nonempty_string_set(self.allowed_variants, "web search variants")
        if self.default_variant not in variants:
            raise ValueError("default web search variant must be allowed")
        fields = _nonempty_string_set(
            self.configuration_fields,
            "web search configuration fields",
            allow_empty=True,
        )
        if "variant" not in fields and variants != frozenset({self.default_variant}):
            raise ValueError("multiple web search variants require the variant configuration field")
        object.__setattr__(self, "allowed_variants", variants)
        object.__setattr__(self, "configuration_fields", fields)

    @property
    def declaration_types(self) -> frozenset[str]:
        return self.allowed_variants

    def encode_declaration(self, spec: ProviderToolSpec) -> JsonObject:
        _require_spec_identity(spec, self.tool)
        configuration = _configuration(spec, self.configuration_fields)
        variant = configuration.pop("variant", self.default_variant)
        if not isinstance(variant, str) or variant not in self.allowed_variants:
            expected = ", ".join(sorted(self.allowed_variants))
            raise OpenAIResponsesError(f"web_search variant must be one of: {expected}")
        return {"type": variant, **configuration}

    def decode_call(
        self,
        item: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> ProviderToolCall:
        del response
        call_id = OPENAI_RESPONSES_JSON.required_string(
            item.get("id"),
            "Responses web search call id",
        )
        status = provider_status(item.get("status"), label="Responses web search status")
        action = item.get("action")
        if action is None:
            arguments: Mapping[str, Any] = {}
        elif isinstance(action, Mapping):
            arguments = cast(Mapping[str, Any], action)
        else:
            raise OpenAIResponsesError("Responses web search action must be an object or null")
        return ProviderToolCall(
            id=call_id,
            tool=self.tool,
            status=status,
            arguments=arguments,
            error=_provider_error(item, "web_search")
            if status is ProviderToolStatus.FAILED
            else None,
            metadata={"responses": {"item": dict(item)}},
        )

    def encode_history(self, call: ProviderToolCall) -> JsonObject:
        _require_call_identity(call, self.tool)
        raw = _native_item(call.metadata)
        if raw is not None:
            if raw.get("type") != self.output_item_type:
                raise OpenAIResponsesError("Responses provider metadata has the wrong item type")
            return raw
        encoded: JsonObject = {
            "type": self.output_item_type,
            "id": call.id,
            "status": call.status.value,
        }
        if call.arguments:
            encoded["action"] = thaw_json_value(call.arguments)
        return encoded

    def stream_event_update(
        self,
        event_type: str,
        value: Mapping[str, Any],
    ) -> OpenAIResponsesProviderToolStreamUpdate:
        suffix = event_type.removeprefix(self.event_prefix)
        status = _provider_event_status(suffix, "web search")
        data: dict[str, object] = {}
        if isinstance(value.get("action"), Mapping):
            data["action"] = value["action"]
        return OpenAIResponsesProviderToolStreamUpdate(status, data)


@dataclass(frozen=True, slots=True)
class OpenAIResponsesImageGenerationTool(OpenAIResponsesProviderToolCodec):
    """Responses image-generation dialect including artifact persistence hooks."""

    tool: ProviderToolId
    configuration_fields: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "action",
                "background",
                "input_fidelity",
                "input_image_mask",
                "moderation",
                "output_compression",
                "output_format",
                "partial_images",
                "quality",
                "size",
            }
        )
    )
    output_item_type: str = field(default="image_generation_call", init=False)
    event_prefix: str = field(default="response.image_generation_call.", init=False)

    def __post_init__(self) -> None:
        _validate_tool_identity(self.tool, "image_generation")
        object.__setattr__(
            self,
            "configuration_fields",
            _nonempty_string_set(
                self.configuration_fields,
                "image generation configuration fields",
                allow_empty=True,
            ),
        )

    @property
    def declaration_types(self) -> frozenset[str]:
        return frozenset({"image_generation"})

    def encode_declaration(self, spec: ProviderToolSpec) -> JsonObject:
        _require_spec_identity(spec, self.tool)
        configuration = _configuration(spec, self.configuration_fields)
        _validate_image_configuration(configuration)
        return {"type": "image_generation", **configuration}

    def decode_call(
        self,
        item: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> ProviderToolCall:
        call_id = OPENAI_RESPONSES_JSON.required_string(
            item.get("id"),
            "Responses image generation call id",
        )
        status = provider_status(
            item.get("status"),
            label="Responses image generation status",
        )
        result = item.get("result")
        output: tuple[ContentPart, ...] = ()
        if result is not None:
            image_base64 = OPENAI_RESPONSES_JSON.required_string(
                result,
                "Responses image generation result",
            )
            if status is ProviderToolStatus.IN_PROGRESS:
                raise OpenAIResponsesError(
                    "in-progress Responses image generation cannot carry a final result"
                )
            output = (
                ContentPart(
                    type="image",
                    data={"base64": image_base64},
                    media_type=_resolve_image_media_type(
                        image_base64,
                        self._configured_media_type(response.get("tools")),
                    ),
                ),
            )
        elif status is ProviderToolStatus.COMPLETED:
            raise OpenAIResponsesError("completed Responses image generation requires a result")
        return ProviderToolCall(
            id=call_id,
            tool=self.tool,
            status=status,
            output=output,
            error=_provider_error(item, "image_generation")
            if status is ProviderToolStatus.FAILED
            else None,
            metadata={"responses": {"item": _without_result(item)}},
        )

    def encode_history(self, call: ProviderToolCall) -> JsonObject:
        _require_call_identity(call, self.tool)
        raw = _native_item(call.metadata)
        encoded = {} if raw is None else raw
        if raw is not None and raw.get("type") != self.output_item_type:
            raise OpenAIResponsesError("Responses provider metadata has the wrong item type")
        result: str | None = None
        if call.output:
            if len(call.output) != 1 or call.output[0].type != "image":
                raise OpenAIResponsesError(
                    "image_generation history requires exactly one image output"
                )
            raw_base64 = call.output[0].data.get("base64")
            if not isinstance(raw_base64, str) or not raw_base64:
                raise OpenAIResponsesError(
                    "image_generation history image requires non-empty base64 data"
                )
            _resolve_image_media_type(raw_base64, call.output[0].media_type)
            result = raw_base64
        if call.status is ProviderToolStatus.COMPLETED and result is None:
            raise OpenAIResponsesError(
                "completed image_generation history requires an image output"
            )
        encoded.update(
            type=self.output_item_type,
            id=call.id,
            status=call.status.value,
            result=result,
        )
        return encoded

    def stream_event_update(
        self,
        event_type: str,
        value: Mapping[str, Any],
    ) -> OpenAIResponsesProviderToolStreamUpdate:
        suffix = event_type.removeprefix(self.event_prefix)
        if suffix == "partial_image":
            image = OPENAI_RESPONSES_JSON.required_string(
                value.get("partial_image_b64"),
                "Responses partial image base64",
            )
            partial_index = _nonnegative_int(
                value.get("partial_image_index"),
                "Responses partial image index",
            )
            if partial_index > 3:
                raise OpenAIResponsesError("Responses partial image index must be at most 3")
            return OpenAIResponsesProviderToolStreamUpdate(
                ProviderToolStatus.IN_PROGRESS,
                {"base64": image, "partial_image_index": partial_index},
            )
        return OpenAIResponsesProviderToolStreamUpdate(
            _provider_event_status(suffix, "image generation")
        )

    def request_requires_artifact_store(self, spec: ProviderToolSpec) -> bool:
        _require_spec_identity(spec, self.tool)
        return True

    def history_requires_artifact_store(self, call: ProviderToolCall) -> bool:
        _require_call_identity(call, self.tool)
        return image_call_has_artifact(call)

    def response_requires_artifact_store(self, call: ProviderToolCall) -> bool:
        _require_call_identity(call, self.tool)
        return image_call_has_inline_result(call)

    async def hydrate_call(
        self,
        call: ProviderToolCall,
        store: OpenAIResponsesArtifactStore,
        context: RunContext,
    ) -> ProviderToolCall:
        _require_call_identity(call, self.tool)
        return await hydrate_image_call(call, store, context)

    async def externalize_call(
        self,
        call: ProviderToolCall,
        store: OpenAIResponsesArtifactStore,
        context: RunContext,
    ) -> ProviderToolCall:
        _require_call_identity(call, self.tool)
        return await externalize_image_call(call, store, context)

    def _configured_media_type(self, value: object) -> str | None:
        if value is None:
            return None
        if not _is_array(value):
            raise OpenAIResponsesError("Responses tools must be an array")
        output_format: str | None = None
        image_tool_seen = False
        for raw_tool in cast(Sequence[object], value):
            declaration = OPENAI_RESPONSES_JSON.mapping(raw_tool, "Responses tool")
            if declaration.get("type") not in self.declaration_types:
                continue
            if image_tool_seen:
                raise OpenAIResponsesError("Responses contains duplicate image_generation tools")
            image_tool_seen = True
            raw_format = declaration.get("output_format")
            if raw_format is not None:
                output_format = OPENAI_RESPONSES_JSON.required_string(
                    raw_format,
                    "Responses image_generation output_format",
                )
        if output_format is None:
            return None
        media_types = {
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
        }
        try:
            return media_types[output_format]
        except KeyError as exc:
            raise OpenAIResponsesError(
                f"unsupported Responses image output format: {output_format}"
            ) from exc


@dataclass(frozen=True, slots=True)
class OpenAIResponsesProviderToolRegistry:
    """Immutable exact-identity registry consumed by the shared Responses codec."""

    codecs: tuple[OpenAIResponsesProviderToolCodec, ...] = ()
    _by_tool: Mapping[ProviderToolId, OpenAIResponsesProviderToolCodec] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _by_output_item: Mapping[str, OpenAIResponsesProviderToolCodec] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _event_codecs: tuple[OpenAIResponsesProviderToolCodec, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        raw_codecs = cast(object, self.codecs)
        if not isinstance(raw_codecs, tuple):
            raise TypeError("Responses provider tool codecs must be a tuple")
        raw_values = cast(tuple[object, ...], raw_codecs)
        by_tool: dict[ProviderToolId, OpenAIResponsesProviderToolCodec] = {}
        by_output: dict[str, OpenAIResponsesProviderToolCodec] = {}
        declarations: dict[str, ProviderToolId] = {}
        prefixes: dict[str, ProviderToolId] = {}
        codecs: list[OpenAIResponsesProviderToolCodec] = []
        for raw_codec in raw_values:
            if not isinstance(raw_codec, OpenAIResponsesProviderToolCodec):
                raise TypeError(
                    "Responses provider tool registry values must be provider tool codecs"
                )
            codec = raw_codec
            codecs.append(codec)
            declaration_types = _validate_codec_shape(codec)
            if codec.tool in by_tool:
                raise ValueError("Responses provider tool identities must be unique")
            if codec.output_item_type in by_output:
                raise ValueError("Responses provider output item types must be unique")
            if any(
                codec.event_prefix.startswith(prefix) or prefix.startswith(codec.event_prefix)
                for prefix in prefixes
            ):
                raise ValueError("Responses provider event prefixes must not overlap")
            for wire_type in declaration_types:
                if wire_type in declarations:
                    raise ValueError("Responses provider declaration types must be unique")
                declarations[wire_type] = codec.tool
            by_tool[codec.tool] = codec
            by_output[codec.output_item_type] = codec
            prefixes[codec.event_prefix] = codec.tool
        typed_codecs = tuple(codecs)
        object.__setattr__(self, "codecs", typed_codecs)
        object.__setattr__(self, "_by_tool", MappingProxyType(by_tool))
        object.__setattr__(self, "_by_output_item", MappingProxyType(by_output))
        object.__setattr__(
            self,
            "_event_codecs",
            tuple(
                sorted(
                    typed_codecs,
                    key=lambda codec: len(codec.event_prefix),
                    reverse=True,
                )
            ),
        )

    @property
    def tools(self) -> frozenset[ProviderToolId]:
        return frozenset(self._by_tool)

    def codec_for_tool(self, tool: ProviderToolId) -> OpenAIResponsesProviderToolCodec:
        try:
            return self._by_tool[tool]
        except KeyError as exc:
            raise OpenAIResponsesError(
                f"Responses profile does not support provider tool: {tool.namespace}/{tool.type}"
            ) from exc

    def codec_for_output_item(self, item_type: str) -> OpenAIResponsesProviderToolCodec | None:
        return self._by_output_item.get(item_type)

    def codec_for_event(self, event_type: str) -> OpenAIResponsesProviderToolCodec | None:
        return next(
            (codec for codec in self._event_codecs if event_type.startswith(codec.event_prefix)),
            None,
        )

    def encode_declaration(self, spec: ProviderToolSpec) -> JsonObject:
        return self.codec_for_tool(spec.tool).encode_declaration(spec)

    def encode_choice(self, spec: ProviderToolSpec) -> JsonObject:
        return self.codec_for_tool(spec.tool).encode_choice(spec)

    def decode_call(
        self,
        item: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> ProviderToolCall | None:
        item_type = OPENAI_RESPONSES_JSON.required_string(
            item.get("type"),
            "Responses output item type",
        )
        codec = self.codec_for_output_item(item_type)
        return None if codec is None else codec.decode_call(item, response)

    def encode_history(self, call: ProviderToolCall) -> JsonObject:
        return self.codec_for_tool(call.tool).encode_history(call)

    def request_requires_artifact_store(self, request: ModelRequest) -> bool:
        return any(
            self.codec_for_tool(spec.tool).request_requires_artifact_store(spec)
            for spec in request.provider_tools
        )

    def history_requires_artifact_store(self, messages: Sequence[Message]) -> bool:
        return any(
            self.codec_for_tool(item.tool).history_requires_artifact_store(item)
            for message in messages
            if message.role == "assistant"
            for item in message.output
            if isinstance(item, ProviderToolCall)
        )

    def response_requires_artifact_store(self, response: ModelResponse) -> bool:
        return any(
            self.codec_for_tool(item.tool).response_requires_artifact_store(item)
            for item in response.output
            if isinstance(item, ProviderToolCall)
        )

    async def hydrate_artifact_history(
        self,
        request: ModelRequest,
        store: OpenAIResponsesArtifactStore,
        context: RunContext,
    ) -> ModelRequest:
        messages: list[Message] = []
        changed = False
        for message in request.messages:
            if message.role != "assistant":
                messages.append(message)
                continue
            output = await self._hydrate_output(message.output, store, context)
            if output == message.output:
                messages.append(message)
                continue
            changed = True
            messages.append(Message.assistant(output, metadata=message.metadata))
        return request if not changed else replace(request, messages=tuple(messages))

    async def externalize_artifacts(
        self,
        response: ModelResponse,
        store: OpenAIResponsesArtifactStore,
        context: RunContext,
    ) -> ModelResponse:
        output: list[ModelOutputItem] = []
        changed = False
        for item in response.output:
            if not isinstance(item, ProviderToolCall):
                output.append(item)
                continue
            externalized = await self.codec_for_tool(item.tool).externalize_call(
                item,
                store,
                context,
            )
            changed = changed or externalized is not item
            output.append(externalized)
        return response if not changed else replace(response, output=tuple(output))

    async def _hydrate_output(
        self,
        output: tuple[ModelOutputItem, ...],
        store: OpenAIResponsesArtifactStore,
        context: RunContext,
    ) -> tuple[ModelOutputItem, ...]:
        hydrated: list[ModelOutputItem] = []
        changed = False
        for item in output:
            if not isinstance(item, ProviderToolCall):
                hydrated.append(item)
                continue
            hydrated_call = await self.codec_for_tool(item.tool).hydrate_call(
                item,
                store,
                context,
            )
            changed = changed or hydrated_call is not item
            hydrated.append(hydrated_call)
        return output if not changed else tuple(hydrated)


def provider_status(value: object, *, label: str) -> ProviderToolStatus:
    """Project one Responses lifecycle value into the provider-neutral enum."""

    status = OPENAI_RESPONSES_JSON.required_string(value, label)
    if status == "completed":
        return ProviderToolStatus.COMPLETED
    if status == "incomplete":
        return ProviderToolStatus.INCOMPLETE
    if status == "failed":
        return ProviderToolStatus.FAILED
    if status in _IN_PROGRESS_STATUSES:
        return ProviderToolStatus.IN_PROGRESS
    raise OpenAIResponsesError(f"unsupported {label}: {status}")


def _validate_codec_shape(codec: OpenAIResponsesProviderToolCodec) -> frozenset[str]:
    raw_tool = cast(object, codec.tool)
    raw_output_item_type = cast(object, codec.output_item_type)
    raw_event_prefix = cast(object, codec.event_prefix)
    if not isinstance(raw_tool, ProviderToolId):
        raise TypeError("Responses provider tool codec requires a ProviderToolId")
    if not isinstance(raw_output_item_type, str) or not raw_output_item_type:
        raise ValueError("Responses provider output item type must be non-empty")
    if (
        not isinstance(raw_event_prefix, str)
        or not raw_event_prefix.startswith("response.")
        or not raw_event_prefix.endswith(".")
    ):
        raise ValueError("Responses provider event prefix must be a response.* namespace")
    return _nonempty_string_set(
        codec.declaration_types,
        "Responses provider declaration types",
    )


def is_terminal_provider_status(status: ProviderToolStatus | None) -> bool:
    return status in _PROVIDER_TERMINAL_STATUSES


def _configuration(
    spec: ProviderToolSpec,
    allowed_fields: frozenset[str],
) -> JsonObject:
    configuration = cast(JsonObject, thaw_json_value(spec.configuration))
    unexpected = set(configuration).difference(allowed_fields)
    if unexpected:
        key = min(unexpected)
        raise OpenAIResponsesError(f"unsupported {spec.tool.type} configuration field: {key}")
    return configuration


def _validate_tool_identity(tool: ProviderToolId, expected_type: str) -> None:
    raw_tool = cast(object, tool)
    if not isinstance(raw_tool, ProviderToolId):
        raise TypeError("Responses provider tool codec requires a ProviderToolId")
    if tool.type != expected_type:
        raise ValueError(f"Responses {expected_type} codec requires tool type={expected_type!r}")


def _require_spec_identity(spec: ProviderToolSpec, tool: ProviderToolId) -> None:
    if spec.tool != tool:
        raise OpenAIResponsesError("Responses provider tool spec does not match its codec")


def _require_call_identity(call: ProviderToolCall, tool: ProviderToolId) -> None:
    if call.tool != tool:
        raise OpenAIResponsesError("Responses provider tool call does not match its codec")


def _nonempty_string_set(
    value: object,
    label: str,
    *,
    allow_empty: bool = False,
) -> frozenset[str]:
    if not isinstance(value, frozenset):
        raise TypeError(f"{label} must be a frozenset")
    values = cast(frozenset[object], value)
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{label} must contain non-empty strings")
    if not values and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    return cast(frozenset[str], values)


def _validate_image_configuration(configuration: Mapping[str, Any]) -> None:
    partial_images = configuration.get("partial_images")
    if partial_images is not None and (
        isinstance(partial_images, bool)
        or not isinstance(partial_images, int)
        or not 0 <= partial_images <= 3
    ):
        raise OpenAIResponsesError("image_generation partial_images must be between 0 and 3")
    output_format = configuration.get("output_format")
    if output_format is not None and output_format not in {"png", "jpeg", "webp"}:
        raise OpenAIResponsesError("image_generation output_format must be png, jpeg, or webp")
    output_compression = configuration.get("output_compression")
    if output_compression is not None and (
        isinstance(output_compression, bool)
        or not isinstance(output_compression, int)
        or not 0 <= output_compression <= 100
    ):
        raise OpenAIResponsesError("image_generation output_compression must be between 0 and 100")


def _provider_error(item: Mapping[str, Any], tool_type: str) -> ErrorInfo:
    raw_error = item.get("error")
    if isinstance(raw_error, Mapping):
        error = cast(Mapping[str, object], raw_error)
        code_value = error.get("code")
        message_value = error.get("message")
        code = code_value if isinstance(code_value, str) and code_value else f"{tool_type}_failed"
        message = (
            message_value
            if isinstance(message_value, str) and message_value
            else f"provider {tool_type} call failed"
        )
        return ErrorInfo(code, message)
    if isinstance(raw_error, str) and raw_error:
        return ErrorInfo(f"{tool_type}_failed", raw_error)
    return ErrorInfo(f"{tool_type}_failed", f"provider {tool_type} call failed")


def _native_item(container: Mapping[str, Any]) -> JsonObject | None:
    raw_container = container.get("responses")
    if not isinstance(raw_container, Mapping):
        return None
    raw_item = cast(Mapping[str, object], raw_container).get("item")
    if not isinstance(raw_item, Mapping):
        return None
    return cast(JsonObject, thaw_json_value(cast(Mapping[str, object], raw_item)))


def _without_result(item: Mapping[str, Any]) -> JsonObject:
    return {key: value for key, value in item.items() if key != "result"}


def _resolve_image_media_type(image_base64: str, configured: str | None) -> str:
    inferred = _infer_image_media_type(image_base64)
    if configured is not None and inferred is not None and configured != inferred:
        raise OpenAIResponsesError(
            "Responses image result does not match the configured output format"
        )
    return configured or inferred or "image/png"


def _infer_image_media_type(image_base64: str) -> str | None:
    try:
        decoded = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise OpenAIResponsesError("Responses image data must contain valid base64") from exc
    if decoded.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if decoded.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(decoded) >= 12 and decoded.startswith(b"RIFF") and decoded[8:12] == b"WEBP":
        return "image/webp"
    return None


def _provider_event_status(suffix: str, label: str) -> ProviderToolStatus:
    if suffix in {"in_progress", "generating", "searching"}:
        return ProviderToolStatus.IN_PROGRESS
    if suffix == "completed":
        return ProviderToolStatus.COMPLETED
    if suffix == "incomplete":
        return ProviderToolStatus.INCOMPLETE
    if suffix == "failed":
        return ProviderToolStatus.FAILED
    raise OpenAIResponsesError(f"unsupported Responses {label} event: {suffix}")


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpenAIResponsesError(f"{label} must be a non-negative integer")
    return value


def _is_array(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)
