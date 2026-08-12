"""Provider-owned tool dialects for Anthropic-compatible Messages APIs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

from jharness.kernel import (
    ContentPart,
    ErrorInfo,
    ProviderToolCall,
    ProviderToolId,
    ProviderToolSpec,
    ProviderToolStatus,
    thaw_json_value,
)
from jharness.models.anthropic.messages.errors import (
    ANTHROPIC_MESSAGES_JSON,
    AnthropicMessagesError,
)

JsonObject = dict[str, Any]

_RESULT_PART_TYPE = "anthropic_server_tool_result"


@dataclass(frozen=True, slots=True)
class AnthropicMessagesServerToolCodec:
    """One immutable Anthropic server-tool declaration and lifecycle codec."""

    tool: ProviderToolId
    declaration_types: frozenset[str]
    default_declaration_type: str
    declaration_name: str
    call_names: frozenset[str]
    result_block_types: frozenset[str]
    configuration_fields: frozenset[str] = field(default_factory=frozenset[str])
    variant_configuration_fields: Mapping[str, frozenset[str]] = field(
        default_factory=lambda: dict[str, frozenset[str]]()
    )

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.tool), ProviderToolId):
            raise TypeError("Anthropic server tool identity must be a ProviderToolId")
        declaration_types = _string_set(
            self.declaration_types,
            "Anthropic server tool declaration types",
        )
        if self.default_declaration_type not in declaration_types:
            raise ValueError("default Anthropic server tool declaration type must be registered")
        _nonempty_string(self.declaration_name, "Anthropic server tool declaration name")
        call_names = _string_set(self.call_names, "Anthropic server tool call names")
        result_types = _string_set(
            self.result_block_types,
            "Anthropic server tool result block types",
        )
        configuration_fields = _string_set(
            self.configuration_fields,
            "Anthropic server tool configuration fields",
            allow_empty=True,
        )
        if "variant" not in configuration_fields and len(declaration_types) != 1:
            raise ValueError(
                "multiple Anthropic server tool declarations require the variant field"
            )
        object.__setattr__(self, "declaration_types", declaration_types)
        object.__setattr__(self, "call_names", call_names)
        object.__setattr__(self, "result_block_types", result_types)
        object.__setattr__(self, "configuration_fields", configuration_fields)
        variant_fields = {
            variant: _string_set(fields, f"{variant} configuration fields", allow_empty=True)
            for variant, fields in self.variant_configuration_fields.items()
        }
        unknown_variants = set(variant_fields).difference(declaration_types)
        if unknown_variants:
            raise ValueError("variant configuration fields require a registered declaration type")
        object.__setattr__(self, "variant_configuration_fields", MappingProxyType(variant_fields))

    def encode_declaration(self, spec: ProviderToolSpec) -> JsonObject:
        """Validate and encode one provider-tool declaration."""

        self._require_spec(spec)
        unsupported = set(spec.configuration).difference(self.configuration_fields)
        if unsupported:
            key = min(unsupported)
            raise AnthropicMessagesError(f"unsupported {self.tool.type} configuration field: {key}")
        configuration = cast(JsonObject, thaw_json_value(spec.configuration))
        variant = configuration.pop("variant", self.default_declaration_type)
        if not isinstance(variant, str) or variant not in self.declaration_types:
            expected = ", ".join(sorted(self.declaration_types))
            raise AnthropicMessagesError(f"{self.tool.type} variant must be one of: {expected}")
        allowed_for_variant = self.variant_configuration_fields.get(variant)
        if allowed_for_variant is not None:
            unsupported = set(configuration).difference(allowed_for_variant)
            if unsupported:
                key = min(unsupported)
                raise AnthropicMessagesError(f"unsupported {variant} configuration field: {key}")
        return {
            "type": variant,
            "name": self.declaration_name,
            **configuration,
        }

    def encode_choice(self, spec: ProviderToolSpec) -> JsonObject:
        """Encode an exact provider-tool choice."""

        self._require_spec(spec)
        return {"type": "tool", "name": self.declaration_name}

    def decode_call(
        self,
        use: Mapping[str, Any] | None,
        result: Mapping[str, Any] | None,
    ) -> ProviderToolCall:
        """Combine one server use and its optional provider result."""

        if use is None and result is None:
            raise ValueError("Anthropic server tool decoding requires a use or result block")
        call_id = self._call_id(use, result)
        call_name: str | None = None
        arguments: Mapping[str, Any] = {}
        use_extra: JsonObject = {}
        if use is not None:
            if use.get("type") != "server_tool_use":
                raise AnthropicMessagesError(
                    "Anthropic server tool use requires type='server_tool_use'"
                )
            call_name = ANTHROPIC_MESSAGES_JSON.required_string(
                use.get("name"),
                "Anthropic server tool use name",
            )
            if call_name not in self.call_names:
                raise AnthropicMessagesError(
                    f"Anthropic server tool name does not match {self.tool.type}: {call_name}"
                )
            raw_input = use.get("input", {})
            if not isinstance(raw_input, Mapping):
                raise AnthropicMessagesError("Anthropic server tool input must be an object")
            arguments = cast(Mapping[str, Any], raw_input)
            use_extra = {
                key: thaw_json_value(value)
                for key, value in use.items()
                if key not in {"type", "id", "name", "input"}
            }
        output: tuple[ContentPart, ...] = ()
        error: ErrorInfo | None = None
        if result is None:
            status = ProviderToolStatus.IN_PROGRESS
        else:
            result_type = ANTHROPIC_MESSAGES_JSON.required_string(
                result.get("type"),
                "Anthropic server tool result type",
            )
            if result_type not in self.result_block_types:
                raise AnthropicMessagesError(
                    f"Anthropic server tool result does not match {self.tool.type}: {result_type}"
                )
            result_content = result.get("content")
            error = _result_error(result_content, self.tool.type)
            status = (
                ProviderToolStatus.FAILED if error is not None else ProviderToolStatus.COMPLETED
            )
            result_extra = {
                key: thaw_json_value(value)
                for key, value in result.items()
                if key not in {"type", "tool_use_id", "content"}
            }
            output = (
                ContentPart(
                    type=_RESULT_PART_TYPE,
                    data={
                        "anthropic": {
                            "type": result_type,
                            "content": thaw_json_value(result_content),
                            **result_extra,
                        }
                    },
                ),
            )
        metadata: JsonObject = {
            "anthropic": {
                "server_tool_use": use is not None,
            }
        }
        native = cast(JsonObject, metadata["anthropic"])
        if call_name is not None:
            native["name"] = call_name
        if use_extra:
            native["use_extra"] = use_extra
        return ProviderToolCall(
            id=call_id,
            tool=self.tool,
            status=status,
            arguments=arguments,
            output=output,
            error=error,
            metadata=metadata,
        )

    def encode_history(self, call: ProviderToolCall) -> list[JsonObject]:
        """Rebuild the exact assistant blocks needed for stateless replay."""

        self._require_call(call)
        native = _native_metadata(call)
        blocks: list[JsonObject] = []
        if native.get("server_tool_use") is True:
            name = native.get("name", self.declaration_name)
            if not isinstance(name, str) or name not in self.call_names:
                raise AnthropicMessagesError(
                    "Anthropic server tool history has an invalid call name"
                )
            use_extra = native.get("use_extra", {})
            if not isinstance(use_extra, Mapping):
                raise AnthropicMessagesError("Anthropic server tool use metadata must be an object")
            blocks.append(
                {
                    "type": "server_tool_use",
                    "id": call.id,
                    "name": name,
                    "input": thaw_json_value(call.arguments),
                    **cast(
                        JsonObject,
                        thaw_json_value(cast(Mapping[str, object], use_extra)),
                    ),
                }
            )
        if call.output:
            if len(call.output) != 1 or call.output[0].type != _RESULT_PART_TYPE:
                raise AnthropicMessagesError(
                    "Anthropic server tool history requires exactly one native result part"
                )
            raw = call.output[0].data.get("anthropic")
            if not isinstance(raw, Mapping):
                raise AnthropicMessagesError(
                    "Anthropic server tool result part requires native data"
                )
            result = cast(
                JsonObject,
                thaw_json_value(cast(Mapping[str, object], raw)),
            )
            result_type = result.pop("type", None)
            if not isinstance(result_type, str) or result_type not in self.result_block_types:
                raise AnthropicMessagesError(
                    "Anthropic server tool history has an invalid result type"
                )
            content = result.pop("content", None)
            blocks.append(
                {
                    "type": result_type,
                    "tool_use_id": call.id,
                    "content": content,
                    **result,
                }
            )
        if not blocks:
            raise AnthropicMessagesError("Anthropic server tool history contains no native blocks")
        return blocks

    def _call_id(
        self,
        use: Mapping[str, Any] | None,
        result: Mapping[str, Any] | None,
    ) -> str:
        use_id = (
            None
            if use is None
            else ANTHROPIC_MESSAGES_JSON.required_string(
                use.get("id"),
                "Anthropic server tool use id",
            )
        )
        result_id = (
            None
            if result is None
            else ANTHROPIC_MESSAGES_JSON.required_string(
                result.get("tool_use_id"),
                "Anthropic server tool result tool_use_id",
            )
        )
        if use_id is not None and result_id is not None and use_id != result_id:
            raise AnthropicMessagesError(
                "Anthropic server tool result references a different use id"
            )
        return use_id or cast(str, result_id)

    def _require_spec(self, spec: ProviderToolSpec) -> None:
        if spec.tool != self.tool:
            raise AnthropicMessagesError("Anthropic server tool spec has the wrong identity")

    def _require_call(self, call: ProviderToolCall) -> None:
        if call.tool != self.tool:
            raise AnthropicMessagesError("Anthropic server tool call has the wrong identity")


@dataclass(frozen=True, slots=True)
class AnthropicMessagesServerToolRegistry:
    """Immutable bidirectional registry for one Messages profile."""

    codecs: tuple[AnthropicMessagesServerToolCodec, ...] = ()
    _by_tool: Mapping[ProviderToolId, AnthropicMessagesServerToolCodec] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _by_call_name: Mapping[str, AnthropicMessagesServerToolCodec] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _by_result_type: Mapping[str, AnthropicMessagesServerToolCodec] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        raw_codecs = cast(object, self.codecs)
        if not isinstance(raw_codecs, tuple):
            raise TypeError("Anthropic server tool codecs must be a tuple")
        codecs = cast(tuple[object, ...], raw_codecs)
        by_tool: dict[ProviderToolId, AnthropicMessagesServerToolCodec] = {}
        by_call_name: dict[str, AnthropicMessagesServerToolCodec] = {}
        by_result_type: dict[str, AnthropicMessagesServerToolCodec] = {}
        declaration_types: set[str] = set()
        validated: list[AnthropicMessagesServerToolCodec] = []
        for raw_codec in codecs:
            if not isinstance(raw_codec, AnthropicMessagesServerToolCodec):
                raise TypeError("Anthropic server tool registry values must be server tool codecs")
            codec = raw_codec
            if codec.tool in by_tool:
                raise ValueError("Anthropic server tool identities must be unique")
            if declaration_types.intersection(codec.declaration_types):
                raise ValueError("Anthropic server tool declaration types must be unique")
            for name in codec.call_names:
                if name in by_call_name:
                    raise ValueError("Anthropic server tool call names must be unique")
                by_call_name[name] = codec
            for result_type in codec.result_block_types:
                if result_type in by_result_type:
                    raise ValueError("Anthropic server tool result types must be unique")
                by_result_type[result_type] = codec
            by_tool[codec.tool] = codec
            declaration_types.update(codec.declaration_types)
            validated.append(codec)
        object.__setattr__(self, "codecs", tuple(validated))
        object.__setattr__(self, "_by_tool", MappingProxyType(by_tool))
        object.__setattr__(self, "_by_call_name", MappingProxyType(by_call_name))
        object.__setattr__(self, "_by_result_type", MappingProxyType(by_result_type))

    @property
    def tools(self) -> frozenset[ProviderToolId]:
        return frozenset(self._by_tool)

    def codec_for_tool(self, tool: ProviderToolId) -> AnthropicMessagesServerToolCodec:
        try:
            return self._by_tool[tool]
        except KeyError as exc:
            raise AnthropicMessagesError(
                "Anthropic Messages profile does not support provider tool: "
                f"{tool.namespace}/{tool.type}"
            ) from exc

    def codec_for_call_name(self, name: str) -> AnthropicMessagesServerToolCodec | None:
        return self._by_call_name.get(name)

    def codec_for_result_type(self, result_type: str) -> AnthropicMessagesServerToolCodec | None:
        return self._by_result_type.get(result_type)

    def encode_declaration(self, spec: ProviderToolSpec) -> JsonObject:
        return self.codec_for_tool(spec.tool).encode_declaration(spec)

    def encode_choice(self, spec: ProviderToolSpec) -> JsonObject:
        return self.codec_for_tool(spec.tool).encode_choice(spec)

    def encode_history(self, call: ProviderToolCall) -> list[JsonObject]:
        return self.codec_for_tool(call.tool).encode_history(call)


def anthropic_messages_web_search_codec(
    tool: ProviderToolId,
    *,
    variants: frozenset[str] = frozenset({"web_search_20250305"}),
    default_variant: str = "web_search_20250305",
) -> AnthropicMessagesServerToolCodec:
    """Build the complete Anthropic web-search server-tool codec."""

    common_fields = frozenset(
        {"allowed_callers", "allowed_domains", "blocked_domains", "max_uses", "user_location"}
    )
    return AnthropicMessagesServerToolCodec(
        tool=tool,
        declaration_types=variants,
        default_declaration_type=default_variant,
        declaration_name="web_search",
        call_names=frozenset({"web_search"}),
        result_block_types=frozenset({"web_search_tool_result"}),
        configuration_fields=frozenset(
            {
                "allowed_callers",
                "allowed_domains",
                "blocked_domains",
                "max_uses",
                "response_inclusion",
                "user_location",
                "variant",
            }
        ),
        variant_configuration_fields={
            variant: common_fields
            | (
                frozenset({"response_inclusion"})
                if variant >= "web_search_20260318"
                else frozenset[str]()
            )
            for variant in variants
        },
    )


def _native_metadata(call: ProviderToolCall) -> Mapping[str, Any]:
    raw = call.metadata.get("anthropic")
    if not isinstance(raw, Mapping):
        raise AnthropicMessagesError("Anthropic server tool history requires native metadata")
    return cast(Mapping[str, Any], raw)


def _result_error(value: object, tool_type: str) -> ErrorInfo | None:
    candidates: Sequence[object]
    if isinstance(value, Mapping):
        candidates = (cast(Mapping[str, object], value),)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        candidates = cast(Sequence[object], value)
    else:
        return None
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        candidate_mapping = cast(Mapping[str, object], candidate)
        candidate_type = candidate_mapping.get("type")
        code = candidate_mapping.get("error_code")
        if not (
            isinstance(candidate_type, str)
            and candidate_type.endswith("_error")
            and isinstance(code, str)
            and code
        ):
            continue
        raw_message = candidate_mapping.get("error_message")
        message = raw_message if isinstance(raw_message, str) and raw_message else code
        return ErrorInfo(code=f"{tool_type}.{code}", message=message)
    return None


def _string_set(value: object, label: str, *, allow_empty: bool = False) -> frozenset[str]:
    if not isinstance(value, frozenset):
        raise TypeError(f"{label} must be a frozenset")
    values = cast(frozenset[object], value)
    if not values and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{label} must contain non-empty strings")
    return cast(frozenset[str], values)


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value
