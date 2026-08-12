"""Streaming conversion for Anthropic Messages."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from jharness.kernel import (
    ContentPart,
    ModelContentDelta,
    ModelDelta,
    ModelProviderToolCallDelta,
    ModelReasoningDelta,
    ModelResponse,
    ModelRuntimeToolCallDelta,
    ModelUsage,
    ModelUsageDelta,
    ProviderToolCall,
    ProviderToolStatus,
    RuntimeToolKind,
    StructuredToolCall,
)
from jharness.models.anthropic.messages.codec import decode_usage
from jharness.models.anthropic.messages.errors import (
    ANTHROPIC_MESSAGES_JSON,
    AnthropicMessagesError,
)
from jharness.models.anthropic.messages.profile import AnthropicMessagesProfile
from jharness.models.anthropic.messages.server_tools import AnthropicMessagesServerToolCodec

JsonObject = dict[str, Any]


@dataclass(slots=True)
class _BlockState:
    block_type: str
    output_index: int
    populated: bool = False
    closed: bool = False
    text_chunks: list[str] = field(default_factory=list[str])
    input_chunks: list[str] = field(default_factory=list[str])
    native_data: JsonObject = field(default_factory=dict[str, Any])
    call_id: str | None = None
    name: str | None = None
    server_codec: AnthropicMessagesServerToolCodec | None = None


class AnthropicMessagesStreamDecoder:
    """Decode one strict Anthropic Messages SSE lifecycle."""

    def __init__(self, profile: AnthropicMessagesProfile) -> None:
        self._profile = profile
        self._finish_reason: str | None = None
        self._raw_stop_reason: str | None = None
        self._model: str | None = None
        self._response_id: str | None = None
        self._usage: ModelUsage | None = None
        self._blocks: dict[int, _BlockState] = {}
        self._output: dict[int, ContentPart | StructuredToolCall | ProviderToolCall] = {}
        self._provider_uses: dict[str, JsonObject] = {}
        self._provider_output_indexes: dict[str, int] = {}
        self._next_output_index = 0
        self._phase: Literal["initial", "active", "delta_seen", "stopped"] = "initial"

    def apply_event(
        self,
        event_name: str | None,
        value: Mapping[str, Any],
    ) -> tuple[bool, list[ModelDelta]]:
        if self._phase == "stopped":
            raise AnthropicMessagesError("Anthropic stream emitted an event after message_stop")
        event_type = _event_type(event_name, value)
        if event_type == "ping":
            return False, []
        if event_type == "message_start":
            return False, self._message_start_events(value)
        if event_type == "content_block_start":
            return False, self._content_block_start_events(value)
        if event_type == "content_block_delta":
            return False, self._content_block_delta_events(value)
        if event_type == "content_block_stop":
            self._content_block_stop(value)
            return False, []
        if event_type == "message_delta":
            return False, self._message_delta_events(value)
        if event_type == "message_stop":
            self._message_stop()
            return True, []
        if event_type == "error":
            raise AnthropicMessagesError("Anthropic stream error event")
        raise AnthropicMessagesError(f"unsupported Anthropic stream event type: {event_type}")

    def completed_response(self) -> ModelResponse:
        if self._phase != "stopped":
            raise AnthropicMessagesError("Anthropic stream completed before message_stop")
        output = tuple(item for _, item in sorted(self._output.items()))
        pending = self._raw_stop_reason == "pause_turn" or any(
            isinstance(item, ProviderToolCall) and item.status is ProviderToolStatus.IN_PROGRESS
            for item in output
        )
        try:
            return ModelResponse(
                output=output,
                finish_reason=self._finish_reason,
                usage=self._usage,
                model_id=self._model,
                response_id=self._response_id,
                provider_turn_pending=pending,
                metadata={
                    "provider": self._profile.name,
                    "type": "message",
                    "role": "assistant",
                },
            )
        except (TypeError, ValueError) as exc:
            raise AnthropicMessagesError(
                f"Anthropic stream produced an invalid response: {exc}"
            ) from exc

    def _message_start_events(self, value: Mapping[str, Any]) -> list[ModelDelta]:
        if self._phase != "initial":
            raise AnthropicMessagesError("Anthropic stream message_start appeared more than once")
        self._phase = "active"
        message = ANTHROPIC_MESSAGES_JSON.mapping(value.get("message"), "Anthropic stream message")
        if message.get("type") != "message":
            raise AnthropicMessagesError("Anthropic stream message_start requires type='message'")
        if message.get("role") != "assistant":
            raise AnthropicMessagesError("Anthropic stream message_start requires role='assistant'")
        content = message.get("content")
        if not isinstance(content, Sequence) or isinstance(content, str | bytes | bytearray):
            raise AnthropicMessagesError("Anthropic stream message_start content must be an array")
        if content:
            raise AnthropicMessagesError("Anthropic stream message_start content must be empty")
        if message.get("id") is not None:
            self._response_id = ANTHROPIC_MESSAGES_JSON.required_string(
                message.get("id"), "Anthropic stream message id"
            )
        if message.get("model") is not None:
            self._model = ANTHROPIC_MESSAGES_JSON.required_string(
                message.get("model"), "Anthropic stream model"
            )
        return self._usage_events(message.get("usage"))

    def _content_block_start_events(self, value: Mapping[str, Any]) -> list[ModelDelta]:
        self._require_active("content_block_start")
        wire_index = _event_index(value)
        if wire_index in self._blocks:
            raise AnthropicMessagesError(
                f"Anthropic content block index started more than once: {wire_index}"
            )
        block = ANTHROPIC_MESSAGES_JSON.mapping(
            value.get("content_block"), "Anthropic content block"
        )
        block_type = _required_type(
            block.get("type"), "Anthropic content block requires non-empty type"
        )
        result_codec = self._profile.server_tools.codec_for_result_type(block_type)
        if result_codec is not None:
            return self._server_result_start(wire_index, block, result_codec)
        output_index = self._allocate_output_index()
        state = _BlockState(block_type=block_type, output_index=output_index)
        self._blocks[wire_index] = state
        if block_type in {"text", "thinking"}:
            field_name = "text" if block_type == "text" else "thinking"
            initial = block.get(field_name)
            if not isinstance(initial, str):
                raise AnthropicMessagesError(f"Anthropic {block_type} block requires {field_name}")
            state.text_chunks.append(initial)
            state.populated = bool(initial)
            if block_type == "thinking":
                state.native_data = {
                    key: value for key, value in block.items() if key not in {"type", "thinking"}
                }
            return _text_deltas(output_index, initial, block_type)
        if block_type == "redacted_thinking":
            data = ANTHROPIC_MESSAGES_JSON.required_string(
                block.get("data"), "Anthropic redacted_thinking data"
            )
            state.populated = True
            state.native_data = dict(block)
            self._output[output_index] = ContentPart(
                type="redacted_thinking",
                data={"anthropic": dict(block)},
            )
            return [
                ModelContentDelta(
                    output_index=output_index,
                    text_delta="",
                    part_type="redacted_thinking",
                    data={"anthropic": {"type": "redacted_thinking", "data": data}},
                )
            ]
        if block_type == "tool_use":
            return self._runtime_tool_start(state, block)
        if block_type == "server_tool_use":
            return self._server_tool_start(state, block)
        raise AnthropicMessagesError(f"unsupported Anthropic stream content block: {block_type}")

    def _runtime_tool_start(
        self,
        state: _BlockState,
        block: Mapping[str, Any],
    ) -> list[ModelDelta]:
        state.call_id = ANTHROPIC_MESSAGES_JSON.required_string(
            block.get("id"), "Anthropic tool_use id"
        )
        state.name = ANTHROPIC_MESSAGES_JSON.required_string(
            block.get("name"), "Anthropic tool_use name"
        )
        initial = _initial_input(block.get("input", {}), "Anthropic tool_use input")
        state.input_chunks.append(initial)
        state.populated = True
        return [
            ModelRuntimeToolCallDelta(
                output_index=state.output_index,
                input_kind=RuntimeToolKind.STRUCTURED,
                input_delta=initial,
                id=state.call_id,
                name=state.name,
            )
        ]

    def _server_tool_start(
        self,
        state: _BlockState,
        block: Mapping[str, Any],
    ) -> list[ModelDelta]:
        call_id = ANTHROPIC_MESSAGES_JSON.required_string(
            block.get("id"), "Anthropic server tool use id"
        )
        name = ANTHROPIC_MESSAGES_JSON.required_string(
            block.get("name"), "Anthropic server tool use name"
        )
        codec = self._profile.server_tools.codec_for_call_name(name)
        if codec is None:
            raise AnthropicMessagesError(f"unsupported Anthropic server tool call: {name}")
        if call_id in self._provider_output_indexes:
            raise AnthropicMessagesError(f"duplicate Anthropic server tool use id: {call_id}")
        initial = _initial_input(block.get("input", {}), "Anthropic server tool input")
        state.call_id = call_id
        state.name = name
        state.server_codec = codec
        state.input_chunks.append(initial)
        state.populated = True
        use = dict(block)
        use["input"] = {}
        self._provider_uses[call_id] = use
        self._provider_output_indexes[call_id] = state.output_index
        self._output[state.output_index] = codec.decode_call(use, None)
        return [
            ModelProviderToolCallDelta(
                output_index=state.output_index,
                id=call_id,
                tool=codec.tool,
                status=ProviderToolStatus.IN_PROGRESS,
                event="input.started",
                data={"name": name, "input_delta": initial},
            )
        ]

    def _server_result_start(
        self,
        wire_index: int,
        block: Mapping[str, Any],
        codec: AnthropicMessagesServerToolCodec,
    ) -> list[ModelDelta]:
        call_id = ANTHROPIC_MESSAGES_JSON.required_string(
            block.get("tool_use_id"),
            "Anthropic server tool result tool_use_id",
        )
        output_index = self._provider_output_indexes.get(call_id)
        use = self._provider_uses.get(call_id)
        if output_index is None:
            output_index = self._allocate_output_index()
            self._provider_output_indexes[call_id] = output_index
        elif use is not None:
            name = ANTHROPIC_MESSAGES_JSON.required_string(
                use.get("name"), "Anthropic server tool use name"
            )
            use_codec = self._profile.server_tools.codec_for_call_name(name)
            if use_codec is not codec:
                raise AnthropicMessagesError(
                    "Anthropic server tool result belongs to a different tool"
                )
        call = codec.decode_call(use, block)
        self._output[output_index] = call
        self._blocks[wire_index] = _BlockState(
            block_type=cast(str, block["type"]),
            output_index=output_index,
            populated=True,
            server_codec=codec,
            call_id=call_id,
        )
        return [
            ModelProviderToolCallDelta(
                output_index=output_index,
                id=call_id,
                tool=codec.tool,
                status=call.status,
                event="result",
                data={"result": dict(block)},
            )
        ]

    def _content_block_delta_events(self, value: Mapping[str, Any]) -> list[ModelDelta]:
        self._require_active("content_block_delta")
        wire_index = _event_index(value)
        state = self._open_block(wire_index, "delta")
        delta = ANTHROPIC_MESSAGES_JSON.mapping(value.get("delta"), "Anthropic content block delta")
        delta_type = _required_type(
            delta.get("type"),
            "Anthropic content block delta requires non-empty type",
        )
        if delta_type == "text_delta":
            return self._text_delta(state, delta, "text", "text")
        if delta_type == "thinking_delta":
            return self._text_delta(state, delta, "thinking", "thinking")
        if delta_type == "signature_delta":
            if state.block_type != "thinking":
                raise AnthropicMessagesError("Anthropic signature_delta requires a thinking block")
            signature = delta.get("signature")
            if not isinstance(signature, str):
                raise AnthropicMessagesError("Anthropic signature delta requires signature")
            if not signature:
                return []
            state.native_data["signature"] = signature
            return [
                ModelContentDelta(
                    output_index=state.output_index,
                    text_delta="",
                    part_type="thinking",
                    data={
                        "anthropic": {
                            "type": "thinking",
                            "signature": signature,
                        }
                    },
                )
            ]
        if delta_type == "citations_delta":
            return self._citation_delta(state, delta)
        if delta_type == "input_json_delta":
            return self._input_delta(state, delta)
        raise AnthropicMessagesError(
            f"unsupported Anthropic content block delta type: {delta_type}"
        )

    def _text_delta(
        self,
        state: _BlockState,
        delta: Mapping[str, Any],
        block_type: str,
        field_name: str,
    ) -> list[ModelDelta]:
        if state.block_type != block_type:
            raise AnthropicMessagesError(
                f"Anthropic {field_name}_delta does not match {state.block_type} block"
            )
        text = delta.get(field_name)
        if not isinstance(text, str):
            raise AnthropicMessagesError(f"Anthropic {field_name} delta requires {field_name}")
        state.text_chunks.append(text)
        state.populated = state.populated or bool(text)
        return _text_deltas(state.output_index, text, block_type)

    def _citation_delta(
        self,
        state: _BlockState,
        delta: Mapping[str, Any],
    ) -> list[ModelDelta]:
        if state.block_type != "text":
            raise AnthropicMessagesError("Anthropic citations_delta requires a text block")
        citation = delta.get("citation")
        if not isinstance(citation, Mapping):
            raise AnthropicMessagesError("Anthropic citations_delta requires citation")
        citation_mapping = cast(Mapping[str, object], citation)
        citations = cast(object, state.native_data.setdefault("citations", []))
        if not isinstance(citations, list):
            raise AnthropicMessagesError("Anthropic text citation state must be an array")
        cast(list[object], citations).append(dict(citation_mapping))
        state.populated = True
        return [
            ModelContentDelta(
                output_index=state.output_index,
                text_delta="",
                data={"anthropic": {"citation": dict(citation_mapping)}},
            )
        ]

    def _input_delta(
        self,
        state: _BlockState,
        delta: Mapping[str, Any],
    ) -> list[ModelDelta]:
        partial = delta.get("partial_json")
        if not isinstance(partial, str):
            raise AnthropicMessagesError("Anthropic input JSON delta requires partial_json")
        if state.block_type not in {"tool_use", "server_tool_use"}:
            raise AnthropicMessagesError(
                f"Anthropic input_json_delta does not match {state.block_type} block"
            )
        state.input_chunks.append(partial)
        state.populated = True
        if not partial:
            return []
        if state.block_type == "tool_use":
            return [
                ModelRuntimeToolCallDelta(
                    output_index=state.output_index,
                    input_kind=RuntimeToolKind.STRUCTURED,
                    input_delta=partial,
                )
            ]
        if state.call_id is None or state.server_codec is None:
            raise AnthropicMessagesError("Anthropic server tool input delta lacks call identity")
        return [
            ModelProviderToolCallDelta(
                output_index=state.output_index,
                id=state.call_id,
                tool=state.server_codec.tool,
                status=ProviderToolStatus.IN_PROGRESS,
                event="input.delta",
                data={"input_delta": partial},
            )
        ]

    def _content_block_stop(self, value: Mapping[str, Any]) -> None:
        self._require_active("content_block_stop")
        wire_index = _event_index(value)
        state = self._open_block(wire_index, "stop")
        if not state.populated:
            raise AnthropicMessagesError(
                f"Anthropic {state.block_type} content block completed without data"
            )
        if state.block_type in {"text", "thinking"}:
            text = "".join(state.text_chunks)
            if not text and not (
                state.block_type == "thinking" and state.native_data.get("signature")
            ):
                raise AnthropicMessagesError(
                    f"Anthropic {state.block_type} content block requires content before stop"
                )
            if state.block_type == "text":
                metadata = (
                    {"anthropic": {"extra": dict(state.native_data)}} if state.native_data else None
                )
                self._output[state.output_index] = ContentPart.text_part(
                    text,
                    metadata=metadata,
                )
            else:
                self._output[state.output_index] = ContentPart(
                    type="thinking",
                    text=text,
                    data={
                        "anthropic": {
                            "type": "thinking",
                            "thinking": text,
                            **state.native_data,
                        }
                    },
                )
        elif state.block_type == "tool_use":
            self._commit_runtime_tool(state)
        elif state.block_type == "server_tool_use":
            self._commit_server_tool(state)
        state.closed = True

    def _commit_runtime_tool(self, state: _BlockState) -> None:
        if state.call_id is None or state.name is None:
            raise AnthropicMessagesError("Anthropic streamed tool call lacks identity")
        arguments = _parse_input(state.input_chunks, "Anthropic streamed tool input")
        self._output[state.output_index] = StructuredToolCall(
            id=state.call_id,
            name=state.name,
            arguments=arguments,
        )

    def _commit_server_tool(self, state: _BlockState) -> None:
        if state.call_id is None or state.server_codec is None:
            raise AnthropicMessagesError("Anthropic streamed server tool call lacks identity")
        arguments = _parse_input(
            state.input_chunks,
            "Anthropic streamed server tool input",
        )
        use = self._provider_uses[state.call_id]
        use["input"] = arguments
        self._output[state.output_index] = state.server_codec.decode_call(use, None)

    def _message_delta_events(self, value: Mapping[str, Any]) -> list[ModelDelta]:
        self._require_active("message_delta")
        if any(not state.closed for state in self._blocks.values()):
            raise AnthropicMessagesError(
                "Anthropic message_delta requires all content blocks to stop"
            )
        delta = ANTHROPIC_MESSAGES_JSON.mapping(value.get("delta"), "Anthropic message delta")
        stop_reason = ANTHROPIC_MESSAGES_JSON.required_string(
            delta.get("stop_reason"), "Anthropic message_delta stop_reason"
        )
        if self._raw_stop_reason is not None:
            raise AnthropicMessagesError(
                "Anthropic stream emitted more than one terminal message_delta"
            )
        self._raw_stop_reason = stop_reason
        self._finish_reason = self._profile.finish_reason(stop_reason)
        self._phase = "delta_seen"
        return self._usage_events(value.get("usage"))

    def _message_stop(self) -> None:
        if self._phase == "initial":
            raise AnthropicMessagesError("Anthropic message_stop requires message_start")
        if self._phase != "delta_seen" or self._finish_reason is None:
            raise AnthropicMessagesError("Anthropic message_stop requires a terminal message_delta")
        if not self._output:
            raise AnthropicMessagesError("Anthropic stream completed without output")
        self._phase = "stopped"

    def _usage_events(self, value: object) -> list[ModelDelta]:
        if self._profile.stream_usage_mode == "omit":
            return []
        usage = decode_usage(value)
        if usage is None:
            return []
        self._usage = _merge_usage(self._usage, usage)
        return [ModelUsageDelta(usage=self._usage)]

    def _allocate_output_index(self) -> int:
        output_index = self._next_output_index
        self._next_output_index += 1
        return output_index

    def _open_block(self, wire_index: int, event: str) -> _BlockState:
        state = self._blocks.get(wire_index)
        if state is None or state.closed:
            raise AnthropicMessagesError(
                f"Anthropic content block {event} requires an open index: {wire_index}"
            )
        return state

    def _require_active(self, event_type: str) -> None:
        if self._phase == "initial":
            raise AnthropicMessagesError(f"Anthropic {event_type} requires message_start")
        if self._phase == "delta_seen":
            raise AnthropicMessagesError(f"Anthropic {event_type} appeared after message_delta")


def _text_deltas(output_index: int, text: str, part_type: str) -> list[ModelDelta]:
    if not text:
        return []
    content = ModelContentDelta(
        output_index=output_index,
        text_delta=text,
        part_type=part_type,
    )
    if part_type == "text":
        return [content]
    return [ModelReasoningDelta(output_index=output_index, text_delta=text), content]


def _initial_input(value: object, label: str) -> str:
    if not isinstance(value, Mapping):
        raise AnthropicMessagesError(f"{label} must be an object")
    if not value:
        return ""
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _parse_input(chunks: Sequence[str], label: str) -> Mapping[str, Any]:
    raw = "".join(chunks) or "{}"
    try:
        value: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AnthropicMessagesError(f"{label} must be valid JSON") from exc
    if not isinstance(value, Mapping):
        raise AnthropicMessagesError(f"{label} must be a JSON object")
    return cast(Mapping[str, Any], value)


def _merge_usage(existing: ModelUsage | None, update: ModelUsage) -> ModelUsage:
    if existing is None:
        return update
    input_tokens = _prefer(update.input_tokens, existing.input_tokens)
    output_tokens = _prefer(update.output_tokens, existing.output_tokens)
    total_tokens = _prefer(update.total_tokens, existing.total_tokens)
    if input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        reasoning_tokens=_prefer(update.reasoning_tokens, existing.reasoning_tokens),
        cache_read_tokens=_prefer(update.cache_read_tokens, existing.cache_read_tokens),
        cache_write_tokens=_prefer(update.cache_write_tokens, existing.cache_write_tokens),
    )


def _prefer(updated: int | None, existing: int | None) -> int | None:
    return updated if updated is not None else existing


def _event_type(event_name: str | None, value: Mapping[str, Any]) -> str:
    raw_type = value.get("type")
    if not isinstance(raw_type, str) or not raw_type:
        raise AnthropicMessagesError("Anthropic stream event payload requires a type")
    if event_name is not None and event_name != raw_type:
        raise AnthropicMessagesError("Anthropic stream event name must match the payload type")
    return raw_type


def _event_index(value: Mapping[str, Any]) -> int:
    index = value.get("index")
    if isinstance(index, bool) or not isinstance(index, int):
        raise AnthropicMessagesError("Anthropic stream event index must be an integer")
    if index < 0:
        raise AnthropicMessagesError("Anthropic stream event index must be >= 0")
    return index


def _required_type(value: object, error_message: str) -> str:
    if not isinstance(value, str) or not value:
        raise AnthropicMessagesError(error_message)
    return value
