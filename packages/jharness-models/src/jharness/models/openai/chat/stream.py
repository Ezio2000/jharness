"""Streaming conversion for OpenAI Chat Completions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeVar, cast

from jharness.kernel import (
    ModelContentDelta,
    ModelDelta,
    ModelResponse,
    ModelRuntimeToolCallDelta,
    ModelUsage,
    ModelUsageDelta,
    RuntimeToolKind,
)
from jharness.models._stream import DeltaAccumulator
from jharness.models.openai.chat.codec import (
    decode_usage,
    optional_service_tier,
    optional_string,
    reject_logprobs,
)
from jharness.models.openai.chat.errors import OPENAI_CHAT_JSON, OpenAIChatError
from jharness.models.openai.chat.profile import OpenAIChatProfile

_MetadataValue = TypeVar("_MetadataValue")
_ContentPartType = Literal["text", "refusal"]
_CONTENT_PART_RANK: dict[_ContentPartType, int] = {
    "text": 0,
    "refusal": 1,
}
_FINISH_REASONS = frozenset({"stop", "length", "tool_calls", "content_filter", "function_call"})
_CHUNK_FIELDS = {
    "id",
    "choices",
    "created",
    "model",
    "object",
    "moderation",
    "obfuscation",
    "service_tier",
    "system_fingerprint",
    "usage",
}


class OpenAIChatStreamDecoder:
    """Decode Chat Completions stream chunks into kernel stream events."""

    def __init__(
        self,
        profile: OpenAIChatProfile,
    ) -> None:
        self._profile = profile
        self._accumulator = DeltaAccumulator(OpenAIChatError)
        self._finish_reason: str | None = None
        self._model: str | None = None
        self._response_id: str | None = None
        self._usage: ModelUsage | None = None
        self._object: str | None = None
        self._created: int | None = None
        self._service_tier: str | None = None
        self._system_fingerprint: str | None = None
        self._obfuscation: list[str] = []
        self._content_part_indexes: dict[_ContentPartType, int] = {}
        self._last_content_part_rank = -1
        self._tool_output_offset: int | None = None
        self._phase: Literal["initial", "active", "finished"] = "initial"

    def apply_chunk(self, value: Mapping[str, Any]) -> list[ModelDelta]:
        self._capture_chunk_metadata(value)
        usage_event = self._capture_usage(value.get("usage"))
        deltas: list[ModelDelta] = [] if usage_event is None else [usage_event]
        choice = self._decode_choice(value.get("choices"), has_usage=usage_event is not None)
        if choice is not None:
            deltas.extend(self._apply_choice(choice))
        for delta in deltas:
            self._accumulator.apply(delta)
        return deltas

    def _capture_usage(self, value: object) -> ModelUsageDelta | None:
        usage = decode_usage(value)
        if usage is None:
            return None
        self._usage = usage if self._usage is None else self._usage.merge_snapshot(usage)
        return ModelUsageDelta(usage=self._usage)

    @staticmethod
    def _decode_choice(value: object, *, has_usage: bool) -> Mapping[str, Any] | None:
        if value is None:
            raise OpenAIChatError("chat completion stream choices must be an array")
        if not isinstance(value, list):
            raise OpenAIChatError("chat completion stream choices must be an array")
        raw_choices = cast(list[object], value)
        if not raw_choices:
            if not has_usage:
                raise OpenAIChatError("chat completion stream empty choices require usage")
            return None
        if len(raw_choices) != 1:
            raise OpenAIChatError("chat completion stream requires exactly one choice per chunk")
        choice = OPENAI_CHAT_JSON.mapping(raw_choices[0], "chat completion stream choice")
        _reject_unknown_fields(
            choice,
            {"index", "delta", "finish_reason", "logprobs"},
            "chat completion stream choice",
        )
        reject_logprobs(choice.get("logprobs"), "chat completion stream choice logprobs")
        return choice

    def _apply_choice(self, choice: Mapping[str, Any]) -> list[ModelDelta]:
        if self._phase == "finished":
            raise OpenAIChatError("chat completion stream emitted a choice after finish_reason")
        if _choice_index(choice) != 0:
            raise OpenAIChatError("chat completion stream choice index must be 0")
        self._phase = "active"
        delta = OPENAI_CHAT_JSON.mapping(choice.get("delta"), "chat completion stream delta")
        deltas = self._deltas_from_wire(delta)
        self._capture_finish_reason(choice.get("finish_reason"))
        return deltas

    def _capture_finish_reason(self, finish_reason: object) -> None:
        if finish_reason is None:
            return
        if not isinstance(finish_reason, str) or finish_reason not in _FINISH_REASONS:
            raise OpenAIChatError("chat completion stream finish_reason has an unsupported value")
        self._finish_reason = finish_reason
        self._phase = "finished"

    def completed_response(self) -> ModelResponse:
        if self._phase == "initial":
            raise OpenAIChatError("chat completion stream completed without a choice")
        if self._phase != "finished":
            raise OpenAIChatError("chat completion stream completed before finish_reason")
        metadata: dict[str, Any] = {
            "provider": self._profile.name,
            "choice_count": 1,
        }
        metadata["object"] = self._object
        metadata["created"] = self._created
        metadata["service_tier"] = self._service_tier
        metadata["system_fingerprint"] = self._system_fingerprint
        if self._obfuscation:
            metadata["obfuscation"] = list(self._obfuscation)
        if not self._accumulator.has_output:
            metadata["openai_chat"] = {"content_null": True}
        return self._accumulator.response(
            finish_reason=self._finish_reason,
            model_id=self._model,
            response_id=self._response_id,
            metadata=metadata,
        )

    def _capture_chunk_metadata(self, value: Mapping[str, Any]) -> None:
        _reject_unknown_fields(value, _CHUNK_FIELDS, "chat completion stream chunk")
        if value.get("object") != "chat.completion.chunk":
            raise OpenAIChatError("chat completion stream object must be 'chat.completion.chunk'")
        if "choices" not in value:
            raise OpenAIChatError("chat completion stream chunk requires choices")
        if value.get("moderation") is not None:
            raise OpenAIChatError("chat completion stream moderation is not supported")
        self._response_id = _consistent_metadata_value(
            self._response_id,
            _required_metadata_str(value.get("id"), "id"),
            "id",
        )
        self._model = _consistent_metadata_value(
            self._model,
            _required_metadata_str(value.get("model"), "model"),
            "model",
        )
        self._object = _consistent_metadata_value(
            self._object,
            "chat.completion.chunk",
            "object",
        )
        self._created = _consistent_metadata_value(
            self._created,
            _required_metadata_int(value.get("created"), "created"),
            "created",
        )
        self._service_tier = _consistent_metadata_value(
            self._service_tier,
            optional_service_tier(value.get("service_tier"), "service_tier"),
            "service_tier",
        )
        self._system_fingerprint = _consistent_metadata_value(
            self._system_fingerprint,
            optional_string(value.get("system_fingerprint"), "chat completion system_fingerprint"),
            "system_fingerprint",
        )
        obfuscation = optional_string(value.get("obfuscation"), "chat completion obfuscation")
        if obfuscation is not None:
            self._obfuscation.append(obfuscation)

    def _deltas_from_wire(self, delta: Mapping[str, Any]) -> list[ModelDelta]:
        _reject_unknown_fields(
            delta,
            {
                "role",
                "content",
                "refusal",
                "tool_calls",
                "reasoning_content",
                "function_call",
            },
            "chat completion stream delta",
        )
        _validate_delta_role(delta.get("role"))
        if "reasoning_content" in delta:
            raise OpenAIChatError("Chat Completions stream deltas do not support reasoning_content")
        if delta.get("function_call") is not None:
            raise OpenAIChatError("Chat Completions stream function_call is not supported")
        content_event = self._content_event(delta.get("content"))
        refusal_event = self._refusal_event(delta.get("refusal"))
        deltas: list[ModelDelta] = [
            item for item in (content_event, refusal_event) if item is not None
        ]
        tool_call_events = self._tool_call_events(delta.get("tool_calls"))
        deltas.extend(tool_call_events)
        return deltas

    def _content_event(self, value: object) -> ModelContentDelta | None:
        content = _optional_delta_text(value, "content")
        if content is None:
            return None
        return ModelContentDelta(
            output_index=self._content_part_index("text"),
            text_delta=content,
        )

    def _refusal_event(self, value: object) -> ModelContentDelta | None:
        refusal = _optional_delta_text(value, "refusal")
        if refusal is None:
            return None
        return ModelContentDelta(
            output_index=self._content_part_index("refusal"),
            text_delta=refusal,
            part_type="refusal",
        )

    def _content_part_index(self, part_type: _ContentPartType) -> int:
        if self._tool_output_offset is not None:
            raise OpenAIChatError("chat completion stream emitted content after tool calls")
        rank = _CONTENT_PART_RANK[part_type]
        if rank < self._last_content_part_rank:
            raise OpenAIChatError(
                f"chat completion stream emitted {part_type} after a later content part"
            )
        existing = self._content_part_indexes.get(part_type)
        if existing is not None:
            return existing
        index = len(self._content_part_indexes)
        self._content_part_indexes[part_type] = index
        self._last_content_part_rank = rank
        return index

    def _tool_call_events(self, value: object) -> list[ModelRuntimeToolCallDelta]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise OpenAIChatError("chat completion stream tool_calls must be an array")
        decoded = (self._tool_call_event(raw_call) for raw_call in cast(list[object], value))
        return [event for event in decoded if event is not None]

    def _tool_call_event(self, value: object) -> ModelRuntimeToolCallDelta | None:
        call = OPENAI_CHAT_JSON.mapping(value, "chat completion stream tool call")
        _reject_unknown_fields(
            call,
            {"index", "id", "type", "function"},
            "chat completion stream tool call",
        )
        raw_call_type = call.get("type")
        if raw_call_type is not None and raw_call_type != "function":
            raise OpenAIChatError(
                f"unsupported chat completion stream tool call type: {raw_call_type}"
            )
        call_index = _tool_call_index(call)
        nested = call.get("function")
        nested_mapping = (
            OPENAI_CHAT_JSON.mapping(nested, "chat completion stream tool function")
            if nested is not None
            else cast(Mapping[str, Any], {})
        )
        _reject_unknown_fields(
            nested_mapping,
            {"name", "arguments"},
            "chat completion stream tool function",
        )
        call_id = OPENAI_CHAT_JSON.optional_string(call.get("id"))
        name = OPENAI_CHAT_JSON.optional_string(nested_mapping.get("name"))
        input_delta = OPENAI_CHAT_JSON.optional_string(nested_mapping.get("arguments"))
        if call_id is None and name is None and input_delta is None:
            return None
        if self._tool_output_offset is None:
            self._tool_output_offset = len(self._content_part_indexes)
        return ModelRuntimeToolCallDelta(
            output_index=self._tool_output_offset + call_index,
            input_kind=RuntimeToolKind.STRUCTURED,
            id=call_id,
            name=name,
            input_delta=input_delta or "",
        )


def _validate_delta_role(role: object) -> None:
    if role is not None and role != "assistant":
        raise OpenAIChatError("chat completion stream delta role must be 'assistant'")


def _optional_delta_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise OpenAIChatError(f"chat completion stream {label} delta must be a string or null")
    return value


def _choice_index(choice: Mapping[str, Any]) -> int:
    if "index" not in choice:
        raise OpenAIChatError("chat completion stream choice requires an index")
    index = choice["index"]
    if isinstance(index, bool) or not isinstance(index, int):
        raise OpenAIChatError("chat completion stream choice index must be an integer")
    return index


def _tool_call_index(call: Mapping[str, Any]) -> int:
    index = call.get("index", 0)
    if isinstance(index, bool) or not isinstance(index, int):
        raise OpenAIChatError("chat completion stream tool call index must be an integer")
    if index < 0:
        raise OpenAIChatError("chat completion stream tool call index must be >= 0")
    return index


def _optional_metadata_str(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise OpenAIChatError(f"chat completion stream {label} must be a string or null")
    if not value:
        raise OpenAIChatError(f"chat completion stream {label} must not be empty")
    return value


def _required_metadata_str(value: object, label: str) -> str:
    result = _optional_metadata_str(value, label)
    if result is None:
        raise OpenAIChatError(f"chat completion stream {label} is required")
    return result


def _optional_metadata_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise OpenAIChatError(f"chat completion stream {label} must be an integer or null")
    return value


def _required_metadata_int(value: object, label: str) -> int:
    result = _optional_metadata_int(value, label)
    if result is None:
        raise OpenAIChatError(f"chat completion stream {label} is required")
    return result


def _consistent_metadata_value(
    existing: _MetadataValue | None,
    update: _MetadataValue | None,
    label: str,
) -> _MetadataValue | None:
    if update is None:
        return existing
    if existing is not None and update != existing:
        raise OpenAIChatError(f"chat completion stream {label} changed between chunks")
    return update


def _reject_unknown_fields(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unexpected = set(value) - allowed
    if unexpected:
        raise OpenAIChatError(f"{label} has unsupported fields: {', '.join(sorted(unexpected))}")
