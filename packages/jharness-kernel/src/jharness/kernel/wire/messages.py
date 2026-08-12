"""Explicit codecs for portable messages and their nested values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from jharness.kernel.errors import ProtocolError
from jharness.kernel.messages import (
    ArtifactRef,
    ContentPart,
    ErrorInfo,
    FreeformToolCall,
    Message,
    ModelOutputItem,
    ProviderToolCall,
    ProviderToolId,
    ProviderToolStatus,
    RuntimeToolCall,
    StructuredToolCall,
    TaskRef,
)
from jharness.kernel.tools import ToolAccepted, ToolFailure, ToolOutcome, ToolSuccess, ToolWaiting
from jharness.kernel.wire._helpers import (
    array,
    decode_document,
    integer,
    json_object,
    object_fields,
    string,
    thaw_object,
    thaw_value,
)

__all__ = [
    "decode_content_part",
    "decode_error_info",
    "decode_message",
    "decode_model_output_item_value",
    "decode_runtime_tool_call",
    "decode_tool_outcome",
    "encode_content_part",
    "encode_error_info",
    "encode_message",
    "encode_model_output_item",
    "encode_runtime_tool_call",
    "encode_tool_outcome",
]

_REGULAR_ROLES = frozenset({"system", "user", "external"})


def encode_message(message: Message) -> dict[str, Any]:
    """Encode one provider-neutral conversation message."""

    if message.role in _REGULAR_ROLES:
        return {
            "role": message.role,
            "parts": [encode_content_part(part) for part in message.parts],
            "metadata": thaw_object(message.metadata),
        }
    if message.role == "assistant":
        return {
            "role": "assistant",
            "output": [encode_model_output_item(item) for item in message.output],
            "metadata": thaw_object(message.metadata),
        }
    if message.outcome is None or message.tool_call_id is None:
        raise ProtocolError("invalid trusted tool message")
    return {
        "role": "tool",
        "tool_call_id": message.tool_call_id,
        "outcome": encode_tool_outcome(message.outcome),
        "metadata": thaw_object(message.metadata),
    }


def decode_message(value: object) -> Message:
    """Decode one portable conversation message."""

    return decode_document(value, "message", decode_message_value)


def decode_message_value(value: object) -> Message:
    mapping = json_object(value, "message")
    role = string(mapping.get("role"), "message role")
    if role in _REGULAR_ROLES:
        fields = object_fields(mapping, "message", {"role", "parts", "metadata"})
        return Message(
            role,
            _decode_parts(fields["parts"], "message parts"),
            metadata=json_object(fields["metadata"], "message metadata"),
        )
    if role == "assistant":
        fields = object_fields(
            mapping,
            "assistant message",
            {"role", "output", "metadata"},
        )
        output = tuple(
            decode_model_output_item_value(item)
            for item in array(fields["output"], "assistant output")
        )
        calls = tuple(
            item for item in output if isinstance(item, RuntimeToolCall | ProviderToolCall)
        )
        if len({call.id for call in calls}) != len(calls):
            raise ProtocolError(
                "assistant tool call ids must be unique",
                code="duplicate_tool_call_id",
            )
        return Message.assistant(
            output,
            metadata=json_object(fields["metadata"], "message metadata"),
        )
    if role == "tool":
        fields = object_fields(
            mapping,
            "tool message",
            {"role", "tool_call_id", "outcome", "metadata"},
        )
        return Message.tool(
            string(fields["tool_call_id"], "tool_call_id", non_empty=True),
            decode_tool_outcome_value(fields["outcome"]),
            metadata=json_object(fields["metadata"], "message metadata"),
        )
    raise ProtocolError(f"message role has unsupported value: {role}")


def encode_content_part(part: ContentPart) -> dict[str, Any]:
    """Encode one text, artifact, or opaque content part."""

    if part.type == "text":
        return {
            "type": "text",
            "text": cast(str, part.text),
            "metadata": thaw_object(part.metadata),
        }
    if part.type == "artifact":
        artifact = part.artifact
        if artifact is None:
            raise ProtocolError("invalid trusted artifact content part")
        return {
            "type": "artifact",
            "artifact": _encode_artifact(artifact),
            "metadata": thaw_object(part.metadata),
        }
    encoded: dict[str, Any] = {
        "type": part.type,
        "metadata": thaw_object(part.metadata),
    }
    encoded.update(
        {
            key: value
            for key, value in (
                ("text", part.text),
                ("uri", part.uri),
                ("media_type", part.media_type),
                ("name", part.name),
            )
            if value is not None
        }
    )
    if part.data:
        encoded["data"] = thaw_object(part.data)
    return encoded


def decode_content_part(value: object) -> ContentPart:
    """Decode one portable content part."""

    return decode_document(value, "content part", decode_content_part_value)


def decode_content_part_value(value: object) -> ContentPart:
    mapping = json_object(value, "content part")
    part_type = string(mapping.get("type"), "content part type", non_empty=True)
    if part_type == "text":
        fields = object_fields(mapping, "text part", {"type", "text", "metadata"})
        return ContentPart.text_part(
            string(fields["text"], "text part text"),
            metadata=json_object(fields["metadata"], "content metadata"),
        )
    if part_type == "artifact":
        fields = object_fields(mapping, "artifact part", {"type", "artifact", "metadata"})
        return ContentPart.artifact_part(
            _decode_artifact(fields["artifact"]),
            metadata=json_object(fields["metadata"], "content metadata"),
        )
    fields = object_fields(
        mapping,
        "opaque part",
        {"type", "metadata"},
        {"text", "uri", "media_type", "name", "data"},
    )
    opaque_data: Mapping[str, Any] = (
        {} if "data" not in fields else json_object(fields["data"], "opaque part data")
    )
    if "data" in fields and not opaque_data:
        raise ProtocolError("opaque part data must not be empty when present")
    return ContentPart(
        type=part_type,
        text=_present_string(fields, "text", "opaque part text"),
        uri=_present_string(fields, "uri", "opaque part uri", non_empty=True),
        media_type=_present_string(
            fields,
            "media_type",
            "opaque part media_type",
            non_empty=True,
        ),
        name=_present_string(fields, "name", "opaque part name", non_empty=True),
        data=opaque_data,
        metadata=json_object(fields["metadata"], "content metadata"),
    )


def _encode_artifact(artifact: ArtifactRef) -> dict[str, Any]:
    encoded: dict[str, Any] = {
        "ref": artifact.ref,
        "metadata": thaw_object(artifact.metadata),
    }
    encoded.update(
        {
            key: value
            for key, value in (
                ("media_type", artifact.media_type),
                ("name", artifact.name),
                ("size_bytes", artifact.size_bytes),
                ("sha256", artifact.sha256),
            )
            if value is not None
        }
    )
    return encoded


def _decode_artifact(value: object) -> ArtifactRef:
    fields = object_fields(
        value,
        "artifact",
        {"ref", "metadata"},
        {"media_type", "name", "size_bytes", "sha256"},
    )
    return ArtifactRef(
        ref=string(fields["ref"], "artifact ref", non_empty=True),
        media_type=_present_string(
            fields,
            "media_type",
            "artifact media_type",
            non_empty=True,
        ),
        name=_present_string(fields, "name", "artifact name", non_empty=True),
        size_bytes=(
            None
            if "size_bytes" not in fields
            else integer(fields["size_bytes"], "artifact size_bytes", minimum=0)
        ),
        sha256=_present_string(fields, "sha256", "artifact sha256", non_empty=True),
        metadata=json_object(fields["metadata"], "artifact metadata"),
    )


def encode_runtime_tool_call(call: RuntimeToolCall) -> dict[str, Any]:
    """Encode one model-requested runtime tool call."""

    common: dict[str, Any] = {
        "input_kind": "structured" if isinstance(call, StructuredToolCall) else "freeform",
        "id": call.id,
        "name": call.name,
    }
    if isinstance(call, StructuredToolCall):
        common["arguments"] = thaw_object(call.arguments)
    else:
        common["input"] = call.input
    return common


def decode_runtime_tool_call(value: object) -> RuntimeToolCall:
    """Decode one portable runtime tool call."""

    return decode_document(value, "runtime tool call", decode_runtime_tool_call_value)


def decode_runtime_tool_call_value(value: object) -> RuntimeToolCall:
    mapping = json_object(value, "runtime tool call")
    input_kind = string(mapping.get("input_kind"), "runtime tool call input_kind")
    if input_kind == "structured":
        fields = object_fields(
            mapping,
            "structured runtime tool call",
            {"input_kind", "id", "name", "arguments"},
        )
        return StructuredToolCall(
            id=string(fields["id"], "tool call id", non_empty=True),
            name=string(fields["name"], "tool call name", non_empty=True),
            arguments=json_object(fields["arguments"], "tool call arguments"),
        )
    if input_kind == "freeform":
        fields = object_fields(
            mapping,
            "freeform runtime tool call",
            {"input_kind", "id", "name", "input"},
        )
        return FreeformToolCall(
            id=string(fields["id"], "tool call id", non_empty=True),
            name=string(fields["name"], "tool call name", non_empty=True),
            input=string(fields["input"], "tool call input"),
        )
    raise ProtocolError(f"runtime tool call input_kind has unsupported value: {input_kind}")


def encode_model_output_item(item: ModelOutputItem) -> dict[str, Any]:
    """Encode one ordered model output item with an explicit discriminator."""

    if isinstance(item, ContentPart):
        return {"kind": "content", "content": encode_content_part(item)}
    if isinstance(item, RuntimeToolCall):
        return {"kind": "runtime_tool_call", **encode_runtime_tool_call(item)}
    return {
        "kind": "provider_tool_call",
        "id": item.id,
        "tool": {
            "namespace": item.tool.namespace,
            "type": item.tool.type,
        },
        "status": item.status.value,
        "arguments": thaw_object(item.arguments),
        "output": [encode_content_part(part) for part in item.output],
        "error": None if item.error is None else encode_error_info(item.error),
        "metadata": thaw_object(item.metadata),
    }


def decode_model_output_item_value(value: object) -> ModelOutputItem:
    mapping = json_object(value, "model output item")
    kind = string(mapping.get("kind"), "model output item kind")
    if kind == "content":
        fields = object_fields(mapping, "content output item", {"kind", "content"})
        return decode_content_part_value(fields["content"])
    if kind == "runtime_tool_call":
        fields = dict(mapping)
        fields.pop("kind")
        return decode_runtime_tool_call_value(fields)
    if kind != "provider_tool_call":
        raise ProtocolError(f"model output item kind has unsupported value: {kind}")
    fields = object_fields(
        mapping,
        "provider tool call output item",
        {
            "kind",
            "id",
            "tool",
            "status",
            "arguments",
            "output",
            "error",
            "metadata",
        },
    )
    tool_fields = object_fields(
        fields["tool"],
        "provider tool identity",
        {"namespace", "type"},
    )
    raw_status = string(fields["status"], "provider tool status")
    try:
        status = ProviderToolStatus(raw_status)
    except ValueError as exc:
        raise ProtocolError(f"provider tool status has unsupported value: {raw_status}") from exc
    raw_error = fields["error"]
    return ProviderToolCall(
        id=string(fields["id"], "provider tool call id", non_empty=True),
        tool=ProviderToolId(
            string(tool_fields["namespace"], "provider tool namespace", non_empty=True),
            string(tool_fields["type"], "provider tool type", non_empty=True),
        ),
        status=status,
        arguments=json_object(fields["arguments"], "provider tool arguments"),
        output=_decode_parts(fields["output"], "provider tool output"),
        error=None if raw_error is None else decode_error_info_value(raw_error),
        metadata=json_object(fields["metadata"], "provider tool metadata"),
    )


def encode_error_info(error: ErrorInfo) -> dict[str, str]:
    """Encode stable portable error details."""

    return {"code": error.code, "message": error.message}


def decode_error_info(value: object) -> ErrorInfo:
    """Decode stable portable error details."""

    return decode_document(value, "error info", decode_error_info_value)


def decode_error_info_value(value: object) -> ErrorInfo:
    fields = object_fields(value, "error info", {"code", "message"})
    return ErrorInfo(
        string(fields["code"], "error code", non_empty=True),
        string(fields["message"], "error message", non_empty=True),
    )


def encode_tool_outcome(outcome: ToolOutcome) -> dict[str, Any]:
    """Encode one model-visible tool outcome."""

    encoded: dict[str, Any] = {
        "kind": outcome.kind,
        "parts": [encode_content_part(part) for part in outcome.parts],
        "structured_content": thaw_value(outcome.structured_content),
    }
    if isinstance(outcome, ToolFailure):
        encoded["error"] = encode_error_info(outcome.error)
    elif isinstance(outcome, ToolAccepted):
        encoded["correlation_id"] = outcome.correlation_id
        encoded["task"] = None if outcome.task is None else _encode_task(outcome.task)
    elif isinstance(outcome, ToolWaiting):
        encoded["task"] = None if outcome.task is None else _encode_task(outcome.task)
    return encoded


def decode_tool_outcome(value: object) -> ToolOutcome:
    """Decode one portable model-visible tool outcome."""

    return decode_document(value, "tool outcome", decode_tool_outcome_value)


def decode_tool_outcome_value(value: object) -> ToolOutcome:
    mapping = json_object(value, "tool outcome")
    kind = string(mapping.get("kind"), "tool outcome kind")
    common = {"kind", "parts", "structured_content"}
    if kind == "success":
        fields = object_fields(mapping, "tool success", common)
        return ToolSuccess(
            _decode_parts(fields["parts"], "tool success parts"),
            fields["structured_content"],
        )
    if kind == "failure":
        fields = object_fields(mapping, "tool failure", {*common, "error"})
        return ToolFailure(
            _decode_parts(fields["parts"], "tool failure parts"),
            decode_error_info_value(fields["error"]),
            fields["structured_content"],
        )
    if kind == "accepted":
        fields = object_fields(
            mapping,
            "tool accepted",
            {*common, "correlation_id", "task"},
        )
        return ToolAccepted(
            _decode_parts(fields["parts"], "tool accepted parts"),
            string(fields["correlation_id"], "correlation_id", non_empty=True),
            _decode_nullable_task(fields["task"]),
            fields["structured_content"],
        )
    if kind == "waiting":
        fields = object_fields(mapping, "tool waiting", {*common, "task"})
        return ToolWaiting(
            _decode_parts(fields["parts"], "tool waiting parts"),
            _decode_nullable_task(fields["task"]),
            fields["structured_content"],
        )
    raise ProtocolError(f"tool outcome kind has unsupported value: {kind}")


def _encode_task(task: TaskRef) -> dict[str, Any]:
    return {
        "id": task.id,
        "status": task.status,
        "metadata": thaw_object(task.metadata),
    }


def _decode_nullable_task(value: object) -> TaskRef | None:
    return None if value is None else _decode_task(value)


def _decode_task(value: object) -> TaskRef:
    fields = object_fields(value, "task", {"id", "status", "metadata"})
    return TaskRef(
        id=string(fields["id"], "task id", non_empty=True),
        status=string(fields["status"], "task status", non_empty=True),
        metadata=json_object(fields["metadata"], "task metadata"),
    )


def _decode_parts(value: object, label: str) -> tuple[ContentPart, ...]:
    return tuple(decode_content_part_value(item) for item in array(value, label))


def _present_string(
    mapping: Mapping[str, Any],
    key: str,
    label: str,
    *,
    non_empty: bool = False,
) -> str | None:
    if key not in mapping:
        return None
    return string(mapping[key], label, non_empty=non_empty)
