"""Message and ordered-output conversion for compatible Responses APIs."""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from jharness.kernel import (
    ContentPart,
    FreeformToolCall,
    Message,
    ModelOutputItem,
    RuntimeToolKind,
    StructuredToolCall,
    thaw_json_value,
)
from jharness.models.openai.errors import OPENAI_RESPONSES_JSON, OpenAIResponsesError
from jharness.models.openai.profiles import OpenAIResponsesProfile

JsonObject = dict[str, Any]


def encode_responses_input(
    messages: Sequence[Message],
    profile: OpenAIResponsesProfile,
) -> list[JsonObject]:
    """Encode durable kernel history as ordered Responses input items."""

    items: list[JsonObject] = []
    pending_calls: dict[str, RuntimeToolKind] = {}
    for message in messages:
        if message.role == "tool":
            if message.tool_call_id is None:
                raise OpenAIResponsesError("tool messages require tool_call_id")
            try:
                input_kind = pending_calls.pop(message.tool_call_id)
            except KeyError as exc:
                raise OpenAIResponsesError(
                    "Responses tool output has no preceding runtime tool call"
                ) from exc
            items.append(_encode_runtime_tool_output(message, input_kind, profile))
        elif message.role == "assistant":
            for item in message.output:
                if isinstance(item, StructuredToolCall | FreeformToolCall):
                    if item.id in pending_calls:
                        raise OpenAIResponsesError(
                            "Responses history contains duplicate pending tool call ids"
                        )
                    pending_calls[item.id] = (
                        RuntimeToolKind.STRUCTURED
                        if isinstance(item, StructuredToolCall)
                        else RuntimeToolKind.FREEFORM
                    )
            items.extend(_encode_assistant_output(message.output, profile))
        else:
            items.append(_encode_regular_message(message, profile))
    return items


def decode_output_items(
    value: object,
    profile: OpenAIResponsesProfile,
    *,
    response: Mapping[str, Any],
) -> list[ModelOutputItem]:
    """Decode the response output array without losing provider item order."""

    if not _is_array(value):
        raise OpenAIResponsesError("Responses output must be an array")
    output: list[ModelOutputItem] = []
    for raw_item in cast(Sequence[object], value):
        item = OPENAI_RESPONSES_JSON.mapping(raw_item, "Responses output item")
        item_type = _required_type(item, "Responses output item")
        if item_type == "message":
            output.extend(_decode_message_item(item))
        elif item_type == "reasoning":
            output.append(_decode_reasoning_item(item, profile))
        elif item_type == "function_call":
            output.append(_decode_structured_tool_call(item, profile))
        elif item_type == "custom_tool_call":
            output.append(_decode_freeform_tool_call(item, profile))
        else:
            provider_call = profile.provider_tool_registry.decode_call(item, response)
            if provider_call is None:
                raise OpenAIResponsesError(f"unsupported Responses output item: {item_type}")
            output.append(provider_call)
    return output


def _encode_regular_message(
    message: Message,
    profile: OpenAIResponsesProfile,
) -> JsonObject:
    role = "user" if message.role == "external" else message.role
    if role not in {"system", "user"}:
        raise OpenAIResponsesError(f"unsupported Responses message role: {message.role}")
    return {
        "type": "message",
        "role": role,
        "content": [_encode_input_part(part, profile) for part in message.parts],
    }


def _encode_runtime_tool_output(
    message: Message,
    input_kind: RuntimeToolKind,
    profile: OpenAIResponsesProfile,
) -> JsonObject:
    if message.tool_call_id is None or message.outcome is None:
        raise OpenAIResponsesError("tool messages require tool_call_id and outcome")
    parts = message.outcome.parts
    if all(part.type == "text" for part in parts):
        output: str | list[JsonObject] = "".join(part.text or "" for part in parts)
    else:
        output = [_encode_input_part(part, profile) for part in parts]
    if input_kind is RuntimeToolKind.FREEFORM and not isinstance(output, str):
        raise OpenAIResponsesError("Responses custom tool output supports text content only")
    return {
        "type": (
            "function_call_output"
            if input_kind is RuntimeToolKind.STRUCTURED
            else "custom_tool_call_output"
        ),
        "call_id": message.tool_call_id,
        "output": output,
    }


def _encode_input_part(
    part: ContentPart,
    profile: OpenAIResponsesProfile,
) -> JsonObject:
    if part.type == "text":
        _require_modality(profile, "text")
        return {"type": "input_text", "text": part.text or ""}
    if part.type == "image":
        return _encode_image_input(part, profile)
    if part.type == "artifact":
        return _encode_artifact_input(part, profile)
    if part.type == "file":
        return _encode_file_input(part, profile)
    raise OpenAIResponsesError(f"unsupported Responses input content part: {part.type}")


def _encode_image_input(
    part: ContentPart,
    profile: OpenAIResponsesProfile,
) -> JsonObject:
    _require_modality(profile, "image")
    image_base64 = part.data.get("base64")
    if part.uri is not None:
        if image_base64 is not None:
            raise OpenAIResponsesError(
                "Responses image input cannot carry both uri and base64 data"
            )
        return {"type": "input_image", "image_url": part.uri}
    if not isinstance(image_base64, str) or not image_base64:
        raise OpenAIResponsesError("Responses image input requires uri or base64 data")
    media_type = _resolve_image_media_type(image_base64, part.media_type)
    if not media_type.startswith("image/"):
        raise OpenAIResponsesError("Responses image input media_type must be image/*")
    return {
        "type": "input_image",
        "image_url": f"data:{media_type};base64,{image_base64}",
    }


def _encode_artifact_input(
    part: ContentPart,
    profile: OpenAIResponsesProfile,
) -> JsonObject:
    if part.artifact is None:
        raise OpenAIResponsesError("Responses artifact input requires an artifact")
    if (part.artifact.media_type or "").startswith("image/"):
        _require_modality(profile, "image")
        return {"type": "input_image", "file_id": part.artifact.ref}
    _require_modality(profile, "file")
    return {"type": "input_file", "file_id": part.artifact.ref}


def _encode_file_input(
    part: ContentPart,
    profile: OpenAIResponsesProfile,
) -> JsonObject:
    _require_modality(profile, "file")
    if part.uri is None:
        raise OpenAIResponsesError("Responses file input requires a uri")
    field = "file_data" if part.uri.startswith("data:") else "file_url"
    encoded: JsonObject = {"type": "input_file", field: part.uri}
    if part.name is not None:
        encoded["filename"] = part.name
    return encoded


def _encode_assistant_output(
    output: Sequence[ModelOutputItem],
    profile: OpenAIResponsesProfile,
) -> list[JsonObject]:
    encoded: list[JsonObject] = []
    message_parts: list[JsonObject] = []
    message_header: JsonObject | None = None

    def flush_message() -> None:
        nonlocal message_header
        if message_parts:
            message = {} if message_header is None else message_header
            message.update(
                type="message",
                role="assistant",
                content=list(message_parts),
            )
            encoded.append(message)
            message_parts.clear()
            message_header = None

    for item in output:
        if isinstance(item, ContentPart) and item.type in {"text", "refusal"}:
            header = _native_message_header(item)
            if message_parts and header != message_header:
                flush_message()
            message_header = header
            message_parts.append(_encode_assistant_part(item))
            continue
        flush_message()
        if isinstance(item, ContentPart):
            encoded.append(_encode_reasoning_part(item, profile))
        elif isinstance(item, StructuredToolCall):
            encoded.append(_encode_structured_tool_call(item))
        elif isinstance(item, FreeformToolCall):
            encoded.append(_encode_freeform_tool_call(item))
        else:
            encoded.append(profile.provider_tool_registry.encode_history(item))
    flush_message()
    return encoded


def _encode_assistant_part(part: ContentPart) -> JsonObject:
    raw = _native_content_part(part)
    if part.type == "text":
        if raw is not None:
            if raw.get("type") != "output_text":
                raise OpenAIResponsesError("Responses text metadata must contain output_text")
            encoded = raw
            encoded["text"] = part.text or ""
            return encoded
        return {"type": "output_text", "text": part.text or "", "annotations": []}
    if part.type != "refusal":
        raise OpenAIResponsesError(f"unsupported Responses assistant content: {part.type}")
    if raw is not None:
        if raw.get("type") != "refusal":
            raise OpenAIResponsesError("Responses refusal data must contain refusal content")
        encoded = raw
        encoded["refusal"] = part.text or ""
        return encoded
    if not part.text:
        raise OpenAIResponsesError("Responses refusal content requires non-empty text")
    return {"type": "refusal", "refusal": part.text}


def _encode_reasoning_part(
    part: ContentPart,
    profile: OpenAIResponsesProfile,
) -> JsonObject:
    if part.type != "reasoning":
        raise OpenAIResponsesError(f"unsupported Responses assistant content: {part.type}")
    raw = _native_item(part.data)
    if raw is not None:
        if raw.get("type") != "reasoning":
            raise OpenAIResponsesError("Responses reasoning data must contain a reasoning item")
        _validate_stateless_reasoning(raw, profile)
        return raw
    if profile.store is False:
        raise OpenAIResponsesError(
            "store=False reasoning history requires provider encrypted_content"
        )
    if part.text is None:
        raise OpenAIResponsesError("Responses reasoning content requires text or native data")
    if profile.reasoning_history_mode == "content":
        return {
            "type": "reasoning",
            "content": [{"type": "reasoning_text", "text": part.text}],
        }
    return {
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": part.text}],
    }


def _encode_structured_tool_call(call: StructuredToolCall) -> JsonObject:
    return {
        "type": "function_call",
        "call_id": call.id,
        "name": call.name,
        "arguments": json.dumps(
            thaw_json_value(call.arguments),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    }


def _encode_freeform_tool_call(call: FreeformToolCall) -> JsonObject:
    return {
        "type": "custom_tool_call",
        "call_id": call.id,
        "name": call.name,
        "input": call.input,
    }


def _decode_message_item(item: Mapping[str, Any]) -> list[ContentPart]:
    if item.get("role") != "assistant":
        raise OpenAIResponsesError("Responses output message requires role='assistant'")
    content = item.get("content")
    if not _is_array(content):
        raise OpenAIResponsesError("Responses output message content must be an array")
    header = {key: value for key, value in item.items() if key != "content"}
    parts: list[ContentPart] = []
    for raw_part in cast(Sequence[object], content):
        block = OPENAI_RESPONSES_JSON.mapping(raw_part, "Responses message content part")
        block_type = _required_type(block, "Responses message content part")
        native = {"item": header, "content": dict(block)}
        if block_type == "output_text":
            text = block.get("text")
            if not isinstance(text, str):
                raise OpenAIResponsesError("Responses output_text requires text")
            annotations = block.get("annotations")
            if annotations is not None and not _is_array(annotations):
                raise OpenAIResponsesError("Responses output_text annotations must be an array")
            parts.append(ContentPart.text_part(text, metadata={"responses": native}))
        elif block_type == "refusal":
            refusal = block.get("refusal")
            if not isinstance(refusal, str) or not refusal:
                raise OpenAIResponsesError("Responses refusal requires non-empty refusal text")
            parts.append(
                ContentPart(
                    type="refusal",
                    text=refusal,
                    data={"responses": native},
                )
            )
        else:
            raise OpenAIResponsesError(
                f"unsupported Responses assistant content part: {block_type}"
            )
    if not parts:
        raise OpenAIResponsesError("Responses output message requires content")
    return parts


def _decode_reasoning_item(
    item: Mapping[str, Any],
    profile: OpenAIResponsesProfile,
) -> ContentPart:
    _validate_stateless_reasoning(item, profile)
    chunks: list[str] = []
    for field, block_type in (("content", "reasoning_text"), ("summary", "summary_text")):
        raw_blocks = item.get(field)
        if raw_blocks is None:
            continue
        if not _is_array(raw_blocks):
            raise OpenAIResponsesError(f"Responses reasoning {field} must be an array")
        for raw_block in cast(Sequence[object], raw_blocks):
            block = OPENAI_RESPONSES_JSON.mapping(raw_block, f"Responses reasoning {field} part")
            if block.get("type") != block_type or not isinstance(block.get("text"), str):
                raise OpenAIResponsesError(
                    f"Responses reasoning {field} requires {block_type} parts"
                )
            chunks.append(cast(str, block["text"]))
    return ContentPart(
        type="reasoning",
        text="".join(chunks),
        data={"responses": {"item": dict(item)}},
    )


def _decode_structured_tool_call(
    item: Mapping[str, Any],
    profile: OpenAIResponsesProfile,
) -> StructuredToolCall:
    _require_runtime_kind(profile, RuntimeToolKind.STRUCTURED)
    status = item.get("status")
    if status is not None:
        status = OPENAI_RESPONSES_JSON.required_string(
            status,
            "Responses function call status",
        )
        if status != "completed":
            raise OpenAIResponsesError(
                "Responses function call must be completed before runtime execution"
            )
    call_id = OPENAI_RESPONSES_JSON.required_string(
        item.get("call_id"),
        "Responses function call call_id",
    )
    name = OPENAI_RESPONSES_JSON.required_string(
        item.get("name"),
        "Responses function call name",
    )
    raw_arguments = OPENAI_RESPONSES_JSON.required_string(
        item.get("arguments"),
        "Responses function call arguments",
    )
    try:
        arguments: object = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise OpenAIResponsesError("Responses function call arguments must be valid JSON") from exc
    if not isinstance(arguments, Mapping):
        raise OpenAIResponsesError("Responses function call arguments must be an object")
    return StructuredToolCall(call_id, name, cast(Mapping[str, Any], arguments))


def _decode_freeform_tool_call(
    item: Mapping[str, Any],
    profile: OpenAIResponsesProfile,
) -> FreeformToolCall:
    _require_runtime_kind(profile, RuntimeToolKind.FREEFORM)
    status = item.get("status")
    if status is not None:
        status = OPENAI_RESPONSES_JSON.required_string(
            status,
            "Responses custom tool call status",
        )
        if status != "completed":
            raise OpenAIResponsesError(
                "Responses custom tool call must be completed before runtime execution"
            )
    call_id = OPENAI_RESPONSES_JSON.required_string(
        item.get("call_id"),
        "Responses custom tool call call_id",
    )
    name = OPENAI_RESPONSES_JSON.required_string(
        item.get("name"),
        "Responses custom tool call name",
    )
    if not profile.allows_freeform_runtime_tool(name):
        raise OpenAIResponsesError(f"{profile.name} does not support freeform runtime tool: {name}")
    raw_input = item.get("input")
    if not isinstance(raw_input, str):
        raise OpenAIResponsesError("Responses custom tool call input must be a string")
    return FreeformToolCall(call_id, name, raw_input)


def _native_content_part(part: ContentPart) -> JsonObject | None:
    container = _native_content_container(part)
    if not isinstance(container, Mapping):
        return None
    raw = cast(Mapping[str, object], container).get("content")
    if not isinstance(raw, Mapping):
        return None
    return cast(JsonObject, thaw_json_value(cast(Mapping[str, object], raw)))


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


def _native_message_header(part: ContentPart) -> JsonObject | None:
    container = _native_content_container(part)
    if not isinstance(container, Mapping):
        return None
    raw = cast(Mapping[str, object], container).get("item")
    if not isinstance(raw, Mapping):
        return None
    header = cast(JsonObject, thaw_json_value(cast(Mapping[str, object], raw)))
    header.pop("content", None)
    return header


def _native_content_container(part: ContentPart) -> object:
    if part.type == "text":
        return part.metadata.get("responses")
    return part.data.get("responses")


def _native_item(container: Mapping[str, Any]) -> JsonObject | None:
    raw_container = container.get("responses")
    if not isinstance(raw_container, Mapping):
        return None
    raw_item = cast(Mapping[str, object], raw_container).get("item")
    if not isinstance(raw_item, Mapping):
        return None
    return cast(JsonObject, thaw_json_value(cast(Mapping[str, object], raw_item)))


def _validate_stateless_reasoning(
    item: Mapping[str, Any],
    profile: OpenAIResponsesProfile,
) -> None:
    if profile.store is not False:
        return
    encrypted = item.get("encrypted_content")
    if not isinstance(encrypted, str) or not encrypted:
        raise OpenAIResponsesError(
            "store=False reasoning items require non-empty encrypted_content"
        )


def _require_modality(profile: OpenAIResponsesProfile, modality: str) -> None:
    if modality not in profile.capabilities.input_modalities:
        raise OpenAIResponsesError(f"{profile.name} does not support {modality} input")


def _require_runtime_kind(
    profile: OpenAIResponsesProfile,
    kind: RuntimeToolKind,
) -> None:
    if kind not in profile.capabilities.runtime_tool_kinds:
        raise OpenAIResponsesError(f"{profile.name} does not support {kind.value} runtime tools")


def _required_type(value: Mapping[str, Any], label: str) -> str:
    return OPENAI_RESPONSES_JSON.required_string(value.get("type"), f"{label} type")


def _is_array(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)
