"""Message and ordered-output conversion for the OpenAI Responses API."""

from __future__ import annotations

import base64
import binascii
import json
import math
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
from jharness.models.openai.responses.errors import OPENAI_RESPONSES_JSON, OpenAIResponsesError
from jharness.models.openai.responses.profile import OpenAIResponsesProfile
from jharness.models.openai.responses.provider_tools import (
    decode_provider_call,
    encode_provider_history,
)

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
            provider_call = decode_provider_call(item, response)
            if provider_call is None:
                raise OpenAIResponsesError(f"unsupported Responses output item: {item_type}")
            if provider_call.tool not in profile.capabilities.provider_tools:
                raise OpenAIResponsesError(
                    "Responses profile does not support provider tool: "
                    f"{provider_call.tool.namespace}/{provider_call.tool.type}"
                )
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
        if part.modality == "image":
            return _encode_image_input(part, profile)
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
        return {"type": "input_image", "image_url": part.uri, "detail": "auto"}
    if not isinstance(image_base64, str) or not image_base64:
        raise OpenAIResponsesError("Responses image input requires uri or base64 data")
    media_type = _resolve_image_media_type(image_base64, part.media_type)
    if not media_type.casefold().startswith("image/"):
        raise OpenAIResponsesError("Responses image input media_type must be image/*")
    return {
        "type": "input_image",
        "image_url": f"data:{media_type};base64,{image_base64}",
        "detail": "auto",
    }


def _encode_artifact_input(
    part: ContentPart,
    profile: OpenAIResponsesProfile,
) -> JsonObject:
    if part.artifact is None:
        raise OpenAIResponsesError("Responses artifact input requires an artifact")
    if part.modality == "image":
        _require_modality(profile, "image")
        return {"type": "input_image", "file_id": part.artifact.ref, "detail": "auto"}
    _require_modality(profile, "file")
    return {"type": "input_file", "file_id": part.artifact.ref}


def _encode_file_input(
    part: ContentPart,
    profile: OpenAIResponsesProfile,
) -> JsonObject:
    _require_modality(profile, "file")
    if part.uri is None:
        raise OpenAIResponsesError("Responses file input requires a uri")
    field = "file_data" if part.uri[:5].casefold() == "data:" else "file_url"
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
            if message_header is None:
                raise OpenAIResponsesError(
                    "Responses assistant history requires the native response output message"
                )
            message = message_header
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
            header = _required_native_message_header(item)
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
            encoded.append(encode_provider_history(item))
    flush_message()
    return encoded


def _encode_assistant_part(part: ContentPart) -> JsonObject:
    raw = _native_content_part(part)
    if part.type == "text":
        if raw is not None:
            encoded = _output_text_block(raw)
            encoded["text"] = part.text or ""
            return encoded
        raise OpenAIResponsesError("Responses text history requires native output_text metadata")
    if part.type != "refusal":
        raise OpenAIResponsesError(f"unsupported Responses assistant content: {part.type}")
    if raw is not None:
        encoded = _refusal_block(raw)
        encoded["refusal"] = part.text or ""
        return encoded
    raise OpenAIResponsesError("Responses refusal history requires native refusal metadata")


def _encode_reasoning_part(
    part: ContentPart,
    profile: OpenAIResponsesProfile,
) -> JsonObject:
    if part.type != "reasoning":
        raise OpenAIResponsesError(f"unsupported Responses assistant content: {part.type}")
    raw = _native_item(part.data)
    if raw is not None:
        return _replay_reasoning_item(raw, profile)
    raise OpenAIResponsesError(
        "Responses reasoning history requires the native provider reasoning item"
    )


def _encode_structured_tool_call(call: StructuredToolCall) -> JsonObject:
    raw = _native_runtime_tool_call(call.metadata, "function_call")
    encoded = (
        validate_runtime_tool_call_item(raw, "function_call")
        if raw is not None
        else {"type": "function_call"}
    )
    encoded["call_id"] = call.id
    encoded["name"] = call.name
    encoded["arguments"] = (
        call.raw_input
        if call.raw_input is not None
        else json.dumps(
            thaw_json_value(call.arguments),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return encoded


def _encode_freeform_tool_call(call: FreeformToolCall) -> JsonObject:
    raw = _native_runtime_tool_call(call.metadata, "custom_tool_call")
    encoded = (
        validate_runtime_tool_call_item(raw, "custom_tool_call")
        if raw is not None
        else {"type": "custom_tool_call"}
    )
    encoded["call_id"] = call.id
    encoded["name"] = call.name
    encoded["input"] = call.input
    return encoded


def _decode_message_item(item: Mapping[str, Any]) -> list[ContentPart]:
    native_item = validate_output_message_item(item)
    header = _response_output_message_header(native_item)
    content = cast(Sequence[object], native_item["content"])
    parts: list[ContentPart] = []
    for raw_part in content:
        block = OPENAI_RESPONSES_JSON.mapping(raw_part, "Responses message content part")
        block_type = _required_type(block, "Responses message content part")
        if block_type == "output_text":
            output_text = _output_text_block(block)
            text = cast(str, output_text["text"])
            native = {"item": header, "content": output_text}
            parts.append(ContentPart.text_part(text, metadata={"responses": native}))
        elif block_type == "refusal":
            refusal_block = _refusal_block(block)
            refusal = cast(str, refusal_block["refusal"])
            native = {"item": header, "content": refusal_block}
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
    return parts


def _decode_reasoning_item(
    item: Mapping[str, Any],
    profile: OpenAIResponsesProfile,
) -> ContentPart:
    native_item = validate_reasoning_item(item, profile, require_replay_state=True)
    chunks: list[str] = []
    for field, block_type in (("content", "reasoning_text"), ("summary", "summary_text")):
        raw_blocks = native_item.get(field)
        if raw_blocks is None:
            continue
        if not _is_array(raw_blocks):
            raise OpenAIResponsesError(f"Responses reasoning {field} must be an array")
        for raw_block in cast(Sequence[object], raw_blocks):
            block = _reasoning_block(raw_block, field=field, block_type=block_type)
            chunks.append(cast(str, block["text"]))
    return ContentPart(
        type="reasoning",
        text="".join(chunks),
        data={"responses": {"item": native_item}},
    )


def _decode_structured_tool_call(
    item: Mapping[str, Any],
    profile: OpenAIResponsesProfile,
) -> StructuredToolCall:
    _require_runtime_kind(profile, RuntimeToolKind.STRUCTURED)
    native_item = validate_runtime_tool_call_item(item, "function_call")
    call_id = OPENAI_RESPONSES_JSON.required_string(
        native_item.get("call_id"),
        "Responses function call call_id",
    )
    name = OPENAI_RESPONSES_JSON.required_string(
        native_item.get("name"),
        "Responses function call name",
    )
    raw_arguments = native_item.get("arguments")
    if not isinstance(raw_arguments, str):
        raise OpenAIResponsesError("Responses function call arguments must be a string")
    try:
        arguments: object = json.loads(raw_arguments)
    except json.JSONDecodeError:
        arguments = None
    metadata = {"responses": {"item": native_item}}
    if isinstance(arguments, Mapping):
        return StructuredToolCall(
            call_id,
            name,
            cast(Mapping[str, Any], arguments),
            metadata=metadata,
        )
    return StructuredToolCall(call_id, name, None, raw_arguments, metadata)


def _decode_freeform_tool_call(
    item: Mapping[str, Any],
    profile: OpenAIResponsesProfile,
) -> FreeformToolCall:
    _require_runtime_kind(profile, RuntimeToolKind.FREEFORM)
    native_item = validate_runtime_tool_call_item(item, "custom_tool_call")
    call_id = OPENAI_RESPONSES_JSON.required_string(
        native_item.get("call_id"),
        "Responses custom tool call call_id",
    )
    name = OPENAI_RESPONSES_JSON.required_string(
        native_item.get("name"),
        "Responses custom tool call name",
    )
    raw_input = native_item.get("input")
    if not isinstance(raw_input, str):
        raise OpenAIResponsesError("Responses custom tool call input must be a string")
    return FreeformToolCall(call_id, name, raw_input, {"responses": {"item": native_item}})


def _native_content_part(part: ContentPart) -> JsonObject | None:
    container = _native_content_container(part)
    if not isinstance(container, Mapping):
        return None
    raw = cast(Mapping[str, object], container).get("content")
    if not isinstance(raw, Mapping):
        return None
    return cast(JsonObject, thaw_json_value(cast(Mapping[str, object], raw)))


def _native_runtime_tool_call(
    metadata: Mapping[str, Any],
    item_type: str,
) -> JsonObject | None:
    container = metadata.get("responses")
    if not isinstance(container, Mapping):
        return None
    raw = cast(Mapping[str, object], container).get("item")
    if not isinstance(raw, Mapping):
        return None
    return validate_runtime_tool_call_item(cast(Mapping[str, Any], raw), item_type)


def validate_runtime_tool_call_item(item: Mapping[str, Any], item_type: str) -> JsonObject:
    input_field = "arguments" if item_type == "function_call" else "input"
    allowed = {"id", "type", "call_id", "name", input_field, "caller", "namespace"}
    if item_type == "function_call":
        allowed.add("status")
    _reject_unknown_fields(item, allowed, f"Responses {item_type}")
    if item.get("type") != item_type:
        raise OpenAIResponsesError(f"Responses runtime tool call requires type={item_type!r}")
    for field in ("call_id", "name"):
        OPENAI_RESPONSES_JSON.required_string(item.get(field), f"Responses {item_type} {field}")
    if not isinstance(item.get(input_field), str):
        raise OpenAIResponsesError(f"Responses {item_type} {input_field} must be a string")
    if "id" in item:
        OPENAI_RESPONSES_JSON.required_string(item.get("id"), f"Responses {item_type} id")
    if item_type == "function_call" and "status" in item:
        status = item.get("status")
        if status not in {"in_progress", "completed", "incomplete"}:
            raise OpenAIResponsesError(
                "Responses function_call status must be in_progress, completed, or incomplete"
            )
    if "namespace" in item:
        OPENAI_RESPONSES_JSON.required_string(
            item.get("namespace"),
            f"Responses {item_type} namespace",
        )
    if "caller" in item and item.get("caller") is not None:
        _runtime_tool_caller(item.get("caller"), item_type)
    return cast(JsonObject, thaw_json_value(item))


def _runtime_tool_caller(value: object, item_type: str) -> None:
    caller = OPENAI_RESPONSES_JSON.mapping(value, f"Responses {item_type} caller")
    caller_type = caller.get("type")
    if caller_type == "direct":
        _reject_unknown_fields(caller, {"type"}, f"Responses {item_type} caller")
        return
    if caller_type == "program":
        _reject_unknown_fields(
            caller,
            {"type", "caller_id"},
            f"Responses {item_type} caller",
        )
        OPENAI_RESPONSES_JSON.required_string(
            caller.get("caller_id"),
            f"Responses {item_type} caller_id",
        )
        return
    raise OpenAIResponsesError(f"Responses {item_type} caller must be direct or program")


def _resolve_image_media_type(image_base64: str, configured: str | None) -> str:
    inferred = _infer_image_media_type(image_base64)
    if (
        configured is not None
        and inferred is not None
        and configured.casefold() != inferred.casefold()
    ):
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
    _response_output_message_header(header)
    return header


def _required_native_message_header(part: ContentPart) -> JsonObject:
    header = _native_message_header(part)
    if header is None:
        raise OpenAIResponsesError(
            "Responses assistant history requires the native response output message"
        )
    return header


def _response_output_message_header(item: Mapping[str, Any]) -> JsonObject:
    allowed = {"id", "type", "status", "role", "phase"}
    unexpected = set(item).difference(allowed | {"content"})
    if unexpected:
        raise OpenAIResponsesError(
            "Responses output message contains unsupported field: " + min(unexpected)
        )
    if item.get("type") != "message":
        raise OpenAIResponsesError("Responses output message requires type='message'")
    OPENAI_RESPONSES_JSON.required_string(item.get("id"), "Responses output message id")
    status = item.get("status")
    if status not in {"in_progress", "completed", "incomplete"}:
        raise OpenAIResponsesError(
            "Responses output message status must be in_progress, completed, or incomplete"
        )
    if item.get("role") != "assistant":
        raise OpenAIResponsesError("Responses output message requires role='assistant'")
    phase = item.get("phase")
    if phase is not None and phase not in {"commentary", "final_answer"}:
        raise OpenAIResponsesError(
            "Responses output message phase must be commentary or final_answer"
        )
    return {
        key: value
        for key, value in cast(JsonObject, thaw_json_value(item)).items()
        if key != "content"
    }


def validate_output_message_item(item: Mapping[str, Any]) -> JsonObject:
    """Validate one standard Responses output-message item."""

    header = _response_output_message_header(item)
    content = item.get("content")
    if not _is_array(content):
        raise OpenAIResponsesError("Responses output message content must be an array")
    validated_content: list[JsonObject] = []
    for raw_part in cast(Sequence[object], content):
        block = OPENAI_RESPONSES_JSON.mapping(raw_part, "Responses message content part")
        block_type = _required_type(block, "Responses message content part")
        if block_type == "output_text":
            validated_content.append(_output_text_block(block))
        elif block_type == "refusal":
            validated_content.append(_refusal_block(block))
        else:
            raise OpenAIResponsesError(
                f"unsupported Responses assistant content part: {block_type}"
            )
    return {**header, "content": validated_content}


def validate_output_content_part(value: Mapping[str, Any]) -> JsonObject:
    """Validate one standard streamed Responses content part."""

    block_type = _required_type(value, "Responses content part")
    if block_type == "output_text":
        return _output_text_block(value)
    if block_type == "refusal":
        return _refusal_block(value)
    if block_type == "reasoning_text":
        return _reasoning_block(value, field="content", block_type="reasoning_text")
    raise OpenAIResponsesError(f"unsupported Responses content part: {block_type}")


def validate_output_text_annotation(value: object) -> JsonObject | None:
    """Validate one nullable annotation from a Responses stream event."""

    return None if value is None else _citation_annotation(value)


def _output_text_block(value: Mapping[str, Any]) -> JsonObject:
    allowed = {"type", "text", "annotations", "logprobs"}
    _reject_unknown_fields(value, allowed, "Responses output_text")
    if value.get("type") != "output_text":
        raise OpenAIResponsesError("Responses text metadata must contain output_text")
    text = value.get("text")
    if not isinstance(text, str):
        raise OpenAIResponsesError("Responses output_text requires text")
    annotations = value.get("annotations")
    if not _is_array(annotations):
        raise OpenAIResponsesError("Responses output_text requires annotations")
    validated_annotations = [
        _citation_annotation(raw) for raw in cast(Sequence[object], annotations)
    ]
    logprobs = value.get("logprobs")
    if logprobs is not None:
        if not _is_array(logprobs):
            raise OpenAIResponsesError("Responses output_text logprobs must be an array")
        validated_logprobs = [
            _logprob(raw, top_level=True) for raw in cast(Sequence[object], logprobs)
        ]
    else:
        validated_logprobs = None
    encoded: JsonObject = {"type": "output_text", "text": text}
    encoded["annotations"] = validated_annotations
    if logprobs is not None:
        encoded["logprobs"] = validated_logprobs
    return encoded


def _refusal_block(value: Mapping[str, Any]) -> JsonObject:
    _reject_unknown_fields(value, {"type", "refusal"}, "Responses refusal")
    if value.get("type") != "refusal":
        raise OpenAIResponsesError("Responses refusal data must contain refusal content")
    refusal = value.get("refusal")
    if not isinstance(refusal, str):
        raise OpenAIResponsesError("Responses refusal requires refusal text")
    return {"type": "refusal", "refusal": refusal}


def _citation_annotation(value: object) -> JsonObject:
    annotation = OPENAI_RESPONSES_JSON.mapping(value, "Responses output_text annotation")
    kind = annotation.get("type")
    schemas: dict[str, tuple[set[str], set[str], set[str]]] = {
        "url_citation": (
            {"type", "start_index", "end_index", "url", "title"},
            {"start_index", "end_index", "url", "title"},
            {"start_index", "end_index"},
        ),
        "file_citation": (
            {"type", "file_id", "filename", "index"},
            {"file_id", "filename", "index"},
            {"index"},
        ),
        "container_file_citation": (
            {"type", "container_id", "file_id", "filename", "start_index", "end_index"},
            {"container_id", "file_id", "filename", "start_index", "end_index"},
            {"start_index", "end_index"},
        ),
        "file_path": (
            {"type", "file_id", "index"},
            {"file_id", "index"},
            {"index"},
        ),
    }
    if kind not in schemas:
        raise OpenAIResponsesError("unsupported Responses output_text annotation type")
    allowed, required, integer_fields = schemas[cast(str, kind)]
    _reject_unknown_fields(annotation, allowed, "Responses output_text annotation")
    encoded: JsonObject = {"type": cast(str, kind)}
    for field in required:
        raw = annotation.get(field)
        if field in integer_fields:
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                raise OpenAIResponsesError(
                    f"Responses output_text annotation {field} must be a non-negative integer"
                )
        elif not isinstance(raw, str) or not raw:
            raise OpenAIResponsesError(
                f"Responses output_text annotation {field} must be a non-empty string"
            )
        encoded[field] = raw
    if kind in {"url_citation", "container_file_citation"} and cast(
        int, encoded["end_index"]
    ) < cast(int, encoded["start_index"]):
        raise OpenAIResponsesError(
            "Responses output_text annotation end_index must not precede start_index"
        )
    return encoded


def _logprob(value: object, *, top_level: bool) -> JsonObject:
    logprob = OPENAI_RESPONSES_JSON.mapping(value, "Responses output_text logprob")
    allowed = {"token", "logprob", "bytes"}
    if top_level:
        allowed.add("top_logprobs")
    _reject_unknown_fields(logprob, allowed, "Responses output_text logprob")
    token = logprob.get("token")
    score = logprob.get("logprob")
    if not isinstance(token, str):
        raise OpenAIResponsesError("Responses output_text logprob token must be a string")
    if isinstance(score, bool) or not isinstance(score, int | float) or not math.isfinite(score):
        raise OpenAIResponsesError("Responses output_text logprob must be a number")
    encoded: JsonObject = {"token": token, "logprob": score}
    raw_bytes = logprob.get("bytes")
    if not _is_array(raw_bytes) or any(
        isinstance(byte, bool) or not isinstance(byte, int) or not 0 <= byte <= 255
        for byte in cast(Sequence[object], raw_bytes)
    ):
        raise OpenAIResponsesError("Responses output_text logprob requires byte values")
    encoded["bytes"] = list(cast(Sequence[object], raw_bytes))
    if top_level:
        top = logprob.get("top_logprobs")
        if not _is_array(top):
            raise OpenAIResponsesError("Responses output_text logprob requires top_logprobs")
        encoded["top_logprobs"] = [
            _logprob(raw, top_level=False) for raw in cast(Sequence[object], top)
        ]
    return encoded


def _reasoning_block(value: object, *, field: str, block_type: str) -> JsonObject:
    block = OPENAI_RESPONSES_JSON.mapping(value, f"Responses reasoning {field} part")
    _reject_unknown_fields(block, {"type", "text"}, f"Responses reasoning {field} part")
    if block.get("type") != block_type or not isinstance(block.get("text"), str):
        raise OpenAIResponsesError(f"Responses reasoning {field} requires {block_type} parts")
    return {"type": block_type, "text": cast(str, block["text"])}


def _reject_unknown_fields(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unexpected = set(value).difference(allowed)
    if unexpected:
        raise OpenAIResponsesError(f"{label} contains unsupported field: {min(unexpected)}")


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
    if profile.store:
        return
    if "reasoning.encrypted_content" not in profile.include:
        raise OpenAIResponsesError(
            "stateless reasoning replay requires include=reasoning.encrypted_content"
        )
    encrypted = item.get("encrypted_content")
    if not isinstance(encrypted, str) or not encrypted:
        raise OpenAIResponsesError(
            "store=False reasoning items require non-empty encrypted_content"
        )


def validate_reasoning_item(
    item: Mapping[str, Any],
    profile: OpenAIResponsesProfile,
    *,
    require_replay_state: bool,
) -> JsonObject:
    fields = frozenset({"id", "type", "status", "content", "summary", "encrypted_content"})
    unexpected = set(item).difference(fields)
    if unexpected:
        raise OpenAIResponsesError(
            "Responses reasoning metadata contains unsupported field: " + min(unexpected)
        )
    if item.get("type") != "reasoning":
        raise OpenAIResponsesError("Responses reasoning data must contain a reasoning item")
    OPENAI_RESPONSES_JSON.required_string(item.get("id"), "Responses reasoning id")
    _validate_reasoning_status(item)
    summary = item.get("summary")
    if not _is_array(summary):
        raise OpenAIResponsesError("Responses reasoning requires summary")
    content = item.get("content")
    if content is not None and not _is_array(content):
        raise OpenAIResponsesError("Responses reasoning content must be an array")
    encrypted = item.get("encrypted_content")
    if "encrypted_content" in item and encrypted is not None and not isinstance(encrypted, str):
        raise OpenAIResponsesError("Responses reasoning encrypted_content must be a string or null")
    if require_replay_state:
        _validate_stateless_reasoning(item, profile)
    encoded: JsonObject = {
        "id": cast(str, item["id"]),
        "type": "reasoning",
        "summary": [
            _reasoning_block(raw, field="summary", block_type="summary_text")
            for raw in cast(Sequence[object], summary)
        ],
    }
    if "status" in item:
        encoded["status"] = item["status"]
    if "encrypted_content" in item:
        encoded["encrypted_content"] = encrypted
    if content is not None:
        encoded["content"] = [
            _reasoning_block(raw, field="content", block_type="reasoning_text") for raw in content
        ]
    return encoded


def _validate_reasoning_status(item: Mapping[str, Any]) -> None:
    if "status" in item and item.get("status") not in {
        "in_progress",
        "completed",
        "incomplete",
    }:
        raise OpenAIResponsesError(
            "Responses reasoning status must be in_progress, completed, or incomplete"
        )


def _replay_reasoning_item(
    item: Mapping[str, Any],
    profile: OpenAIResponsesProfile,
) -> JsonObject:
    return validate_reasoning_item(item, profile, require_replay_state=True)


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
