"""Message conversion for OpenAI Chat Completions."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from typing import Any, cast

from jharness.kernel import (
    ContentPart,
    Message,
    ProviderToolCall,
    RuntimeToolCall,
)
from jharness.models.openai.chat.errors import OPENAI_CHAT_JSON, OpenAIChatError
from jharness.models.openai.chat.profile import OpenAIChatProfile

JsonValue = Any
JsonObject = dict[str, JsonValue]


def encode_chat_message(
    message: Message,
    profile: OpenAIChatProfile,
) -> JsonObject:
    role = "user" if message.role == "external" else message.role
    if role == "tool":
        if message.tool_call_id is None or message.outcome is None:
            raise OpenAIChatError("tool messages require tool_call_id and outcome")
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": _text_only_content(message.outcome.parts, "tool"),
        }

    content_parts = message.parts
    runtime_calls: tuple[RuntimeToolCall, ...] = ()
    if role == "assistant":
        if any(isinstance(item, ProviderToolCall) for item in message.output):
            raise OpenAIChatError("Chat Completions cannot encode provider tool output history")
        assistant_parts = tuple(item for item in message.output if isinstance(item, ContentPart))
        runtime_calls = message.runtime_tool_calls()
        content_parts = _without_reasoning(assistant_parts)
    content = encode_message_content(content_parts, role, profile)
    data: JsonObject = {"role": role, "content": content}
    if role == "assistant" and not content_parts and _assistant_content_was_null(message):
        data["content"] = None
    if role == "assistant" and runtime_calls:
        from jharness.models.openai.chat.tools import encode_assistant_tool_calls

        data["tool_calls"] = encode_assistant_tool_calls(runtime_calls)
        if content == "":
            data.pop("content")
    return data


def encode_message_content(
    parts: Sequence[ContentPart],
    role: str,
    profile: OpenAIChatProfile,
) -> str | list[JsonObject]:
    if role == "system":
        return _text_only_content(parts, "system")
    if role == "assistant":
        return _encode_assistant_content(parts)
    if role == "tool":
        return _text_only_content(parts, role)
    if not parts:
        return ""
    if all(part.type == "text" for part in parts):
        return "".join(part.text or "" for part in parts)
    return [encode_content_part(part, profile) for part in parts]


def encode_content_part(part: ContentPart, profile: OpenAIChatProfile) -> JsonObject:
    if part.type == "text":
        return {"type": "text", "text": part.text or ""}
    if part.type == "image":
        if "image" not in profile.capabilities.input_modalities:
            raise OpenAIChatError(f"{profile.name} does not support image input")
        uri = _required_uri(part, "image")
        return {"type": "image_url", "image_url": {"url": uri}}
    if part.type in {"audio", "input_audio"}:
        if "audio" not in profile.capabilities.input_modalities:
            raise OpenAIChatError(f"{profile.name} does not support audio input")
        data, audio_format = _audio_data_and_format(part)
        return {"type": "input_audio", "input_audio": {"data": data, "format": audio_format}}
    if part.type in {"artifact", "file"}:
        return _encode_file_content_part(part, profile)
    raise OpenAIChatError(f"unsupported content part type for Chat Completions: {part.type}")


def _encode_file_content_part(part: ContentPart, profile: OpenAIChatProfile) -> JsonObject:
    artifact = part.artifact
    _require_file_input_modality(profile)
    if artifact is not None:
        return {"type": "file", "file": {"file_id": artifact.ref}}
    file_data = _base64_data(_required_uri(part, "file"), "file")
    return {
        "type": "file",
        "file": {"file_data": file_data, "filename": part.name or "file"},
    }


def _require_file_input_modality(profile: OpenAIChatProfile) -> None:
    if "file" not in profile.capabilities.input_modalities:
        raise OpenAIChatError(f"{profile.name} does not support file input")


def decode_message_content(value: object, annotations: object = None) -> list[ContentPart]:
    if value is None:
        if annotations not in (None, []):
            raise OpenAIChatError("chat completion annotations require string content")
        return []
    if isinstance(value, str):
        return [
            ContentPart.text_part(
                value,
                metadata={"openai": {"annotations": _annotations(annotations)}},
            )
        ]
    raise OpenAIChatError("chat completion message content must be a string or null")


def decode_message_refusal(value: object) -> list[ContentPart]:
    if value is None:
        return []
    if not isinstance(value, str):
        raise OpenAIChatError("chat completion message refusal must be a string or null")
    block = {"type": "refusal", "refusal": value}
    return [ContentPart(type="refusal", text=value, data={"openai": block})]


def _encode_assistant_content(
    parts: Sequence[ContentPart],
) -> str | list[JsonObject]:
    if not parts:
        return ""
    if all(part.type == "text" for part in parts):
        if any(_text_annotations(part) for part in parts):
            raise OpenAIChatError(
                "Chat Completions cannot replay assistant message annotations in a standard request"
            )
        return "".join(part.text or "" for part in parts)
    if len(parts) != 1 or parts[0].type != "refusal":
        raise OpenAIChatError("assistant history must contain text parts only or one refusal part")
    part = parts[0]
    wire_block = _wire_block(part)
    if wire_block is None:
        refusal = part.text
        if not isinstance(refusal, str):
            raise OpenAIChatError("assistant refusal parts require text or OpenAI data")
        wire_block = {"type": "refusal", "refusal": refusal}
    return [wire_block]


def _without_reasoning(parts: Sequence[ContentPart]) -> tuple[ContentPart, ...]:
    if any(part.type == "reasoning" for part in parts):
        raise OpenAIChatError("Chat Completions does not support assistant reasoning history")
    return tuple(parts)


def _wire_block(part: ContentPart) -> JsonObject | None:
    raw = part.data.get("openai")
    if raw is None:
        return None
    block = OPENAI_CHAT_JSON.mapping(raw, "OpenAI-native content part")
    block_type = block.get("type")
    if block_type != "refusal":
        raise OpenAIChatError(f"unsupported OpenAI-native assistant content part: {block_type}")
    if block_type != part.type:
        raise OpenAIChatError("OpenAI-native content type must match the ContentPart type")
    unexpected = set(block).difference({"type", "refusal"})
    if unexpected:
        raise OpenAIChatError(
            "OpenAI-native refusal contains unsupported field: " + min(unexpected)
        )
    refusal = block.get("refusal")
    if not isinstance(refusal, str):
        raise OpenAIChatError("OpenAI-native refusal parts require refusal text")
    return {"type": "refusal", "refusal": refusal}


def _annotations(value: object) -> list[JsonObject]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise OpenAIChatError("chat completion annotations must be an array")
    annotations: list[JsonObject] = []
    for raw in cast(list[object], value):
        annotation = OPENAI_CHAT_JSON.mapping(raw, "chat completion annotation")
        if set(annotation) != {"type", "url_citation"} or annotation.get("type") != "url_citation":
            raise OpenAIChatError("unsupported chat completion annotation")
        citation = OPENAI_CHAT_JSON.mapping(
            annotation.get("url_citation"), "chat completion url_citation"
        )
        if set(citation) != {"start_index", "end_index", "title", "url"}:
            raise OpenAIChatError("chat completion url_citation has unsupported fields")
        start = citation.get("start_index")
        end = citation.get("end_index")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end < start
        ):
            raise OpenAIChatError("chat completion url_citation indexes are invalid")
        title = citation.get("title")
        url = citation.get("url")
        if not isinstance(title, str) or not isinstance(url, str):
            raise OpenAIChatError("chat completion url_citation title and url must be strings")
        annotations.append(
            {
                "type": "url_citation",
                "url_citation": {
                    "start_index": start,
                    "end_index": end,
                    "title": title,
                    "url": url,
                },
            }
        )
    return annotations


def _text_annotations(part: ContentPart) -> object:
    native: object = part.metadata.get("openai")
    if not isinstance(native, Mapping):
        return None
    return cast(Mapping[str, Any], native).get("annotations")


def _assistant_content_was_null(message: Message) -> bool:
    marker: object = message.metadata.get("openai_chat")
    if marker is None:
        return False
    if not isinstance(marker, Mapping):
        raise OpenAIChatError("OpenAI Chat assistant history marker must be an object")
    mapping = cast(Mapping[str, Any], marker)
    if set(mapping) != {"content_null"} or mapping.get("content_null") is not True:
        raise OpenAIChatError("invalid OpenAI Chat assistant history marker")
    return True


def _text_only_content(parts: Sequence[ContentPart], role: str) -> str:
    unsupported = [part.type for part in parts if part.type != "text"]
    if unsupported:
        raise OpenAIChatError(f"{role} messages only support text content parts")
    return "".join(part.text or "" for part in parts)


def _required_uri(part: ContentPart, label: str) -> str:
    if part.uri is None:
        raise OpenAIChatError(f"{label} input requires a uri")
    return part.uri


def _audio_data_and_format(part: ContentPart) -> tuple[str, str]:
    uri = _required_uri(part, "audio")
    media_type = part.media_type
    if media_type is None and uri[:5].casefold() == "data:":
        media_type = uri[5:].partition(",")[0].partition(";")[0]
    if not isinstance(media_type, str):
        raise OpenAIChatError("audio input requires media_type or a data URI media type")
    normalized = media_type.casefold()
    audio_formats = {"audio/wav": "wav", "audio/mpeg": "mp3", "audio/mp3": "mp3"}
    audio_format = audio_formats.get(normalized)
    if audio_format is None:
        raise OpenAIChatError("Chat Completions input_audio supports WAV or MP3 only")
    return _base64_data(uri, "audio"), audio_format


def _base64_data(value: str, label: str) -> str:
    encoded = value
    if value[:5].casefold() == "data:":
        header, separator, encoded = value.partition(",")
        if not separator or "base64" not in header.casefold().split(";")[1:]:
            raise OpenAIChatError(f"{label} data URI must use base64 encoding")
    try:
        base64.b64decode(encoded, validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise OpenAIChatError(f"{label} input requires base64 data, not a URL") from exc
    return encoded
