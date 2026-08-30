"""Message conversion for Anthropic Messages."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from jharness.kernel import (
    ArtifactRef,
    ContentPart,
    Message,
    ModelOutputItem,
    ProviderToolCall,
    StructuredToolCall,
)
from jharness.models.anthropic.messages.errors import (
    ANTHROPIC_MESSAGES_JSON,
    AnthropicMessagesError,
)
from jharness.models.anthropic.messages.profile import AnthropicMessagesProfile
from jharness.models.anthropic.messages.server_tools import (
    decode_call as decode_web_search_call,
)
from jharness.models.anthropic.messages.server_tools import (
    encode_history as encode_web_search_history,
)
from jharness.models.anthropic.messages.server_tools import (
    is_result_type as is_web_search_result_type,
)

JsonValue = Any
JsonObject = dict[str, JsonValue]

_NATIVE_BLOCK_TYPES_BY_ROLE = {
    "system": {"text"},
    "user": {"text", "image", "document", "search_result", "container_upload"},
    "assistant": {"thinking", "redacted_thinking", "container_upload"},
}


def encode_messages(
    messages: Sequence[Message],
    profile: AnthropicMessagesProfile,
) -> tuple[str | list[JsonObject] | None, list[JsonObject]]:
    system_blocks: list[JsonObject] = []
    encoded_messages: list[JsonObject] = []
    tool_result_blocks: list[JsonObject] = []
    conversation_started = False
    for message in messages:
        if message.role == "system":
            if conversation_started:
                raise AnthropicMessagesError(
                    "Anthropic Messages requires all system content before messages"
                )
            system_blocks.extend(_encode_system_parts(message.parts, profile))
            continue
        if message.role == "tool":
            conversation_started = True
            tool_result_blocks.append(encode_tool_result_block(message, profile))
            continue
        if tool_result_blocks:
            encoded_messages.append({"role": "user", "content": tool_result_blocks})
            tool_result_blocks = []
        encoded_messages.append(encode_message(message, profile))
        conversation_started = True
    if tool_result_blocks:
        encoded_messages.append({"role": "user", "content": tool_result_blocks})
    return _system_value(system_blocks), encoded_messages


def encode_message(
    message: Message,
    profile: AnthropicMessagesProfile,
) -> JsonObject:
    if message.role == "tool":
        return {"role": "user", "content": [encode_tool_result_block(message, profile)]}

    role = "user" if message.role == "external" else message.role
    if role not in {"user", "assistant"}:
        raise AnthropicMessagesError(f"unsupported Anthropic message role: {message.role}")
    if role == "assistant":
        content = _encode_assistant_output(message.output, profile)
    else:
        content = encode_message_content(message.parts, role, profile)
    return {"role": role, "content": content}


def encode_tool_result_block(message: Message, profile: AnthropicMessagesProfile) -> JsonObject:
    if message.tool_call_id is None or message.outcome is None:
        raise AnthropicMessagesError("tool messages require tool_call_id and outcome")
    block: JsonObject = {
        "type": "tool_result",
        "tool_use_id": message.tool_call_id,
        "content": encode_tool_result_content(message.outcome.parts, profile),
    }
    if message.outcome.kind == "failure":
        block["is_error"] = True
    return block


def encode_message_content(
    parts: Sequence[ContentPart],
    role: str,
    profile: AnthropicMessagesProfile,
) -> str | list[JsonObject]:
    if role == "assistant":
        blocks = [_encode_assistant_part(part, profile) for part in parts]
        if not blocks:
            return ""
        if all(block.get("type") == "text" for block in blocks):
            return "".join(cast(str, block.get("text", "")) for block in blocks)
        return blocks
    if not parts:
        return ""
    if all(part.type == "text" for part in parts):
        return "".join(part.text or "" for part in parts)
    return [encode_user_content_part(part, profile) for part in parts]


def encode_user_content_part(part: ContentPart, profile: AnthropicMessagesProfile) -> JsonObject:
    wire_block = _wire_block(part, role="user", profile=profile)
    if wire_block is not None:
        return wire_block
    if part.type == "text":
        return {"type": "text", "text": part.text or ""}
    if part.type == "image":
        if "image" not in profile.capabilities.input_modalities:
            raise AnthropicMessagesError(f"{profile.name} does not support image input")
        return {"type": "image", "source": _encode_media_source(part, "image")}
    if part.type in {"artifact", "file"}:
        return _encode_artifact_or_file(part, profile)
    if part.type == "video":
        raise AnthropicMessagesError(f"{profile.name} does not support video input")
    raise AnthropicMessagesError(
        f"unsupported content part type for Anthropic Messages: {part.type}"
    )


def _encode_artifact_or_file(
    part: ContentPart,
    profile: AnthropicMessagesProfile,
) -> JsonObject:
    if part.modality == "image":
        if "image" not in profile.capabilities.input_modalities:
            raise AnthropicMessagesError(f"{profile.name} does not support image input")
        return {"type": "image", "source": _encode_media_source(part, "image")}
    if "file" not in profile.capabilities.input_modalities:
        raise AnthropicMessagesError(f"{profile.name} does not support file input")
    route = _file_route(part)
    if route == "image":
        if "image" not in profile.capabilities.input_modalities:
            raise AnthropicMessagesError(f"{profile.name} does not support image input")
        return {"type": "image", "source": _encode_media_source(part, "image")}
    if route == "container_upload":
        return {"type": "container_upload", "file_id": _artifact_ref(part, "file")}
    block: JsonObject = {"type": "document", "source": _encode_media_source(part, "file")}
    name = part.artifact.name if part.artifact is not None else part.name
    if name:
        block["title"] = name
    return block


def encode_tool_result_content(
    parts: Sequence[ContentPart], profile: AnthropicMessagesProfile
) -> str | list[JsonObject]:
    if not parts:
        return ""
    if all(part.type == "text" for part in parts):
        return "".join(part.text or "" for part in parts)
    return [encode_user_content_part(part, profile) for part in parts]


def decode_content_blocks(
    value: object,
    profile: AnthropicMessagesProfile,
) -> list[ModelOutputItem]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise AnthropicMessagesError("Anthropic response content must be an array")
    blocks = [
        ANTHROPIC_MESSAGES_JSON.mapping(item, "Anthropic content block")
        for item in cast(Sequence[object], value)
    ]
    results = _index_server_results(blocks, profile)
    consumed_results: set[str] = set()
    return [
        decoded
        for block in blocks
        if (
            decoded := _decode_content_block(
                block,
                profile,
                results,
                consumed_results,
            )
        )
        is not None
    ]


def _index_server_results(
    blocks: Sequence[Mapping[str, Any]],
    profile: AnthropicMessagesProfile,
) -> dict[str, Mapping[str, Any]]:
    results: dict[str, Mapping[str, Any]] = {}
    for block in blocks:
        block_type = _content_block_type(block)
        if not is_web_search_result_type(block_type):
            continue
        call_id = ANTHROPIC_MESSAGES_JSON.required_string(
            block.get("tool_use_id"),
            "Anthropic server tool result tool_use_id",
        )
        if call_id in results:
            raise AnthropicMessagesError(f"duplicate Anthropic server tool result: {call_id}")
        results[call_id] = block
    return results


def _decode_content_block(
    block: Mapping[str, Any],
    profile: AnthropicMessagesProfile,
    results: Mapping[str, Mapping[str, Any]],
    consumed_results: set[str],
) -> ModelOutputItem | None:
    block_type = _content_block_type(block)
    if block_type == "tool_use":
        from jharness.models.anthropic.messages.tools import decode_tool_uses

        return decode_tool_uses((block,))[0]
    if block_type == "server_tool_use":
        return _decode_server_use(block, profile, results, consumed_results)
    if is_web_search_result_type(block_type):
        call_id = ANTHROPIC_MESSAGES_JSON.required_string(
            block.get("tool_use_id"),
            "Anthropic server tool result tool_use_id",
        )
        return None if call_id in consumed_results else decode_web_search_call(None, block)
    decoder = _CONTENT_BLOCK_DECODERS.get(block_type)
    if decoder is None:
        raise AnthropicMessagesError(f"unsupported Anthropic assistant content block: {block_type}")
    return decoder(block)


def _decode_server_use(
    block: Mapping[str, Any],
    profile: AnthropicMessagesProfile,
    results: Mapping[str, Mapping[str, Any]],
    consumed_results: set[str],
) -> ProviderToolCall:
    name = ANTHROPIC_MESSAGES_JSON.required_string(
        block.get("name"),
        "Anthropic server tool use name",
    )
    if name != "web_search":
        raise AnthropicMessagesError(f"unsupported Anthropic server tool call: {name}")
    call_id = ANTHROPIC_MESSAGES_JSON.required_string(
        block.get("id"),
        "Anthropic server tool use id",
    )
    paired = results.get(call_id)
    result = None
    if paired is not None:
        result = paired
        consumed_results.add(call_id)
    return decode_web_search_call(block, result)


def _encode_assistant_output(
    output: Sequence[ModelOutputItem],
    profile: AnthropicMessagesProfile,
) -> str | list[JsonObject]:
    blocks: list[JsonObject] = []
    for item in output:
        if isinstance(item, ContentPart):
            blocks.append(_encode_assistant_part(item, profile))
            continue
        if isinstance(item, StructuredToolCall):
            from jharness.models.anthropic.messages.tools import (
                encode_assistant_tool_uses,
            )

            blocks.extend(encode_assistant_tool_uses((item,)))
            continue
        if isinstance(item, ProviderToolCall):
            blocks.extend(encode_web_search_history(item))
            continue
        raise AnthropicMessagesError(
            "Anthropic assistant history contains an unsupported output item"
        )
    if not blocks:
        return ""
    if all(set(block) == {"type", "text"} and block.get("type") == "text" for block in blocks):
        return "".join(cast(str, block.get("text", "")) for block in blocks)
    return blocks


def _content_block_type(block: Mapping[str, Any]) -> str:
    block_type = block.get("type")
    if not isinstance(block_type, str) or not block_type:
        raise AnthropicMessagesError("Anthropic content block requires non-empty type")
    return block_type


def _decode_text_block(block: Mapping[str, Any]) -> ContentPart:
    _validate_wire_block(block, "text")
    if "cache_control" in block:
        raise AnthropicMessagesError("Anthropic response text blocks do not support cache_control")
    text = block.get("text")
    if not isinstance(text, str):
        raise AnthropicMessagesError("Anthropic text block requires text")
    extra = {key: value for key, value in block.items() if key not in {"type", "text"}}
    return ContentPart.text_part(
        text,
        metadata={"anthropic": {"extra": extra}} if extra else None,
    )


def _decode_thinking_block(block: Mapping[str, Any]) -> ContentPart:
    _validate_wire_block(block, "thinking")
    thinking = block.get("thinking")
    if not isinstance(thinking, str):
        raise AnthropicMessagesError("Anthropic thinking block requires thinking text")
    return ContentPart(
        type="thinking",
        text=thinking,
        data={"anthropic": dict(block)},
    )


def _decode_redacted_thinking_block(block: Mapping[str, Any]) -> ContentPart:
    _validate_wire_block(block, "redacted_thinking")
    data = block.get("data")
    if not isinstance(data, str):
        raise AnthropicMessagesError("Anthropic redacted_thinking block requires data")
    return ContentPart(
        type="redacted_thinking",
        data={"anthropic": dict(block)},
    )


_CONTENT_BLOCK_DECODERS: Mapping[str, Callable[[Mapping[str, Any]], ContentPart]] = {
    "container_upload": lambda block: _decode_container_upload_block(block),
    "redacted_thinking": _decode_redacted_thinking_block,
    "text": _decode_text_block,
    "thinking": _decode_thinking_block,
}


def _decode_container_upload_block(block: Mapping[str, Any]) -> ContentPart:
    _validate_wire_block(block, "container_upload")
    if "cache_control" in block:
        raise AnthropicMessagesError(
            "Anthropic response container_upload blocks do not support cache_control"
        )
    file_id = ANTHROPIC_MESSAGES_JSON.required_string(
        block.get("file_id"), "Anthropic container_upload file_id"
    )
    return ContentPart.artifact_part(ArtifactRef(file_id, metadata={"anthropic": dict(block)}))


def _encode_system_parts(
    parts: Sequence[ContentPart], profile: AnthropicMessagesProfile
) -> list[JsonObject]:
    blocks: list[JsonObject] = []
    for part in parts:
        wire_block = _wire_block(part, role="system", profile=profile)
        if wire_block is not None:
            blocks.append(wire_block)
            continue
        if part.type == "text":
            blocks.append({"type": "text", "text": part.text or ""})
            continue
        raise AnthropicMessagesError(f"system messages do not support {part.type!r} content parts")
    return blocks


def _system_value(blocks: Sequence[JsonObject]) -> list[JsonObject] | None:
    if not blocks:
        return None
    return [dict(block) for block in blocks]


def _encode_assistant_part(  # noqa: C901
    part: ContentPart, profile: AnthropicMessagesProfile
) -> JsonObject:
    if part.type == "artifact" and part.artifact is not None:
        native = part.artifact.metadata.get("anthropic")
        if native is not None:
            mapping = ANTHROPIC_MESSAGES_JSON.mapping(native, "Anthropic-native artifact")
            if mapping.get("type") != "container_upload":
                raise AnthropicMessagesError(
                    "Anthropic assistant artifacts must be container_upload blocks"
                )
            _validate_wire_block(mapping, "container_upload")
            if "cache_control" in mapping:
                raise AnthropicMessagesError(
                    "Anthropic assistant container_upload blocks do not support cache_control"
                )
            if mapping.get("file_id") != part.artifact.ref:
                raise AnthropicMessagesError(
                    "Anthropic container_upload artifact id does not match native block"
                )
            return dict(mapping)
    wire_block = _wire_block(part, role="assistant", profile=profile)
    if wire_block is not None:
        return wire_block
    if part.type == "text":
        return {
            "type": "text",
            "text": part.text or "",
            **_anthropic_text_extra(part),
        }
    if part.type == "thinking":
        block: JsonObject = {"type": "thinking", "thinking": part.text or ""}
        signature = _anthropic_metadata_str(part, "signature")
        if not signature:
            raise AnthropicMessagesError("thinking parts require anthropic metadata signature")
        block["signature"] = signature
        return block
    if part.type == "redacted_thinking":
        data = _anthropic_metadata_str(part, "data")
        if data is None:
            raise AnthropicMessagesError("redacted_thinking parts require anthropic metadata data")
        return {"type": "redacted_thinking", "data": data}
    raise AnthropicMessagesError("assistant messages only support text or Anthropic-native parts")


def _wire_block(
    part: ContentPart,
    *,
    role: str,
    profile: AnthropicMessagesProfile,
) -> JsonObject | None:
    value = part.data.get("anthropic")
    if value is None:
        return None
    mapping = ANTHROPIC_MESSAGES_JSON.mapping(value, "Anthropic-native content part")
    block_type = mapping.get("type")
    if not isinstance(block_type, str) or not block_type:
        raise AnthropicMessagesError("Anthropic-native content part requires a non-empty type")
    allowed_types = _NATIVE_BLOCK_TYPES_BY_ROLE[role]
    if block_type not in allowed_types:
        raise AnthropicMessagesError(
            f"Anthropic-native {block_type!r} blocks are not allowed for {role} messages"
        )
    if block_type == "image" and "image" not in profile.capabilities.input_modalities:
        raise AnthropicMessagesError(f"{profile.name} does not support image input")
    if block_type == "document" and "file" not in profile.capabilities.input_modalities:
        raise AnthropicMessagesError(f"{profile.name} does not support file input")
    _validate_wire_block(mapping, block_type)
    return dict(mapping)


def _validate_wire_block(  # noqa: C901
    block: Mapping[str, Any], block_type: str
) -> None:
    if block_type == "text":
        _reject_unknown_fields(block, {"type", "text", "citations", "cache_control"}, "text")
        if not isinstance(block.get("text"), str):
            raise AnthropicMessagesError("Anthropic-native text blocks require text")
        validate_citations(block.get("citations"))
        _validate_cache_control(block.get("cache_control"))
        return
    if block_type == "image":
        _reject_unknown_fields(
            block, {"type", "source", "cache_control", "transformations"}, "image"
        )
        _validate_media_source(block.get("source"), "image")
        _validate_cache_control(block.get("cache_control"))
        _validate_image_transformations(block.get("transformations"))
        return
    if block_type == "document":
        _reject_unknown_fields(
            block,
            {"type", "source", "title", "context", "citations", "cache_control"},
            "document",
        )
        _validate_media_source(block.get("source"), "document")
        for key in ("title", "context"):
            if key in block and block[key] is not None and not isinstance(block[key], str):
                raise AnthropicMessagesError(
                    f"Anthropic-native document {key} must be a string or null"
                )
        _validate_document_citations_config(block.get("citations"))
        _validate_cache_control(block.get("cache_control"))
        return
    if block_type == "search_result":
        _reject_unknown_fields(
            block,
            {"type", "source", "title", "content", "citations", "cache_control"},
            "search_result",
        )
        _require_source_strings(block, "source", "title")
        _validate_search_result_content(block.get("content"))
        _validate_document_citations_config(block.get("citations"))
        _validate_cache_control(block.get("cache_control"))
        return
    if block_type == "container_upload":
        _reject_unknown_fields(block, {"type", "file_id", "cache_control"}, "container_upload")
        _require_source_strings(block, "file_id")
        _validate_cache_control(block.get("cache_control"))
        return
    if block_type == "thinking":
        _reject_unknown_fields(block, {"type", "thinking", "signature"}, "thinking")
        if not isinstance(block.get("thinking"), str):
            raise AnthropicMessagesError("Anthropic-native thinking blocks require thinking text")
        signature = block.get("signature")
        if not isinstance(signature, str):
            raise AnthropicMessagesError("Anthropic-native thinking blocks require a signature")
        return
    if block_type == "redacted_thinking":
        _reject_unknown_fields(block, {"type", "data"}, "redacted_thinking")
        data = block.get("data")
        if not isinstance(data, str):
            raise AnthropicMessagesError("Anthropic-native redacted_thinking blocks require data")
        return
    raise AnthropicMessagesError(f"unsupported Anthropic-native content block: {block_type}")


def _reject_unknown_fields(block: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(block).difference(allowed)
    if unknown:
        raise AnthropicMessagesError(
            f"Anthropic-native {label} block has unsupported field: {min(unknown)}"
        )


def validate_citations(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise AnthropicMessagesError("Anthropic citations must be an array")
    citations = cast(Sequence[object], value)
    for item in citations:
        if not isinstance(item, Mapping):
            raise AnthropicMessagesError("Anthropic citations must contain objects")
        _validate_citation(cast(Mapping[str, object], item))


def _validate_citation(citation: Mapping[str, object]) -> None:
    citation_type = citation.get("type")
    if citation_type == "char_location":
        _validate_document_citation(
            citation,
            {"start_char_index", "end_char_index"},
            "char_location",
        )
        _validate_citation_range(citation, "start_char_index", "end_char_index", minimum=0)
        return
    if citation_type == "page_location":
        _validate_document_citation(
            citation,
            {"start_page_number", "end_page_number"},
            "page_location",
        )
        _validate_citation_range(citation, "start_page_number", "end_page_number", minimum=1)
        return
    if citation_type == "content_block_location":
        _validate_document_citation(
            citation,
            {"start_block_index", "end_block_index"},
            "content_block_location",
        )
        _validate_citation_range(citation, "start_block_index", "end_block_index", minimum=0)
        return
    if citation_type == "web_search_result_location":
        _require_citation_fields(
            citation,
            {"type", "cited_text", "encrypted_index", "url"},
            {"title"},
            "web_search_result_location",
        )
        _require_citation_strings(citation, "cited_text", "encrypted_index", "url")
        _validate_nullable_citation_string(citation, "title")
        return
    if citation_type == "search_result_location":
        _require_citation_fields(
            citation,
            {
                "type",
                "cited_text",
                "end_block_index",
                "search_result_index",
                "source",
                "start_block_index",
            },
            {"title"},
            "search_result_location",
        )
        _require_citation_strings(citation, "cited_text", "source")
        _require_citation_integers(
            citation,
            "end_block_index",
            "search_result_index",
            "start_block_index",
        )
        _validate_citation_nonnegative(citation, "search_result_index")
        _validate_citation_range(citation, "start_block_index", "end_block_index", minimum=0)
        _validate_nullable_citation_string(citation, "title")
        return
    raise AnthropicMessagesError("Anthropic citation type is invalid")


def _validate_document_citation(
    citation: Mapping[str, object],
    index_fields: set[str],
    citation_type: str,
) -> None:
    _require_citation_fields(
        citation,
        {"type", "cited_text", "document_index", *index_fields},
        {"document_title", "file_id"},
        citation_type,
    )
    _require_citation_strings(citation, "cited_text")
    _require_citation_integers(citation, "document_index", *sorted(index_fields))
    _validate_nullable_citation_string(citation, "document_title")
    _validate_nullable_citation_string(citation, "file_id")


def _require_citation_fields(
    citation: Mapping[str, object],
    required: set[str],
    optional: set[str],
    citation_type: str,
) -> None:
    unknown = set(citation).difference(required | optional)
    if unknown:
        raise AnthropicMessagesError(
            f"Anthropic {citation_type} citation has unsupported field: {min(unknown)}"
        )
    missing = required.difference(citation)
    if missing:
        raise AnthropicMessagesError(
            f"Anthropic {citation_type} citation requires field: {min(missing)}"
        )


def _require_citation_strings(citation: Mapping[str, object], *fields: str) -> None:
    for field in fields:
        if not isinstance(citation[field], str):
            raise AnthropicMessagesError(f"Anthropic citation {field} must be a string")


def _require_citation_integers(citation: Mapping[str, object], *fields: str) -> None:
    for field in fields:
        value = citation[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise AnthropicMessagesError(f"Anthropic citation {field} must be an integer")


def _validate_citation_nonnegative(citation: Mapping[str, object], field: str) -> None:
    value = cast(int, citation[field])
    if value < 0:
        raise AnthropicMessagesError(f"Anthropic citation {field} must be non-negative")


def _validate_citation_range(
    citation: Mapping[str, object], start_field: str, end_field: str, *, minimum: int
) -> None:
    if "document_index" in citation:
        _validate_citation_nonnegative(citation, "document_index")
    start = cast(int, citation[start_field])
    end = cast(int, citation[end_field])
    if start < minimum or end < minimum or end < start:
        raise AnthropicMessagesError(
            f"Anthropic citation {start_field}/{end_field} has an invalid range"
        )


def _validate_nullable_citation_string(citation: Mapping[str, object], field: str) -> None:
    if field in citation and citation[field] is not None and not isinstance(citation[field], str):
        raise AnthropicMessagesError(f"Anthropic citation {field} must be a string or null")


def _validate_cache_control(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise AnthropicMessagesError("Anthropic cache_control must be an object")
    cache_control = cast(Mapping[str, object], value)
    _reject_unknown_fields(cache_control, {"type", "ttl"}, "cache_control")
    if cache_control.get("type") != "ephemeral":
        raise AnthropicMessagesError("Anthropic cache_control.type must be 'ephemeral'")
    ttl = cache_control.get("ttl")
    if ttl is not None and ttl not in {"5m", "1h"}:
        raise AnthropicMessagesError("Anthropic cache_control.ttl must be '5m' or '1h'")


def _validate_image_transformations(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise AnthropicMessagesError("Anthropic image transformations must be an object")
    transformations = cast(Mapping[str, object], value)
    _reject_unknown_fields(transformations, {"oversized_image"}, "image transformations")
    oversized_image = transformations.get("oversized_image")
    if oversized_image is not None and oversized_image not in {"downsize", "error"}:
        raise AnthropicMessagesError(
            "Anthropic image transformations.oversized_image must be 'downsize' or 'error'"
        )


def _validate_document_citations_config(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise AnthropicMessagesError("Anthropic document citations must be an object")
    config = cast(Mapping[str, object], value)
    _reject_unknown_fields(config, {"enabled"}, "document citations")
    enabled = config.get("enabled")
    if enabled is not None and not isinstance(enabled, bool):
        raise AnthropicMessagesError("Anthropic document citations.enabled must be a boolean")


def _validate_search_result_content(value: object) -> None:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise AnthropicMessagesError(
            "Anthropic search_result content must be an array of text blocks"
        )
    for item in cast(Sequence[object], value):
        if not isinstance(item, Mapping):
            raise AnthropicMessagesError("Anthropic search_result content blocks must be objects")
        _validate_wire_block(cast(Mapping[str, Any], item), "text")


def _validate_media_source(value: object, block_type: str) -> None:
    source = ANTHROPIC_MESSAGES_JSON.mapping(value, f"Anthropic-native {block_type} source")
    source_type = source.get("type")
    if source_type == "base64":
        _reject_unknown_fields(source, {"type", "media_type", "data"}, "base64 source")
        _require_source_strings(source, "media_type", "data")
        media_type = cast(str, source["media_type"])
        expected = (
            {"image/jpeg", "image/png", "image/gif", "image/webp"}
            if block_type == "image"
            else {"application/pdf"}
        )
        if media_type not in expected:
            raise AnthropicMessagesError(
                f"Anthropic {block_type} base64 source has unsupported media_type: {media_type}"
            )
        return
    if source_type == "url":
        _reject_unknown_fields(source, {"type", "url"}, "URL source")
        _require_source_strings(source, "url")
        return
    if source_type == "file":
        _reject_unknown_fields(source, {"type", "file_id"}, "file source")
        _require_source_strings(source, "file_id")
        return
    if block_type == "document" and source_type == "text":
        _reject_unknown_fields(source, {"type", "media_type", "data"}, "text source")
        _require_source_strings(source, "media_type", "data")
        if source["media_type"] != "text/plain":
            raise AnthropicMessagesError(
                "Anthropic document text source media_type must be text/plain"
            )
        return
    if block_type == "document" and source_type == "content":
        _reject_unknown_fields(source, {"type", "content"}, "content source")
        _validate_document_content_source(source.get("content"))
        return
    raise AnthropicMessagesError(f"unsupported Anthropic-native {block_type} source type")


def _validate_document_content_source(value: object) -> None:
    if isinstance(value, str):
        return
    if not isinstance(value, Sequence) or isinstance(value, bytes | bytearray):
        raise AnthropicMessagesError(
            "Anthropic document content source requires a string or iterable of text/image blocks"
        )
    for item in cast(Sequence[object], value):
        if not isinstance(item, Mapping):
            raise AnthropicMessagesError("Anthropic document content source blocks must be objects")
        block = cast(Mapping[str, Any], item)
        nested_type = block.get("type")
        if nested_type == "text":
            _validate_wire_block(block, "text")
            continue
        if nested_type == "image":
            _validate_wire_block(block, "image")
            continue
        raise AnthropicMessagesError(
            "Anthropic document content source blocks must be text or image"
        )


def _require_source_strings(source: Mapping[str, Any], *fields: str) -> None:
    for field in fields:
        value = source.get(field)
        if not isinstance(value, str) or not value:
            raise AnthropicMessagesError(f"Anthropic source requires non-empty {field}")


def _anthropic_metadata_str(part: ContentPart, key: str) -> str | None:
    metadata = part.metadata.get("anthropic")
    if not isinstance(metadata, Mapping):
        return None
    metadata_mapping = cast(Mapping[str, object], metadata)
    value = metadata_mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise AnthropicMessagesError(f"Anthropic metadata {key} must be a string")
    return value


def _anthropic_text_extra(part: ContentPart) -> JsonObject:
    metadata = part.metadata.get("anthropic")
    if not isinstance(metadata, Mapping):
        return {}
    extra = cast(Mapping[str, object], metadata).get("extra")
    if extra is None:
        return {}
    if not isinstance(extra, Mapping):
        raise AnthropicMessagesError("Anthropic text metadata extra must be an object")
    extra_mapping = cast(Mapping[str, Any], extra)
    _reject_unknown_fields(extra_mapping, {"citations", "cache_control"}, "text metadata")
    validate_citations(extra_mapping.get("citations"))
    _validate_cache_control(extra_mapping.get("cache_control"))
    return dict(extra_mapping)


def _encode_media_source(part: ContentPart, label: str) -> JsonObject:
    if part.artifact is not None:
        if label == "image":
            _validate_image_media_type(_effective_media_type(part), "files")
        return {"type": "file", "file_id": part.artifact.ref}
    if part.uri is None:
        raise AnthropicMessagesError(f"{label} input requires a uri or artifact")
    if part.uri[:5].casefold() == "data:":
        media_type, data = _parse_data_url(part.uri)
        media_type = (part.media_type or media_type).casefold()
        if label == "file":
            if media_type == "application/pdf":
                return {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": data,
                }
            if media_type == "text/plain":
                return {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": _decode_base64_text(data),
                }
            raise AnthropicMessagesError(
                "Anthropic document data URLs must use application/pdf or text/plain"
            )
        _validate_image_media_type(media_type, "data URLs")
        return {
            "type": "base64",
            "media_type": media_type,
            "data": data,
        }
    if (
        label == "file"
        and part.media_type is not None
        and part.media_type.casefold() != "application/pdf"
    ):
        raise AnthropicMessagesError("Anthropic document URL inputs must be PDFs")
    if label == "image":
        _validate_image_media_type(part.media_type, "URL inputs")
    return {"type": "url", "url": part.uri}


def _validate_image_media_type(media_type: str | None, source: str) -> None:
    if media_type is not None and media_type.casefold() not in {
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
    }:
        raise AnthropicMessagesError(
            f"Anthropic image {source} must use image/jpeg, image/png, image/gif, or image/webp"
        )


def _file_route(part: ContentPart) -> str:
    media_type = _effective_media_type(part)
    if media_type in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
        return "image"
    if media_type in {"application/pdf", "text/plain"}:
        return "document"
    if media_type is None and part.artifact is None:
        return "document"
    return "container_upload"


def _effective_media_type(part: ContentPart) -> str | None:
    media_type = part.media_type
    if media_type is None and part.artifact is not None:
        media_type = part.artifact.media_type
    if media_type is None and part.uri is not None and part.uri[:5].casefold() == "data:":
        media_type = part.uri[5:].partition(",")[0].partition(";")[0]
    return None if media_type is None else media_type.casefold()


def _artifact_ref(part: ContentPart, label: str) -> str:
    if part.artifact is None:
        raise AnthropicMessagesError(f"Anthropic {label} input requires an artifact reference")
    return part.artifact.ref


def _parse_data_url(uri: str) -> tuple[str, str]:
    header, separator, data = uri.partition(",")
    if separator != "," or header[:5].casefold() != "data:":
        raise AnthropicMessagesError("data URL input must contain media type and base64 data")
    metadata = header[5:]
    values = metadata.split(";")
    media_type = values[0] or "application/octet-stream"
    if not any(value.casefold() == "base64" for value in values[1:]):
        raise AnthropicMessagesError("data URL input must use base64 encoding")
    if not data:
        raise AnthropicMessagesError("data URL input requires base64 data")
    return media_type, data


def _decode_base64_text(data: str) -> str:
    try:
        decoded = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AnthropicMessagesError("text/plain data URL contains invalid base64 data") from exc
    try:
        return decoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AnthropicMessagesError("text/plain data URL must decode as UTF-8") from exc
