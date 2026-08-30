"""Closed exact handling for the two supported OpenAI Responses hosted tools."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
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

OPENAI_RESPONSES_WEB_SEARCH = ProviderToolId("openai.responses", "web_search")
OPENAI_RESPONSES_IMAGE_GENERATION = ProviderToolId("openai.responses", "image_generation")
SUPPORTED_PROVIDER_TOOLS = frozenset(
    {OPENAI_RESPONSES_WEB_SEARCH, OPENAI_RESPONSES_IMAGE_GENERATION}
)
_WEB_SEARCH_ITEM = "web_search_call"
_IMAGE_GENERATION_ITEM = "image_generation_call"
_TERMINAL = frozenset(
    {ProviderToolStatus.COMPLETED, ProviderToolStatus.INCOMPLETE, ProviderToolStatus.FAILED}
)


def encode_provider_declaration(spec: ProviderToolSpec) -> JsonObject:
    configuration = _configuration(spec)
    if spec.tool == OPENAI_RESPONSES_WEB_SEARCH:
        _validate_web_search_configuration(configuration)
        return {"type": "web_search", **configuration}
    if spec.tool == OPENAI_RESPONSES_IMAGE_GENERATION:
        _validate_image_configuration(configuration)
        return {"type": "image_generation", **configuration}
    raise _unsupported_tool(spec.tool)


def encode_provider_choice(spec: ProviderToolSpec) -> JsonObject:
    return {"type": cast(str, encode_provider_declaration(spec)["type"])}


def decode_provider_call(
    item: Mapping[str, Any], response: Mapping[str, Any]
) -> ProviderToolCall | None:
    item_type = OPENAI_RESPONSES_JSON.required_string(
        item.get("type"), "Responses output item type"
    )
    if item_type == _WEB_SEARCH_ITEM:
        return _decode_web_search_call(item)
    if item_type == _IMAGE_GENERATION_ITEM:
        return _decode_image_generation_call(item, response)
    return None


def encode_provider_history(call: ProviderToolCall) -> JsonObject:
    if call.tool == OPENAI_RESPONSES_WEB_SEARCH:
        return _encode_web_search_history(call)
    if call.tool == OPENAI_RESPONSES_IMAGE_GENERATION:
        return _encode_image_generation_history(call)
    raise _unsupported_tool(call.tool)


def provider_tool_for_output_item(item_type: str) -> ProviderToolId | None:
    return {
        _WEB_SEARCH_ITEM: OPENAI_RESPONSES_WEB_SEARCH,
        _IMAGE_GENERATION_ITEM: OPENAI_RESPONSES_IMAGE_GENERATION,
    }.get(item_type)


def provider_tool_for_event(event_type: str) -> ProviderToolId | None:
    if event_type.startswith("response.web_search_call."):
        return OPENAI_RESPONSES_WEB_SEARCH
    if event_type.startswith("response.image_generation_call."):
        return OPENAI_RESPONSES_IMAGE_GENERATION
    return None


def decode_provider_item_status(
    item: Mapping[str, Any], tool: ProviderToolId
) -> ProviderToolStatus:
    if tool == OPENAI_RESPONSES_WEB_SEARCH:
        _validate_item(item, _WEB_SEARCH_ITEM, {"id", "type", "status", "action"})
        _web_search_action(item.get("action"))
        return _web_search_status(item.get("status"))
    if tool == OPENAI_RESPONSES_IMAGE_GENERATION:
        _validate_item(item, _IMAGE_GENERATION_ITEM, {"id", "type", "status", "result"})
        if "result" in item and item["result"] is not None:
            OPENAI_RESPONSES_JSON.required_string(
                item["result"], "Responses image generation result"
            )
        return _image_generation_status(item.get("status"))
    raise _unsupported_tool(tool)


def decode_provider_stream_event(
    event_type: str, value: Mapping[str, Any]
) -> tuple[ProviderToolId, ProviderToolStatus, Mapping[str, object]]:
    tool = provider_tool_for_event(event_type)
    if tool is None:
        raise OpenAIResponsesError(f"unsupported Responses provider event type: {event_type}")
    suffix = event_type.rsplit(".", 1)[-1]
    if tool == OPENAI_RESPONSES_WEB_SEARCH:
        _validate_stream_event_fields(value, {"item_id", "output_index"}, event_type)
        return tool, _web_search_status(suffix), {}
    if suffix != "partial_image":
        _validate_stream_event_fields(value, {"item_id", "output_index"}, event_type)
        return tool, _image_generation_status(suffix), {}
    _validate_stream_event_fields(
        value,
        {
            "item_id",
            "output_index",
            "partial_image_b64",
            "partial_image_index",
            "background",
            "output_format",
            "quality",
            "size",
        },
        event_type,
    )
    image = OPENAI_RESPONSES_JSON.required_string(
        value.get("partial_image_b64"), "Responses partial image base64"
    )
    index = _nonnegative_int(value.get("partial_image_index"), "Responses partial image index")
    if index > 3:
        raise OpenAIResponsesError("Responses partial image index must be at most 3")
    data: dict[str, object] = {"base64": image, "partial_image_index": index}
    for field, values in {
        "background": {"auto", "opaque", "transparent"},
        "output_format": {"png", "jpeg", "webp"},
        "quality": {"auto", "low", "medium", "high"},
    }.items():
        if field in value:
            raw = value[field]
            if raw is None:
                continue
            if raw not in values:
                raise OpenAIResponsesError(f"Responses partial image {field} is invalid")
            data[field] = cast(str, raw)
    if "size" in value and value["size"] is not None:
        data["size"] = OPENAI_RESPONSES_JSON.required_string(
            value["size"], "Responses partial image size"
        )
    return tool, ProviderToolStatus.IN_PROGRESS, data


def is_terminal_provider_status(status: ProviderToolStatus | None) -> bool:
    return status in _TERMINAL


def request_requires_artifact_store(request: ModelRequest) -> bool:
    return any(spec.tool == OPENAI_RESPONSES_IMAGE_GENERATION for spec in request.provider_tools)


def history_requires_artifact_store(messages: Sequence[Message]) -> bool:
    return any(
        item.tool == OPENAI_RESPONSES_IMAGE_GENERATION and image_call_has_artifact(item)
        for message in messages
        if message.role == "assistant"
        for item in message.output
        if isinstance(item, ProviderToolCall)
    )


def response_requires_artifact_store(response: ModelResponse) -> bool:
    return any(
        item.tool == OPENAI_RESPONSES_IMAGE_GENERATION and image_call_has_inline_result(item)
        for item in response.output
        if isinstance(item, ProviderToolCall)
    )


async def hydrate_artifact_history(
    request: ModelRequest, store: OpenAIResponsesArtifactStore, context: RunContext
) -> ModelRequest:
    messages: list[Message] = []
    changed = False
    for message in request.messages:
        if message.role != "assistant":
            messages.append(message)
            continue
        output: list[ModelOutputItem] = []
        for item in message.output:
            if (
                isinstance(item, ProviderToolCall)
                and item.tool == OPENAI_RESPONSES_IMAGE_GENERATION
            ):
                hydrated = await hydrate_image_call(item, store, context)
                changed = changed or hydrated is not item
                output.append(hydrated)
            else:
                output.append(item)
        messages.append(
            message
            if tuple(output) == message.output
            else Message.assistant(output, metadata=message.metadata)
        )
    return request if not changed else replace(request, messages=tuple(messages))


async def externalize_artifacts(
    response: ModelResponse, store: OpenAIResponsesArtifactStore, context: RunContext
) -> ModelResponse:
    output: list[ModelOutputItem] = []
    changed = False
    for item in response.output:
        if isinstance(item, ProviderToolCall) and item.tool == OPENAI_RESPONSES_IMAGE_GENERATION:
            externalized = await externalize_image_call(item, store, context)
            changed = changed or externalized is not item
            output.append(externalized)
        else:
            output.append(item)
    return response if not changed else replace(response, output=tuple(output))


def _decode_web_search_call(item: Mapping[str, Any]) -> ProviderToolCall:
    _validate_item(item, _WEB_SEARCH_ITEM, {"id", "type", "status", "action"})
    call_id = OPENAI_RESPONSES_JSON.required_string(item.get("id"), "Responses web search call id")
    status = _web_search_status(item.get("status"))
    action = _web_search_action(item.get("action"))
    return ProviderToolCall(
        id=call_id,
        tool=OPENAI_RESPONSES_WEB_SEARCH,
        status=status,
        arguments=action,
        error=ErrorInfo("web_search_failed", "provider web_search call failed")
        if status is ProviderToolStatus.FAILED
        else None,
        metadata={"responses": {"item": dict(item)}},
    )


def _decode_image_generation_call(
    item: Mapping[str, Any], response: Mapping[str, Any]
) -> ProviderToolCall:
    _validate_item(item, _IMAGE_GENERATION_ITEM, {"id", "type", "status", "result"})
    call_id = OPENAI_RESPONSES_JSON.required_string(
        item.get("id"), "Responses image generation call id"
    )
    status = _image_generation_status(item.get("status"))
    result = item.get("result")
    output: tuple[ContentPart, ...] = ()
    if result is not None:
        image = OPENAI_RESPONSES_JSON.required_string(result, "Responses image generation result")
        if status is ProviderToolStatus.IN_PROGRESS:
            raise OpenAIResponsesError(
                "in-progress Responses image generation cannot carry a final result"
            )
        output = (
            ContentPart(
                type="image",
                data={"base64": image},
                media_type=_resolve_image_media_type(
                    image, _configured_image_media_type(response.get("tools"))
                ),
            ),
        )
    return ProviderToolCall(
        id=call_id,
        tool=OPENAI_RESPONSES_IMAGE_GENERATION,
        status=status,
        output=output,
        error=ErrorInfo("image_generation_failed", "provider image_generation call failed")
        if status is ProviderToolStatus.FAILED
        else None,
        metadata={"responses": {"item": _without_result(item)}},
    )


def _encode_web_search_history(call: ProviderToolCall) -> JsonObject:
    _require_call_tool(call, OPENAI_RESPONSES_WEB_SEARCH)
    if call.status is ProviderToolStatus.INCOMPLETE:
        raise OpenAIResponsesError("Responses web_search does not support incomplete status")
    return {
        "type": _WEB_SEARCH_ITEM,
        "id": call.id,
        "status": _history_status(call, _WEB_SEARCH_ITEM, _web_search_status),
        "action": _web_search_action(thaw_json_value(call.arguments)),
    }


def _encode_image_generation_history(call: ProviderToolCall) -> JsonObject:
    _require_call_tool(call, OPENAI_RESPONSES_IMAGE_GENERATION)
    if call.status is ProviderToolStatus.INCOMPLETE:
        raise OpenAIResponsesError("Responses image_generation does not support incomplete status")
    result: str | None = None
    if call.output:
        if len(call.output) != 1 or call.output[0].type != "image":
            raise OpenAIResponsesError("image_generation history requires exactly one image output")
        raw = call.output[0].data.get("base64")
        if not isinstance(raw, str) or not raw:
            raise OpenAIResponsesError(
                "image_generation history image requires non-empty base64 data"
            )
        _resolve_image_media_type(raw, call.output[0].media_type)
        result = raw
    return {
        "type": _IMAGE_GENERATION_ITEM,
        "id": call.id,
        "status": _history_status(call, _IMAGE_GENERATION_ITEM, _image_generation_status),
        "result": result,
    }


def _configuration(spec: ProviderToolSpec) -> JsonObject:
    if spec.tool not in SUPPORTED_PROVIDER_TOOLS:
        raise _unsupported_tool(spec.tool)
    return cast(JsonObject, thaw_json_value(spec.configuration))


def _validate_web_search_configuration(configuration: Mapping[str, Any]) -> None:
    _reject_unknown(
        configuration,
        {"external_web_access", "filters", "search_context_size", "user_location"},
        "web_search configuration",
    )
    external = configuration.get("external_web_access")
    if external is not None and not isinstance(external, bool):
        raise OpenAIResponsesError("web_search external_web_access must be a bool")
    context = configuration.get("search_context_size")
    if context is not None and context not in {"low", "medium", "high"}:
        raise OpenAIResponsesError("web_search search_context_size must be low, medium, or high")
    filters = configuration.get("filters")
    if filters is not None:
        typed = OPENAI_RESPONSES_JSON.mapping(filters, "web_search filters")
        _reject_unknown(typed, {"allowed_domains"}, "web_search filters")
        domains = typed.get("allowed_domains")
        if domains is not None and (
            not _is_array(domains)
            or len(cast(Sequence[object], domains)) > 100
            or any(not isinstance(domain, str) or not domain for domain in domains)
        ):
            raise OpenAIResponsesError(
                "web_search allowed_domains must contain at most 100 non-empty strings"
            )
    _validate_web_search_location(configuration.get("user_location"))


def _validate_web_search_location(location: object) -> None:
    if location is None:
        return
    typed = OPENAI_RESPONSES_JSON.mapping(location, "web_search user_location")
    _reject_unknown(
        typed, {"type", "country", "city", "region", "timezone"}, "web_search user_location"
    )
    if "type" in typed and typed["type"] != "approximate":
        raise OpenAIResponsesError("web_search user_location.type must be approximate")
    country = typed.get("country")
    if (
        "country" in typed
        and country is not None
        and (not isinstance(country, str) or len(country) != 2)
    ):
        raise OpenAIResponsesError("web_search user_location.country must be two letters")
    for field in ("city", "region", "timezone"):
        if field in typed and typed[field] is not None and not isinstance(typed[field], str):
            raise OpenAIResponsesError(f"web_search user_location.{field} must be a string")


def _validate_image_configuration(configuration: Mapping[str, Any]) -> None:
    _reject_unknown(
        configuration,
        {
            "action",
            "background",
            "input_fidelity",
            "input_image_mask",
            "model",
            "moderation",
            "output_compression",
            "output_format",
            "partial_images",
            "quality",
            "size",
        },
        "image_generation configuration",
    )
    for field, values in {
        "action": {"auto", "generate", "edit"},
        "background": {"auto", "opaque", "transparent"},
        "input_fidelity": {"low", "high"},
        "moderation": {"auto", "low"},
        "output_format": {"png", "jpeg", "webp"},
        "quality": {"auto", "low", "medium", "high"},
    }.items():
        if (
            field in configuration
            and configuration[field] is not None
            and configuration[field] not in values
        ):
            raise OpenAIResponsesError(f"image_generation {field} is invalid")
    for field in ("model", "size"):
        if field in configuration and (
            not isinstance(configuration[field], str) or not configuration[field]
        ):
            raise OpenAIResponsesError(f"image_generation {field} must be a non-empty string")
    if "input_image_mask" in configuration:
        typed = OPENAI_RESPONSES_JSON.mapping(
            configuration["input_image_mask"], "image_generation input_image_mask"
        )
        _reject_unknown(typed, {"file_id", "image_url"}, "image_generation input_image_mask")
        sources = [key for key in ("file_id", "image_url") if key in typed]
        if len(sources) != 1 or not isinstance(typed[sources[0]], str) or not typed[sources[0]]:
            raise OpenAIResponsesError(
                "image_generation input_image_mask requires one non-empty source"
            )
    for field, low, high in (("partial_images", 0, 3), ("output_compression", 0, 100)):
        if field in configuration:
            value = configuration[field]
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise OpenAIResponsesError(f"image_generation {field} is out of range")


def _web_search_action(value: object) -> JsonObject:
    action = OPENAI_RESPONSES_JSON.mapping(value, "Responses web search action")
    schemas = {
        "search": {"type", "query", "queries", "sources"},
        "open_page": {"type", "url"},
        "find_in_page": {"type", "url", "pattern"},
    }
    action_type = action.get("type")
    if action_type not in schemas:
        raise OpenAIResponsesError("unsupported Responses web search action type")
    _reject_unknown(action, schemas[cast(str, action_type)], f"web search {action_type} action")
    if action_type == "search":
        _validate_web_search_query(action)
    else:
        required = ("url", "pattern") if action_type == "find_in_page" else ()
        for field in required:
            if not isinstance(action.get(field), str):
                raise OpenAIResponsesError(
                    f"Responses web search {action_type}.{field} must be a string"
                )
        if (
            action_type == "open_page"
            and "url" in action
            and action["url"] is not None
            and not isinstance(action["url"], str)
        ):
            raise OpenAIResponsesError(
                "Responses web search open_page.url must be a string or null"
            )
    _validate_web_search_sources(action.get("sources"))
    return {key: thaw_json_value(raw) for key, raw in action.items()}


def _validate_web_search_query(action: Mapping[str, Any]) -> None:
    query = action.get("query")
    queries = action.get("queries")
    if query is not None and not isinstance(query, str):
        raise OpenAIResponsesError("Responses web search search.query must be a string")
    if queries is not None and (
        not _is_array(queries) or any(not isinstance(x, str) for x in queries)
    ):
        raise OpenAIResponsesError("Responses web search search.queries must be strings")


def _validate_web_search_sources(value: object) -> None:
    if value is None:
        return
    if not _is_array(value):
        raise OpenAIResponsesError("Responses web search search.sources must be an array")
    for raw_source in cast(Sequence[object], value):
        source = OPENAI_RESPONSES_JSON.mapping(raw_source, "Responses web search source")
        _reject_unknown(source, {"type", "url"}, "Responses web search source")
        if source.get("type") != "url":
            raise OpenAIResponsesError("Responses web search source type must be url")
        if not isinstance(source.get("url"), str):
            raise OpenAIResponsesError("Responses web search source url must be a string")


def _history_status(
    call: ProviderToolCall,
    item_type: str,
    decode_status: Callable[[object], ProviderToolStatus],
) -> str:
    responses = call.metadata.get("responses")
    if responses is None:
        return call.status.value
    typed_responses = OPENAI_RESPONSES_JSON.mapping(responses, f"Responses {item_type} metadata")
    raw_item = typed_responses.get("item")
    if raw_item is None:
        return call.status.value
    item = OPENAI_RESPONSES_JSON.mapping(raw_item, f"Responses {item_type} metadata item")
    if item.get("type") != item_type or item.get("id") != call.id:
        raise OpenAIResponsesError(
            f"Responses {item_type} metadata item does not match the provider call"
        )
    status = OPENAI_RESPONSES_JSON.required_string(
        item.get("status"), f"Responses {item_type} metadata status"
    )
    if decode_status(status) is not call.status:
        raise OpenAIResponsesError(
            f"Responses {item_type} metadata status does not match the provider call"
        )
    return status


def _configured_image_media_type(value: object) -> str | None:
    if value is None:
        return None
    if not _is_array(value):
        raise OpenAIResponsesError("Responses tools must be an array")
    declarations = [
        OPENAI_RESPONSES_JSON.mapping(raw, "Responses tool")
        for raw in cast(Sequence[object], value)
    ]
    images = [
        declaration for declaration in declarations if declaration.get("type") == "image_generation"
    ]
    if len(images) > 1:
        raise OpenAIResponsesError("Responses contains duplicate image_generation tools")
    output_format = images[0].get("output_format") if images else None
    if output_format is None:
        return None
    mapping = {"jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
    try:
        return mapping[
            OPENAI_RESPONSES_JSON.required_string(output_format, "Responses image output_format")
        ]
    except KeyError as exc:
        raise OpenAIResponsesError("unsupported Responses image output format") from exc


def _validate_item(item: Mapping[str, Any], expected_type: str, allowed: set[str]) -> None:
    _reject_unknown(item, allowed, f"Responses {expected_type}")
    if item.get("type") != expected_type:
        raise OpenAIResponsesError(f"Responses provider item requires type={expected_type!r}")
    OPENAI_RESPONSES_JSON.required_string(item.get("id"), f"Responses {expected_type} id")


def _validate_stream_event_fields(
    value: Mapping[str, Any], fields: set[str], event_type: str
) -> None:
    _reject_unknown(value, {"type", "sequence_number", *fields}, f"Responses {event_type}")


def _web_search_status(value: object) -> ProviderToolStatus:
    return _status(value, "Responses web search status", {"in_progress", "searching"})


def _image_generation_status(value: object) -> ProviderToolStatus:
    return _status(value, "Responses image generation status", {"in_progress", "generating"})


def _status(value: object, label: str, active: set[str]) -> ProviderToolStatus:
    status = OPENAI_RESPONSES_JSON.required_string(value, label)
    if status in active:
        return ProviderToolStatus.IN_PROGRESS
    if status == "completed":
        return ProviderToolStatus.COMPLETED
    if status == "failed":
        return ProviderToolStatus.FAILED
    raise OpenAIResponsesError(f"unsupported {label}: {status}")


def _require_call_tool(call: ProviderToolCall, tool: ProviderToolId) -> None:
    if call.tool != tool:
        raise _unsupported_tool(call.tool)


def _unsupported_tool(tool: ProviderToolId) -> OpenAIResponsesError:
    return OpenAIResponsesError(
        f"Responses profile does not support provider tool: {tool.namespace}/{tool.type}"
    )


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value).difference(allowed)
    if unknown:
        raise OpenAIResponsesError(f"{label} contains unsupported field: {min(unknown)}")


def _without_result(item: Mapping[str, Any]) -> JsonObject:
    return {key: thaw_json_value(value) for key, value in item.items() if key != "result"}


def _resolve_image_media_type(image: str, configured: str | None) -> str:
    inferred = _infer_image_media_type(image)
    if configured is not None and inferred is not None and configured != inferred:
        raise OpenAIResponsesError("Responses image result does not match configured output format")
    return configured or inferred or "image/png"


def _infer_image_media_type(image: str) -> str | None:
    try:
        decoded = base64.b64decode(image, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise OpenAIResponsesError("Responses image data must contain valid base64") from exc
    if decoded.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if decoded.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(decoded) >= 12 and decoded.startswith(b"RIFF") and decoded[8:12] == b"WEBP":
        return "image/webp"
    return None


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpenAIResponsesError(f"{label} must be a non-negative integer")
    return value


def _is_array(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)
