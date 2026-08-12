"""Exercise the complete JHarness/DeepSeek Responses protocol intersection.

The API key is read only from ``DEEPSEEK_API_KEY``.  The default run executes
offline codec checks plus real DeepSeek HTTP/SSE calls, runtime tool loops, a
stateless multi-turn conversation, and provider-hosted web search.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import secrets
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, cast

from jharness.kernel import (
    Checkpoint,
    Completed,
    ContentPart,
    DeltaSink,
    Event,
    EventKind,
    FreeformToolCall,
    FreeformToolSpec,
    Message,
    Model,
    ModelCapabilities,
    ModelOptions,
    ModelRequest,
    ModelResponse,
    ProviderToolCall,
    ProviderToolStatus,
    ResponseFormat,
    RunContext,
    RunLimits,
    Runtime,
    SettledResult,
    StructuredToolCall,
    StructuredToolSpec,
    ToolChoice,
    ToolContext,
    ToolResult,
    ToolSuccess,
    thaw_json_value,
)
from jharness.kernel.diagnostics import build_trace, verify_trace
from jharness.kernel.wire import (
    decode_checkpoint,
    decode_event,
    decode_model_response,
    encode_checkpoint,
    encode_event,
    encode_model_response,
)
from jharness.models.deepseek import (
    DEEPSEEK_RESPONSES_WEB_SEARCH,
    DeepSeekResponsesEffort,
    deepseek_openai_responses_profile,
    deepseek_responses_web_search,
)
from jharness.models.openai import OpenAIResponsesModel
from jharness.toolkit import ToolRegistry

from .offline_contract import run_offline_contract

_MODEL = "deepseek-v4-flash"
_DEFAULT_BASE_URL = "https://api.deepseek.com"
_GUIDE_URL = "https://api-docs.deepseek.com/guides/responses_api/"
_URL = re.compile(r"https?://[^\s<>\]\[)('\\\"]+")
_REQUIRED_DELTA_KINDS = frozenset(
    {"content", "reasoning", "tool_call", "provider_tool_call", "usage"}
)
_REQUIRED_OUTPUT_KINDS = frozenset(
    {
        "ContentPart:reasoning",
        "ContentPart:text",
        "FreeformToolCall",
        "ProviderToolCall",
        "StructuredToolCall",
    }
)


@dataclass(frozen=True, slots=True)
class LiveConfig:
    api_key: str
    base_url: str
    timeout: float


@dataclass(frozen=True, slots=True)
class ObservedRun:
    checkpoint: Checkpoint
    events: tuple[Event, ...]

    @property
    def delta_kinds(self) -> frozenset[str]:
        kinds: set[str] = set()
        for event in self.events:
            if event.kind is not EventKind.MODEL_DELTA:
                continue
            kind = event.data.get("kind")
            if isinstance(kind, str):
                kinds.add(kind)
        return frozenset(kinds)


@dataclass(frozen=True, slots=True)
class ScenarioReport:
    name: str
    response_ids: tuple[str, ...]
    output_kinds: tuple[str, ...]
    delta_kinds: tuple[str, ...]
    final_text: str
    details: Mapping[str, object]


class RecordingModel:
    """Record effective requests and terminal responses around a real model.

    ``first_turn_choice`` is useful for a Runtime tool loop: it forces exactly
    one tool-producing turn, then switches the post-tool turn to ``none`` so
    the model can produce the final answer.
    """

    def __init__(
        self,
        delegate: Model,
        *,
        first_turn_choice: ToolChoice | None = None,
    ) -> None:
        self._delegate = delegate
        self._first_turn_choice = first_turn_choice
        self.requests: list[ModelRequest] = []
        self.responses: list[ModelResponse] = []

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._delegate.capabilities

    async def invoke(
        self,
        request: ModelRequest,
        context: RunContext,
        *,
        stream: bool,
        emit_delta: DeltaSink | None,
    ) -> ModelResponse:
        effective = request
        if self._first_turn_choice is not None:
            has_tool_output = any(message.role == "tool" for message in request.messages)
            choice = ToolChoice(type="none") if has_tool_output else self._first_turn_choice
            effective = replace(request, tool_choice=choice)
        self.requests.append(effective)
        response = await self._delegate.invoke(
            effective,
            context,
            stream=stream,
            emit_delta=emit_delta,
        )
        _validate_live_response(response)
        self.responses.append(response)
        return response


class SumTool:
    spec = StructuredToolSpec(
        "calculate_sum",
        "Add two integers and return their sum.",
        {
            "type": "object",
            "properties": {
                "left": {"type": "integer"},
                "right": {"type": "integer"},
            },
            "required": ["left", "right"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"sum": {"type": "integer"}},
            "required": ["sum"],
            "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    async def invoke(self, call: StructuredToolCall, context: ToolContext) -> ToolResult:
        del context
        left = call.arguments.get("left")
        right = call.arguments.get("right")
        if (
            isinstance(left, bool)
            or not isinstance(left, int)
            or isinstance(right, bool)
            or not isinstance(right, int)
        ):
            raise ValueError("calculate_sum requires integer left and right arguments")
        self.calls.append((left, right))
        total = left + right
        structured = {"sum": total}
        return SettledResult(
            ToolSuccess(
                (ContentPart.text_part(json.dumps(structured, separators=(",", ":"))),),
                structured,
            )
        )


class InMemoryApplyPatchTool:
    """A non-mutating apply_patch implementation for protocol testing."""

    spec = FreeformToolSpec(
        "apply_patch",
        "Accept a patch for this demo without changing the filesystem.",
        {
            "type": "object",
            "properties": {
                "accepted": {"type": "boolean"},
                "filesystem_mutated": {"type": "boolean"},
            },
            "required": ["accepted", "filesystem_mutated"],
            "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        self.inputs: list[str] = []

    async def invoke(self, call: FreeformToolCall, context: ToolContext) -> ToolResult:
        del context
        self.inputs.append(call.input)
        structured = {"accepted": True, "filesystem_mutated": False}
        return SettledResult(
            ToolSuccess(
                (
                    ContentPart.text_part(
                        "Patch accepted by the in-memory demo; no file was changed."
                    ),
                ),
                structured,
            )
        )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _model(config: LiveConfig, effort: DeepSeekResponsesEffort | None) -> OpenAIResponsesModel:
    return OpenAIResponsesModel(
        base_url=config.base_url,
        api_key=config.api_key,
        model=_MODEL,
        profile=deepseek_openai_responses_profile(effort=effort),
        timeout=config.timeout,
    )


def _limits() -> RunLimits:
    return RunLimits(
        max_planning_steps=4,
        max_tool_calls=2,
        timeout_seconds=240,
    )


def _validate_live_response(response: ModelResponse) -> None:
    """Validate every portable ModelResponse field and its v0 wire round-trip."""

    wire = encode_model_response(response)
    _require(
        set(wire)
        == {
            "output",
            "finish_reason",
            "usage",
            "model_id",
            "response_id",
            "provider_turn_pending",
            "metadata",
        },
        "portable ModelResponse wire fields changed",
    )
    _require(decode_model_response(wire) == response, "ModelResponse wire round-trip failed")
    _require(bool(response.output), "DeepSeek returned no ordered output items")
    _require(response.finish_reason is not None, "finish_reason was not decoded")
    _require(response.model_id == _MODEL, "unexpected terminal response model_id")
    _require(bool(response.response_id), "terminal response_id was not decoded")
    _require(not response.provider_turn_pending, "terminal response cannot be provider-pending")
    _require(response.usage is not None, "DeepSeek terminal usage was not decoded")
    _require(
        response.usage is not None and response.usage.total_tokens is not None,
        "DeepSeek total_tokens was not decoded",
    )
    _require(response.metadata.get("object") == "response", "response object metadata missing")
    _require(
        response.metadata.get("status") in {"completed", "incomplete"},
        "response status metadata missing",
    )
    if response.usage is not None:
        usage_wire = cast(Mapping[str, object], wire["usage"])
        _require(
            set(usage_wire)
            == {
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "reasoning_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
            },
            "portable ModelUsage wire fields changed",
        )


async def _observe_start(
    runtime: Runtime,
    messages: Sequence[Message],
    *,
    stream: bool,
) -> ObservedRun:
    invocation = runtime.start(messages, stream=stream)
    events = tuple([event async for event in invocation.events()])
    checkpoint = await invocation.result()
    _require(
        isinstance(checkpoint.snapshot.state, Completed),
        "runtime stopped with "
        f"{checkpoint.snapshot.status}: {checkpoint.snapshot.state!r}; "
        f"last_event={events[-1].data if events else None!r}",
    )
    _require(
        decode_checkpoint(encode_checkpoint(checkpoint)) == checkpoint,
        "Checkpoint v0 wire round-trip failed",
    )
    for event in events:
        _require(decode_event(encode_event(event)) == event, "Event v0 wire round-trip failed")
    verification = verify_trace(build_trace(events, "start"))
    _require(
        verification.final_checkpoint_id == checkpoint.id,
        "verified trace does not end at the returned checkpoint",
    )
    return ObservedRun(checkpoint, events)


def _visible_text(response: ModelResponse) -> str:
    # Reasoning is retained and asserted as protocol history, but the report exposes
    # only user-facing text/refusal content.
    return "".join(
        part.text or "" for part in response.visible_parts() if part.type in {"text", "refusal"}
    )


def _output_kinds(responses: Sequence[ModelResponse]) -> tuple[str, ...]:
    kinds: set[str] = set()
    for response in responses:
        for item in response.output:
            if isinstance(item, ContentPart):
                kinds.add(f"ContentPart:{item.type}")
            else:
                kinds.add(type(item).__name__)
    return tuple(sorted(kinds))


def _response_ids(responses: Sequence[ModelResponse]) -> tuple[str, ...]:
    ids: list[str] = []
    for response in responses:
        _require(response.response_id is not None, "response id disappeared after validation")
        ids.append(cast(str, response.response_id))
    return tuple(ids)


def _event_input_kinds(events: Sequence[Event]) -> frozenset[str]:
    kinds: set[str] = set()
    for event in events:
        if event.kind is EventKind.MODEL_DELTA and event.data.get("kind") == "tool_call":
            input_kind = event.data.get("input_kind")
            if isinstance(input_kind, str):
                kinds.add(input_kind)
    return frozenset(kinds)


def _provider_lifecycle_actions(
    events: Sequence[Event],
    event_name: str,
) -> tuple[Mapping[str, object], ...]:
    actions: list[Mapping[str, object]] = []
    for event in events:
        if (
            event.kind is not EventKind.MODEL_DELTA
            or event.data.get("kind") != "provider_tool_call"
            or event.data.get("event") != event_name
        ):
            continue
        raw_data = event.data.get("data")
        if not isinstance(raw_data, Mapping):
            continue
        action = cast(Mapping[object, object], raw_data).get("action")
        if isinstance(action, Mapping):
            actions.append(cast(Mapping[str, object], action))
    return tuple(actions)


def _wire_input_types(model: OpenAIResponsesModel, request: ModelRequest) -> tuple[str, ...]:
    return tuple(cast(str, item["type"]) for item in _wire_input_items(model, request))


def _wire_input_items(
    model: OpenAIResponsesModel,
    request: ModelRequest,
) -> tuple[Mapping[str, object], ...]:
    payload = model.codec.encode_request(request)
    for forbidden in ("store", "include", "previous_response_id"):
        _require(forbidden not in payload, f"DeepSeek stateless request emitted {forbidden}")
    raw_input = payload.get("input")
    _require(isinstance(raw_input, list), "Responses input was not encoded as an item list")
    result: list[Mapping[str, object]] = []
    for raw_item in cast(list[object], raw_input):
        _require(isinstance(raw_item, Mapping), "Responses input item is not an object")
        item = cast(Mapping[str, object], raw_item)
        item_type = item.get("type")
        _require(isinstance(item_type, str), "Responses input item has no type")
        result.append(item)
    return tuple(result)


def _native_provider_history(
    calls: Sequence[ProviderToolCall],
) -> tuple[Mapping[str, object], ...]:
    items: list[Mapping[str, object]] = []
    for call in calls:
        responses = call.metadata.get("responses")
        _require(isinstance(responses, Mapping), "provider call has no Responses metadata")
        raw_item = cast(Mapping[object, object], responses).get("item")
        _require(isinstance(raw_item, Mapping), "provider call has no native Responses item")
        items.append(cast(Mapping[str, object], thaw_json_value(raw_item)))
    return tuple(items)


def _json_object(text: str) -> dict[str, Any]:
    value: object = json.loads(text)
    _require(isinstance(value, dict), "structured response was not a JSON object")
    mapping = cast(dict[object, object], value)
    _require(all(isinstance(key, str) for key in mapping), "JSON object keys must be strings")
    return cast(dict[str, Any], mapping)


def _citation_urls(response: ModelResponse) -> tuple[str, ...]:
    urls: list[str] = []
    for part in response.visible_parts():
        native = part.metadata.get("responses")
        if not isinstance(native, Mapping):
            continue
        content = cast(Mapping[object, object], native).get("content")
        if not isinstance(content, Mapping):
            continue
        annotations = cast(Mapping[object, object], content).get("annotations")
        if not isinstance(annotations, Sequence) or isinstance(annotations, str | bytes):
            continue
        for annotation in cast(Sequence[object], annotations):
            if not isinstance(annotation, Mapping):
                continue
            raw_url = cast(Mapping[object, object], annotation).get("url")
            if isinstance(raw_url, str):
                urls.append(raw_url)
    return tuple(dict.fromkeys(urls))


def extract_text_urls(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(match.rstrip("`*.,;:") for match in _URL.findall(text)))


async def _basic_scenario(config: LiveConfig) -> ScenarioReport:
    print("[live] basic non-streaming terminal response", file=sys.stderr, flush=True)
    wire_model = _model(config, "none")
    recording = RecordingModel(wire_model)
    run = await _observe_start(
        Runtime(
            model=recording,
            model_options=ModelOptions(max_output_tokens=64),
            tool_choice=ToolChoice(type="none"),
            limits=_limits(),
        ),
        (Message.user("Reply with exactly JHARNESS_RESPONSE_OK and nothing else."),),
        stream=False,
    )
    _require(len(recording.responses) == 1, "basic scenario expected one model response")
    text = _visible_text(recording.responses[0])
    _require("JHARNESS_RESPONSE_OK" in text, "basic end-to-end response marker missing")
    return ScenarioReport(
        "basic_nonstream",
        _response_ids(recording.responses),
        _output_kinds(recording.responses),
        tuple(sorted(run.delta_kinds)),
        text,
        {"event_count": len(run.events), "checkpoint_revision": run.checkpoint.snapshot.revision},
    )


async def _reasoning_scenario(config: LiveConfig) -> ScenarioReport:
    print("[live] reasoning + text + usage SSE", file=sys.stderr, flush=True)
    wire_model = _model(config, "high")
    recording = RecordingModel(wire_model)
    run = await _observe_start(
        Runtime(
            model=recording,
            model_options=ModelOptions(max_output_tokens=512),
            tool_choice=ToolChoice(type="none"),
            limits=_limits(),
        ),
        (
            Message.user(
                "Compute 37 multiplied by 41. Think through the arithmetic, then give the "
                "number and one short verification sentence."
            ),
        ),
        stream=True,
    )
    _require(
        {"reasoning", "content", "usage"}.issubset(run.delta_kinds),
        "reasoning stream did not expose reasoning/content/usage deltas",
    )
    response = recording.responses[0]
    _require(
        any(isinstance(item, ContentPart) and item.type == "reasoning" for item in response.output),
        "terminal response did not retain reasoning history",
    )
    text = _visible_text(response)
    _require("1517" in text, "reasoning scenario returned an unexpected answer")
    reasoning_chars = sum(
        len(cast(str, event.data.get("text_delta", "")))
        for event in run.events
        if event.kind is EventKind.MODEL_DELTA and event.data.get("kind") == "reasoning"
    )
    return ScenarioReport(
        "reasoning_stream",
        _response_ids(recording.responses),
        _output_kinds(recording.responses),
        tuple(sorted(run.delta_kinds)),
        text,
        {"reasoning_delta_chars": reasoning_chars, "event_count": len(run.events)},
    )


async def _structured_output_scenario(config: LiveConfig) -> ScenarioReport:
    print("[live] JSON object + strict JSON Schema", file=sys.stderr, flush=True)
    wire_model = _model(config, "none")
    recording = RecordingModel(wire_model)
    object_run = await _observe_start(
        Runtime(
            model=recording,
            response_format=ResponseFormat("json_object"),
            model_options=ModelOptions(max_output_tokens=128),
            tool_choice=ToolChoice(type="none"),
            limits=_limits(),
        ),
        (
            Message.user(
                'Return one JSON object with exactly {"status":"ok","count":3}. '
                "Do not use Markdown."
            ),
        ),
        stream=True,
    )
    object_value = _json_object(_visible_text(recording.responses[-1]))
    _require(object_value == {"status": "ok", "count": 3}, "JSON object mode mismatch")

    schema = {
        "type": "object",
        "properties": {
            "protocol": {"type": "string", "const": "responses"},
            "passed": {"type": "boolean"},
        },
        "required": ["protocol", "passed"],
        "additionalProperties": False,
    }
    schema_run = await _observe_start(
        Runtime(
            model=recording,
            response_format=ResponseFormat("json_schema", schema=schema, strict=True),
            model_options=ModelOptions(max_output_tokens=128),
            tool_choice=ToolChoice(type="none"),
            limits=_limits(),
        ),
        (Message.user('Return protocol="responses" and passed=true in the required schema.'),),
        stream=True,
    )
    schema_value = _json_object(_visible_text(recording.responses[-1]))
    _require(
        schema_value == {"protocol": "responses", "passed": True},
        "strict JSON Schema output mismatch",
    )
    delta_kinds = object_run.delta_kinds.union(schema_run.delta_kinds)
    return ScenarioReport(
        "structured_output",
        _response_ids(recording.responses),
        _output_kinds(recording.responses),
        tuple(sorted(delta_kinds)),
        json.dumps(schema_value, ensure_ascii=False, sort_keys=True),
        {"json_object": object_value, "json_schema": schema_value},
    )


async def _function_tool_scenario(config: LiveConfig) -> ScenarioReport:
    print(
        "[live] structured function call -> local tool -> final response",
        file=sys.stderr,
        flush=True,
    )
    wire_model = _model(config, "none")
    sum_tool = SumTool()
    recording = RecordingModel(
        wire_model,
        first_turn_choice=ToolChoice(type="runtime", name="calculate_sum"),
    )
    run = await _observe_start(
        Runtime(
            model=recording,
            tools=ToolRegistry((sum_tool,)),
            model_options=ModelOptions(max_output_tokens=256),
            tool_choice=ToolChoice(type="auto"),
            limits=_limits(),
        ),
        (
            Message.user(
                "Call calculate_sum exactly once with left=19 and right=23. After receiving "
                "the tool result, answer with the sum."
            ),
        ),
        stream=True,
    )
    _require(sum_tool.calls == [(19, 23)], "structured tool was not called exactly as requested")
    _require(len(recording.responses) == 2, "structured tool loop expected two model turns")
    _require(
        any(isinstance(item, StructuredToolCall) for item in recording.responses[0].output),
        "first structured-tool response contained no StructuredToolCall",
    )
    _require("structured" in _event_input_kinds(run.events), "structured tool delta missing")
    history_types = _wire_input_types(wire_model, recording.requests[-1])
    _require(
        {"function_call", "function_call_output"}.issubset(history_types),
        "structured call/output were not replayed in complete history",
    )
    text = _visible_text(recording.responses[-1])
    _require("42" in text, "structured tool final answer did not use the tool result")
    return ScenarioReport(
        "structured_runtime_tool",
        _response_ids(recording.responses),
        _output_kinds(recording.responses),
        tuple(sorted(run.delta_kinds)),
        text,
        {"tool_calls": len(sum_tool.calls), "replayed_input_types": history_types},
    )


async def _apply_patch_scenario(config: LiveConfig) -> ScenarioReport:
    print(
        "[live] freeform apply_patch call -> in-memory tool -> final response",
        file=sys.stderr,
        flush=True,
    )
    wire_model = _model(config, "none")
    patch_tool = InMemoryApplyPatchTool()
    recording = RecordingModel(
        wire_model,
        first_turn_choice=ToolChoice(type="required"),
    )
    run = await _observe_start(
        Runtime(
            model=recording,
            tools=ToolRegistry((patch_tool,)),
            model_options=ModelOptions(max_output_tokens=512),
            tool_choice=ToolChoice(type="auto"),
            limits=_limits(),
        ),
        (
            Message.user(
                "Use apply_patch exactly once to propose adding a virtual file demo.txt whose "
                "only line is RESPONSE_PROTOCOL_OK. This is an in-memory protocol test."
            ),
        ),
        stream=True,
    )
    _require(len(patch_tool.inputs) == 1, "apply_patch was not invoked exactly once")
    patch_input = patch_tool.inputs[0]
    _require(
        "*** Begin Patch" in patch_input and "*** End Patch" in patch_input,
        "DeepSeek custom tool input was not an apply_patch payload",
    )
    _require(
        any(isinstance(item, FreeformToolCall) for item in recording.responses[0].output),
        "first apply_patch response contained no FreeformToolCall",
    )
    _require("freeform" in _event_input_kinds(run.events), "freeform tool delta missing")
    history_types = _wire_input_types(wire_model, recording.requests[-1])
    _require(
        {"custom_tool_call", "custom_tool_call_output"}.issubset(history_types),
        "custom call/output were not replayed in complete history",
    )
    text = _visible_text(recording.responses[-1])
    return ScenarioReport(
        "freeform_apply_patch",
        _response_ids(recording.responses),
        _output_kinds(recording.responses),
        tuple(sorted(run.delta_kinds)),
        text,
        {
            "patch_chars": len(patch_input),
            "filesystem_mutated": False,
            "replayed_input_types": history_types,
        },
    )


async def _web_search_scenario(config: LiveConfig) -> ScenarioReport:
    print(
        "[live] provider-hosted web search SSE + stateless follow-up",
        file=sys.stderr,
        flush=True,
    )
    marker = f"SEARCH-{secrets.token_hex(6).upper()}"
    wire_model = _model(config, "none")
    recording = RecordingModel(wire_model)
    provider_tool = deepseek_responses_web_search()
    search_run = await _observe_start(
        Runtime(
            model=recording,
            provider_tools=(provider_tool,),
            tool_choice=ToolChoice(
                type="provider",
                provider_tool=DEEPSEEK_RESPONSES_WEB_SEARCH,
            ),
            model_options=ModelOptions(max_output_tokens=2048),
            limits=_limits(),
        ),
        (
            Message.system(
                "Use the provider-hosted search tool for fresh web facts. Preserve source URLs."
            ),
            Message.user(
                f"Conversation marker {marker}. Search the public web for the current official "
                "DeepSeek Responses API guide title and URL on api-docs.deepseek.com. Perform "
                "the built-in search now; do not rely only on memory."
            ),
        ),
        stream=True,
    )
    first_response = recording.responses[-1]
    calls = first_response.provider_tool_calls()
    _require(bool(calls), "DeepSeek returned no provider-hosted web_search_call")
    _require(
        all(call.tool == DEEPSEEK_RESPONSES_WEB_SEARCH for call in calls),
        "web search response contained an unexpected provider tool identity",
    )
    completed_search_ids = {
        call.id
        for call in calls
        if call.status is ProviderToolStatus.COMPLETED and call.arguments.get("type") == "search"
    }
    _require(
        bool(completed_search_ids),
        "no provider-hosted search action completed",
    )
    _require(
        any(bool(call.arguments) for call in calls),
        "web search calls contained no decoded action/query arguments",
    )
    _require(
        "provider_tool_call" in search_run.delta_kinds,
        "web search stream exposed no provider_tool_call delta",
    )
    completed_search_done_count = sum(
        1
        for event in search_run.events
        if event.kind is EventKind.MODEL_DELTA
        and event.data.get("kind") == "provider_tool_call"
        and event.data.get("event") == "response.output_item.done"
        and event.data.get("status") == ProviderToolStatus.COMPLETED.value
        and event.data.get("id") in completed_search_ids
    )
    _require(
        completed_search_done_count > 0,
        "completed provider search call had no matching output_item.done delta",
    )
    lifecycle_actions = _provider_lifecycle_actions(
        search_run.events,
        "response.web_search_call.completed",
    )

    history = (*search_run.checkpoint.snapshot.history,)
    follow_up = Message.user(
        "Using the search results already present in the complete history, answer with the "
        "official guide title and full source URL. Also repeat the conversation marker from "
        "the earlier user message."
    )
    synthesis_run = await _observe_start(
        Runtime(
            model=recording,
            provider_tools=(provider_tool,),
            tool_choice=ToolChoice(type="none"),
            model_options=ModelOptions(max_output_tokens=512),
            limits=_limits(),
        ),
        (*history, follow_up),
        stream=True,
    )
    final_response = recording.responses[-1]
    text = _visible_text(final_response)
    _require(marker in text, "stateless follow-up lost the prior conversation marker")
    _require("Responses API" in text, "search synthesis did not return the guide title")
    replayed_request = recording.requests[-1]
    _require(
        replayed_request.messages == (*history, follow_up),
        "Runtime did not preserve the complete ordered message history",
    )
    replayed_items = _wire_input_items(wire_model, replayed_request)
    replayed_types = tuple(cast(str, item["type"]) for item in replayed_items)
    replayed_provider_items = tuple(
        item for item in replayed_items if item["type"] == "web_search_call"
    )
    _require(
        replayed_provider_items == _native_provider_history(calls),
        "native provider search history changed or was reordered during replay",
    )
    citation_urls = tuple(
        dict.fromkeys(
            (
                *_citation_urls(first_response),
                *_citation_urls(final_response),
            )
        )
    )
    source_urls = tuple(dict.fromkeys((*citation_urls, *extract_text_urls(text))))
    _require(
        _GUIDE_URL in source_urls,
        "canonical DeepSeek Responses guide URL did not survive search synthesis",
    )
    action_types = tuple(
        sorted(
            {
                cast(str, call.arguments.get("type"))
                for call in calls
                if isinstance(call.arguments.get("type"), str)
            }
        )
    )
    statuses = tuple(sorted({call.status.value for call in calls}))
    delta_kinds = search_run.delta_kinds.union(synthesis_run.delta_kinds)
    return ScenarioReport(
        "hosted_web_search_multiturn",
        _response_ids(recording.responses),
        _output_kinds(recording.responses),
        tuple(sorted(delta_kinds)),
        text,
        {
            "search_call_count": len(calls),
            "search_action_types": action_types,
            "search_statuses": statuses,
            "history_marker": marker,
            "completed_search_done_count": completed_search_done_count,
            "lifecycle_action_count": len(lifecycle_actions),
            "citation_urls": citation_urls,
            "source_urls": source_urls,
            "replayed_input_types": replayed_types,
            "stateless_history_message_count": len(recording.requests[-1].messages),
        },
    )


_SCENARIOS = {
    "basic": _basic_scenario,
    "reasoning": _reasoning_scenario,
    "structured-output": _structured_output_scenario,
    "function-tool": _function_tool_scenario,
    "apply-patch": _apply_patch_scenario,
    "web-search": _web_search_scenario,
}


async def _run_live(config: LiveConfig, only: str) -> tuple[ScenarioReport, ...]:
    selected = tuple(_SCENARIOS) if only == "all" else (only,)
    reports = [await _SCENARIOS[name](config) for name in selected]
    if only == "all":
        delta_kinds = frozenset(kind for report in reports for kind in report.delta_kinds)
        output_kinds = frozenset(kind for report in reports for kind in report.output_kinds)
        _require(
            _REQUIRED_DELTA_KINDS.issubset(delta_kinds),
            f"live delta coverage incomplete: {sorted(_REQUIRED_DELTA_KINDS - delta_kinds)}",
        )
        _require(
            _REQUIRED_OUTPUT_KINDS.issubset(output_kinds),
            f"live output coverage incomplete: {sorted(_REQUIRED_OUTPUT_KINDS - output_kinds)}",
        )
    return tuple(reports)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="run codec/profile contract checks without making network calls",
    )
    parser.add_argument(
        "--only",
        choices=("all", *_SCENARIOS),
        default="all",
        help="run one live scenario after the offline checks (default: all)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DEEPSEEK_BASE_URL", _DEFAULT_BASE_URL),
        help=f"DeepSeek API base URL (default: {_DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("DEEPSEEK_TIMEOUT_SECONDS", "180")),
        help="HTTP phase timeout in seconds (default: 180)",
    )
    return parser


def _live_config(arguments: argparse.Namespace) -> LiveConfig:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY must be set for live scenarios; use --offline-only otherwise"
        )
    base_url = cast(str, arguments.base_url).rstrip("/")
    timeout = cast(float, arguments.timeout)
    _require(bool(base_url), "--base-url must not be empty")
    _require(timeout > 0, "--timeout must be > 0")
    return LiveConfig(api_key=api_key, base_url=base_url, timeout=timeout)


async def _async_main(arguments: argparse.Namespace) -> dict[str, object]:
    print(
        "[offline] DeepSeek Responses profile and codec contract",
        file=sys.stderr,
        flush=True,
    )
    offline = run_offline_contract()
    if cast(bool, arguments.offline_only):
        return {"offline_contract": offline, "live": []}
    reports = await _run_live(_live_config(arguments), cast(str, arguments.only))
    live = [asdict(report) for report in reports]
    delta_kinds = sorted({kind for report in reports for kind in report.delta_kinds})
    output_kinds = sorted({kind for report in reports for kind in report.output_kinds})
    return {
        "offline_contract": offline,
        "live": live,
        "live_coverage": {
            "delta_kinds": delta_kinds,
            "output_kinds": output_kinds,
        },
    }


def main() -> None:
    arguments = _parser().parse_args()
    report = asyncio.run(_async_main(arguments))
    print(json.dumps(thaw_json_value(report), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
