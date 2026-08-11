"""Typed SSE event conversion for compatible Responses APIs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, cast

from jharness.kernel import (
    ModelContentDelta,
    ModelDelta,
    ModelProviderToolCallDelta,
    ModelReasoningDelta,
    ModelResponse,
    ModelToolCallDelta,
    ModelUsageDelta,
    ProviderToolId,
    ProviderToolStatus,
)
from jharness.models.openai.errors import OPENAI_RESPONSES_JSON, OpenAIResponsesError
from jharness.models.openai.profiles import OpenAIResponsesProfile
from jharness.models.openai.responses_api.codec import OpenAIResponsesCodec, decode_usage
from jharness.models.openai.responses_api.messages import provider_status


@dataclass(slots=True)
class _ItemState:
    item_id: str
    item_type: str
    call_id: str | None = None
    name: str | None = None
    status: ProviderToolStatus | None = None
    closed: bool = False


@dataclass(slots=True)
class _PartState:
    part_type: str
    closed: bool = False


_PROVIDER_TERMINAL_STATUSES = frozenset(
    {
        ProviderToolStatus.COMPLETED,
        ProviderToolStatus.INCOMPLETE,
        ProviderToolStatus.FAILED,
    }
)


class OpenAIResponsesStreamDecoder:
    """Validate a Responses event stream and expose portable live deltas."""

    def __init__(
        self,
        codec: OpenAIResponsesCodec,
        profile: OpenAIResponsesProfile,
    ) -> None:
        self._codec = codec
        self._profile = profile
        self._phase: Literal["initial", "active", "terminal"] = "initial"
        self._response_id: str | None = None
        self._sequence_number: int | None = None
        self._items: dict[int, _ItemState] = {}
        self._parts: dict[tuple[int, int], _PartState] = {}
        self._summary_parts: dict[tuple[int, int], _PartState] = {}
        self._done_events: set[tuple[str, int, int]] = set()
        self._response: ModelResponse | None = None

    def apply_event(
        self,
        event_name: str | None,
        value: Mapping[str, Any],
    ) -> tuple[bool, list[ModelDelta]]:
        """Apply one typed event and report whether it is terminal."""

        if self._phase == "terminal":
            raise OpenAIResponsesError("Responses stream emitted an event after termination")
        event_type = _event_type(event_name, value)
        self._validate_sequence_number(value)
        if event_type == "response.created":
            self._response_event(value, expected_status="in_progress", created=True)
            return False, []
        if event_type == "response.in_progress":
            self._response_event(value, expected_status="in_progress", created=False)
            return False, []
        self._require_started(event_type)
        handler = self._event_handlers().get(event_type)
        if handler is not None:
            return False, handler(value)
        if event_type.startswith("response.image_generation_call."):
            return False, self._image_generation_event(event_type, value)
        if event_type.startswith("response.web_search_call."):
            return False, self._web_search_event(event_type, value)
        if event_type in {
            "response.completed",
            "response.incomplete",
            "response.failed",
        }:
            return True, self._terminal_event(event_type, value)
        if event_type == "error":
            raise OpenAIResponsesError("Responses stream emitted an error event")
        raise OpenAIResponsesError(f"unsupported Responses stream event type: {event_type}")

    def _event_handlers(self) -> Mapping[str, Any]:
        return {
            "response.output_item.added": self._output_item_added,
            "response.output_item.done": self._output_item_done,
            "response.content_part.added": partial(self._content_part_event, done=False),
            "response.content_part.done": partial(self._content_part_event, done=True),
            "response.reasoning_summary_part.added": partial(
                self._reasoning_summary_part_event,
                done=False,
            ),
            "response.reasoning_summary_part.done": partial(
                self._reasoning_summary_part_event,
                done=True,
            ),
            "response.output_text.delta": partial(self._content_delta, part_type="text"),
            "response.output_text.annotation.added": self._output_text_annotation_added,
            "response.refusal.delta": partial(self._content_delta, part_type="refusal"),
            "response.reasoning_text.delta": partial(
                self._reasoning_delta,
                "response.reasoning_text.delta",
            ),
            "response.reasoning_summary_text.delta": partial(
                self._reasoning_delta,
                "response.reasoning_summary_text.delta",
            ),
            "response.function_call_arguments.delta": self._function_arguments_delta,
            **{
                event_type: partial(self._validate_done_event, event_type)
                for event_type in (
                    "response.output_text.done",
                    "response.refusal.done",
                    "response.reasoning_text.done",
                    "response.reasoning_summary_text.done",
                    "response.function_call_arguments.done",
                )
            },
        }

    def completed_response(self) -> ModelResponse:
        """Return the terminal full response, never a delta reconstruction."""

        if self._phase != "terminal" or self._response is None:
            raise OpenAIResponsesError("Responses stream ended before a terminal response")
        return self._response

    def _response_event(
        self,
        value: Mapping[str, Any],
        *,
        expected_status: str,
        created: bool,
    ) -> None:
        if created:
            if self._phase != "initial":
                raise OpenAIResponsesError("Responses stream response.created appeared twice")
        elif self._phase == "initial":
            raise OpenAIResponsesError("Responses response.in_progress requires response.created")
        response = OPENAI_RESPONSES_JSON.mapping(
            value.get("response"),
            "Responses stream response",
        )
        if response.get("object") != "response" or response.get("status") != expected_status:
            raise OpenAIResponsesError(
                "Responses stream response requires object='response' and "
                f"status={expected_status!r}"
            )
        response_id = OPENAI_RESPONSES_JSON.required_string(
            response.get("id"),
            "Responses stream response id",
        )
        if self._response_id is not None and self._response_id != response_id:
            raise OpenAIResponsesError("Responses stream response id changed")
        self._response_id = response_id
        self._phase = "active"

    def _output_item_added(self, value: Mapping[str, Any]) -> list[ModelDelta]:
        output_index = _output_index(value)
        if output_index != len(self._items):
            raise OpenAIResponsesError("Responses output items must be added in index order")
        item = OPENAI_RESPONSES_JSON.mapping(
            value.get("item"),
            "Responses output item",
        )
        item_id = OPENAI_RESPONSES_JSON.required_string(
            item.get("id"),
            "Responses output item id",
        )
        item_type = OPENAI_RESPONSES_JSON.required_string(
            item.get("type"),
            "Responses output item type",
        )
        if any(state.item_id == item_id for state in self._items.values()):
            raise OpenAIResponsesError("Responses output item id must be unique")
        state = _ItemState(item_id, item_type)
        self._items[output_index] = state
        if item_type in {"message", "reasoning"}:
            return []
        if item_type == "function_call":
            state.call_id = OPENAI_RESPONSES_JSON.required_string(
                item.get("call_id"),
                "Responses function call call_id",
            )
            state.name = OPENAI_RESPONSES_JSON.required_string(
                item.get("name"),
                "Responses function call name",
            )
            arguments = item.get("arguments", "")
            if not isinstance(arguments, str):
                raise OpenAIResponsesError("Responses function call arguments must be a string")
            return [
                ModelToolCallDelta(
                    output_index=output_index,
                    arguments_delta=arguments,
                    id=state.call_id,
                    name=state.name,
                )
            ]
        if item_type in {"image_generation_call", "web_search_call"}:
            state.status = provider_status(
                item.get("status"),
                label=f"Responses {item_type} status",
            )
            return [
                self._provider_delta(
                    output_index,
                    state,
                    event="response.output_item.added",
                    data={},
                )
            ]
        raise OpenAIResponsesError(f"unsupported Responses streamed output item: {item_type}")

    def _output_item_done(self, value: Mapping[str, Any]) -> list[ModelDelta]:
        output_index = _output_index(value)
        state = self._open_item(output_index)
        item = OPENAI_RESPONSES_JSON.mapping(
            value.get("item"),
            "Responses completed output item",
        )
        self._validate_item_identity(state, item)
        if any(
            index == output_index and not part.closed for (index, _), part in self._parts.items()
        ) or any(
            index == output_index and not part.closed
            for (index, _), part in self._summary_parts.items()
        ):
            raise OpenAIResponsesError("Responses output item completed with open content parts")
        state.closed = True
        if state.item_type not in {"image_generation_call", "web_search_call"}:
            return []
        updated = provider_status(
            item.get("status"),
            label=f"Responses {state.item_type} status",
        )
        if state.status is updated:
            return []
        state.status = updated
        return [
            self._provider_delta(
                output_index,
                state,
                event="response.output_item.done",
                data={},
            )
        ]

    def _content_part_event(
        self,
        value: Mapping[str, Any],
        *,
        done: bool,
    ) -> list[ModelDelta]:
        output_index = _output_index(value)
        content_index = _content_index(value)
        state = self._open_item(output_index)
        allowed_types = {
            "message": frozenset({"output_text", "refusal"}),
            "reasoning": frozenset({"reasoning_text"}),
        }.get(state.item_type)
        if allowed_types is None:
            raise OpenAIResponsesError(
                "Responses content parts require a message or reasoning item"
            )
        _validate_item_id(value, state)
        part = OPENAI_RESPONSES_JSON.mapping(
            value.get("part"),
            "Responses content part",
        )
        part_type = OPENAI_RESPONSES_JSON.required_string(
            part.get("type"),
            "Responses content part type",
        )
        if part_type not in allowed_types:
            raise OpenAIResponsesError(f"unsupported Responses content part: {part_type}")
        key = (output_index, content_index)
        existing = self._parts.get(key)
        if not done:
            if existing is not None:
                raise OpenAIResponsesError("Responses content part was added twice")
            self._parts[key] = _PartState(part_type)
            return []
        if existing is None or existing.closed:
            raise OpenAIResponsesError("Responses content_part.done requires an open part")
        if existing.part_type != part_type:
            raise OpenAIResponsesError("Responses content part type changed")
        existing.closed = True
        return []

    def _reasoning_summary_part_event(
        self,
        value: Mapping[str, Any],
        *,
        done: bool,
    ) -> list[ModelDelta]:
        output_index = _output_index(value)
        summary_index = _nonnegative_int(
            value.get("summary_index"),
            "Responses reasoning summary_index",
        )
        state = self._open_item(output_index)
        if state.item_type != "reasoning":
            raise OpenAIResponsesError("Responses reasoning summary requires a reasoning item")
        _validate_item_id(value, state)
        part = OPENAI_RESPONSES_JSON.mapping(
            value.get("part"),
            "Responses reasoning summary part",
        )
        if part.get("type") != "summary_text" or not isinstance(part.get("text"), str):
            raise OpenAIResponsesError("Responses reasoning summary requires summary_text")
        key = (output_index, summary_index)
        existing = self._summary_parts.get(key)
        if not done:
            if existing is not None:
                raise OpenAIResponsesError("Responses reasoning summary part was added twice")
            self._summary_parts[key] = _PartState("summary_text")
            return []
        if existing is None or existing.closed:
            raise OpenAIResponsesError(
                "Responses reasoning_summary_part.done requires an open part"
            )
        existing.closed = True
        return []

    def _content_delta(
        self,
        value: Mapping[str, Any],
        *,
        part_type: str,
    ) -> list[ModelDelta]:
        output_index = _output_index(value)
        content_index = _content_index(value)
        state = self._open_item(output_index)
        if state.item_type != "message":
            raise OpenAIResponsesError("Responses text deltas require a message item")
        _validate_item_id(value, state)
        expected_wire_type = "output_text" if part_type == "text" else "refusal"
        part = self._parts.get((output_index, content_index))
        if part is None or part.closed or part.part_type != expected_wire_type:
            raise OpenAIResponsesError("Responses content delta requires a matching open part")
        delta = value.get("delta")
        if not isinstance(delta, str):
            raise OpenAIResponsesError("Responses content delta requires a string")
        if not delta:
            return []
        return [
            ModelContentDelta(
                output_index=output_index,
                content_index=content_index,
                part_type=part_type,
                text_delta=delta,
            )
        ]

    def _output_text_annotation_added(self, value: Mapping[str, Any]) -> list[ModelDelta]:
        output_index = _output_index(value)
        content_index = _content_index(value)
        state = self._open_item(output_index)
        if state.item_type != "message":
            raise OpenAIResponsesError("Responses output text annotation requires a message item")
        _validate_item_id(value, state)
        part = self._parts.get((output_index, content_index))
        if part is None or part.closed or part.part_type != "output_text":
            raise OpenAIResponsesError(
                "Responses output text annotation requires an open output_text part"
            )
        _nonnegative_int(
            value.get("annotation_index"),
            "Responses annotation_index",
        )
        annotation = OPENAI_RESPONSES_JSON.mapping(
            value.get("annotation"),
            "Responses output text annotation",
        )
        OPENAI_RESPONSES_JSON.required_string(
            annotation.get("type"),
            "Responses output text annotation type",
        )
        return []

    def _reasoning_delta(
        self,
        event_type: str,
        value: Mapping[str, Any],
    ) -> list[ModelDelta]:
        output_index = _output_index(value)
        state = self._open_item(output_index)
        if state.item_type != "reasoning":
            raise OpenAIResponsesError("Responses reasoning delta requires a reasoning item")
        _validate_item_id(value, state)
        field = (
            "summary_index"
            if event_type == "response.reasoning_summary_text.delta"
            else "content_index"
        )
        raw_index = value.get(field, 0)
        content_index = _nonnegative_int(raw_index, f"Responses reasoning {field}")
        if field == "summary_index":
            part = self._summary_parts.get((output_index, content_index))
            if part is None or part.closed:
                raise OpenAIResponsesError(
                    "Responses reasoning summary delta requires an open summary part"
                )
        else:
            part = self._parts.get((output_index, content_index))
            if part is None or part.closed or part.part_type != "reasoning_text":
                raise OpenAIResponsesError(
                    "Responses reasoning delta requires an open reasoning_text part"
                )
        delta = value.get("delta")
        if not isinstance(delta, str):
            raise OpenAIResponsesError("Responses reasoning delta requires a string")
        return (
            [
                ModelReasoningDelta(
                    output_index=output_index,
                    content_index=content_index,
                    text_delta=delta,
                )
            ]
            if delta
            else []
        )

    def _function_arguments_delta(self, value: Mapping[str, Any]) -> list[ModelDelta]:
        output_index = _output_index(value)
        state = self._open_item(output_index)
        if state.item_type != "function_call" or state.call_id is None or state.name is None:
            raise OpenAIResponsesError(
                "Responses function arguments delta requires a function_call item"
            )
        _validate_item_id(value, state)
        delta = value.get("delta")
        if not isinstance(delta, str):
            raise OpenAIResponsesError("Responses function arguments delta requires a string")
        return (
            [ModelToolCallDelta(output_index=output_index, arguments_delta=delta)] if delta else []
        )

    def _image_generation_event(
        self,
        event_type: str,
        value: Mapping[str, Any],
    ) -> list[ModelDelta]:
        output_index, state = self._provider_event_item(value, "image_generation_call")
        suffix = event_type.removeprefix("response.image_generation_call.")
        if suffix == "partial_image":
            image = OPENAI_RESPONSES_JSON.required_string(
                value.get("partial_image_b64"),
                "Responses partial image base64",
            )
            partial_index = _nonnegative_int(
                value.get("partial_image_index"),
                "Responses partial image index",
            )
            if partial_index > 3:
                raise OpenAIResponsesError("Responses partial image index must be at most 3")
            return self._changed_provider_status(
                output_index,
                state,
                ProviderToolStatus.IN_PROGRESS,
                event_type,
                data={"base64": image, "partial_image_index": partial_index},
            )
        status = _provider_event_status(suffix, "image generation")
        return self._changed_provider_status(output_index, state, status, event_type)

    def _web_search_event(
        self,
        event_type: str,
        value: Mapping[str, Any],
    ) -> list[ModelDelta]:
        output_index, state = self._provider_event_item(value, "web_search_call")
        suffix = event_type.removeprefix("response.web_search_call.")
        status = _provider_event_status(suffix, "web search")
        data: dict[str, object] = {}
        if isinstance(value.get("action"), Mapping):
            data["action"] = value["action"]
        return self._changed_provider_status(
            output_index,
            state,
            status,
            event_type,
            data=data,
        )

    def _provider_event_item(
        self,
        value: Mapping[str, Any],
        expected_type: str,
    ) -> tuple[int, _ItemState]:
        output_index = _output_index(value)
        state = self._open_item(output_index)
        if state.item_type != expected_type:
            raise OpenAIResponsesError(f"Responses provider event requires {expected_type} item")
        _validate_item_id(value, state)
        return output_index, state

    def _changed_provider_status(
        self,
        output_index: int,
        state: _ItemState,
        status: ProviderToolStatus,
        event_type: str,
        *,
        data: Mapping[str, object] | None = None,
    ) -> list[ModelDelta]:
        current = state.status
        if current in _PROVIDER_TERMINAL_STATUSES:
            if status is ProviderToolStatus.IN_PROGRESS:
                raise OpenAIResponsesError(
                    "Responses provider tool status cannot return to in_progress"
                )
            if status is not current:
                raise OpenAIResponsesError(
                    "Responses provider tool lifecycle emitted conflicting terminal statuses"
                )
            return []
        state.status = status
        return [
            self._provider_delta(
                output_index,
                state,
                event=event_type,
                data={} if data is None else data,
            )
        ]

    def _provider_delta(
        self,
        output_index: int,
        state: _ItemState,
        *,
        event: str,
        data: Mapping[str, object],
    ) -> ModelProviderToolCallDelta:
        return ModelProviderToolCallDelta(
            output_index=output_index,
            id=state.item_id,
            tool=self._provider_tool(
                "image_generation" if state.item_type == "image_generation_call" else "web_search"
            ),
            status=state.status,
            event=event,
            data=data,
        )

    def _provider_tool(self, tool_type: str) -> ProviderToolId:
        try:
            return self._profile.provider_tool(tool_type)
        except ValueError as exc:
            raise OpenAIResponsesError(str(exc)) from exc

    def _validate_done_event(
        self,
        event_type: str,
        value: Mapping[str, Any],
    ) -> list[ModelDelta]:
        output_index = _output_index(value)
        state = self._open_item(output_index)
        _validate_item_id(value, state)
        expected = {
            "response.function_call_arguments.done": "function_call",
            "response.output_text.done": "message",
            "response.reasoning_summary_text.done": "reasoning",
            "response.reasoning_text.done": "reasoning",
            "response.refusal.done": "message",
        }[event_type]
        if state.item_type != expected:
            raise OpenAIResponsesError(f"Responses {event_type} does not match its output item")
        part_index = -1
        if event_type == "response.reasoning_summary_text.done":
            part_index = _nonnegative_int(
                value.get("summary_index"),
                "Responses reasoning summary_index",
            )
            part = self._summary_parts.get((output_index, part_index))
            if part is None or part.closed or part.part_type != "summary_text":
                raise OpenAIResponsesError(
                    "Responses reasoning summary done requires an open summary_text part"
                )
        elif event_type != "response.function_call_arguments.done":
            part_index = _content_index(value)
            expected_part_type = {
                "response.output_text.done": "output_text",
                "response.reasoning_text.done": "reasoning_text",
                "response.refusal.done": "refusal",
            }[event_type]
            part = self._parts.get((output_index, part_index))
            if part is None or part.closed or part.part_type != expected_part_type:
                raise OpenAIResponsesError(
                    f"Responses {event_type} requires a matching open content part"
                )
        done_key = (event_type, output_index, part_index)
        if done_key in self._done_events:
            raise OpenAIResponsesError(f"Responses {event_type} was emitted twice")
        field = {
            "response.function_call_arguments.done": "arguments",
            "response.output_text.done": "text",
            "response.reasoning_summary_text.done": "text",
            "response.reasoning_text.done": "text",
            "response.refusal.done": "refusal",
        }[event_type]
        if not isinstance(value.get(field), str):
            raise OpenAIResponsesError(f"Responses {event_type} requires {field}")
        self._done_events.add(done_key)
        return []

    def _terminal_event(
        self,
        event_type: str,
        value: Mapping[str, Any],
    ) -> list[ModelDelta]:
        if any(not item.closed for item in self._items.values()):
            raise OpenAIResponsesError("Responses stream terminated with open output items")
        if any(not part.closed for part in self._summary_parts.values()):
            raise OpenAIResponsesError("Responses stream terminated with open reasoning parts")
        response = OPENAI_RESPONSES_JSON.mapping(
            value.get("response"),
            "Responses terminal stream response",
        )
        expected_status = event_type.removeprefix("response.")
        if response.get("status") != expected_status:
            raise OpenAIResponsesError("Responses terminal event type must match response status")
        response_id = OPENAI_RESPONSES_JSON.required_string(
            response.get("id"),
            "Responses terminal response id",
        )
        if self._response_id != response_id:
            raise OpenAIResponsesError("Responses terminal response id changed")
        self._validate_terminal_output(response.get("output"))
        decoded = self._codec.decode_response(response)
        self._response = decoded
        self._phase = "terminal"
        usage = decode_usage(response.get("usage"))
        return [] if usage is None else [ModelUsageDelta(usage)]

    def _validate_terminal_output(self, value: object) -> None:
        if not _is_array(value):
            raise OpenAIResponsesError("Responses terminal output must be an array")
        items = cast(Sequence[object], value)
        if len(items) != len(self._items):
            raise OpenAIResponsesError("Responses terminal output does not match streamed items")
        for output_index, raw_item in enumerate(items):
            state = self._items.get(output_index)
            if state is None:
                raise OpenAIResponsesError("Responses terminal output index was not streamed")
            item = OPENAI_RESPONSES_JSON.mapping(raw_item, "Responses terminal output item")
            self._validate_item_identity(state, item)

    @staticmethod
    def _validate_item_identity(state: _ItemState, item: Mapping[str, Any]) -> None:
        if item.get("id") != state.item_id or item.get("type") != state.item_type:
            raise OpenAIResponsesError("Responses streamed output item identity changed")
        if state.item_type == "function_call" and (
            item.get("call_id") != state.call_id or item.get("name") != state.name
        ):
            raise OpenAIResponsesError("Responses streamed function identity changed")

    def _open_item(self, output_index: int) -> _ItemState:
        state = self._items.get(output_index)
        if state is None or state.closed:
            raise OpenAIResponsesError(
                f"Responses event requires an open output item at index {output_index}"
            )
        return state

    def _require_started(self, event_type: str) -> None:
        if self._phase == "initial":
            raise OpenAIResponsesError(f"Responses {event_type} requires response.created")

    def _validate_sequence_number(self, value: Mapping[str, Any]) -> None:
        raw = value.get("sequence_number")
        if raw is None:
            return
        sequence_number = _nonnegative_int(raw, "Responses sequence_number")
        if self._sequence_number is not None and sequence_number <= self._sequence_number:
            raise OpenAIResponsesError("Responses sequence_number must increase")
        self._sequence_number = sequence_number


def _event_type(event_name: str | None, value: Mapping[str, Any]) -> str:
    event_type = OPENAI_RESPONSES_JSON.required_string(
        value.get("type"),
        "Responses stream event type",
    )
    if event_name is not None and event_name != event_type:
        raise OpenAIResponsesError("Responses SSE event name must match the payload type")
    return event_type


def _output_index(value: Mapping[str, Any]) -> int:
    return _nonnegative_int(value.get("output_index"), "Responses output_index")


def _content_index(value: Mapping[str, Any]) -> int:
    return _nonnegative_int(value.get("content_index"), "Responses content_index")


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpenAIResponsesError(f"{label} must be a non-negative integer")
    return value


def _validate_item_id(value: Mapping[str, Any], state: _ItemState) -> None:
    item_id = OPENAI_RESPONSES_JSON.required_string(
        value.get("item_id"),
        "Responses event item_id",
    )
    if item_id != state.item_id:
        raise OpenAIResponsesError("Responses event item_id changed")


def _provider_event_status(suffix: str, label: str) -> ProviderToolStatus:
    if suffix in {"in_progress", "generating", "searching"}:
        return ProviderToolStatus.IN_PROGRESS
    if suffix == "completed":
        return ProviderToolStatus.COMPLETED
    if suffix == "incomplete":
        return ProviderToolStatus.INCOMPLETE
    if suffix == "failed":
        return ProviderToolStatus.FAILED
    raise OpenAIResponsesError(f"unsupported Responses {label} event: {suffix}")


def _is_array(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)
