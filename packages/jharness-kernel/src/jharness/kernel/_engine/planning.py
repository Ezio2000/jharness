"""One model effect translated into one typed Change."""

from __future__ import annotations

from collections.abc import Awaitable, Mapping
from time import time
from typing import Any, Protocol, cast

from jharness.kernel._engine.change import Change, failed, insert, limited, suspend
from jharness.kernel._engine.deadline import (
    Deadline,
    EffectInterrupted,
    WorkDeadlineReached,
    await_effect,
)
from jharness.kernel._validation import expect_instance
from jharness.kernel.checkpoint import ModelTurnFact, ModelTurnResult
from jharness.kernel.control import ControlInbox, Insert, Pause
from jharness.kernel.errors import ModelError
from jharness.kernel.events import EventKind
from jharness.kernel.limits import LimitReason, RunLimits
from jharness.kernel.messages import (
    ContentPart,
    Message,
    ProviderToolCall,
    ProviderToolStatus,
    ToolCall,
)
from jharness.kernel.models import (
    Model,
    ModelCapabilities,
    ModelContentDelta,
    ModelDelta,
    ModelOptions,
    ModelProviderToolCallDelta,
    ModelReasoningDelta,
    ModelRequest,
    ModelResponse,
    ModelToolCallDelta,
    ModelUsage,
    ModelUsageDelta,
    ProviderToolSpec,
    ResponseFormat,
    ToolChoice,
)
from jharness.kernel.snapshot import RunSnapshot
from jharness.kernel.state import (
    Completed,
    Limited,
    PendingToolCalls,
    Planning,
    ToolsPending,
)
from jharness.kernel.tools import ToolCatalog

_TEXT_LIKE = frozenset({"text", "reasoning", "thinking", "redacted_thinking", "refusal"})


class Emit(Protocol):
    def __call__(self, kind: EventKind, data: Mapping[str, Any]) -> Awaitable[None]: ...


class PlanningStep:
    __slots__ = (
        "_capabilities",
        "_catalog",
        "_emit",
        "_limits",
        "_model",
        "_options",
        "_provider_tools",
        "_response_format",
        "_stream",
        "_tool_choice",
    )

    def __init__(
        self,
        *,
        model: Model,
        capabilities: ModelCapabilities,
        catalog: ToolCatalog,
        limits: RunLimits,
        options: ModelOptions,
        provider_tools: tuple[ProviderToolSpec, ...],
        tool_choice: ToolChoice,
        response_format: ResponseFormat | None,
        stream: bool,
        emit: Emit,
    ) -> None:
        self._model = model
        self._capabilities = capabilities
        self._catalog = catalog
        self._limits = limits
        self._options = options
        self._provider_tools = provider_tools
        self._tool_choice = tool_choice
        self._response_format = response_format
        self._stream = stream
        self._emit = emit

    async def run(
        self, snapshot: RunSnapshot, *, deadline: Deadline, inbox: ControlInbox
    ) -> Change:
        try:
            request = ModelRequest(
                messages=tuple(snapshot.history),
                runtime_tools=self._catalog.specs(),
                provider_tools=self._provider_tools,
                options=self._options,
                tool_choice=self._tool_choice,
                response_format=self._response_format,
            )
            await self._emit(
                EventKind.MODEL_STARTED,
                {"planning_step": snapshot.metrics.planning_steps + 1},
            )
            response = await self._invoke(request, snapshot, deadline, inbox)
        except EffectInterrupted as interrupted:
            return _interrupted(interrupted)
        except WorkDeadlineReached:
            return limited(LimitReason.DEADLINE)
        except ModelError as exc:
            return failed("model_provider_error", exc.info.message)
        except Exception as exc:
            return failed("model_protocol_error", str(exc) or exc.__class__.__name__)

        await self._emit(
            EventKind.MODEL_FINISHED,
            {
                "finish_reason": response.finish_reason,
                "runtime_tool_call_count": len(response.runtime_tool_calls()),
                "provider_tool_call_count": len(response.provider_tool_calls()),
                "usage": usage_data(response.usage),
            },
        )
        return self._change(snapshot, response)

    async def _invoke(
        self,
        request: ModelRequest,
        snapshot: RunSnapshot,
        deadline: Deadline,
        inbox: ControlInbox,
    ) -> ModelResponse:
        _validate_request(request, self._capabilities)
        use_stream = self._stream and self._capabilities.streaming

        async def emit_delta(delta: ModelDelta) -> None:
            await self._emit(EventKind.MODEL_DELTA, delta_data(delta))

        response = await await_effect(
            self._model.invoke(
                request,
                snapshot.context,
                stream=use_stream,
                emit_delta=emit_delta if use_stream else None,
            ),
            deadline=deadline,
            inbox=inbox,
        )
        response = expect_instance(response, ModelResponse, "model response")
        _validate_response(response, request, self._capabilities)
        return response

    def _change(self, snapshot: RunSnapshot, response: ModelResponse) -> Change:
        calls = response.runtime_tool_calls()
        parts = response.visible_parts()
        total_tokens = snapshot.metrics.usage.total_tokens
        if response.usage is not None and response.usage.total_tokens is not None:
            total_tokens = (total_tokens or 0) + response.usage.total_tokens
        over_tokens = (
            self._limits.max_total_tokens is not None
            and total_tokens is not None
            and total_tokens > self._limits.max_total_tokens
        )
        if over_tokens:
            state = Limited(LimitReason.MAX_TOTAL_TOKENS)
        elif calls:
            state = ToolsPending(PendingToolCalls(calls))
        else:
            state = Completed(parts)
        return Change(
            fact=ModelTurnFact(
                at=time(),
                result=ModelTurnResult(state.kind),
                part_count=len(parts),
                tool_call_ids=tuple(call.id for call in calls),
                finish_reason=response.finish_reason,
                usage=response.usage,
                limit_reason=state.reason if isinstance(state, Limited) else None,
            ),
            state=state,
            append=(response.to_assistant_message(),),
            planning_steps=1,
            usage=response.usage,
        )


def _interrupted(interrupted: EffectInterrupted) -> Change:
    control = interrupted.control
    if isinstance(control, Pause):
        return suspend(Planning(), control.suspension)
    if isinstance(control, Insert):
        return insert(control)
    raise TypeError("unsupported planning interruption")


def usage_data(usage: ModelUsage | None) -> dict[str, int | None] | None:
    if usage is None:
        return None
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "reasoning_tokens": usage.reasoning_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
    }


def delta_data(delta: ModelDelta) -> Mapping[str, Any]:
    if isinstance(delta, ModelContentDelta):
        return {
            "kind": "content",
            "output_index": delta.output_index,
            "content_index": delta.content_index,
            "part_type": delta.part_type,
            "text_delta": delta.text_delta,
            "data": delta.data,
        }
    if isinstance(delta, ModelToolCallDelta):
        return {
            "kind": "tool_call",
            "output_index": delta.output_index,
            "id": delta.id,
            "name": delta.name,
            "arguments_delta": delta.arguments_delta,
        }
    if isinstance(delta, ModelReasoningDelta):
        return {
            "kind": "reasoning",
            "output_index": delta.output_index,
            "content_index": delta.content_index,
            "text_delta": delta.text_delta,
        }
    if isinstance(delta, ModelProviderToolCallDelta):
        return {
            "kind": "provider_tool_call",
            "output_index": delta.output_index,
            "id": delta.id,
            "tool": {
                "namespace": delta.tool.namespace,
                "type": delta.tool.type,
            },
            "status": None if delta.status is None else delta.status.value,
            "event": delta.event,
            "data": delta.data,
        }
    if not isinstance(cast(object, delta), ModelUsageDelta):
        raise TypeError("model emitted an invalid delta")
    return {"kind": "usage", "usage": usage_data(delta.usage)}


def _validate_request(request: ModelRequest, capabilities: ModelCapabilities) -> None:
    _validate_request_tools(request, capabilities)
    if request.options.seed is not None and not capabilities.seed:
        raise ValueError("model does not support seed")
    response_format = request.response_format
    if response_format is not None:
        if response_format.type == "json_object" and not capabilities.json_mode:
            raise ValueError("model does not support JSON object mode")
        if response_format.type == "json_schema" and not capabilities.structured_output:
            raise ValueError("model does not support structured output")
    # Provider-tool output is governed by the requested ProviderToolId, not by
    # the model's direct output modalities.
    unsupported_modalities = {
        _part_modality(part)
        for message in request.messages
        for part in _message_input_parts(message)
        if _part_modality(part) not in capabilities.input_modalities
    }
    if unsupported_modalities:
        raise ValueError(
            "model does not support input modalities: " + ", ".join(sorted(unsupported_modalities))
        )


def _validate_request_tools(request: ModelRequest, capabilities: ModelCapabilities) -> None:
    if request.runtime_tools and not capabilities.runtime_tools:
        raise ValueError("model does not support runtime tools")
    if request.provider_tools:
        unsupported = tuple(
            spec.tool
            for spec in request.provider_tools
            if spec.tool not in capabilities.provider_tools
        )
        if unsupported:
            raise ValueError("model does not support requested provider tools")
    if request.tool_choice.type not in capabilities.tool_choice_types:
        raise ValueError(f"model does not support tool_choice={request.tool_choice.type!r}")
    has_tools = bool(request.runtime_tools or request.provider_tools)
    if (
        has_tools
        and request.tool_choice.type != "none"
        and not request.tool_choice.allow_parallel_tool_calls
        and not capabilities.parallel_tool_call_control
    ):
        raise ValueError("model cannot disable parallel tool calls")


def _validate_response(
    response: ModelResponse,
    request: ModelRequest,
    capabilities: ModelCapabilities,
) -> None:
    calls = response.runtime_tool_calls()
    provider_calls = response.provider_tool_calls()
    _validate_response_tools(calls, provider_calls, request, capabilities)
    _validate_response_modalities(response, capabilities)
    if not calls and not provider_calls and not response.visible_parts():
        raise ValueError("terminal model response requires visible output")


def _validate_response_tools(
    calls: tuple[ToolCall, ...],
    provider_calls: tuple[ProviderToolCall, ...],
    request: ModelRequest,
    capabilities: ModelCapabilities,
) -> None:
    if calls and not capabilities.runtime_tools:
        raise ValueError("model returned unsupported runtime tool calls")
    requested_provider_tools = {spec.tool for spec in request.provider_tools}
    if any(call.tool not in capabilities.provider_tools for call in provider_calls):
        raise ValueError("model returned an unsupported provider tool call")
    if any(call.tool not in requested_provider_tools for call in provider_calls):
        raise ValueError("model returned an unrequested provider tool call")
    if any(call.status is ProviderToolStatus.IN_PROGRESS for call in provider_calls):
        raise ValueError("terminal model response contains an in-progress provider tool call")
    if len(calls) > 1 and (
        not capabilities.parallel_tool_calls or not request.tool_choice.allow_parallel_tool_calls
    ):
        raise ValueError("model returned disallowed parallel runtime tool calls")
    _validate_response_tool_choice(calls, provider_calls, request.tool_choice)


def _validate_response_tool_choice(
    calls: tuple[ToolCall, ...],
    provider_calls: tuple[ProviderToolCall, ...],
    choice: ToolChoice,
) -> None:
    all_calls = (*calls, *provider_calls)
    if choice.type == "none" and all_calls:
        raise ValueError("model returned tool calls for tool_choice=none")
    if choice.type == "required" and not all_calls:
        raise ValueError("model omitted required tool call")
    if choice.type == "runtime" and (
        provider_calls or not calls or any(call.name != choice.name for call in calls)
    ):
        raise ValueError("model returned a tool other than the selected runtime tool")
    if choice.type == "provider" and (
        calls
        or not provider_calls
        or any(call.tool != choice.provider_tool for call in provider_calls)
    ):
        raise ValueError("model returned a tool other than the selected provider tool")


def _validate_response_modalities(
    response: ModelResponse,
    capabilities: ModelCapabilities,
) -> None:
    unsupported_modalities = {
        _part_modality(item)
        for item in response.output
        if isinstance(item, ContentPart)
        and _part_modality(item) not in capabilities.output_modalities
    }
    if unsupported_modalities:
        raise ValueError(
            "model returned unsupported output modalities: "
            + ", ".join(sorted(unsupported_modalities))
        )


def _message_input_parts(message: Message) -> tuple[ContentPart, ...]:
    if message.role == "assistant":
        return tuple(item for item in message.output if isinstance(item, ContentPart))
    if message.role == "tool" and message.outcome is not None:
        return message.outcome.parts
    return message.parts


def _part_modality(part: ContentPart) -> str:
    if part.type in _TEXT_LIKE:
        return "text"
    if part.type in {"image", "input_image", "output_image"}:
        return "image"
    if part.type in {"audio", "input_audio", "output_audio"}:
        return "audio"
    if part.type in {"video", "input_video", "output_video"}:
        return "video"
    media_type = part.media_type or (None if part.artifact is None else part.artifact.media_type)
    if media_type is not None:
        family = media_type.partition("/")[0]
        if family in {"image", "audio", "video"}:
            return family
    return "file"
