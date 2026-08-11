# pyright: reportUnnecessaryTypeIgnoreComment=error

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from math import inf, nan
from typing import Any, TypeAlias, cast

import pytest

from jharness.kernel import (
    ContentPart,
    DeltaSink,
    Message,
    Model,
    ModelCapabilities,
    ModelContentDelta,
    ModelDelta,
    ModelError,
    ModelErrorInfo,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ProviderToolId,
    RunContext,
    Runtime,
)
from jharness.models.decorators import FallbackModel, RetryingModel


@dataclass(frozen=True, slots=True)
class _Invocation:
    request: ModelRequest
    context: RunContext
    stream: bool
    emit_delta: DeltaSink | None


@dataclass(frozen=True, slots=True)
class _EmitThen:
    delta: ModelDelta
    outcome: ModelResponse | BaseException


_Action: TypeAlias = ModelResponse | BaseException | _EmitThen


class _ScriptModel:
    def __init__(
        self,
        actions: Sequence[_Action],
        *,
        capabilities: ModelCapabilities | None = None,
        label: str = "",
        call_order: list[str] | None = None,
    ) -> None:
        self.actions = deque(actions)
        self.calls: list[_Invocation] = []
        self._capabilities = capabilities or ModelCapabilities(streaming=True)
        self.label = label
        self.call_order = call_order

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    async def invoke(
        self,
        request: ModelRequest,
        context: RunContext,
        *,
        stream: bool,
        emit_delta: DeltaSink | None,
    ) -> ModelResponse:
        self.calls.append(_Invocation(request, context, stream, emit_delta))
        if self.call_order is not None:
            self.call_order.append(self.label)
        action = self.actions.popleft()
        if isinstance(action, _EmitThen):
            if emit_delta is None:
                raise AssertionError("scripted delta requires a sink")
            await emit_delta(action.delta)
            if isinstance(action.outcome, BaseException):
                raise action.outcome
            return action.outcome
        if isinstance(action, BaseException):
            raise action
        return action


_REQUEST = ModelRequest(messages=(Message.user("hello"),))
_CONTEXT = RunContext("run-1", 1.0)


def _response(text: str) -> ModelResponse:
    return ModelResponse(output=(ContentPart.text_part(text),))


def _error(code: str, *, retryable: bool = True) -> ModelError:
    return ModelError(
        ModelErrorInfo(
            code=code,
            message=code,
            provider="test",
            retryable=retryable,
        )
    )


def _error_with_retry_after(value: object) -> ModelError:
    return ModelError(
        ModelErrorInfo(
            code="retry-after",
            message="retry-after",
            provider="test",
            retryable=True,
            metadata={"retry_after": value},
        )
    )


async def _invoke(
    model: Model,
    *,
    stream: bool = False,
    emit_delta: DeltaSink | None = None,
) -> ModelResponse:
    return await model.invoke(
        _REQUEST,
        _CONTEXT,
        stream=stream,
        emit_delta=emit_delta,
    )


async def test_retry_returns_first_success_without_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(
        "jharness.models.decorators._sleep_before_retry",
        record_sleep,
    )
    response = _response("primary")
    base = _ScriptModel((response,))
    model = RetryingModel(base, max_attempts=3)

    result = await _invoke(model)

    assert result is response
    assert len(base.calls) == 1
    assert sleeps == []
    assert model.capabilities is base.capabilities
    assert isinstance(model, Model)


async def test_retry_retries_only_retryable_model_errors_and_forwards_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(
        "jharness.models.decorators._sleep_before_retry",
        record_sleep,
    )
    response = _response("recovered")
    base = _ScriptModel((_error("first"), _error("second"), response))
    model = RetryingModel(
        base,
        max_attempts=3,
        backoff_initial_seconds=0.5,
        backoff_multiplier=2,
        backoff_max_seconds=0.75,
        jitter_ratio=0,
    )

    result = await _invoke(model)

    assert result is response
    assert sleeps == [0.5, 0.75]
    assert len(base.calls) == 3
    assert all(call.request is _REQUEST for call in base.calls)
    assert all(call.context is _CONTEXT for call in base.calls)
    assert all(call.stream is False for call in base.calls)
    assert all(call.emit_delta is None for call in base.calls)


@pytest.mark.parametrize(
    ("failure", "attempts"),
    (
        (_error("not-retryable", retryable=False), 3),
        (ValueError("protocol failure"), 3),
    ),
)
async def test_retry_propagates_non_retryable_failures_unchanged(
    failure: BaseException,
    attempts: int,
) -> None:
    base = _ScriptModel((failure, _response("unused")))

    with pytest.raises(type(failure)) as caught:
        await _invoke(
            RetryingModel(
                base,
                max_attempts=attempts,
                backoff_initial_seconds=0,
                backoff_max_seconds=0,
                jitter_ratio=0,
            )
        )

    assert caught.value is failure
    assert len(base.calls) == 1


async def test_retry_exhaustion_raises_the_last_model_error_unchanged() -> None:
    first = _error("first")
    last = _error("last")
    base = _ScriptModel((first, last))

    with pytest.raises(ModelError) as caught:
        await _invoke(
            RetryingModel(
                base,
                max_attempts=2,
                backoff_initial_seconds=0,
                backoff_max_seconds=0,
                jitter_ratio=0,
            )
        )

    assert caught.value is last
    assert len(base.calls) == 2


async def test_retry_propagates_cancellation_without_retry() -> None:
    cancellation = asyncio.CancelledError()
    base = _ScriptModel((cancellation, _response("unused")))

    with pytest.raises(asyncio.CancelledError) as caught:
        await _invoke(RetryingModel(base, max_attempts=2))

    assert caught.value is cancellation
    assert len(base.calls) == 1


async def test_cancellation_during_retry_backoff_stops_further_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeping = asyncio.Event()

    async def blocked_sleep(_delay: float) -> None:
        sleeping.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(
        "jharness.models.decorators._sleep_before_retry",
        blocked_sleep,
    )
    base = _ScriptModel((_error("retry"), _response("unused")))
    task = asyncio.create_task(_invoke(RetryingModel(base, max_attempts=2)))
    await sleeping.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(base.calls) == 1


@pytest.mark.parametrize(
    ("keywords", "pattern"),
    (
        ({"max_attempts": 0}, "max_attempts"),
        ({"max_attempts": -1}, "max_attempts"),
        ({"max_attempts": True}, "max_attempts"),
        ({"max_attempts": 1.5}, "max_attempts"),
        ({"backoff_initial_seconds": -1}, "backoff_initial_seconds"),
        ({"backoff_initial_seconds": inf}, "backoff_initial_seconds"),
        ({"backoff_initial_seconds": 10**400}, "backoff_initial_seconds"),
        ({"backoff_multiplier": 0.5}, "backoff_multiplier"),
        ({"backoff_multiplier": nan}, "backoff_multiplier"),
        (
            {"backoff_initial_seconds": 2, "backoff_max_seconds": 1},
            "backoff_max_seconds",
        ),
        ({"jitter_ratio": -0.1}, "jitter_ratio"),
        ({"jitter_ratio": 1.1}, "jitter_ratio"),
    ),
)
def test_retry_validates_policy_options(
    keywords: dict[str, object],
    pattern: str,
) -> None:
    with pytest.raises(ValueError, match=pattern):
        RetryingModel(_ScriptModel((_response("ok"),)), **cast(Any, keywords))


@pytest.mark.parametrize(
    ("retry_after", "backoff", "maximum", "jitter_ratio", "expected_bounds"),
    (
        (None, 5.0, 5.0, 0.2, (4.0, 5.0)),
        (2.0, 0.25, 5.0, 0.2, (2.0, 2.4)),
        (0.1, 0.25, 5.0, 0.2, (0.2, 0.3)),
        (5.0, 0.25, 5.0, 0.2, (5.0, 5.0)),
    ),
)
async def test_retry_samples_jitter_from_the_bounded_interval(
    monkeypatch: pytest.MonkeyPatch,
    retry_after: float | None,
    backoff: float,
    maximum: float,
    jitter_ratio: float,
    expected_bounds: tuple[float, float],
) -> None:
    bounds: list[tuple[float, float]] = []
    sleeps: list[float] = []

    def midpoint(lower: float, upper: float) -> float:
        bounds.append((lower, upper))
        return (lower + upper) / 2

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("jharness.models.decorators.uniform", midpoint)
    monkeypatch.setattr("jharness.models.decorators._sleep_before_retry", record_sleep)
    failure = _error("retry") if retry_after is None else _error_with_retry_after(retry_after)
    recovered = _response("recovered")
    model = RetryingModel(
        _ScriptModel((failure, recovered)),
        max_attempts=2,
        backoff_initial_seconds=backoff,
        backoff_max_seconds=maximum,
        jitter_ratio=jitter_ratio,
    )

    assert await _invoke(model) is recovered
    assert len(bounds) == 1
    assert bounds[0] == pytest.approx(expected_bounds)
    assert sleeps == pytest.approx([sum(expected_bounds) / 2])


@pytest.mark.parametrize(
    ("retry_after", "now", "maximum", "expected"),
    (
        (2.0, 100.0, 5.0, 2.0),
        ("3.5", 100.0, 5.0, 3.5),
        ("2", 100.0, 2.0, 2.0),
        ("Thu, 01 Jan 1970 00:01:42 GMT", 100.0, 5.0, 2.0),
        ("Thu, 01 Jan 1970 00:01:42", 100.0, 5.0, 0.5),
        ("", 100.0, 5.0, 0.5),
        ("invalid", 100.0, 5.0, 0.5),
        ("nan", 100.0, 5.0, 0.5),
        (-1, 100.0, 5.0, 0.5),
    ),
)
async def test_retry_honors_supported_retry_after_values(
    monkeypatch: pytest.MonkeyPatch,
    retry_after: object,
    now: float,
    maximum: float,
    expected: float,
) -> None:
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("jharness.models.decorators.wall_time", lambda: now)
    monkeypatch.setattr("jharness.models.decorators._sleep_before_retry", record_sleep)
    recovered = _response("recovered")
    base = _ScriptModel((_error_with_retry_after(retry_after), recovered))
    model = RetryingModel(
        base,
        max_attempts=2,
        backoff_initial_seconds=0.5,
        backoff_max_seconds=maximum,
        jitter_ratio=0,
    )

    assert await _invoke(model) is recovered
    assert sleeps == [expected]
    assert len(base.calls) == 2


@pytest.mark.parametrize(
    "retry_after",
    (
        17,
        "17",
        10**400,
        "inf",
        "1e999",
        "Thu, 01 Jan 1970 00:01:50 GMT",
    ),
)
async def test_retry_abandons_retry_after_beyond_backoff_cap(
    monkeypatch: pytest.MonkeyPatch,
    retry_after: object,
) -> None:
    failure = _error_with_retry_after(retry_after)
    base = _ScriptModel((failure, _response("too-early")))

    async def unexpected_sleep(_delay: float) -> None:
        raise AssertionError("cap-rejected retry must not sleep")

    def unexpected_uniform(_lower: float, _upper: float) -> float:
        raise AssertionError("cap-rejected retry must not sample jitter")

    monkeypatch.setattr("jharness.models.decorators.wall_time", lambda: 100.0)
    monkeypatch.setattr("jharness.models.decorators.uniform", unexpected_uniform)
    monkeypatch.setattr(
        "jharness.models.decorators._sleep_before_retry",
        unexpected_sleep,
    )
    model = RetryingModel(
        base,
        max_attempts=2,
        backoff_initial_seconds=0.5,
        backoff_max_seconds=2,
        jitter_ratio=0,
    )

    with pytest.raises(ModelError) as caught:
        await _invoke(model)

    assert caught.value is failure
    assert len(base.calls) == 1


async def test_retry_keeps_monotonic_budget_when_wall_clock_jumps_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop_now = 500.0
    sleeps: list[float] = []
    recovered = _response("recovered")
    base = _ScriptModel((_error("retry"), recovered))

    async def advance_loop(delay: float) -> None:
        nonlocal loop_now
        sleeps.append(delay)
        loop_now += delay

    monkeypatch.setattr(
        "jharness.models.decorators.wall_time",
        lambda: 100.0 if not base.calls else 10_000.0,
    )
    monkeypatch.setattr("jharness.models.decorators._loop_time", lambda: loop_now)
    monkeypatch.setattr("jharness.models.decorators._sleep_before_retry", advance_loop)
    context = RunContext("deadline-run", 99.0, deadline=101.0)
    model = RetryingModel(
        base,
        max_attempts=2,
        backoff_initial_seconds=0.5,
        backoff_max_seconds=0.5,
        jitter_ratio=0,
    )

    assert await model.invoke(_REQUEST, context, stream=False, emit_delta=None) is recovered
    assert sleeps == [0.5]
    assert len(base.calls) == 2


@pytest.mark.parametrize(
    ("deadline", "backoff", "retry_after"),
    (
        (100.0, 0.0, None),
        (100.5, 1.0, None),
        (101.0, 0.0, "2"),
    ),
)
async def test_retry_does_not_start_an_attempt_that_cannot_fit_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
    deadline: float,
    backoff: float,
    retry_after: str | None,
) -> None:
    failure = _error("retry") if retry_after is None else _error_with_retry_after(retry_after)
    base = _ScriptModel((failure, _response("too-late")))

    async def unexpected_sleep(_delay: float) -> None:
        raise AssertionError("deadline-rejected retry must not sleep")

    monkeypatch.setattr("jharness.models.decorators.wall_time", lambda: 100.0)
    monkeypatch.setattr(
        "jharness.models.decorators._sleep_before_retry",
        unexpected_sleep,
    )
    context = RunContext("deadline-run", 99.0, deadline=deadline)
    model = RetryingModel(
        base,
        max_attempts=2,
        backoff_initial_seconds=backoff,
        backoff_max_seconds=max(backoff, 5.0 if retry_after is not None else backoff),
        jitter_ratio=0,
    )

    with pytest.raises(ModelError) as caught:
        await model.invoke(_REQUEST, context, stream=False, emit_delta=None)

    assert caught.value is failure
    assert len(base.calls) == 1


async def test_retry_stops_if_deadline_expires_during_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wall_now = 100.0
    loop_now = 500.0
    failure = _error("retry")
    base = _ScriptModel((failure, _response("too-late")))

    async def expire_during_sleep(_delay: float) -> None:
        nonlocal loop_now, wall_now
        loop_now = 501.0
        wall_now = 0.0

    monkeypatch.setattr("jharness.models.decorators.wall_time", lambda: wall_now)
    monkeypatch.setattr("jharness.models.decorators._loop_time", lambda: loop_now)
    monkeypatch.setattr(
        "jharness.models.decorators._sleep_before_retry",
        expire_during_sleep,
    )
    context = RunContext("deadline-run", 99.0, deadline=101.0)
    model = RetryingModel(
        base,
        max_attempts=2,
        backoff_initial_seconds=0.5,
        backoff_max_seconds=0.5,
        jitter_ratio=0,
    )

    with pytest.raises(ModelError) as caught:
        await model.invoke(_REQUEST, context, stream=False, emit_delta=None)

    assert caught.value is failure
    assert len(base.calls) == 1


async def test_retry_is_allowed_before_the_first_delta() -> None:
    delta = ModelContentDelta(output_index=0, text_delta="visible", content_index=0)
    response = _response("done")
    base = _ScriptModel((_error("before-delta"), _EmitThen(delta, response)))
    observed: list[ModelDelta] = []

    async def emit_delta(item: ModelDelta, /) -> None:
        observed.append(item)

    result = await _invoke(
        RetryingModel(
            base,
            max_attempts=2,
            backoff_initial_seconds=0,
            backoff_max_seconds=0,
            jitter_ratio=0,
        ),
        stream=True,
        emit_delta=emit_delta,
    )

    assert result is response
    assert observed == [delta]
    assert len(base.calls) == 2
    assert base.calls[0].emit_delta is base.calls[1].emit_delta


async def test_retry_is_forbidden_after_the_first_delta() -> None:
    delta = ModelContentDelta(output_index=0, text_delta="partial", content_index=0)
    failure = _error("after-delta")
    base = _ScriptModel((_EmitThen(delta, failure), _response("unused")))
    observed: list[ModelDelta] = []

    async def emit_delta(item: ModelDelta, /) -> None:
        observed.append(item)

    with pytest.raises(ModelError) as caught:
        await _invoke(
            RetryingModel(base, max_attempts=2),
            stream=True,
            emit_delta=emit_delta,
        )

    assert caught.value is failure
    assert observed == [delta]
    assert len(base.calls) == 1


@pytest.mark.parametrize(
    "sink_failure",
    (
        ValueError("sink failed"),
        _error("sink-model-error"),
    ),
)
async def test_retry_propagates_sink_failure_unchanged_without_retry(
    sink_failure: BaseException,
) -> None:
    delta = ModelContentDelta(output_index=0, text_delta="partial", content_index=0)
    base = _ScriptModel((_EmitThen(delta, _response("unused")), _response("unused")))

    async def emit_delta(_item: ModelDelta, /) -> None:
        raise sink_failure

    with pytest.raises(type(sink_failure)) as caught:
        await _invoke(
            RetryingModel(base, max_attempts=2),
            stream=True,
            emit_delta=emit_delta,
        )

    assert caught.value is sink_failure
    assert len(base.calls) == 1


async def test_fallback_uses_backup_only_for_retryable_primary_failure() -> None:
    response = _response("fallback")
    primary = _ScriptModel((_error("primary"),))
    backup = _ScriptModel((response,))
    model = FallbackModel(primary, backup)

    result = await _invoke(model)

    assert result is response
    assert len(primary.calls) == 1
    assert len(backup.calls) == 1
    assert primary.calls[0].request is backup.calls[0].request is _REQUEST
    assert primary.calls[0].context is backup.calls[0].context is _CONTEXT
    assert primary.calls[0].emit_delta is backup.calls[0].emit_delta is None
    assert isinstance(model, Model)


async def test_fallback_does_not_call_backup_after_primary_success() -> None:
    response = _response("primary")
    primary = _ScriptModel((response,))
    backup = _ScriptModel((_response("unused"),))

    result = await _invoke(FallbackModel(primary, backup))

    assert result is response
    assert len(primary.calls) == 1
    assert backup.calls == []


@pytest.mark.parametrize(
    "failure",
    (
        _error("not-retryable", retryable=False),
        ValueError("protocol failure"),
        asyncio.CancelledError(),
    ),
)
async def test_fallback_propagates_non_retryable_failures_without_calling_backup(
    failure: BaseException,
) -> None:
    primary = _ScriptModel((failure,))
    backup = _ScriptModel((_response("unused"),))

    with pytest.raises(type(failure)) as caught:
        await _invoke(FallbackModel(primary, backup))

    assert caught.value is failure
    assert len(primary.calls) == 1
    assert backup.calls == []


async def test_fallback_raises_backup_failure_unchanged() -> None:
    primary_failure = _error("primary")
    backup_failure = _error("backup", retryable=False)
    primary = _ScriptModel((primary_failure,))
    backup = _ScriptModel((backup_failure,))

    with pytest.raises(ModelError) as caught:
        await _invoke(FallbackModel(primary, backup))

    assert caught.value is backup_failure
    assert len(primary.calls) == len(backup.calls) == 1


async def test_fallback_keeps_monotonic_budget_when_wall_clock_jumps_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop_now = 500.0
    response = _response("fallback")
    primary = _ScriptModel((_error("primary"),))
    backup = _ScriptModel((response,))
    monkeypatch.setattr(
        "jharness.models.decorators.wall_time",
        lambda: 100.0 if not primary.calls else 10_000.0,
    )
    monkeypatch.setattr("jharness.models.decorators._loop_time", lambda: loop_now)
    context = RunContext("deadline-run", 99.0, deadline=101.0)

    result = await FallbackModel(primary, backup).invoke(
        _REQUEST,
        context,
        stream=False,
        emit_delta=None,
    )

    assert result is response
    assert len(primary.calls) == len(backup.calls) == 1


async def test_fallback_stops_when_inner_retry_exhausts_monotonic_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wall_now = 100.0
    loop_now = 500.0
    failure = _error("primary")
    primary = _ScriptModel((failure, _response("too-late")))
    backup = _ScriptModel((_response("backup-too-late"),))

    async def expire_during_sleep(_delay: float) -> None:
        nonlocal loop_now, wall_now
        loop_now = 501.0
        wall_now = 0.0

    monkeypatch.setattr("jharness.models.decorators.wall_time", lambda: wall_now)
    monkeypatch.setattr("jharness.models.decorators._loop_time", lambda: loop_now)
    monkeypatch.setattr(
        "jharness.models.decorators._sleep_before_retry",
        expire_during_sleep,
    )
    context = RunContext("deadline-run", 99.0, deadline=101.0)
    model = FallbackModel(
        RetryingModel(
            primary,
            max_attempts=2,
            backoff_initial_seconds=0.5,
            backoff_max_seconds=0.5,
            jitter_ratio=0,
        ),
        backup,
    )

    with pytest.raises(ModelError) as caught:
        await model.invoke(_REQUEST, context, stream=False, emit_delta=None)

    assert caught.value is failure
    assert len(primary.calls) == 1
    assert backup.calls == []


async def test_fallback_does_not_start_backup_after_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = _error("primary")
    primary = _ScriptModel((failure,))
    backup = _ScriptModel((_response("too-late"),))
    monkeypatch.setattr("jharness.models.decorators.wall_time", lambda: 100.0)
    context = RunContext("deadline-run", 99.0, deadline=100.0)

    with pytest.raises(ModelError) as caught:
        await FallbackModel(primary, backup).invoke(
            _REQUEST,
            context,
            stream=False,
            emit_delta=None,
        )

    assert caught.value is failure
    assert len(primary.calls) == 1
    assert backup.calls == []


_BOOLEAN_CAPABILITY_FIELDS = (
    "streaming",
    "runtime_tools",
    "parallel_tool_calls",
    "parallel_tool_call_control",
    "structured_output",
    "json_mode",
    "seed",
    "usage_reporting",
)

_CAPABILITY_FIELDS = (
    "streaming",
    "runtime_tools",
    "tool_choice_types",
    "parallel_tool_calls",
    "parallel_tool_call_control",
    "input_modalities",
    "output_modalities",
    "provider_tools",
    "structured_output",
    "json_mode",
    "seed",
    "usage_reporting",
)


def test_fallback_capability_inventory_tracks_kernel_contract() -> None:
    assert tuple(field.name for field in fields(ModelCapabilities)) == _CAPABILITY_FIELDS


@pytest.mark.parametrize("field", _BOOLEAN_CAPABILITY_FIELDS)
def test_fallback_capabilities_are_the_safe_intersection(field: str) -> None:
    all_enabled = ModelCapabilities(streaming=True)
    changes: dict[str, object] = {field: False}
    if field == "runtime_tools":
        changes["tool_choice_types"] = frozenset({"auto", "none"})
    one_disabled = replace(all_enabled, **cast(Any, changes))
    primary_disabled = FallbackModel(
        _ScriptModel((_response("unused"),), capabilities=one_disabled),
        _ScriptModel((_response("unused"),), capabilities=all_enabled),
    ).capabilities
    backup_disabled = FallbackModel(
        _ScriptModel((_response("unused"),), capabilities=all_enabled),
        _ScriptModel((_response("unused"),), capabilities=one_disabled),
    ).capabilities

    assert getattr(primary_disabled, field) is False
    assert getattr(backup_disabled, field) is False
    assert all(
        getattr(primary_disabled, other) is True and getattr(backup_disabled, other) is True
        for other in _BOOLEAN_CAPABILITY_FIELDS
        if other != field
    )


def test_fallback_capabilities_intersect_exact_tool_choice_types() -> None:
    primary = ModelCapabilities(
        tool_choice_types=frozenset({"auto", "none", "required", "runtime"})
    )
    backup = ModelCapabilities(tool_choice_types=frozenset({"auto", "none"}))

    capabilities = FallbackModel(
        _ScriptModel((_response("unused"),), capabilities=primary),
        _ScriptModel((_response("unused"),), capabilities=backup),
    ).capabilities

    assert capabilities.tool_choice_types == frozenset({"auto", "none"})


def test_fallback_capabilities_intersect_modalities_and_provider_tools() -> None:
    web_search = ProviderToolId("test", "web_search")
    image_generation = ProviderToolId("test", "image_generation")
    primary = ModelCapabilities(
        input_modalities=frozenset({"text", "image"}),
        output_modalities=frozenset({"text", "image"}),
        provider_tools=frozenset({web_search, image_generation}),
    )
    backup = ModelCapabilities(
        input_modalities=frozenset({"text", "file"}),
        output_modalities=frozenset({"text"}),
        provider_tools=frozenset({web_search}),
    )

    capabilities = FallbackModel(
        _ScriptModel((_response("unused"),), capabilities=primary),
        _ScriptModel((_response("unused"),), capabilities=backup),
    ).capabilities

    assert capabilities.input_modalities == frozenset({"text"})
    assert capabilities.output_modalities == frozenset({"text"})
    assert capabilities.provider_tools == frozenset({web_search})


async def test_fallback_is_allowed_before_the_first_delta() -> None:
    delta = ModelContentDelta(output_index=0, text_delta="fallback", content_index=0)
    response = _response("done")
    primary = _ScriptModel((_error("before-delta"),))
    backup = _ScriptModel((_EmitThen(delta, response),))
    observed: list[ModelDelta] = []

    async def emit_delta(item: ModelDelta, /) -> None:
        observed.append(item)

    result = await _invoke(
        FallbackModel(primary, backup),
        stream=True,
        emit_delta=emit_delta,
    )

    assert result is response
    assert observed == [delta]
    assert len(primary.calls) == len(backup.calls) == 1
    assert primary.calls[0].emit_delta is backup.calls[0].emit_delta


async def test_fallback_is_forbidden_after_the_first_delta() -> None:
    delta = ModelContentDelta(output_index=0, text_delta="primary", content_index=0)
    failure = _error("after-delta")
    primary = _ScriptModel((_EmitThen(delta, failure),))
    backup = _ScriptModel((_response("unused"),))
    observed: list[ModelDelta] = []

    async def emit_delta(item: ModelDelta, /) -> None:
        observed.append(item)

    with pytest.raises(ModelError) as caught:
        await _invoke(
            FallbackModel(primary, backup),
            stream=True,
            emit_delta=emit_delta,
        )

    assert caught.value is failure
    assert observed == [delta]
    assert backup.calls == []


@pytest.mark.parametrize(
    "sink_failure",
    (
        ValueError("sink failed"),
        _error("sink-model-error"),
    ),
)
async def test_fallback_propagates_sink_failure_without_calling_backup(
    sink_failure: BaseException,
) -> None:
    primary = _ScriptModel(
        (
            _EmitThen(
                ModelContentDelta(output_index=0, text_delta="partial", content_index=0),
                _response("unused"),
            ),
        )
    )
    backup = _ScriptModel((_response("unused"),))

    async def emit_delta(_item: ModelDelta, /) -> None:
        raise sink_failure

    with pytest.raises(type(sink_failure)) as caught:
        await _invoke(
            FallbackModel(primary, backup),
            stream=True,
            emit_delta=emit_delta,
        )

    assert caught.value is sink_failure
    assert backup.calls == []


async def test_direct_retry_and_fallback_composition_uses_expected_call_order() -> None:
    order: list[str] = []
    primary = _ScriptModel(
        (_error("primary-1"), _error("primary-2")),
        label="primary",
        call_order=order,
    )
    backup = _ScriptModel(
        (_error("backup-1"), _response("recovered")),
        label="backup",
        call_order=order,
    )
    model = FallbackModel(
        RetryingModel(
            primary,
            max_attempts=2,
            backoff_initial_seconds=0,
            backoff_max_seconds=0,
            jitter_ratio=0,
        ),
        RetryingModel(
            backup,
            max_attempts=2,
            backoff_initial_seconds=0,
            backoff_max_seconds=0,
            jitter_ratio=0,
        ),
    )

    result = await _invoke(model)

    assert result.visible_parts()[0].text == "recovered"
    assert order == ["primary", "primary", "backup", "backup"]


async def test_nested_retry_delta_commit_prevents_outer_fallback() -> None:
    delta = ModelContentDelta(output_index=0, text_delta="committed", content_index=0)
    failure = _error("after-delta")
    primary = _ScriptModel((_EmitThen(delta, failure), _response("unused")))
    backup = _ScriptModel((_response("unused"),))
    observed: list[ModelDelta] = []

    async def emit_delta(item: ModelDelta, /) -> None:
        observed.append(item)

    model = FallbackModel(RetryingModel(primary, max_attempts=2), backup)
    with pytest.raises(ModelError) as caught:
        await _invoke(model, stream=True, emit_delta=emit_delta)

    assert caught.value is failure
    assert observed == [delta]
    assert len(primary.calls) == 1
    assert backup.calls == []


async def test_retrying_and_fallback_models_are_directly_composable() -> None:
    primary = _ScriptModel((_response("primary"),))
    backup = _ScriptModel((_response("backup"),))
    model = FallbackModel(
        RetryingModel(
            primary,
            max_attempts=2,
            backoff_initial_seconds=0,
            backoff_max_seconds=0,
            jitter_ratio=0,
        ),
        RetryingModel(
            backup,
            max_attempts=2,
            backoff_initial_seconds=0,
            backoff_max_seconds=0,
            jitter_ratio=0,
        ),
    )

    result = await _invoke(model)

    assert isinstance(model, Model)
    assert result.visible_parts()[0].text == "primary"
    assert len(primary.calls) == 1
    assert backup.calls == []


async def test_nested_fallback_models_follow_expression_order() -> None:
    order: list[str] = []
    primary = _ScriptModel(
        (_error("primary"),),
        label="primary",
        call_order=order,
    )
    second = _ScriptModel(
        (_error("second"),),
        label="second",
        call_order=order,
    )
    third = _ScriptModel(
        (_response("third"),),
        label="third",
        call_order=order,
    )
    model = FallbackModel(FallbackModel(primary, second), third)

    result = await _invoke(model)

    assert result.visible_parts()[0].text == "third"
    assert order == ["primary", "second", "third"]


def test_model_decorators_reject_invalid_models_at_the_boundary() -> None:
    valid = _ScriptModel((_response("ok"),))
    invalid = cast(Model, object())
    model_class = cast(Model, _ScriptModel)

    with pytest.raises(TypeError, match="model must implement Model"):
        RetryingModel(invalid)
    with pytest.raises(TypeError, match="model must implement Model"):
        RetryingModel(model_class)
    with pytest.raises(TypeError, match="primary must implement Model"):
        FallbackModel(invalid, valid)
    with pytest.raises(TypeError, match="backup must implement Model"):
        FallbackModel(valid, invalid)


async def test_runtime_observes_nested_attempts_as_one_logical_model_operation() -> None:
    primary = _ScriptModel((_error("first"), _error("second")))
    final = ModelResponse(
        output=(ContentPart.text_part("from fallback"),),
        usage=ModelUsage(input_tokens=2, output_tokens=3, total_tokens=5),
    )
    backup = _ScriptModel((final,))
    model = FallbackModel(
        RetryingModel(
            primary,
            max_attempts=2,
            backoff_initial_seconds=0,
            backoff_max_seconds=0,
            jitter_ratio=0,
        ),
        backup,
    )
    invocation = Runtime(model=model).start((Message.user("hello"),))
    events = invocation.events()
    result_task = asyncio.create_task(invocation.result())
    observed = [event async for event in events]
    checkpoint = await result_task

    assert checkpoint.snapshot.status == "completed"
    assert checkpoint.snapshot.metrics.planning_steps == 1
    assert checkpoint.snapshot.metrics.usage.total_tokens == 5
    assert len(checkpoint.snapshot.history) == 2
    assert checkpoint.snapshot.history[-1].visible_parts()[0].text == "from fallback"
    assert [event.kind.value for event in observed].count("model_started") == 1
    assert [event.kind.value for event in observed].count("model_finished") == 1
    assert len(primary.calls) == 2
    assert len(backup.calls) == 1
