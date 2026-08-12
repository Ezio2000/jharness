"""Composable retry and fallback models."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from math import inf, isfinite
from random import uniform
from time import time as wall_time

from jharness.kernel import (
    DeltaSink,
    Model,
    ModelCapabilities,
    ModelDelta,
    ModelError,
    ModelRequest,
    ModelResponse,
    RunContext,
)

__all__ = ["FallbackModel", "RetryingModel"]


@dataclass(frozen=True, slots=True)
class RetryingModel:
    """Retry retryable ``ModelError`` failures before a stream publishes a delta."""

    model: Model
    max_attempts: int = 2
    backoff_initial_seconds: float = 0.25
    backoff_multiplier: float = 2.0
    backoff_max_seconds: float = 5.0
    jitter_ratio: float = 0.2

    def __post_init__(self) -> None:
        _require_model(self.model, "model")
        _validate_retry_options(
            max_attempts=self.max_attempts,
            backoff_initial_seconds=self.backoff_initial_seconds,
            backoff_multiplier=self.backoff_multiplier,
            backoff_max_seconds=self.backoff_max_seconds,
            jitter_ratio=self.jitter_ratio,
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        return self.model.capabilities

    async def invoke(
        self,
        request: ModelRequest,
        context: RunContext,
        *,
        stream: bool,
        emit_delta: DeltaSink | None,
    ) -> ModelResponse:
        deadline = _monotonic_deadline(context)
        relay = _DeltaRelay(emit_delta) if emit_delta is not None else None
        attempt = 1
        backoff = self.backoff_initial_seconds
        while True:
            try:
                return await self.model.invoke(
                    request,
                    context,
                    stream=stream,
                    emit_delta=relay,
                )
            except ModelError as exc:
                if relay is not None and relay.started:
                    raise
                if not exc.info.retryable or attempt >= self.max_attempts:
                    raise
                delay = _retry_delay(
                    backoff,
                    error=exc,
                    jitter_ratio=self.jitter_ratio,
                    maximum=self.backoff_max_seconds,
                )
                if delay is None or not _retry_fits_deadline(delay, deadline):
                    raise
                await _sleep_before_retry(delay)
                if _deadline_expired(deadline):
                    raise
                attempt += 1
                backoff = min(
                    self.backoff_max_seconds,
                    backoff * self.backoff_multiplier,
                )


@dataclass(frozen=True, slots=True)
class FallbackModel:
    """Use a fallback model after a retryable failure before any published delta."""

    primary: Model
    backup: Model

    def __post_init__(self) -> None:
        _require_model(self.primary, "primary")
        _require_model(self.backup, "backup")

    @property
    def capabilities(self) -> ModelCapabilities:
        primary = self.primary.capabilities
        backup = self.backup.capabilities
        provider_tools = primary.provider_tools & backup.provider_tools
        runtime_tool_kinds = primary.runtime_tool_kinds & backup.runtime_tool_kinds
        tool_choice_types = primary.tool_choice_types & backup.tool_choice_types
        if not provider_tools:
            tool_choice_types = tool_choice_types.difference({"provider"})
        if not runtime_tool_kinds:
            tool_choice_types = tool_choice_types.difference({"runtime"})
        return ModelCapabilities(
            streaming=primary.streaming and backup.streaming,
            runtime_tool_kinds=runtime_tool_kinds,
            tool_choice_types=tool_choice_types,
            parallel_runtime_tool_calls=(
                primary.parallel_runtime_tool_calls and backup.parallel_runtime_tool_calls
            ),
            parallel_runtime_tool_call_control=(
                primary.parallel_runtime_tool_call_control
                and backup.parallel_runtime_tool_call_control
            ),
            input_modalities=primary.input_modalities & backup.input_modalities,
            output_modalities=primary.output_modalities & backup.output_modalities,
            provider_tools=provider_tools,
            structured_output=primary.structured_output and backup.structured_output,
            json_mode=primary.json_mode and backup.json_mode,
            seed=primary.seed and backup.seed,
            usage_reporting=primary.usage_reporting and backup.usage_reporting,
        )

    async def invoke(
        self,
        request: ModelRequest,
        context: RunContext,
        *,
        stream: bool,
        emit_delta: DeltaSink | None,
    ) -> ModelResponse:
        deadline = _monotonic_deadline(context)
        relay = _DeltaRelay(emit_delta) if emit_delta is not None else None
        try:
            return await self.primary.invoke(
                request,
                context,
                stream=stream,
                emit_delta=relay,
            )
        except ModelError as exc:
            if relay is not None and relay.started:
                raise
            if not exc.info.retryable:
                raise
            if _deadline_expired(deadline):
                raise
        return await self.backup.invoke(
            request,
            context,
            stream=stream,
            emit_delta=relay,
        )


@dataclass(slots=True)
class _DeltaRelay:
    sink: DeltaSink
    started: bool = False

    async def __call__(self, delta: ModelDelta, /) -> None:
        self.started = True
        await self.sink(delta)


def _require_model(value: object, label: str) -> Model:
    if isinstance(value, type) or not isinstance(value, Model):
        raise TypeError(f"{label} must implement Model")
    return value


def _validate_retry_options(
    *,
    max_attempts: object,
    backoff_initial_seconds: object,
    backoff_multiplier: object,
    backoff_max_seconds: object,
    jitter_ratio: object,
) -> None:
    if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    initial = _nonnegative_finite(backoff_initial_seconds, "backoff_initial_seconds")
    multiplier = _finite(backoff_multiplier, "backoff_multiplier")
    if multiplier < 1:
        raise ValueError("backoff_multiplier must be >= 1")
    maximum = _nonnegative_finite(backoff_max_seconds, "backoff_max_seconds")
    if maximum < initial:
        raise ValueError("backoff_max_seconds must be >= backoff_initial_seconds")
    jitter = _finite(jitter_ratio, "jitter_ratio")
    if not 0 <= jitter <= 1:
        raise ValueError("jitter_ratio must be between 0 and 1")


def _nonnegative_finite(value: object, label: str) -> float:
    number = _finite(value, label)
    if number < 0:
        raise ValueError(f"{label} must be nonnegative")
    return number


def _finite(value: object, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _retry_delay(
    backoff: float,
    *,
    error: ModelError,
    jitter_ratio: float,
    maximum: float,
) -> float | None:
    retry_after = _retry_after_seconds(error)
    if retry_after is not None and retry_after > maximum:
        return None
    retry_floor = 0.0 if retry_after is None else retry_after
    baseline = max(backoff, retry_floor)
    lower = max(0.0, backoff * (1 - jitter_ratio), retry_floor)
    upper = min(maximum, baseline * (1 + jitter_ratio))
    return uniform(lower, upper)


def _retry_after_seconds(error: ModelError) -> float | None:
    raw = error.info.metadata.get("retry_after")
    if isinstance(raw, int | float) and not isinstance(raw, bool):
        try:
            seconds = float(raw)
        except OverflowError:
            return inf if raw > 0 else None
        return seconds if seconds >= 0 else None
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        seconds = None
    if seconds is not None:
        return seconds if seconds >= 0 else None
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            return None
        return max(0.0, parsed.timestamp() - wall_time())
    except (OverflowError, TypeError, ValueError):
        return None


async def _sleep_before_retry(delay: float) -> None:
    await asyncio.sleep(delay)


def _loop_time() -> float:
    return asyncio.get_running_loop().time()


def _monotonic_deadline(context: RunContext) -> float | None:
    if context.deadline is None:
        return None
    return _loop_time() + max(0.0, context.deadline - wall_time())


def _retry_fits_deadline(delay: float, deadline: float | None) -> bool:
    if deadline is None:
        return True
    return delay < deadline - _loop_time()


def _deadline_expired(deadline: float | None) -> bool:
    return deadline is not None and _loop_time() >= deadline
