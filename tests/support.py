"""Small kernel-only helpers for runtime integration tests."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from jharness.kernel import (
    ApprovalAllow,
    ApprovalDecision,
    ApprovalDeny,
    ApprovalPolicy,
    ApprovalRequest,
    Checkpoint,
    DeltaSink,
    Event,
    Invocation,
    Model,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    RunContext,
)

ResponseFactory = Callable[[int, ModelRequest], ModelResponse]


class DeterministicModel(Model):
    def __init__(self, respond: ResponseFactory) -> None:
        self._respond = respond
        self.turns = 0

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()

    async def invoke(
        self,
        request: ModelRequest,
        context: RunContext,
        *,
        stream: bool,
        emit_delta: DeltaSink | None,
    ) -> ModelResponse:
        del context, stream, emit_delta
        response = self._respond(self.turns, request)
        self.turns += 1
        return response


class AllowAll(ApprovalPolicy):
    async def decide(self, requests: tuple[ApprovalRequest, ...]) -> tuple[ApprovalDecision, ...]:
        return tuple(ApprovalAllow(request.call.id) for request in requests)


class DenyAll(ApprovalPolicy):
    async def decide(self, requests: tuple[ApprovalRequest, ...]) -> tuple[ApprovalDecision, ...]:
        return tuple(ApprovalDeny(request.call.id, "test denial") for request in requests)


async def collect_invocation(invocation: Invocation) -> tuple[Checkpoint, list[Event]]:
    events = invocation.events()
    result_task = asyncio.create_task(invocation.result())
    observed = [event async for event in events]
    return await result_task, observed
