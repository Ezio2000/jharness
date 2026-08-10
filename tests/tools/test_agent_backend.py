from __future__ import annotations

import asyncio
from time import time

import pytest

from jharness.kernel import (
    ContentPart,
    DeltaSink,
    Model,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    RunContext,
    RunLimits,
    Runtime,
    ToolCall,
)
from jharness.toolkit import ToolRegistry
from jharness.tools import AskQuestionTool
from jharness.tools.agent import (
    AgentBackend,
    AgentBackendError,
    AgentRequest,
    InMemoryAgentBackend,
)


class _StaticModel(Model):
    def __init__(self, response: ModelResponse | Exception) -> None:
        self.response = response

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
        del request, context, stream, emit_delta
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class _ControlledModel(Model):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False
        self.requests: list[tuple[ModelRequest, RunContext]] = []

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
        del stream, emit_delta
        self.requests.append((request, context))
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return ModelResponse((ContentPart.text_part("child result"),), finish_reason="stop")


def _parent(run_id: str = "parent", *, depth: int | None = None) -> RunContext:
    metadata: dict[str, object] = {"tenant": "test"}
    if depth is not None:
        metadata["agent_depth"] = depth
    return RunContext(run_id, time(), time() + 60, metadata=metadata)


def _backend(
    model: _ControlledModel,
    *,
    system_prompt: str | None = None,
) -> InMemoryAgentBackend:
    return InMemoryAgentBackend(Runtime(model=model), system_prompt=system_prompt)


async def test_in_memory_backend_concurrent_start_is_idempotent() -> None:
    model = _ControlledModel()
    backend = _backend(model)
    request = AgentRequest("delegate", "do the work")
    parent = _parent()

    first, second = await asyncio.gather(
        backend.start_or_get(request, parent=parent, parent_tool_call_id="call"),
        backend.start_or_get(request, parent=parent, parent_tool_call_id="call"),
    )

    assert isinstance(backend, AgentBackend)
    assert first == second
    await asyncio.wait_for(model.started.wait(), timeout=1)
    assert len(model.requests) == 1

    model.release.set()
    terminal = await backend.wait_for_terminal(first.agent_id, requester=parent)
    assert terminal.status == "completed"
    assert terminal.result == "child result"


async def test_in_memory_backend_rejects_conflicting_idempotency_request() -> None:
    model = _ControlledModel()
    backend = _backend(model)
    parent = _parent()
    first = await backend.start_or_get(
        AgentRequest("first", "first prompt"),
        parent=parent,
        parent_tool_call_id="call",
    )

    with pytest.raises(AgentBackendError, match="different Agent request") as raised:
        await backend.start_or_get(
            AgentRequest("second", "second prompt"),
            parent=parent,
            parent_tool_call_id="call",
        )
    assert raised.value.code == "agent_conflict"

    model.release.set()
    await backend.wait_for_terminal(first.agent_id, requester=parent)


async def test_in_memory_backend_hides_agents_from_other_runs() -> None:
    model = _ControlledModel()
    backend = _backend(model)
    owner = _parent("owner")
    intruder = _parent("intruder")
    snapshot = await backend.start_or_get(
        AgentRequest("private", "private prompt", background=True),
        parent=owner,
        parent_tool_call_id="call",
    )

    operations = (
        backend.get(snapshot.agent_id, requester=intruder),
        backend.wait_or_get(
            snapshot.agent_id,
            requester=intruder,
            requester_tool_call_id="wait",
        ),
        backend.cancel(
            snapshot.agent_id,
            requester=intruder,
            requester_tool_call_id="cancel",
        ),
        backend.wait_for_terminal(snapshot.agent_id, requester=intruder),
    )
    for operation in operations:
        with pytest.raises(AgentBackendError, match="Agent not found") as raised:
            await operation
        assert raised.value.code == "agent_not_found"

    model.release.set()
    await backend.wait_for_terminal(snapshot.agent_id, requester=owner)


async def test_in_memory_backend_cancel_is_concurrent_and_idempotent() -> None:
    model = _ControlledModel()
    backend = _backend(model)
    parent = _parent()
    running = await backend.start_or_get(
        AgentRequest("cancel", "keep working"),
        parent=parent,
        parent_tool_call_id="call",
    )
    await asyncio.wait_for(model.started.wait(), timeout=1)

    first, second = await asyncio.gather(
        backend.cancel(
            running.agent_id,
            requester=parent,
            requester_tool_call_id="cancel-1",
        ),
        backend.cancel(
            running.agent_id,
            requester=parent,
            requester_tool_call_id="cancel-2",
        ),
    )

    assert first == second
    assert first.status == "cancelled"
    assert first.cancellation_requested is True
    assert model.cancelled is True
    assert (
        await backend.cancel(
            running.agent_id,
            requester=parent,
            requester_tool_call_id="cancel-3",
        )
        == first
    )


async def test_in_memory_backend_maps_child_failures_and_limits() -> None:
    parent = _parent()
    failed_backend = InMemoryAgentBackend(Runtime(model=_StaticModel(RuntimeError("boom"))))
    failed = await failed_backend.start_or_get(
        AgentRequest("failed", "fail"),
        parent=parent,
        parent_tool_call_id="failed",
    )
    failed = await failed_backend.wait_for_terminal(failed.agent_id, requester=parent)
    assert failed.status == "failed"
    assert failed.error is not None
    assert failed.error.code == "model_protocol_error"
    assert failed.error.message == "boom"

    limited_backend = InMemoryAgentBackend(
        Runtime(
            model=_StaticModel(
                ModelResponse(
                    (ContentPart.text_part("too expensive"),),
                    usage=ModelUsage(total_tokens=2),
                )
            ),
            limits=RunLimits(max_total_tokens=1),
        )
    )
    limited = await limited_backend.start_or_get(
        AgentRequest("limited", "use tokens"),
        parent=parent,
        parent_tool_call_id="limited",
    )
    limited = await limited_backend.wait_for_terminal(limited.agent_id, requester=parent)
    assert limited.status == "failed"
    assert limited.error is not None
    assert limited.error.code == "child_limited"
    assert "max_total_tokens" in limited.error.message


async def test_in_memory_backend_maps_unsupported_child_suspension() -> None:
    model = _StaticModel(
        ModelResponse(
            tool_calls=(
                ToolCall(
                    "ask",
                    "AskQuestion",
                    {"questions": [{"id": "confirm", "kind": "confirm", "prompt": "Continue?"}]},
                ),
            )
        )
    )
    backend = InMemoryAgentBackend(Runtime(model=model, tools=ToolRegistry((AskQuestionTool(),))))
    parent = _parent()
    started = await backend.start_or_get(
        AgentRequest("suspend", "ask for input"),
        parent=parent,
        parent_tool_call_id="suspend",
    )
    terminal = await backend.wait_for_terminal(started.agent_id, requester=parent)

    assert terminal.status == "failed"
    assert terminal.error is not None
    assert terminal.error.code == "child_suspended"


async def test_in_memory_backend_maps_supervision_failure_without_leaking_details() -> None:
    model = _ControlledModel()
    backend = _backend(model)
    expired_parent = RunContext("expired", 1.0, 1.0)
    started = await backend.start_or_get(
        AgentRequest("expired", "cannot start"),
        parent=expired_parent,
        parent_tool_call_id="expired",
    )
    terminal = await backend.wait_for_terminal(started.agent_id, requester=expired_parent)

    assert terminal.status == "failed"
    assert terminal.error is not None
    assert terminal.error.code == "child_host_error"
    assert terminal.error.message == "Child Agent supervision failed."


async def test_in_memory_backend_builds_child_messages_and_nested_context() -> None:
    model = _ControlledModel()
    model.release.set()
    backend = _backend(model, system_prompt="trusted child policy")
    parent = _parent(depth=2)
    started = await backend.start_or_get(
        AgentRequest("nested", "child prompt"),
        parent=parent,
        parent_tool_call_id="delegate",
    )
    terminal = await backend.wait_for_terminal(started.agent_id, requester=parent)

    assert terminal.status == "completed"
    request, context = model.requests[0]
    assert [message.role for message in request.messages] == ["system", "user"]
    assert request.messages[0].parts[0].text == "trusted child policy"
    assert request.messages[1].parts[0].text == "child prompt"
    assert context.parent_run_id == parent.run_id
    assert context.parent_tool_call_id == "delegate"
    assert context.deadline == parent.deadline
    assert context.run_kind == "agent"
    assert context.metadata["tenant"] == "test"
    assert context.metadata["agent_depth"] == 3
    assert context.metadata["agent_id"] == started.agent_id
