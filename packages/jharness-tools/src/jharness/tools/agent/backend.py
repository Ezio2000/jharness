"""Agent execution port and the default in-process supervisor."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace
from time import time
from typing import Protocol, cast, runtime_checkable
from uuid import uuid4

from jharness.kernel import (
    Completed,
    ErrorInfo,
    Failed,
    Invocation,
    Limited,
    Message,
    RunContext,
    Runtime,
    Suspended,
    Suspension,
)
from jharness.tools.agent.models import AgentBackendError, AgentRequest, AgentSnapshot

_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_LOGGER = logging.getLogger(__name__)


@runtime_checkable
class AgentBackend(Protocol):
    """Create, observe, wait for, and cancel Host-owned child Agents.

    Implementations own authorization, idempotency, persistence, supervision, and
    deriving a Child Runtime from trusted Host configuration. Methods must be safe for
    concurrent calls from multiple immutable tool instances.
    """

    async def start_or_get(
        self,
        request: AgentRequest,
        *,
        parent: RunContext,
        parent_tool_call_id: str,
    ) -> AgentSnapshot:
        """Idempotently create or return the Agent for one parent tool call.

        For foreground requests, creation must also establish durable completion
        delivery so a fast Child cannot race the parent's waiting checkpoint.
        """

        ...

    async def get(
        self,
        agent_id: str,
        *,
        requester: RunContext,
    ) -> AgentSnapshot:
        """Return the current Agent snapshot without waiting."""

        ...

    async def wait_or_get(
        self,
        agent_id: str,
        *,
        requester: RunContext,
        requester_tool_call_id: str,
    ) -> AgentSnapshot:
        """Atomically establish completion delivery or return a terminal snapshot."""

        ...

    async def cancel(
        self,
        agent_id: str,
        *,
        requester: RunContext,
        requester_tool_call_id: str,
    ) -> AgentSnapshot:
        """Idempotently request cancellation and return the resulting snapshot."""

        ...


@dataclass(slots=True)
class _AgentRecord:
    request: AgentRequest
    owner_run_id: str
    snapshot: AgentSnapshot
    child_context: RunContext
    started: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    invocation: Invocation | None = None
    task: asyncio.Task[None] | None = None
    cancellation_sent: bool = False


class InMemoryAgentBackend:
    """Supervise child Agents within one process and one asyncio event loop.

    ``child_runtime`` is an immutable Runtime configured by the trusted Host and can
    safely create concurrent single-use child Invocations. Agent snapshots survive
    parent suspension but not process restart. Hosts needing crash recovery or shared
    workers should provide another :class:`AgentBackend` implementation. Records are
    retained for this backend's lifetime to preserve idempotent retry results.
    """

    __slots__ = (
        "_by_start_key",
        "_child_runtime",
        "_lock",
        "_records",
        "_system_prompt",
    )

    def __init__(
        self,
        child_runtime: Runtime,
        *,
        system_prompt: str | None = None,
    ) -> None:
        if not isinstance(cast(object, child_runtime), Runtime):
            raise TypeError("child_runtime must be a Runtime")
        if system_prompt is not None:
            if not isinstance(cast(object, system_prompt), str):
                raise TypeError("system_prompt must be a string or None")
            if not system_prompt:
                raise ValueError("system_prompt must not be empty")
        self._child_runtime = child_runtime
        self._system_prompt = system_prompt
        self._records: dict[str, _AgentRecord] = {}
        self._by_start_key: dict[tuple[str, str], str] = {}
        self._lock = asyncio.Lock()

    async def start_or_get(
        self,
        request: AgentRequest,
        *,
        parent: RunContext,
        parent_tool_call_id: str,
    ) -> AgentSnapshot:
        """Start one child Invocation, idempotently keyed by its parent tool call."""

        request = _request(request)
        parent = _context(parent, "parent")
        parent_tool_call_id = _non_empty_string(parent_tool_call_id, "parent_tool_call_id")
        key = (parent.run_id, parent_tool_call_id)

        async with self._lock:
            existing_id = self._by_start_key.get(key)
            if existing_id is None:
                record = self._new_record(request, parent, parent_tool_call_id)
                self._records[record.snapshot.agent_id] = record
                self._by_start_key[key] = record.snapshot.agent_id
                record.task = asyncio.create_task(self._run_child(record))
            else:
                record = self._records[existing_id]
                if record.request != request:
                    raise AgentBackendError(
                        "agent_conflict",
                        "The parent tool call already owns a different Agent request.",
                    )

        await record.started.wait()
        return record.snapshot

    async def get(self, agent_id: str, *, requester: RunContext) -> AgentSnapshot:
        """Return an owned Agent's current immutable snapshot."""

        return self._authorized_record(agent_id, requester).snapshot

    async def wait_or_get(
        self,
        agent_id: str,
        *,
        requester: RunContext,
        requester_tool_call_id: str,
    ) -> AgentSnapshot:
        """Register the parent waiter without racing terminal publication."""

        _non_empty_string(requester_tool_call_id, "requester_tool_call_id")
        record = self._authorized_record(agent_id, requester)
        # Completion is retained by the per-Agent event and immutable terminal snapshot,
        # so this implementation needs no waiter-specific state.
        return record.snapshot

    async def cancel(
        self,
        agent_id: str,
        *,
        requester: RunContext,
        requester_tool_call_id: str,
    ) -> AgentSnapshot:
        """Request one cooperative child pause and await its terminal snapshot."""

        _non_empty_string(requester_tool_call_id, "requester_tool_call_id")
        record = self._authorized_record(agent_id, requester)
        if record.snapshot.status in _TERMINAL_STATUSES:
            return record.snapshot

        signal_cancellation = not record.cancellation_sent
        if signal_cancellation:
            record.cancellation_sent = True
            record.snapshot = replace(record.snapshot, cancellation_requested=True)

        await record.started.wait()
        if signal_cancellation:
            invocation = cast(Invocation, record.invocation)
            invocation.pause(
                Suspension(
                    reason="agent_cancelled",
                    source="AgentCancel",
                    wait_id=record.snapshot.agent_id,
                )
            )
        await record.done.wait()
        return record.snapshot

    async def wait_for_terminal(
        self,
        agent_id: str,
        *,
        requester: RunContext,
    ) -> AgentSnapshot:
        """Wait until an owned Agent publishes its terminal snapshot."""

        record = self._authorized_record(agent_id, requester)
        await record.done.wait()
        return record.snapshot

    def _new_record(
        self,
        request: AgentRequest,
        parent: RunContext,
        parent_tool_call_id: str,
    ) -> _AgentRecord:
        agent_id = f"agent:{uuid4()}"
        parent_depth = _agent_depth(parent)
        metadata = dict(parent.metadata)
        metadata.update({"agent_id": agent_id, "agent_depth": parent_depth + 1})
        return _AgentRecord(
            request=request,
            owner_run_id=parent.run_id,
            snapshot=AgentSnapshot(
                agent_id,
                request.description,
                "running",
                request.background,
            ),
            child_context=RunContext(
                run_id=f"{agent_id}:run",
                started_at=time(),
                deadline=parent.deadline,
                parent_run_id=parent.run_id,
                parent_tool_call_id=parent_tool_call_id,
                run_kind="agent",
                metadata=metadata,
            ),
        )

    def _authorized_record(self, agent_id: str, requester: RunContext) -> _AgentRecord:
        agent_id = _non_empty_string(agent_id, "agent_id")
        requester = _context(requester, "requester")
        record = self._records.get(agent_id)
        if record is None or record.owner_run_id != requester.run_id:
            raise AgentBackendError("agent_not_found", "Agent not found.")
        return record

    async def _run_child(self, record: _AgentRecord) -> None:
        try:
            messages = [Message.user(record.request.prompt)]
            if self._system_prompt is not None:
                messages.insert(0, Message.system(self._system_prompt))
            invocation = self._child_runtime.start(messages, context=record.child_context)
            record.invocation = invocation
            record.started.set()
            checkpoint = await invocation.result()
            record.snapshot = _terminal_snapshot(record, checkpoint.snapshot.state)
        except Exception:
            _LOGGER.exception("Child Agent supervision failed")
            record.snapshot = AgentSnapshot(
                record.snapshot.agent_id,
                record.request.description,
                "failed",
                record.request.background,
                error=ErrorInfo("child_host_error", "Child Agent supervision failed."),
            )
        finally:
            record.started.set()
            record.invocation = None
            record.task = None
            record.done.set()


def _terminal_snapshot(record: _AgentRecord, state: object) -> AgentSnapshot:
    current = record.snapshot
    if isinstance(state, Completed):
        result = "\n".join(part.text for part in state.parts if part.text is not None)
        return AgentSnapshot(
            current.agent_id,
            record.request.description,
            "completed",
            record.request.background,
            result=result,
        )
    if isinstance(state, Suspended) and state.suspension.reason == "agent_cancelled":
        return AgentSnapshot(
            current.agent_id,
            record.request.description,
            "cancelled",
            record.request.background,
            cancellation_requested=True,
        )
    if isinstance(state, Failed):
        error = state.error
    elif isinstance(state, Limited):
        error = ErrorInfo("child_limited", f"Child reached limit: {state.reason.value}")
    else:
        error = ErrorInfo(
            "child_suspended",
            "Child requires an unsupported external resume.",
        )
    return AgentSnapshot(
        current.agent_id,
        record.request.description,
        "failed",
        record.request.background,
        error=error,
    )


def _request(value: object) -> AgentRequest:
    if not isinstance(value, AgentRequest):
        raise TypeError("request must be an AgentRequest")
    return value


def _context(value: object, label: str) -> RunContext:
    if not isinstance(value, RunContext):
        raise TypeError(f"{label} must be a RunContext")
    return value


def _non_empty_string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if not value:
        raise ValueError(f"{label} must not be empty")
    return value


def _agent_depth(parent: RunContext) -> int:
    value = parent.metadata.get("agent_depth", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


__all__ = ["AgentBackend", "InMemoryAgentBackend"]
