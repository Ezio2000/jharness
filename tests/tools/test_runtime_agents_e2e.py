from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from jharness.kernel import (
    Checkpoint,
    Completed,
    ContentPart,
    DeltaSink,
    Message,
    Model,
    ModelCapabilities,
    ModelOptions,
    ModelRequest,
    ModelResponse,
    RunContext,
    RunLimits,
    Runtime,
    StructuredToolCall,
    Suspended,
    ToolAccepted,
    ToolChoice,
    ToolSuccess,
    ToolWaiting,
    thaw_json_value,
)
from jharness.kernel.wire import decode_checkpoint, encode_checkpoint
from jharness.toolkit import ToolRegistry
from jharness.tools import ReadTool
from jharness.tools.agent import (
    AgentCancelTool,
    AgentGetTool,
    AgentTool,
    AgentWaitTool,
    InMemoryAgentBackend,
    extract_agent_wait,
    resume_agent,
)

_Mode = Literal["foreground", "background_wait", "cancel"]


@dataclass(frozen=True, slots=True)
class _Harness:
    model: _EndToEndModel
    backend: InMemoryAgentBackend
    parent_runtime: Runtime
    system_prompt: str


class _EndToEndModel(Model):
    def __init__(
        self,
        mode: _Mode,
        *,
        expected_options: ModelOptions,
        expected_tool_choice: ToolChoice,
    ) -> None:
        self.mode = mode
        self.expected_options = expected_options
        self.expected_tool_choice = expected_tool_choice
        self.release_child = asyncio.Event()
        self.child_entered = asyncio.Event()
        self.child_model_cancelled = False
        self.parent_requests: list[ModelRequest] = []
        self.child_requests: list[tuple[ModelRequest, RunContext]] = []
        self.observed_completion: dict[str, object] | None = None

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
        if context.run_kind == "agent":
            return await self._invoke_child(request, context)
        return self._invoke_parent(request)

    async def _invoke_child(
        self,
        request: ModelRequest,
        context: RunContext,
    ) -> ModelResponse:
        self.child_requests.append((request, context))
        assert request.options == self.expected_options
        assert request.tool_choice == self.expected_tool_choice
        assert [spec.name for spec in request.runtime_tools] == ["Read"]
        assert context.parent_run_id is not None
        assert context.parent_tool_call_id is not None

        tool_messages = [message for message in request.messages if message.role == "tool"]
        if not tool_messages:
            self.child_entered.set()
            try:
                await self.release_child.wait()
            except asyncio.CancelledError:
                self.child_model_cancelled = True
                raise
            return ModelResponse(
                (StructuredToolCall("child-read", "Read", {"file_path": "evidence.txt"}),)
            )

        outcome = tool_messages[-1].outcome
        assert isinstance(outcome, ToolSuccess)
        text = outcome.parts[0].text
        assert text is not None and "E2E evidence from disk" in text
        return ModelResponse(
            (
                ContentPart.text_part(
                    "Child completed after inherited Read tool: E2E evidence from disk"
                ),
            ),
            finish_reason="stop",
        )

    def _invoke_parent(self, request: ModelRequest) -> ModelResponse:
        self.parent_requests.append(request)
        completion = _external_completion(request)
        if completion is not None:
            self.observed_completion = completion
            return ModelResponse(
                (
                    ContentPart.text_part(
                        f"Parent observed {completion['status']}: {completion.get('result', '')}"
                    ),
                ),
                finish_reason="stop",
            )

        tool_messages = [message for message in request.messages if message.role == "tool"]
        if not tool_messages:
            return ModelResponse(
                (
                    StructuredToolCall(
                        "delegate-child",
                        "Agent",
                        {
                            "description": "Read delegated evidence",
                            "prompt": "Read evidence.txt and report its evidence.",
                            "background": self.mode != "foreground",
                        },
                    ),
                )
            )

        if self.mode == "foreground":
            outcome = tool_messages[0].outcome
            if not isinstance(outcome, ToolSuccess):
                raise AssertionError("foreground Parent should resume through a completion")
            payload = thaw_json_value(outcome.structured_content)
            assert isinstance(payload, dict) and payload["status"] == "completed"
            return ModelResponse(
                (ContentPart.text_part("Parent observed fast Child completion."),),
                finish_reason="stop",
            )

        agent_id = _accepted_agent_id(tool_messages)
        if self.mode == "background_wait":
            return self._background_wait_step(tool_messages, agent_id)
        if self.mode == "cancel":
            return self._cancel_step(tool_messages, agent_id)
        raise AssertionError("foreground Parent should resume through an external completion")

    def _background_wait_step(
        self,
        tool_messages: list[Message],
        agent_id: str,
    ) -> ModelResponse:
        if len(tool_messages) == 1:
            return ModelResponse(
                (StructuredToolCall("get-child", "AgentGet", {"agent_id": agent_id}),)
            )
        assert len(tool_messages) == 2
        get_outcome = tool_messages[-1].outcome
        assert isinstance(get_outcome, ToolSuccess)
        payload = thaw_json_value(get_outcome.structured_content)
        assert isinstance(payload, dict) and payload["status"] == "running"
        return ModelResponse(
            (StructuredToolCall("wait-child", "AgentWait", {"agent_id": agent_id}),)
        )

    def _cancel_step(
        self,
        tool_messages: list[Message],
        agent_id: str,
    ) -> ModelResponse:
        if len(tool_messages) == 1:
            return ModelResponse(
                (StructuredToolCall("cancel-child", "AgentCancel", {"agent_id": agent_id}),)
            )
        cancel_outcome = tool_messages[-1].outcome
        assert isinstance(cancel_outcome, ToolSuccess)
        payload = thaw_json_value(cancel_outcome.structured_content)
        assert isinstance(payload, dict) and payload["status"] == "cancelled"
        return ModelResponse(
            (ContentPart.text_part("Parent observed cancelled Child."),),
            finish_reason="stop",
        )


def _accepted_agent_id(tool_messages: list[Message]) -> str:
    accepted = tool_messages[0].outcome
    assert isinstance(accepted, ToolAccepted)
    return accepted.correlation_id


def _external_completion(request: ModelRequest) -> dict[str, object] | None:
    messages = [
        message
        for message in request.messages
        if message.role == "external" and message.metadata.get("kind") == "agent_completion"
    ]
    if not messages:
        return None
    assert len(messages) == 1
    text = messages[0].parts[0].text
    assert text is not None
    value = json.loads(text.removeprefix("Agent completion:\n"))
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _harness(tmp_path: Path, mode: _Mode) -> _Harness:
    (tmp_path / "evidence.txt").write_text("E2E evidence from disk\n", encoding="utf-8")
    options = ModelOptions(
        model="inherited-test-model",
        temperature=0,
        seed=7,
        metadata={"profile": "parent-runtime"},
    )
    tool_choice = ToolChoice(allow_parallel_runtime_tool_calls=False)
    limits = RunLimits(
        max_planning_steps=8,
        max_tool_calls=8,
        timeout_seconds=10,
        max_tool_concurrency=1,
        max_tool_batch_size=1,
    )
    system_prompt = "Shared parent policy inherited by the Child."
    model = _EndToEndModel(
        mode,
        expected_options=options,
        expected_tool_choice=tool_choice,
    )
    child_tools = (ReadTool(tmp_path),)
    child_runtime = Runtime(
        model=model,
        tools=ToolRegistry(child_tools),
        limits=limits,
        model_options=options,
        tool_choice=tool_choice,
    )
    backend = InMemoryAgentBackend(child_runtime, system_prompt=system_prompt)
    parent_runtime = Runtime(
        model=model,
        tools=ToolRegistry(
            (
                *child_tools,
                AgentTool(backend),
                AgentGetTool(backend),
                AgentWaitTool(backend),
                AgentCancelTool(backend),
            )
        ),
        limits=limits,
        model_options=options,
        tool_choice=tool_choice,
    )
    return _Harness(model, backend, parent_runtime, system_prompt)


async def _start_parent(harness: _Harness) -> Checkpoint:
    return await harness.parent_runtime.start(
        (
            Message.system(harness.system_prompt),
            Message.user("Delegate the evidence inspection."),
        )
    ).result()


async def _resume_parent(
    harness: _Harness,
    checkpoint: Checkpoint,
    agent_id: str,
) -> Checkpoint:
    snapshot = await harness.backend.wait_for_terminal(
        agent_id,
        requester=checkpoint.snapshot.context,
    )
    return await resume_agent(harness.parent_runtime, checkpoint, snapshot).result()


def test_real_foreground_agent_runs_child_tool_and_resumes_parent(tmp_path: Path) -> None:
    async def scenario() -> None:
        harness = _harness(tmp_path, "foreground")
        parent_paused = await _start_parent(harness)
        assert isinstance(parent_paused.snapshot.state, Suspended)
        wait = extract_agent_wait(parent_paused)
        assert wait.source == "Agent"

        running = await harness.backend.get(
            wait.agent_id,
            requester=parent_paused.snapshot.context,
        )
        assert running.status == "running"
        await asyncio.wait_for(harness.model.child_entered.wait(), timeout=1)
        child_request, child_context = harness.model.child_requests[0]
        assert child_context.parent_run_id == parent_paused.snapshot.context.run_id
        assert child_context.parent_tool_call_id == "delegate-child"
        assert child_context.deadline == parent_paused.snapshot.context.deadline
        assert child_context.metadata["agent_depth"] == 1
        assert child_request.messages[0].parts[0].text == harness.system_prompt

        harness.model.release_child.set()
        terminal = await harness.backend.wait_for_terminal(
            wait.agent_id,
            requester=parent_paused.snapshot.context,
        )
        assert terminal.status == "completed"
        assert terminal.result is not None and "E2E evidence from disk" in terminal.result

        restored_parent = decode_checkpoint(encode_checkpoint(parent_paused))
        completed_parent = await _resume_parent(harness, restored_parent, wait.agent_id)
        assert isinstance(completed_parent.snapshot.state, Completed)
        parent_text = completed_parent.snapshot.state.parts[0].text
        assert parent_text is not None and "E2E evidence from disk" in parent_text
        assert harness.model.observed_completion is not None
        assert harness.model.observed_completion["status"] == "completed"

    asyncio.run(scenario())


def test_fast_foreground_agent_cannot_race_parent_completion_delivery(tmp_path: Path) -> None:
    async def scenario() -> None:
        harness = _harness(tmp_path, "foreground")
        harness.model.release_child.set()
        checkpoint = await _start_parent(harness)

        if isinstance(checkpoint.snapshot.state, Suspended):
            wait = extract_agent_wait(checkpoint)
            checkpoint = await _resume_parent(harness, checkpoint, wait.agent_id)

        assert isinstance(checkpoint.snapshot.state, Completed)
        assert len(harness.model.child_requests) == 2

    asyncio.run(scenario())


def test_real_background_agent_get_wait_and_resume_parent(tmp_path: Path) -> None:
    async def scenario() -> None:
        harness = _harness(tmp_path, "background_wait")
        parent_paused = await _start_parent(harness)
        assert isinstance(parent_paused.snapshot.state, Suspended)
        wait = extract_agent_wait(parent_paused)
        assert wait.source == "AgentWait"

        running = await harness.backend.get(
            wait.agent_id,
            requester=parent_paused.snapshot.context,
        )
        assert running.background is True
        assert running.status == "running"
        outcomes = [
            message.outcome for message in parent_paused.snapshot.history if message.role == "tool"
        ]
        assert isinstance(outcomes[0], ToolAccepted)
        assert isinstance(outcomes[1], ToolSuccess)
        assert isinstance(outcomes[2], ToolWaiting)
        get_payload = thaw_json_value(outcomes[1].structured_content)
        assert isinstance(get_payload, dict) and get_payload["status"] == "running"

        harness.model.release_child.set()
        completed_parent = await _resume_parent(harness, parent_paused, wait.agent_id)
        assert isinstance(completed_parent.snapshot.state, Completed)
        terminal = await harness.backend.get(
            wait.agent_id,
            requester=parent_paused.snapshot.context,
        )
        assert terminal.status == "completed"
        assert harness.model.observed_completion is not None
        assert harness.model.observed_completion["status"] == "completed"

    asyncio.run(scenario())


def test_real_background_agent_cancel_pauses_active_child(tmp_path: Path) -> None:
    async def scenario() -> None:
        harness = _harness(tmp_path, "cancel")
        completed_parent = await _start_parent(harness)
        assert isinstance(completed_parent.snapshot.state, Completed)

        tool_outcomes = [
            message.outcome
            for message in completed_parent.snapshot.history
            if message.role == "tool"
        ]
        accepted = tool_outcomes[0]
        assert isinstance(accepted, ToolAccepted)
        assert isinstance(tool_outcomes[1], ToolSuccess)
        cancel_payload = thaw_json_value(tool_outcomes[1].structured_content)
        assert isinstance(cancel_payload, dict)
        assert cancel_payload["status"] == "cancelled"

        terminal = await harness.backend.get(
            accepted.correlation_id,
            requester=completed_parent.snapshot.context,
        )
        assert terminal.status == "cancelled"
        assert terminal.cancellation_requested is True
        assert harness.model.child_model_cancelled is True

    asyncio.run(scenario())
