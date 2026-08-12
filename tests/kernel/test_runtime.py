from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Mapping

import pytest

from jharness.kernel import (
    ApprovalDecision,
    ApprovalDeny,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalSuspend,
    Checkpoint,
    CommitError,
    Completed,
    ContentPart,
    DeltaSink,
    DurableCommit,
    Event,
    EventKind,
    Failed,
    HistoryRewrite,
    Invocation,
    Message,
    Model,
    ModelCapabilities,
    ModelContentDelta,
    ModelOptions,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    PendingToolCalls,
    Planning,
    ProviderToolCall,
    ProviderToolId,
    ProviderToolSpec,
    ProviderToolStatus,
    RequestError,
    RunContext,
    RunHistory,
    RunLimits,
    RunRepository,
    RunSnapshot,
    Runtime,
    RuntimeToolCall,
    SettledResult,
    StructuredToolCall,
    StructuredToolSpec,
    Suspended,
    Suspension,
    SuspensionSelector,
    ToolBinding,
    ToolCatalog,
    ToolCatalogProvider,
    ToolChoice,
    ToolContext,
    ToolExecution,
    ToolFailure,
    ToolResult,
    ToolSuccess,
    ToolWaiting,
    WaitingResult,
)


class ScriptModel(Model):
    def __init__(
        self,
        responses: list[ModelResponse],
        *,
        streaming: bool = False,
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        self.responses = deque(responses)
        self.requests: list[ModelRequest] = []
        self._capabilities = (
            ModelCapabilities(streaming=streaming) if capabilities is None else capabilities
        )

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
        del context
        self.requests.append(request)
        if stream and emit_delta is not None:
            await emit_delta(ModelContentDelta(0, "live"))
        return self.responses.popleft()


class BlockingModel(Model):
    def __init__(self, final: ModelResponse) -> None:
        self.calls = 0
        self.started = asyncio.Event()
        self.final = final

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
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            await asyncio.Event().wait()
        return self.final


ToolEffect = Callable[[ToolContext], Awaitable[ToolResult]]


class Binding:
    __slots__ = ("_call", "_effect", "_spec")

    def __init__(
        self, call: StructuredToolCall, spec: StructuredToolSpec, effect: ToolEffect
    ) -> None:
        self._call = call
        self._spec = spec
        self._effect = effect

    @property
    def call(self) -> StructuredToolCall:
        return self._call

    @property
    def spec(self) -> StructuredToolSpec:
        return self._spec

    async def invoke(self, context: ToolContext) -> ToolResult:
        return await self._effect(context)


class Catalog(ToolCatalog):
    def __init__(self, effects: Mapping[str, tuple[StructuredToolSpec, ToolEffect]]) -> None:
        self.effects = dict(effects)

    def specs(self) -> tuple[StructuredToolSpec, ...]:
        return tuple(spec for spec, _ in self.effects.values())

    def spec(self, name: str) -> StructuredToolSpec | None:
        item = self.effects.get(name)
        return None if item is None else item[0]

    def bind(self, call: RuntimeToolCall) -> ToolBinding:
        if not isinstance(call, StructuredToolCall):
            raise TypeError("test catalog accepts structured calls only")
        spec, effect = self.effects[call.name]
        return Binding(call, spec, effect)


class CatalogProvider(ToolCatalogProvider):
    def __init__(self, catalog: ToolCatalog) -> None:
        self.catalog = catalog

    async def open_catalog(self) -> ToolCatalog:
        return self.catalog


def final(text: str = "done", *, usage: ModelUsage | None = None) -> ModelResponse:
    return ModelResponse((ContentPart.text_part(text),), finish_reason="stop", usage=usage)


async def success(context: ToolContext) -> ToolResult:
    del context
    return SettledResult(ToolSuccess((ContentPart.text_part("tool-ok"),), {"ok": True}))


def tool_provider(
    effect: ToolEffect = success,
    *,
    execution: ToolExecution | None = None,
    name: str = "lookup",
) -> CatalogProvider:
    spec = StructuredToolSpec(
        name,
        "Lookup",
        {"type": "object"},
        execution=ToolExecution() if execution is None else execution,
    )
    return CatalogProvider(Catalog({name: (spec, effect)}))


async def collect(invocation: Invocation) -> tuple[Checkpoint, list[Event]]:
    events = invocation.events()
    result_task = asyncio.create_task(invocation.result())
    observed = [event async for event in events]
    return await result_task, observed


def test_start_rejects_invalid_planning_history_before_creating_invocation() -> None:
    call = StructuredToolCall("call-1", "lookup")
    model = ScriptModel([final()])

    with pytest.raises(ValueError, match="unresolved"):
        Runtime(model=model).start((Message.user("go"), Message.assistant((call,))))

    assert model.requests == []


async def test_final_run_result_and_events_share_one_execution() -> None:
    model = ScriptModel([final()], streaming=True)
    invocation = Runtime(model=model).start((Message.user("hello"),), stream=True)
    checkpoint, events = await collect(invocation)
    assert checkpoint.snapshot.status == "completed"
    assert checkpoint.snapshot.revision == 1
    assert [event.kind for event in events] == [
        EventKind.INVOCATION_STARTED,
        EventKind.CHECKPOINT_COMMITTED,
        EventKind.MODEL_STARTED,
        EventKind.MODEL_DELTA,
        EventKind.MODEL_FINISHED,
        EventKind.CHECKPOINT_COMMITTED,
        EventKind.INVOCATION_STOPPED,
    ]
    assert await invocation.result() is checkpoint
    assert len(model.requests) == 1


async def test_model_request_receives_complete_history() -> None:
    messages = tuple(Message.user(str(index)) for index in range(256))
    call = StructuredToolCall("call-complete-history", "lookup")
    model = ScriptModel([ModelResponse((call,)), final()])

    checkpoint = await Runtime(model=model, tools=tool_provider()).start(messages).result()

    assert model.requests[0].messages == messages
    assert model.requests[1].messages == tuple(checkpoint.snapshot.history)[:-1]


async def test_runtime_rejects_unsupported_exact_tool_choice_before_invocation() -> None:
    model = ScriptModel(
        [final()],
        capabilities=ModelCapabilities(tool_choice_types=frozenset({"auto"})),
    )

    checkpoint = (
        await Runtime(model=model, tool_choice=ToolChoice("none"))
        .start((Message.user("hello"),))
        .result()
    )

    assert isinstance(checkpoint.snapshot.state, Failed)
    assert checkpoint.snapshot.state.error.message == "model does not support tool_choice='none'"
    assert model.requests == []


async def test_runtime_rejects_unsupported_parallel_control_before_invocation() -> None:
    model = ScriptModel(
        [final()],
        capabilities=ModelCapabilities(parallel_runtime_tool_call_control=False),
    )

    checkpoint = (
        await Runtime(
            model=model,
            tools=tool_provider(),
            tool_choice=ToolChoice(allow_parallel_runtime_tool_calls=False),
        )
        .start((Message.user("hello"),))
        .result()
    )

    assert isinstance(checkpoint.snapshot.state, Failed)
    assert (
        checkpoint.snapshot.state.error.message
        == "model cannot disable parallel runtime tool calls"
    )
    assert model.requests == []


async def test_runtime_accepts_serial_request_when_model_cannot_call_in_parallel() -> None:
    model = ScriptModel(
        [final()],
        capabilities=ModelCapabilities(
            parallel_runtime_tool_calls=False,
            parallel_runtime_tool_call_control=False,
        ),
    )

    checkpoint = (
        await Runtime(
            model=model,
            tools=tool_provider(),
            tool_choice=ToolChoice(allow_parallel_runtime_tool_calls=False),
        )
        .start((Message.user("hello"),))
        .result()
    )

    assert checkpoint.snapshot.status == "completed"
    assert len(model.requests) == 1


async def test_provider_tool_selection_ignores_runtime_parallel_control() -> None:
    provider_id = ProviderToolId("deepseek.responses", "web_search")
    provider_call = ProviderToolCall(
        "search-1",
        provider_id,
        ProviderToolStatus.COMPLETED,
        {"query": "JHarness"},
    )
    model = ScriptModel(
        [ModelResponse((provider_call,))],
        capabilities=ModelCapabilities(
            tool_choice_types=frozenset({"auto", "none", "required", "runtime", "provider"}),
            parallel_runtime_tool_calls=True,
            parallel_runtime_tool_call_control=False,
            provider_tools=frozenset({provider_id}),
        ),
    )

    checkpoint = (
        await Runtime(
            model=model,
            tools=tool_provider(),
            provider_tools=(ProviderToolSpec(provider_id),),
            tool_choice=ToolChoice(
                "provider",
                provider_tool=provider_id,
                allow_parallel_runtime_tool_calls=False,
            ),
        )
        .start((Message.user("search"),))
        .result()
    )

    assert checkpoint.snapshot.status == "completed"
    assert len(model.requests) == 1


async def test_provider_only_request_ignores_runtime_parallel_control() -> None:
    provider_id = ProviderToolId("deepseek.responses", "web_search")
    model = ScriptModel(
        [final()],
        capabilities=ModelCapabilities(
            parallel_runtime_tool_calls=True,
            parallel_runtime_tool_call_control=False,
            provider_tools=frozenset({provider_id}),
        ),
    )

    checkpoint = (
        await Runtime(
            model=model,
            provider_tools=(ProviderToolSpec(provider_id),),
            tool_choice=ToolChoice(allow_parallel_runtime_tool_calls=False),
        )
        .start((Message.user("search"),))
        .result()
    )

    assert checkpoint.snapshot.status == "completed"
    assert len(model.requests) == 1


async def test_runtime_rejects_unsupported_seed_before_invocation() -> None:
    model = ScriptModel(
        [final()],
        capabilities=ModelCapabilities(seed=False),
    )

    checkpoint = (
        await Runtime(
            model=model,
            model_options=ModelOptions(seed=7),
        )
        .start((Message.user("hello"),))
        .result()
    )

    assert isinstance(checkpoint.snapshot.state, Failed)
    assert checkpoint.snapshot.state.error.message == "model does not support seed"
    assert model.requests == []


async def test_provider_tool_call_is_observed_but_never_scheduled_by_runtime() -> None:
    provider_id = ProviderToolId("deepseek.responses", "web_search")
    provider_call = ProviderToolCall(
        "search-1",
        provider_id,
        ProviderToolStatus.COMPLETED,
        {"query": "JHarness"},
        metadata={"action": "search"},
    )
    model = ScriptModel(
        [ModelResponse((provider_call, ContentPart.text_part("answer")))],
        capabilities=ModelCapabilities(provider_tools=frozenset({provider_id})),
    )
    checkpoint, events = await collect(
        Runtime(
            model=model,
            provider_tools=(ProviderToolSpec(provider_id),),
        ).start((Message.user("search"),))
    )

    assert checkpoint.snapshot.status == "completed"
    assert checkpoint.snapshot.metrics.tool_calls == 0
    assistant = checkpoint.snapshot.history[-1]
    assert assistant.provider_tool_calls() == (provider_call,)
    finished = next(event for event in events if event.kind is EventKind.MODEL_FINISHED)
    assert finished.data["runtime_tool_call_count"] == 0
    assert finished.data["provider_tool_call_count"] == 1


async def test_provider_only_output_completes_with_empty_visible_projection() -> None:
    provider_id = ProviderToolId("deepseek.responses", "web_search")
    calls = tuple(
        ProviderToolCall(
            f"search-{index}",
            provider_id,
            ProviderToolStatus.COMPLETED,
            {"query": f"query-{index}"},
        )
        for index in range(3)
    )
    model = ScriptModel(
        [ModelResponse(calls)],
        capabilities=ModelCapabilities(
            parallel_runtime_tool_calls=False,
            provider_tools=frozenset({provider_id}),
        ),
    )
    checkpoint, events = await collect(
        Runtime(
            model=model,
            provider_tools=(ProviderToolSpec(provider_id),),
            tool_choice=ToolChoice(allow_parallel_runtime_tool_calls=False),
        ).start((Message.user("search only"),))
    )

    assert checkpoint.snapshot.state == Completed(())
    assert checkpoint.snapshot.history[-1].output == calls
    assert checkpoint.snapshot.metrics.tool_calls == 0
    assert not any(
        event.kind
        in {
            EventKind.TOOL_BATCH_SELECTED,
            EventKind.TOOL_STARTED,
            EventKind.TOOL_FINISHED,
        }
        for event in events
    )


async def test_provider_output_does_not_expand_direct_output_modalities() -> None:
    provider_id = ProviderToolId("openai.responses", "image_generation")
    image = ContentPart(
        "image",
        uri="https://example.invalid/generated.png",
        media_type="image/png",
    )
    provider_call = ProviderToolCall(
        "image-1",
        provider_id,
        ProviderToolStatus.COMPLETED,
        output=(image,),
    )
    capabilities = ModelCapabilities(
        output_modalities=frozenset({"text"}),
        provider_tools=frozenset({provider_id}),
    )
    provider_result = (
        await Runtime(
            model=ScriptModel([ModelResponse((provider_call,))], capabilities=capabilities),
            provider_tools=(ProviderToolSpec(provider_id),),
        )
        .start((Message.user("draw"),))
        .result()
    )
    direct_result = (
        await Runtime(
            model=ScriptModel([ModelResponse((image,))], capabilities=capabilities),
        )
        .start((Message.user("draw"),))
        .result()
    )

    assert provider_result.snapshot.state == Completed((image,))
    assert isinstance(direct_result.snapshot.state, Failed)
    assert direct_result.snapshot.state.error.code == "model_protocol_error"


async def test_mixed_provider_and_runtime_calls_schedule_only_runtime_call() -> None:
    provider_id = ProviderToolId("deepseek.responses", "web_search")
    provider_call = ProviderToolCall(
        "search-1",
        provider_id,
        ProviderToolStatus.COMPLETED,
    )
    runtime_call = StructuredToolCall("runtime-1", "lookup", {"query": "JHarness"})
    model = ScriptModel(
        [ModelResponse((provider_call, runtime_call)), final()],
        capabilities=ModelCapabilities(provider_tools=frozenset({provider_id})),
    )
    checkpoint = (
        await Runtime(
            model=model,
            tools=tool_provider(),
            provider_tools=(ProviderToolSpec(provider_id),),
        )
        .start((Message.user("search then lookup"),))
        .result()
    )

    assert checkpoint.snapshot.status == "completed"
    assert checkpoint.snapshot.metrics.tool_calls == 1
    assert checkpoint.snapshot.history[1].output == (provider_call, runtime_call)
    assert [
        message.tool_call_id for message in checkpoint.snapshot.history if message.role == "tool"
    ] == [runtime_call.id]


async def test_invalid_dynamic_model_request_becomes_failed_checkpoint() -> None:
    model = ScriptModel([final()])
    checkpoint, events = await collect(
        Runtime(
            model=model,
            tool_choice=ToolChoice("runtime", name="missing"),
        ).start((Message.user("go"),))
    )

    assert isinstance(checkpoint.snapshot.state, Failed)
    assert checkpoint.snapshot.state.error.code == "model_protocol_error"
    assert model.requests == []
    assert EventKind.MODEL_STARTED not in [event.kind for event in events]


def test_runtime_rejects_invalid_static_provider_configuration() -> None:
    provider_id = ProviderToolId("deepseek.responses", "web_search")
    spec = ProviderToolSpec(provider_id)
    model = ScriptModel([final()])

    with pytest.raises(ValueError, match="provider tools must be unique"):
        Runtime(model=model, provider_tools=(spec, spec))
    with pytest.raises(ValueError, match="unavailable provider tool"):
        Runtime(
            model=model,
            tool_choice=ToolChoice("provider", provider_tool=provider_id),
        )


async def test_pending_provider_turn_continues_with_adjacent_history() -> None:
    provider_id = ProviderToolId("deepseek.responses", "web_search")
    call = ProviderToolCall("search-running", provider_id, ProviderToolStatus.IN_PROGRESS)
    model = ScriptModel(
        [
            ModelResponse(
                (call, ContentPart.text_part("working")),
                provider_turn_pending=True,
            ),
            final("done"),
        ],
        capabilities=ModelCapabilities(provider_tools=frozenset({provider_id})),
    )
    checkpoint = (
        await Runtime(
            model=model,
            provider_tools=(ProviderToolSpec(provider_id),),
        )
        .start((Message.user("search"),))
        .result()
    )

    assert isinstance(checkpoint.snapshot.state, Completed)
    assert len(model.requests) == 2
    assert model.requests[1].messages[-1].provider_tool_calls() == (call,)
    assert checkpoint.snapshot.metrics.planning_steps == 2


async def test_pending_provider_turn_defers_insert_until_continuation_clears() -> None:
    provider_id = ProviderToolId("fixture.provider", "search")
    provider_call = ProviderToolCall(
        "search-running",
        provider_id,
        ProviderToolStatus.IN_PROGRESS,
    )

    class DeferredInsertModel(Model):
        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []
            self.continuation_started = asyncio.Event()
            self.release_continuation = asyncio.Event()

        @property
        def capabilities(self) -> ModelCapabilities:
            return ModelCapabilities(provider_tools=frozenset({provider_id}))

        async def invoke(
            self,
            request: ModelRequest,
            context: RunContext,
            *,
            stream: bool,
            emit_delta: DeltaSink | None,
        ) -> ModelResponse:
            del context, stream, emit_delta
            self.requests.append(request)
            if len(self.requests) == 1:
                return ModelResponse((provider_call,), provider_turn_pending=True)
            if len(self.requests) == 2:
                self.continuation_started.set()
                await self.release_continuation.wait()
                return final("provider continuation complete")
            return final("after deferred insert")

    model = DeferredInsertModel()
    invocation = Runtime(
        model=model,
        provider_tools=(ProviderToolSpec(provider_id),),
    ).start((Message.user("search"),))
    collect_task = asyncio.create_task(collect(invocation))
    await model.continuation_started.wait()
    invocation.insert(Message.external("live context"))
    await asyncio.sleep(0)
    model.release_continuation.set()
    checkpoint, events = await collect_task

    assert isinstance(checkpoint.snapshot.state, Completed)
    assert checkpoint.snapshot.metrics.planning_steps == 3
    assert [message.role for message in checkpoint.snapshot.history] == [
        "user",
        "assistant",
        "assistant",
        "external",
        "assistant",
    ]
    assert [[message.role for message in request.messages] for request in model.requests] == [
        ["user"],
        ["user", "assistant"],
        ["user", "assistant", "assistant", "external"],
    ]
    model_turns = [
        event.data["fact"]["data"]
        for event in events
        if event.kind is EventKind.CHECKPOINT_COMMITTED
        and event.data["fact"]["kind"] == "model_turn"
    ]
    assert [(fact["result"], fact["provider_turn_pending"]) for fact in model_turns] == [
        ("planning", True),
        ("planning", False),
        ("completed", False),
    ]


async def test_result_only_rejects_late_event_subscription() -> None:
    invocation = Runtime(model=ScriptModel([final()])).start((Message.user("hello"),))
    assert object.__getattribute__(invocation, "_queue") is None
    first = await invocation.result()
    assert object.__getattribute__(invocation, "_queue") is None
    assert await invocation.result() is first
    with pytest.raises(RuntimeError, match="result-only"):
        invocation.events()


async def test_event_subscription_allocates_the_observation_queue_lazily() -> None:
    invocation = Runtime(model=ScriptModel([final()])).start((Message.user("hello"),))
    assert object.__getattribute__(invocation, "_queue") is None
    events = invocation.events()
    assert isinstance(object.__getattribute__(invocation, "_queue"), asyncio.Queue)
    with pytest.raises(RuntimeError, match="consumed only once"):
        invocation.events()
    assert [event async for event in events][-1].kind is EventKind.INVOCATION_STOPPED
    assert object.__getattribute__(invocation, "_queue") is None


async def test_finished_invocation_discards_all_control_operations() -> None:
    invocation = Runtime(model=ScriptModel([final()])).start((Message.user("hello"),))
    checkpoint = await invocation.result()
    assert object.__getattribute__(invocation, "_execute") is None

    invocation.pause(Suspension("late", "host"))
    invocation.insert(Message.external("late"))
    invocation.cancel_tool("late-call")

    control = object.__getattribute__(invocation, "_control")
    assert object.__getattribute__(control, "_active") is None
    assert not object.__getattribute__(control, "_pending")
    assert await invocation.result() is checkpoint


async def test_tool_result_is_committed_in_model_order() -> None:
    call = StructuredToolCall("call-1", "lookup", {"q": "x"})
    model = ScriptModel([ModelResponse((call,)), final()])
    checkpoint, events = await collect(
        Runtime(model=model, tools=tool_provider()).start((Message.user("go"),))
    )
    assert checkpoint.snapshot.revision == 3
    assert [message.role for message in checkpoint.snapshot.history] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    outcome = checkpoint.snapshot.history[2].outcome
    assert isinstance(outcome, ToolSuccess)
    assert [event.kind for event in events].count(EventKind.CHECKPOINT_COMMITTED) == 4
    selected = [event for event in events if event.kind is EventKind.TOOL_BATCH_SELECTED]
    assert len(selected) == 1
    assert selected[0].data["call_ids"] == (call.id,)
    assert selected[0].data["remaining_count"] == 0


async def test_serial_tool_batches_do_not_materialize_the_remaining_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = tuple(StructuredToolCall(f"call-{index}", "lookup") for index in range(40))

    def fail_prefix(_: PendingToolCalls, _count: int) -> tuple[StructuredToolCall, ...]:
        raise AssertionError("tool selection materialized the pending suffix")

    monkeypatch.setattr(PendingToolCalls, "prefix", fail_prefix)
    checkpoint = (
        await Runtime(
            model=ScriptModel([ModelResponse(calls), final()]),
            tools=tool_provider(),
            limits=RunLimits(max_tool_calls=len(calls), max_tool_batch_size=len(calls)),
        )
        .start((Message.user("go"),))
        .result()
    )

    assert checkpoint.snapshot.metrics.tool_calls == len(calls)
    tail = checkpoint.snapshot.history.iter_tail(len(calls) + 1)
    assert [message.tool_call_id for message in tail if message.role == "tool"] == [
        call.id for call in calls
    ]


async def test_waiting_result_suspends_and_exact_resume_completes() -> None:
    suspension = Suspension("external_work", "tool", wait_id="wait-1")

    async def waiting(context: ToolContext) -> ToolResult:
        del context
        return WaitingResult(ToolWaiting((ContentPart.text_part("waiting"),)), suspension)

    call = StructuredToolCall("call-1", "lookup")
    model = ScriptModel([ModelResponse((call,)), final("resumed")])
    runtime = Runtime(model=model, tools=tool_provider(waiting))
    paused = await runtime.start((Message.user("go"),)).result()
    assert isinstance(paused.snapshot.state, Suspended)
    assert paused.snapshot.revision == 2
    resumed = await runtime.resume(paused, selector=SuspensionSelector(wait_id="wait-1")).result()
    assert resumed.snapshot.status == "completed"
    assert resumed.snapshot.revision == 4
    with pytest.raises(RequestError, match="does not match") as mismatch:
        runtime.resume(paused, selector=SuspensionSelector(wait_id="other"))
    assert mismatch.value.code == "suspension_mismatch"


async def test_pause_and_insert_interrupt_model_without_partial_commit() -> None:
    pause_model = BlockingModel(final())
    pause_invocation = Runtime(model=pause_model).start((Message.user("go"),))
    pause_events = pause_invocation.events()
    pause_task = asyncio.create_task(pause_invocation.result())
    async for event in pause_events:
        if event.kind is EventKind.MODEL_STARTED:
            pause_invocation.pause(Suspension("user", "host"))
    paused = await pause_task
    assert paused.snapshot.status == "suspended"
    assert paused.snapshot.revision == 1

    insert_model = BlockingModel(final())
    insert_invocation = Runtime(model=insert_model).start((Message.user("go"),))
    insert_events = insert_invocation.events()
    insert_task = asyncio.create_task(insert_invocation.result())
    inserted_once = False
    async for event in insert_events:
        if event.kind is EventKind.MODEL_STARTED and not inserted_once:
            inserted_once = True
            insert_invocation.insert(Message.external("new information"))
    inserted = await insert_task
    assert inserted.snapshot.status == "completed"
    assert [message.role for message in inserted.snapshot.history] == [
        "user",
        "external",
        "assistant",
    ]


async def test_total_token_limit_commits_complete_model_observation_then_limits() -> None:
    model = ScriptModel([final(usage=ModelUsage(total_tokens=6))])
    checkpoint = (
        await Runtime(model=model, limits=RunLimits(max_total_tokens=5))
        .start((Message.user("go"),))
        .result()
    )
    assert checkpoint.snapshot.status == "limited"
    assert checkpoint.snapshot.metrics.planning_steps == 1
    assert checkpoint.snapshot.metrics.usage.total_tokens == 6


class RejectSecondCommit(RunRepository):
    def __init__(self) -> None:
        self.commits: list[DurableCommit] = []

    async def commit(self, commit: DurableCommit) -> None:
        if self.commits:
            raise RuntimeError("storage unavailable")
        self.commits.append(commit)


class FalseyRepository(RunRepository):
    def __init__(self) -> None:
        self.commits: list[DurableCommit] = []

    def __bool__(self) -> bool:
        return False

    async def commit(self, commit: DurableCommit) -> None:
        self.commits.append(commit)


class BlockingStartCommit(RunRepository):
    def __init__(self) -> None:
        self.attempts = 0
        self.cancelled = False

    async def commit(self, commit: DurableCommit) -> None:
        del commit
        self.attempts += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class CancellationIgnoringModel(Model):
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.finished = asyncio.Event()
        self.cancellations = 0

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
        try:
            while not self.release.is_set():
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    self.cancellations += 1
        finally:
            self.finished.set()
        return final()


async def test_start_commit_is_bounded_by_the_invocation_work_deadline() -> None:
    repository = BlockingStartCommit()
    model = ScriptModel([final()])
    invocation = Runtime(
        model=model,
        limits=RunLimits(timeout_seconds=0.02),
        repository=repository,
        repository_timeout=1.0,
    ).start((Message.user("go"),))

    with pytest.raises(
        CommitError,
        match="start checkpoint commit exceeded work deadline",
    ) as caught:
        await invocation.result()

    assert caught.value.last_checkpoint is None
    assert repository.attempts == 1
    assert repository.cancelled
    assert model.requests == []


async def test_runtime_preserves_an_explicit_falsey_repository() -> None:
    repository = FalseyRepository()

    checkpoint = (
        await Runtime(model=ScriptModel([final()]), repository=repository)
        .start((Message.user("go"),))
        .result()
    )

    assert checkpoint.snapshot.status == "completed"
    assert [commit.checkpoint.snapshot.revision for commit in repository.commits] == [0, 1]


async def test_noncompliant_port_is_reported_after_bounded_cleanup() -> None:
    model = CancellationIgnoringModel()
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    reports: list[Mapping[str, object]] = []
    loop.set_exception_handler(lambda _loop, context: reports.append(context))
    try:
        checkpoint = (
            await Runtime(
                model=model,
                limits=RunLimits(timeout_seconds=0.01),
            )
            .start((Message.user("go"),))
            .result()
        )
        assert checkpoint.snapshot.status == "limited"
        assert model.cancellations >= 1
        assert any("abandoned a port task" in str(item.get("message")) for item in reports)
    finally:
        model.release.set()
        await asyncio.wait_for(model.finished.wait(), timeout=1)
        loop.set_exception_handler(previous_handler)


async def test_repository_failure_preserves_last_checkpoint_and_stops_observation() -> None:
    repository = RejectSecondCommit()
    invocation = Runtime(model=ScriptModel([final()]), repository=repository).start(
        (Message.user("go"),)
    )
    events = invocation.events()
    with pytest.raises(CommitError, match="storage unavailable") as caught:
        async for _ in events:
            pass
    assert caught.value.last_checkpoint is repository.commits[0].checkpoint


class DenyPolicy(ApprovalPolicy):
    async def decide(self, requests: tuple[ApprovalRequest, ...]) -> tuple[ApprovalDecision, ...]:
        return tuple(ApprovalDeny(request.call.id, "denied") for request in requests)


class SuspendPolicy(ApprovalPolicy):
    async def decide(self, requests: tuple[ApprovalRequest, ...]) -> tuple[ApprovalDecision, ...]:
        return tuple(
            ApprovalSuspend(request.call.id, Suspension("approval", "policy"))
            for request in requests
        )


async def test_approval_deny_is_model_visible_and_suspend_invokes_nothing() -> None:
    invoked = 0

    async def effect(context: ToolContext) -> ToolResult:
        nonlocal invoked
        del context
        invoked += 1
        return SettledResult(ToolSuccess((ContentPart.text_part("unexpected"),)))

    call = StructuredToolCall("call-1", "lookup")
    denied_model = ScriptModel([ModelResponse((call,)), final()])
    denied = (
        await Runtime(
            model=denied_model,
            tools=tool_provider(effect),
            approval_policy=DenyPolicy(),
        )
        .start((Message.user("go"),))
        .result()
    )
    assert isinstance(denied.snapshot.history[2].outcome, ToolFailure)
    assert invoked == 0

    suspended_model = ScriptModel([ModelResponse((call,))])
    suspended = (
        await Runtime(
            model=suspended_model,
            tools=tool_provider(effect),
            approval_policy=SuspendPolicy(),
        )
        .start((Message.user("go"),))
        .result()
    )
    assert isinstance(suspended.snapshot.state, Suspended)
    assert invoked == 0


class OneRewrite:
    def __init__(self) -> None:
        self.used = False

    async def reduce(self, snapshot: RunSnapshot) -> HistoryRewrite | None:
        del snapshot
        if self.used:
            return None
        self.used = True
        return HistoryRewrite(RunHistory((Message.user("summary"),)), "compact")


async def test_history_rewrite_is_a_separate_checkpoint() -> None:
    checkpoint, events = await collect(
        Runtime(model=ScriptModel([final()]), history_reducer=OneRewrite()).start(
            (Message.user("old"), Message.user("history"))
        )
    )
    facts = [
        event.data["fact"]["kind"]
        for event in events
        if event.kind is EventKind.CHECKPOINT_COMMITTED
    ]
    assert facts == ["started", "history_rewrite", "model_turn"]
    assert checkpoint.snapshot.history[0].parts[0].text == "summary"


async def test_consumer_close_returns_last_committed_checkpoint() -> None:
    invocation = Runtime(model=BlockingModel(final())).start((Message.user("go"),))
    events = invocation.events()
    assert (await anext(events)).kind is EventKind.INVOCATION_STARTED
    assert (await anext(events)).kind is EventKind.CHECKPOINT_COMMITTED
    assert (await anext(events)).kind is EventKind.MODEL_STARTED
    await events.aclose()
    checkpoint = await invocation.result()
    assert checkpoint.snapshot.state == Planning()
