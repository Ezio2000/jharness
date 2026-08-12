# Kernel v0 State Machine

## Lifecycle

Portable lifecycle is one flat discriminated union:

```text
Planning(provider_turn_pending: bool)
ToolsPending(non-empty ordered runtime tool calls, provider_turn_pending: bool)
Suspended(resume_to: Planning | ToolsPending, suspension)
Completed(final content)
Failed(error)
Limited(limit reason)
```

Wire kinds are `planning`, `tools_pending`, `suspended`, `completed`, `failed`,
and `limited`. There is no wrapper status, continuation, or outcome object, and
no independent mutable status field.

## Legal Transitions

```text
Planning -> ToolsPending
Planning -> Suspended(resume_to=Planning)
Planning -> Completed | Failed | Limited

ToolsPending -> ToolsPending
ToolsPending -> Planning
ToolsPending -> Suspended(resume_to=ToolsPending | Planning)
ToolsPending -> Failed | Limited

Suspended(resume_to=S) -> S
```

`Completed`, `Failed`, and `Limited` have no outgoing transitions.

## Checkpoints

Every durable change creates exactly one `Checkpoint(id, snapshot, fact)` and
increments snapshot revision exactly once. Fact kinds are `started`, `resumed`,
`model_turn`, `tool_batch`, `conversation_insert`, `history_rewrite`, and
`control`.

A model turn is durable only after `Model.invoke` returns a complete response.
The complete non-empty ordered output is persisted as one assistant message.
Provider-executed tool calls stay in that output; only runtime tool calls become
pending work. A serial runtime tool call is a tool batch of one. A parallel
batch is durable only after every selected runtime call has a normalized
result. A checkpoint stores tool messages in model call order.

`provider_turn_pending=true` is a provider-neutral durable continuation fact.
It preserves unfinished provider-owned work without creating host tool work.
While set, history reduction, conversation insertion, and resume-appended
messages are forbidden so the next model request sees the provider turn and its
continuation adjacently. Runtime calls may still enter `ToolsPending`; the flag
survives every selected runtime batch and returns to `Planning(true)` when the
last runtime call settles. A later model response clears the flag by returning
`provider_turn_pending=false`.

## Metrics

- `planning_steps` increases by one for each committed complete model response.
- `tool_calls` increases by the number of committed runtime tool messages.
- usage accumulates only reported fields from committed model responses.
- counters never decrease.

Interrupted model calls and uncommitted parallel results do not advance
metrics.

## Completion

A complete model response always commits its ordered assistant output, usage,
and one planning-step increment together. Its model-turn fact has a `result`
that determines the after state: `planning` records either a pending provider
turn or a cleared continuation with deferred conversation inserts,
`completed` uses its visible-part count, `tools_pending` uses its ordered
runtime call ids, and `limited` records
`max_total_tokens`. Visible parts consist of direct content plus output from any
terminal provider-tool status (`completed`, `incomplete`, or `failed`) in output
order. Provider tool calls never create
`ToolsPending`. No separate terminal checkpoint is required for the same
response.

## Suspension

`Suspended.resume_to` stores the exact active state. A waiting tool result is
committed before the suspended checkpoint becomes visible. Its model-visible
outcome is written once to `ToolMessage`; its host-only suspension is written
only to `Suspended`. If calls remain, `resume_to` is `ToolsPending`; otherwise
it is `Planning`.

Approval suspension occurs before any selected call is invoked or committed and
therefore preserves the complete selected prefix in `resume_to=ToolsPending`.

## Failure and Limits

Tool validation, denial, and implementation failures become model-visible tool
outcomes. Model, protocol, and infrastructure failures move to `Failed` when a
terminal checkpoint is possible.

Repository failure is outside the state-machine transition. The attempted
checkpoint is not authoritative and the last successfully committed checkpoint
remains current.
