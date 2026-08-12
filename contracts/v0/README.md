# JHarness v0 Contracts

This directory is the source of truth for provider-neutral behavior and portable JSON
shapes. Provider payloads, Python object layout, and repository storage formats are not
part of this contract.

## Behavior

| Concern | Contract |
| --- | --- |
| Lifecycle, metrics, and checkpoints | [State machine](state-machine.md) |
| Start, continue, resume, controls, and deadlines | [Run control](run-control.md) |
| Atomic persistence and cancellation safety | [Repository](repository.md) |
| Tool binding, approval, batching, and execution | [Tool scheduling](tool-scheduling.md) |
| Complete and streaming model responses | [Model stream](model-stream.md) |
| Trace construction and deterministic verification | [Run trace](run-trace.md) |

## Portable Schemas

| Schema | Owns |
| --- | --- |
| `messages.schema.json` | Content, artifacts, ordered output items, tool calls, outcomes, and messages |
| `model-request.schema.json` | Runtime/provider tools, model options, tool choice, response format, and request |
| `model-response.schema.json` | Non-empty ordered model output and usage |
| `model-error.schema.json` | Provider-neutral model failure |
| `tools.schema.json` | Tool specifications, execution facts, and risk |
| `tool-result.schema.json` | Tool outcomes and waiting suspension |
| `approval.schema.json` | Approval requests and decisions |
| `limits.schema.json` | Run budgets and concurrency |
| `state.schema.json` | Lifecycle, suspension, and metrics |
| `run-context.schema.json` | Run identity, deadline, and host correlation |
| `run-snapshot.schema.json` | Revisioned durable aggregate |
| `checkpoint.schema.json` | Checkpoint, fact, and compact run view |
| `run-request.schema.json` | Start, continue, and resume requests |
| `events.schema.json` | Invocation observation and checkpoint events |
| `run-trace.schema.json` | Trace header and entries |

Schema IDs use `https://jharness.invalid/spec/v0/<file>.schema.json` and resolve
offline within this directory. Only versioned top-level envelopes carry
`schema_version`.

## Portable JSON Boundary

- A portable value may contain at most 128 object or array containers on any path,
  counting the top-level container. Object keys must be strings; cycles, non-JSON
  values, and non-finite floating-point numbers are rejected.
- An `integer` field requires lexical integer form without a fraction or exponent;
  booleans are not integers. Integer fields have no additional IEEE-754 range limit.
- A `number` field accepts integer or floating-point form. Integer form must be within
  `[-9007199254740991, 9007199254740991]` before finite floating-point conversion.
- Unconstrained opaque JSON preserves scalar types and follows the same JSON and depth
  rules. Opaque content either omits `data` or carries a non-empty object.

## Core Invariants

- `Checkpoint` is the complete portable recovery value, and `Invocation.result()`
  returns the last authoritative checkpoint.
- Lifecycle is exactly `Planning`, `ToolsPending`, `Suspended`, `Completed`, `Failed`,
  or `Limited`.
- `Suspended.resume_to` is exactly `Planning` or `ToolsPending`; terminal checkpoints
  cannot continue or resume.
- One durable boundary increments the revision once and records one semantic fact.
- Repository commits atomically check revision, parent, history base, and run-scoped
  checkpoint idempotency.
- Every model request receives the complete current durable history.
- Model modalities describe what content the model accepts or directly emits;
  tool ownership independently determines whether the runtime or provider
  executes a call.
- Assistant messages and complete model responses preserve one non-empty
  ordered output of content, runtime tool calls, and provider tool calls.
- Only runtime tool calls enter `ToolsPending`; provider tool calls are executed
  remotely and remain ordered assistant output.
- Runtime tools have an explicit `structured` JSON-object or `freeform` string
  input kind; specifications, calls, and deltas preserve that kind.
- Active states preserve `provider_turn_pending`; while true, provider
  continuation history remains adjacent and cannot be rewritten or interrupted
  by appended messages. Live inserts are durably applied after the continuation
  clears and before the next model turn.
- Parallel tool execution requires parallel, read-only, and idempotent execution
  facts; durable results remain in model order.
- Only `checkpoint_committed` advances durable trace state. Other events are live
  observation.
- Portable tools and models each expose one invocation operation.
- Decoders reject unknown shapes, invalid discriminators, non-finite numbers, invalid
  integer representations, excessive nesting, and local cross-field invariants. Trace
  transition semantics require [`verify_trace()` after decoding](run-trace.md#verification).
