# Architecture

JHarness keeps durable execution in a dependency-free kernel and injects every
deployment choice through a narrow public protocol.

## Packages

| Package | Owns |
| --- | --- |
| `jharness.kernel` | Runtime, immutable state, model/tool ports, limits, checkpoints, events, diagnostics, repository protocol, and v0 wire codecs |
| `jharness.toolkit` | Tool registration, function adaptation, JSON Schema validation, retry, and circuit breaking |
| `jharness.models` | Provider clients and profiles plus retry/fallback model composition |
| `jharness.repository` | Memory, SQLite, MySQL, and Redis implementations |
| `jharness.tools` | Filesystem, shell, interaction, and child-agent tools |

Among JHarness packages, each integration package depends only on the exact matching
`jharness-kernel`; the integration packages do not depend on one another. Each
distribution owns one `jharness.<component>` namespace portion, and no distribution
owns `jharness/__init__.py`.

## Execution

`Runtime` is immutable configuration. Each `start`, `continue_from`, or `resume` call
creates one single-use `Invocation`, which runs the same engine and returns its last
committed `Checkpoint`.

The lifecycle has six states:

- `Planning` invokes the configured model once.
- `ToolsPending` executes a non-empty remaining suffix of model tool calls.
- `Suspended` preserves the exact active state for a later `resume`.
- `Completed`, `Failed`, and `Limited` are terminal.

Model output either completes the run or schedules tools. Tool results are committed
in model order even when execution is concurrent. Parallel batches require calls
declared parallel, read-only, and idempotent. Limits cap planning steps, tool-call
count, batch size, concurrency, progress buffering, and optional elapsed time. The
token limit is checked after a complete response from provider-reported cumulative
usage, so that response may cross the threshold and missing usage cannot trigger it.

An invocation can be result-only or expose one ordered event iterator. Abandoning that
iterator before completion cancels its execution. `MODEL_DELTA` and `TOOL_PROGRESS`
are lossy when their buffer allowance is exhausted; checkpoint events are not.

## Durable Boundary

Each successfully committed durable boundary produces:

1. an immutable `Checkpoint` containing the complete recovery state and history;
2. a `DurableCommit` containing the expected revision, content digest, and explicit
   history change;
3. one atomic repository commit before `CHECKPOINT_COMMITTED` is emitted.

Exact commit retries are idempotent and conflicting revisions fail. Without an
explicit `RunRepository`, execution uses an invocation-local ephemeral repository and
does not look up a run by ID. The host must retain the returned checkpoint or provide
a repository for recovery.

Portable JSON is encoded explicitly by `jharness.kernel.wire`. Provider payloads stay
inside `jharness.models`, and storage layouts stay inside
`jharness.repository`. Optional traces are built and verified through
`jharness.kernel.diagnostics` without replaying models or tools.

## Extension Boundary

The host must supply a `Model` and may also supply a `ToolCatalogProvider`,
`ApprovalPolicy`, `BatchPolicy`, `HistoryReducer`, and `RunRepository`. It chooses and
configures retry and fallback composition, and owns credentials, authorization,
isolation, observation, and backend lifecycle. Extensions compose around these ports;
they do not replace the kernel state machine.

Normative details live in the contracts:

| Concern | Contract |
| --- | --- |
| States and checkpoints | [State machine](../contracts/v0/state-machine.md) |
| Start, continue, resume, controls, and deadlines | [Run control](../contracts/v0/run-control.md) |
| Tool selection, approval, execution, and commit | [Tool scheduling](../contracts/v0/tool-scheduling.md) |
| Streaming rules | [Model stream](../contracts/v0/model-stream.md) |
| Repository atomicity and idempotency | [Repository](../contracts/v0/repository.md) |
| Trace ordering and verification | [Run trace](../contracts/v0/run-trace.md) |
