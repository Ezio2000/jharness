# Architecture

JHarness keeps durable execution in a dependency-free kernel and injects deployment
choices through narrow public protocols. The [package overview](../README.md#install)
lists the five independently installable distributions.

## Packages

Each integration package depends only on the exact matching `jharness-kernel`, never
on another integration. Every distribution owns one `jharness.<component>` namespace
portion; none owns `jharness/__init__.py`.

| Layer | Owns |
| --- | --- |
| Host application | Credentials, authorization, isolation, deployment policy, artifact persistence, HTTP client and backend lifecycle |
| `jharness.kernel` | Immutable domain values, model/tool ports, state transitions, approval, limits, checkpoints, events, diagnostics, repository protocol, and explicit v0 wire codecs |
| `jharness.toolkit` | Tool registration, function adaptation, JSON Schema validation, retry, and circuit breaking |
| `jharness.models` | Provider HTTP/SSE clients, protocol codecs, capability profiles, and retry/fallback composition |
| `jharness.repository` | Memory, SQLite, MySQL, and Redis storage implementations |
| `jharness.tools` | Filesystem, shell, interaction, and child-agent tools |
| Remote provider | Model inference and provider-hosted tool execution |

## Execution

`Runtime` is immutable configuration. Each `start`, `continue_from`, or `resume` call
creates one single-use `Invocation`, runs the same engine, and returns its last
committed `Checkpoint`.

The lifecycle has six states:

- `Planning` invokes the configured model once.
- `ToolsPending` executes a non-empty remaining suffix of `RuntimeToolCall` values.
- `Suspended` preserves the exact active state for a later `resume`.
- `Completed`, `Failed`, and `Limited` are terminal.

Model responses with runtime calls enter `ToolsPending`; responses without them
complete from visible content, including final or partial provider-tool output.
Runtime tool results commit in model order even when execution is concurrent.
Parallel batches require calls declared parallel, read-only, and idempotent.

Limits cap planning steps, runtime tool-call count, batch size, concurrency, progress
buffering, and optional elapsed time. The token limit is checked after a complete
response against provider-reported cumulative usage: that response may cross the
threshold, and missing usage cannot trigger it.

An invocation can be result-only or expose one ordered event iterator. Abandoning the
iterator before completion cancels execution. `MODEL_DELTA` and `TOOL_PROGRESS` may be
dropped when their buffer allowance is exhausted; checkpoint events are retained.

## Model Boundary

`ModelCapabilities` declares native input/output modalities, supported tool-choice
types, runtime/provider tools, parallel behavior, structured output, seed, streaming,
and usage. Each protocol profile owns one immutable capability value plus wire policy;
its client returns `profile.capabilities` directly. Supplier factories compose
profiles, while shared codecs remain independent of supplier names.

| Value | Meaning |
| --- | --- |
| `ContentPart` | Native text, image, audio, video, or file content |
| `RuntimeToolCall` | Host work executed through `ToolCatalogProvider`, approval, batching, and tool-result history |
| `ProviderToolCall` | Work executed remotely and recorded by the runtime |
| `ModelResponse.output` / assistant `Message.output` | One ordered, interleaved sequence of content and both call kinds |
| Model deltas | Live observations addressed by `output_index` and optional `content_index`; the terminal full response is authoritative |

`ToolChoice` selects `auto`, `none`, `required`, or an exact `runtime`/`provider` target.
Unsupported choices and unsupported requests to disable parallel runtime calls are
rejected before model invocation. Provider-only selection is outside that parallel
control. Provider tool identities are namespaced; equal wire names from different
suppliers do not identify the same capability.

Native image capability and hosted image generation are separate declarations. For
example, a hosted tool can return an image while the model's native output is text.
The [model guide](model-adapters.md) covers profile selection, ordered projections,
provider configuration, and the durable artifact-store requirements for generated images.

## Durable Boundary

Each successful boundary produces an immutable `Checkpoint` with complete recovery
state and history, and a `DurableCommit` with expected revision, digest, and explicit
history change. The repository commits atomically before `CHECKPOINT_COMMITTED` is
emitted. Exact retries are idempotent; conflicting revisions fail.

Without an explicit `RunRepository`, execution uses an invocation-local ephemeral
repository and does not look up runs by ID. The host must retain the returned checkpoint
or provide a repository for recovery.

`jharness.kernel.wire` explicitly encodes portable JSON. Provider envelopes remain in
`jharness.models`; selected native data may be retained in explicit `data`, `metadata`,
or provider-tool fields for lossless history round-trips. Storage layouts belong to
`jharness.repository`. Optional `jharness.kernel.diagnostics` traces can be built and
verified without replaying models or tools.

## Extension Boundary

The host supplies a `Model` and optionally a `ToolCatalogProvider`, `ApprovalPolicy`,
`BatchPolicy`, `HistoryReducer`, and `RunRepository`. Retry and fallback compose around
these ports; extensions do not replace the state machine.

Toolkit [function adapters](../packages/jharness-toolkit/README.md#function-adapters)
bind named business parameters or one freeform string, with explicit keyword-only
`ToolContext` injection. Schemas stay host-declared; signature inspection checks Python
argument binding. JSON-compatible returns become successful results, while native
`ToolResult` values pass through with their existing lifecycle semantics.

| Normative behavior | Contract |
| --- | --- |
| States and checkpoints | [State machine](../contracts/v0/state-machine.md) |
| Start, continue, resume, controls, and deadlines | [Run control](../contracts/v0/run-control.md) |
| Tool selection, approval, execution, and commit | [Tool scheduling](../contracts/v0/tool-scheduling.md) |
| Streaming | [Model stream](../contracts/v0/model-stream.md) |
| Repository atomicity and idempotency | [Repository](../contracts/v0/repository.md) |
| Trace ordering and verification | [Run trace](../contracts/v0/run-trace.md) |
