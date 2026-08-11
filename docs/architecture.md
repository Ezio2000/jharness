# Architecture

JHarness keeps durable execution in a dependency-free kernel and injects every
deployment choice through a narrow public protocol.

## Packages

| Package | Owns |
| --- | --- |
| `jharness.kernel` | Runtime, immutable state, model/tool ports, limits, checkpoints, events, diagnostics, repository protocol, and v0 wire codecs |
| `jharness.toolkit` | Tool registration, function adaptation, JSON Schema validation, retry, and circuit breaking |
| `jharness.models` | Provider HTTP/SSE clients, explicit wire codecs, capability profiles, and retry/fallback model composition |
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
- `ToolsPending` executes a non-empty remaining suffix of runtime-owned `ToolCall` values.
- `Suspended` preserves the exact active state for a later `resume`.
- `Completed`, `Failed`, and `Limited` are terminal.

Each model response has one ordered `output` sequence containing `ContentPart`,
runtime-owned `ToolCall`, and provider-owned `ProviderToolCall` items. The runtime
schedules only `ToolCall` values. A response with runtime calls enters `ToolsPending`;
a response without them completes from its visible content, including content returned
inside a completed provider tool call. A `ProviderToolCall` records work already run by
the remote provider and never enters the JHarness tool scheduler.

Runtime tool results are committed in model order even when execution is concurrent.
Parallel batches require calls declared parallel, read-only, and idempotent. Limits cap
planning steps, runtime tool-call count, batch size, concurrency, progress buffering,
and optional elapsed time. The token limit is checked after a complete response from
provider-reported cumulative usage, so that response may cross the threshold and
missing usage cannot trigger it.

An invocation can be result-only or expose one ordered event iterator. Abandoning that
iterator before completion cancels its execution. `MODEL_DELTA` and `TOOL_PROGRESS`
are lossy when their buffer allowance is exhausted; checkpoint events are not.

## Model Boundary

The model abstraction separates three questions that provider APIs often combine:

| Question | Kernel representation | Execution owner | Boundary rule |
| --- | --- | --- | --- |
| What media can the model itself understand or produce? | Exact `ModelCapabilities.input_modalities` and `output_modalities`; `ContentPart` carries the data | The selected model | Modalities describe native media such as text, image, audio, video, and file. They do not imply a tool or describe the output of a provider-hosted tool. |
| Which host functions can the model request? | `ModelRequest.runtime_tools`, `ToolChoice(type="runtime")`, and output `ToolCall` | JHarness `Runtime`, through the host-supplied `ToolCatalogProvider` | Only these calls enter `ToolsPending`, approval, batching, execution, and tool-result history. |
| Which remote capabilities may the model invoke? | Namespaced `ProviderToolId`, `ProviderToolSpec`, `ToolChoice(type="provider")`, and output `ProviderToolCall` | The remote provider | The adapter validates provider-specific configuration and maps its wire events and result content. The runtime records the call but never executes it. |
| Which tool-selection policies can the endpoint honor? | Exact `ModelCapabilities.tool_choice_types`, `parallel_tool_calls`, and `parallel_tool_call_control` | Kernel validates; adapter encodes | Unsupported selection types and requests to disable parallel calls are rejected before provider invocation. |
| What actually happened in the response? | Ordered `ModelResponse.output` and assistant `Message.output` | Kernel preserves; adapter maps | Content and both tool-call kinds remain interleaved in protocol order instead of being split into lossy parallel arrays. |
| What happened while streaming? | Deltas addressed by `output_index` and, where needed, `content_index`; provider-tool progress uses `ModelProviderToolCallDelta` | Adapter decodes; host observes | Deltas are live-only observations. The terminal full `ModelResponse` is authoritative and is the value committed to history. |

Consequently, image understanding is declared with the `image` input modality, and a
model that natively emits images declares the `image` output modality. Image generation
through a hosted tool is instead declared with a provider tool identity such as
`ProviderToolId("openai.responses", "image_generation")`: OpenAI performs that
operation, and its `ProviderToolCall.output` may contain an image even when the model's
native output modality is text.

`ToolChoice` uses `auto`, `none`, or `required` for a general policy and `runtime` or
`provider` for an exact target. Provider tool identifiers are namespaced because equal
wire names from different suppliers are not interchangeable. Convenience methods such
as `runtime_tool_calls()`, `provider_tool_calls()`, and `visible_parts()` are derived
projections; there are no mirrored legacy `parts` and `tool_calls` response fields to
reconcile.

`ModelCapabilities` is the single model-feature declaration. A protocol profile owns
that immutable value together with only wire-level policy. Model clients return
`profile.capabilities` directly. Supplier factories compose protocol profiles; shared
codecs never branch on a supplier name or namespace. This keeps endpoint variation at
the profile boundary without duplicating capability flags or embedding vendor policy
in the kernel.

## Layer Ownership

| Layer | Owns | Does not own |
| --- | --- | --- |
| Host application | Credentials, authorization, runtime tool implementations, artifact persistence, HTTP client lifecycle, and deployment policy | Provider wire parsing or kernel state transitions |
| `jharness.models` adapter | Endpoint URL shape, authentication headers, JSON/SSE codecs, provider error mapping, and immutable protocol profiles | Runtime tool execution, checkpoint persistence, lifecycle decisions, or supplier-name branching in shared codecs |
| `jharness.kernel` model port | Portable requests, capabilities, ordered output, validation, streaming observations, and immutable history values | Supplier payload fields or remote tool execution |
| JHarness runtime engine | Planning, scheduling only `ToolCall`, approval, limits, durable transitions, and terminal projection | Executing `ProviderToolCall` or interpreting its provider-specific configuration |
| Remote model provider | Model inference and any requested provider-hosted tools | Host runtime tools and JHarness durability |

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

Portable JSON is encoded explicitly by `jharness.kernel.wire`. Its ordered output is
the durable provider-neutral fact. Provider HTTP/SSE envelopes and codecs stay inside
`jharness.models`; an adapter may retain selected provider data only through explicit
portable `data`, `metadata`, or provider-tool fields needed for lossless history
round-trips. Storage layouts stay inside `jharness.repository`. Optional traces are
built and verified through `jharness.kernel.diagnostics` without replaying models or
tools.

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
