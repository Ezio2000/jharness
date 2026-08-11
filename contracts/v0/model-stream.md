# Kernel v0 Model Streaming

Model streaming is live observation, not durable state. There is one model
operation:

```text
Model.invoke(request, context, *, stream, emit_delta) -> ModelResponse
```

## Complete Request and Response

Every `ModelRequest` carries the complete durable message history plus two
disjoint tool declarations:

- `runtime_tools` are functions that the JHarness runtime may bind, approve,
  schedule, and execute;
- `provider_tools` are namespaced provider capabilities that the remote model
  service executes. Each declaration contains a `ProviderToolId(namespace,
  type)` and provider-owned configuration.

`tool_choice.type` is `auto`, `none`, `required`, `runtime`, or `provider`.
`runtime` targets one declared runtime tool by name; `provider` targets one
declared provider tool by `ProviderToolId`. `required` accepts either ownership
class. `allow_parallel_tool_calls` constrains runtime-owned calls returned for
host scheduling; the count and order of provider-owned items do not prove how
the remote provider executed them.

Every complete `ModelResponse` contains one non-empty ordered `output`. Its
items are exactly:

- `content`, carrying one `ContentPart`;
- `runtime_tool_call`, carrying id, name, and JSON arguments;
- `provider_tool_call`, carrying id, namespaced tool identity, status,
  arguments, provider-produced content, optional failure, and metadata.

The adapter preserves the provider's semantically observable item order.
Runtime and provider tool-call ids are unique across one output. Provider tool
status is `in_progress`, `completed`, `incomplete`, or `failed`; a failed call
has an error, and an in-progress call has no final output. Provider-specific wire payloads
and versioned tool names are adapter concerns, not portable message shapes.

An assistant message persists that same ordered output without splitting
content from calls. Only `runtime_tool_call` creates `ToolsPending` work.
Provider-tool output is projected into visible content in its output position.
That projection may be empty when the complete response consists only of
provider-tool facts; the ordered assistant output remains the durable result.

## Live Deltas

When `stream=false`, the model returns a complete response and does not call the
sink. When `stream=true`, `emit_delta` may receive only five provider-neutral
delta variants:

- `content`
- `tool_call`
- `reasoning`
- `provider_tool_call`
- `usage`

There are no started or completed stream items. `model_started` and
`model_finished` remain invocation observation events around `Model.invoke`;
they are not values in the model stream.

The provider adapter owns stream assembly and always returns one complete
`ModelResponse`. Every non-usage delta carries a zero-based `output_index` into
the ordered provider response. Content and reasoning deltas additionally carry
`content_index`; runtime tool-call deltas accumulate id, name, and JSON
arguments at their output position. Provider-tool deltas carry id,
`ProviderToolId`, optional normalized status, optional provider event name, and
opaque event data. Usage deltas merge field by field; an omitted value does not
clear a value already reported. There is no tool invocation mode.

Provider-tool progress, including partial image data, is live-only. The adapter
materializes the final provider call and its content in the returned output;
partial event data never becomes durable history.

Only the returned response can produce a `model_turn` checkpoint. Partial text,
reasoning, calls, and usage never enter snapshot history or metrics. Kernel does
not run a second response accumulator.

Adapters await each `emit_delta` call in stream order. Closing or cancelling the
invocation closes the provider stream before control returns. Pause,
conversation insertion, provider failure, iterator failure, or deadline before
return preserves the last committed checkpoint and discards partial deltas.

The delta sink is host code. Its exception propagates unchanged after provider
resources are closed; it is not normalized as a provider `ModelError`. Transport,
provider payload, iterator, and stream-protocol failures are normalized.

An adapter applies a finite default HTTP timeout unless a caller explicitly disables
that transport timeout; the run deadline remains authoritative in either case.
Complete JSON response bodies and HTTP error bodies have a finite configurable byte
limit. SSE input is strictly UTF-8, recognizes only CR/LF line endings, and is bounded
per line and per event before accumulation. Provider `Retry-After` values remain
available as error metadata, and semantic stream errors preserve their payload status
and code even when delivered under HTTP 2xx.

Kernel does not retry a partially observed stream. Retry and fallback decorators must
not expose deltas from a failed attempt before switching attempts or models.
