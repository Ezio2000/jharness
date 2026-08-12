# Model Adapters

`jharness-models` translates the provider-neutral `jharness.kernel.Model` protocol to
three explicit wire APIs:

| Import | Wire API | Configuration |
| --- | --- | --- |
| `jharness.models.openai` | OpenAI Chat Completions | `OpenAIChatCompletionsModel` and `OpenAIChatCompletionsProfile` |
| `jharness.models.openai` | OpenAI Responses | `OpenAIResponsesModel` and `OpenAIResponsesProfile` |
| `jharness.models.anthropic` | Anthropic Messages | `AnthropicModel` and `AnthropicProfile` |
| `jharness.models.deepseek` | Compatible Chat Completions, Messages, or native Responses endpoint | DeepSeek profile factories for the concrete adapters above |

Install the adapter package with `uv add jharness-models`. Provider APIs are not
flattened into `jharness.models`; import from the namespaces shown above.

## Configure a Provider

```python
import os

from jharness.models.openai import OpenAIChatCompletionsModel

model = OpenAIChatCompletionsModel(
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.environ["OPENAI_API_KEY"],
    model=os.environ["OPENAI_MODEL"],
)
```

The Anthropic equivalent is:

```python
from jharness.models.anthropic import AnthropicModel

model = AnthropicModel(
    base_url=os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
    api_key=os.environ["ANTHROPIC_API_KEY"],
    model=os.environ["ANTHROPIC_MODEL"],
)
```

OpenAI Responses is a separate adapter rather than a mode on Chat Completions:

```python
import os

from jharness.models.openai import OpenAIResponsesModel

model = OpenAIResponsesModel(
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.environ["OPENAI_API_KEY"],
    model=os.environ["OPENAI_MODEL"],
)
```

Profiles declare endpoint capabilities and request-shape differences. The runtime
checks those capabilities before invocation instead of guessing from a model name.
Every profile carries one immutable `ModelCapabilities` value, and the model client
returns that same value unchanged. There is no second set of per-feature profile
booleans for the client to translate.

The default `OpenAIResponsesProfile` is deliberately conservative: text input and
output, runtime functions, streaming, and usage only. It does not claim image/file
input, structured output, JSON mode, or provider-hosted tools for an arbitrary model
identifier. The host opts into only features verified for its selected model by
supplying an explicit profile.

For example, narrow Chat Completions to text input while retaining its other default
capabilities:

```python
from dataclasses import replace

from jharness.models.openai import OpenAIChatCompletionsProfile

default = OpenAIChatCompletionsProfile()
text_only = OpenAIChatCompletionsProfile(
    name="text-only-chat",
    capabilities=replace(
        default.capabilities,
        input_modalities=frozenset({"text"}),
    ),
)
```

`ModelCapabilities.tool_choice_types` declares the exact accepted choice vocabulary;
`parallel_runtime_tool_calls` and `parallel_runtime_tool_call_control` describe runtime-owned calls
only. The latter says whether a model that may return parallel runtime calls can honor
`allow_parallel_runtime_tool_calls=False`; provider-only selection neither requires that
control nor emits its wire field. `seed` declares whether `ModelOptions.seed` is
accepted. Protocol-only wire choices—such as whether automatic tool choice is explicit,
how reasoning history is encoded, or which fields a hosted tool accepts—remain profile
policy rather than kernel semantics.

## Capability and Execution Boundaries

The kernel values deliberately do not copy a supplier's feature list:

| Capability | How it is declared | Who performs it |
| --- | --- | --- |
| Image understanding | `"image"` in `ModelCapabilities.input_modalities` and an image `ContentPart` in the request | The model |
| Native image output | `"image"` in `ModelCapabilities.output_modalities` | The model |
| Host function call | `RuntimeToolSpec` (`StructuredToolSpec` or `FreeformToolSpec`) in `ModelRequest.runtime_tools`; returned as the matching `RuntimeToolCall` | JHarness runtime and the host tool catalog |
| Hosted image generation or web search | Namespaced `ProviderToolId` in `ModelCapabilities.provider_tools`, requested with `ProviderToolSpec`, and returned as `ProviderToolCall` | The remote provider |
| Mixed protocol result | `ModelResponse.output` containing ordered `ContentPart`, `RuntimeToolCall`, and `ProviderToolCall` values | The adapter maps it; the kernel preserves it |

A hosted image-generation result may contain an image in
`ProviderToolCall.output` even when the model itself advertises only text output. The
provider tool declaration says how and where the image is produced; it is not a
substitute for declaring image understanding.

The current adapters expose these boundaries as follows:

| Adapter/profile | Default model input | Native model output | Runtime tools | Provider-hosted tools | Conversation rule |
| --- | --- | --- | --- | --- | --- |
| OpenAI Chat Completions | Text and image | Text | Function tools | None | Complete JHarness history is encoded as messages |
| Anthropic Messages | Text, image, and file | Text | Client `tool_use` blocks | Profile-installed server-tool codecs | Complete JHarness history is encoded as Messages blocks |
| OpenAI Responses default | Text | Text | Function tools | None | Complete ordered history is encoded as Responses input items with `store=false` |
| OpenAI Responses explicit profile | Host-declared subset of text, image, and file | Text | Function tools | Host-declared image generation and/or web search | Profile storage policy and complete ordered history are authoritative |
| DeepSeek Responses profile | Text only | Text | Function and `apply_patch` freeform tools | `deepseek.responses/web_search` | Strictly stateless; complete history is resent |
| DeepSeek Anthropic profile | Text only | Text | Client `tool_use` blocks | `deepseek.anthropic/web_search` | Complete Messages history, including opaque search results, is replayed exactly |

Profiles remain authoritative. The runtime rejects request modalities and tool
identities that the selected explicit profile does not advertise before network
invocation.

### Profile Ownership

| Layer | Declares | Must not do |
| --- | --- | --- |
| Kernel `ModelCapabilities` | Exact model modalities, tool-choice types, runtime/provider tools, parallel behavior, structured output, seed, streaming, and usage | Name a supplier or encode HTTP/SSE fields |
| Protocol profile | One `ModelCapabilities` plus immutable wire policies for Chat Completions, Responses, or Messages | Duplicate capabilities as `supports_*` flags |
| Supplier factory | Compose a protocol profile for one concrete endpoint, such as DeepSeek thinking or Responses | Add supplier checks to a shared codec |
| Protocol codec/client | Consume the profile, validate wire data, and expose `profile.capabilities` unchanged | Infer features from model names or translate a second capability representation |

### Vision and Hosted Image Generation

Use a public `ContentPart` to supply an image for model understanding:

```python
import os

from jharness.kernel import ContentPart, Message

message = Message(
    "user",
    parts=(
        ContentPart.text_part("Describe the important visual details."),
        ContentPart(
            type="image",
            uri=os.environ["INPUT_IMAGE_URL"],
            media_type="image/jpeg",
        ),
    ),
)
```

Declare OpenAI-hosted image generation in an explicit model profile. Generated bytes
must be externalized through a host-owned `ResponsesArtifactStore` before the model
response can enter durable history:

```python
import asyncio
import os
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

from jharness.kernel import ArtifactRef, ProviderToolId, ProviderToolSpec, Runtime, ToolChoice
from jharness.models.openai import (
    OpenAIResponsesModel,
    OpenAIResponsesProfile,
    ResponsesArtifactStore,
    ResponsesImageGenerationTool,
    ResponsesProviderToolRegistry,
)


class LocalImageArtifacts(ResponsesArtifactStore):
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, digest: str) -> Path:
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("invalid image artifact digest")
        return self.root / digest[:2] / digest

    @staticmethod
    def _write_atomically(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if sha256(path.read_bytes()).hexdigest() != path.name:
                raise ValueError("existing image artifact is corrupt")
            return
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(data)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    async def save_image(self, data, *, media_type, call_id, context):
        del call_id, context
        digest = sha256(data).hexdigest()
        await asyncio.to_thread(self._write_atomically, self._path(digest), data)
        return ArtifactRef(
            f"sha256:{digest}",
            media_type=media_type,
            size_bytes=len(data),
            sha256=digest,
        )

    async def load_image(self, artifact, *, call_id, context):
        del call_id, context
        digest = artifact.sha256
        if digest is None or artifact.ref != f"sha256:{digest}":
            raise ValueError("invalid image artifact reference")
        data = await asyncio.to_thread(self._path(digest).read_bytes)
        if sha256(data).hexdigest() != digest:
            raise ValueError("stored image artifact is corrupt")
        return data


image_generation = ProviderToolId("openai.responses", "image_generation")
base = OpenAIResponsesProfile()
profile = OpenAIResponsesProfile(
    capabilities=replace(
        base.capabilities,
        tool_choice_types=base.capabilities.tool_choice_types | {"provider"},
        provider_tools=frozenset({image_generation}),
    ),
    provider_tool_registry=ResponsesProviderToolRegistry(
        (ResponsesImageGenerationTool(tool=image_generation),)
    ),
)
model = OpenAIResponsesModel(
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.environ["OPENAI_API_KEY"],
    model=os.environ["OPENAI_MODEL"],
    profile=profile,
    artifact_store=LocalImageArtifacts(Path("artifacts").resolve()),
)
runtime = Runtime(
    model=model,
    provider_tools=(
        ProviderToolSpec(
            image_generation,
            {"size": "1024x1024", "output_format": "png"},
        ),
    ),
    tool_choice=ToolChoice(type="provider", provider_tool=image_generation),
)
```

The client persists the decoded image before returning `ModelResponse` and replaces
its inline base64 with an `ArtifactRef`. Before a later model turn, it loads that
artifact into an invocation-local wire request. Durable checkpoints and repository
history therefore never retain generated image base64. Partial streaming images remain
live-only. The JHarness tool registry is not involved.

`call_id` is provider-controlled and response-scoped. An artifact store must not use it
as a filesystem path or assume it is globally unique. A successful save is durable;
repeated saves of the same bytes are idempotent, and an existing reference is never
reassigned to different content. The returned `ArtifactRef` must remain stable across
process restarts and carry exact `size_bytes` and SHA-256 metadata. A load either returns
the exact referenced bytes or fails the model turn.

Artifact saving happens before the response checkpoint is committed. Cancellation,
subsequent validation failure, or repository failure can therefore leave an unreferenced
save. The host must retain reachable artifacts for at least as long as their checkpoints,
configure the same store when recovering a run, and garbage-collect staged artifacts
that never become reachable from committed history. Content-addressed storage, as in the
example, makes repeated saves safe and simplifies that cleanup.

### Responses Storage Policy

The default OpenAI Responses profile sends `store=false` and requests
`reasoning.encrypted_content`, allowing reasoning items to round-trip in complete
history without relying on provider storage. Set `store=True` explicitly only when the
host permits provider retention; `include` may then be empty. `store` and `include` are
first-class profile fields and cannot be overridden through `extra_request_body`.
Compatible endpoints that do not implement these fields use `store=None` and an empty
`include`, as the DeepSeek Responses profile does.

## DeepSeek Profiles

DeepSeek factories configure one of the concrete adapters:

```python
import os

from jharness.models.deepseek import deepseek_openai_chat_profile
from jharness.models.openai import OpenAIChatCompletionsModel

model = OpenAIChatCompletionsModel(
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=os.environ["DEEPSEEK_API_KEY"],
    model=os.environ["DEEPSEEK_MODEL"],
    profile=deepseek_openai_chat_profile(thinking=True, effort="high"),
)
```

`deepseek_anthropic_profile` configures `AnthropicModel` instead. Both factories
require an explicit `thinking` value; `effort` accepts `"high"` or `"max"` only when
thinking is enabled.

DeepSeek's native Responses endpoint uses the Responses adapter with its own strict
profile:

```python
import os

from jharness.models.deepseek import deepseek_openai_responses_profile
from jharness.models.openai import OpenAIResponsesModel

model = OpenAIResponsesModel(
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    api_key=os.environ["DEEPSEEK_API_KEY"],
    model="deepseek-v4-flash",
    profile=deepseek_openai_responses_profile(effort="none"),
)
```

This profile accepts only `deepseek-v4-flash`, text input and output, runtime function
tools, the exact freeform runtime tool `apply_patch`, and the provider-hosted
`deepseek.responses/web_search` tool. Web search accepts the proven `web_search` and
`web_search_2025_08_26` wire variants. It rejects image or file input, image generation,
unmodeled provider tools, `seed`, and requests to disable parallel runtime calls while
runtime tools remain selectable. These checks happen locally instead of relying on fields
the compatible endpoint may silently ignore. DeepSeek Responses is stateless: every
request carries the complete ordered history, and the codec omits `store` and
`previous_response_id` instead of relying on provider-managed conversation state. It
never uses a response ID as a continuation handle. See the
[DeepSeek Responses API guide](https://api-docs.deepseek.com/guides/responses_api/)
for the upstream endpoint behavior.

DeepSeek can emit `response.web_search_call.completed` after an individual search
action has finished even when the later `response.output_item.done` item reports that
action as failed. The DeepSeek profile therefore exposes that lifecycle event as an
advisory provider-tool delta while keeping the portable call status `in_progress`.
The `response.output_item.done` item finalizes the live `completed`, `incomplete`, or
`failed` status. The terminal full response remains the authoritative durable value
and must report the same provider-tool status.

## Retry and Fallback

Retry and fallback wrap any kernel `Model`:

```python
from jharness.models.decorators import FallbackModel, RetryingModel

model = FallbackModel(
    RetryingModel(primary_model, max_attempts=3),
    RetryingModel(backup_model, max_attempts=2),
)
```

`max_attempts` includes the first call. `RetryingModel` retries only a `ModelError`
whose `info.retryable` flag is true, using bounded exponential backoff with jitter.
It treats numeric or HTTP-date `retry_after` metadata as a lower bound and, where the
configured maximum leaves room, applies positive jitter above it. If the lower bound
exceeds the maximum or the delay cannot fit before `RunContext.deadline`, the error is
propagated instead of retried. The decorator converts the deadline once to a monotonic
budget and checks it again before the next attempt. `Runtime` applies the deadline to
in-flight model work.

`FallbackModel` calls its backup only after the primary raises a retryable
`ModelError`. Its advertised capabilities are the field-by-field intersection of the
two models, including exact tool-choice and provider-tool intersections, preventing
the runtime from relying on a capability either model marks unsupported.
Endpoint-specific request-shape compatibility remains the host's responsibility.

Neither decorator switches attempts once the first streaming delta is offered to the
host sink. This prevents deltas from separate provider attempts from being presented
as one response. Non-retryable errors, protocol errors, sink failures, and
cancellation propagate unchanged.

## Ordered Responses and Streaming

Chat Completions and Messages are normalized into the same ordered kernel result as
Responses. Messages retains native block order; Chat Completions places its content
before the provider-ordered call array because that wire protocol exposes them as
separate fields. Neither adapter exposes separate content and tool-call result arrays
to the kernel. Responses function and custom items use the provider `call_id` as
`RuntimeToolCall.id`,
while hosted tool items become namespaced `ProviderToolCall` values.

Streaming events use `output_index` and, for nested content, `content_index`.
Provider-hosted tool progress is exposed as live-only
`ModelProviderToolCallDelta`; it is not scheduled as runtime work. For Responses SSE,
the full response carried by the terminal event is decoded as the authoritative
`ModelResponse`.

## Transport Boundary

All adapter clients accept an optional host-owned `httpx.AsyncClient`; the host must
close it. Without one, each invocation creates and closes its own client. The default
transport timeout is 10 seconds for connection setup and 60 seconds for other HTTP
phases. Passing `timeout=None` disables the HTTP phase timeout, not the run deadline.

Complete responses and SSE streams produce the same `ModelResponse` type. Streaming
deltas are ordered and backpressured through the host sink. Provider transport,
payload, and stream failures become structured `ModelError` values; exceptions raised
by the host sink remain unchanged. Chat Completions and Messages complete JSON and
HTTP error bodies share an 8 MiB default bound. Responses raises the body, SSE line,
and SSE event defaults to 64 MiB because an image-generation result can contain inline
image data. Every bound is configurable with the corresponding positive
`max_response_body_bytes`, `max_sse_line_bytes`, or `max_sse_event_bytes` option.

Provider HTTP/SSE envelopes and codecs stay in `jharness.models`. An adapter may retain
selected native item data in explicit `ContentPart.data`, `metadata`, or
`ProviderToolCall` fields when the complete ordered history must round-trip, but those
opaque details do not become general kernel semantics. The package implements OpenAI
Chat Completions, Anthropic Messages, and OpenAI-compatible Responses. Hosted tools
are available only when explicitly advertised by the selected profile and requested
through `ProviderToolSpec`. Responses installs independent codecs for image generation
and web search; Anthropic Messages installs independent server-tool codecs, including
DeepSeek's verified web search. Chat Completions remains a runtime-tool protocol unless
a concrete Chat-compatible endpoint exposes a stable hosted-call lifecycle.
Provider-managed conversation state, batch jobs, and file-upload management remain
outside this package.
