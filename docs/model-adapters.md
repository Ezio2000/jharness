# Model Adapters

`jharness-models` translates the provider-neutral `jharness.kernel.Model` protocol to
three explicit wire APIs:

| Import | Wire API | Configuration |
| --- | --- | --- |
| `jharness.models.openai` | OpenAI Chat (Chat Completions API) | `OpenAIChatModel` and `OpenAIChatProfile` |
| `jharness.models.openai` | OpenAI Responses | `OpenAIResponsesModel` and `OpenAIResponsesProfile` |
| `jharness.models.anthropic` | Anthropic Messages | `AnthropicMessagesModel` and `AnthropicMessagesProfile` |

Install the adapter package with `uv add jharness-models`. Provider APIs are not
flattened into `jharness.models`; import from the namespaces shown above.

## Configure a Provider

```python
import os

from jharness.models.openai import OpenAIChatModel

model = OpenAIChatModel(
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.environ["OPENAI_API_KEY"],
    model=os.environ["OPENAI_MODEL"],
)
```

The Anthropic Messages equivalent is:

```python
from jharness.models.anthropic import AnthropicMessagesModel

model = AnthropicMessagesModel(
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

Each profile owns one immutable `ModelCapabilities`, returned unchanged by its client.
The runtime checks requests against that declaration before network invocation.
Capabilities are host assertions for the exact model and endpoint, not live discovery:
an overstated profile may still be rejected, ignored, or degraded by the provider.

`profile.name` identifies the adapter in `ModelResponse.metadata["provider"]` and
`ModelErrorInfo.provider`. `to_assistant_message()` preserves it in durable history and
traces. Defaults are `openai-chat`, `openai-responses`, and `anthropic-messages`.

The default `OpenAIResponsesProfile` supports text input/output, runtime function and
custom tools, streaming, and usage. It enables no image/file input, structured output,
JSON mode, or hosted tools. `AnthropicMessagesProfile` also enables no hosted tools.
Official profile factories declare the supported hosted-tool identities; select them
only when the endpoint and model implement their advertised capabilities. Profiles
configure the standard protocol wire and cannot inject custom codecs or field sets.

For example, narrow Chat Completions to text input while retaining its other default
capabilities:

```python
from dataclasses import replace

from jharness.models.openai import OpenAIChatProfile

default = OpenAIChatProfile()
text_only = OpenAIChatProfile(
    name="text-only-chat",
    capabilities=replace(
        default.capabilities,
        input_modalities=frozenset({"text"}),
    ),
)
```

`tool_choice_types` declares the accepted choice vocabulary. The capability flags
`parallel_runtime_tool_calls` and `parallel_runtime_tool_call_control` apply to runtime
calls; the latter controls whether `allow_parallel_runtime_tool_calls=False` is
supported. Provider-only selection neither requires that control nor emits its wire
field. `seed` declares support for `ModelOptions.seed`.

## Capability and Execution Boundaries

Native modalities describe what the model understands or produces. Runtime tools are
executed by JHarness; provider-hosted tools are executed remotely. A hosted image tool
can therefore return an image even when the model's native output modality is text.
The [architecture](architecture.md#model-boundary) defines these kernel values and
ownership boundaries.

| Adapter/profile | Default model input | Native model output | Runtime tools | Hosted tools |
| --- | --- | --- | --- | --- |
| OpenAI Chat | Text, image | Text | Function calls | None |
| Anthropic Messages | Text, image, file | Text, container-upload file references | Client `tool_use` | Official web-search preset |
| OpenAI Responses default | Text | Text | Function and custom calls | None |
| OpenAI Responses official/explicit | Host-declared subset of text, image, file | Text | Function and custom calls | Official web-search and image-generation presets |

All adapters encode complete JHarness history. Responses storage and stateless
reasoning replay are described under [storage policy](#responses-storage-policy).

### Hosted-Tool Presets

A profile declares which provider tools may be requested;
`Runtime(..., provider_tools=...)` enables them for the invocation. Hosted presets
bypass local binding, approval, batching, and execution.

OpenAI Responses exposes web search and image generation:

```python
import os

from jharness.kernel import Runtime, ToolChoice
from jharness.models.openai import (
    OPENAI_RESPONSES_WEB_SEARCH,
    OpenAIResponsesModel,
    openai_responses_profile,
    openai_responses_web_search,
)

model = OpenAIResponsesModel(
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.environ["OPENAI_API_KEY"],
    model=os.environ["OPENAI_MODEL"],
    profile=openai_responses_profile(),
)
runtime = Runtime(
    model=model,
    provider_tools=(
        openai_responses_web_search({"search_context_size": "high"}),
    ),
    tool_choice=ToolChoice(
        type="provider",
        provider_tool=OPENAI_RESPONSES_WEB_SEARCH,
    ),
)
```

Anthropic Messages exposes its versioned server-side web search through the same
public pattern:

```python
import os

from jharness.kernel import Runtime
from jharness.models.anthropic import (
    AnthropicMessagesModel,
    anthropic_messages_profile,
    anthropic_messages_web_search,
)

model = AnthropicMessagesModel(
    base_url=os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
    api_key=os.environ["ANTHROPIC_API_KEY"],
    model=os.environ["ANTHROPIC_MODEL"],
    profile=anthropic_messages_profile(),
)
runtime = Runtime(
    model=model,
    provider_tools=(
        anthropic_messages_web_search(
            {"variant": "web_search_20260318", "max_uses": 3},
        ),
    ),
)
```

The Anthropic preset accepts the current `web_search_20250305`,
`web_search_20260209`, and `web_search_20260318` variants. With no `variant`, it uses
the basic `web_search_20250305` declaration; select a later capability-keyed variant
explicitly when the chosen model and deployment support it.

The generic profile classes allow narrower capability sets. Third-party endpoints
must implement the selected protocol's standard wire contract; vendor-specific
deviations require a user-owned adapter.

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

An `ArtifactRef` whose `media_type` is `image/*` is also an image modality even though
the provider transports it by file id. Responses encodes it as `input_image` with
`file_id`. Anthropic Messages maps JPEG, PNG, GIF, and WebP files to `image`, PDF and
plain text to `document`, and datasets or other MIME types to `container_upload`.
Chat Completions always uses the standard nested `file` content-part shape, so an
artifact also requires the profile's `"file"` input capability; top-level `file_id`
and `file_data` variants are not accepted.

Declare OpenAI-hosted image generation in an explicit model profile. Generated bytes
must be externalized through a host-owned `OpenAIResponsesArtifactStore` before the model
response can enter durable history:

```python
import asyncio
import os
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

from jharness.kernel import ArtifactRef, Runtime, ToolChoice
from jharness.models.openai import (
    OPENAI_RESPONSES_IMAGE_GENERATION,
    OpenAIResponsesArtifactStore,
    OpenAIResponsesModel,
    openai_responses_image_generation,
    openai_responses_profile,
)


class LocalImageArtifacts(OpenAIResponsesArtifactStore):
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


model = OpenAIResponsesModel(
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.environ["OPENAI_API_KEY"],
    model=os.environ["OPENAI_MODEL"],
    profile=openai_responses_profile(),
    artifact_store=LocalImageArtifacts(Path("artifacts").resolve()),
)
runtime = Runtime(
    model=model,
    provider_tools=(
        openai_responses_image_generation(
            {"size": "1024x1024", "output_format": "png"},
        ),
    ),
    tool_choice=ToolChoice(
        type="provider",
        provider_tool=OPENAI_RESPONSES_IMAGE_GENERATION,
    ),
)
```

The client saves decoded image bytes before returning `ModelResponse`, replacing
base64 with an `ArtifactRef`. Later turns load those bytes into an invocation-local
request. Durable history contains references; partial streaming images remain live-only.

The store must durably and idempotently save bytes, keep references stable across
restarts, and return exact `size_bytes` and SHA-256 metadata. Loads return the exact
referenced bytes or fail the turn. Provider-controlled, response-scoped `call_id` values
are neither filesystem paths nor globally unique storage keys.

Saving precedes checkpoint commit, so cancellation, validation, or repository failure
can leave unreferenced artifacts. Retain reachable artifacts for the checkpoint
lifetime, use the same store during recovery, and collect uncommitted saves.
Content-addressed storage, as above, supports idempotency and cleanup.

### Responses Storage Policy

The default OpenAI Responses profile sends `store=false` and requests
`reasoning.encrypted_content` through the standard `include` field so native reasoning
items can be replayed statelessly. Set `store=True` explicitly only when the host permits
provider retention; callers may then choose an empty `include`. Generic kernel reasoning
is never synthesized into a native Responses reasoning item, and an item without its
original encrypted content cannot be replayed in a stateless request.

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
Fallback composition assumes that each model already implements its selected standard
protocol contract; endpoint dialect handling is outside these adapters.

Neither decorator switches attempts once the first streaming delta is offered to the
host sink. This prevents deltas from separate provider attempts from being presented
as one response. Non-retryable errors, protocol errors, sink failures, and
cancellation propagate unchanged.

## Ordered Responses and Streaming

Use `runtime_tool_calls()`, `provider_tool_calls()`, and `visible_parts()` to project
the ordered output without maintaining separate result arrays.

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

Adapters retain selected native data in `ContentPart.data`, `metadata`, or
`ProviderToolCall` fields for complete history round-trips. These fields do not add
provider-specific kernel semantics. Hosted-tool mappings are fixed by each protocol;
Chat Completions supports runtime tools only. Provider-managed conversation state,
batch jobs, and file-upload management remain outside this package.
