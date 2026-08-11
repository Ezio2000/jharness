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
Custom profiles can enable only features the selected endpoint actually supports.
Every profile carries one immutable `ModelCapabilities` value, and the model client
returns that same value unchanged. There is no second set of per-feature profile
booleans for the client to translate.

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
`parallel_tool_call_control` says whether `allow_parallel_tool_calls=False` can be
honored; and `seed` declares whether `ModelOptions.seed` is accepted. Protocol-only
wire choices—such as whether automatic tool choice is explicit, how reasoning history
is encoded, or which fields a hosted tool accepts—remain profile policy rather than
kernel semantics.

## Capability and Execution Boundaries

The kernel values deliberately do not copy a supplier's feature list:

| Capability | How it is declared | Who performs it |
| --- | --- | --- |
| Image understanding | `"image"` in `ModelCapabilities.input_modalities` and an image `ContentPart` in the request | The model |
| Native image output | `"image"` in `ModelCapabilities.output_modalities` | The model |
| Host function call | `ToolSpec` in `ModelRequest.runtime_tools`; returned as `ToolCall` | JHarness runtime and the host tool catalog |
| Hosted image generation or web search | Namespaced `ProviderToolId` in `ModelCapabilities.provider_tools`, requested with `ProviderToolSpec`, and returned as `ProviderToolCall` | The remote provider |
| Mixed protocol result | `ModelResponse.output` containing ordered `ContentPart`, `ToolCall`, and `ProviderToolCall` values | The adapter maps it; the kernel preserves it |

A hosted image-generation result may contain an image in
`ProviderToolCall.output` even when the model itself advertises only text output. The
provider tool declaration says how and where the image is produced; it is not a
substitute for declaring image understanding.

The current adapters expose these boundaries as follows:

| Adapter/profile | Default model input | Native model output | Runtime tools | Provider-hosted tools | Conversation rule |
| --- | --- | --- | --- | --- | --- |
| OpenAI Chat Completions | Text and image | Text | Function tools | None | Complete JHarness history is encoded as messages |
| Anthropic Messages | Text, image, and file | Text | Client `tool_use` blocks | None | Complete JHarness history is encoded as Messages blocks |
| OpenAI Responses | Text, image, and file | Text | Function tools | `openai.responses/image_generation` and `openai.responses/web_search` | Ordered history is encoded as Responses input items; JHarness does not rely on a stored conversation ID |
| DeepSeek Responses profile | Text only | Text | Function tools | `deepseek.responses/web_search` | Strictly stateless; complete history is resent |

Profiles remain authoritative. A custom compatible endpoint may advertise a narrower
set, and the runtime rejects unsupported request modalities or tool identities before
network invocation.

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

Declare OpenAI-hosted image generation separately and pass it to `Runtime`:

```python
from jharness.kernel import ProviderToolId, ProviderToolSpec, Runtime, ToolChoice

image_generation = ProviderToolId("openai.responses", "image_generation")
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

The Responses adapter sends this as an `image_generation` hosted tool. OpenAI executes
it, and the adapter records the call and its image result as one `ProviderToolCall` in
ordered output. The JHarness tool registry is not involved.

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
tools, and the provider-hosted `deepseek.responses/web_search` tool. It rejects image
or file input, image generation, unmodeled provider tools, `seed`, and attempts to
disable parallel tool calls. Web-search configuration must be omitted because the
endpoint does not apply it. These checks happen locally instead of relying on fields
the compatible endpoint may silently ignore. DeepSeek Responses is stateless: every
request carries the complete ordered history, and the codec omits `store` and
`previous_response_id` instead of relying on provider-managed conversation state. It
never uses a response ID as a continuation handle. See the
[DeepSeek Responses API guide](https://api-docs.deepseek.com/guides/responses_api/)
for the upstream endpoint behavior.

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
to the kernel. Responses function items use the provider `call_id` as `ToolCall.id`,
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
through `ProviderToolSpec`; current Responses support covers image generation and web
search, while the Chat Completions and Messages adapters expose runtime tools only.
Provider-managed conversation state, batch jobs, and file-upload management remain
outside this package.
