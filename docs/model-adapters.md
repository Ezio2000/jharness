# Model Adapters

`jharness-models` translates the provider-neutral `jharness.kernel.Model` protocol to
two explicit wire APIs:

| Import | Wire API | Configuration |
| --- | --- | --- |
| `jharness.models.openai` | OpenAI Chat Completions | `OpenAIChatCompletionsModel` and `OpenAIChatCompletionsProfile` |
| `jharness.models.anthropic` | Anthropic Messages | `AnthropicModel` and `AnthropicProfile` |
| `jharness.models.deepseek` | Either API above | DeepSeek profile factories |

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

Profiles declare endpoint capabilities and request-shape differences. The runtime
checks those capabilities before invocation instead of guessing from a model name.
Custom profiles can enable only features the selected endpoint actually supports.

DeepSeek factories configure one of the concrete adapters:

```python
from jharness.models.deepseek import deepseek_openai_chat_profile
from jharness.models.openai import OpenAIChatCompletionsModel

model = OpenAIChatCompletionsModel(
    base_url="https://api.deepseek.com",
    api_key=api_key,
    model=model_name,
    profile=deepseek_openai_chat_profile(thinking=True, effort="high"),
)
```

`deepseek_anthropic_profile` configures `AnthropicModel` instead. Both factories
require an explicit `thinking` value; `effort` accepts `"high"` or `"max"` only when
thinking is enabled.

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
two models, preventing the runtime from relying on a capability either model marks
unsupported. Endpoint-specific request-shape compatibility remains the host's
responsibility.

Neither decorator switches attempts once the first streaming delta is offered to the
host sink. This prevents deltas from separate provider attempts from being presented
as one response. Non-retryable errors, protocol errors, sink failures, and
cancellation propagate unchanged.

## Transport Boundary

Both adapters accept an optional host-owned `httpx.AsyncClient`; the host must close
it. Without one, each invocation creates and closes its own client. The default
transport timeout is 10 seconds for connection setup and 60 seconds for other HTTP
phases. Passing `timeout=None` disables the HTTP phase timeout, not the run deadline.

Complete responses and SSE streams produce the same `ModelResponse` type. Streaming
deltas are ordered and backpressured through the host sink. Provider transport,
payload, and stream failures become structured `ModelError` values; exceptions raised
by the host sink remain unchanged. Complete JSON response bodies and HTTP error bodies
share an 8 MiB default bound, configurable with a positive
`max_response_body_bytes`. SSE line and event sizes have additional positive,
configurable bounds.

Provider payloads and codecs stay in `jharness.models`; they never become durable
kernel wire data. The package implements OpenAI Chat Completions and Anthropic
Messages, not OpenAI Responses, provider-managed conversations, hosted tools, batch
jobs, or file-upload APIs.
