# jharness-models

OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages adapters; DeepSeek
profiles; and provider-neutral model composition for the JHarness kernel.

```bash
uv add jharness-models
```

```python
from jharness.models.openai import OpenAIResponsesModel
```

| Adapter | Runtime tools | Provider-hosted tools | Ordered output |
| --- | --- | --- | --- |
| OpenAI Chat Completions | Function calls | None | Content and calls are normalized into `ModelResponse.output` |
| Anthropic Messages | Client `tool_use` | Profile-installed server-tool codecs | Native block order is retained |
| OpenAI Responses | Function and freeform calls | Profile-installed image generation and web search codecs | Native Responses item order is retained |

DeepSeek's native Responses endpoint uses `OpenAIResponsesModel` with
`deepseek_openai_responses_profile`. That profile is text-only, accepts only
`deepseek-v4-flash`, exposes provider-hosted web search plus the exact freeform
`apply_patch` runtime tool, and forces stateless requests with complete history. The
DeepSeek Anthropic profile independently exposes its verified server-side web search.

Model modalities describe what the model itself understands or produces. Tool
ownership is separate: `RuntimeToolCall` is executed by the JHarness runtime, while a
`ProviderToolCall` records work already executed by the supplier. Both remain
interleaved with `ContentPart` values in ordered output.

Each protocol profile contains the exact immutable `ModelCapabilities` returned by
its model client. The default Responses profile is conservative and text-only;
model-specific modalities, structured output, and hosted tools require explicit host
opt-in. Tool selection is declared as a set of supported types rather than a coarse
boolean. Supplier factories only compose protocol capabilities and wire policies; the
shared codecs contain no supplier-name branches.

OpenAI Responses sends `store=false` by default and requests encrypted reasoning
history. Hosted image generation additionally requires a host-owned
`ResponsesArtifactStore`; generated base64 is persisted externally and durable history
contains only integrity-bearing `ArtifactRef` values. Stores must be durable,
idempotent, safe for provider-controlled call ids, available during run recovery, and
responsible for retention and garbage collection of uncommitted saves.

Retry and fallback compose directly around model values:

```python
from jharness.models.decorators import FallbackModel, RetryingModel

model = FallbackModel(
    RetryingModel(primary_model, max_attempts=3),
    RetryingModel(backup_model, max_attempts=2),
)
```

Installing this distribution installs the exact matching `jharness-kernel` version.
Provider configuration and composition details are in the
[model adapter guide](https://github.com/Ezio2000/jharness/blob/main/docs/model-adapters.md).
