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
| Anthropic Messages | Client `tool_use` | None | Native block order is retained |
| OpenAI Responses | Function calls | Image generation and web search | Native Responses item order is retained |

DeepSeek's native Responses endpoint uses `OpenAIResponsesModel` with
`deepseek_openai_responses_profile`. That profile is text-only, accepts only
`deepseek-v4-flash`, exposes provider-hosted web search, and forces stateless requests
with complete history.

Model modalities describe what the model itself understands or produces. Tool
ownership is separate: `ToolCall` is executed by the JHarness runtime, while a
`ProviderToolCall` records work already executed by the supplier. Both remain
interleaved with `ContentPart` values in ordered output.

Each protocol profile contains the exact immutable `ModelCapabilities` returned by
its model client. Tool selection is declared as a set of supported types rather than a
coarse boolean. Supplier factories only compose protocol capabilities and wire
policies; the shared codecs contain no supplier-name branches.

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
