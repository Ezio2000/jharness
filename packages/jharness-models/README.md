# jharness-models

OpenAI Chat (Chat Completions API), OpenAI Responses, and Anthropic Messages adapters;
DeepSeek profiles; and provider-neutral model composition for the JHarness kernel.

```bash
uv add jharness-models
```

```python
from jharness.models.openai import OpenAIResponsesModel
```

| Adapter | Runtime tools | Provider-hosted tools | Ordered output |
| --- | --- | --- | --- |
| OpenAI Chat | Function calls | None | Content and calls are normalized into `ModelResponse.output` |
| Anthropic Messages | Client `tool_use` | Official web-search preset or explicit server-tool codecs | Native block order is retained |
| OpenAI Responses | Function and freeform calls | Official web-search and image-generation presets or explicit codecs | Native Responses item order is retained |

DeepSeek's native Responses endpoint uses `OpenAIResponsesModel` with
`deepseek_responses_profile`. That profile is text-only, accepts only
`deepseek-v4-flash`, exposes provider-hosted web search plus the exact freeform
`apply_patch` runtime tool, and forces stateless requests with complete history. The
DeepSeek Messages profile independently exposes its verified server-side web search.

Model modalities describe what the model itself understands or produces. Tool
ownership is separate: `RuntimeToolCall` is executed by the JHarness runtime, while a
`ProviderToolCall` records work already executed by the supplier. Both remain
interleaved with `ContentPart` values in ordered output.

Each protocol profile contains the exact immutable `ModelCapabilities` returned by
its model client. The default Responses and Messages profile classes remain
provider-tool neutral. The official `openai_responses_profile()` and
`anthropic_messages_profile()` factories install their hosted-tool identities and
codecs, but do not add a tool to any request. The host must still pass an explicit
`ProviderToolSpec` factory result to `Runtime`, and selecting an official profile is
the host's confirmation that the chosen endpoint and model support its advertised
capabilities. Tool selection is declared as a set of supported types rather than a
coarse boolean. Supplier factories only compose protocol capabilities and wire
policies; the shared codecs contain no supplier-name branches.

```python
from jharness.kernel import Runtime
from jharness.models.openai import (
    OpenAIResponsesModel,
    openai_responses_profile,
    openai_responses_web_search,
)

model = OpenAIResponsesModel(..., profile=openai_responses_profile())
runtime = Runtime(
    model=model,
    provider_tools=(openai_responses_web_search(),),
)
```

OpenAI Responses sends `store=false` by default and requests encrypted reasoning
history. Hosted image generation additionally requires a host-owned
`OpenAIResponsesArtifactStore`; generated base64 is persisted externally and durable history
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
