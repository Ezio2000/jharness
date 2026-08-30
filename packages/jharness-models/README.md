# jharness-models

OpenAI Chat (Chat Completions API), OpenAI Responses, and Anthropic Messages adapters,
plus provider-neutral model composition for the JHarness kernel.

```bash
uv add jharness-models
```

```python
from jharness.models.openai import OpenAIResponsesModel
```

| Adapter | Runtime tools | Provider-hosted tools | Ordered output |
| --- | --- | --- | --- |
| OpenAI Chat | Function calls | None | Content and calls are normalized into `ModelResponse.output` |
| Anthropic Messages | Client `tool_use` | Official web-search preset | Native block order is retained |
| OpenAI Responses | Function and custom calls | Official web-search and image-generation presets | Native Responses item order is retained |

Model modalities describe what the model itself understands or produces. Tool
ownership is separate: `RuntimeToolCall` is executed by the JHarness runtime, while a
`ProviderToolCall` records work already executed by the supplier. Both remain
interleaved with `ContentPart` values in ordered output.

Each protocol profile contains the exact immutable `ModelCapabilities` returned by
its model client. The default Responses and Messages profile classes remain
provider-tool neutral. The official `openai_responses_profile()` and
`anthropic_messages_profile()` factories install their hosted-tool identities and
capabilities, but do not add a tool to any request. Hosted-tool mapping is a closed,
protocol-owned union; profiles cannot inject custom wire codecs. The host must still
pass an explicit `ProviderToolSpec` factory result to `Runtime`, and selecting an official profile is
the host's confirmation that the chosen endpoint and model support its advertised
capabilities. Tool selection is declared as a set of supported types rather than a
coarse boolean. Supplier factories only compose protocol capabilities and wire
policies; each adapter emits only its documented protocol wire.

Capabilities are host assertions rather than live endpoint discovery. JHarness blocks
requests outside the selected profile, but an overstated profile may still reach a
provider that rejects, ignores, or degrades the claimed feature. Use an exact,
model-appropriate profile. Image MIME `ArtifactRef` values count as image inputs:
Responses emits `input_image`, Anthropic Messages emits an image file source, and Chat
uses the standard nested `file` content part and therefore also requires file input.

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
history; assistant and reasoning history is replayed only from complete native Responses
output items. Hosted image generation additionally requires a host-owned
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
